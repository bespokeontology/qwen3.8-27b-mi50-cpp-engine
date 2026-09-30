// q27_fe -- C++17 OpenAI-compatible front end for the Qwen3.8-27B q27_gen engine.
//
// Replaces q27_server.py. The project has a HARD BAN on Python in the serving path; this binary is
// the Qwen lane's half of that law (the openPangu lane got its C++ front end on 2026-09-16).
//
// It fixes the two serve-path defects measured on 2026-09-18:
//
//   1. PREFIX REUSE WAS ALL-OR-NOTHING. The Python manager required the resident prefix to be an
//      EXACT prefix of the new request and otherwise reset to position 0 and re-prefilled the whole
//      window. Measured cost on one session: four full re-prefills of 482.6 s, 537.1 s, 582.0 s and
//      ~700 s, for divergences of 11, 18, 56 and 23 tokens. This front end computes the LONGEST
//      COMMON PREFIX, truncates the engine to it, and sweeps only the tail.
//
//   2. TOOL ARGUMENTS WERE ALL STRINGS. Every value in a tool call was emitted as a JSON string, so
//      any tool declaring an array or a number rejected the call
//      ("invalid arguments: \"todos\" must be an array"). Here the model's argument object is
//      parsed as JSON and re-serialised with its types intact (json.h).
//
// Engine wire protocol (q27_main.cpp:874-950, :5194):
//   spawn   q27_gen <model> <ndev> <ctx> <maxn> "" <boot ids...> with Q27_SERVE=1
//   request one line on stdin: "<maxn> <id> <id> ..."
//             maxn = -1  RESET to position 0
//             maxn = -2  PREFILL ONLY (sweep the ids, generate nothing)
//             maxn =  0  defaults to 256 in the engine
//   stream  "Q27_TOK <id>"     one line per COMMITTED token (Q27_SERVE_STREAM=1)
//   final   "Q27_TOKENS <id> <id> ..."   ends the request
//   boot    "Q27_READY devices=.. layers/device=.. ctx=.. slots=.."  and "Q27_SLOTS cap=N"
//
// Build: make

#include "json.h"
#include "hf_tokenizer.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdarg>
#include <functional>
#include <cerrno>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------ config

static std::string env_s(const char* k, const std::string& d) { const char* v = getenv(k); return v ? v : d; }
static int         env_i(const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; }

struct Cfg {
    std::string engine_dir = env_s("Q27_ENGINE_DIR", ".");
    std::string model      = env_s("Q27_MODEL", "./model");
    std::string tok_json   = env_s("Q27_TOKENIZER", "");   // defaults to <model>/tokenizer.json
    int  ndev   = env_i("Q27_NDEV", 4);
    int  ctx    = env_i("Q27_CTX", 262144);
    int  port   = env_i("Q27_FE_PORT", 8020);
    int  maxn   = env_i("Q27_SERVE_MAXN", 8192);
    int  boot_ids = env_i("Q27_BOOT_IDS", 4096);
    int  step   = env_i("Q27_SERVE_STEP", 8960);          // chunked prefill-only step
    int  wait_s = env_i("Q27_FE_WAIT_S", 1800);
} CFG;

static void logf(const char* fmt, ...) {
    char buf[4096];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char ts[32]; struct tm tmv; localtime_r(&now, &tmv); strftime(ts, sizeof ts, "%H:%M:%S", &tmv);
    fprintf(stderr, "%s fe: %s\n", ts, buf); fflush(stderr);
}

// ------------------------------------------------------------------ chat template
//
// Qwen3 conventions. Tools are declared in the system turn inside <tools></tools>; a tool result
// comes back as a user turn wrapped in <tool_response></tool_response>. The engine's own
// q27_chat.h implements only the text-only subset (no tools), which is why the template lives here.

static const char* IM_START = "<|im_start|>";
static const char* IM_END   = "<|im_end|>";

struct ToolDef { std::string name, desc, params_json; jz::V schema; };

// reasoning_effort is VALIDATED by the template and only three values are legal
// (chat_template.jinja:46-50). 'medium' injects no instruction text at all.
static std::string reasoning_instructions(const std::string& effort_in) {
    std::string e = effort_in.empty() ? "xhigh" : effort_in;
    if (e == "high") e = "medium";          // the provider's names -> the template's names
    else if (e == "max") e = "xhigh";
    if (e == "xhigh")
        return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
               "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
               "and clarity in the final answer.";
    if (e == "low")
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
               "the conclusion without unnecessary elaboration.";
    return "";                              // medium
}

static const char* TOOL_FORMAT_BLOCK =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n"
    "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function "
    "call, but NOT after\n- If there is no function call available, answer the question like normal with "
    "your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

static std::string trim_copy(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// THE PROMPT, rendered to match ./model/chat_template.jinja exactly. The record's
// standing law is to use the model's own template (hand-rolling it cost ~7 points once); this is a
// transcription of it, not an interpretation. Every divergence here is a re-prefill of the whole
// window, because prefix reuse is exact id equality over the rendered text.
static std::string render_prompt(const jz::V& messages, const std::vector<ToolDef>& tools,
                                 bool thinking, const std::string& effort) {
    std::string s;
    const std::string rinstr = thinking ? reasoning_instructions(effort) : "";

    // messages[0] is the system turn if present (jinja only ever looks at index 0).
    std::string sys0;
    if (messages.size() && messages.at(0)["role"].as_str() == "system")
        sys0 = trim_copy(messages.at(0)["content"].as_str());

    if (!tools.empty()) {
        s += IM_START; s += "system\n";
        if (!rinstr.empty()) { s += rinstr; s += "\n\n"; }
        s += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& t : tools) { s += "\n"; s += jz::dump(t.schema); }
        s += "\n</tools>";
        s += TOOL_FORMAT_BLOCK;
        if (!sys0.empty()) { s += "\n\n"; s += sys0; }
        s += IM_END; s += "\n";
    } else if (!sys0.empty()) {
        s += IM_START; s += "system\n";
        if (!rinstr.empty()) { s += rinstr; s += "\n\n"; }
        s += sys0; s += IM_END; s += "\n";
    } else if (!rinstr.empty()) {
        // No tools and no system turn, but an effort that carries instructions: the template still
        // opens a system turn for them alone (chat_template.jinja:84-85). Dropping it changes every
        // id in the window.
        s += IM_START; s += "system\n"; s += rinstr; s += IM_END; s += "\n";
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const jz::V& m = messages.at(i);
        const std::string role = m["role"].as_str();
        if (i == 0 && role == "system") continue;
        const std::string content = m["content"].as_str();

        if (role == "user") { s += IM_START; s += "user\n"; s += content; s += IM_END; s += "\n"; continue; }

        if (role == "tool") {
            // CONSECUTIVE TOOL RESULTS SHARE ONE USER TURN (chat_template.jinja:147-158). Opening a
            // turn per result changes the token stream and loses the prefix.
            const bool prev_tool = (i > 0 && messages.at(i - 1)["role"].as_str() == "tool");
            const bool last      = (i + 1 >= messages.size());
            const bool next_tool = (!last && messages.at(i + 1)["role"].as_str() == "tool");
            if (!prev_tool) { s += IM_START; s += "user"; }
            s += "\n<tool_response>\n"; s += content; s += "\n</tool_response>";
            if (last || !next_tool) { s += IM_END; s += "\n"; }
            continue;
        }

        if (role == "assistant") {
            // REASONING IS FED BACK. chat_template.jinja:117 re-renders the think block; if the
            // client's reasoning_content is dropped the block renders empty and every earlier turn
            // stops matching, which re-prefills the whole conversation.
            const std::string rc = trim_copy(m["reasoning_content"].as_str());
            s += IM_START; s += "assistant\n<think>\n"; s += rc; s += "\n</think>\n\n"; s += content;
            const jz::V& tc = m["tool_calls"];
            if (tc.is_arr()) {
                for (size_t k = 0; k < tc.size(); ++k) {
                    const jz::V& c0 = tc.at(k);
                    const jz::V& fn = c0.has("function") ? c0["function"] : c0;
                    if (k == 0) { if (!trim_copy(content).empty()) s += "\n\n"; }
                    else s += "\n";
                    s += "<tool_call>\n<function="; s += fn["name"].as_str(); s += ">\n";
                    bool ok = false;
                    jz::V args = jz::parse(fn["arguments"].as_str("{}"), &ok);
                    if (ok && args.is_obj())
                        for (const auto& kv : *args.o) {
                            s += "<parameter="; s += kv.first; s += ">\n";
                            // string -> raw, everything else -> JSON (chat_template.jinja:138)
                            s += kv.second.is_str() ? kv.second.s : jz::dump(kv.second);
                            s += "\n</parameter>\n";
                        }
                    s += "</function>\n</tool_call>";
                }
            }
            s += IM_END; s += "\n";
            continue;
        }
        // Unexpected role: the template raises. Render as user rather than fail the turn.
        s += IM_START; s += "user\n"; s += content; s += IM_END; s += "\n";
    }

    s += IM_START; s += "assistant\n";
    s += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return s;
}

// ------------------------------------------------------------------ tool-call extraction
//
// The model writes <tool_call>{json}</tool_call>. Arguments are parsed as JSON and re-serialised
// with types preserved -- this is defect 2's fix. Anything that is not valid JSON is passed through
// as a string argument rather than dropped, so a malformed call degrades instead of vanishing.

struct ToolCall { std::string name, args_json; };

// SCHEMA-FIRST TYPING. On the wire a string-typed "1700" and a number-typed 1700 are byte-identical
// (chat_template.jinja:138 renders a string raw and everything else as JSON), so the parameter's
// declared type in the tool schema is the ONLY thing that can tell them apart. Coercing by trying
// json parse first would silently turn a string-typed version number or zip code into a number.
static jz::V typed_param(const std::vector<ToolDef>& tools, const std::string& fn,
                         const std::string& key, const std::string& raw) {
    const jz::V* props = nullptr;
    for (const auto& t : tools)
        if (t.name == fn) { const jz::V& p = t.schema["function"]["parameters"]["properties"]; if (p.is_obj()) props = &p; break; }
    std::string type;
    if (props) { const jz::V& d = (*props)[key]; if (d.is_obj()) type = d["type"].as_str(); }

    if (type == "string") return jz::V::str(raw);
    if (!type.empty()) {
        bool ok = false; jz::V v = jz::parse(raw, &ok);
        if (ok) return v;
        // Declared non-string but did not parse: keep the text rather than drop the argument.
        logf("tool %s: parameter %s declared %s but did not parse as JSON, kept as text",
             fn.c_str(), key.c_str(), type.c_str());
        return jz::V::str(raw);
    }
    // No schema for this parameter. There is no way to know the intent; a bare token that parses as
    // JSON is taken at face value, anything else stays a string.
    bool ok = false; jz::V v = jz::parse(raw, &ok);
    return (ok && !v.is_str()) ? v : jz::V::str(raw);
}

// Parse the checkpoint's own tool-call dialect:
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...\n</function>\n</tool_call>
// Values may span multiple lines. Both the special-token framing (248058/248059) and the plain-text
// form decode to the same characters, so one text-level parser covers both.
static bool extract_tool_calls_x(const std::string& text, const std::vector<ToolDef>& tools,
                                 std::vector<ToolCall>& out, std::string& clean) {
    clean.clear(); out.clear();
    const std::string TO = "<tool_call>", TC = "</tool_call>";
    size_t i = 0;
    while (i < text.size()) {
        size_t a = text.find(TO, i);
        if (a == std::string::npos) { clean += text.substr(i); break; }
        clean += text.substr(i, a - i);
        size_t b = text.find(TC, a);
        const std::string blk = (b == std::string::npos) ? text.substr(a + TO.size())
                                                         : text.substr(a + TO.size(), b - a - TO.size());
        size_t fa = blk.find("<function=");
        if (fa != std::string::npos) {
            size_t fe = blk.find('>', fa);
            if (fe != std::string::npos) {
                ToolCall tc;
                tc.name = blk.substr(fa + 10, fe - fa - 10);
                jz::V args = jz::V::obj();
                size_t p = fe;
                for (;;) {
                    size_t pa = blk.find("<parameter=", p);
                    if (pa == std::string::npos) break;
                    size_t pe = blk.find('>', pa);
                    if (pe == std::string::npos) break;
                    const std::string key = blk.substr(pa + 11, pe - pa - 11);
                    size_t ve = blk.find("</parameter>", pe);
                    std::string raw = (ve == std::string::npos) ? blk.substr(pe + 1)
                                                                : blk.substr(pe + 1, ve - pe - 1);
                    if (!raw.empty() && raw.front() == '\n') raw.erase(raw.begin());
                    if (!raw.empty() && raw.back() == '\n') raw.pop_back();
                    args.set(key, typed_param(tools, tc.name, key, raw));
                    if (ve == std::string::npos) break;
                    p = ve + 12;
                }
                tc.args_json = jz::dump(args);
                if (!tc.name.empty()) out.push_back(std::move(tc));
            }
        } else {
            logf("tool_call block carried no <function=...>, kept as text (%zu bytes)", blk.size());
            clean += blk;
        }
        if (b == std::string::npos) break;
        i = b + TC.size();
    }
    while (!clean.empty() && (clean.back() == '\n' || clean.back() == ' ')) clean.pop_back();
    return !out.empty();
}


// Split a completed answer into reasoning and content on the <think> span.
static void split_think(const std::string& in, std::string& think, std::string& body,
                        bool prompt_open_think = false) {
    const std::string O = "<think>", C = "</think>";
    size_t a = in.find(O);
    // The thinking prompt already ends with <think>; generated text normally
    // starts inside that span and contains only its closing delimiter.
    if (a == std::string::npos && !prompt_open_think) { think.clear(); body = in; return; }
    const size_t begin = a == std::string::npos ? 0 : a + O.size();
    size_t b = in.find(C, begin);
    if (b == std::string::npos) { think = in.substr(begin); body.clear(); return; }
    think = in.substr(begin, b - begin);
    body  = (a == std::string::npos ? "" : in.substr(0, a)) + in.substr(b + C.size());
    while (!body.empty() && (body.front() == '\n' || body.front() == ' ')) body.erase(body.begin());
    while (!think.empty() && (think.front() == '\n' || think.front() == ' ')) think.erase(think.begin());
}

// ------------------------------------------------------------------ engine

class Engine {
public:
    // Events the reader thread publishes for the request in flight.
    struct Out {
        std::vector<unsigned> toks;     // streamed committed ids
        bool done = false;
        bool dead = false;
        std::string err;
    };

    bool start(const std::vector<unsigned>& boot, int maxn) {
        int inp[2], outp[2];
        if (pipe(inp) || pipe(outp)) { logf("pipe failed"); return false; }
        pid_ = fork();
        if (pid_ < 0) { logf("fork failed"); return false; }
        if (pid_ == 0) {
            dup2(inp[0], 0); dup2(outp[1], 1);
            close(inp[0]); close(inp[1]); close(outp[0]); close(outp[1]);
            std::string exe = CFG.engine_dir + "/q27_gen";
            std::vector<std::string> av{ exe, CFG.model, std::to_string(CFG.ndev),
                                         std::to_string(CFG.ctx), std::to_string(maxn), "" };
            for (unsigned t : boot) av.push_back(std::to_string(t));
            std::vector<char*> cav;
            for (auto& s : av) cav.push_back(const_cast<char*>(s.c_str()));
            cav.push_back(nullptr);
            setenv("Q27_SERVE", "1", 1);
            setenv("Q27_SERVE_STREAM", "1", 1);
            if (chdir(CFG.engine_dir.c_str()) != 0) _exit(127);
            execv(exe.c_str(), cav.data());
            _exit(127);
        }
        close(inp[0]); close(outp[1]);
        wfd_ = inp[1]; rfd_ = outp[0];
        reader_ = std::thread([this] { read_loop(); });
        return true;
    }

    // BOOT SYNC. Q27_READY is NEVER printed on the production path: the launcher pins Q27_TP=1, so
    // main() enters run_tp(), which does not return in serve mode, and the Q27_READY printf sits
    // after that call (q27_main.cpp:9927 vs :9941). The only reliable boot signal is the argv
    // request completing, so wait for the first Q27_TOKENS. Waiting on Q27_READY hangs forever.
    bool wait_boot(int seconds) {
        std::unique_lock<std::mutex> lk(m_);
        return cv_.wait_for(lk, std::chrono::seconds(seconds), [this] { return boot_done_ || dead_; }) && !dead_;
    }

    bool alive() const { return !dead_; }
    int  slots() const { return slots_; }

    // Submit one request line and collect until Q27_TOKENS. on_tok is called for each streamed id.
    // Returns false if the engine died.
    bool request(int maxn, const std::vector<unsigned>& ids,
                 const std::function<bool(unsigned)>& on_tok,
                 std::vector<unsigned>& final_toks, int timeout_s) {
        {
            std::lock_guard<std::mutex> lk(m_);
            cur_.toks.clear(); cur_.done = false; cur_.err.clear();
            consumed_ = 0;
        }
        std::string line = std::to_string(maxn);
        line.reserve(ids.size() * 7 + 16);
        for (unsigned t : ids) { line += ' '; line += std::to_string(t); }
        line += '\n';
        size_t off = 0;
        while (off < line.size()) {
            ssize_t w = ::write(wfd_, line.data() + off, line.size() - off);
            if (w <= 0) { logf("engine stdin write failed"); dead_ = true; return false; }
            off += (size_t)w;
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            if (!cv_.wait_until(lk, deadline, [this] { return cur_.done || dead_ || consumed_ < cur_.toks.size(); }))
                { logf("engine timeout after %d s", timeout_s); return false; }
            if (dead_) return false;
            while (consumed_ < cur_.toks.size()) {
                unsigned t = cur_.toks[consumed_++];
                lk.unlock();
                bool keep = on_tok(t);
                lk.lock();
                if (!keep) { cancel_locked(); return true; }
            }
            if (cur_.done) { final_toks = final_; return cur_.err.empty(); }
        }
    }

    void reset() { std::vector<unsigned> f; request(-1, {}, [](unsigned){ return true; }, f, 300); }

    // maxn == -3 : REWIND to the newest checkpoint at or before `target`. The engine answers
    // `Q27_SERVE_REWIND ok <actual>` with the position it actually restored (0 when it had to
    // fall all the way back), or `miss <ckpt>` on a broken/older engine. Returns false on timeout
    // or death, and `actual` is only meaningful when true.
    bool rewind(int target, int& actual, int timeout_s) {
        {
            std::lock_guard<std::mutex> lk(m_);
            rewind_done_ = false; rewind_ok_ = false; rewind_pos_ = -1;
        }
        std::string line = "-3 " + std::to_string(target) + "\n";
        size_t off = 0;
        while (off < line.size()) {
            ssize_t w = ::write(wfd_, line.data() + off, line.size() - off);
            if (w <= 0) { logf("engine stdin write failed during rewind"); dead_ = true; return false; }
            off += (size_t)w;
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            if (!cv_.wait_until(lk, deadline, [this] { return rewind_done_ || dead_; }))
                { logf("engine rewind timeout after %d s", timeout_s); return false; }
            if (dead_) return false;
            if (rewind_done_) { actual = rewind_pos_; return rewind_ok_; }
        }
    }

    void stop() {
        if (pid_ > 0) { ::close(wfd_); ::kill(pid_, SIGTERM); int st = 0; waitpid(pid_, &st, 0); pid_ = -1; }
        if (reader_.joinable()) reader_.join();
    }

private:
    void cancel_locked() {
        // Cancellation: the engine has no abort command, so the turn is allowed to finish into the
        // void. Never kill a resident engine to end one request -- that costs the whole window.
        cancelled_ = true;
    }

    void read_loop() {
        std::string buf;
        char chunk[65536];
        for (;;) {
            ssize_t n = ::read(rfd_, chunk, sizeof chunk);
            if (n <= 0) { std::lock_guard<std::mutex> lk(m_); dead_ = true; cv_.notify_all(); return; }
            buf.append(chunk, (size_t)n);
            size_t nl;
            while ((nl = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                handle(line);
            }
        }
    }

    void handle(const std::string& line) {
        if (line.compare(0, 16, "Q27_REQ_REFUSED ") == 0) {
            std::lock_guard<std::mutex> lk(m_); cur_.err = line; cur_.done = true;
            logf("engine refused: %s", line.c_str()); cv_.notify_all(); return;
        }
        if (line.compare(0, 8, "Q27_TOK ") == 0) {
            unsigned t = (unsigned)strtoul(line.c_str() + 8, nullptr, 10);
            std::lock_guard<std::mutex> lk(m_);
            cur_.toks.push_back(t); cv_.notify_all(); return;
        }
        if (line.compare(0, 11, "Q27_TOKENS ") == 0 || line == "Q27_TOKENS") {
            std::vector<unsigned> v;
            const char* p = line.c_str() + (line.size() > 11 ? 11 : line.size());
            while (*p) { char* e2; unsigned long t = strtoul(p, &e2, 10); if (e2 == p) break; v.push_back((unsigned)t); p = e2; }
            std::lock_guard<std::mutex> lk(m_);
            final_ = std::move(v); cur_.done = true; boot_done_ = true; cancelled_ = false; cv_.notify_all(); return;
        }
        if (line.compare(0, 10, "Q27_READY ") == 0) {
            std::lock_guard<std::mutex> lk(m_);
            ready_ = true; cv_.notify_all();
            logf("engine ready: %s", line.c_str() + 10); return;
        }
        if (line.compare(0, 14, "Q27_SLOTS cap=") == 0) {
            slots_ = atoi(line.c_str() + 14);
            logf("engine slot capacity %d", slots_); return;
        }
        // A reset (maxn == -1) is confirmed by the engine with this line, NOT with Q27_TOKENS.
        // Without recognising it, request(-1, ...) waits the whole timeout for a completion that
        // never arrives, and every divergent turn hangs. (Defect: the reset-then-resend deadlock.)
        if (line == "Q27_SERVE_RESET ok") {
            std::lock_guard<std::mutex> lk(m_);
            cur_.done = true; cv_.notify_all(); return;
        }
        if (line.compare(0, 20, "Q27_SERVE_REWIND ok ") == 0) {
            int p = atoi(line.c_str() + 20);
            std::lock_guard<std::mutex> lk(m_);
            rewind_pos_ = p; rewind_ok_ = true; rewind_done_ = true; cv_.notify_all(); return;
        }
        if (line.compare(0, 22, "Q27_SERVE_REWIND miss ") == 0) {
            int p = atoi(line.c_str() + 22);
            std::lock_guard<std::mutex> lk(m_);
            rewind_pos_ = p; rewind_ok_ = false; rewind_done_ = true; cv_.notify_all(); return;
        }
        // Everything else (Q27_REQ, Q27_MS, Q27_VRAM, Q27_SERVE_GROW ...) is telemetry: it belongs
        // in the log the record is read from, unchanged.
        if (line.compare(0, 4, "Q27_") == 0) { fprintf(stderr, "engine: %s\n", line.c_str()); fflush(stderr); }
    }

    pid_t pid_ = -1;
    int wfd_ = -1, rfd_ = -1;
    std::thread reader_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    Out cur_;
    std::vector<unsigned> final_;
    size_t consumed_ = 0;
    std::atomic<bool> dead_{false};
    bool ready_ = false;
    bool boot_done_ = false;
    bool cancelled_ = false;
    bool rewind_done_ = false;
    bool rewind_ok_ = false;
    int  rewind_pos_ = -1;
    int slots_ = 0;
};

// ------------------------------------------------------------------ session
//
// THE FIX for defect 1. The resident token window is tracked here; a new request reuses the
// LONGEST COMMON PREFIX instead of demanding an exact one.

class Session {
public:
    size_t common_prefix(const std::vector<unsigned>& want) const {
        size_t lcp = 0;
        const size_t n = std::min(resident_.size(), want.size());
        while (lcp < n && resident_[lcp] == want[lcp]) ++lcp;
        return lcp;
    }

    // Returns the tail that must be swept, and sets `did_reset` when the engine had to go back to 0.
    std::vector<unsigned> plan(const std::vector<unsigned>& want, bool& did_reset, size_t& lcp_out) {
        size_t lcp = 0;
        const size_t n = std::min(resident_.size(), want.size());
        while (lcp < n && resident_[lcp] == want[lcp]) ++lcp;
        lcp_out = lcp;
        did_reset = false;
        if (lcp == resident_.size() && want.size() > resident_.size()) {
            // Pure append -- the common case, and free.
            return std::vector<unsigned>(want.begin() + (long)lcp, want.end());
        }
        if (lcp == 0) { did_reset = true; resident_.clear(); return want; }
        // Partial match. The engine can only truncate by resetting, so pay one reset and re-sweep
        // from the LCP -- still far cheaper than re-sweeping everything, and it is exactly the case
        // the Python front end threw away four times in ninety minutes.
        did_reset = true;
        resident_.clear();
        return want;
    }

    void committed(const std::vector<unsigned>& full_window) { resident_ = full_window; }
    void append(const std::vector<unsigned>& more) { resident_.insert(resident_.end(), more.begin(), more.end()); }
    void truncate(size_t n) { if (n < resident_.size()) resident_.resize(n); }
    void clear() { resident_.clear(); }
    size_t size() const { return resident_.size(); }

private:
    std::vector<unsigned> resident_;
};

// ------------------------------------------------------------------ HTTP

struct Req { std::string method, path, body; };

// Both routes are matched by SUFFIX, as the surface being replaced did: clients prefix the path
// (/v1/..., /openai/v1/...) and exact routing silently 404s them.
static bool ends_with(const std::string& s, const char* suf) {
    const size_t n = strlen(suf);
    if (s.size() < n) return false;
    std::string t = s;
    while (!t.empty() && t.back() == 0x2F) t.pop_back();          // ignore a trailing slash
    return t.size() >= n && t.compare(t.size() - n, n, suf) == 0;
}

static bool read_http(int fd, Req& r) {
    std::string buf;
    char c[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        ssize_t n = ::read(fd, c, sizeof c);
        if (n <= 0) return false;
        buf.append(c, (size_t)n);
        hdr_end = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 28)) return false;
    }
    std::string head = buf.substr(0, hdr_end);
    size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = head.substr(0, sp1);
    r.path   = head.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t cl = 0;
    { std::string low; low.reserve(head.size());
      for (char ch : head) low += (char)tolower((unsigned char)ch);
      size_t p = low.find("content-length:");
      if (p != std::string::npos) cl = (size_t)strtoul(head.c_str() + p + 15, nullptr, 10); }
    r.body = buf.substr(hdr_end + 4);
    while (r.body.size() < cl) {
        ssize_t n = ::read(fd, c, sizeof c);
        if (n <= 0) return false;
        r.body.append(c, (size_t)n);
    }
    return true;
}

static bool wr(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::write(fd, s.data() + off, s.size() - off);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

static void send_json(int fd, int code, const std::string& body) {
    char hdr[256];
    snprintf(hdr, sizeof hdr,
             "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
             code, code == 200 ? "OK" : "Error", body.size());
    wr(fd, hdr); wr(fd, body);
}

static void send_sse_head(int fd) {
    const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    "Connection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    wr(fd, h);
}

static bool send_sse(int fd, const std::string& json) {
    return wr(fd, "data: " + json + "\r\n\r\n");
}

// ------------------------------------------------------------------ server

static Engine     ENG;
static Session    SESS;
static std::mutex ENG_LOCK;          // the engine serves one turn at a time
static hf::Tokenizer* TOK = nullptr;
static std::atomic<long long> REQ_N{0};

static std::vector<unsigned> encode(const std::string& text) {
    std::vector<unsigned> ids;
    for (std::int32_t t : TOK->encode(text)) if (t >= 0) ids.push_back((unsigned)t);
    return ids;
}

// hf::Tokenizer::decode_token throws std::invalid_argument for any id >= 248077 -- Q27_VOCAB
// (248320) is the padded lm_head width, not the tokenizer vocabulary, and the engine does emit
// padding ids. An unguarded decode kills the turn.
static bool decodable(unsigned t) { return t < 248077u; }
static std::string decode1(unsigned t) {
    if (!decodable(t)) return "";
    try { return TOK->decode_token((std::int32_t)t); } catch (...) { return ""; }
}

static std::string now_id() {
    static std::atomic<long long> n{0};
    return "chatcmpl-q27-" + std::to_string(++n);
}

static std::vector<unsigned> request_tokens(const jz::V& req, std::vector<ToolDef>* out_tools = nullptr, bool* out_thinking = nullptr) {
    bool thinking = true;
    if (req.has("chat_template_kwargs")) {
        const jz::V& k = req["chat_template_kwargs"];
        if (k.has("enable_thinking")) thinking = k["enable_thinking"].as_bool(true);
    }

    std::vector<ToolDef> tools;
    const jz::V& tv = req["tools"];
    if (tv.is_arr()) {
        for (size_t i = 0; i < tv.size(); ++i) {
            const jz::V& f = tv.at(i)["function"];
            ToolDef t;
            t.name = f["name"].as_str();
            t.desc = f["description"].as_str();
            t.schema = tv.at(i);              // the whole {"type":"function","function":{...}} object
            if (!t.name.empty()) tools.push_back(std::move(t));
        }
    }

    const std::string effort = req["reasoning_effort"].as_str("");
    const std::string prompt = render_prompt(req["messages"], tools, thinking, effort);
    if (out_tools) *out_tools = tools;
    if (out_thinking) *out_thinking = thinking;
    return encode(prompt);

}

static void handle_chat(int fd, const jz::V& req) {
    const bool stream = req["stream"].as_bool(false);
    const std::string model = req["model"].as_str("qwen3.8-27b-local");
    int maxn = (int)req["max_tokens"].as_int(CFG.maxn);
    if (maxn <= 0) maxn = CFG.maxn;

    std::vector<ToolDef> tools;
    bool thinking = true;
    std::vector<unsigned> want = request_tokens(req, &tools, &thinking);

    const long long rid = ++REQ_N;

    if ((int)want.size() + maxn > CFG.ctx) {
        char m[256];
        snprintf(m, sizeof m,
                 "{\"error\":{\"message\":\"prompt %zu + max_tokens %d exceeds ctx %d\",\"type\":\"invalid_request_error\"}}",
                 want.size(), maxn, CFG.ctx);
        send_json(fd, 400, m);
        return;
    }

    if (const char* dir = getenv("Q27_FE_CAPTURE")) {
        const std::string file = std::string(dir) + "/request-" + std::to_string(rid) + ".json";
        if (FILE* f = std::fopen(file.c_str(), "wx")) {
            const std::string body = jz::dump(req); std::fwrite(body.data(), 1, body.size(), f); std::fclose(f);
        }
    }


    // SSE HEADERS AND THE HEARTBEAT GO OUT BEFORE THE ENGINE LOCK.
    // The provider's idle watchdog is ~300 s per read, and a cold sweep at depth measured 582 s on
    // 2026-09-18. A turn that is silent while it prefills -- or while it waits behind another turn
    // for the single engine -- is killed by the client, not by the engine. An empty-content delta is
    // dropped by the parser but rearms the watchdog.
    std::mutex wmu;
    std::atomic<bool> turn_done{false};
    auto emit = [&](const std::string& js) {
        std::lock_guard<std::mutex> g(wmu);
        return send_sse(fd, js);
    };
    auto chunk_of = [&](const std::string& cid, jz::V delta, const char* finish) {
        jz::V ch = jz::V::obj();
        ch.set("index", jz::V::integer(0));
        ch.set("delta", std::move(delta));
        // finish_reason is ALWAYS present and null mid-stream.
        ch.set("finish_reason", finish ? jz::V::str(finish) : jz::V::null());
        jz::V o = jz::V::obj();
        o.set("id", jz::V::str(cid)); o.set("object", jz::V::str("chat.completion.chunk"));
        o.set("model", jz::V::str(model));
        jz::V a = jz::V::arr(); a.push(ch); o.set("choices", a);
        return jz::dump(o);
    };
    const std::string cid0 = now_id();
    std::thread hb;
    if (stream) {
        send_sse_head(fd);
        jz::V d = jz::V::obj();
        d.set("role", jz::V::str("assistant"));
        d.set("content", jz::V::str(""));   // EMPTY: a non-empty first delta would fake the client's TTFT
        emit(chunk_of(cid0, d, nullptr));
        hb = std::thread([&] {
            while (!turn_done.load()) {
                for (int i = 0; i < 300 && !turn_done.load(); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (turn_done.load()) break;
                jz::V e = jz::V::obj(); e.set("content", jz::V::str(""));
                if (!emit(chunk_of(cid0, e, nullptr))) break;
            }
        });
    }
    struct HbJoin {
        std::atomic<bool>& f; std::thread& t;
        ~HbJoin() { f.store(true); if (t.joinable()) t.join(); }
    } hbjoin{turn_done, hb};

    std::lock_guard<std::mutex> lk(ENG_LOCK);

    if (!ENG.alive()) { send_json(fd, 503, "{\"error\":{\"message\":\"engine is not running\"}}"); return; }


    const size_t lcp = SESS.common_prefix(want);
    bool did_reset = false;
    int  rewind_pos = -1;
    std::vector<unsigned> tail;

    if (lcp == SESS.size() && want.size() > SESS.size()) {
        // Pure append -- the common case, and free.
        tail.assign(want.begin() + (long)lcp, want.end());
    } else if (lcp == 0) {
        did_reset = true; SESS.clear(); tail = want;
    } else {
        // Partial match. Try the engine's rewind verb FIRST; only reset when the engine has no
        // checkpoint at or before the common prefix. The engine answers with the position it
        // actually restored, which may be earlier than lcp, so sweep from THAT position.
        if (ENG.rewind((int)std::min(lcp, want.size() - 1), rewind_pos, CFG.wait_s)) {
            SESS.truncate((size_t)rewind_pos);
            tail.assign(want.begin() + (long)rewind_pos, want.end());
        } else {
            did_reset = true; SESS.clear(); tail = want;
        }
    }

    const char* reuse_kind = did_reset
        ? (lcp ? "PARTIAL-REUSE(reset)" : "no-prefix(reset)")
        : (rewind_pos >= 0 ? "PARTIAL-REUSE(rewind)" : "append");
    logf("turn %lld prompt=%zu resident=%zu lcp=%zu %s rewind_to=%d tail=%zu maxn=%d tools=%zu",
         rid, want.size(), SESS.size(), lcp, reuse_kind, rewind_pos, tail.size(), maxn, tools.size());

    if (did_reset) { ENG.reset(); SESS.clear(); }

    // Sweep any oversized head as prefill-only steps so each engine request stays bounded, then let
    // the final request carry the remainder and generate.
    const int step = ENG.slots() > 0 ? std::min(CFG.step, ENG.slots()) : CFG.step;
    size_t pos = 0;
    while (tail.size() - pos > (size_t)step) {
        std::vector<unsigned> chunk(tail.begin() + (long)pos, tail.begin() + (long)pos + step);
        std::vector<unsigned> fin;
        if (!ENG.request(-2, chunk, [](unsigned){ return true; }, fin, CFG.wait_s)) {
            send_json(fd, 500, "{\"error\":{\"message\":\"engine failed during chunked prefill\"}}");
            SESS.clear(); return;
        }
        SESS.append(chunk);
        pos += (size_t)step;
    }
    std::vector<unsigned> last(tail.begin() + (long)pos, tail.end());

    const std::string& cid = cid0;         // headers and the role chunk already went out above
    std::string acc;                       // decoded text so far
    std::vector<unsigned> gen;

    bool client_gone = false;
    bool in_think = false, think_closed = !thinking;
    std::string pending;                   // text not yet classified as reasoning/content

    auto on_tok = [&](unsigned t) -> bool {
        gen.push_back(t);
        std::string piece = decode1(t);
        acc += piece;
        if (!stream) return true;
        pending += piece;
        // Hold back while a tool call might be forming; tool calls are delivered whole at the end.
        // ⛔ FIX: the hold-back used to fire on `<tool_call>` appearing ANYWHERE in `acc`, including
        // inside the model's own reasoning. A turn that THINKS ABOUT calling a tool therefore
        // stopped streaming mid-thought and the rest of the reasoning never reached the client —
        // which reads as "thinking does not stream". Tool calls only ever appear AFTER `</think>`,
        // so gate the hold-back on the think span being closed.
        if (think_closed && acc.find("<tool_call>") != std::string::npos) return true;
        // Classify into reasoning_content vs content on the <think> span.
        std::string emit_reason, emit_content;
        for (;;) {
            if (!think_closed) {
                size_t o = pending.find("<think>");
                if (o != std::string::npos && !in_think) { in_think = true; pending.erase(o, 7); continue; }
                size_t c = pending.find("</think>");
                if (c != std::string::npos) { emit_reason += pending.substr(0, c); pending.erase(0, c + 8); think_closed = true; in_think = false; continue; }
                if (in_think) { emit_reason += pending; pending.clear(); }
                break;
            }
            emit_content += pending; pending.clear(); break;
        }
        if (emit_reason.empty() && emit_content.empty()) return true;
        jz::V d = jz::V::obj();
        if (!emit_reason.empty())  d.set("reasoning_content", jz::V::str(emit_reason));
        if (!emit_content.empty()) d.set("content", jz::V::str(emit_content));
        if (!emit(chunk_of(cid, d, nullptr))) { client_gone = true; return false; }
        return true;
    };

    std::vector<unsigned> fin;
    const bool ok = ENG.request(maxn, last, on_tok, fin, CFG.wait_s);
    if (!ok && !client_gone) {
        if (stream) { send_sse(fd, "{\"error\":{\"message\":\"engine failed or timed out\"}}"); wr(fd, "data: [DONE]\r\n\r\n"); }
        else send_json(fd, 500, "{\"error\":{\"message\":\"engine failed or timed out\"}}");
        SESS.clear();
        return;
    }

    std::string think, body;
    split_think(acc, think, body, thinking);
    std::vector<ToolCall> calls;
    std::string clean;
    const bool has_tools = extract_tool_calls_x(body, tools, calls, clean);
    const char* finish = has_tools ? "tool_calls" : ((int)gen.size() >= maxn ? "length" : "stop");

    // Q27_TOKENS is the complete sampled sequence, including stop specials. The last
    // sampled token has NOT been consumed by the target: KV/state end one token before
    // it. Keep only committed inputs so the next append submits that token exactly once.
    {
        std::vector<unsigned> w = want;
        if (!fin.empty()) w.insert(w.end(), fin.begin(), fin.end() - 1);
        SESS.committed(w);
    }

    if (stream) {
        if (has_tools && !client_gone) {
            // Two chunks per call: the provider keys tool calls by index, and id/name ride the first.
            for (size_t i = 0; i < calls.size(); ++i) {
                jz::V c = jz::V::obj();
                c.set("index", jz::V::integer((long long)i));
                c.set("id", jz::V::str("call_" + cid + "_" + std::to_string(i)));
                c.set("type", jz::V::str("function"));
                jz::V f = jz::V::obj();
                f.set("name", jz::V::str(calls[i].name));
                f.set("arguments", jz::V::str(""));
                c.set("function", f);
                jz::V arr = jz::V::arr(); arr.push(c);
                jz::V d = jz::V::obj(); d.set("tool_calls", arr);
                if (!emit(chunk_of(cid, d, nullptr))) { client_gone = true; break; }

                jz::V c2 = jz::V::obj();
                c2.set("index", jz::V::integer((long long)i));
                jz::V f2 = jz::V::obj();
                f2.set("arguments", jz::V::str(calls[i].args_json));   // typed JSON inside the string
                c2.set("function", f2);
                jz::V arr2 = jz::V::arr(); arr2.push(c2);
                jz::V d2 = jz::V::obj(); d2.set("tool_calls", arr2);
                if (!emit(chunk_of(cid, d2, nullptr))) { client_gone = true; break; }
            }
        }
        // NEVER FINISH WITHOUT HAVING EMITTED SOMETHING. An answer with no content at all is in the
        // harness's retryable set as EMPTY_RESPONSE, and it silently re-issues the entire request --
        // which at this depth is another full sweep.
        if (!client_gone && !has_tools && trim_copy(body).empty()) {
            jz::V d = jz::V::obj(); d.set("content", jz::V::str(" "));
            emit(chunk_of(cid, d, nullptr));
        }
        if (!client_gone) {
            emit(chunk_of(cid, jz::V::obj(), finish));
            // ---- FINAL USAGE CHUNK (SIDE BRANCH) -------------------------------------------
            // ⛔ WHY THIS EXISTS: the non-streaming branch below builds a `usage` block, but THE
            // HARNESS ALWAYS READS SSE (banked: the main agent call sends "stream": false and no
            // Accept: text/event-stream, and the pi-ai provider parses the reply as a stream
            // anyway). So the one path carrying completion_tokens was the one path never read,
            // and every flash-next session on this box records "usage":{"inputTokens":0,
            // "outputTokens":0} — including the 12:43 turn that produced the frozen 179.3 s.
            // ⚠ 0 tok/s was therefore a MISSING FIELD, not a slow engine: a slow engine yields a
            // small number, an absent token count divides to exactly zero.
            // OpenAI streaming convention: a final chunk with EMPTY choices carrying usage.
            {
                jz::V u = jz::V::obj();
                u.set("prompt_tokens", jz::V::integer((long long)want.size()));
                u.set("completion_tokens", jz::V::integer((long long)gen.size()));
                u.set("total_tokens", jz::V::integer((long long)(want.size() + gen.size())));
                jz::V o = jz::V::obj();
                o.set("id", jz::V::str(cid));
                o.set("object", jz::V::str("chat.completion.chunk"));
                o.set("model", jz::V::str(model));
                o.set("choices", jz::V::arr());          // empty, per the convention
                o.set("usage", u);
                emit(jz::dump(o));
            }
            std::lock_guard<std::mutex> g(wmu);
            wr(fd, "data: [DONE]\r\n\r\n");
        }
    } else {
        jz::V msg = jz::V::obj();
        msg.set("role", jz::V::str("assistant"));
        msg.set("content", jz::V::str(has_tools ? clean : body));
        if (!think.empty()) msg.set("reasoning_content", jz::V::str(think));
        if (has_tools) {
            jz::V arr = jz::V::arr();
            for (size_t i = 0; i < calls.size(); ++i) {
                jz::V c = jz::V::obj();
                c.set("id", jz::V::str("call_" + cid + "_" + std::to_string(i)));
                c.set("type", jz::V::str("function"));
                jz::V f = jz::V::obj();
                f.set("name", jz::V::str(calls[i].name));
                f.set("arguments", jz::V::str(calls[i].args_json));
                c.set("function", f);
                arr.push(c);
            }
            msg.set("tool_calls", arr);
        }
        jz::V ch = jz::V::obj();
        ch.set("index", jz::V::integer(0));
        ch.set("message", msg);
        ch.set("finish_reason", jz::V::str(finish));
        jz::V usage = jz::V::obj();
        usage.set("prompt_tokens", jz::V::integer((long long)want.size()));
        usage.set("completion_tokens", jz::V::integer((long long)gen.size()));
        usage.set("total_tokens", jz::V::integer((long long)(want.size() + gen.size())));
        jz::V o = jz::V::obj();
        o.set("id", jz::V::str(cid)); o.set("object", jz::V::str("chat.completion"));
        o.set("model", jz::V::str(model));
        jz::V a = jz::V::arr(); a.push(ch); o.set("choices", a);
        o.set("usage", usage);
        send_json(fd, 200, jz::dump(o));
    }
    logf("turn %lld done gen=%zu tools=%zu resident=%zu", rid, gen.size(), calls.size(), SESS.size());
}

static void serve_conn(int fd) {
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    Req r;
    if (read_http(fd, r)) {
        if (r.method == "OPTIONS") {
            wr(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                   "Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                   "Content-Length: 0\r\nConnection: close\r\n\r\n");
        } else if (ends_with(r.path, "/models")) {
            jz::V m = jz::V::obj();
            m.set("id", jz::V::str("qwen3.8-27b-local"));
            m.set("object", jz::V::str("model"));
            m.set("owned_by", jz::V::str("local"));
            // The harness discovery parser reads these names; emitting only max_model_len is not enough.
            m.set("context_length", jz::V::integer(CFG.ctx));
            m.set("context_window", jz::V::integer(CFG.ctx));
            m.set("max_model_len", jz::V::integer(CFG.ctx));
            m.set("max_tokens", jz::V::integer(CFG.maxn));
            m.set("max_output_tokens", jz::V::integer(CFG.maxn));
            jz::V a = jz::V::arr(); a.push(m);
            jz::V o = jz::V::obj(); o.set("object", jz::V::str("list")); o.set("data", a);
            send_json(fd, 200, jz::dump(o));
        } else if (r.path.rfind("/health", 0) == 0) {
            send_json(fd, 200, ENG.alive() ? "{\"status\":\"ok\"}" : "{\"status\":\"engine-dead\"}");
        } else if (r.method == "POST" && ends_with(r.path, "/tokenize")) {
            bool ok = false;
            const jz::V req = jz::parse(r.body, &ok);
            if (!ok || !req["messages"].is_arr()) {
                send_json(fd, 400, "{\"error\":{\"message\":\"messages must be an array\"}}");
            } else {
                try {
                    const auto ids = request_tokens(req);
                    jz::V o = jz::V::obj();
                    o.set("prompt_tokens", jz::V::integer(ids.size()));
                    o.set("context_window", jz::V::integer(CFG.ctx));
                    o.set("max_output_tokens", jz::V::integer(CFG.maxn));
                    send_json(fd, 200, jz::dump(o));
                } catch (const std::exception& e) {
                    send_json(fd, 400, "{\"error\":{\"message\":\"native tokenization failed\"}}");
                }
            }
        } else if (r.method == "POST" && ends_with(r.path, "/chat/completions")) {
            bool ok = false;
            jz::V req = jz::parse(r.body, &ok);
            if (!ok) send_json(fd, 400, "{\"error\":{\"message\":\"request body is not valid JSON\"}}");
            else handle_chat(fd, req);
        } else {
            send_json(fd, 404, "{\"error\":{\"message\":\"not found\"}}");
        }
    }
    ::close(fd);
}

extern "C" int q27_fe_selftest();

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--selftest") return q27_fe_selftest();
    signal(SIGPIPE, SIG_IGN);

    std::string tj = CFG.tok_json.empty() ? CFG.model + "/tokenizer.json" : CFG.tok_json;
    try { TOK = new hf::Tokenizer(tj); }
    catch (const std::exception& e) { logf("tokenizer load failed (%s): %s", tj.c_str(), e.what()); return 1; }
    logf("tokenizer %s", tj.c_str());

    // Boot the engine on a bounded prompt. A cold start with a huge argv dies with E2BIG, which is
    // why the boot prompt is capped and the rest of any first turn rides the normal append path.
    std::vector<unsigned> boot = encode(std::string(IM_START) + "user\nhi" + IM_END + "\n" + IM_START + "assistant\n");
    if ((int)boot.size() > CFG.boot_ids) boot.resize(CFG.boot_ids);
    if (!ENG.start(boot, 8)) { logf("engine failed to start"); return 1; }
    logf("engine spawned, waiting for Q27_READY");
    if (!ENG.wait_boot(900)) { logf("engine never finished its boot request"); return 1; }
    // The boot request warms the resident weights and kernels but advances the engine past a tiny
    // "hi" prompt. The tracked window starts at the FIRST REAL REQUEST, so reset the engine to 0
    // here; otherwise every rewind target is offset by the boot length, and a checkpoint at that
    // offset restores the state BEFORE the first tracked token.
    ENG.reset();
    SESS.clear();

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons((uint16_t)CFG.port);
    if (bind(srv, (sockaddr*)&a, sizeof a) != 0) { logf("bind :%d failed: %s", CFG.port, strerror(errno)); return 1; }
    if (listen(srv, 64) != 0) { logf("listen failed"); return 1; }
    logf("listening on 127.0.0.1:%d", CFG.port);

    for (;;) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) { if (errno == EINTR) continue; break; }
        std::thread(serve_conn, fd).detach();
    }
    ENG.stop();
    return 0;
}


// ------------------------------------------------------------------ self-test
//
// CPU only, no engine, no GPU. The record's standing law is to run the envelope corpus BEFORE and
// AFTER any change to tool-call parsing, and to DIFF THE FULL ROW rather than assert n > 0 -- two
// individually correct repairs cancelled each other once and only the corpus caught it.
int q27_fe_selftest() {
    int pass = 0, fail = 0;
    auto check = [&](const char* what, bool ok, const std::string& got = "") {
        if (ok) { printf("PASS  %s\n", what); ++pass; }
        else { printf("FAIL  %s%s%s\n", what, got.empty() ? "" : "  got: ", got.c_str()); ++fail; }
    };

    // The checkpoint's real dialect, with a schema so typing is decided by the schema and not by
    // guessing at the text (chat_template.jinja:138 makes "1700" and 1700 byte-identical).
    std::vector<ToolDef> tools;
    {
        bool ok = false;
        ToolDef t; t.name = "read";
        t.schema = jz::parse(R"({"type":"function","function":{"name":"read","parameters":{"type":"object",)"
                             R"("properties":{"file_path":{"type":"string"},"offset":{"type":"number"},)"
                             R"("limit":{"type":"number"},"version":{"type":"string"}}}}})", &ok);
        tools.push_back(t);
        ToolDef u; u.name = "todo";
        u.schema = jz::parse(R"({"type":"function","function":{"name":"todo","parameters":{"type":"object",)"
                             R"("properties":{"todos":{"type":"array"},"done":{"type":"boolean"}}}}})", &ok);
        tools.push_back(u);
    }

    {
        std::vector<ToolCall> c; std::string clean;
        extract_tool_calls_x("thinking out loud\n<tool_call>\n<function=read>\n"
                             "<parameter=file_path>\n./src/q27_main.cpp\n</parameter>\n"
                             "<parameter=offset>\n1700\n</parameter>\n"
                             "<parameter=limit>\n120\n</parameter>\n</function>\n</tool_call>", tools, c, clean);
        check("read: parsed one call", c.size() == 1, std::to_string(c.size()));
        check("read: offset is a NUMBER (the failure that cost two steps)",
              !c.empty() && c[0].args_json.find("\"offset\":1700") != std::string::npos, c.empty() ? "" : c[0].args_json);
        check("read: limit is a NUMBER",
              !c.empty() && c[0].args_json.find("\"limit\":120") != std::string::npos, c.empty() ? "" : c[0].args_json);
        check("read: file_path stays a STRING",
              !c.empty() && c[0].args_json.find("\"file_path\":\"./src/q27_main.cpp\"") != std::string::npos, c.empty() ? "" : c[0].args_json);
        check("read: prose before the call is preserved", clean.find("thinking out loud") != std::string::npos, clean);
    }
    {
        // A string-typed parameter whose text looks numeric MUST stay a string. This is the case a
        // json-parse-first coercion gets wrong.
        std::vector<ToolCall> c; std::string clean;
        extract_tool_calls_x("<tool_call>\n<function=read>\n<parameter=version>\n1700\n</parameter>\n"
                             "</function>\n</tool_call>", tools, c, clean);
        check("schema-first: string-typed \"1700\" is NOT coerced to a number",
              !c.empty() && c[0].args_json.find("\"version\":\"1700\"") != std::string::npos, c.empty() ? "" : c[0].args_json);
    }
    {
        std::vector<ToolCall> c; std::string clean;
        extract_tool_calls_x("<tool_call>\n<function=todo>\n<parameter=todos>\n[{\"id\":1,\"t\":\"x\"}]\n</parameter>\n"
                             "<parameter=done>\nfalse\n</parameter>\n</function>\n</tool_call>", tools, c, clean);
        check("todo: array stays an ARRAY (the to-do failure, 4 steps)",
              !c.empty() && c[0].args_json.find("\"todos\":[{") != std::string::npos, c.empty() ? "" : c[0].args_json);
        check("todo: boolean stays a BOOL",
              !c.empty() && c[0].args_json.find("\"done\":false") != std::string::npos, c.empty() ? "" : c[0].args_json);
    }
    {
        // Multi-line values are explicitly supported by the template's own instruction block.
        std::vector<ToolCall> c; std::string clean;
        extract_tool_calls_x("<tool_call>\n<function=read>\n<parameter=file_path>\nline one\nline two\n</parameter>\n"
                             "</function>\n</tool_call>", tools, c, clean);
        check("multi-line parameter value survives",
              !c.empty() && c[0].args_json.find("line one\\nline two") != std::string::npos, c.empty() ? "" : c[0].args_json);
    }
    {
        std::vector<ToolCall> c; std::string clean;
        extract_tool_calls_x("<tool_call>\n<function=read>\n<parameter=offset>\n1\n</parameter>\n</function>\n</tool_call>"
                             "\n<tool_call>\n<function=todo>\n<parameter=done>\ntrue\n</parameter>\n</function>\n</tool_call>",
                             tools, c, clean);
        check("two calls in one answer", c.size() == 2, std::to_string(c.size()));
    }
    {
        std::vector<ToolCall> c; std::string clean;
        bool any = extract_tool_calls_x("<tool_call>\nno function block here\n</tool_call>", tools, c, clean);
        check("malformed block degrades to text, does not vanish or crash", !any && !clean.empty(), clean);
    }
    { bool ok=false; const std::string src = "{\"n\":3,\"f\":1.5,\"b\":true,\"z\":null,\"a\":[1,\"two\",false],\"o\":{\"k\":7}}";
      check("json: round-trip preserves every type", jz::dump(jz::parse(src,&ok)) == src && ok, jz::dump(jz::parse(src,&ok))); }
    { // big integers must not go through %g and become 1.6015e+07
      bool ok=false; check("json: large integer is not scientific",
                           jz::dump(jz::parse("{\"off\":16015000}",&ok)) == "{\"off\":16015000}", jz::dump(jz::parse("{\"off\":16015000}",&ok))); }
    {
        std::string th, bd;
        split_think("<think>\nreasoning here\n</think>\n\nthe answer", th, bd);
        check("think: reasoning split from content", th.find("reasoning here") != std::string::npos && bd == "the answer", bd);
        split_think("reasoning here\n</think>\n\n2+2=4", th, bd, true);
        check("think: prompt-open span separates non-streaming answer", th == "reasoning here\n" && bd == "2+2=4", bd);
        split_think("unfinished reasoning", th, bd, true);
        check("think: truncated prompt-open span has no final content", th == "unfinished reasoning" && bd.empty(), bd);
        split_think("2+2=4", th, bd, false);
        check("think: disabled thinking preserves plain content", th.empty() && bd == "2+2=4", bd);
    }
    {
        bool ok = false;
        jz::V msgs = jz::parse(R"([{"role":"user","content":"hi"}])", &ok);
        std::string p = render_prompt(msgs, {}, true, "high");
        check("template: thinking=true ends with an OPEN think span",
              p.size() > 20 && p.compare(p.size() - 21, 21, "<|im_start|>assistant\n<think>\n") != 0
                  ? p.find("<|im_start|>assistant\n<think>\n") != std::string::npos : true);
        check("template: reasoning_effort high -> medium injects NO instruction text",
              p.find("Reasoning effort is set to") == std::string::npos);
        std::string p2 = render_prompt(msgs, {}, false, "");
        check("template: thinking=false emits a CLOSED empty think span",
              p2.find("<|im_start|>assistant\n<think>\n\n</think>\n\n") != std::string::npos);
        std::string p3 = render_prompt(msgs, {}, true, "max");
        check("template: reasoning_effort max -> xhigh injects its sentence",
              p3.find("Reasoning effort is set to xhigh") != std::string::npos);
    }
    {
        // Consecutive tool results share ONE user turn (chat_template.jinja:147-158).
        bool ok = false;
        jz::V msgs = jz::parse(R"([{"role":"user","content":"q"},{"role":"tool","content":"a"},{"role":"tool","content":"b"}])", &ok);
        std::string p = render_prompt(msgs, {}, false, "");
        size_t n = 0, at = 0;
        while ((at = p.find("<|im_start|>user", at)) != std::string::npos) { ++n; at += 4; }
        check("template: two consecutive tool results share one user turn", n == 2, std::to_string(n));
    }
    {
        // Reasoning MUST be fed back, or every earlier turn stops matching.
        bool ok = false;
        jz::V msgs = jz::parse(R"([{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"because"},{"role":"user","content":"again"}])", &ok);
        std::string p = render_prompt(msgs, {}, false, "");
        check("template: assistant reasoning is fed back into the prompt",
              p.find("<think>\nbecause\n</think>") != std::string::npos);
    }
    {
        Session s;
        std::vector<unsigned> a{1,2,3,4,5};
        bool rst = false; size_t lcp = 0;
        s.committed(a);
        auto tail = s.plan({1,2,3,4,5,6,7}, rst, lcp);
        check("session: pure append reuses everything", !rst && lcp == 5 && tail.size() == 2, std::to_string(tail.size()));
        s.committed(a);
        tail = s.plan({1,2,3,9,9}, rst, lcp);
        check("session: divergence reports the true LCP (3), not zero", lcp == 3, std::to_string(lcp));
    }
    check("decode guard: padding ids above the tokenizer vocabulary are not decoded", !decodable(248077u) && decodable(248046u));

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
