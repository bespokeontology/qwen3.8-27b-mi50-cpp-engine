// Minimal JSON reader/writer for the Q27 C++ front end. No third-party libraries (project law).
// Only what an OpenAI chat-completions surface needs: objects, arrays, strings, numbers, bools,
// null, with exact round-tripping of string escapes and NUMBER TYPES PRESERVED.
//
// Type preservation is not a nicety here. The Python front end this replaces emitted every tool
// argument as a JSON string, so any tool declaring "todos": array or "offset": number rejected the
// call outright ("invalid arguments: \"offset\" must be a number"). Values carry their type from
// parse to serialise.
#pragma once
#include <cstdint>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace jz {

enum class T { Null, Bool, Num, Str, Arr, Obj };

struct V;
using Arr = std::vector<V>;
// Ordered object: OpenAI clients do not care about key order, but stable output makes diffs and
// transcripts readable, so insertion order is kept rather than sorted.
using Obj = std::vector<std::pair<std::string, V>>;

struct V {
    T t = T::Null;
    bool b = false;
    double n = 0;
    bool n_is_int = false;          // print 3 rather than 3.0 when the source had no fraction
    std::string s;
    std::shared_ptr<Arr> a;
    std::shared_ptr<Obj> o;

    V() = default;
    static V null()                 { return V(); }
    static V boolean(bool x)        { V v; v.t = T::Bool; v.b = x; return v; }
    static V num(double x, bool i)  { V v; v.t = T::Num; v.n = x; v.n_is_int = i; return v; }
    static V integer(long long x)   { return num((double)x, true); }
    static V str(std::string x)     { V v; v.t = T::Str; v.s = std::move(x); return v; }
    static V arr()                  { V v; v.t = T::Arr; v.a = std::make_shared<Arr>(); return v; }
    static V obj()                  { V v; v.t = T::Obj; v.o = std::make_shared<Obj>(); return v; }

    bool is_obj() const { return t == T::Obj && o; }
    bool is_arr() const { return t == T::Arr && a; }
    bool is_str() const { return t == T::Str; }
    bool is_num() const { return t == T::Num; }

    // Object access. Returns a null V when absent, so callers can chain without checking.
    const V& operator[](const std::string& k) const {
        static const V nil;
        if (!is_obj()) return nil;
        for (const auto& kv : *o) if (kv.first == k) return kv.second;
        return nil;
    }
    bool has(const std::string& k) const {
        if (!is_obj()) return false;
        for (const auto& kv : *o) if (kv.first == k) return true;
        return false;
    }
    void set(const std::string& k, V v) {
        if (!is_obj()) { t = T::Obj; o = std::make_shared<Obj>(); }
        for (auto& kv : *o) if (kv.first == k) { kv.second = std::move(v); return; }
        o->emplace_back(k, std::move(v));
    }
    void push(V v) {
        if (!is_arr()) { t = T::Arr; a = std::make_shared<Arr>(); }
        a->push_back(std::move(v));
    }
    size_t size() const { return is_arr() ? a->size() : (is_obj() ? o->size() : 0); }
    const V& at(size_t i) const { static const V nil; return (is_arr() && i < a->size()) ? (*a)[i] : nil; }

    std::string as_str(const std::string& d = "") const { return is_str() ? s : d; }
    double      as_num(double d = 0) const { return is_num() ? n : d; }
    long long   as_int(long long d = 0) const { return is_num() ? (long long)llround(n) : d; }
    bool        as_bool(bool d = false) const { return t == T::Bool ? b : d; }
};

// ---------------------------------------------------------------- serialise

inline void esc(const std::string& in, std::string& out) {
    out += '"';
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += (char)c;
        }
    }
    out += '"';
}

inline void dump(const V& v, std::string& out) {
    switch (v.t) {
        case T::Null: out += "null"; break;
        case T::Bool: out += v.b ? "true" : "false"; break;
        case T::Num: {
            char buf[40];
            if (v.n_is_int && std::fabs(v.n) < 9.2e18) snprintf(buf, sizeof buf, "%lld", (long long)llround(v.n));
            else {
                snprintf(buf, sizeof buf, "%.17g", v.n);
                // trim to the shortest representation that round-trips
                for (int p = 1; p <= 17; ++p) {
                    char t2[40]; snprintf(t2, sizeof t2, "%.*g", p, v.n);
                    if (strtod(t2, nullptr) == v.n) { snprintf(buf, sizeof buf, "%s", t2); break; }
                }
            }
            out += buf; break;
        }
        case T::Str: esc(v.s, out); break;
        case T::Arr: {
            out += '[';
            if (v.a) for (size_t i = 0; i < v.a->size(); ++i) { if (i) out += ','; dump((*v.a)[i], out); }
            out += ']'; break;
        }
        case T::Obj: {
            out += '{';
            if (v.o) { bool first = true; for (const auto& kv : *v.o) { if (!first) out += ','; first = false; esc(kv.first, out); out += ':'; dump(kv.second, out); } }
            out += '}'; break;
        }
    }
}

inline std::string dump(const V& v) { std::string s; dump(v, s); return s; }

// ---------------------------------------------------------------- parse

struct P {
    const char* p; const char* e; bool ok = true;
    explicit P(const std::string& s) : p(s.data()), e(s.data() + s.size()) {}
    void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    bool lit(const char* w) { size_t n = strlen(w); if ((size_t)(e - p) < n || memcmp(p, w, n)) return false; p += n; return true; }

    static void utf8(unsigned cp, std::string& out) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    }

    std::string str() {
        std::string out;
        if (p >= e || *p != '"') { ok = false; return out; }
        ++p;
        while (p < e && *p != '"') {
            if (*p == '\\') {
                ++p; if (p >= e) { ok = false; return out; }
                switch (*p) {
                    case '"': out += '"'; break;   case '\\': out += '\\'; break;
                    case '/': out += '/'; break;   case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;  case 't': out += '\t'; break;
                    case 'b': out += '\b'; break;  case 'f': out += '\f'; break;
                    case 'u': {
                        if (e - p < 5) { ok = false; return out; }
                        unsigned cp = (unsigned)strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16);
                        p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && e - p >= 7 && p[1] == '\\' && p[2] == 'u') {
                            unsigned lo = (unsigned)strtoul(std::string(p + 3, p + 7).c_str(), nullptr, 16);
                            if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
                        }
                        utf8(cp, out); break;
                    }
                    default: ok = false; return out;
                }
                ++p;
            } else out += *p++;
        }
        if (p >= e) { ok = false; return out; }
        ++p;
        return out;
    }

    V val() {
        ws();
        if (p >= e) { ok = false; return V(); }
        switch (*p) {
            case 'n': if (lit("null")) return V(); ok = false; return V();
            case 't': if (lit("true")) return V::boolean(true); ok = false; return V();
            case 'f': if (lit("false")) return V::boolean(false); ok = false; return V();
            case '"': return V::str(str());
            case '[': {
                ++p; V v = V::arr(); ws();
                if (p < e && *p == ']') { ++p; return v; }
                for (;;) { v.push(val()); if (!ok) return v; ws();
                    if (p < e && *p == ',') { ++p; continue; }
                    if (p < e && *p == ']') { ++p; return v; }
                    ok = false; return v; }
            }
            case '{': {
                ++p; V v = V::obj(); ws();
                if (p < e && *p == '}') { ++p; return v; }
                for (;;) { ws(); std::string k = str(); if (!ok) return v; ws();
                    if (p >= e || *p != ':') { ok = false; return v; }
                    ++p; v.set(k, val()); if (!ok) return v; ws();
                    if (p < e && *p == ',') { ++p; continue; }
                    if (p < e && *p == '}') { ++p; return v; }
                    ok = false; return v; }
            }
            default: {
                const char* st = p;
                if (p < e && (*p == '-' || *p == '+')) ++p;
                bool isint = true;
                while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
                    if (*p == '.' || *p == 'e' || *p == 'E') isint = false;
                    ++p;
                }
                if (p == st) { ok = false; return V(); }
                return V::num(strtod(std::string(st, p).c_str(), nullptr), isint);
            }
        }
    }
};

inline V parse(const std::string& s, bool* ok = nullptr) {
    P pp(s); V v = pp.val(); pp.ws();
    if (ok) *ok = pp.ok;
    return v;
}

}  // namespace jz
