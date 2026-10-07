# Qwen3.8-27B native engine, AMD gfx906. ROCm 5.7.1.
# Real dependency generation (-MMD -MP): the estate has a recorded incident where stale objects
# and a changed struct offset masqueraded for a whole session as a GPU algorithm failure.
HIPCC   ?= /opt/rocm/bin/hipcc
ARCH    ?= gfx906
CH ?= 256                        # Q27_PF_CH: prefill chunk width (256 = the shipped layer-split geometry)
TSLOT ?= 256                     # Q27_PF_TSLOT: wide-FFN tile span (256 = per 256 positions, 1024 = whole layer)
EXP ?= 0                         # EXP=1 compiles the closed experimental kernels back in (they cost decode: receipt v93)
CXXFLAGS = -O3 -std=c++17 --offload-arch=$(ARCH) -Iinclude -Isrc/tok -MMD -MP -Wall -Wno-unused-function -DQ27_PF_CH=$(CH) -DQ27_PF_TSLOT=$(TSLOT) $(if $(filter 1,$(EXP)),-DQ27_EXPERIMENTS=1,)
LDFLAGS  = -lpcre2-8 -L/opt/rocm/lib -lrocblas

SRC  = src/q27_nvfp4.hip src/q27_fp8.hip src/q27_elem.hip src/q27_gdn.hip src/q27_attn.hip src/q27_wide.hip \
       src/q27_util.hip src/q27_epi.hip src/q27_rb.cpp \
       src/q27_load.cpp src/q27_dflash2.cpp src/q27_df2_run.cpp src/q27_main.cpp src/tok/hf_tokenizer.cpp
OBJ  = $(addsuffix .o,$(basename $(SRC)))
DEP  = $(OBJ:.o=.d)

all: p2p-gate q27_gen q27_tok

# native tokenizer driver (encode/decode/chat/golden), plain g++, no ROCm needed
q27_tok: tools/q27_tok.cpp src/tok/hf_tokenizer.cpp src/tok/hf_tokenizer.h src/tok/q27_chat.h
	g++ -O2 -std=c++17 -Isrc/tok tools/q27_tok.cpp src/tok/hf_tokenizer.cpp -lpcre2-8 -o $@

q27_gen: $(OBJ)
	$(HIPCC) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.hip
	$(HIPCC) $(CXXFLAGS) -c $< -o $@
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# per-file compile check, used by the build agents
check: $(OBJ)
	@echo "all objects built for $(ARCH)"

# ISA dump for the dependency audit: count s_waitcnt vmcnt(0) and saveexec, never instructions
isa/%.s: src/%.hip
	@mkdir -p isa
	$(HIPCC) $(CXXFLAGS) -S -o $@ -c $<

# ---------------------------------------------------------------------------
# P2P POISON GUARD. On this box P2P fails 0/12 directed pairs and fails
# DESTRUCTIVELY: six ordered pairs among cards 0,1,3 write into the SENDER's
# own VRAM. A collective that accidentally took a peer path would corrupt the
# sending card's weights mid-token while still looking coherent. Every
# cross-card byte in this engine must go through pinned host memory, so the
# absence of these symbols is a safety property and is enforced, not assumed.
# P2P POLICY (revised 2026-09-10). The original gate banned peer APIs AND
# hipMemcpyDeviceToDevice TOGETHER, on the strength of a ROCm 5.15.0-190 finding: a peer write landed
# at the destination offset inside the SENDER's own buffer and destroyed the sender's VRAM. Two things
# are now known.
#   1. That was 5.15.0-190. On 6.8.0-138 all 12 directed pairs are correct on three transports with
#      source-collateral clean (P2P_WINDOW2 receipt; independently re-verified on scratch buffers with
#      sender-integrity checks). The constraint carried its own lift condition -- re-measure on 6.8 --
#      which was met. Keeping the ban afterwards was a stale workaround promoted into architecture.
#   2. hipMemcpyDeviceToDevice is a SAME-DEVICE copy, never a peer operation. It was collateral damage
#      and is no longer flagged.
# Host staging remains the DEFAULT and the fallback: at the 20,480 B TP payload P2P still TIES it
# (33.92 vs 34.46 us rejected on an impossibility argument), because a ~7.8 us fixed cost per transfer
# is the entire transfer at that size. Peer use must sit behind a runtime self-test, not an assumption.
p2p-gate:
	@peer=$$(grep -rnE 'hipDeviceEnablePeerAccess|hipDeviceCanAccessPeer|hipMemcpyPeer|hipIpc' \
	  --include=*.hip --include=*.cpp --include=*.h src include 2>/dev/null | grep -v '^\s*//'); \
	if [ -n "$$peer" ]; then \
	  echo "p2p-gate: peer API present -- permitted on 6.8; host staging stays the fallback:"; \
	  echo "$$peer"; \
	else echo "p2p-gate: no peer API in the tree (host-staged path)"; fi

audit: p2p-gate isa/q27_nvfp4.s isa/q27_gdn.s isa/q27_attn.s
	@for f in $^; do \
	  echo "== $$f: vmcnt(0)=$$(grep -c 'vmcnt(0)' $$f) saveexec=$$(grep -c saveexec $$f) \
global_load=$$(grep -c global_load $$f) ds_bpermute=$$(grep -c ds_bpermute $$f)"; done

# Offline syntax check of the HOST TUs against tools/hipstub — no ROCm, no GPU needed.
# Run this after ANY edit to q27_main.cpp / q27_load.cpp so a typo cannot burn a card window.
syntax:
	@for f in src/q27_main.cpp src/q27_load.cpp src/q27_rb.cpp; do \
	  printf "%-22s " "$$f"; \
	  if g++ -fsyntax-only -std=c++17 -Iinclude -Isrc/tok -Itools/hipstub $$f 2>/tmp/q27syn.$$$$; then echo OK; \
	  else echo FAIL; cat /tmp/q27syn.$$$$; rm -f /tmp/q27syn.$$$$; exit 1; fi; rm -f /tmp/q27syn.$$$$; done

clean:
	rm -f $(OBJ) $(DEP) q27_gen; rm -rf isa
.PHONY: all check audit clean p2p-poison syntax
-include $(DEP)
