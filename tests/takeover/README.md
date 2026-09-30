# Corrected production kernel regression checks

Build with `make component-build` from the repository root. Stop model services before running GPU checks; tests run sequentially on the three devices.

```bash
./tests/takeover/check_flags
./tests/takeover/pack_gu_pair --selftest
./tests/takeover/check_attn
./tests/takeover/check_gdn
./native/q27_fe --selftest
```

The tests include actual product HIP source, with the separate frozen quantizer body as the oracle. Attention covers **108 cases** across three GPUs, two head counts, NR1/3/4, 2/7/128 splits, and zero/asymmetric input. GDN covers **54 cases** across three GPUs, ragged head spans, NR1/3/4 and zero/asymmetric state. Full outputs, codes, scales, recurrent state, snapshots, padding and guards compare exactly.

The old fused boundary produced **43956 attention** and **15792 GDN** code/scale mismatches in these fixtures. Restoring BF16 rounding made all 162 cases exact. Original before/after logs are under `receipts/corrected-20260929/components/`.

Flag checks cover invalid consumer combinations and valid controls. The nibble packer covers **65536** packed-byte combinations plus three row shapes, and rejects four invalid inputs. It is a CPU layout test, not a GPU throughput result.

The older tests in the parent directory describe historical four-card fixtures. These targeted kernel checks do not replace task, continuation, rewind, reset, long-context or transport qualification.
