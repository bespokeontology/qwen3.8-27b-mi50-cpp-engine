# Authority Matrix

All rows produced by one binary (`q27_gen`, sha256 `497227d4…`), same environment,
differing only in speculation flags. 256 new tokens requested per run; runs that emit an
end-of-sequence token stop earlier and the actual committed count is used.
Authority metric: committed output tokens / decode wall clock.

Column key: `P1/P2/P3` = P(accept >= 1/2/3 drafts). `round_ms` = mean speculative round.
`draft_ms`, `layers_ms`, `headsync_ms`, `tail_ms` are per-round phase means on card 0;
for DFlash2 the block executor's cost appears in `tail_ms`, not `draft_ms`.
Blank cells: not run, or not applicable (plain has no speculative statistics).

```
workload  prompt_tok  path   K  prefill_ms  rounds  tok_per_round  P1     P2     P3     round_ms  draft_ms  layers_ms  headsync_ms  tail_ms  decode_ms  committed_tok_s
prose1k   1024        plain  0  577.6       3719.2  68.559                                                                                              
prose1k   1024        chain  2  626.1       105     2.448          0.800  0.648  0.000  23.579    2.591     20.152     0.455        0.256    2475.8     103.737
prose1k   1024        chain  3  624.6       94      2.723          0.766  0.596  0.362  33.048    3.692     15.045     13.713       0.365    3106.6     82.281
prose1k   1024        chain  7  625.5       87      2.966          0.701  0.494  0.322  50.151    7.483     14.559     27.134       0.592    4363.2     59.131
prose1k   1024        df2    3  634.0       124     2.065          0.548  0.331  0.185  54.287    7.922     13.963     14.880       17.473   6731.7     38.004
prose1k   1024        df2    7  651.3       114     2.246          0.553  0.316  0.202  68.386    8.583     15.144     26.732       17.840   7796.0     32.798
code483   483         plain  0  489.2       3667.7  69.524                                                                                              
code483   483         chain  2  526.9       105     2.438          0.819  0.619  0.000  23.844    2.548     20.446     0.456        0.275    2503.7     102.237
code483   483         chain  7  526.4       71      3.606          0.732  0.577  0.451  50.430    7.679     14.833     27.010       0.577    3580.6     71.497
code483   483         df2    3  537.7       125     2.056          0.560  0.304  0.192  45.695    3.642     14.556     14.202       13.249   5711.9     44.964
code483   483         df2    7  547.9       112     2.277          0.562  0.312  0.205  59.498    4.129     14.518     27.156       13.609   6663.9     38.197
prose8k   8191        plain  0  6413.0      1720.8  61.597                                                                                              
prose8k   8191        chain  2  6484.2      36      3.000          1.000  1.000  0.000  26.407    4.405     21.419     0.455        0.030    950.7      113.520
prose8k   8191        chain  3  6482.9      27      3.926          1.000  1.000  0.926  38.333    6.448     14.294     17.333       0.074    1035.0     102.415
prose8k   8191        chain  7  6483.6      18      5.889          1.000  1.000  0.944  59.258    12.211    15.902     30.130       0.518    1066.7     99.375
prose8k   8191        df2    3                                                                                                                          
prose8k   8191        df2    7                                                                                                                          
code6k8   6782        chain  2  5088.3      120     2.133          0.700  0.433  0.000  25.135    2.922     21.183     0.455        0.428    3016.2     84.801
code6k8   6782        chain  7  5035.0      110     2.327          0.600  0.382  0.218  54.485    8.279     14.517     30.748       0.561    5993.4     42.674
code6k8   6782        df2    3  5231.2      157     1.637          0.401  0.159  0.076  95.838    40.717    15.144     15.769       24.149   15046.7    17.078
code6k8   6782        df2    7                                                                                                                          
```

## Failed runs

`prose8k df2 K=3`, `prose8k df2 K=7`, `code6k8 df2 K=7` aborted. The DFlash2 draft block
cost rises to 8,020 ms at position 8,204 and the round then trips the staging done-flag
guard. See Limitations in the README. Chained and plain paths complete normally at 8K.

## Prompt characteristics

| prompt | tokens | distinct ids | lexical diversity |
|---|---:|---:|---:|
| p1k | 1024 | 386 | 37.7% |
| p8k | 8191 | 444 | 5.4% |
| pcode | 483 | 237 | 49.1% |
| pcode8k | 6782 | 848 | 12.5% |

The low diversity of `p8k` (5.4%) is why the chained drafter reaches P(>=1) = 1.000 on it.
Those rows measure an easy-prediction regime and are not representative.
