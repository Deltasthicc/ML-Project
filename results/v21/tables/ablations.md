### Novelties 3, 4 and the new augmentation - H_tglite70k (~66K, full v2.1 recipe) minus ablation (seed-paired over the shared seeds; tuned mode; Betti-0 lower is better, so a negative Betti-0 difference favours H_tglite70k)

| ablation (removed) | run | headline - ablation: unseen Dice | headline - ablation: unseen clDice | headline - ablation: unseen Betti-0 (raw) | headline - ablation: gap |
|---|---|---|---|---|---|
| novelty 3 (skeleton distillation) | Z_no_kdskel | -0.0051 [±0.0148] not clear (n=3) | +0.0019 [±0.0073] not clear (n=3) | -11.4963 [±15.7837] not clear (n=3) | +0.0021 [±0.0129] not clear (n=3) |
| all distillation | Z_no_kd | -0.0033 [±0.0044] not clear (n=3) | +0.0068 [±0.0114] not clear (n=3) | -9.1538 [±12.5467] not clear (n=3) | +0.0023 [±0.0040] not clear (n=3) |
| novelty 4 (skeleton consistency) | Z_pixcons | -0.0091 [±0.0045] clear (n=3) | +0.0011 [±0.0105] not clear (n=3) | -3.9476 [±20.0068] not clear (n=3) | +0.0088 [±0.0044] clear (n=3) |
| new augmentation (RandConv + Fourier) | Z_noaug | -0.0094 [±0.0267] not clear (n=3) | +0.0053 [±0.0108] not clear (n=3) | +3.2969 [±19.8757] not clear (n=3) | +0.0091 [±0.0244] not clear (n=3) |
| all v2.1 training changes | Z_v2recipe | -0.0081 [±0.0188] not clear (n=3) | +0.0015 [±0.0143] not clear (n=3) | -9.2167 [±16.5123] not clear (n=3) | +0.0026 [±0.0160] not clear (n=3) |