### INT8 (ONNX Runtime, static QDQ, calibrated on 8 DRIVE-train images) - H_tglite70k (~66K) seed 0

* CPU 512x512, 4 threads: fp32 16.3 ms -> int8 24.8 ms
* file size: 266 KB -> 109 KB

| dataset | Dice fp32 | Dice int8 | n |
|---|---|---|---|
| drive | 0.8105 | 0.8105 | 20 |
| stare | 0.8058 | 0.8063 | 20 |
| chase_db1 | 0.7633 | 0.7634 | 28 |
| hrf | 0.7313 | 0.7316 | 45 |
| fives | 0.8113 | 0.8123 | 200 |