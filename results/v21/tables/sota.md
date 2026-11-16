### Published results (context only - training data, scoring resolution and thresholds differ) vs ours
### Dice / clDice in %, 'unseen' = mean of STARE, CHASE_DB1, HRF. Ours: threshold chosen on DRIVE-val, native-resolution scoring (network-resolution in brackets).

| model | protocol | params | gflops | stare | chase_db1 | hrf | unseen | cldice | betti0 | uncertainty | unseen_scale_sel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DSAE-Net [19] | cross-dataset tests | 78K | 4.76 | 81.92 | 81.46 | 80.12 | 81.17 | - | - | no | - |
| LightVesselNet [12] | train one, test four | 75K | 1.4 | - | - | - | - | - | - | no | - |
| TFFM [21] | source-only (Fundus-AVSeg) | n/r | n/r | - | - | - | 79.9-82.2 | 68.7-73.6 | 24.2-50.5 | no | - |
| DGSSA [15] | multi-source DG | n/r | n/r | - | - | - | 77.98 | - | - | no | - |
| WaveRNet [16] | multi-source DG (no HRF) | >90M | n/r | - | - | - | 69.49 | - | - | no | - |
| TG-LiteNet v1 (E, MC) | source-only DRIVE | 0.21M | ~5.5 | - | - | - | 75.6 | 83.3 | 42.7 | MC dropout | - |
| TG-LiteNet v2.1 ~66K (H_tglite70k) | source-only DRIVE (native res.) | 66K | 2.73 | 81.0 | 76.5 | 72.9 | 76.8 (net-res 75.7) | 82.0 | 32.1 | connectivity score + entropy | 77.9 |
| v2.1 ~66K, 5-seed ensemble | source-only DRIVE (native res.) | 66K | 2.73 | 81.7 | 77.2 | 73.0 | 77.3 (net-res 76.4) | 82.7 | 26.5 | connectivity score + entropy | 78.5 |
| v2.1 R_lite_w16 (size-rule headline) | source-only DRIVE (native res.) | 212K | 5.53 | 81.2 | 77.7 | 73.6 | 77.5 (net-res 76.7) | 82.6 | 29.0 | connectivity score + entropy | 78.5 |
| v2.1 R_unet_w8 | source-only DRIVE (native res.) | 486K | 6.05 | 81.2 | 77.5 | 73.6 | 77.5 (net-res 76.3) | 82.6 | 31.8 | connectivity score + entropy | 78.5 |
| TG-LiteNet v2 (E) | source-only DRIVE (native res.) | 486K | 6.05 | 81.9 | 78.7 | 73.3 | 78.0 (net-res 77.0) | 82.5 | 39.2 | connectivity score + entropy | 78.9 |
| v2 stage C U-Net (31M) | source-only DRIVE (native res.) | 31036K | 384.74 | 82.8 | 78.9 | 74.9 | 78.9 (net-res 77.7) | 83.6 | 35.5 | connectivity score + entropy | 75.1 |