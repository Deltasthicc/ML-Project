### Novelty 2 - centreline-calibre-guided test-time scale selection (scales 1, 1.5, 2, 2.5; the scale whose predicted vessel calibre, measured on the prediction's own skeleton, is closest to the model's DRIVE-train calibre wins). Unseen Dice in the tuned mode at scale 1 vs with selection; delta = selected - scale 1 (seed-paired).

| model | n | unseen tuned (scale 1) | unseen scale-selected | delta unseen | delta hrf | delta stare | delta chase_db1 | delta fives | delta drive |
|---|---|---|---|---|---|---|---|---|---|
| ENS_H_tglite70k[ens] | 1 | 0.773 | 0.785 | +0.0119 (n=1) | +0.0358 | +0.0000 | +0.0000 | -0.0683 | +0.0000 |
| H_tglite70k[det] | 5 | 0.768±0.004 | 0.779±0.004 | +0.0114 [±0.0005] clear (n=5) | +0.0343 | +0.0000 | +0.0000 | -0.0608 | +0.0000 |
| R_lite_w16[det] | 3 | 0.775±0.011 | 0.785±0.008 | +0.0094 [±0.0074] clear (n=3) | +0.0276 | +0.0007 | +0.0000 | -0.0259 | +0.0000 |
| R_unet_w8[det] | 3 | 0.775±0.010 | 0.785±0.009 | +0.0102 [±0.0076] clear (n=3) | +0.0307 | +0.0000 | +0.0000 | -0.0526 | +0.0000 |
| Z_no_kd[det] | 3 | 0.772±0.002 | 0.779±0.006 | +0.0074 [±0.0092] not clear (n=3) | +0.0297 | +0.0000 | -0.0076 | -0.0673 | +0.0000 |
| Z_no_kdskel[det] | 3 | 0.774±0.004 | 0.773±0.007 | -0.0003 [±0.0153] not clear (n=3) | +0.0274 | +0.0000 | -0.0283 | -0.0669 | +0.0000 |
| Z_noaug[det] | 3 | 0.778±0.007 | 0.780±0.008 | +0.0023 [±0.0334] not clear (n=3) | +0.0258 | +0.0000 | -0.0190 | -0.0566 | +0.0000 |
| Z_pixcons[det] | 3 | 0.778±0.005 | 0.773±0.009 | -0.0049 [±0.0197] not clear (n=3) | +0.0301 | +0.0000 | -0.0449 | -0.0660 | +0.0000 |
| Z_v2recipe[det] | 3 | 0.777±0.005 | 0.782±0.007 | +0.0056 [±0.0158] not clear (n=3) | +0.0282 | +0.0000 | -0.0113 | -0.0504 | +0.0000 |
| A_unet[det] | 5 | 0.762±0.015 | 0.742±0.034 | -0.0197 [±0.0446] not clear (n=5) | -0.0044 | +0.0000 | -0.0546 | -0.0400 | +0.0000 |
| C_unet_topo_dg[det] | 5 | 0.789±0.008 | 0.751±0.034 | -0.0383 [±0.0349] clear (n=5) | +0.0160 | +0.0001 | -0.1309 | -0.0510 | +0.0000 |
| D_compact[det] | 5 | 0.780±0.002 | 0.772±0.035 | -0.0072 [±0.0428] not clear (n=5) | +0.0265 | +0.0000 | -0.0481 | -0.0350 | +0.0000 |
| ENS_A_unet[ens] | 1 | 0.764 | 0.762 | -0.0020 (n=1) | -0.0060 | +0.0000 | +0.0000 | -0.0262 | +0.0000 |
| ENS_D_compact[ens] | 1 | 0.783 | 0.792 | +0.0089 (n=1) | +0.0266 | +0.0000 | +0.0000 | -0.0312 | +0.0000 |
| E_tglitenet[det] | 5 | 0.780±0.007 | 0.789±0.005 | +0.0095 [±0.0023] clear (n=5) | +0.0286 | +0.0000 | +0.0000 | -0.0413 | +0.0000 |
| E_tglitenet[mc] | 5 | 0.778±0.007 | 0.786±0.005 | +0.0085 [±0.0057] clear (n=5) | +0.0311 | +0.0000 | -0.0056 | -0.0407 | +0.0000 |

### DRIVE-val check of the rule (rule used for claims only if it passed)

* A_unet seed0 det: val Dice 0.8225 -> 0.8225, passed  (shrink 0.67: rule +0.005 / oracle +0.006; shrink 0.5: rule +0.006 / oracle +0.010; shrink 0.4: rule +0.019 / oracle +0.025)
* A_unet seed1 det: val Dice 0.8194 -> 0.8194, passed  (shrink 0.67: rule +0.001 / oracle +0.006; shrink 0.5: rule +0.005 / oracle +0.011; shrink 0.4: rule +0.009 / oracle +0.021)
* A_unet seed2 det: val Dice 0.8212 -> 0.8212, passed  (shrink 0.67: rule -0.009 / oracle +0.001; shrink 0.5: rule -0.007 / oracle +0.001; shrink 0.4: rule -0.013 / oracle +0.000)
* A_unet seed3 det: val Dice 0.8203 -> 0.8203, passed  (shrink 0.67: rule -0.011 / oracle +0.000; shrink 0.5: rule -0.017 / oracle +0.001; shrink 0.4: rule -0.026 / oracle +0.000)
* A_unet seed4 det: val Dice 0.8185 -> 0.8185, passed  (shrink 0.67: rule -0.007 / oracle +0.001; shrink 0.5: rule -0.006 / oracle +0.001; shrink 0.4: rule -0.005 / oracle +0.007)
* C_unet_topo_dg seed0 det: val Dice 0.8240 -> 0.8240, passed  (shrink 0.67: rule +0.001 / oracle +0.006; shrink 0.5: rule +0.009 / oracle +0.014; shrink 0.4: rule +0.040 / oracle +0.042)
* C_unet_topo_dg seed1 det: val Dice 0.8224 -> 0.8224, passed  (shrink 0.67: rule +0.006 / oracle +0.009; shrink 0.5: rule +0.013 / oracle +0.018; shrink 0.4: rule +0.043 / oracle +0.047)
* C_unet_topo_dg seed2 det: val Dice 0.8217 -> 0.8217, passed  (shrink 0.67: rule -0.008 / oracle +0.003; shrink 0.5: rule -0.000 / oracle +0.008; shrink 0.4: rule +0.033 / oracle +0.036)
* C_unet_topo_dg seed3 det: val Dice 0.8252 -> 0.8252, passed  (shrink 0.67: rule +0.009 / oracle +0.010; shrink 0.5: rule +0.021 / oracle +0.023; shrink 0.4: rule +0.048 / oracle +0.052)
* C_unet_topo_dg seed4 det: val Dice 0.8238 -> 0.8238, passed  (shrink 0.67: rule -0.008 / oracle +0.003; shrink 0.5: rule +0.002 / oracle +0.009; shrink 0.4: rule +0.030 / oracle +0.035)
* D_compact seed0 det: val Dice 0.8185 -> 0.8185, passed  (shrink 0.67: rule +0.009 / oracle +0.014; shrink 0.5: rule +0.028 / oracle +0.031; shrink 0.4: rule +0.073 / oracle +0.077)
* D_compact seed1 det: val Dice 0.8158 -> 0.8158, passed  (shrink 0.67: rule +0.008 / oracle +0.013; shrink 0.5: rule +0.023 / oracle +0.028; shrink 0.4: rule +0.068 / oracle +0.072)
* D_compact seed2 det: val Dice 0.8166 -> 0.8166, passed  (shrink 0.67: rule +0.005 / oracle +0.009; shrink 0.5: rule +0.020 / oracle +0.024; shrink 0.4: rule +0.058 / oracle +0.063)
* D_compact seed3 det: val Dice 0.8172 -> 0.8172, passed  (shrink 0.67: rule +0.002 / oracle +0.008; shrink 0.5: rule +0.017 / oracle +0.023; shrink 0.4: rule +0.052 / oracle +0.056)
* D_compact seed4 det: val Dice 0.8181 -> 0.8181, passed  (shrink 0.67: rule +0.009 / oracle +0.013; shrink 0.5: rule +0.025 / oracle +0.028; shrink 0.4: rule +0.068 / oracle +0.072)
* E_tglitenet seed0 det: val Dice 0.8189 -> 0.8189, passed  (shrink 0.67: rule +0.005 / oracle +0.011; shrink 0.5: rule +0.022 / oracle +0.027; shrink 0.4: rule +0.067 / oracle +0.070)
* E_tglitenet seed0 mc: val Dice 0.8192 -> 0.8192, passed  (shrink 0.67: rule +0.004 / oracle +0.012; shrink 0.5: rule +0.020 / oracle +0.025; shrink 0.4: rule +0.064 / oracle +0.068)
* E_tglitenet seed1 det: val Dice 0.8161 -> 0.8161, passed  (shrink 0.67: rule +0.006 / oracle +0.012; shrink 0.5: rule +0.023 / oracle +0.027; shrink 0.4: rule +0.073 / oracle +0.076)
* E_tglitenet seed1 mc: val Dice 0.8144 -> 0.8144, passed  (shrink 0.67: rule +0.008 / oracle +0.011; shrink 0.5: rule +0.020 / oracle +0.025; shrink 0.4: rule +0.073 / oracle +0.076)
* E_tglitenet seed2 det: val Dice 0.8163 -> 0.8163, passed  (shrink 0.67: rule +0.009 / oracle +0.014; shrink 0.5: rule +0.023 / oracle +0.027; shrink 0.4: rule +0.063 / oracle +0.068)
* E_tglitenet seed2 mc: val Dice 0.8150 -> 0.8150, passed  (shrink 0.67: rule +0.013 / oracle +0.017; shrink 0.5: rule +0.030 / oracle +0.033; shrink 0.4: rule +0.066 / oracle +0.071)
* E_tglitenet seed3 det: val Dice 0.8143 -> 0.8143, passed  (shrink 0.67: rule +0.017 / oracle +0.019; shrink 0.5: rule +0.041 / oracle +0.044; shrink 0.4: rule +0.093 / oracle +0.099)
* E_tglitenet seed3 mc: val Dice 0.8147 -> 0.8147, passed  (shrink 0.67: rule +0.018 / oracle +0.020; shrink 0.5: rule +0.042 / oracle +0.044; shrink 0.4: rule +0.096 / oracle +0.102)
* E_tglitenet seed4 det: val Dice 0.8182 -> 0.8182, passed  (shrink 0.67: rule +0.002 / oracle +0.009; shrink 0.5: rule +0.019 / oracle +0.026; shrink 0.4: rule +0.062 / oracle +0.065)
* E_tglitenet seed4 mc: val Dice 0.8176 -> 0.8176, passed  (shrink 0.67: rule +0.008 / oracle +0.012; shrink 0.5: rule +0.028 / oracle +0.032; shrink 0.4: rule +0.069 / oracle +0.074)
* H_tglite70k seed0 det: val Dice 0.8129 -> 0.8129, passed  (shrink 0.67: rule +0.012 / oracle +0.017; shrink 0.5: rule +0.032 / oracle +0.038; shrink 0.4: rule +0.091 / oracle +0.096)
* H_tglite70k seed1 det: val Dice 0.8115 -> 0.8115, passed  (shrink 0.67: rule +0.007 / oracle +0.018; shrink 0.5: rule +0.028 / oracle +0.035; shrink 0.4: rule +0.082 / oracle +0.085)
* H_tglite70k seed2 det: val Dice 0.8112 -> 0.8112, passed  (shrink 0.67: rule +0.012 / oracle +0.018; shrink 0.5: rule +0.032 / oracle +0.045; shrink 0.4: rule +0.090 / oracle +0.093)
* H_tglite70k seed3 det: val Dice 0.8139 -> 0.8139, passed  (shrink 0.67: rule +0.011 / oracle +0.016; shrink 0.5: rule +0.033 / oracle +0.038; shrink 0.4: rule +0.094 / oracle +0.097)
* H_tglite70k seed4 det: val Dice 0.8118 -> 0.8118, passed  (shrink 0.67: rule +0.016 / oracle +0.018; shrink 0.5: rule +0.038 / oracle +0.042; shrink 0.4: rule +0.088 / oracle +0.097)
* R_lite_w16 seed0 det: val Dice 0.8142 -> 0.8142, passed  (shrink 0.67: rule +0.011 / oracle +0.016; shrink 0.5: rule +0.033 / oracle +0.036; shrink 0.4: rule +0.077 / oracle +0.081)
* R_lite_w16 seed1 det: val Dice 0.8146 -> 0.8146, passed  (shrink 0.67: rule +0.007 / oracle +0.011; shrink 0.5: rule +0.023 / oracle +0.027; shrink 0.4: rule +0.069 / oracle +0.073)
* R_lite_w16 seed2 det: val Dice 0.8160 -> 0.8160, passed  (shrink 0.67: rule -0.006 / oracle +0.005; shrink 0.5: rule +0.008 / oracle +0.016; shrink 0.4: rule +0.056 / oracle +0.057)
* R_unet_w8 seed0 det: val Dice 0.8153 -> 0.8153, passed  (shrink 0.67: rule -0.002 / oracle +0.008; shrink 0.5: rule +0.010 / oracle +0.023; shrink 0.4: rule +0.072 / oracle +0.073)
* R_unet_w8 seed1 det: val Dice 0.8143 -> 0.8143, passed  (shrink 0.67: rule +0.005 / oracle +0.013; shrink 0.5: rule +0.021 / oracle +0.028; shrink 0.4: rule +0.081 / oracle +0.084)
* R_unet_w8 seed2 det: val Dice 0.8112 -> 0.8112, passed  (shrink 0.67: rule +0.013 / oracle +0.015; shrink 0.5: rule +0.032 / oracle +0.035; shrink 0.4: rule +0.077 / oracle +0.081)
* Z_no_kd seed0 det: val Dice 0.8138 -> 0.8138, passed  (shrink 0.67: rule +0.009 / oracle +0.016; shrink 0.5: rule +0.028 / oracle +0.035; shrink 0.4: rule +0.080 / oracle +0.087)
* Z_no_kd seed1 det: val Dice 0.8101 -> 0.8101, passed  (shrink 0.67: rule +0.004 / oracle +0.013; shrink 0.5: rule +0.023 / oracle +0.034; shrink 0.4: rule +0.088 / oracle +0.091)
* Z_no_kd seed2 det: val Dice 0.8108 -> 0.8108, passed  (shrink 0.67: rule +0.009 / oracle +0.016; shrink 0.5: rule +0.033 / oracle +0.038; shrink 0.4: rule +0.082 / oracle +0.090)
* Z_no_kdskel seed0 det: val Dice 0.8127 -> 0.8127, passed  (shrink 0.67: rule +0.006 / oracle +0.012; shrink 0.5: rule +0.024 / oracle +0.031; shrink 0.4: rule +0.080 / oracle +0.083)
* Z_no_kdskel seed1 det: val Dice 0.8107 -> 0.8107, passed  (shrink 0.67: rule +0.010 / oracle +0.015; shrink 0.5: rule +0.028 / oracle +0.036; shrink 0.4: rule +0.085 / oracle +0.089)
* Z_no_kdskel seed2 det: val Dice 0.8130 -> 0.8130, passed  (shrink 0.67: rule +0.007 / oracle +0.012; shrink 0.5: rule +0.030 / oracle +0.033; shrink 0.4: rule +0.081 / oracle +0.085)
* Z_noaug seed0 det: val Dice 0.8124 -> 0.8124, passed  (shrink 0.67: rule +0.003 / oracle +0.011; shrink 0.5: rule +0.026 / oracle +0.033; shrink 0.4: rule +0.082 / oracle +0.084)
* Z_noaug seed1 det: val Dice 0.8089 -> 0.8089, passed  (shrink 0.67: rule +0.009 / oracle +0.014; shrink 0.5: rule +0.030 / oracle +0.034; shrink 0.4: rule +0.091 / oracle +0.094)
* Z_noaug seed2 det: val Dice 0.8127 -> 0.8127, passed  (shrink 0.67: rule -0.006 / oracle +0.006; shrink 0.5: rule +0.009 / oracle +0.018; shrink 0.4: rule +0.065 / oracle +0.066)
* Z_pixcons seed0 det: val Dice 0.8138 -> 0.8138, passed  (shrink 0.67: rule +0.004 / oracle +0.012; shrink 0.5: rule +0.023 / oracle +0.033; shrink 0.4: rule +0.081 / oracle +0.084)
* Z_pixcons seed1 det: val Dice 0.8133 -> 0.8133, passed  (shrink 0.67: rule -0.005 / oracle +0.010; shrink 0.5: rule +0.014 / oracle +0.025; shrink 0.4: rule +0.069 / oracle +0.070)
* Z_pixcons seed2 det: val Dice 0.8117 -> 0.8117, passed  (shrink 0.67: rule +0.013 / oracle +0.018; shrink 0.5: rule +0.038 / oracle +0.040; shrink 0.4: rule +0.091 / oracle +0.095)
* Z_v2recipe seed0 det: val Dice 0.8172 -> 0.8172, passed  (shrink 0.67: rule +0.006 / oracle +0.013; shrink 0.5: rule +0.024 / oracle +0.030; shrink 0.4: rule +0.072 / oracle +0.075)
* Z_v2recipe seed1 det: val Dice 0.8140 -> 0.8140, passed  (shrink 0.67: rule +0.009 / oracle +0.013; shrink 0.5: rule +0.033 / oracle +0.036; shrink 0.4: rule +0.081 / oracle +0.087)
* Z_v2recipe seed2 det: val Dice 0.8126 -> 0.8126, passed  (shrink 0.67: rule +0.008 / oracle +0.013; shrink 0.5: rule +0.029 / oracle +0.032; shrink 0.4: rule +0.071 / oracle +0.076)