### Novelty 1 - label-free reliability of whole images. Spearman rho between the score and the image error (1 - Dice, tuned mode), averaged over the unseen datasets, then over seeds; AURC gain = % of the random-to-oracle gap closed when the least reliable images are rejected first.
### Scores: rel_frag (fragment density, primary), rel_endpoints, rel_smallfrac (connectivity, label-free, single pass); ent_mean (mean entropy; MC / ensemble mean for [mc] / [ens]); var_mean (MC / ensemble variance).

| model | score | rho_mean | rho_sd | n | gain_mean | gain_sd |
|---|---|---|---|---|---|---|
| A_unet[det] | ent_mean | -0.021 | 0.091 | 5 | -9.2 | 6.1 |
| A_unet[det] | rel_endpoints | 0.110 | 0.132 | 5 | 5.6 | 8.7 |
| A_unet[det] | rel_frag | 0.063 | 0.125 | 5 | -0.9 | 12.3 |
| A_unet[det] | rel_smallfrac | 0.147 | 0.171 | 5 | 3.0 | 16.0 |
| C_unet_topo_dg[det] | ent_mean | 0.075 | 0.037 | 5 | -0.5 | 2.5 |
| C_unet_topo_dg[det] | rel_endpoints | 0.220 | 0.057 | 5 | 18.6 | 5.0 |
| C_unet_topo_dg[det] | rel_frag | 0.222 | 0.085 | 5 | 17.3 | 6.4 |
| C_unet_topo_dg[det] | rel_smallfrac | 0.325 | 0.089 | 5 | 27.3 | 7.1 |
| D_compact[det] | ent_mean | 0.075 | 0.039 | 5 | -1.4 | 4.3 |
| D_compact[det] | rel_endpoints | 0.182 | 0.088 | 5 | 15.5 | 8.4 |
| D_compact[det] | rel_frag | 0.093 | 0.082 | 5 | 7.6 | 8.5 |
| D_compact[det] | rel_smallfrac | 0.237 | 0.067 | 5 | 20.9 | 10.1 |
| ENS_A_unet[ens] | ent_mean | -0.022 | - | 1 | -7.2 | - |
| ENS_A_unet[ens] | rel_endpoints | 0.029 | - | 1 | -4.9 | - |
| ENS_A_unet[ens] | rel_frag | 0.036 | - | 1 | -8.1 | - |
| ENS_A_unet[ens] | rel_smallfrac | 0.038 | - | 1 | -11.6 | - |
| ENS_A_unet[ens] | var_mean | 0.076 | - | 1 | -1.4 | - |
| ENS_D_compact[ens] | ent_mean | 0.097 | - | 1 | 0.1 | - |
| ENS_D_compact[ens] | rel_endpoints | 0.039 | - | 1 | 0.8 | - |
| ENS_D_compact[ens] | rel_frag | -0.100 | - | 1 | -9.1 | - |
| ENS_D_compact[ens] | rel_smallfrac | 0.089 | - | 1 | 5.3 | - |
| ENS_D_compact[ens] | var_mean | 0.266 | - | 1 | 18.8 | - |
| ENS_H_tglite70k[ens] | ent_mean | 0.163 | - | 1 | -0.9 | - |
| ENS_H_tglite70k[ens] | rel_endpoints | 0.149 | - | 1 | 13.7 | - |
| ENS_H_tglite70k[ens] | rel_frag | 0.049 | - | 1 | 2.8 | - |
| ENS_H_tglite70k[ens] | rel_smallfrac | 0.120 | - | 1 | 17.8 | - |
| ENS_H_tglite70k[ens] | var_mean | 0.242 | - | 1 | 6.8 | - |
| E_tglitenet[det] | ent_mean | 0.036 | 0.128 | 5 | -3.6 | 7.8 |
| E_tglitenet[det] | rel_endpoints | 0.182 | 0.116 | 5 | 10.8 | 9.2 |
| E_tglitenet[det] | rel_frag | 0.106 | 0.113 | 5 | 5.7 | 13.1 |
| E_tglitenet[det] | rel_smallfrac | 0.199 | 0.082 | 5 | 16.9 | 4.6 |
| E_tglitenet[mc] | ent_mean | -0.049 | 0.093 | 5 | -10.6 | 9.3 |
| E_tglitenet[mc] | rel_endpoints | 0.162 | 0.115 | 5 | 10.7 | 12.0 |
| E_tglitenet[mc] | rel_frag | 0.058 | 0.084 | 5 | 3.1 | 9.9 |
| E_tglitenet[mc] | rel_smallfrac | 0.180 | 0.108 | 5 | 16.9 | 7.5 |
| E_tglitenet[mc] | var_mean | -0.046 | 0.115 | 5 | -3.2 | 13.6 |
| H_tglite70k[det] | ent_mean | 0.215 | 0.084 | 5 | 4.8 | 8.1 |
| H_tglite70k[det] | rel_endpoints | 0.223 | 0.030 | 5 | 17.3 | 2.5 |
| H_tglite70k[det] | rel_frag | 0.078 | 0.056 | 5 | 6.0 | 3.9 |
| H_tglite70k[det] | rel_smallfrac | 0.159 | 0.038 | 5 | 14.5 | 1.7 |
| R_lite_w16[det] | ent_mean | 0.208 | 0.063 | 3 | 9.2 | 9.4 |
| R_lite_w16[det] | rel_endpoints | 0.307 | 0.054 | 3 | 26.0 | 4.8 |
| R_lite_w16[det] | rel_frag | 0.256 | 0.091 | 3 | 23.5 | 10.5 |
| R_lite_w16[det] | rel_smallfrac | 0.316 | 0.087 | 3 | 27.9 | 9.2 |
| R_unet_w8[det] | ent_mean | 0.210 | 0.037 | 3 | 8.8 | 9.4 |
| R_unet_w8[det] | rel_endpoints | 0.261 | 0.053 | 3 | 22.2 | 2.8 |
| R_unet_w8[det] | rel_frag | 0.227 | 0.059 | 3 | 19.9 | 9.3 |
| R_unet_w8[det] | rel_smallfrac | 0.306 | 0.037 | 3 | 28.3 | 10.5 |
| Z_no_kd[det] | ent_mean | 0.152 | 0.076 | 3 | 1.8 | 9.5 |
| Z_no_kd[det] | rel_endpoints | 0.136 | 0.081 | 3 | 13.7 | 8.7 |
| Z_no_kd[det] | rel_frag | -0.024 | 0.029 | 3 | 1.1 | 6.0 |
| Z_no_kd[det] | rel_smallfrac | 0.095 | 0.092 | 3 | 12.8 | 6.2 |
| Z_no_kdskel[det] | ent_mean | 0.177 | 0.056 | 3 | 0.1 | 4.9 |
| Z_no_kdskel[det] | rel_endpoints | 0.233 | 0.069 | 3 | 19.5 | 4.8 |
| Z_no_kdskel[det] | rel_frag | 0.114 | 0.092 | 3 | 10.0 | 8.2 |
| Z_no_kdskel[det] | rel_smallfrac | 0.230 | 0.102 | 3 | 18.8 | 9.9 |
| Z_noaug[det] | ent_mean | -0.071 | 0.114 | 3 | -10.9 | 12.4 |
| Z_noaug[det] | rel_endpoints | 0.278 | 0.013 | 3 | 23.5 | 3.6 |
| Z_noaug[det] | rel_frag | 0.232 | 0.020 | 3 | 17.9 | 1.1 |
| Z_noaug[det] | rel_smallfrac | 0.288 | 0.064 | 3 | 25.2 | 7.9 |
| Z_pixcons[det] | ent_mean | 0.190 | 0.051 | 3 | 2.1 | 4.4 |
| Z_pixcons[det] | rel_endpoints | 0.256 | 0.017 | 3 | 21.2 | 3.2 |
| Z_pixcons[det] | rel_frag | 0.175 | 0.035 | 3 | 13.7 | 4.6 |
| Z_pixcons[det] | rel_smallfrac | 0.233 | 0.036 | 3 | 19.4 | 3.8 |
| Z_v2recipe[det] | ent_mean | 0.027 | 0.051 | 3 | -3.5 | 4.1 |
| Z_v2recipe[det] | rel_endpoints | 0.326 | 0.056 | 3 | 24.8 | 4.8 |
| Z_v2recipe[det] | rel_frag | 0.259 | 0.051 | 3 | 21.3 | 7.6 |
| Z_v2recipe[det] | rel_smallfrac | 0.242 | 0.098 | 3 | 21.0 | 4.7 |

### Pre-registered comparisons (seed-paired t interval)

* A_unet[det]: rel_frag - entropy (rho, unseen mean): +0.0839 [±0.1081] not clear (n=5)
* C_unet_topo_dg[det]: rel_frag - entropy (rho, unseen mean): +0.1472 [±0.1275] clear (n=5)
* D_compact[det]: rel_frag - entropy (rho, unseen mean): +0.0182 [±0.1034] not clear (n=5)
* ENS_A_unet[ens]: rel_frag - entropy (rho, unseen mean): +0.0585 (n=1)
* ENS_D_compact[ens]: rel_frag - entropy (rho, unseen mean): -0.1974 (n=1)
* ENS_H_tglite70k[ens]: rel_frag - entropy (rho, unseen mean): -0.1146 (n=1)
* E_tglitenet[det]: rel_frag - entropy (rho, unseen mean): +0.0698 [±0.1767] not clear (n=5)
* E_tglitenet[mc]: rel_frag - entropy (rho, unseen mean): +0.1073 [±0.0625] clear (n=5)
* H_tglite70k[det]: rel_frag - entropy (rho, unseen mean): -0.1374 [±0.1275] clear (n=5)
* R_lite_w16[det]: rel_frag - entropy (rho, unseen mean): +0.0478 [±0.2729] not clear (n=3)
* R_unet_w8[det]: rel_frag - entropy (rho, unseen mean): +0.0166 [±0.1356] not clear (n=3)
* Z_no_kd[det]: rel_frag - entropy (rho, unseen mean): -0.1761 [±0.1383] clear (n=3)
* Z_no_kdskel[det]: rel_frag - entropy (rho, unseen mean): -0.0628 [±0.1526] not clear (n=3)
* Z_noaug[det]: rel_frag - entropy (rho, unseen mean): +0.3027 [±0.3343] not clear (n=3)
* Z_pixcons[det]: rel_frag - entropy (rho, unseen mean): -0.0151 [±0.1703] not clear (n=3)
* Z_v2recipe[det]: rel_frag - entropy (rho, unseen mean): +0.2321 [±0.1390] clear (n=3)
* E: rel_frag(det) - MC-dropout entropy (rho, unseen mean): +0.1547 [±0.1095] clear (n=5)