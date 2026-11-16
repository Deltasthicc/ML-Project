window.SITE = {
 "models": [
  {
   "key": "ENS_H_tglite70k[ens]",
   "label": "5 x 66K ensemble",
   "version": "v2.1",
   "kind": "ensemble",
   "n": 1,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.7,
   "drive": 0.814,
   "stare": 0.817,
   "chase": 0.772,
   "hrf": 0.73,
   "fives": 0.828,
   "raw": 0.741,
   "tuned": 0.773,
   "tuned_sd": null,
   "sel": 0.785,
   "sel_sd": null,
   "cldice": 0.827,
   "betti_raw": 26.549,
   "betti_clean": 4.275,
   "gap": 0.041,
   "net_res": 0.764
  },
  {
   "key": "H_tglite70k[det]",
   "label": "TG-LiteNet v2.1 (66K)",
   "version": "v2.1",
   "kind": "headline",
   "n": 5,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.7,
   "drive": 0.81,
   "stare": 0.81,
   "chase": 0.765,
   "hrf": 0.729,
   "fives": 0.81,
   "raw": 0.733,
   "tuned": 0.768,
   "tuned_sd": 0.004,
   "sel": 0.779,
   "sel_sd": 0.004,
   "cldice": 0.82,
   "betti_raw": 32.103,
   "betti_clean": 4.485,
   "gap": 0.042,
   "net_res": 0.757
  },
  {
   "key": "R_lite_w16[det]",
   "label": "Depthwise 212K (size-rule headline)",
   "version": "v2.1",
   "kind": "reference",
   "n": 3,
   "params": 212000.0,
   "gflops": 5.53,
   "cpu_ms": 106.7,
   "drive": 0.814,
   "stare": 0.812,
   "chase": 0.777,
   "hrf": 0.736,
   "fives": 0.822,
   "raw": 0.745,
   "tuned": 0.775,
   "tuned_sd": 0.011,
   "sel": 0.785,
   "sel_sd": 0.008,
   "cldice": 0.826,
   "betti_raw": 29.027,
   "betti_clean": 4.67,
   "gap": 0.039,
   "net_res": 0.767
  },
  {
   "key": "R_unet_w8[det]",
   "label": "U-Net w8 486K (v2.1 recipe)",
   "version": "v2.1",
   "kind": "reference",
   "n": 3,
   "params": 486000.0,
   "gflops": 6.05,
   "cpu_ms": 55.0,
   "drive": 0.812,
   "stare": 0.812,
   "chase": 0.775,
   "hrf": 0.736,
   "fives": 0.822,
   "raw": 0.739,
   "tuned": 0.775,
   "tuned_sd": 0.01,
   "sel": 0.785,
   "sel_sd": 0.009,
   "cldice": 0.826,
   "betti_raw": 31.785,
   "betti_clean": 4.48,
   "gap": 0.037,
   "net_res": 0.763
  },
  {
   "key": "Z_no_kd[det]",
   "label": "66K without distillation",
   "version": "v2.1",
   "kind": "ablation",
   "n": 3,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.7,
   "drive": 0.811,
   "stare": 0.814,
   "chase": 0.77,
   "hrf": 0.732,
   "fives": 0.815,
   "raw": 0.741,
   "tuned": 0.772,
   "tuned_sd": 0.002,
   "sel": 0.779,
   "sel_sd": 0.006,
   "cldice": 0.813,
   "betti_raw": 41.425,
   "betti_clean": 4.528,
   "gap": 0.039,
   "net_res": 0.763
  },
  {
   "key": "Z_no_kdskel[det]",
   "label": "66K without N3 (centreline distillation)",
   "version": "v2.1",
   "kind": "ablation",
   "n": 3,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.7,
   "drive": 0.813,
   "stare": 0.814,
   "chase": 0.772,
   "hrf": 0.734,
   "fives": 0.82,
   "raw": 0.739,
   "tuned": 0.774,
   "tuned_sd": 0.004,
   "sel": 0.773,
   "sel_sd": 0.007,
   "cldice": 0.818,
   "betti_raw": 43.767,
   "betti_clean": 4.551,
   "gap": 0.039,
   "net_res": 0.764
  },
  {
   "key": "Z_noaug[det]",
   "label": "66K without RandConv + Fourier",
   "version": "v2.1",
   "kind": "ablation",
   "n": 3,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.7,
   "drive": 0.81,
   "stare": 0.812,
   "chase": 0.785,
   "hrf": 0.738,
   "fives": 0.829,
   "raw": 0.74,
   "tuned": 0.778,
   "tuned_sd": 0.007,
   "sel": 0.78,
   "sel_sd": 0.008,
   "cldice": 0.815,
   "betti_raw": 28.974,
   "betti_clean": 4.524,
   "gap": 0.032,
   "net_res": 0.766
  },
  {
   "key": "Z_pixcons[det]",
   "label": "66K with pixel consistency (no N4)",
   "version": "v2.1",
   "kind": "ablation",
   "n": 3,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.7,
   "drive": 0.81,
   "stare": 0.817,
   "chase": 0.779,
   "hrf": 0.737,
   "fives": 0.82,
   "raw": 0.742,
   "tuned": 0.778,
   "tuned_sd": 0.005,
   "sel": 0.773,
   "sel_sd": 0.009,
   "cldice": 0.819,
   "betti_raw": 36.219,
   "betti_clean": 4.472,
   "gap": 0.032,
   "net_res": 0.767
  },
  {
   "key": "Z_v2recipe[det]",
   "label": "66K with the v2 recipe",
   "version": "v2.1",
   "kind": "ablation",
   "n": 3,
   "params": 66000.0,
   "gflops": 2.73,
   "cpu_ms": 24.2,
   "drive": 0.815,
   "stare": 0.816,
   "chase": 0.783,
   "hrf": 0.73,
   "fives": 0.83,
   "raw": 0.75,
   "tuned": 0.777,
   "tuned_sd": 0.005,
   "sel": 0.782,
   "sel_sd": 0.007,
   "cldice": 0.819,
   "betti_raw": 41.488,
   "betti_clean": 4.701,
   "gap": 0.039,
   "net_res": 0.768
  },
  {
   "key": "A_unet[det]",
   "label": "A  U-Net",
   "version": "v2",
   "kind": "stage",
   "n": 5,
   "params": 31036000.0,
   "gflops": 384.74,
   "cpu_ms": 935.9,
   "drive": 0.822,
   "stare": 0.798,
   "chase": 0.769,
   "hrf": 0.718,
   "fives": 0.78,
   "raw": 0.759,
   "tuned": 0.762,
   "tuned_sd": 0.015,
   "sel": 0.742,
   "sel_sd": 0.034,
   "cldice": 0.832,
   "betti_raw": 72.799,
   "betti_clean": 4.511,
   "gap": 0.061,
   "net_res": 0.762
  },
  {
   "key": "C_unet_topo_dg[det]",
   "label": "C  + DG augmentation",
   "version": "v2",
   "kind": "stage",
   "n": 5,
   "params": 31036000.0,
   "gflops": 384.74,
   "cpu_ms": 1003.4,
   "drive": 0.824,
   "stare": 0.828,
   "chase": 0.789,
   "hrf": 0.749,
   "fives": 0.842,
   "raw": 0.761,
   "tuned": 0.789,
   "tuned_sd": 0.008,
   "sel": 0.751,
   "sel_sd": 0.034,
   "cldice": 0.836,
   "betti_raw": 35.464,
   "betti_clean": 4.594,
   "gap": 0.035,
   "net_res": 0.777
  },
  {
   "key": "D_compact[det]",
   "label": "D  compact U-Net (486K)",
   "version": "v2",
   "kind": "stage",
   "n": 5,
   "params": 486000.0,
   "gflops": 6.05,
   "cpu_ms": 30.9,
   "drive": 0.818,
   "stare": 0.818,
   "chase": 0.784,
   "hrf": 0.737,
   "fives": 0.846,
   "raw": 0.749,
   "tuned": 0.78,
   "tuned_sd": 0.002,
   "sel": 0.772,
   "sel_sd": 0.035,
   "cldice": 0.825,
   "betti_raw": 40.452,
   "betti_clean": 4.567,
   "gap": 0.039,
   "net_res": 0.769
  },
  {
   "key": "ENS_A_unet[ens]",
   "label": "5 x U-Net ensemble",
   "version": "v2",
   "kind": "ensemble",
   "n": 1,
   "params": 155182000.0,
   "gflops": 1923.68,
   "cpu_ms": 4679.7,
   "drive": 0.827,
   "stare": 0.802,
   "chase": 0.775,
   "hrf": 0.717,
   "fives": 0.79,
   "raw": 0.765,
   "tuned": 0.764,
   "tuned_sd": null,
   "sel": 0.762,
   "sel_sd": null,
   "cldice": 0.84,
   "betti_raw": 64.507,
   "betti_clean": 4.424,
   "gap": 0.062,
   "net_res": 0.766
  },
  {
   "key": "ENS_D_compact[ens]",
   "label": "5 x compact ensemble",
   "version": "v2",
   "kind": "ensemble",
   "n": 1,
   "params": 2432000.0,
   "gflops": 30.24,
   "cpu_ms": 154.4,
   "drive": 0.82,
   "stare": 0.823,
   "chase": 0.788,
   "hrf": 0.738,
   "fives": 0.856,
   "raw": 0.754,
   "tuned": 0.783,
   "tuned_sd": null,
   "sel": 0.792,
   "sel_sd": null,
   "cldice": 0.829,
   "betti_raw": 31.403,
   "betti_clean": 4.46,
   "gap": 0.037,
   "net_res": 0.773
  },
  {
   "key": "E_tglitenet[det]",
   "label": "E  TG-LiteNet v2 (486K)",
   "version": "v2",
   "kind": "stage",
   "n": 5,
   "params": 486000.0,
   "gflops": 6.05,
   "cpu_ms": 30.0,
   "drive": 0.817,
   "stare": 0.819,
   "chase": 0.787,
   "hrf": 0.733,
   "fives": 0.847,
   "raw": 0.751,
   "tuned": 0.78,
   "tuned_sd": 0.007,
   "sel": 0.789,
   "sel_sd": 0.005,
   "cldice": 0.825,
   "betti_raw": 39.18,
   "betti_clean": 4.417,
   "gap": 0.037,
   "net_res": 0.77
  },
  {
   "key": "E_tglitenet[mc]",
   "label": "E  + MC dropout",
   "version": "v2",
   "kind": "stage",
   "n": 5,
   "params": 486000.0,
   "gflops": 6.05,
   "cpu_ms": 30.0,
   "drive": 0.816,
   "stare": 0.818,
   "chase": 0.786,
   "hrf": 0.729,
   "fives": 0.845,
   "raw": 0.752,
   "tuned": 0.778,
   "tuned_sd": 0.007,
   "sel": 0.786,
   "sel_sd": 0.005,
   "cldice": 0.824,
   "betti_raw": 39.773,
   "betti_clean": 4.422,
   "gap": 0.038,
   "net_res": 0.769
  }
 ],
 "stages": {
  "raw": {
   "A": {
    "drive_dice": [
     0.82482,
     0.00255
    ],
    "stare_dice": [
     0.79305,
     0.00819
    ],
    "chase_db1_dice": [
     0.76716,
     0.00372
    ],
    "hrf_dice": [
     0.71624,
     0.01013
    ],
    "fives_dice": [
     0.79519,
     0.00517
    ],
    "unseen_dice": [
     0.75882,
     0.00711
    ],
    "drive_cldice": [
     0.8246,
     0.00583
    ],
    "stare_cldice": [
     0.86202,
     0.0073
    ],
    "chase_db1_cldice": [
     0.83322,
     0.00535
    ],
    "hrf_cldice": [
     0.81293,
     0.006
    ],
    "fives_cldice": [
     0.82058,
     0.00893
    ],
    "unseen_cldice": [
     0.83606,
     0.00279
    ],
    "drive_betti0_err": [
     74.99,
     14.35703
    ],
    "stare_betti0_err": [
     32.82,
     5.448
    ],
    "chase_db1_betti0_err": [
     51.97857,
     8.10103
    ],
    "hrf_betti0_err": [
     133.61333,
     15.58354
    ],
    "fives_betti0_err": [
     80.398,
     11.27145
    ],
    "unseen_betti0_err": [
     72.80397,
     8.33926
    ]
   },
   "B": {
    "drive_dice": [
     0.82132,
     0.00214
    ],
    "stare_dice": [
     0.79648,
     0.00456
    ],
    "chase_db1_dice": [
     0.76579,
     0.00365
    ],
    "hrf_dice": [
     0.7115,
     0.00544
    ],
    "fives_dice": [
     0.80893,
     0.00782
    ],
    "unseen_dice": [
     0.75792,
     0.00389
    ],
    "drive_cldice": [
     0.83683,
     0.00335
    ],
    "stare_cldice": [
     0.86808,
     0.00865
    ],
    "chase_db1_cldice": [
     0.83917,
     0.01225
    ],
    "hrf_cldice": [
     0.81795,
     0.00575
    ],
    "fives_cldice": [
     0.83543,
     0.00836
    ],
    "unseen_cldice": [
     0.84173,
     0.00688
    ],
    "drive_betti0_err": [
     42.81,
     6.94716
    ],
    "stare_betti0_err": [
     22.5,
     4.00827
    ],
    "chase_db1_betti0_err": [
     32.47143,
     6.72953
    ],
    "hrf_betti0_err": [
     46.59111,
     13.47181
    ],
    "fives_betti0_err": [
     45.201,
     10.66275
    ],
    "unseen_betti0_err": [
     33.85418,
     6.49372
    ]
   },
   "C": {
    "drive_dice": [
     0.82423,
     0.00123
    ],
    "stare_dice": [
     0.80329,
     0.00455
    ],
    "chase_db1_dice": [
     0.76668,
     0.00432
    ],
    "hrf_dice": [
     0.71199,
     0.00395
    ],
    "fives_dice": [
     0.84264,
     0.0026
    ],
    "unseen_dice": [
     0.76065,
     0.00361
    ],
    "drive_cldice": [
     0.84648,
     0.0009
    ],
    "stare_cldice": [
     0.86652,
     0.00525
    ],
    "chase_db1_cldice": [
     0.85134,
     0.00323
    ],
    "hrf_cldice": [
     0.81898,
     0.00431
    ],
    "fives_cldice": [
     0.87968,
     0.00425
    ],
    "unseen_cldice": [
     0.84561,
     0.00183
    ],
    "drive_betti0_err": [
     34.35,
     4.72533
    ],
    "stare_betti0_err": [
     27.14,
     2.69059
    ],
    "chase_db1_betti0_err": [
     23.47143,
     1.83075
    ],
    "hrf_betti0_err": [
     55.83556,
     6.0646
    ],
    "fives_betti0_err": [
     36.951,
     3.72953
    ],
    "unseen_betti0_err": [
     35.48233,
     2.98521
    ]
   },
   "D": {
    "drive_dice": [
     0.81908,
     0.00114
    ],
    "stare_dice": [
     0.79037,
     0.00215
    ],
    "chase_db1_dice": [
     0.75826,
     0.00499
    ],
    "hrf_dice": [
     0.69919,
     0.00368
    ],
    "fives_dice": [
     0.83779,
     0.00476
    ],
    "unseen_dice": [
     0.74927,
     0.00191
    ],
    "drive_cldice": [
     0.83964,
     0.00144
    ],
    "stare_cldice": [
     0.86109,
     0.00126
    ],
    "chase_db1_cldice": [
     0.84308,
     0.00205
    ],
    "hrf_cldice": [
     0.80118,
     0.00245
    ],
    "fives_cldice": [
     0.87756,
     0.00163
    ],
    "unseen_cldice": [
     0.83512,
     0.00115
    ],
    "drive_betti0_err": [
     42.37,
     1.4537
    ],
    "stare_betti0_err": [
     30.5,
     2.12867
    ],
    "chase_db1_betti0_err": [
     28.09286,
     2.50199
    ],
    "hrf_betti0_err": [
     62.78667,
     9.84256
    ],
    "fives_betti0_err": [
     44.824,
     2.08757
    ],
    "unseen_betti0_err": [
     40.45984,
     3.76278
    ]
   },
   "E": {
    "drive_dice": [
     0.81749,
     0.00136
    ],
    "stare_dice": [
     0.79188,
     0.00789
    ],
    "chase_db1_dice": [
     0.76278,
     0.00753
    ],
    "hrf_dice": [
     0.69774,
     0.00258
    ],
    "fives_dice": [
     0.84098,
     0.0061
    ],
    "unseen_dice": [
     0.7508,
     0.00585
    ],
    "drive_cldice": [
     0.84023,
     0.0037
    ],
    "stare_cldice": [
     0.86302,
     0.0066
    ],
    "chase_db1_cldice": [
     0.84452,
     0.00288
    ],
    "hrf_cldice": [
     0.7994,
     0.00806
    ],
    "fives_cldice": [
     0.87707,
     0.00512
    ],
    "unseen_cldice": [
     0.83565,
     0.00201
    ],
    "drive_betti0_err": [
     40.02,
     4.4008
    ],
    "stare_betti0_err": [
     30.63,
     1.39893
    ],
    "chase_db1_betti0_err": [
     28.67857,
     3.30304
    ],
    "hrf_betti0_err": [
     58.21778,
     4.96664
    ],
    "fives_betti0_err": [
     44.699,
     2.26677
    ],
    "unseen_betti0_err": [
     39.17545,
     1.44815
    ]
   }
  },
  "clean": {
   "A": {
    "drive_dice": [
     0.82221,
     0.00357
    ],
    "stare_dice": [
     0.79566,
     0.00767
    ],
    "chase_db1_dice": [
     0.76669,
     0.00304
    ],
    "hrf_dice": [
     0.71398,
     0.00999
    ],
    "fives_dice": [
     0.78226,
     0.00707
    ],
    "unseen_dice": [
     0.75878,
     0.00653
    ],
    "drive_cldice": [
     0.81888,
     0.00857
    ],
    "stare_cldice": [
     0.86985,
     0.00482
    ],
    "chase_db1_cldice": [
     0.82842,
     0.01059
    ],
    "hrf_cldice": [
     0.80384,
     0.00901
    ],
    "fives_cldice": [
     0.79333,
     0.01357
    ],
    "unseen_cldice": [
     0.83404,
     0.0053
    ],
    "drive_betti0_err": [
     1.74,
     0.1245
    ],
    "stare_betti0_err": [
     3.03,
     0.13509
    ],
    "chase_db1_betti0_err": [
     1.89286,
     0.60609
    ],
    "hrf_betti0_err": [
     8.60889,
     0.52437
    ],
    "fives_betti0_err": [
     3.089,
     0.09832
    ],
    "unseen_betti0_err": [
     4.51058,
     0.09398
    ]
   },
   "B": {
    "drive_dice": [
     0.81921,
     0.00173
    ],
    "stare_dice": [
     0.79963,
     0.00384
    ],
    "chase_db1_dice": [
     0.76697,
     0.00365
    ],
    "hrf_dice": [
     0.71034,
     0.00557
    ],
    "fives_dice": [
     0.79553,
     0.00854
    ],
    "unseen_dice": [
     0.75898,
     0.00365
    ],
    "drive_cldice": [
     0.83,
     0.00399
    ],
    "stare_cldice": [
     0.87335,
     0.00596
    ],
    "chase_db1_cldice": [
     0.83412,
     0.01177
    ],
    "hrf_cldice": [
     0.80714,
     0.0059
    ],
    "fives_cldice": [
     0.80855,
     0.01034
    ],
    "unseen_cldice": [
     0.8382,
     0.00613
    ],
    "drive_betti0_err": [
     1.84,
     0.1084
    ],
    "stare_betti0_err": [
     2.98,
     0.27065
    ],
    "chase_db1_betti0_err": [
     1.83571,
     0.30991
    ],
    "hrf_betti0_err": [
     8.45333,
     0.37072
    ],
    "fives_betti0_err": [
     3.014,
     0.13297
    ],
    "unseen_betti0_err": [
     4.42302,
     0.14067
    ]
   },
   "C": {
    "drive_dice": [
     0.82385,
     0.00078
    ],
    "stare_dice": [
     0.80785,
     0.00414
    ],
    "chase_db1_dice": [
     0.76837,
     0.00399
    ],
    "hrf_dice": [
     0.71143,
     0.00359
    ],
    "fives_dice": [
     0.84061,
     0.00438
    ],
    "unseen_dice": [
     0.76255,
     0.00331
    ],
    "drive_cldice": [
     0.84328,
     0.00138
    ],
    "stare_cldice": [
     0.87615,
     0.00409
    ],
    "chase_db1_cldice": [
     0.85016,
     0.00513
    ],
    "hrf_cldice": [
     0.81057,
     0.00543
    ],
    "fives_cldice": [
     0.87263,
     0.01147
    ],
    "unseen_cldice": [
     0.84563,
     0.00241
    ],
    "drive_betti0_err": [
     1.64,
     0.05477
    ],
    "stare_betti0_err": [
     3.53,
     0.38987
    ],
    "chase_db1_betti0_err": [
     1.65714,
     0.16091
    ],
    "hrf_betti0_err": [
     8.59556,
     0.2728
    ],
    "fives_betti0_err": [
     2.992,
     0.05369
    ],
    "unseen_betti0_err": [
     4.59423,
     0.1678
    ]
   },
   "D": {
    "drive_dice": [
     0.81826,
     0.00075
    ],
    "stare_dice": [
     0.79519,
     0.00212
    ],
    "chase_db1_dice": [
     0.7611,
     0.00472
    ],
    "hrf_dice": [
     0.69778,
     0.00323
    ],
    "fives_dice": [
     0.83675,
     0.0037
    ],
    "unseen_dice": [
     0.75135,
     0.00161
    ],
    "drive_cldice": [
     0.83495,
     0.0019
    ],
    "stare_cldice": [
     0.86999,
     0.00212
    ],
    "chase_db1_cldice": [
     0.84339,
     0.00258
    ],
    "hrf_cldice": [
     0.78803,
     0.00344
    ],
    "fives_cldice": [
     0.87209,
     0.00464
    ],
    "unseen_cldice": [
     0.8338,
     0.00196
    ],
    "drive_betti0_err": [
     1.68,
     0.14405
    ],
    "stare_betti0_err": [
     3.47,
     0.44525
    ],
    "chase_db1_betti0_err": [
     1.65,
     0.32478
    ],
    "hrf_betti0_err": [
     8.58222,
     0.47151
    ],
    "fives_betti0_err": [
     3.082,
     0.07488
    ],
    "unseen_betti0_err": [
     4.56741,
     0.23417
    ]
   },
   "E": {
    "drive_dice": [
     0.81658,
     0.0025
    ],
    "stare_dice": [
     0.79648,
     0.00745
    ],
    "chase_db1_dice": [
     0.7656,
     0.00718
    ],
    "hrf_dice": [
     0.69631,
     0.00133
    ],
    "fives_dice": [
     0.84018,
     0.00414
    ],
    "unseen_dice": [
     0.7528,
     0.00521
    ],
    "drive_cldice": [
     0.83494,
     0.00691
    ],
    "stare_cldice": [
     0.87106,
     0.0045
    ],
    "chase_db1_cldice": [
     0.8448,
     0.00599
    ],
    "hrf_cldice": [
     0.78502,
     0.01285
    ],
    "fives_cldice": [
     0.87147,
     0.01497
    ],
    "unseen_cldice": [
     0.83363,
     0.0049
    ],
    "drive_betti0_err": [
     1.66,
     0.15166
    ],
    "stare_betti0_err": [
     3.35,
     0.3937
    ],
    "chase_db1_betti0_err": [
     1.68571,
     0.39722
    ],
    "hrf_betti0_err": [
     8.22222,
     0.58479
    ],
    "fives_betti0_err": [
     3.091,
     0.20154
    ],
    "unseen_betti0_err": [
     4.41931,
     0.25536
    ]
   }
  },
  "tuned": {
   "A": {
    "drive_dice": [
     0.82217,
     0.00224
    ],
    "stare_dice": [
     0.79808,
     0.01329
    ],
    "chase_db1_dice": [
     0.7688,
     0.01002
    ],
    "hrf_dice": [
     0.71779,
     0.02195
    ],
    "fives_dice": [
     0.77977,
     0.00789
    ],
    "unseen_dice": [
     0.76155,
     0.01491
    ],
    "drive_cldice": [
     0.81772,
     0.00497
    ],
    "stare_cldice": [
     0.86908,
     0.00406
    ],
    "chase_db1_cldice": [
     0.82584,
     0.00894
    ],
    "hrf_cldice": [
     0.8002,
     0.01298
    ],
    "fives_cldice": [
     0.78836,
     0.00999
    ],
    "unseen_cldice": [
     0.83171,
     0.00752
    ],
    "drive_betti0_err": [
     1.78,
     0.14832
    ],
    "stare_betti0_err": [
     3.15,
     0.2622
    ],
    "chase_db1_betti0_err": [
     1.92857,
     0.50948
    ],
    "hrf_betti0_err": [
     8.67556,
     0.54763
    ],
    "fives_betti0_err": [
     3.101,
     0.08272
    ],
    "unseen_betti0_err": [
     4.58471,
     0.10318
    ]
   },
   "B": {
    "drive_dice": [
     0.81802,
     0.00144
    ],
    "stare_dice": [
     0.81279,
     0.00577
    ],
    "chase_db1_dice": [
     0.77773,
     0.005
    ],
    "hrf_dice": [
     0.73428,
     0.01046
    ],
    "fives_dice": [
     0.79606,
     0.00958
    ],
    "unseen_dice": [
     0.77493,
     0.00661
    ],
    "drive_cldice": [
     0.82393,
     0.00456
    ],
    "stare_cldice": [
     0.87233,
     0.00605
    ],
    "chase_db1_cldice": [
     0.82793,
     0.00997
    ],
    "hrf_cldice": [
     0.79791,
     0.00944
    ],
    "fives_cldice": [
     0.80023,
     0.01046
    ],
    "unseen_cldice": [
     0.83272,
     0.00572
    ],
    "drive_betti0_err": [
     1.86,
     0.06519
    ],
    "stare_betti0_err": [
     2.92,
     0.2515
    ],
    "chase_db1_betti0_err": [
     2.15,
     0.84568
    ],
    "hrf_betti0_err": [
     8.21778,
     0.83819
    ],
    "fives_betti0_err": [
     3.111,
     0.21637
    ],
    "unseen_betti0_err": [
     4.42926,
     0.10992
    ]
   },
   "C": {
    "drive_dice": [
     0.82375,
     0.00079
    ],
    "stare_dice": [
     0.82849,
     0.00505
    ],
    "chase_db1_dice": [
     0.78946,
     0.006
    ],
    "hrf_dice": [
     0.7488,
     0.01294
    ],
    "fives_dice": [
     0.84158,
     0.00772
    ],
    "unseen_dice": [
     0.78892,
     0.00789
    ],
    "drive_cldice": [
     0.83263,
     0.00407
    ],
    "stare_cldice": [
     0.87883,
     0.00255
    ],
    "chase_db1_cldice": [
     0.84016,
     0.00451
    ],
    "hrf_cldice": [
     0.78964,
     0.00566
    ],
    "fives_cldice": [
     0.85709,
     0.01119
    ],
    "unseen_cldice": [
     0.83621,
     0.00335
    ],
    "drive_betti0_err": [
     1.67,
     0.04472
    ],
    "stare_betti0_err": [
     3.46,
     0.307
    ],
    "chase_db1_betti0_err": [
     1.57857,
     0.14151
    ],
    "hrf_betti0_err": [
     8.58222,
     0.32675
    ],
    "fives_betti0_err": [
     2.997,
     0.07103
    ],
    "unseen_betti0_err": [
     4.54026,
     0.15717
    ]
   },
   "D": {
    "drive_dice": [
     0.81835,
     0.00051
    ],
    "stare_dice": [
     0.81801,
     0.00254
    ],
    "chase_db1_dice": [
     0.7842,
     0.00154
    ],
    "hrf_dice": [
     0.73651,
     0.00452
    ],
    "fives_dice": [
     0.84617,
     0.00328
    ],
    "unseen_dice": [
     0.77958,
     0.00222
    ],
    "drive_cldice": [
     0.82532,
     0.00346
    ],
    "stare_cldice": [
     0.87078,
     0.00182
    ],
    "chase_db1_cldice": [
     0.835,
     0.00496
    ],
    "hrf_cldice": [
     0.76861,
     0.00815
    ],
    "fives_cldice": [
     0.85851,
     0.00806
    ],
    "unseen_cldice": [
     0.8248,
     0.00409
    ],
    "drive_betti0_err": [
     1.67,
     0.1255
    ],
    "stare_betti0_err": [
     3.32,
     0.2515
    ],
    "chase_db1_betti0_err": [
     1.81429,
     0.4734
    ],
    "hrf_betti0_err": [
     8.29333,
     0.58698
    ],
    "fives_betti0_err": [
     3.112,
     0.14264
    ],
    "unseen_betti0_err": [
     4.47587,
     0.16921
    ]
   },
   "E": {
    "drive_dice": [
     0.81656,
     0.00262
    ],
    "stare_dice": [
     0.81893,
     0.00287
    ],
    "chase_db1_dice": [
     0.78745,
     0.00427
    ],
    "hrf_dice": [
     0.73322,
     0.01369
    ],
    "fives_dice": [
     0.84726,
     0.00625
    ],
    "unseen_dice": [
     0.77987,
     0.00679
    ],
    "drive_cldice": [
     0.82557,
     0.00318
    ],
    "stare_cldice": [
     0.87271,
     0.00255
    ],
    "chase_db1_cldice": [
     0.83669,
     0.00432
    ],
    "hrf_cldice": [
     0.76531,
     0.00705
    ],
    "fives_cldice": [
     0.85689,
     0.00948
    ],
    "unseen_cldice": [
     0.82491,
     0.00328
    ],
    "drive_betti0_err": [
     1.74,
     0.09618
    ],
    "stare_betti0_err": [
     3.43,
     0.30537
    ],
    "chase_db1_betti0_err": [
     1.82857,
     0.30136
    ],
    "hrf_betti0_err": [
     8.42222,
     0.33738
    ],
    "fives_betti0_err": [
     2.892,
     0.03213
    ],
    "unseen_betti0_err": [
     4.56026,
     0.12339
    ]
   }
  }
 },
 "stage_labels": {
  "A": "A  U-Net",
  "B": "B  + topology loss",
  "C": "C  + DG augmentation",
  "D": "D  compact U-Net (486K)",
  "E": "E  TG-LiteNet v2 (486K)"
 },
 "factorial": [
  {
   "key": "A_unet[det]",
   "backbone": "U-Net",
   "topology": false,
   "dg": false,
   "n": 5,
   "dice_raw": 0.7588158957461931,
   "cldice_raw": 0.8360550376747519,
   "betti_raw": 72.80396825396824,
   "dice_tuned": 0.7615539012518131,
   "gap_tuned": 0.0606149831223472
  },
  {
   "key": "X_unet_dg[det]",
   "backbone": "U-Net",
   "topology": false,
   "dg": true,
   "n": 3,
   "dice_raw": 0.7735524261523735,
   "cldice_raw": 0.8382869269645538,
   "betti_raw": 91.06940035273368,
   "dice_tuned": 0.7848360818185189,
   "gap_tuned": 0.0417937881388985
  },
  {
   "key": "B_unet_topo[det]",
   "backbone": "U-Net",
   "topology": true,
   "dg": false,
   "n": 5,
   "dice_raw": 0.7579243396294354,
   "cldice_raw": 0.8417323712010802,
   "betti_raw": 33.85417989417989,
   "dice_tuned": 0.7749307495861297,
   "gap_tuned": 0.0430844289894363
  },
  {
   "key": "C_unet_topo_dg[det]",
   "backbone": "U-Net",
   "topology": true,
   "dg": true,
   "n": 5,
   "dice_raw": 0.7606546400451609,
   "cldice_raw": 0.8456136138064082,
   "betti_raw": 35.48232804232804,
   "dice_tuned": 0.7889158910190607,
   "gap_tuned": 0.0348327682044459
  },
  {
   "key": "X_compact_plain[det]",
   "backbone": "Compact",
   "topology": false,
   "dg": false,
   "n": 3,
   "dice_raw": 0.7542069060978759,
   "cldice_raw": 0.8251098420349358,
   "betti_raw": 98.768430335097,
   "dice_tuned": 0.7531331967666789,
   "gap_tuned": 0.0653149243826384
  },
  {
   "key": "X_compact_dg[det]",
   "backbone": "Compact",
   "topology": false,
   "dg": true,
   "n": 3,
   "dice_raw": 0.7597733386938798,
   "cldice_raw": 0.826758283346314,
   "betti_raw": 88.98827160493828,
   "dice_tuned": 0.7727911637767334,
   "gap_tuned": 0.0455369704010774
  },
  {
   "key": "X_compact_topo[det]",
   "backbone": "Compact",
   "topology": true,
   "dg": false,
   "n": 3,
   "dice_raw": 0.7468836231625149,
   "cldice_raw": 0.8354767698416031,
   "betti_raw": 44.486860670194005,
   "dice_tuned": 0.7623735934725926,
   "gap_tuned": 0.0518393770436657
  },
  {
   "key": "D_compact[det]",
   "backbone": "Compact",
   "topology": true,
   "dg": true,
   "n": 5,
   "dice_raw": 0.7492743950658765,
   "cldice_raw": 0.8351197692735726,
   "betti_raw": 40.45984126984127,
   "dice_tuned": 0.7795756111524043,
   "gap_tuned": 0.0387716165262658
  }
 ],
 "scale_delta": {
  "H_tglite70k[det]": {
   "stare": 0.0,
   "chase_db1": 0.0,
   "hrf": 0.034255424083043516,
   "fives": -0.0607516399912732
  },
  "R_lite_w16[det]": {
   "stare": 0.0007028102416611152,
   "chase_db1": 0.0,
   "hrf": 0.027611845949214175,
   "fives": -0.025858216783023174
  },
  "R_unet_w8[det]": {
   "stare": 0.0,
   "chase_db1": 0.0,
   "hrf": 0.0307125551882577,
   "fives": -0.052632567143792906
  },
  "E_tglitenet[det]": {
   "stare": 0.0,
   "chase_db1": 0.0,
   "hrf": 0.028593454547114795,
   "fives": -0.04131709363858573
  },
  "D_compact[det]": {
   "stare": 0.0,
   "chase_db1": -0.0481343662248233,
   "hrf": 0.026489430579450413,
   "fives": -0.035029899746742976
  },
  "C_unet_topo_dg[det]": {
   "stare": 8.539402368183957e-05,
   "chase_db1": -0.13086740901443633,
   "hrf": 0.016010462157051507,
   "fives": -0.05098928687721893
  },
  "A_unet[det]": {
   "stare": 0.0,
   "chase_db1": -0.05462489844646332,
   "hrf": -0.004355128644856321,
   "fives": -0.0400283285800167
  }
 },
 "scale_unseen": {
  "ENS_H_tglite70k[ens]": {
   "text": "+0.0119 (n=1)"
  },
  "H_tglite70k[det]": {
   "text": "+0.0114 [\u00b10.0005] clear (n=5)"
  },
  "R_lite_w16[det]": {
   "text": "+0.0094 [\u00b10.0074] clear (n=3)"
  },
  "R_unet_w8[det]": {
   "text": "+0.0102 [\u00b10.0076] clear (n=3)"
  },
  "Z_no_kd[det]": {
   "text": "+0.0074 [\u00b10.0092] not clear (n=3)"
  },
  "Z_no_kdskel[det]": {
   "text": "-0.0003 [\u00b10.0153] not clear (n=3)"
  },
  "Z_noaug[det]": {
   "text": "+0.0023 [\u00b10.0334] not clear (n=3)"
  },
  "Z_pixcons[det]": {
   "text": "-0.0049 [\u00b10.0197] not clear (n=3)"
  },
  "Z_v2recipe[det]": {
   "text": "+0.0056 [\u00b10.0158] not clear (n=3)"
  },
  "A_unet[det]": {
   "text": "-0.0197 [\u00b10.0446] not clear (n=5)"
  },
  "C_unet_topo_dg[det]": {
   "text": "-0.0383 [\u00b10.0349] clear (n=5)"
  },
  "D_compact[det]": {
   "text": "-0.0072 [\u00b10.0428] not clear (n=5)"
  },
  "ENS_A_unet[ens]": {
   "text": "-0.0020 (n=1)"
  },
  "ENS_D_compact[ens]": {
   "text": "+0.0089 (n=1)"
  },
  "E_tglitenet[det]": {
   "text": "+0.0095 [\u00b10.0023] clear (n=5)"
  },
  "E_tglitenet[mc]": {
   "text": "+0.0085 [\u00b10.0057] clear (n=5)"
  }
 },
 "scale_share": {
  "chase_db1": {
   "1.0": 1.0,
   "1.5": 0.0,
   "2.0": 0.0,
   "2.5": 0.0
  },
  "drive": {
   "1.0": 1.0,
   "1.5": 0.0,
   "2.0": 0.0,
   "2.5": 0.0
  },
  "fives": {
   "1.0": 0.895,
   "1.5": 0.0,
   "2.0": 0.013,
   "2.5": 0.092
  },
  "hrf": {
   "1.0": 0.08444444444444445,
   "1.5": 0.9155555555555556,
   "2.0": 0.0,
   "2.5": 0.0
  },
  "stare": {
   "1.0": 1.0,
   "1.5": 0.0,
   "2.0": 0.0,
   "2.5": 0.0
  }
 },
 "ablations": {
  "Z_no_kdskel": {
   "dice": {
    "delta": -0.0051,
    "ci": 0.0148,
    "clear": false
   },
   "cldice": {
    "delta": 0.0019,
    "ci": 0.0073,
    "clear": false
   },
   "betti": {
    "delta": -11.4963,
    "ci": 15.7837,
    "clear": false
   }
  },
  "Z_no_kd": {
   "dice": {
    "delta": -0.0033,
    "ci": 0.0044,
    "clear": false
   },
   "cldice": {
    "delta": 0.0068,
    "ci": 0.0114,
    "clear": false
   },
   "betti": {
    "delta": -9.1538,
    "ci": 12.5467,
    "clear": false
   }
  },
  "Z_pixcons": {
   "dice": {
    "delta": -0.0091,
    "ci": 0.0045,
    "clear": true
   },
   "cldice": {
    "delta": 0.0011,
    "ci": 0.0105,
    "clear": false
   },
   "betti": {
    "delta": -3.9476,
    "ci": 20.0068,
    "clear": false
   }
  },
  "Z_noaug": {
   "dice": {
    "delta": -0.0094,
    "ci": 0.0267,
    "clear": false
   },
   "cldice": {
    "delta": 0.0053,
    "ci": 0.0108,
    "clear": false
   },
   "betti": {
    "delta": 3.2969,
    "ci": 19.8757,
    "clear": false
   }
  },
  "Z_v2recipe": {
   "dice": {
    "delta": -0.0081,
    "ci": 0.0188,
    "clear": false
   },
   "cldice": {
    "delta": 0.0015,
    "ci": 0.0143,
    "clear": false
   },
   "betti": {
    "delta": -9.2167,
    "ci": 16.5123,
    "clear": false
   }
  }
 },
 "novelty1": {
  "E_mc": {
   "delta": 0.1073,
   "ci": 0.0625,
   "clear": true
  },
  "E_det": {
   "delta": 0.0698,
   "ci": 0.1767,
   "clear": false
  },
  "H": {
   "delta": -0.1374,
   "ci": 0.1275,
   "clear": true
  },
  "E_vs_mc": {
   "delta": 0.1547,
   "ci": 0.1095,
   "clear": true
  }
 },
 "uncertainty": {
  "A_unet[det]": {
   "rho_px": 0.3380919243255047,
   "rho_img": -0.034335737121186524,
   "gain": -0.9334444696195453,
   "ece": 0.042165915624573,
   "ece_ts": 0.045600406154252066,
   "px_err@100": 0.05568470377415887,
   "px_err@90": 0.0223476773047223
  },
  "C_unet_topo_dg[det]": {
   "rho_px": 0.3430486876413907,
   "rho_img": 0.00830039297963231,
   "gain": 1.7058516114840636,
   "ece": 0.04459884372808546,
   "ece_ts": 0.05153912573207753,
   "px_err@100": 0.0563490578980633,
   "px_err@90": 0.021997453823738567
  },
  "E_tglitenet[det]": {
   "rho_px": 0.34231118999597693,
   "rho_img": -0.08115831504768613,
   "gain": -9.877230660434138,
   "ece": 0.046345384518091594,
   "ece_ts": 0.05326635754841627,
   "px_err@100": 0.05878001023783643,
   "px_err@90": 0.024145052611049236
  },
  "E_tglitenet[mc]": {
   "rho_px": 0.34123894960060364,
   "rho_img": -0.07756741588194282,
   "gain": -8.58056004180561,
   "ece": 0.0469725995382713,
   "ece_ts": 0.05410270079972867,
   "px_err@100": 0.05840085258175213,
   "px_err@90": 0.0235683462728571
  },
  "ENS_D_compact[ens]": {
   "rho_px": 0.34359017110630385,
   "rho_img": -0.02632109175153437,
   "gain": 2.0809917110871843,
   "ece": 0.04634122579782213,
   "ece_ts": 0.05360830470940133,
   "px_err@100": 0.057911378243770496,
   "px_err@90": 0.0241933804460574
  }
 },
 "datasets": [
  {
   "id": "drive",
   "n": 40,
   "w": 565,
   "h": 584,
   "vessel_pct": 12.6,
   "fov": 540,
   "calibre_native": 3.5,
   "calibre_norm": 3.4,
   "green": "96 \u00b1 19",
   "fov_mask": "official"
  },
  {
   "id": "stare",
   "n": 20,
   "w": 700,
   "h": 605,
   "vessel_pct": 10.5,
   "fov": 638,
   "calibre_native": 4.3,
   "calibre_norm": 3.5,
   "green": "109 \u00b1 24",
   "fov_mask": "estimated"
  },
  {
   "id": "chase_db1",
   "n": 28,
   "w": 999,
   "h": 960,
   "vessel_pct": 10.3,
   "fov": 899,
   "calibre_native": 6.1,
   "calibre_norm": 3.6,
   "green": "62 \u00b1 26",
   "fov_mask": "estimated"
  },
  {
   "id": "hrf",
   "n": 45,
   "w": 3504,
   "h": 2336,
   "vessel_pct": 9.1,
   "fov": 3262,
   "calibre_native": 8.8,
   "calibre_norm": 1.4,
   "green": "60 \u00b1 17",
   "fov_mask": "official"
  },
  {
   "id": "fives",
   "n": 200,
   "w": 2048,
   "h": 2048,
   "vessel_pct": 9.7,
   "fov": 1969,
   "calibre_native": 13.2,
   "calibre_norm": 3.5,
   "green": "53 \u00b1 17",
   "fov_mask": "estimated"
  }
 ],
 "decisions": {
  "arch": {
   "L_cap": {
    "val_dice": [
     0.809458394212999,
     0.811434117102485,
     0.8100648093248223
    ],
    "cpu_ms": 24.713150050956756,
    "params": 66125,
    "mean_val_dice": 0.8103191068801022
   },
   "L_wnet": {
    "val_dice": [
     0.8144369054106942,
     0.8087753034808836,
     0.808048951995184
    ],
    "cpu_ms": 29.112499963957816,
    "params": 64262,
    "mean_val_dice": 0.8104203869622539
   }
  },
  "size_rule": {
   "H_tglite70k": {
    "mean_val_dice": 0.8084697863929202,
    "params": 66125
   },
   "R_lite_w16": {
    "mean_val_dice": 0.8139588123964329,
    "params": 211569
   },
   "R_unet_w8": {
    "mean_val_dice": 0.8111212926140413,
    "params": 486409
   }
  },
  "backbone": {
   "K_lite_w16": {
    "mean_val_dice": 0.8163242943271358,
    "val_dice": [
     0.8158255647296793,
     0.8163412675701466,
     0.816806050681582
    ],
    "cpu_ms": 108.78839995712042,
    "params": 211569,
    "eligible": true,
    "within_tol": true
   },
   "K_hybrid_w16": {
    "mean_val_dice": 0.8171662734602317,
    "val_dice": [
     0.8181246795146159,
     0.8168208454594252,
     0.8165532954066542
    ],
    "cpu_ms": 89.18830001493916,
    "params": 259777,
    "eligible": true,
    "within_tol": true
   },
   "K_unet_w8": {
    "mean_val_dice": 0.8161628635218113,
    "val_dice": [
     0.8161173764220809,
     0.8158769689509405,
     0.8164942451924126
    ],
    "cpu_ms": 38.506649958435446,
    "params": 486409,
    "eligible": true,
    "within_tol": true
   },
   "K_unet_w16": {
    "mean_val_dice": 0.8182139898225844,
    "val_dice": [
     0.8182988849511634,
     0.8173163308815138,
     0.8190267536350762
    ],
    "cpu_ms": 86.03950001997873,
    "params": 1942289,
    "eligible": true,
    "within_tol": true
   }
  },
  "backbone_choice": "K_unet_w8",
  "frozen_v2": "2026-11-11T18:52:55",
  "frozen_v21": "2026-11-14T05:32:53"
 },
 "int8": {
  "fp32_ms": 16.3,
  "int8_ms": 24.8,
  "fp32_kb": 266,
  "int8_kb": 109
 },
 "efficiency": {
  "A_unet": {
   "params_M": 31.0365,
   "gflops": 384.74,
   "cpu_4t": 935.9,
   "gpu_fp32": 27.81
  },
  "B_unet_topo": {
   "params_M": 31.0365,
   "gflops": 384.74,
   "cpu_4t": 988.7,
   "gpu_fp32": 27.65
  },
  "C_unet_topo_dg": {
   "params_M": 31.0365,
   "gflops": 384.74,
   "cpu_4t": 1003.4,
   "gpu_fp32": 27.61
  },
  "D_compact": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.9,
   "gpu_fp32": 3.4
  },
  "E_tglitenet": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.0,
   "gpu_fp32": 3.32
  },
  "K_hybrid_w16": {
   "params_M": 0.2598,
   "gflops": 15.13,
   "cpu_4t": 89.2,
   "gpu_fp32": 5.53
  },
  "K_lite_w16": {
   "params_M": 0.2116,
   "gflops": 5.53,
   "cpu_4t": 108.8,
   "gpu_fp32": 6.88
  },
  "K_unet_w16": {
   "params_M": 1.9423,
   "gflops": 24.11,
   "cpu_4t": 86.0,
   "gpu_fp32": 5.32
  },
  "K_unet_w8": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 38.5,
   "gpu_fp32": 3.31
  },
  "S_compact_w24": {
   "params_M": 4.3676,
   "gflops": 54.18,
   "cpu_4t": 163.3,
   "gpu_fp32": 8.69
  },
  "S_compact_w32": {
   "params_M": 7.7625,
   "gflops": 96.27,
   "cpu_4t": 260.4,
   "gpu_fp32": 11.58
  },
  "S_compact_w8": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 29.9,
   "gpu_fp32": 3.41
  },
  "V_compact_cbdice": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 39.1,
   "gpu_fp32": 3.37
  },
  "V_compact_srl": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.9,
   "gpu_fp32": 3.44
  },
  "W_compact_warm": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.8,
   "gpu_fp32": 3.33
  },
  "W_unet_topo_dg_warm": {
   "params_M": 31.0365,
   "gflops": 384.74,
   "cpu_4t": 906.3,
   "gpu_fp32": 27.53
  },
  "X_compact_dg": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.0,
   "gpu_fp32": 3.46
  },
  "X_compact_plain": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.4,
   "gpu_fp32": 3.42
  },
  "X_compact_topo": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.6,
   "gpu_fp32": 3.48
  },
  "X_unet_dg": {
   "params_M": 31.0365,
   "gflops": 384.74,
   "cpu_4t": 957.4,
   "gpu_fp32": 27.56
  },
  "Z_compact_nocons": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 30.6,
   "gpu_fp32": 3.43
  },
  "Z_compact_noskel": {
   "params_M": 0.4864,
   "gflops": 6.05,
   "cpu_4t": 31.7,
   "gpu_fp32": 3.39
  }
 },
 "n_trainings": {
  "v2": 76,
  "v21": 38
 },
 "viewer": {
  "drive": {
   "id": "14",
   "native_hw": [
    584,
    565
   ],
   "maps": {
    "v21_H_tglite70k": {
     "threshold": 0.65,
     "dice_normalised": 0.8089
    },
    "v21_H_tglite70k_scalesel": {
     "threshold": 0.65,
     "dice_normalised": 0.8089
    },
    "v21_R_lite_w16": {
     "threshold": 0.6,
     "dice_normalised": 0.8198
    },
    "v21_Z_v2recipe": {
     "threshold": 0.7,
     "dice_normalised": 0.8211
    },
    "v2_E_tglitenet": {
     "threshold": 0.7,
     "dice_normalised": 0.818
    },
    "v2_C_unet_topo_dg": {
     "threshold": 0.65,
     "dice_normalised": 0.8233
    }
   },
   "selected_scale": 1.0
  },
  "stare": {
   "id": "im0162",
   "native_hw": [
    605,
    700
   ],
   "maps": {
    "v21_H_tglite70k": {
     "threshold": 0.65,
     "dice_normalised": 0.7832
    },
    "v21_H_tglite70k_scalesel": {
     "threshold": 0.65,
     "dice_normalised": 0.7832
    },
    "v21_R_lite_w16": {
     "threshold": 0.6,
     "dice_normalised": 0.7879
    },
    "v21_Z_v2recipe": {
     "threshold": 0.7,
     "dice_normalised": 0.7877
    },
    "v2_E_tglitenet": {
     "threshold": 0.7,
     "dice_normalised": 0.7885
    },
    "v2_C_unet_topo_dg": {
     "threshold": 0.65,
     "dice_normalised": 0.7968
    }
   },
   "selected_scale": 1.0
  },
  "chase_db1": {
   "id": "Image_02L",
   "native_hw": [
    960,
    999
   ],
   "maps": {
    "v21_H_tglite70k": {
     "threshold": 0.65,
     "dice_normalised": 0.754
    },
    "v21_H_tglite70k_scalesel": {
     "threshold": 0.65,
     "dice_normalised": 0.754
    },
    "v21_R_lite_w16": {
     "threshold": 0.6,
     "dice_normalised": 0.7752
    },
    "v21_Z_v2recipe": {
     "threshold": 0.7,
     "dice_normalised": 0.7853
    },
    "v2_E_tglitenet": {
     "threshold": 0.7,
     "dice_normalised": 0.7862
    },
    "v2_C_unet_topo_dg": {
     "threshold": 0.65,
     "dice_normalised": 0.794
    }
   },
   "selected_scale": 1.0
  },
  "hrf": {
   "id": "07_dr",
   "native_hw": [
    2336,
    3504
   ],
   "maps": {
    "v21_H_tglite70k": {
     "threshold": 0.65,
     "dice_normalised": 0.6917
    },
    "v21_H_tglite70k_scalesel": {
     "threshold": 0.65,
     "dice_normalised": 0.7394
    },
    "v21_R_lite_w16": {
     "threshold": 0.6,
     "dice_normalised": 0.6943
    },
    "v21_Z_v2recipe": {
     "threshold": 0.7,
     "dice_normalised": 0.6956
    },
    "v2_E_tglitenet": {
     "threshold": 0.7,
     "dice_normalised": 0.7023
    },
    "v2_C_unet_topo_dg": {
     "threshold": 0.65,
     "dice_normalised": 0.7054
    }
   },
   "selected_scale": 1.5
  },
  "fives": {
   "id": "120_G",
   "native_hw": [
    2048,
    2048
   ],
   "maps": {
    "v21_H_tglite70k": {
     "threshold": 0.65,
     "dice_normalised": 0.8509
    },
    "v21_H_tglite70k_scalesel": {
     "threshold": 0.65,
     "dice_normalised": 0.8509
    },
    "v21_R_lite_w16": {
     "threshold": 0.6,
     "dice_normalised": 0.8799
    },
    "v21_Z_v2recipe": {
     "threshold": 0.7,
     "dice_normalised": 0.9045
    },
    "v2_E_tglitenet": {
     "threshold": 0.7,
     "dice_normalised": 0.9091
    },
    "v2_C_unet_topo_dg": {
     "threshold": 0.65,
     "dice_normalised": 0.9114
    }
   },
   "selected_scale": 1.0
  }
 }
};
