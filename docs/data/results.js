window.RESULTS = {
 "datasets": [
  "drive",
  "stare",
  "chase_db1",
  "hrf"
 ],
 "stages": [
  {
   "tag": "A",
   "label": "U-Net",
   "exp": "A_unet",
   "params_m": 31.0365,
   "gflops": 384.7,
   "cpu_ms": 1834.0,
   "mc_passes": 1,
   "seeds": 5,
   "drive_dice": 0.8301,
   "drive_dice_std": 0.0014,
   "drive_cldice": 0.8303,
   "drive_cldice_std": 0.0025,
   "drive_betti0_err": 53.91,
   "drive_betti0_err_std": 2.9074,
   "stare_dice": 0.7994,
   "stare_dice_std": 0.0061,
   "stare_cldice": 0.8662,
   "stare_cldice_std": 0.0023,
   "stare_betti0_err": 36.46,
   "stare_betti0_err_std": 5.6647,
   "chase_db1_dice": 0.7764,
   "chase_db1_dice_std": 0.0033,
   "chase_db1_cldice": 0.834,
   "chase_db1_cldice_std": 0.0025,
   "chase_db1_betti0_err": 53.8643,
   "chase_db1_betti0_err_std": 5.1307,
   "hrf_dice": 0.7115,
   "hrf_dice_std": 0.0099,
   "hrf_cldice": 0.8062,
   "hrf_cldice_std": 0.0045,
   "hrf_betti0_err": 134.6578,
   "hrf_betti0_err_std": 32.8187,
   "unseen_dice": 0.7624,
   "unseen_dice_std": 0.0063,
   "unseen_cldice": 0.8355,
   "unseen_cldice_std": 0.0022,
   "unseen_betti0_err": 74.994,
   "unseen_betti0_err_std": 14.356,
   "gap": 0.0676
  },
  {
   "tag": "B",
   "label": "+ topology loss",
   "exp": "B_unet_topo",
   "params_m": 31.0365,
   "gflops": 384.7,
   "cpu_ms": 1834.0,
   "mc_passes": 1,
   "seeds": 5,
   "drive_dice": 0.8268,
   "drive_dice_std": 0.0007,
   "drive_cldice": 0.8454,
   "drive_cldice_std": 0.0008,
   "drive_betti0_err": 34.12,
   "drive_betti0_err_std": 2.1942,
   "stare_dice": 0.7943,
   "stare_dice_std": 0.0033,
   "stare_cldice": 0.8739,
   "stare_cldice_std": 0.0028,
   "stare_betti0_err": 23.01,
   "stare_betti0_err_std": 2.1597,
   "chase_db1_dice": 0.772,
   "chase_db1_dice_std": 0.004,
   "chase_db1_cldice": 0.8401,
   "chase_db1_cldice_std": 0.0065,
   "chase_db1_betti0_err": 31.8786,
   "chase_db1_betti0_err_std": 3.4518,
   "hrf_dice": 0.6932,
   "hrf_dice_std": 0.0054,
   "hrf_cldice": 0.8137,
   "hrf_cldice_std": 0.0063,
   "hrf_betti0_err": 40.1956,
   "hrf_betti0_err_std": 3.5989,
   "unseen_dice": 0.7532,
   "unseen_dice_std": 0.004,
   "unseen_cldice": 0.8426,
   "unseen_cldice_std": 0.0034,
   "unseen_betti0_err": 31.6947,
   "unseen_betti0_err_std": 2.2707,
   "gap": 0.0736
  },
  {
   "tag": "C",
   "label": "+ DG augmentation",
   "exp": "C_unet_topo_dg",
   "params_m": 31.0365,
   "gflops": 384.7,
   "cpu_ms": 1834.0,
   "mc_passes": 1,
   "seeds": 5,
   "drive_dice": 0.8259,
   "drive_dice_std": 0.0009,
   "drive_cldice": 0.849,
   "drive_cldice_std": 0.0011,
   "drive_betti0_err": 29.7,
   "drive_betti0_err_std": 3.3079,
   "stare_dice": 0.8105,
   "stare_dice_std": 0.0014,
   "stare_cldice": 0.8759,
   "stare_cldice_std": 0.0017,
   "stare_betti0_err": 23.1,
   "stare_betti0_err_std": 1.7521,
   "chase_db1_dice": 0.7687,
   "chase_db1_dice_std": 0.0024,
   "chase_db1_cldice": 0.8523,
   "chase_db1_cldice_std": 0.0019,
   "chase_db1_betti0_err": 22.9786,
   "chase_db1_betti0_err_std": 1.0569,
   "hrf_dice": 0.7051,
   "hrf_dice_std": 0.0023,
   "hrf_cldice": 0.8119,
   "hrf_cldice_std": 0.0006,
   "hrf_betti0_err": 37.7156,
   "hrf_betti0_err_std": 4.0848,
   "unseen_dice": 0.7614,
   "unseen_dice_std": 0.0009,
   "unseen_cldice": 0.8467,
   "unseen_cldice_std": 0.0011,
   "unseen_betti0_err": 27.9314,
   "unseen_betti0_err_std": 2.1112,
   "gap": 0.0645
  },
  {
   "tag": "D",
   "label": "+ compact backbone",
   "exp": "D_lite_topo_dg",
   "params_m": 0.2116,
   "gflops": 5.5,
   "cpu_ms": 209.0,
   "mc_passes": 1,
   "seeds": 5,
   "drive_dice": 0.8217,
   "drive_dice_std": 0.0008,
   "drive_cldice": 0.8381,
   "drive_cldice_std": 0.0027,
   "drive_betti0_err": 44.34,
   "drive_betti0_err_std": 7.6101,
   "stare_dice": 0.8013,
   "stare_dice_std": 0.0045,
   "stare_cldice": 0.8709,
   "stare_cldice_std": 0.0059,
   "stare_betti0_err": 34.14,
   "stare_betti0_err_std": 4.9399,
   "chase_db1_dice": 0.7633,
   "chase_db1_dice_std": 0.0013,
   "chase_db1_cldice": 0.8414,
   "chase_db1_cldice_std": 0.0048,
   "chase_db1_betti0_err": 31.7643,
   "chase_db1_betti0_err_std": 5.6267,
   "hrf_dice": 0.7051,
   "hrf_dice_std": 0.004,
   "hrf_cldice": 0.7848,
   "hrf_cldice_std": 0.0066,
   "hrf_betti0_err": 68.2267,
   "hrf_betti0_err_std": 10.9018,
   "unseen_dice": 0.7566,
   "unseen_dice_std": 0.0028,
   "unseen_cldice": 0.8323,
   "unseen_cldice_std": 0.0022,
   "unseen_betti0_err": 44.7103,
   "unseen_betti0_err_std": 6.4905,
   "gap": 0.0651
  },
  {
   "tag": "E",
   "label": "TG-LiteNet",
   "exp": "E_tglitenet",
   "params_m": 0.2116,
   "gflops": 5.5,
   "cpu_ms": 209.0,
   "mc_passes": 1,
   "seeds": 5,
   "drive_dice": 0.8216,
   "drive_dice_std": 0.0006,
   "drive_cldice": 0.8386,
   "drive_cldice_std": 0.0035,
   "drive_betti0_err": 41.59,
   "drive_betti0_err_std": 4.4368,
   "stare_dice": 0.8011,
   "stare_dice_std": 0.0052,
   "stare_cldice": 0.8703,
   "stare_cldice_std": 0.0045,
   "stare_betti0_err": 32.04,
   "stare_betti0_err_std": 3.5418,
   "chase_db1_dice": 0.762,
   "chase_db1_dice_std": 0.0034,
   "chase_db1_cldice": 0.8428,
   "chase_db1_cldice_std": 0.003,
   "chase_db1_betti0_err": 27.7714,
   "chase_db1_betti0_err_std": 2.0492,
   "hrf_dice": 0.7029,
   "hrf_dice_std": 0.0056,
   "hrf_cldice": 0.7852,
   "hrf_cldice_std": 0.009,
   "hrf_betti0_err": 66.0356,
   "hrf_betti0_err_std": 3.8727,
   "unseen_dice": 0.7553,
   "unseen_dice_std": 0.0045,
   "unseen_cldice": 0.8328,
   "unseen_cldice_std": 0.0022,
   "unseen_betti0_err": 41.949,
   "unseen_betti0_err_std": 2.8439,
   "gap": 0.0663
  },
  {
   "tag": "E-MC",
   "label": "TG-LiteNet + MC dropout",
   "exp": "E_tglitenet",
   "params_m": 0.2116,
   "gflops": 5.5,
   "cpu_ms": 209.0,
   "mc_passes": 20,
   "seeds": 5,
   "drive_dice": 0.8216,
   "drive_dice_std": 0.0007,
   "drive_cldice": 0.8384,
   "drive_cldice_std": 0.0031,
   "drive_betti0_err": 41.78,
   "drive_betti0_err_std": 4.89,
   "stare_dice": 0.8019,
   "stare_dice_std": 0.0048,
   "stare_cldice": 0.8712,
   "stare_cldice_std": 0.0042,
   "stare_betti0_err": 32.42,
   "stare_betti0_err_std": 3.7695,
   "chase_db1_dice": 0.7627,
   "chase_db1_dice_std": 0.0035,
   "chase_db1_cldice": 0.8429,
   "chase_db1_cldice_std": 0.0028,
   "chase_db1_betti0_err": 28.55,
   "chase_db1_betti0_err_std": 1.7824,
   "hrf_dice": 0.7032,
   "hrf_dice_std": 0.0053,
   "hrf_cldice": 0.7842,
   "hrf_cldice_std": 0.009,
   "hrf_betti0_err": 67.0133,
   "hrf_betti0_err_std": 4.1964,
   "unseen_dice": 0.756,
   "unseen_dice_std": 0.0043,
   "unseen_cldice": 0.8328,
   "unseen_cldice_std": 0.0022,
   "unseen_betti0_err": 42.6611,
   "unseen_betti0_err_std": 2.9835,
   "gap": 0.0657
  }
 ],
 "controls": [
  {
   "exp": "A_unet",
   "label": "A  U-Net",
   "seeds": 5,
   "fixed": {
    "dice": 0.7624,
    "cldice": 0.8355,
    "betti0_err": 74.994
   },
   "clean": {
    "dice": 0.763,
    "cldice": 0.8355,
    "betti0_err": 4.6721
   },
   "tuned": {
    "dice": 0.7566,
    "cldice": 0.8369,
    "betti0_err": 69.4429
   },
   "thr_range": [
    0.45,
    0.5
   ]
  },
  {
   "exp": "B_unet_topo",
   "label": "B  + topology",
   "seeds": 5,
   "fixed": {
    "dice": 0.7532,
    "cldice": 0.8426,
    "betti0_err": 31.6947
   },
   "clean": {
    "dice": 0.7544,
    "cldice": 0.841,
    "betti0_err": 4.9415
   },
   "tuned": {
    "dice": 0.7706,
    "cldice": 0.8383,
    "betti0_err": 38.4258
   },
   "thr_range": [
    0.55,
    0.65
   ]
  },
  {
   "exp": "C_unet_topo_dg",
   "label": "C  + DG aug.",
   "seeds": 5,
   "fixed": {
    "dice": 0.7614,
    "cldice": 0.8467,
    "betti0_err": 27.9314
   },
   "clean": {
    "dice": 0.7632,
    "cldice": 0.8471,
    "betti0_err": 4.6948
   },
   "tuned": {
    "dice": 0.7817,
    "cldice": 0.843,
    "betti0_err": 37.1375
   },
   "thr_range": [
    0.55,
    0.65
   ]
  },
  {
   "exp": "D_lite_topo_dg",
   "label": "D  compact",
   "seeds": 5,
   "fixed": {
    "dice": 0.7566,
    "cldice": 0.8323,
    "betti0_err": 44.7103
   },
   "clean": {
    "dice": 0.7585,
    "cldice": 0.8281,
    "betti0_err": 4.9593
   },
   "tuned": {
    "dice": 0.7715,
    "cldice": 0.8293,
    "betti0_err": 49.7685
   },
   "thr_range": [
    0.5,
    0.65
   ]
  },
  {
   "exp": "E_tglitenet",
   "label": "E  TG-LiteNet",
   "seeds": 5,
   "fixed": {
    "dice": 0.7553,
    "cldice": 0.8328,
    "betti0_err": 41.949
   },
   "clean": {
    "dice": 0.757,
    "cldice": 0.8288,
    "betti0_err": 4.9116
   },
   "tuned": {
    "dice": 0.7707,
    "cldice": 0.83,
    "betti0_err": 47.403
   },
   "thr_range": [
    0.5,
    0.65
   ]
  },
  {
   "exp": "X_unet_dg",
   "label": "U-Net + DG only",
   "seeds": 3,
   "fixed": {
    "dice": 0.7708,
    "cldice": 0.8375,
    "betti0_err": 72.9604
   },
   "clean": {
    "dice": 0.7718,
    "cldice": 0.8391,
    "betti0_err": 4.8008
   },
   "tuned": {
    "dice": 0.777,
    "cldice": 0.836,
    "betti0_err": 80.2423
   },
   "thr_range": [
    0.5,
    0.55
   ]
  },
  {
   "exp": "S_unet_b8",
   "label": "plain U-Net, base 8",
   "seeds": 3,
   "fixed": {
    "dice": 0.758,
    "cldice": 0.8329,
    "betti0_err": 48.8848
   },
   "clean": {
    "dice": 0.7594,
    "cldice": 0.8299,
    "betti0_err": 4.7399
   },
   "tuned": {
    "dice": 0.7642,
    "cldice": 0.8314,
    "betti0_err": 51.2517
   },
   "thr_range": [
    0.5,
    0.55
   ]
  },
  {
   "exp": "S_unet_b16",
   "label": "plain U-Net, base 16",
   "seeds": 3,
   "fixed": {
    "dice": 0.7573,
    "cldice": 0.8398,
    "betti0_err": 33.6596
   },
   "clean": {
    "dice": 0.7592,
    "cldice": 0.8389,
    "betti0_err": 4.8107
   },
   "tuned": {
    "dice": 0.7771,
    "cldice": 0.8364,
    "betti0_err": 38.7804
   },
   "thr_range": [
    0.55,
    0.65
   ]
  },
  {
   "exp": "X_lite_dg",
   "label": "compact + DG only",
   "seeds": 3,
   "fixed": {
    "dice": 0.7671,
    "cldice": 0.8265,
    "betti0_err": 87.7004
   },
   "clean": {
    "dice": 0.7677,
    "cldice": 0.825,
    "betti0_err": 4.6041
   },
   "tuned": {
    "dice": 0.7572,
    "cldice": 0.8281,
    "betti0_err": 84.79
   },
   "thr_range": [
    0.4,
    0.5
   ]
  }
 ],
 "tradeoff": [
  {
   "exp": "A_unet",
   "name": "A",
   "main": true,
   "cpu_ms": 1834.0,
   "params_m": 31.036,
   "dice": 0.7624,
   "cldice": 0.8355,
   "betti0": 75.0
  },
  {
   "exp": "B_unet_topo",
   "name": "B",
   "main": true,
   "cpu_ms": 1834.0,
   "params_m": 31.037,
   "dice": 0.7532,
   "cldice": 0.8426,
   "betti0": 31.7
  },
  {
   "exp": "C_unet_topo_dg",
   "name": "C",
   "main": true,
   "cpu_ms": 1834.0,
   "params_m": 31.037,
   "dice": 0.7614,
   "cldice": 0.8467,
   "betti0": 27.9
  },
  {
   "exp": "D_lite_topo_dg",
   "name": "D",
   "main": true,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7566,
   "cldice": 0.8323,
   "betti0": 44.7
  },
  {
   "exp": "E_tglitenet",
   "name": "E",
   "main": true,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7553,
   "cldice": 0.8328,
   "betti0": 41.9
  },
  {
   "exp": "S_unet_b16",
   "name": "plain U-Net, base 16",
   "main": true,
   "cpu_ms": 187.0,
   "params_m": 1.942,
   "dice": 0.7573,
   "cldice": 0.8398,
   "betti0": 33.7
  },
  {
   "exp": "S_unet_b8",
   "name": "plain U-Net, base 8",
   "main": true,
   "cpu_ms": 67.0,
   "params_m": 0.486,
   "dice": 0.758,
   "cldice": 0.8329,
   "betti0": 48.9
  },
  {
   "exp": "V_lite_cbdice",
   "name": "V_lite_cbdice",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7448,
   "cldice": 0.834,
   "betti0": 39.1
  },
  {
   "exp": "V_lite_cldice_srl",
   "name": "V_lite_cldice_srl",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7335,
   "cldice": 0.8325,
   "betti0": 36.5
  },
  {
   "exp": "V_lite_srl",
   "name": "V_lite_srl",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7432,
   "cldice": 0.8279,
   "betti0": 55.3
  },
  {
   "exp": "W_lite_w24",
   "name": "W_lite_w24",
   "main": false,
   "cpu_ms": 343.0,
   "params_m": 0.464,
   "dice": 0.7641,
   "cldice": 0.8352,
   "betti0": 40.0
  },
  {
   "exp": "W_lite_w32",
   "name": "W_lite_w32",
   "main": false,
   "cpu_ms": 487.0,
   "params_m": 0.815,
   "dice": 0.7594,
   "cldice": 0.8377,
   "betti0": 34.3
  },
  {
   "exp": "W_lite_w48",
   "name": "W_lite_w48",
   "main": false,
   "cpu_ms": 838.0,
   "params_m": 1.81,
   "dice": 0.7591,
   "cldice": 0.8389,
   "betti0": 30.1
  },
  {
   "exp": "W_lite_w8",
   "name": "W_lite_w8",
   "main": false,
   "cpu_ms": 76.0,
   "params_m": 0.057,
   "dice": 0.7537,
   "cldice": 0.8222,
   "betti0": 61.1
  },
  {
   "exp": "X_lite_dg",
   "name": "compact + DG only",
   "main": true,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7671,
   "cldice": 0.8265,
   "betti0": 87.7
  },
  {
   "exp": "X_lite_plain",
   "name": "X_lite_plain",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7452,
   "cldice": 0.8235,
   "betti0": 85.8
  },
  {
   "exp": "X_lite_topo",
   "name": "X_lite_topo",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7469,
   "cldice": 0.8285,
   "betti0": 45.6
  },
  {
   "exp": "X_unet_dg",
   "name": "U-Net + DG only",
   "main": true,
   "cpu_ms": 1834.0,
   "params_m": 31.036,
   "dice": 0.7708,
   "cldice": 0.8375,
   "betti0": 73.0
  },
  {
   "exp": "Z_lite_nocons",
   "name": "Z_lite_nocons",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7587,
   "cldice": 0.8328,
   "betti0": 52.2
  },
  {
   "exp": "Z_lite_noskel",
   "name": "Z_lite_noskel",
   "main": false,
   "cpu_ms": 209.0,
   "params_m": 0.212,
   "dice": 0.7608,
   "cldice": 0.8314,
   "betti0": 53.8
  }
 ],
 "dataset_info": {
  "drive": {
   "n": 40,
   "w": 565,
   "h": 584,
   "vessel_pct": 12.9,
   "fov_px": 527,
   "calibre_native": 3.5,
   "calibre_norm": 3.5,
   "green": "97 \u00b1 17"
  },
  "stare": {
   "n": 20,
   "w": 700,
   "h": 605,
   "vessel_pct": 10.5,
   "fov_px": 638,
   "calibre_native": 4.3,
   "calibre_norm": 3.5,
   "green": "109 \u00b1 24"
  },
  "chase_db1": {
   "n": 28,
   "w": 999,
   "h": 960,
   "vessel_pct": 10.3,
   "fov_px": 899,
   "calibre_native": 6.1,
   "calibre_norm": 3.6,
   "green": "62 \u00b1 26"
  },
  "hrf": {
   "n": 45,
   "w": 3504,
   "h": 2336,
   "vessel_pct": 9.1,
   "fov_px": 3262,
   "calibre_native": 8.8,
   "calibre_norm": 1.4,
   "green": "60 \u00b1 17"
  }
 },
 "factorial": [
  {
   "backbone": "unet",
   "topology": false,
   "dg": false,
   "exp": "A_unet",
   "dice": 0.7624,
   "cldice": 0.8355,
   "betti0": 75.0,
   "gap": 0.0676
  },
  {
   "backbone": "unet",
   "topology": true,
   "dg": false,
   "exp": "B_unet_topo",
   "dice": 0.7532,
   "cldice": 0.8426,
   "betti0": 31.7,
   "gap": 0.0736
  },
  {
   "backbone": "unet",
   "topology": false,
   "dg": true,
   "exp": "X_unet_dg",
   "dice": 0.7708,
   "cldice": 0.8375,
   "betti0": 73.0,
   "gap": 0.06
  },
  {
   "backbone": "unet",
   "topology": true,
   "dg": true,
   "exp": "C_unet_topo_dg",
   "dice": 0.7614,
   "cldice": 0.8467,
   "betti0": 27.9,
   "gap": 0.0645
  },
  {
   "backbone": "lite",
   "topology": false,
   "dg": false,
   "exp": "X_lite_plain",
   "dice": 0.7452,
   "cldice": 0.8235,
   "betti0": 85.8,
   "gap": 0.0776
  },
  {
   "backbone": "lite",
   "topology": true,
   "dg": false,
   "exp": "X_lite_topo",
   "dice": 0.7469,
   "cldice": 0.8285,
   "betti0": 45.6,
   "gap": 0.0739
  },
  {
   "backbone": "lite",
   "topology": false,
   "dg": true,
   "exp": "X_lite_dg",
   "dice": 0.7671,
   "cldice": 0.8265,
   "betti0": 87.7,
   "gap": 0.0573
  },
  {
   "backbone": "lite",
   "topology": true,
   "dg": true,
   "exp": "D_lite_topo_dg",
   "dice": 0.7566,
   "cldice": 0.8323,
   "betti0": 44.7,
   "gap": 0.0651
  }
 ],
 "risk": {
  "pixel_cov": [
   1.0,
   0.9,
   0.8,
   0.5
  ],
  "image_cov": [
   1.0,
   0.9,
   0.8,
   0.7,
   0.6,
   0.5
  ],
  "datasets": {
   "drive": {
    "rho_px": 0.311,
    "rho_img": 0.151,
    "pixel": [
     0.0472,
     0.0162,
     0.0062,
     0.0019
    ],
    "image": [
     0.1784,
     0.178,
     0.1765,
     0.177,
     0.1774,
     0.1775
    ]
   },
   "stare": {
    "rho_px": 0.318,
    "rho_img": 0.085,
    "pixel": [
     0.0471,
     0.0178,
     0.0036,
     0.0003
    ],
    "image": [
     0.1981,
     0.1965,
     0.1963,
     0.1963,
     0.1971,
     0.1987
    ]
   },
   "chase_db1": {
    "rho_px": 0.341,
    "rho_img": 0.477,
    "pixel": [
     0.0578,
     0.025,
     0.0079,
     0.0013
    ],
    "image": [
     0.2373,
     0.2331,
     0.2299,
     0.2275,
     0.2249,
     0.2242
    ]
   },
   "hrf": {
    "rho_px": 0.34,
    "rho_img": -0.172,
    "pixel": [
     0.065,
     0.0349,
     0.0131,
     0.0036
    ],
    "image": [
     0.2968,
     0.2984,
     0.301,
     0.3029,
     0.3049,
     0.3068
    ]
   }
  }
 },
 "failure": [
  {
   "exp": "A_unet",
   "kind": "fp_rate_by_zone",
   "group": "centre",
   "value": 0.0555
  },
  {
   "exp": "A_unet",
   "kind": "fp_rate_by_zone",
   "group": "edge",
   "value": 0.0354
  },
  {
   "exp": "A_unet",
   "kind": "fp_rate_by_zone",
   "group": "middle",
   "value": 0.0493
  },
  {
   "exp": "A_unet",
   "kind": "fp_share_far",
   "group": "spurious (> 2 px from any vessel)",
   "value": 0.1438
  },
  {
   "exp": "A_unet",
   "kind": "recall_by_width",
   "group": "medium (2-4 px)",
   "value": 0.9348
  },
  {
   "exp": "A_unet",
   "kind": "recall_by_width",
   "group": "thick (> 4 px)",
   "value": 0.9533
  },
  {
   "exp": "A_unet",
   "kind": "recall_by_width",
   "group": "thin  (<= 2 px)",
   "value": 0.8196
  },
  {
   "exp": "A_unet",
   "kind": "recall_by_zone",
   "group": "centre",
   "value": 0.9146
  },
  {
   "exp": "A_unet",
   "kind": "recall_by_zone",
   "group": "edge",
   "value": 0.8859
  },
  {
   "exp": "A_unet",
   "kind": "recall_by_zone",
   "group": "middle",
   "value": 0.9279
  },
  {
   "exp": "A_unet",
   "kind": "stray_pieces_per_image",
   "group": "pieces touching no true vessel",
   "value": 35.8241
  },
  {
   "exp": "A_unet",
   "kind": "stray_pixels_share",
   "group": "of all predicted vessel pixels",
   "value": 0.0094
  },
  {
   "exp": "B_unet_topo",
   "kind": "fp_rate_by_zone",
   "group": "centre",
   "value": 0.0614
  },
  {
   "exp": "B_unet_topo",
   "kind": "fp_rate_by_zone",
   "group": "edge",
   "value": 0.0382
  },
  {
   "exp": "B_unet_topo",
   "kind": "fp_rate_by_zone",
   "group": "middle",
   "value": 0.0538
  },
  {
   "exp": "B_unet_topo",
   "kind": "fp_share_far",
   "group": "spurious (> 2 px from any vessel)",
   "value": 0.1474
  },
  {
   "exp": "B_unet_topo",
   "kind": "recall_by_width",
   "group": "medium (2-4 px)",
   "value": 0.9365
  },
  {
   "exp": "B_unet_topo",
   "kind": "recall_by_width",
   "group": "thick (> 4 px)",
   "value": 0.9528
  },
  {
   "exp": "B_unet_topo",
   "kind": "recall_by_width",
   "group": "thin  (<= 2 px)",
   "value": 0.8333
  },
  {
   "exp": "B_unet_topo",
   "kind": "recall_by_zone",
   "group": "centre",
   "value": 0.9201
  },
  {
   "exp": "B_unet_topo",
   "kind": "recall_by_zone",
   "group": "edge",
   "value": 0.8855
  },
  {
   "exp": "B_unet_topo",
   "kind": "recall_by_zone",
   "group": "middle",
   "value": 0.9324
  },
  {
   "exp": "B_unet_topo",
   "kind": "stray_pieces_per_image",
   "group": "pieces touching no true vessel",
   "value": 18.2648
  },
  {
   "exp": "B_unet_topo",
   "kind": "stray_pixels_share",
   "group": "of all predicted vessel pixels",
   "value": 0.0081
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "fp_rate_by_zone",
   "group": "centre",
   "value": 0.0607
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "fp_rate_by_zone",
   "group": "edge",
   "value": 0.0385
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "fp_rate_by_zone",
   "group": "middle",
   "value": 0.0516
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "fp_share_far",
   "group": "spurious (> 2 px from any vessel)",
   "value": 0.1695
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "recall_by_width",
   "group": "medium (2-4 px)",
   "value": 0.9451
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "recall_by_width",
   "group": "thick (> 4 px)",
   "value": 0.9612
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "recall_by_width",
   "group": "thin  (<= 2 px)",
   "value": 0.855
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "recall_by_zone",
   "group": "centre",
   "value": 0.9245
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "recall_by_zone",
   "group": "edge",
   "value": 0.9066
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "recall_by_zone",
   "group": "middle",
   "value": 0.9352
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "stray_pieces_per_image",
   "group": "pieces touching no true vessel",
   "value": 18.2672
  },
  {
   "exp": "C_unet_topo_dg",
   "kind": "stray_pixels_share",
   "group": "of all predicted vessel pixels",
   "value": 0.0096
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "fp_rate_by_zone",
   "group": "centre",
   "value": 0.0574
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "fp_rate_by_zone",
   "group": "edge",
   "value": 0.035
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "fp_rate_by_zone",
   "group": "middle",
   "value": 0.0486
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "fp_share_far",
   "group": "spurious (> 2 px from any vessel)",
   "value": 0.1583
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "recall_by_width",
   "group": "medium (2-4 px)",
   "value": 0.941
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "recall_by_width",
   "group": "thick (> 4 px)",
   "value": 0.9544
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "recall_by_width",
   "group": "thin  (<= 2 px)",
   "value": 0.8205
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "recall_by_zone",
   "group": "centre",
   "value": 0.9093
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "recall_by_zone",
   "group": "edge",
   "value": 0.8808
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "recall_by_zone",
   "group": "middle",
   "value": 0.9245
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "stray_pieces_per_image",
   "group": "pieces touching no true vessel",
   "value": 20.0672
  },
  {
   "exp": "D_lite_topo_dg",
   "kind": "stray_pixels_share",
   "group": "of all predicted vessel pixels",
   "value": 0.0117
  },
  {
   "exp": "E_tglitenet",
   "kind": "fp_rate_by_zone",
   "group": "centre",
   "value": 0.0567
  },
  {
   "exp": "E_tglitenet",
   "kind": "fp_rate_by_zone",
   "group": "edge",
   "value": 0.0355
  },
  {
   "exp": "E_tglitenet",
   "kind": "fp_rate_by_zone",
   "group": "middle",
   "value": 0.0484
  },
  {
   "exp": "E_tglitenet",
   "kind": "fp_share_far",
   "group": "spurious (> 2 px from any vessel)",
   "value": 0.1541
  },
  {
   "exp": "E_tglitenet",
   "kind": "recall_by_width",
   "group": "medium (2-4 px)",
   "value": 0.941
  },
  {
   "exp": "E_tglitenet",
   "kind": "recall_by_width",
   "group": "thick (> 4 px)",
   "value": 0.9539
  },
  {
   "exp": "E_tglitenet",
   "kind": "recall_by_width",
   "group": "thin  (<= 2 px)",
   "value": 0.8216
  },
  {
   "exp": "E_tglitenet",
   "kind": "recall_by_zone",
   "group": "centre",
   "value": 0.9084
  },
  {
   "exp": "E_tglitenet",
   "kind": "recall_by_zone",
   "group": "edge",
   "value": 0.8811
  },
  {
   "exp": "E_tglitenet",
   "kind": "recall_by_zone",
   "group": "middle",
   "value": 0.9246
  },
  {
   "exp": "E_tglitenet",
   "kind": "stray_pieces_per_image",
   "group": "pieces touching no true vessel",
   "value": 19.7169
  },
  {
   "exp": "E_tglitenet",
   "kind": "stray_pixels_share",
   "group": "of all predicted vessel pixels",
   "value": 0.0113
  }
 ],
 "viewer": {
  "drive": {
   "id": "11",
   "w": 565,
   "h": 584
  },
  "stare": {
   "id": "im0162",
   "w": 578,
   "h": 500
  },
  "chase_db1": {
   "id": "Image_08L",
   "w": 602,
   "h": 579
  },
  "hrf": {
   "id": "08_g",
   "w": 566,
   "h": 377
  }
 }
};
