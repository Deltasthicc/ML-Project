### v2.1 decisions (DRIVE-val only)

```json
{
  "rules_frozen": true,
  "rules_frozen_at": "2026-11-13T12:45:28",
  "rules_sha256": {
    "protocol/PROTOCOL_v2.1.md": "08d55bd6fae9ec1ffe87bead540d4c63944dbb1efe0a325b45474f2954725c4d",
    "configs/v21.yaml": "87486a1eb30824d6d8a924da753f511508b9a85bf873868e1843b9fce9de652b"
  },
  "arch": "L_cap",
  "arch_selection": {
    "candidates": {
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
    "rule": "mean DRIVE-val Dice within 0.005 of the best candidate; lowest CPU latency"
  },
  "weights": {
    "loss": {
      "lambda_topo": 0.5
    },
    "distill": {
      "lambda_kd_skel": 1.0
    }
  },
  "weight_selection": {
    "grid": {
      "G_t025_k050": {
        "lambda_topo": 0.25,
        "lambda_kd_skel": 0.5,
        "score": 0.8302695084592144,
        "val_dice": 0.8112058137641922,
        "val_cldice": 0.8493332031542368
      },
      "G_t025_k100": {
        "lambda_topo": 0.25,
        "lambda_kd_skel": 1.0,
        "score": 0.8299934312430954,
        "val_dice": 0.8102766051601882,
        "val_cldice": 0.8497102573260027
      },
      "G_t050_k050": {
        "lambda_topo": 0.5,
        "lambda_kd_skel": 0.5,
        "score": 0.8297377629770245,
        "val_dice": 0.8104462556577421,
        "val_cldice": 0.8490292702963069
      },
      "G_t050_k100": {
        "lambda_topo": 0.5,
        "lambda_kd_skel": 1.0,
        "score": 0.8312219634517113,
        "val_dice": 0.8090461457784988,
        "val_cldice": 0.8533977811249238
      }
    },
    "choice": "G_t050_k100",
    "rule": "max mean (val Dice + val clDice)/2 at the best checkpoint, 2 seeds; ties -> defaults"
  },
  "headline": "R_lite_w16",
  "size_rule": {
    "candidates": {
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
    "rule": "smallest model whose mean DRIVE-val Dice (seeds [0, 1, 2]) is within 0.005 of the best"
  },
  "frozen": true,
  "frozen_at": "2026-11-14T05:32:53"
}
```
