# Auto-generated findings (facts only)

Deltas are (second − first) in mean over seeds; brackets are 95 % Welch intervals.


## RQ1 topology: A → B (adds soft-clDice + skeleton head)

- drive       dice        A_unet → B_unet_topo: -0.0031 [±0.0028] CI excludes 0 (worsens)
- drive       cldice      A_unet → B_unet_topo: +0.0140 [±0.0058] CI excludes 0 (improves)
- drive       betti0_err  A_unet → B_unet_topo: -17.7000 [±5.4374] CI excludes 0 (improves)
- drive       auc         A_unet → B_unet_topo: -0.0000 [±0.0012] CI includes 0
- unseen      dice        A_unet → B_unet_topo: -0.0075 [±0.0101] CI includes 0
- unseen      cldice      A_unet → B_unet_topo: +0.0054 [±0.0105] CI includes 0
- unseen      betti0_err  A_unet → B_unet_topo: -35.1695 [±16.2475] CI excludes 0 (improves)
- unseen      auc         A_unet → B_unet_topo: -0.0011 [±0.0033] CI includes 0
- gap_unseen  dice        A_unet → B_unet_topo: +0.0044 [±0.0076] CI includes 0
- gap_unseen  cldice      A_unet → B_unet_topo: +0.0086 [±0.0097] CI includes 0
- gap_unseen  auc         A_unet → B_unet_topo: +0.0011 [±0.0033] CI includes 0

## RQ2 generalisation: B → C (adds DG augmentation + consistency)

- drive       dice        B_unet_topo → C_unet_topo_dg: -0.0006 [±0.0017] CI includes 0
- drive       cldice      B_unet_topo → C_unet_topo_dg: +0.0033 [±0.0025] CI excludes 0 (improves)
- drive       betti0_err  B_unet_topo → C_unet_topo_dg: -6.0833 [±8.8499] CI includes 0
- drive       auc         B_unet_topo → C_unet_topo_dg: +0.0003 [±0.0004] CI includes 0
- unseen      dice        B_unet_topo → C_unet_topo_dg: +0.0105 [±0.0084] CI excludes 0 (improves)
- unseen      cldice      B_unet_topo → C_unet_topo_dg: +0.0037 [±0.0108] CI includes 0
- unseen      betti0_err  B_unet_topo → C_unet_topo_dg: -2.2007 [±4.1014] CI includes 0
- unseen      auc         B_unet_topo → C_unet_topo_dg: +0.0023 [±0.0039] CI includes 0
- gap_unseen  dice        B_unet_topo → C_unet_topo_dg: -0.0111 [±0.0063] CI excludes 0 (improves)
- gap_unseen  cldice      B_unet_topo → C_unet_topo_dg: -0.0004 [±0.0107] CI includes 0
- gap_unseen  auc         B_unet_topo → C_unet_topo_dg: -0.0020 [±0.0039] CI includes 0

## RQ3 trade-off: C → D (compact backbone)

- drive       dice        C_unet_topo_dg → D_lite_topo_dg: -0.0046 [±0.0021] CI excludes 0 (worsens)
- drive       cldice      C_unet_topo_dg → D_lite_topo_dg: -0.0090 [±0.0028] CI excludes 0 (worsens)
- drive       betti0_err  C_unet_topo_dg → D_lite_topo_dg: +15.6000 [±13.6950] CI excludes 0 (worsens)
- drive       auc         C_unet_topo_dg → D_lite_topo_dg: -0.0024 [±0.0004] CI excludes 0 (worsens)
- unseen      dice        C_unet_topo_dg → D_lite_topo_dg: -0.0052 [±0.0070] CI includes 0
- unseen      cldice      C_unet_topo_dg → D_lite_topo_dg: -0.0124 [±0.0020] CI excludes 0 (worsens)
- unseen      betti0_err  C_unet_topo_dg → D_lite_topo_dg: +16.0742 [±9.3356] CI excludes 0 (worsens)
- unseen      auc         C_unet_topo_dg → D_lite_topo_dg: -0.0038 [±0.0003] CI excludes 0 (worsens)
- gap_unseen  dice        C_unet_topo_dg → D_lite_topo_dg: +0.0007 [±0.0048] CI includes 0
- gap_unseen  cldice      C_unet_topo_dg → D_lite_topo_dg: +0.0035 [±0.0039] CI includes 0
- gap_unseen  auc         C_unet_topo_dg → D_lite_topo_dg: +0.0013 [±0.0006] CI excludes 0 (worsens)

## E (dropout trained in, deterministic eval) vs D

- drive       dice        D_lite_topo_dg → E_tglitenet: -0.0001 [±0.0021] CI includes 0
- drive       cldice      D_lite_topo_dg → E_tglitenet: +0.0006 [±0.0027] CI includes 0
- drive       betti0_err  D_lite_topo_dg → E_tglitenet: -1.7167 [±14.2039] CI includes 0
- drive       auc         D_lite_topo_dg → E_tglitenet: +0.0001 [±0.0004] CI includes 0
- unseen      dice        D_lite_topo_dg → E_tglitenet: -0.0012 [±0.0081] CI includes 0
- unseen      cldice      D_lite_topo_dg → E_tglitenet: +0.0001 [±0.0031] CI includes 0
- unseen      betti0_err  D_lite_topo_dg → E_tglitenet: -1.3558 [±9.2964] CI includes 0
- unseen      auc         D_lite_topo_dg → E_tglitenet: -0.0002 [±0.0004] CI includes 0
- gap_unseen  dice        D_lite_topo_dg → E_tglitenet: +0.0011 [±0.0068] CI includes 0
- gap_unseen  cldice      D_lite_topo_dg → E_tglitenet: +0.0005 [±0.0041] CI includes 0
- gap_unseen  auc         D_lite_topo_dg → E_tglitenet: +0.0003 [±0.0005] CI includes 0

## MC averaging effect: E det → E MC

- drive       dice        E_tglitenet → E_tglitenet[mc]: +0.0002 [±0.0011] CI includes 0
- drive       cldice      E_tglitenet → E_tglitenet[mc]: -0.0003 [±0.0014] CI includes 0
- drive       betti0_err  E_tglitenet → E_tglitenet[mc]: +0.5000 [±13.4878] CI includes 0
- drive       auc         E_tglitenet → E_tglitenet[mc]: +0.0001 [±0.0001] CI excludes 0 (improves)
- unseen      dice        E_tglitenet → E_tglitenet[mc]: +0.0007 [±0.0087] CI includes 0
- unseen      cldice      E_tglitenet → E_tglitenet[mc]: -0.0000 [±0.0031] CI includes 0
- unseen      betti0_err  E_tglitenet → E_tglitenet[mc]: +0.7079 [±8.4433] CI includes 0
- unseen      auc         E_tglitenet → E_tglitenet[mc]: +0.0003 [±0.0004] CI includes 0
- gap_unseen  dice        E_tglitenet → E_tglitenet[mc]: -0.0005 [±0.0076] CI includes 0
- gap_unseen  cldice      E_tglitenet → E_tglitenet[mc]: -0.0002 [±0.0040] CI includes 0
- gap_unseen  auc         E_tglitenet → E_tglitenet[mc]: -0.0001 [±0.0005] CI includes 0

## Overall: A → E (MC)

- drive       dice        A_unet → E_tglitenet[mc]: -0.0082 [±0.0030] CI excludes 0 (worsens)
- drive       cldice      A_unet → E_tglitenet[mc]: +0.0087 [±0.0062] CI excludes 0 (improves)
- drive       betti0_err  A_unet → E_tglitenet[mc]: -9.4000 [±13.3964] CI includes 0
- drive       auc         A_unet → E_tglitenet[mc]: -0.0019 [±0.0012] CI excludes 0 (worsens)
- unseen      dice        A_unet → E_tglitenet[mc]: -0.0028 [±0.0102] CI includes 0
- unseen      cldice      A_unet → E_tglitenet[mc]: -0.0033 [±0.0028] CI excludes 0 (worsens)
- unseen      betti0_err  A_unet → E_tglitenet[mc]: -21.9439 [±13.9947] CI excludes 0 (improves)
- unseen      auc         A_unet → E_tglitenet[mc]: -0.0026 [±0.0017] CI excludes 0 (worsens)
- gap_unseen  dice        A_unet → E_tglitenet[mc]: -0.0054 [±0.0079] CI includes 0
- gap_unseen  cldice      A_unet → E_tglitenet[mc]: +0.0119 [±0.0082] CI excludes 0 (worsens)
- gap_unseen  auc         A_unet → E_tglitenet[mc]: +0.0006 [±0.0021] CI includes 0

## Cost

- parameters: U-Net 31.04 M vs TG-LiteNet 0.212 M  (×147 fewer)
- FLOPs @512²: 384.7 vs 5.53 GFLOPs (×70)
- CPU latency (4 threads): 2915 ms vs 429 ms; MC dropout multiplies the latter by T=20

## RQ4 uncertainty (stage E, MC mode)

- drive      rho_px(var)=0.312  rho_px(ent)=0.314  rho_img(var)=0.175  AURC(var)=0.1777 (random 0.1781, oracle 0.1663)  pixel err@90% cov: 0.0162 vs 0.0473 at 100%
- stare      rho_px(var)=0.318  rho_px(ent)=0.326  rho_img(var)=0.143  AURC(var)=0.1948 (random 0.1985, oracle 0.1759)  pixel err@90% cov: 0.0182 vs 0.0473 at 100%
- chase_db1  rho_px(var)=0.340  rho_px(ent)=0.343  rho_img(var)=0.466  AURC(var)=0.2263 (random 0.2369, oracle 0.2098)  pixel err@90% cov: 0.0252 vs 0.0578 at 100%
- hrf        rho_px(var)=0.341  rho_px(ent)=0.353  rho_img(var)=-0.195  AURC(var)=0.3078 (random 0.2971, oracle 0.2516)  pixel err@90% cov: 0.0354 vs 0.0653 at 100%
