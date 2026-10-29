# Auto-generated findings (facts only)

Deltas are (second − first) in mean over seeds; brackets are 95 % paired-t intervals over shared seeds.


## RQ1 topology: A → B (adds soft-clDice + skeleton head)

- drive       dice        A_unet → B_unet_topo: -0.0033 [±0.0019] CI excludes 0 (worsens)
- drive       cldice      A_unet → B_unet_topo: +0.0151 [±0.0037] CI excludes 0 (improves)
- drive       betti0_err  A_unet → B_unet_topo: -19.7900 [±4.0945] CI excludes 0 (improves)
- drive       auc         A_unet → B_unet_topo: -0.0001 [±0.0005] CI includes 0
- unseen      dice        A_unet → B_unet_topo: -0.0092 [±0.0035] CI excludes 0 (worsens)
- unseen      cldice      A_unet → B_unet_topo: +0.0071 [±0.0043] CI excludes 0 (improves)
- unseen      betti0_err  A_unet → B_unet_topo: -43.2993 [±16.0324] CI excludes 0 (improves)
- unseen      auc         A_unet → B_unet_topo: -0.0010 [±0.0009] CI excludes 0 (worsens)
- gap_unseen  dice        A_unet → B_unet_topo: +0.0060 [±0.0036] CI excludes 0 (worsens)
- gap_unseen  cldice      A_unet → B_unet_topo: +0.0080 [±0.0019] CI excludes 0 (worsens)
- gap_unseen  auc         A_unet → B_unet_topo: +0.0010 [±0.0008] CI excludes 0 (worsens)

## RQ2 generalisation: B → C (adds DG augmentation + consistency)

- drive       dice        B_unet_topo → C_unet_topo_dg: -0.0008 [±0.0016] CI includes 0
- drive       cldice      B_unet_topo → C_unet_topo_dg: +0.0037 [±0.0019] CI excludes 0 (improves)
- drive       betti0_err  B_unet_topo → C_unet_topo_dg: -4.4200 [±4.1425] CI excludes 0 (improves)
- drive       auc         B_unet_topo → C_unet_topo_dg: +0.0003 [±0.0003] CI excludes 0 (improves)
- unseen      dice        B_unet_topo → C_unet_topo_dg: +0.0083 [±0.0045] CI excludes 0 (improves)
- unseen      cldice      B_unet_topo → C_unet_topo_dg: +0.0041 [±0.0041] CI includes 0
- unseen      betti0_err  B_unet_topo → C_unet_topo_dg: -3.7633 [±3.3938] CI excludes 0 (improves)
- unseen      auc         B_unet_topo → C_unet_topo_dg: +0.0019 [±0.0016] CI excludes 0 (improves)
- gap_unseen  dice        B_unet_topo → C_unet_topo_dg: -0.0091 [±0.0041] CI excludes 0 (improves)
- gap_unseen  cldice      B_unet_topo → C_unet_topo_dg: -0.0005 [±0.0033] CI includes 0
- gap_unseen  auc         B_unet_topo → C_unet_topo_dg: -0.0016 [±0.0017] CI includes 0

## RQ3 trade-off: C → D (compact backbone)

- drive       dice        C_unet_topo_dg → D_lite_topo_dg: -0.0042 [±0.0019] CI excludes 0 (worsens)
- drive       cldice      C_unet_topo_dg → D_lite_topo_dg: -0.0110 [±0.0041] CI excludes 0 (worsens)
- drive       betti0_err  C_unet_topo_dg → D_lite_topo_dg: +14.6400 [±11.9629] CI excludes 0 (worsens)
- drive       auc         C_unet_topo_dg → D_lite_topo_dg: -0.0027 [±0.0009] CI excludes 0 (worsens)
- unseen      dice        C_unet_topo_dg → D_lite_topo_dg: -0.0049 [±0.0026] CI excludes 0 (worsens)
- unseen      cldice      C_unet_topo_dg → D_lite_topo_dg: -0.0143 [±0.0037] CI excludes 0 (worsens)
- unseen      betti0_err  C_unet_topo_dg → D_lite_topo_dg: +16.7789 [±10.3269] CI excludes 0 (worsens)
- unseen      auc         C_unet_topo_dg → D_lite_topo_dg: -0.0043 [±0.0010] CI excludes 0 (worsens)
- gap_unseen  dice        C_unet_topo_dg → D_lite_topo_dg: +0.0006 [±0.0029] CI includes 0
- gap_unseen  cldice      C_unet_topo_dg → D_lite_topo_dg: +0.0034 [±0.0038] CI includes 0
- gap_unseen  auc         C_unet_topo_dg → D_lite_topo_dg: +0.0016 [±0.0005] CI excludes 0 (worsens)

## E (dropout trained in, deterministic eval) vs D

- drive       dice        D_lite_topo_dg → E_tglitenet: -0.0002 [±0.0005] CI includes 0
- drive       cldice      D_lite_topo_dg → E_tglitenet: +0.0005 [±0.0017] CI includes 0
- drive       betti0_err  D_lite_topo_dg → E_tglitenet: -2.7500 [±7.5372] CI includes 0
- drive       auc         D_lite_topo_dg → E_tglitenet: +0.0001 [±0.0002] CI includes 0
- unseen      dice        D_lite_topo_dg → E_tglitenet: -0.0013 [±0.0025] CI includes 0
- unseen      cldice      D_lite_topo_dg → E_tglitenet: +0.0004 [±0.0033] CI includes 0
- unseen      betti0_err  D_lite_topo_dg → E_tglitenet: -2.7613 [±8.3201] CI includes 0
- unseen      auc         D_lite_topo_dg → E_tglitenet: -0.0000 [±0.0007] CI includes 0
- gap_unseen  dice        D_lite_topo_dg → E_tglitenet: +0.0011 [±0.0027] CI includes 0
- gap_unseen  cldice      D_lite_topo_dg → E_tglitenet: +0.0001 [±0.0019] CI includes 0
- gap_unseen  auc         D_lite_topo_dg → E_tglitenet: +0.0001 [±0.0006] CI includes 0

## MC averaging effect: E det → E MC

- drive       dice        E_tglitenet → E_tglitenet[mc]: +0.0001 [±0.0002] CI includes 0
- drive       cldice      E_tglitenet → E_tglitenet[mc]: -0.0001 [±0.0004] CI includes 0
- drive       betti0_err  E_tglitenet → E_tglitenet[mc]: +0.1900 [±1.1307] CI includes 0
- drive       auc         E_tglitenet → E_tglitenet[mc]: +0.0001 [±0.0001] CI excludes 0 (improves)
- unseen      dice        E_tglitenet → E_tglitenet[mc]: +0.0006 [±0.0005] CI excludes 0 (improves)
- unseen      cldice      E_tglitenet → E_tglitenet[mc]: -0.0000 [±0.0003] CI includes 0
- unseen      betti0_err  E_tglitenet → E_tglitenet[mc]: +0.7121 [±0.3012] CI excludes 0 (worsens)
- unseen      auc         E_tglitenet → E_tglitenet[mc]: +0.0003 [±0.0000] CI excludes 0 (improves)
- gap_unseen  dice        E_tglitenet → E_tglitenet[mc]: -0.0006 [±0.0005] CI excludes 0 (improves)
- gap_unseen  cldice      E_tglitenet → E_tglitenet[mc]: -0.0001 [±0.0006] CI includes 0
- gap_unseen  auc         E_tglitenet → E_tglitenet[mc]: -0.0001 [±0.0001] CI excludes 0 (improves)

## Overall: A → E (MC)

- drive       dice        A_unet → E_tglitenet[mc]: -0.0084 [±0.0019] CI excludes 0 (worsens)
- drive       cldice      A_unet → E_tglitenet[mc]: +0.0082 [±0.0049] CI excludes 0 (improves)
- drive       betti0_err  A_unet → E_tglitenet[mc]: -12.1300 [±6.5557] CI excludes 0 (improves)
- drive       auc         A_unet → E_tglitenet[mc]: -0.0022 [±0.0014] CI excludes 0 (worsens)
- unseen      dice        A_unet → E_tglitenet[mc]: -0.0065 [±0.0072] CI includes 0
- unseen      cldice      A_unet → E_tglitenet[mc]: -0.0027 [±0.0027] CI includes 0
- unseen      betti0_err  A_unet → E_tglitenet[mc]: -32.3329 [±19.8212] CI excludes 0 (improves)
- unseen      auc         A_unet → E_tglitenet[mc]: -0.0031 [±0.0014] CI excludes 0 (worsens)
- gap_unseen  dice        A_unet → E_tglitenet[mc]: -0.0020 [±0.0078] CI includes 0
- gap_unseen  cldice      A_unet → E_tglitenet[mc]: +0.0109 [±0.0046] CI excludes 0 (worsens)
- gap_unseen  auc         A_unet → E_tglitenet[mc]: +0.0009 [±0.0010] CI includes 0

## Cost

- parameters: U-Net 31.04 M vs TG-LiteNet 0.212 M  (×147 fewer)
- FLOPs @512²: 384.7 vs 5.53 GFLOPs (×70)
- CPU latency (4 threads): 2915 ms vs 429 ms; MC dropout multiplies the latter by T=20

## RQ4 uncertainty (stage E, MC mode)

- drive      rho_px(var)=0.311  rho_px(ent)=0.313  rho_img(var)=0.151  AURC(var)=0.1783 (random 0.1784, oracle 0.1665)  pixel err@90% cov: 0.0162 vs 0.0472 at 100%
- stare      rho_px(var)=0.318  rho_px(ent)=0.325  rho_img(var)=0.085  AURC(var)=0.1971 (random 0.1981, oracle 0.1754)  pixel err@90% cov: 0.0178 vs 0.0471 at 100%
- chase_db1  rho_px(var)=0.341  rho_px(ent)=0.343  rho_img(var)=0.477  AURC(var)=0.2268 (random 0.2373, oracle 0.2103)  pixel err@90% cov: 0.0250 vs 0.0578 at 100%
- hrf        rho_px(var)=0.340  rho_px(ent)=0.352  rho_img(var)=-0.172  AURC(var)=0.3054 (random 0.2968, oracle 0.2516)  pixel err@90% cov: 0.0349 vs 0.0650 at 100%
