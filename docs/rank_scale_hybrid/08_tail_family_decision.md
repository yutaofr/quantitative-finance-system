# Tail Family Decision

## Experiment Config

- Preregistration: `docs/rank_scale_hybrid/07_tail_family_preregistration.md`
- k_fixed: `2.682572`
- Tail family: variance-one Student-t, MLE nu per window and pooled
- Input: T5 z_t / k_fixed = u_t for windows 2017, 2018, 2020

## Nu Estimates

- Window_2017: nu = 100.0000
- Window_2018: nu = 100.0000
- Window_2020: nu = 45.7288

- nu_drift: `54.2712` (threshold: 30.0)
- NU_DRIFT_OK: `False`

## Protocol Decision

- PASS_PER_WINDOW: `False`
- PASS_POOLED: `False`
- Failed conditions: `CALIBRATION_BREACH:Window_2017:tau_0.10:0.1000, CALIBRATION_BREACH:Window_2017:tau_0.25:0.1731, CALIBRATION_BREACH:Window_2017:tau_0.75:0.3654, CALIBRATION_BREACH:Window_2017:tau_0.90:0.0538, CALIBRATION_BREACH:Window_2018:tau_0.25:0.4423, CALIBRATION_BREACH:Window_2018:tau_0.75:0.1731, CALIBRATION_BREACH:Window_2020:tau_0.10:0.0600, CALIBRATION_BREACH:Window_2020:tau_0.25:0.1300, CALIBRATION_BREACH:Window_2020:tau_0.75:0.2300, KS_REJECT:Window_2017:p=0.0001, KS_REJECT:Window_2018:p=0.0000, KS_REJECT:Window_2020:p=0.0020`

## Conclusion

FAIL: Student-t with global k_fixed does not provide adequate quantile calibration under the preregistered rules. Failed conditions: CALIBRATION_BREACH:Window_2017:tau_0.10:0.1000; CALIBRATION_BREACH:Window_2017:tau_0.25:0.1731; CALIBRATION_BREACH:Window_2017:tau_0.75:0.3654; CALIBRATION_BREACH:Window_2017:tau_0.90:0.0538; CALIBRATION_BREACH:Window_2018:tau_0.25:0.4423; CALIBRATION_BREACH:Window_2018:tau_0.75:0.1731; CALIBRATION_BREACH:Window_2020:tau_0.10:0.0600; CALIBRATION_BREACH:Window_2020:tau_0.25:0.1300; CALIBRATION_BREACH:Window_2020:tau_0.75:0.2300; KS_REJECT:Window_2017:p=0.0001; KS_REJECT:Window_2018:p=0.0000; KS_REJECT:Window_2020:p=0.0020.

## Boundary

This decision does not modify production code, SRD §18 constants, or the law layer.
It does not reopen trigger audit, crisis architecture, hard switch, override, or MoE.
