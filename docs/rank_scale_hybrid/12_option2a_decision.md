# Option 2A Decision

## Experiment Config

- Preregistration: `docs/rank_scale_hybrid/11_option2a_preregistration.md`
- Model: one-step residual persistence location correction
- Candidate A: Gaussian, MLE (c, k_new) per window
- Candidate B: Student-t nu=10, MLE (c, k_new) per window

## Candidate A Results

- Window_2017: c=0.9483, k=0.9899, mean_v=-0.0111, std_v=0.9999, cal@0.25=0.0962, cal@0.75=0.1346
- Window_2018: c=0.9346, k=1.1633, mean_v=0.0023, std_v=1.0000, cal@0.25=0.0962, cal@0.75=0.1346
- Window_2020: c=0.8083, k=1.6236, mean_v=0.0591, std_v=0.9983, cal@0.25=0.0900, cal@0.75=0.0100

## Candidate B Results

- Window_2017: c=0.9350, k=0.9109, mean_v=0.0150, std_v=1.0873, cal@0.25=0.0962, cal@0.75=0.0962
- Window_2018: c=0.9192, k=0.9784, mean_v=-0.0288, std_v=1.1893, cal@0.25=0.0577, cal@0.75=0.0962
- Window_2020: c=0.7727, k=1.5777, mean_v=0.0877, std_v=1.0269, cal@0.25=0.0900, cal@0.75=0.0300

## Protocol Decision

- Overall: `FAIL`
- PASS_A: `False`
- PASS_B: `False`

## Conclusion

FAIL: joint location-scale correction does not adequately absorb location bias under the preregistered criteria. Candidate A failed conditions: cal_025_breach:Window_2017:0.0962; cal_075_breach:Window_2017:0.1346; corr_next_not_positive:Window_2018:-0.1759962467102941; cal_025_breach:Window_2018:0.0962; cal_075_breach:Window_2018:0.1346; corr_next_not_positive:Window_2020:-0.2785473355432425; rank_next_not_positive:Window_2020:-0.24086956521739128; cal_025_breach:Window_2020:0.0900. Candidate B failed conditions: cal_025_breach:Window_2017:0.0962; cal_075_breach:Window_2017:0.0962; corr_next_not_positive:Window_2018:-0.15864707728818453; cal_075_breach:Window_2018:0.0962; corr_next_not_positive:Window_2020:-0.2870518364847856; rank_next_not_positive:Window_2020:-0.26347826086956516; cal_025_breach:Window_2020:0.0900. Research line terminated.

## Boundary

This decision does not modify production code, SRD §18 constants, or the law layer.
It does not reopen trigger audit, crisis architecture, hard switch, override, or MoE.
