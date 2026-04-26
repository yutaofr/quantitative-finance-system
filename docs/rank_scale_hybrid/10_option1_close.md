# Option 1 Close

`fixed T5 + downstream k + tail family` has been rejected by quantile calibration: per-window
calibration errors at τ ∈ {0.10, 0.25, 0.75, 0.90} exceed the preregistered 0.05 threshold
across all three windows (2017, 2018, 2020), with KS rejection in all windows.

The primary failure mode is **location bias**: z_t has regime-dependent conditional mean
(positive in bull windows, negative in bear/crisis windows), which a scalar scale correction k
cannot absorb; this is confirmed by the mirror-symmetric tau_0.25 / tau_0.75 errors between
the 2017 (right-shifted) and 2018 (left-shifted) windows.

This line is closed. No further experimentation may use Option 1 (fixed T5 + downstream k)
as a foundation for shape-layer extension; the successor is Option 2A (joint location-scale
falsification, preregistered in `11_option2a_preregistration.md`).
