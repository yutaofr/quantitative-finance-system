# Cycle Evaluation Protocol

Status: FROZEN FOR THIS AUDIT.

## 1. Task Objects

Task 1 is `cycle_score_t`, a continuous weekly score. High score means expansion. Low score means contraction.

Task 2 is `cycle_state_t`, a discrete weekly state with exactly three preregistered states: `EXPANSION`, `SLOWDOWN`, and `CONTRACTION`. Scores are mapped by terciles, with the lowest tercile labeled `CONTRACTION`, the middle tercile labeled `SLOWDOWN`, and the highest tercile labeled `EXPANSION`.

## 2. Labels

Primary endogenous forward market label: 52-week forward log return, 13-week forward realized weekly volatility, and 13-week forward max drawdown from local `NASDAQXNDX` prices. The frozen composite is `z(forward_return) - 0.5*z(forward_risk) - 0.5*z(forward_drawdown)`.

External sanity-check label: preregistered crisis windows `2000-03-24..2002-10-11`, `2008-09-12..2009-03-13`, and `2020-02-14..2020-03-27`. These windows are used only for lead-time and sanity overlay. They are never used for training.

## 3. Metrics

Directionality: Pearson and rank correlation of `cycle_score_t` with future return, future risk, future drawdown, and the composite label.

Classification: balanced accuracy, macro-F1, confusion matrix, and per-state precision/recall for `cycle_state_t`.

Lead time: first contraction signal inside the 26-week pre-crisis window, transition/crisis lead count, and false-alarm lead frequency outside crisis plus lead windows.

Stability: state persistence, transition churn rate, one-step `EXPANSION` to `CONTRACTION` flip frequency, score autocorrelation, and score mean absolute change.

Cross-window consistency: all metrics are reported for 2017, 2018H2, 2020H1, 2008, and 2000-2002 where data exists. The protocol also reports window dispersion and worst-window composite correlation.

## 4. Success Standards

Layer 1 usability gate: finite output, at least 3 aligned observations, no global direction undefined, and transition churn not above 0.65.

Layer 2 decision gate: positive return direction, negative risk direction, positive rank-return direction, negative rank-risk direction, balanced accuracy above constant-chance 1/3, at least one positive crisis lead, and false-alarm frequency not above 0.50.

Layer 3 cross-window survival gate: worst-window composite correlation must be positive and cross-window dispersion must not exceed 0.50.

Object result labels are `PASS`, `PARTIAL`, or `FAIL` by layer. The system-level decision is one of `CURRENT_SYSTEM_HAS_NO_CYCLE_CAPABILITY`, `CURRENT_SYSTEM_HAS_LOCAL_BUT_NOT_GENERAL_CYCLE_CAPABILITY`, or `CURRENT_SYSTEM_HAS_AUDITABLE_CYCLE_CAPABILITY`.

## 5. Evidence Boundaries

通过方向性 ≠ 通过周期判定

通过危机窗口 ≠ 可泛化到所有周期

通过单一标签体系 ≠ 真正识别经济周期
