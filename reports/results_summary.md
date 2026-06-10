# Macro-Factor Sector Ranking Model: Results Summary

## Research Question

Can a small set of macroeconomic indicators and leakage-safe sector momentum features help rank U.S. sector ETFs for next-month relative performance?

## Data Scopes

The project evaluates two scopes. The 11-sector recent-regime scope includes all 11 U.S. sector ETFs but begins later because XLC and XLRE have shorter histories. The 9-sector long-history scope excludes XLC and XLRE to test robustness over a longer sample.

## Modeling Approach

The core model is an interpretable walk-forward Ridge Regression setup. Each sector gets a separate model, and each month sectors are ranked by predicted next-month returns. The project also compares macro-only, momentum-only, combined macro-plus-momentum, simple relative momentum, and equal-weight benchmark variants.

## Walk-Forward and Leakage Controls

Models use expanding-window training with no random train-test split. Training rows are strictly earlier than each prediction month. Macro features are lagged where appropriate, and sector momentum and volatility features use trailing completed monthly returns only. Date metadata, targets, and ranks are excluded from model inputs.

## Recent 11-Sector Result

In the recent 11-sector scope, the combined macro-plus-momentum model performed better than the macro-only and equal-weight benchmarks over the tested out-of-sample period. This result is promising but covers a short, recent market regime.

## Long-History 9-Sector Result

In the longer 9-sector scope, the combined model was not consistently better than equal weight. This weakens the evidence that the enhanced signal is broadly persistent across regimes.

## Transaction-Cost Conclusion

Transaction costs do not materially change the favorable recent-regime 11-sector conclusion for the combined model, but they reinforce the weaker long-history conclusion because active strategies already trail equal weight before costs in that scope.

## Limitations

This is not evidence of a deployable trading strategy. The results are regime-dependent, use simplified monthly rebalancing assumptions, and rely on standard FRED downloads that may include revised historical values. A future improvement would use ALFRED vintage data for stricter point-in-time macro feature construction.

## Overall Interpretation

The project shows that sector-specific momentum and volatility can improve a macro-only model in the recent 11-sector sample, but the long-history evidence is mixed to weak. The strongest conclusion is methodological: controlled walk-forward testing and same-row ablations are essential before interpreting apparent sector-ranking performance.
