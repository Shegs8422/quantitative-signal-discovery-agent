# Investment Assurance Brief — Signal Discovery Fork

**Date:** 2026-11-12 | **Scope:** accepted/near-miss signals produced by the fork, built into L/S portfolios and stress-evaluated for deployability.

This document is the assurance artifact: what was tested, how, what passed, what failed, and what it cannot attest.

---

## Tooling under assessment

`portfolio_sim.py` (fork addition): dollar-neutral top/bottom L/S construction (k=2–4 per leg on 12–28 names; deciles on 381), vol targeting (10% ann., causal 63d lookback, cap 2.0x), one-way 10 bps transaction costs, 50 bps/yr borrow on the short leg, close-to-close execution, monthly/weekly/daily rebalance. Universes: `bond_etfs`, `global_bond_etfs`, `sp500`; 2012–2025.

## Signals under autopsy

### Signal 1 — Bond ETF "volatility-adjusted momentum" (accepted by upstream rule, and again under HAC config)
| Construction | Ann. return | Sharpe | Max DD | Turnover/period | Verdict |
|---|---|---|---|---|---|
| Daily rebalance | −32.7% | −3.27 | −99.6% | 104%/day (≈26%/yr cost drag) | Destroyed by costs |
| Weekly rebalance | −13.3% | −1.33 | −86.6% | 39%/week | Destroyed by costs |
| Monthly rebalance | −5.0% | −0.50 | −57.6% | 14%/month | Destroyed by costs |

Gross (pre-cost) edge ≈ breakeven-positive (Sharpe 0.41); annual years mostly negative except 2019 (+2.5%) and 2022 (+10.6%, the hiking regime); beta vs panel +0.30; residualized IC ≈ 0 (p = 0.42) → rates beta, not cross-sectional alpha.

### Signal 2 — S&P 500 "price deviation from linear-weighted mean" (mean reversion)
| Monthly | +0.98%/yr | Sharpe 0.09 | Max DD −26.5% | Positive, unreliable, ~zero information over panel |

### Signal 3 — Global bond ETF 60d risk-adjusted momentum (accepted; positive direction)
Bootstrap CI barely excludes 0; decile spread t = −0.04 (p = 0.97); Bonferroni (6 trials) fails at p = 0.042; sub-period IC decays +0.033 → +0.012; **portfolio sim −2.34%/yr, Sharpe −0.25, max DD −36%** → not investable.

## Verdict

**No accepted signal is investable as constructed, at any tested rebalance frequency, universe, or direction.** The workflow's acceptance criteria identified statistically significant cross-sectional structure; portfolio construction with real frictions removed all of it. Central truth of the engagement:

> Discovery ≠ deployability. Significance of ICs does not survive contact with costs, turnover, beta, and regime dependence.

## What the tooling guarantees (upstream does not)

- Overlap-adjusted inference (HAC) everywhere meaningful: evaluator keys, workflow gate, reports
- Costs and turnover are first-class, priced into every artifact
- Multiple-testing registry (`output/trials.jsonl`), Bonferroni-annotated
- Direction reported, never hidden by |IC|
- Full pipeline reproducible from saved code with one command

## What it cannot and does not attest

1. **Out-of-sample validity.** All evaluation is in-sample (same data generated and tested signals). Live capital requires untouched holdout or paper trading.
2. **Point-in-time universe.** Current constituents → survivorship bias; delisted streams missing.
3. **Microstructure reality.** No market impact, borrow desk, HTB regimes, halts, or full corporate-action audit beyond Yahoo adjustments.
4. **Regulatory readiness.** No compliance sign-off, fiduciary review, independent model validation, or limit framework. "Investment assurance" in the legal sense is out of reach here; only technical assurance is claimed.
5. **Capacity.** Buckets of 2–38 names at observed turnover are pseudo-liquid; impact would worsen with AUM.
6. **FX hedge costs** on unhedged country panels; local-currency attribution exists but hedge-cost modeling does not.

## Governance checklist (before any capital moves)

- [ ] Independent model validator reproduces every report/portfolio artifact from raw data
- [ ] Out-of-sample holdout not used in any research iteration (incl. this one)
- [ ] Live paper trading ≥ 3 months, cost-charged
- [ ] Capacity & borrow study with a real prime brokerage desk
- [ ] Risk committee sign-off incl. stress tests (−20% equity gap, +200bp rates)
- [ ] Ongoing trial tracking across ALL research iterations, not only reports

## Honest final word

The fork does not promise returns. It promises no alpha claim passes without HAC significance, cost accounting, direction disclosure, beta diagnostics, regime stability and a multiple-testing context. Applied to this session's discoveries, that gate says: **do not allocate.** That is assurance working as designed.
