---
title: "PurgedWalkForwardValidator"
description: "López de Prado purged-and-embargoed walk-forward cross-validation for time-ordered financial data."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Evaluation`

López de Prado purged-and-embargoed walk-forward cross-validation for time-ordered financial data.
Generates rolling-origin train/test splits while removing training samples whose label horizon
overlaps the test window (purge) and a buffer of samples immediately after each test fold (embargo).

## For Beginners

When you test a trading model, you must never let it train on data that
secretly contains the answers to the test. Because each label here looks forward in time (e.g. "the
return over the next 5 days"), a training point sitting just before the test period overlaps with it
and would leak the answer. "Purging" deletes those contaminated training points; the "embargo" deletes
a few extra points right after the test window for good measure. The result is a list of clean
(train, test) splits that walk forward through time the way real trading does.

## How It Works

Standard k-fold CV leaks information when labels are computed over a forward horizon (e.g. the label
for sample `i` depends on returns up to `i + h`): a training sample near a test fold can
"see the future" through its overlapping label, inflating measured performance. Purging drops those
overlapping training samples; the embargo additionally drops a gap right after the test fold to
neutralize serial correlation that survives purging. Walk-forward (rolling-origin) ordering keeps the
test fold strictly after the training data, mirroring live deployment.

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Int32,Int32,Int32,Int32,Boolean)` | Builds the purged + embargoed walk-forward splits. |

