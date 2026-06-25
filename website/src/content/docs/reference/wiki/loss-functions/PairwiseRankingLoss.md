---
title: "PairwiseRankingLoss"
description: "Implements the pairwise RankNet learning-to-rank loss with an optional tail-weighting knob."
section: "Reference"
---

_Loss Functions_

Implements the pairwise RankNet learning-to-rank loss with an optional tail-weighting knob.

## For Beginners

Most loss functions ask "how close is each predicted number to the
true number?". A ranking loss asks a different, often more useful, question: "did you put the
items in the right *order*?". If you only care about buying the best things and selling
the worst things, the exact predicted values do not matter — only their order does.
This loss looks at every pair of items, checks whether the one with the higher true value also
got the higher predicted score, and nudges the model when it got a pair backwards.
The tail-weighting option lets you tell the model "I care much more about getting the
top and bottom right than the middle", which is exactly what a long/short trader wants.

## How It Works

This loss treats every prediction vector and target vector as a single *ranking group*
(for example, all stocks in one asset class on one date). It does not learn to match each
target value pointwise; instead it learns the correct **relative order** of the items.
For every ordered pair (i, j) where the true score of i is greater than the true score of j,
the RankNet loss penalizes the model when it predicts s_i ≤ s_j:

loss(i, j) = log(1 + exp(-(s_i - s_j)))

where s_i and s_j are the predicted scores. Summed over all such pairs and averaged by the
number of pairs, this yields a smooth, convex, gradient-friendly surrogate for "fraction of
pairs ordered incorrectly". With the default tail weight (1.0) it is the standard RankNet
loss of Burges et al. (2005).

**Tail weighting.** In cross-sectional trading only the extremes are actionable: you go
long the top names and short the bottom names, and the middle of the ranking is never traded.
The optional `tailWeightPower` knob makes each pair contribute in proportion to how
extreme its two items are in the target distribution. Each item is assigned an extremity in
[0, 1] measuring its distance from the median target (0 at the median, 1 at the most extreme
top or bottom name). A pair's weight is

w(i, j) = (1 + max(extremity_i, extremity_j))^tailWeightPower

With `tailWeightPower = 0` every weight is 1 and you recover plain RankNet (backward
compatible). With a positive power the biggest movers dominate the loss, so the model spends
its capacity getting the tradeable tails right.

**How it plugs in.** This is a standard `ILossFunction`, so any
gradient-trained model in AiDotNet can use it. For a neural-network cross-sectional ranker:

Each training example's feature matrix is one cross-section (one date / one segment) and the
target vector is the signed forward returns; the network outputs one score per name and the
loss ranks them.

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new PairwiseRankingLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"PairwiseRankingLoss = {value:F4}");
```

