---
title: "AlphaInvesting<T>"
description: "Alpha-Investing for online feature selection with FDR control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Online`

Alpha-Investing for online feature selection with FDR control.

## For Beginners

Think of it like managing a budget for discoveries.
You start with some "discovery wealth." Each time you test a feature, you spend
some wealth. When you find a significant feature, you earn wealth back. This
ensures you don't make too many false discoveries while still finding real ones.

## How It Works

Alpha-Investing is an online algorithm for controlling the false discovery rate
when testing features sequentially. It maintains a "wealth" that increases when
null hypotheses are rejected and decreases when tests are performed.

