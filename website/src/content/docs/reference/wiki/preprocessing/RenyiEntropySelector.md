---
title: "RenyiEntropySelector<T>"
description: "Renyi Entropy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Renyi Entropy based Feature Selection.

## For Beginners

Renyi entropy is a family of entropy measures. With
alpha=1, it equals Shannon entropy. Different alpha values emphasize different
aspects of the distribution - low alpha focuses on rare events, high alpha on
common events.

## How It Works

Selects features based on their Renyi entropy, which generalizes Shannon entropy
with a tunable parameter alpha.

