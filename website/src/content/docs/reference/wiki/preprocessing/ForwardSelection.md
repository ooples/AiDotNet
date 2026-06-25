---
title: "ForwardSelection<T>"
description: "Forward Selection (Sequential Forward Selection) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Forward Selection (Sequential Forward Selection) for feature selection.

## For Beginners

Forward Selection is like building a team one person
at a time. At each step, you try adding each remaining candidate and keep the one
who helps the most. You continue until your team is the desired size. Simple but
can miss feature interactions that only appear together.

## How It Works

Starts with an empty set and greedily adds one feature at a time that provides
the best improvement until the desired number of features is reached.

