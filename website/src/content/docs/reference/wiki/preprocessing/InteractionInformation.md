---
title: "InteractionInformation<T>"
description: "Interaction Information for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Interaction Information for feature selection.

## For Beginners

This method finds features that work together in
unexpected ways. Sometimes two features alone are useless, but together they're
very predictive (synergy). Other times, features duplicate each other's information
(redundancy). Interaction Information captures both.

## How It Works

Interaction Information measures the amount of information bound up in a set of
variables that is not present in any subset. It can be positive (synergy) or
negative (redundancy), revealing higher-order dependencies.

