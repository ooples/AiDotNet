---
title: "FilterWrapper<T>"
description: "Filter-Wrapper hybrid feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Filter-Wrapper hybrid feature selection.

## For Beginners

Think of it as a two-stage interview process.
First, resumes are quickly screened (filter) to remove obviously unqualified
candidates. Then, the remaining candidates go through detailed interviews
(wrapper) to select the best ones. This is faster than interviewing everyone.

## How It Works

Filter-Wrapper combines the speed of filter methods with the accuracy of
wrapper methods. It first uses a filter to reduce the feature space, then
applies a wrapper method on the reduced set for fine-tuning.

