---
title: "AdjustedMutualInformation<T>"
description: "Adjusted Mutual Information correcting for chance agreement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Adjusted Mutual Information correcting for chance agreement.

## For Beginners

Just by chance, two random groupings will
share some information. AMI subtracts this expected chance agreement
to give a cleaner measure of true dependency.

## How It Works

Adjusted Mutual Information (AMI) corrects for the fact that random
labelings have non-zero expected MI. It accounts for the number of
clusters to give a more accurate measure of agreement.

