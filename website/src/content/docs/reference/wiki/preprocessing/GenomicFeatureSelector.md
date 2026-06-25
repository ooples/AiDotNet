---
title: "GenomicFeatureSelector<T>"
description: "Feature selection for genomic/bioinformatics data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.DomainSpecific`

Feature selection for genomic/bioinformatics data.

## For Beginners

Genomic data often has millions of features (genes)
but only a few hundred samples (patients). This selector uses techniques
specifically designed for this challenging scenario, finding the few genes
that truly matter for distinguishing between conditions.

## How It Works

GenomicFeatureSelector is designed for high-dimensional genomic data such as
gene expression profiles, SNPs, or methylation data. It addresses the "large p,
small n" problem common in genomics and can incorporate gene grouping.

