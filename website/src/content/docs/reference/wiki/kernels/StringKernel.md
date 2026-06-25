---
title: "StringKernel<T>"
description: "Implements various string kernels for comparing text/sequence data in Gaussian Processes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements various string kernels for comparing text/sequence data in Gaussian Processes.

## For Beginners

String kernels allow Gaussian Processes to work directly with
text or sequence data without needing to convert them to fixed-length feature vectors.

The key insight: We can define a kernel (similarity measure) between strings that
captures meaningful notions of similarity:

- Spectrum kernel: Counts shared substrings
- Subsequence kernel: Counts shared (possibly non-contiguous) subsequences
- Edit distance kernel: Based on how many edits to transform one string to another

Applications:

- Text classification (spam detection, sentiment analysis)
- Bioinformatics (DNA/protein sequence comparison)
- Natural language processing
- Any domain with sequential/string data

## How It Works

**Note:** This class does NOT implement IKernelFunction<T> by design. Unlike numeric
kernels that operate on Vector<T> feature vectors, string kernels operate directly on
text data. To use string kernels with standard GP models, use this class to compute a
kernel matrix from your text data, then use that matrix with a custom kernel implementation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StringKernel(StringKernel<>.KernelType,Int32,Double,Double)` | Initializes a new string kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Type` | Gets the kernel type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(String,String)` | Calculates the string kernel value between two strings. |
| `CalculateBagOfWordsKernel(String,String)` | Calculates the bag of words kernel. |
| `CalculateEditDistanceKernel(String,String)` | Calculates the edit distance (Levenshtein) kernel. |
| `CalculateSpectrumKernel(String,String)` | Calculates the spectrum (k-mer) kernel. |
| `CalculateSubsequenceKernel(String,String)` | Calculates the subsequence kernel with gap penalty. |
| `CalculateSubsequenceKernelSelf(String)` | Calculates subsequence kernel of a string with itself (for normalization). |
| `ComputeEditDistance(String,String)` | Computes the Levenshtein edit distance between two strings. |
| `ComputeKernelMatrix(String[])` | Creates a kernel matrix for a collection of strings. |
| `GetKmerCounts(String)` | Gets k-mer counts for a string. |
| `GetWordCounts(String)` | Gets word counts for a string. |

