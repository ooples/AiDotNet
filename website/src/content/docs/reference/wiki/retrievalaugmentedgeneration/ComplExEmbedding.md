---
title: "ComplExEmbedding<T>"
description: "ComplEx embedding model: uses complex-valued embeddings with Hermitian dot product scoring."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

ComplEx embedding model: uses complex-valued embeddings with Hermitian dot product scoring.

## For Beginners

ComplEx is particularly good at modeling:

- Symmetric relations: "married_to" (score(A,r,B) = score(B,r,A))
- Antisymmetric relations: "parent_of" (score(A,r,B) ≠ score(B,r,A))

It achieves this by using complex numbers, which naturally distinguish direction.

## How It Works

ComplEx (Trouillon et al., 2016) represents entities and relations as complex vectors.
Score: Re(⟨h, r, conj(t)⟩) = Σ(hRe·rRe·tRe + hRe·rImag·tImag + hImag·rRe·tImag - hImag·rImag·tRe).
Training uses logistic loss with optional N3 regularization.
Higher scores indicate more plausible triples.

Complex numbers are represented as paired real/imaginary T[] arrays (length 2*dim each)
to maintain generic T type parameter compatibility.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDistanceBased` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `StableSigmoid(Double)` | Numerically stable sigmoid: 1 / (1 + exp(-x)). |
| `StableSoftplus(Double)` | Numerically stable softplus: log(1 + exp(x)). |

