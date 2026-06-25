---
title: "MathDataLoaderOptions"
description: "Configuration options for the Hendrycks MATH benchmark loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the Hendrycks MATH benchmark loader.

## How It Works

MATH (Hendrycks et al. 2021) — 12,500 competition math problems
(7,500 train / 5,000 test) drawn from AMC/AIME/HMMT competitions
across 7 subject areas (algebra, counting/probability, geometry,
intermediate algebra, number theory, prealgebra, precalculus).
Each problem has a difficulty level (1..5) and a step-by-step
LaTeX-formatted reference solution. Standard advanced math
reasoning benchmark.

## Properties

| Property | Summary |
|:-----|:--------|
| `LevelFilter` | Difficulty level filter (1..5). |
| `SubjectFilter` | Optional subject filter (case-insensitive substring of subject directory name). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

