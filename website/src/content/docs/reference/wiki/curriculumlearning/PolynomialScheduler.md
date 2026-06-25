---
title: "PolynomialScheduler<T>"
description: "Curriculum scheduler with polynomial progression curve."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Curriculum scheduler with polynomial progression curve.

## For Beginners

This scheduler uses a polynomial curve to control
progression speed. The power parameter determines the curve shape.

## How It Works

**Power Parameter Effects:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolynomialScheduler(Int32,Double,,)` | Initializes a new instance of the `PolynomialScheduler` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDataFraction` | Gets the current data fraction using polynomial curve. |
| `GetStatistics` | Gets scheduler-specific statistics. |

