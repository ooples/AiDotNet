---
title: "Quartile<T>"
description: "Computes and stores the quartiles (Q1, Q2, Q3) of a numeric dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Statistics`

Computes and stores the quartiles (Q1, Q2, Q3) of a numeric dataset.

## For Beginners

Quartiles divide your data into four equal parts, giving you a quick way to understand
the distribution of your values.

Think of quartiles like dividing a line of people by height into four equal groups:

- Q1 (First Quartile): The height where 25% of people are shorter and 75% are taller
- Q2 (Second Quartile): The height where 50% are shorter and 50% are taller (this is also called the median)
- Q3 (Third Quartile): The height where 75% are shorter and 25% are taller

Quartiles help you understand:

- Where the "middle half" of your data lies (between Q1 and Q3)
- If your data is skewed (if the distance from Q1 to Q2 is different from Q2 to Q3)
- What values might be considered outliers (typically those below Q1-1.5×IQR or above Q3+1.5×IQR)

For example, if test scores have Q1=70, Q2=80, and Q3=90, you know half the scores are between 70 and 90,
and the median score is 80.

## How It Works

The Quartile class calculates the three standard quartiles of a dataset: the first quartile (Q1, 25th percentile),
the second quartile (Q2, 50th percentile or median), and the third quartile (Q3, 75th percentile).
These quartiles divide the dataset into four equal parts and provide insights into the distribution of the data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Quartile(Vector<>)` | Initializes a new instance of the Quartile class with the provided dataset. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Q1` | Gets the first quartile (Q1, 25th percentile) of the dataset. |
| `Q2` | Gets the second quartile (Q2, 50th percentile, median) of the dataset. |
| `Q3` | Gets the third quartile (Q3, 75th percentile) of the dataset. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | The numeric operations appropriate for the generic type T. |
| `_sortedData` | The sorted vector of data values. |

