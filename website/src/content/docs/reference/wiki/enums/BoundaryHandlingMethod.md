---
title: "BoundaryHandlingMethod"
description: "Specifies how to handle boundaries when processing data that extends beyond the available range."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies how to handle boundaries when processing data that extends beyond the available range.

## For Beginners

When working with operations like convolutions, filtering, or sampling, 
you often need to access data points outside the boundaries of your dataset. 

For example, if you have an image and want to apply a filter to every pixel, what happens 
at the edges where the filter would need pixels that don't exist? Boundary handling methods 
provide different ways to solve this problem.

Think of it like trying to read a word at the edge of a page - you need to decide what to do 
when part of the word would be off the page.

## Fields

| Field | Summary |
|:-----|:--------|
| `Periodic` | Treats the data as if it repeats infinitely in all directions. |
| `Symmetric` | Reflects the data at boundaries as if there were a mirror at each edge. |
| `ZeroPadding` | Fills values outside the boundaries with zeros. |

