---
title: "NumericTypeExtensions"
description: "Provides extension methods for working with numeric types in AI and machine learning contexts."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Provides extension methods for working with numeric types in AI and machine learning contexts.

## For Beginners

This class contains helper methods that let you check what kind of numbers
you're working with and convert between different number formats. In AI and machine learning,
we often need to work with both regular numbers (like 1, 2.5, -3) and complex numbers
(which have both a real and imaginary part, like 3+2i).

## Methods

| Method | Summary |
|:-----|:--------|
| `IsComplexType` | Determines whether a type represents a complex number. |
| `IsRealType` | Determines whether a type represents a real number. |
| `ToRealOrComplex(Complex<>)` | Converts a complex number to either its real part or keeps it as a complex number, depending on the target type. |

