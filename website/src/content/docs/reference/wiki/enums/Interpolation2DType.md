---
title: "Interpolation2DType"
description: "Specifies different methods for interpolating 2D data points to create a continuous surface."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different methods for interpolating 2D data points to create a continuous surface.

## For Beginners

Interpolation is like "filling in the blanks" between known data points. 
Imagine you have temperature readings from several weather stations across a city, and you 
want to estimate the temperature at locations between these stations. Interpolation methods 
are different mathematical techniques to make these estimates.

Each method has different strengths:

- Some are faster but less accurate
- Some preserve certain properties of your data better than others
- Some work better for smooth data, others for data with sharp changes

The right choice depends on your specific data and what properties you want to preserve.

## Fields

| Field | Summary |
|:-----|:--------|
| `Bicubic` | A smoother interpolation method that uses cubic polynomials in both x and y directions. |
| `Bilinear` | A simple, fast interpolation method that uses linear interpolation in both x and y directions. |
| `CubicConvolution` | An interpolation method that preserves the sharpness of edges while providing smooth results elsewhere. |
| `Kriging` | A geostatistical method that uses spatial correlation between data points. |
| `MovingLeastSquares` | A flexible method that fits local polynomial functions to nearby data points. |
| `MultiQuadratic` | An interpolation method using radial basis functions with multiquadratic form. |
| `ShepardsMethod` | A distance-weighted interpolation method that gives more influence to nearby points. |
| `ThinPlateSpline` | A flexible interpolation method that minimizes the bending energy of a thin metal plate. |

