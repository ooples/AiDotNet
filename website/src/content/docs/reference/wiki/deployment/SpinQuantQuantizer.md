---
title: "SpinQuantQuantizer<T, TInput, TOutput>"
description: "SpinQuant quantizer - uses learned rotation matrices to reduce outliers before quantization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Strategies`

SpinQuant quantizer - uses learned rotation matrices to reduce outliers before quantization.
Applies orthogonal transformations to weight matrices to minimize quantization error.

## For Beginners

SpinQuant "rotates" the data in a mathematical sense to spread
out outliers more evenly. This makes quantization more accurate because extreme values
cause the most problems during compression.

## How It Works

**How It Works:**

**Key Features:**

**Reference:** Liu et al., "SpinQuant: LLM quantization with learned rotations" (ICLR 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpinQuantQuantizer(QuantizationConfiguration,Int32,Double,Int32)` | Initializes a new instance of the SpinQuantQuantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `CalibrationWarnings` | Gets any warnings generated during calibration. |
| `IsCalibrated` | Gets whether the quantizer has been calibrated. |
| `Mode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRotation(Vector<>,Matrix<>)` | Applies rotation matrix to weight vector. |
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `CayleyToRotation(Matrix<>)` | Converts Cayley parameter (skew-symmetric matrix) to rotation matrix. |
| `ComputeQuantizationError(Vector<>,Matrix<>,)` | Computes quantization error for a weight block with given rotation. |
| `ComputeRotationGradient(Vector<>,Matrix<>,Int32,QuantizationConfiguration)` | Computes gradient of quantization error with respect to rotation parameters. |
| `ComputeScaleFactors(Vector<>,QuantizationConfiguration)` | Computes scale factors from parameter statistics. |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `InvertMatrix(Matrix<>)` | Inverts a matrix using Gaussian elimination with partial pivoting. |
| `LearnRotationMatrix(Vector<>,QuantizationConfiguration)` | Learns optimal rotation matrix to minimize quantization error. |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeSymmetric(Vector<>,QuantizationConfiguration)` | Performs symmetric quantization on rotated parameters. |

