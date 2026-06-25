---
title: "CalibrationMethod"
description: "Calibration methods for quantization - techniques to determine optimal scaling factors when converting high-precision models to low-precision formats."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Calibration methods for quantization - techniques to determine optimal scaling factors
when converting high-precision models to low-precision formats.

## How It Works

**For Beginners:** Quantization compresses AI models by using smaller numbers (e.g., 8-bit
instead of 32-bit). Calibration is the process of figuring out the best way to map large
numbers to small numbers without losing too much accuracy.

Think of it like adjusting a thermostat - you need to find the right scale to represent
temperatures accurately. Different calibration methods use different strategies:

- **None**: No calibration, uses default scaling. Fastest but least accurate.
- **MinMax**: Simple approach using the min/max values seen in data. Fast and usually good enough.
- **Histogram**: Analyzes distribution of values to find better scaling. More accurate than MinMax.
- **Entropy**: Uses information theory (KL divergence) to minimize information loss. Most accurate but slowest.
- **MSE**: Minimizes the mean squared error between original and quantized values.
- **Percentile**: Ignores outliers by using percentiles instead of absolute min/max. Good for noisy data.

For most cases, **MinMax** or **Histogram** provides a good balance of speed and accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Entropy` | Entropy-based calibration - uses KL divergence to minimize information loss. |
| `Histogram` | Histogram-based calibration - analyzes value distribution using percentiles. |
| `MSE` | Mean Squared Error (MSE) based calibration. |
| `MinMax` | Min-Max calibration - simple range-based scaling. |
| `None` | No calibration - uses default symmetric scaling. |
| `Percentile` | Percentile-based calibration - uses percentiles to handle outliers. |

