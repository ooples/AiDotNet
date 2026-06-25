---
title: "ZeroTerminalSNRSchedule<T>"
description: "Zero Terminal SNR noise schedule ensuring signal-to-noise ratio reaches exactly zero at the final timestep."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers.NoiseSchedules`

Zero Terminal SNR noise schedule ensuring signal-to-noise ratio reaches exactly zero at the final timestep.

## For Beginners

This fixes a common issue in diffusion training where the model
never sees completely noisy images. By ensuring SNR reaches zero, the model learns to
generate from pure noise, improving overall sample quality.

## How It Works

Standard beta schedules often have non-zero SNR at the final timestep, meaning the model
never sees pure noise during training. Zero Terminal SNR rescales the schedule so that
alpha_cumprod[T] = 0, ensuring the final timestep is pure noise.

Reference: Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed", WACV 2024

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Vector<>)` | Rescales alpha cumulative products to enforce zero terminal SNR. |

