---
title: "AudioAugmentationSettings"
description: "Audio-specific augmentation settings with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Audio-specific augmentation settings with industry-standard defaults.

## For Beginners

These settings control how audio data is augmented.
Defaults are based on best practices from audiomentations and torchaudio.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableNoise` | Gets or sets whether background noise is enabled. |
| `EnablePitchShift` | Gets or sets whether pitch shifting is enabled. |
| `EnableTimeShift` | Gets or sets whether time shift is enabled. |
| `EnableTimeStretch` | Gets or sets whether time stretching is enabled. |
| `EnableVolumeChange` | Gets or sets whether volume change is enabled. |
| `MaxTimeShift` | Gets or sets the maximum time shift as a fraction of audio length. |
| `MaxTimeStretch` | Gets or sets the maximum time stretch factor. |
| `MinTimeStretch` | Gets or sets the minimum time stretch factor. |
| `NoiseSNR` | Gets or sets the signal-to-noise ratio in decibels. |
| `PitchShiftRange` | Gets or sets the pitch shift range in semitones. |
| `VolumeChangeRange` | Gets or sets the volume change range in decibels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

