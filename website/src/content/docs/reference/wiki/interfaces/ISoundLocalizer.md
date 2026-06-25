---
title: "ISoundLocalizer<T>"
description: "Interface for sound localization models that estimate the spatial position of sound sources."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for sound localization models that estimate the spatial position of sound sources.

## For Beginners

Sound localization is like closing your eyes and pointing
to where a sound is coming from.

How it works (like human hearing):

1. Sound reaches one ear slightly before the other (ITD - Interaural Time Difference)
2. Sound is slightly louder in the closer ear (ILD - Interaural Level Difference)
3. Head shape affects high frequencies differently for each ear
4. Brain combines all cues to determine direction

What's measured:

- Azimuth: Left-right angle (0° = front, 90° = right, -90° = left)
- Elevation: Up-down angle (0° = level, 90° = above)
- Distance: How far away (harder to estimate from audio alone)

Use cases:

- Spatial audio for VR/AR (place sounds correctly in 3D)
- Smart speakers (know which direction user is speaking from)
- Security (detect where intruder sounds come from)
- Robotics (navigate toward or away from sounds)
- Audio surveillance (track moving sound sources)
- Hearing aids (enhance sounds from specific directions)

## How It Works

Sound localization estimates where sound is coming from in 3D space. This requires
multi-channel audio (stereo or more) and uses differences in timing, loudness, and
spectral content between channels to determine direction.

## Properties

| Property | Summary |
|:-----|:--------|
| `ArrayConfig` | Gets the microphone array geometry if applicable. |
| `RequiredChannels` | Gets the number of audio channels required. |
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportsDistanceEstimation` | Gets whether this model can estimate distance (not just direction). |
| `SupportsMultipleSourceTracking` | Gets whether this model can track multiple simultaneous sources. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Beamform(Tensor<>,Double,Double)` | Beamforms audio to focus on a specific direction. |
| `ComputeSpatialSpectrum(Tensor<>,Double)` | Computes spatial power spectrum for visualization. |
| `EstimateDirections(Tensor<>,Int32)` | Estimates direction of arrival (DOA) for dominant sources. |
| `Localize(Tensor<>)` | Localizes sound sources in multi-channel audio. |
| `LocalizeAsync(Tensor<>,CancellationToken)` | Localizes sound sources asynchronously. |
| `TrackSources(Tensor<>,Double)` | Tracks sound source positions over time. |

