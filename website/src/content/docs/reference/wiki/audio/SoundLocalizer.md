---
title: "SoundLocalizer<T>"
description: "Sound source localization using microphone arrays."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Localization`

Sound source localization using microphone arrays.

## For Beginners

Sound localization finds where sounds come from:

- Uses 2+ microphones to detect time differences in sound arrival
- Calculates direction (azimuth/elevation) or 3D position of sound sources
- Works like how human ears localize sounds

Usage:

## How It Works

Estimates the direction of arrival (DOA) of sound sources using multiple microphones.
Supports various algorithms: GCC-PHAT, MUSIC, SRP-PHAT.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoundLocalizer(Double[0:,0:],SoundLocalizerOptions)` | Creates a new SoundLocalizer with specified microphone positions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MicrophonePositions` | Gets the microphone positions (meters). |
| `NumMicrophones` | Gets the number of microphones in the array. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateCircularArray(Int32,Double,SoundLocalizerOptions)` | Creates a SoundLocalizer for a circular microphone array. |
| `CreateLinearArray(Int32,Double,SoundLocalizerOptions)` | Creates a SoundLocalizer for a linear microphone array. |
| `EstimateTdoa(Tensor<>,Tensor<>)` | Estimates time difference of arrival (TDOA) between two channels. |
| `Localize(Tensor<>[])` | Localizes sound sources from multi-channel audio. |
| `LocalizeAsync(Tensor<>[],CancellationToken)` | Localizes sound sources asynchronously. |

