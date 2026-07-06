---
title: "Audio Processing"
description: "Speech recognition, classification, and audio features."
order: 6
section: "Tutorials"
---

AiDotNet ships an audio toolkit — speech recognition, classification, feature extraction, and music analysis. Unlike tabular and image models, audio components are used **directly** (each is its own class) rather than through a single `AiModelBuilder` call.

## Speech Recognition (Whisper)

`WhisperModel<T>` (in `AiDotNet.Audio.Whisper`) transcribes speech. Construct it with a `WhisperModelSize` — sizes range from `Tiny` (fastest) to `Large` (most accurate) — and transcribe audio through the model's own API. See the Whisper sample for an end-to-end run.

## Audio Feature Extraction

Feature extractors turn raw audio into representations models can learn from. `ChromaExtractor` captures pitch-class energy (useful for music).

```csharp
using AiDotNet.Audio.Features;

var chroma = new ChromaExtractor<float>();
Console.WriteLine("Created a chroma feature extractor.");
```

## The Audio Toolkit

| Area | Classes | Namespace |
|:-----|:--------|:----------|
| Speech | `WhisperModel<T>` | `AiDotNet.Audio.Whisper` |
| Classification | `GenreClassifier`, `AudioEventDetector` | `AiDotNet.Audio.Classification` |
| Features | `ChromaExtractor`, and other spectral extractors | `AiDotNet.Audio.Features` |
| Music analysis | `BeatTracker`, `ChordRecognizer`, `KeyDetector` | `AiDotNet.Audio.MusicAnalysis` |
| Enhancement | denoisers such as `DeepFilterNet`, `DCCRN` | `AiDotNet.Audio` |

## Pattern: Features → Classifier

A common pipeline is to extract audio features, then train a standard classifier on them through the facade — exactly like any tabular classification task. Extract features for each clip into a row vector, stack them into a `Matrix`, and pass the labels to `DataLoaders.FromArrays(...)` with a `RandomForestClassifier` or similar.

## Best Practices

1. **Match the sample rate**: resample audio to the rate your extractor/model expects.
2. **Pick the right Whisper size**: `Tiny`/`Base` for latency, `Large` for accuracy.
3. **Normalize features**: scale extracted features before training a classifier.
4. **Window long audio**: process long clips in overlapping windows.

## Notes

Audio components are used directly rather than via a single `AiModelBuilder` call: construct the model/extractor you need (e.g. `WhisperModel`, `ChromaExtractor`, `GenreClassifier`) and call its own methods. A unified audio facade is not part of `AiModelBuilder` today.

## Next Steps

- [Whisper Sample](/samples/audio/SpeechRecognition/Whisper/)
- [Classification Tutorial](/docs/tutorials/classification/)
