---
title: "IEmotionRecognizer<T>"
description: "Defines the contract for speech emotion recognition models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for speech emotion recognition models.

## For Beginners

This is like reading emotions from someone's voice!

How humans convey emotion in speech:

- Pitch: Higher when excited/happy, lower when sad
- Speed: Faster when angry/excited, slower when sad
- Volume: Louder when angry, softer when sad/fearful
- Voice quality: Breathy, tense, relaxed

Applications:

- Call centers: Detect frustrated customers for escalation
- Mental health: Monitor patient emotional state
- Voice assistants: Respond appropriately to user mood
- Gaming: Adapt game difficulty/story based on player emotion
- Market research: Analyze focus group reactions

Challenges:

- Cultural differences in emotional expression
- Speaker variability (age, gender, accent)
- Context dependency (same words can mean different emotions)
- Mixed emotions (happy but nervous)

## How It Works

Speech Emotion Recognition (SER) identifies emotional states from voice:

- Basic emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Arousal: Low (calm) to High (excited)
- Valence: Negative to Positive
- Dominance: Submissive to Dominant

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Gets the sample rate this recognizer operates at. |
| `SupportedEmotions` | Gets the list of emotions this model can detect. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractEmotionFeatures(Tensor<>)` | Extracts emotion-relevant features from audio. |
| `GetArousal(Tensor<>)` | Gets arousal (activation) level from speech. |
| `GetEmotionProbabilities(Tensor<>)` | Gets probabilities for all supported emotions. |
| `GetValence(Tensor<>)` | Gets valence (positivity) level from speech. |
| `RecognizeEmotion(Tensor<>)` | Recognizes the primary emotion in speech audio. |
| `RecognizeEmotionTimeSeries(Tensor<>,Int32,Int32)` | Recognizes emotions over time (for longer recordings). |

