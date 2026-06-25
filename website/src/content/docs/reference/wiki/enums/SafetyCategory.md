---
title: "SafetyCategory"
description: "Comprehensive taxonomy of safety and harm categories for content classification."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Comprehensive taxonomy of safety and harm categories for content classification.

## For Beginners

These are the specific types of harmful content that the
safety system can detect. Each category represents a different kind of risk,
from hate speech to deepfakes to copyright violations. You can configure which
categories to check for and what action to take for each.

## How It Works

This enum defines a hierarchical taxonomy of content safety categories covering
all known types of harmful, inappropriate, or policy-violating content across
text, image, audio, and video modalities.

**References:**

- UnsafeBench 11-category taxonomy (Qu et al., 2024)
- WildGuard 13-risk-category classification (Allen AI, 2024)
- OmniSafeBench-MM 9 risk domains with 50 fine-grained categories (2025)
- MM-SafetyBench 13 scenarios (Liu et al., ECCV 2024)
- EU AI Act risk classification (Articles 5, 6, 50, 52)

## Fields

| Field | Summary |
|:-----|:--------|
| `AIGenerated` | Content detected as AI-generated (text, image, audio, or video). |
| `Bias` | Biased content that treats demographic groups inequitably. |
| `CopyrightViolation` | Content that infringes copyright or reproduces protected works. |
| `Deepfake` | Deepfake content — AI-generated or manipulated media impersonating real people. |
| `Dehumanization` | Content that dehumanizes or degrades individuals or groups. |
| `Discrimination` | Discriminatory content or policies targeting protected groups. |
| `Disinformation` | Disinformation — deliberately false content intended to deceive. |
| `Doxxing` | Doxxing — publishing private information to identify or locate someone. |
| `DrugManufacturing` | Instructions for manufacturing illegal drugs. |
| `FinancialAdvice` | Unqualified financial advice that could cause harm. |
| `Fraud` | Fraud, scam, or social engineering content. |
| `Hallucination` | Hallucinated or fabricated content not grounded in facts or source material. |
| `Harassment` | Targeted harassment or bullying of individuals. |
| `HateSpeech` | Hate speech targeting protected groups based on race, religion, gender, etc. |
| `IllegalActivities` | Instructions or promotion of illegal activities. |
| `Impersonation` | Impersonation of real individuals or organizations. |
| `JailbreakAttempt` | Jailbreak attempt — attempts to bypass safety measures. |
| `LegalAdvice` | Unqualified legal advice that could cause harm. |
| `Malware` | Malware, hacking tools, or cyberattack instructions. |
| `Manipulated` | Content that has been digitally manipulated or altered. |
| `MedicalAdvice` | Unqualified medical advice that could cause harm. |
| `Misinformation` | Misinformation — unintentionally false or misleading content. |
| `ModelExtraction` | Attempts to extract or replicate the model itself. |
| `PIIExposure` | Exposure of personally identifiable information (PII). |
| `PolicyViolation` | Content that violates a topic restriction or custom policy rule. |
| `PromptInjection` | Prompt injection — attempts to override system instructions. |
| `SexualExplicit` | Sexually explicit content including pornography or graphic sexual acts. |
| `SexualMinors` | Child sexual abuse material (CSAM). |
| `SexualSuggestive` | Sexually suggestive content that is not explicitly graphic. |
| `SocialEngineering` | Social engineering content designed to manipulate victims. |
| `Stereotyping` | Stereotyping content that reinforces harmful generalizations. |
| `SurveillanceEnabling` | Content that enables mass surveillance. |
| `TrainingDataLeakage` | Leakage of memorized training data. |
| `TransparencyViolation` | Content that lacks transparency about AI involvement (regulatory compliance). |
| `ViolenceGraphic` | Graphic depictions of violence, gore, or injury. |
| `ViolenceSelfHarm` | Content promoting, glorifying, or instructing self-harm. |
| `ViolenceSuicide` | Content promoting, glorifying, or instructing suicide. |
| `ViolenceTerrorism` | Content promoting, glorifying, or instructing terrorism. |
| `ViolenceThreat` | Threats of violence against individuals or groups. |
| `ViolenceWeapons` | Content promoting or depicting weapons use. |
| `Watermarked` | Content that contains a digital watermark (informational, not harmful). |
| `WeaponsInstructions` | Instructions for manufacturing or using weapons. |

