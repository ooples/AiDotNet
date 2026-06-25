---
title: "EUAIActComplianceChecker<T>"
description: "Checks compliance with the EU AI Act requirements for AI systems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Compliance`

Checks compliance with the EU AI Act requirements for AI systems.

## For Beginners

The EU AI Act is a law that requires AI systems used in Europe to
meet certain safety standards. High-risk systems need transparency, human oversight,
and watermarking. This module checks whether your AI system has the right safety features
enabled for compliance.

## How It Works

The EU AI Act (Regulation 2024/1689) establishes a risk-based framework for AI systems
in the European Union. This module checks whether an AI system's safety pipeline meets
the Act's requirements based on the system's risk classification.

**Key requirements checked:**

- Article 50: AI-generated content must be machine-detectable (watermarking)
- Article 52: Transparency obligations for certain AI systems
- Article 6/Annex III: High-risk AI systems require safety management systems
- Articles 9-15: Requirements for high-risk systems (data governance, accuracy, cybersecurity)

**References:**

- EU AI Act (Regulation 2024/1689), effective August 2024
- ENISA guidance on AI cybersecurity for the EU AI Act (2024)
- EU AI Act compliance frameworks survey (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EUAIActComplianceChecker(SafetyConfig)` | Initializes a new EU AI Act compliance checker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` | Evaluates text for compliance issues and checks pipeline configuration. |

