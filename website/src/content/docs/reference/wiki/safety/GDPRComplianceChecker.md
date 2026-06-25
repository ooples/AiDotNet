---
title: "GDPRComplianceChecker<T>"
description: "Checks compliance with GDPR requirements related to AI and personal data processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Compliance`

Checks compliance with GDPR requirements related to AI and personal data processing.

## For Beginners

GDPR is a European privacy law. If your AI system handles
personal information (names, emails, etc.), you need to detect and protect that data.
This module checks that your safety pipeline has the right protections enabled.

## How It Works

The General Data Protection Regulation (GDPR, 2016/679) imposes strict requirements on
processing personal data. AI systems that process personal data must comply with data
minimization, purpose limitation, and individual rights (right to explanation, right to
erasure). This module checks whether appropriate PII safeguards are in place.

**Key requirements checked:**

- Article 5(1)(c): Data minimization — only process necessary personal data
- Article 13/14: Right to information — transparency about data processing
- Article 17: Right to erasure — ability to delete personal data
- Article 22: Automated decision-making — right to human review
- Article 35: Data Protection Impact Assessment for high-risk processing

**References:**

- GDPR (Regulation 2016/679), Articles 5, 13-14, 17, 22, 35
- EDPB Guidelines on AI and GDPR (2024)
- CNIL AI Action Plan: GDPR compliance for AI systems (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GDPRComplianceChecker(SafetyConfig)` | Initializes a new GDPR compliance checker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` | Evaluates text for GDPR compliance issues. |

