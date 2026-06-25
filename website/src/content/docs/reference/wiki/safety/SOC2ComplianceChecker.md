---
title: "SOC2ComplianceChecker<T>"
description: "Checks compliance with SOC 2 requirements for AI system security and availability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Compliance`

Checks compliance with SOC 2 requirements for AI system security and availability.

## For Beginners

SOC 2 is a security standard for service organizations. If your
company processes customer data through AI, auditors will check that you have proper
safety controls. This module verifies that your AI safety pipeline meets those requirements.

## How It Works

SOC 2 (Service Organization Control 2) requires organizations to demonstrate that their
systems meet the Trust Services Criteria: Security, Availability, Processing Integrity,
Confidentiality, and Privacy. This module checks whether the safety pipeline has adequate
controls for AI-specific SOC 2 concerns.

**Key requirements checked:**

- CC6.1: Logical access controls — input validation, jailbreak prevention
- CC7.2: System monitoring — safety event logging and alerting
- CC8.1: Change management — safety configuration validation
- PI1.1: Processing integrity — output validation and quality checks
- C1.1: Confidentiality — PII detection and data classification

**References:**

- AICPA SOC 2 Trust Services Criteria (2022, updated 2024)
- SOC 2 for AI systems: Emerging best practices (2024)
- AI governance and SOC 2 compliance (ISACA, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SOC2ComplianceChecker(SafetyConfig)` | Initializes a new SOC 2 compliance checker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` | Evaluates text for SOC 2 compliance issues. |

