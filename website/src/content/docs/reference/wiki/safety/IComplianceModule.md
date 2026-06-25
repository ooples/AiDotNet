---
title: "IComplianceModule<T>"
description: "Interface for regulatory compliance checking modules."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Compliance`

Interface for regulatory compliance checking modules.

## For Beginners

A compliance module checks if your AI system meets legal
requirements. Different laws around the world require AI systems to be transparent,
protect personal data, and maintain audit trails. This module checks all of that.

## How It Works

Compliance modules evaluate AI system configurations and outputs against regulatory
requirements such as the EU AI Act, GDPR, and SOC2. They check for required
transparency, data protection, watermarking, and audit trail compliance.

## Properties

| Property | Summary |
|:-----|:--------|
| `RegulationName` | Gets the name of the regulation this module checks compliance for. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateCompliance(SafetyConfig)` | Evaluates the current safety configuration for compliance. |

