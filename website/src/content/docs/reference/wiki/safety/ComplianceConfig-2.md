---
title: "ComplianceConfig"
description: "Configuration for regulatory compliance."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for regulatory compliance.

## For Beginners

Compliance settings enable regulatory framework-specific checks.
The EU AI Act requires transparency and watermarking for AI-generated content.
GDPR requires PII detection and erasure support. SOC2 requires audit logging.
Enabling a compliance mode automatically enables the safety features required by that regulation.

## Properties

| Property | Summary |
|:-----|:--------|
| `EUAIAct` | Gets or sets whether EU AI Act compliance mode is enabled. |
| `GDPR` | Gets or sets whether GDPR compliance mode is enabled. |
| `SOC2` | Gets or sets whether SOC2 compliance mode is enabled. |

