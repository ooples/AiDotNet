---
title: "ComplianceModuleBase<T>"
description: "Abstract base class for regulatory compliance checking modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Compliance`

Abstract base class for regulatory compliance checking modules.

## For Beginners

This base class provides common code for all compliance checkers.
Each checker type extends this and adds its own checks for specific laws and regulations.

## How It Works

Provides shared infrastructure for compliance modules including configuration
access and common compliance check utilities. Concrete implementations provide
the actual compliance checking logic (EU AI Act, GDPR, SOC2).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplianceModuleBase(SafetyConfig)` | Initializes the compliance module base. |

## Properties

| Property | Summary |
|:-----|:--------|
| `RegulationName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateCompliance(SafetyConfig)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `Config` | The safety configuration to evaluate for compliance. |

