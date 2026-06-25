---
title: "Contradiction"
description: "Represents a detected contradiction between reasoning steps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents a detected contradiction between reasoning steps.

## For Beginners

This class stores information about a contradiction that was found,
including which steps conflict and why.

## Properties

| Property | Summary |
|:-----|:--------|
| `Explanation` | Description of why these steps contradict each other. |
| `Severity` | Severity of the contradiction (0.0 = minor inconsistency, 1.0 = direct logical contradiction). |
| `Step1Number` | The step number of the first conflicting statement. |
| `Step2Number` | The step number of the second conflicting statement. |

