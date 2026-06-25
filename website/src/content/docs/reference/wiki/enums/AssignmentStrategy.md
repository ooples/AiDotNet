---
title: "AssignmentStrategy"
description: "Strategy for assigning requests to model versions during A/B testing."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Strategy for assigning requests to model versions during A/B testing.

## For Beginners

This determines how traffic is distributed between different model versions:

- **Random**: Each request is randomly assigned based on traffic split percentages.

Use when you want pure statistical randomness.

- **Sticky**: Users consistently get the same version (based on user ID hash).

Use when you want each user to have a consistent experience across sessions.

- **Gradual**: Gradually shifts traffic from old to new version over time.

Use when you want to slowly roll out a new version to minimize risk.

## Fields

| Field | Summary |
|:-----|:--------|
| `Gradual` | Gradually shift traffic from old to new version over time. |
| `Random` | Each request randomly assigned based on traffic split. |
| `Sticky` | Users consistently get the same version (based on user ID hash). |

