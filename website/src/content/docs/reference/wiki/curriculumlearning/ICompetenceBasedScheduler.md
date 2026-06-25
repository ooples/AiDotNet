---
title: "ICompetenceBasedScheduler<T>"
description: "Interface for competence-based curriculum schedulers."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Interface for competence-based curriculum schedulers.

## Properties

| Property | Summary |
|:-----|:--------|
| `CompetenceThreshold` | Gets or sets the competence threshold to advance to next phase. |
| `CurrentCompetence` | Gets the current competence level of the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `HasMasteredCurrentContent` | Gets whether the model has mastered the current curriculum content. |
| `UpdateCompetence(CurriculumEpochMetrics<>)` | Updates the competence estimate based on model performance. |

