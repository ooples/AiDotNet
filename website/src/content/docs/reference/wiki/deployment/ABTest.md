---
title: "ABTest"
description: "Represents a single A/B test configuration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Configuration`

Represents a single A/B test configuration.

## For Beginners

An A/B test compares two or more model versions to determine
which performs better. This class defines a single test with its versions and traffic allocation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ControlVersion` | Gets or sets the control version (baseline) for this test. |
| `Description` | Gets or sets the description of this test. |
| `EndDate` | Gets or sets the end date of the test. |
| `IsActive` | Gets or sets whether this test is active (default: true). |
| `MinimumImprovementThreshold` | Gets or sets the minimum improvement threshold to consider the treatment a winner (default: 0.01 = 1%). |
| `Name` | Gets or sets the unique name for this test. |
| `PrimaryMetric` | Gets or sets the primary metric to compare (e.g., "accuracy", "latency"). |
| `StartDate` | Gets or sets the start date of the test. |
| `TreatmentTrafficPercentage` | Gets or sets the traffic percentage for the treatment version (0.0 to 1.0). |
| `TreatmentVersion` | Gets or sets the treatment version (new version being tested). |

