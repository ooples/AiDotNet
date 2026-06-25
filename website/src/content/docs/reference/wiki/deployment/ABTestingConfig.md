---
title: "ABTestingConfig"
description: "Configuration for A/B testing - comparing multiple model versions by splitting traffic."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for A/B testing - comparing multiple model versions by splitting traffic.

## For Beginners

A/B testing lets you try out a new model version on a small percentage
of users before fully deploying it. This helps you:

- Test new models in production safely
- Compare performance between versions with real users
- Gradually roll out changes to minimize risk
- Make data-driven decisions about which model is better

How it works:
You specify how to split traffic between versions. For example:

- Version 1.0: 80% of traffic (current stable version)
- Version 2.0: 20% of traffic (new experimental version)

Then you monitor metrics like accuracy, latency, and user satisfaction to decide
which version is better.

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `AssignmentStrategy` | Gets or sets the strategy for assigning users to versions (default: Random). |
| `ControlVersion` | Gets or sets the control group version (baseline for comparison). |
| `DefaultTrafficSplit` | Gets or sets the default traffic split percentage for new test versions (default: 0.5). |
| `Enabled` | Gets or sets whether A/B testing is enabled (default: false). |
| `MinSampleSize` | Gets or sets the minimum sample size per version before comparing results (default: 1000). |
| `TestDurationDays` | Gets or sets the duration in days for the A/B test (default: 7). |
| `Tests` | Gets or sets the list of defined A/B tests. |
| `TrackAssignments` | Gets or sets whether to track experiment assignment for each request (default: true). |
| `TrafficSplit` | Gets or sets the traffic split configuration. |

