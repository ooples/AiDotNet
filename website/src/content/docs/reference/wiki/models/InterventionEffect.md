---
title: "InterventionEffect<T>"
description: "Represents the effect of an intervention in a time series or sequential data, capturing the starting point, duration, and magnitude of the effect."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents the effect of an intervention in a time series or sequential data, capturing the starting point,
duration, and magnitude of the effect.

## For Beginners

This class represents a change or treatment that was applied at a specific point in time and had some measurable effect.

For example, you might use this to model:

- The effect of a marketing campaign that ran for 2 weeks
- The impact of a policy change that was implemented on a specific date
- The result of a medical treatment that was administered for a certain period

The class stores three key pieces of information:

- When the intervention started (as an index in a sequence or time series)
- How long the intervention lasted
- How strong the effect was (positive or negative)

This information is useful for analyzing cause-and-effect relationships in data
and understanding how specific actions impact outcomes over time.

## How It Works

This class models an intervention effect, which is a change or treatment applied to a system at a specific point 
in time that continues for a certain duration and produces a measurable effect. Interventions are commonly analyzed 
in time series analysis, causal inference, and experimental studies to understand how specific actions or events 
affect outcomes over time. The class captures the essential information about an intervention: when it started, 
how long it lasted, and how strong its effect was.

## Properties

| Property | Summary |
|:-----|:--------|
| `Duration` | Gets or sets the duration of the intervention in time units or sequence steps. |
| `Effect` | Gets or sets the magnitude of the intervention's effect on the outcome variable. |
| `StartIndex` | Gets or sets the starting index of the intervention in the time series or sequence. |

