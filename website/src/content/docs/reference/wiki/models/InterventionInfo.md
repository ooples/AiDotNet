---
title: "InterventionInfo"
description: "Represents information about an intervention in a time series or sequential data, specifying when it started and how long it lasted."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents information about an intervention in a time series or sequential data, specifying when it started
and how long it lasted.

## For Beginners

This class describes when a change or treatment was applied and how long it lasted.

For example, you might use this to specify:

- When a marketing campaign started and how long it ran
- When a policy change was implemented and whether it's still in effect
- When a medical treatment began and its duration

Unlike the InterventionEffect class, this class only stores information about
timing and duration, not about the strength or direction of the effect.
This makes it useful for planning analyses or defining intervention periods
before measuring their impacts.

## How It Works

This class provides a simple representation of an intervention's timing and duration in a time series or sequential 
dataset. An intervention is a deliberate change, treatment, or event that occurs at a specific point in time and may 
continue for a certain duration. This class captures only the timing aspects of the intervention without including 
information about its effect or magnitude. It is useful for defining when interventions occurred in causal analysis, 
time series experiments, or when modeling the impact of specific events.

## Properties

| Property | Summary |
|:-----|:--------|
| `Duration` | Gets or sets the duration of the intervention in time units or sequence steps. |
| `StartIndex` | Gets or sets the starting index of the intervention in the time series or sequence. |

