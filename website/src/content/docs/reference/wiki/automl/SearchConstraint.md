---
title: "SearchConstraint"
description: "Defines a constraint for AutoML search to limit the search space or enforce requirements."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Defines a constraint for AutoML search to limit the search space or enforce requirements.

## Properties

| Property | Summary |
|:-----|:--------|
| `Expression` | Gets or sets the constraint expression or rule. |
| `IsHardConstraint` | Gets or sets whether this constraint is a hard constraint (must be satisfied) or soft constraint (preferred). |
| `MaxValue` | Gets or sets the maximum value for range constraints. |
| `Metadata` | Gets or sets additional metadata for the constraint. |
| `MinValue` | Gets or sets the minimum value for range constraints. |
| `Name` | Gets or sets the name of the constraint. |
| `ParameterNames` | Gets or sets the parameter names involved in this constraint. |
| `Type` | Gets or sets the type of constraint. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a clone of this search constraint. |

