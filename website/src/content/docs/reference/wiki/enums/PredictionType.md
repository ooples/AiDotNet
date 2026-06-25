---
title: "PredictionType"
description: "Specifies the type of prediction task that a machine learning model performs."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of prediction task that a machine learning model performs.

## How It Works

**For Beginners:** This enum helps you tell the library what kind of prediction you're trying to make.
Think of it as telling the AI system what type of question you're asking:

- Are you asking a yes/no question? Use BinaryClassification.
- Are you asking "how much" or "what value"? Use Regression.

Choosing the right prediction type helps the AI model understand what you're trying to accomplish
and use the appropriate techniques for your specific problem.

## Fields

| Field | Summary |
|:-----|:--------|
| `BinaryClassification` | Represents a binary classification task where the output is one of two possible classes. |
| `MultiClass` | Represents a multi-class classification task where the output is one of many possible classes. |
| `MultiLabel` | Represents a multi-label classification task where multiple labels can be true at the same time. |
| `Regression` | Represents a regression task where the output is a continuous numerical value. |

