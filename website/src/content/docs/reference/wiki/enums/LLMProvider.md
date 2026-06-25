---
title: "LLMProvider"
description: "Defines the large language model (LLM) providers available for AI agent assistance during model building and inference."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the large language model (LLM) providers available for AI agent assistance during model building and inference.

## For Beginners

This enum lets you choose which AI company's language model to use for helping
build your machine learning models.

Think of these as different AI assistants you can hire:

- **OpenAI**: Created GPT-3.5 and GPT-4, known for strong general reasoning
- **Anthropic**: Created Claude, designed to be helpful, harmless, and honest
- **Azure OpenAI**: Microsoft's enterprise version of OpenAI models with added security

Each provider requires an API key (like a password) to use their services. You would choose based on:

- Which service you already have an account with
- Pricing and rate limits
- Data privacy requirements (Azure OpenAI keeps data in your region)
- Specific model capabilities you need

For example, if your company uses Microsoft Azure, you might choose AzureOpenAI to keep
everything within your existing cloud infrastructure and compliance policies.

## How It Works

This enum specifies which LLM provider to use when enabling agent assistance in the AiModelBuilder.
Different providers offer different models with varying capabilities, pricing, and performance characteristics.
The selected provider determines which API will be called for agent operations such as model selection,
hyperparameter tuning, and conversational assistance.

## Fields

| Field | Summary |
|:-----|:--------|
| `Anthropic` | Anthropic's Claude family of models including Claude 2, Claude 3 Haiku, Claude 3 Sonnet, and Claude 3 Opus. |
| `AzureOpenAI` | Microsoft Azure-hosted OpenAI models with enterprise features, compliance, and regional data residency. |
| `OpenAI` | OpenAI's GPT family of models including GPT-3.5, GPT-4, GPT-4-turbo, and GPT-4o. |

