---
title: "SlackConfiguration"
description: "Configuration for Slack notification service."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TrainingMonitoring.Notifications`

Configuration for Slack notification service.

## Properties

| Property | Summary |
|:-----|:--------|
| `BotToken` | Gets or sets the bot token (alternative to webhook). |
| `Channel` | Gets or sets the channel to post to (required when using bot token). |
| `IconEmoji` | Gets or sets the bot icon emoji. |
| `TimeoutMs` | Gets or sets the request timeout in milliseconds. |
| `Username` | Gets or sets the bot username. |
| `WebhookUrl` | Gets or sets the webhook URL. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForBotToken(String,String)` | Creates a configuration using a bot token. |
| `ForWebhook(String)` | Creates a configuration using a webhook URL. |
| `IsValid` | Validates the configuration. |

