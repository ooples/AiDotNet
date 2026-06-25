---
title: "SlackNotificationService"
description: "Slack notification service using webhooks or bot tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Notifications`

Slack notification service using webhooks or bot tokens.

## How It Works

**For Beginners:** This service sends training notifications to Slack.
You can use either an incoming webhook (simpler) or a bot token (more flexible).

Webhook Example:

Bot Token Example:

To create a webhook:

1. Go to https://api.slack.com/apps
2. Create a new app or select existing
3. Go to Incoming Webhooks
4. Add a webhook to your workspace
5. Copy the webhook URL

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlackNotificationService(SlackConfiguration,HttpClient)` | Creates a new Slack notification service. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsConfigured` |  |
| `ServiceName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes the Slack service resources. |
| `Dispose(Boolean)` | Disposes the Slack service resources. |
| `Send(TrainingNotification)` |  |
| `SendAsync(TrainingNotification,CancellationToken)` |  |
| `TestConnectionAsync(CancellationToken)` |  |

