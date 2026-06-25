---
title: "INotificationService"
description: "Interface for notification services."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TrainingMonitoring.Notifications`

Interface for notification services.

## How It Works

**For Beginners:** Notification services allow you to receive alerts
about your training progress via email, Slack, or other channels.
This is especially useful for long-running training jobs where you
want to be notified of important events like completion, failures,
or when a new best model is found.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsConfigured` | Gets whether the service is properly configured and ready to send. |
| `ServiceName` | Gets the name of this notification service. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Send(TrainingNotification)` | Sends a notification synchronously. |
| `SendAsync(TrainingNotification,CancellationToken)` | Sends a notification asynchronously. |
| `TestConnectionAsync(CancellationToken)` | Tests the connection to the notification service. |

