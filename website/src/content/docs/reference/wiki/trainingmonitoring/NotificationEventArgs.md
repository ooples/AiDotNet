---
title: "NotificationEventArgs"
description: "Event arguments for notification events."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Notifications`

Event arguments for notification events.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NotificationEventArgs(TrainingNotification,String,Boolean,Exception)` | Creates notification event arguments. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Exception` | Gets any exception that occurred during sending. |
| `Notification` | Gets the notification that was sent. |
| `ServiceName` | Gets the service name that handled the notification. |
| `Success` | Gets whether the notification was sent successfully. |

