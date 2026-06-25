---
title: "NotificationManager"
description: "Manages multiple notification services and provides unified notification delivery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Notifications`

Manages multiple notification services and provides unified notification delivery.

## How It Works

**For Beginners:** The NotificationManager allows you to send notifications
through multiple channels (email, Slack, etc.) at once. It supports:

- Sending to all registered services
- Filtering by severity (only send errors via email, info via Slack)
- Buffering notifications to avoid spam
- Background sending for non-blocking operation

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NotificationManager(Boolean,Nullable<TimeSpan>)` | Creates a new notification manager. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferFlushInterval` | Gets the buffer flush interval. |
| `BufferingEnabled` | Gets whether buffering is enabled. |
| `DefaultMinimumSeverity` | Gets the default minimum severity for all services. |
| `MaxBufferSize` | Gets the maximum buffer size before automatic flush. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddService(INotificationService)` | Adds a notification service. |
| `Dispose` | Disposes the notification manager and all services. |
| `Dispose(Boolean)` | Disposes the notification manager and all services. |
| `FlushAsync(CancellationToken)` | Flushes the notification buffer immediately. |
| `GetService(String)` | Gets a registered service by name. |
| `GetServiceNames` | Gets all registered service names. |
| `IsTypeEnabled(NotificationType)` | Checks if a notification type is enabled. |
| `RemoveService(String)` | Removes a notification service. |
| `Send(TrainingNotification)` | Sends a notification synchronously. |
| `SendAsync(TrainingNotification,CancellationToken)` | Sends a notification to all applicable services. |
| `SendBackground(TrainingNotification)` | Sends notifications in the background (fire-and-forget). |
| `SetMinimumSeverity(String,NotificationSeverity)` | Sets the minimum severity for a specific service. |
| `SetTypeEnabled(NotificationType,Boolean)` | Enables or disables a notification type. |
| `TestAllServicesAsync(CancellationToken)` | Tests all registered services. |

## Events

| Event | Summary |
|:-----|:--------|
| `NotificationFailed` | Event raised when a notification fails to send. |
| `NotificationSent` | Event raised when a notification is sent. |

