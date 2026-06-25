---
title: "EmailNotificationService"
description: "Email notification service using SMTP."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Notifications`

Email notification service using SMTP.

## How It Works

**For Beginners:** This service sends training notifications via email.
It supports various email providers like Gmail, Outlook, and custom SMTP servers.

Example usage:

Note: For Gmail, you need to use an "App Password" (not your regular password).
Enable 2FA and generate an app password in your Google Account settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EmailNotificationService(EmailConfiguration)` | Creates a new email notification service. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsConfigured` |  |
| `ServiceName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes the email service resources. |
| `Dispose(Boolean)` | Disposes the email service resources. |
| `Send(TrainingNotification)` |  |
| `SendAsync(TrainingNotification,CancellationToken)` |  |
| `TestConnectionAsync(CancellationToken)` |  |

