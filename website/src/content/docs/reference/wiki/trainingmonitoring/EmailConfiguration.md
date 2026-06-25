---
title: "EmailConfiguration"
description: "Configuration for email notification service."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TrainingMonitoring.Notifications`

Configuration for email notification service.

## Properties

| Property | Summary |
|:-----|:--------|
| `CcAddresses` | Gets or sets CC email addresses. |
| `EnableSsl` | Gets or sets whether to use SSL/TLS. |
| `FromAddress` | Gets or sets the sender email address. |
| `FromName` | Gets or sets the sender display name. |
| `Password` | Gets or sets the password for SMTP authentication. |
| `SmtpHost` | Gets or sets the SMTP server host. |
| `SmtpPort` | Gets or sets the SMTP server port. |
| `TimeoutMs` | Gets or sets the connection timeout in milliseconds. |
| `ToAddresses` | Gets or sets the recipient email addresses. |
| `UseHtml` | Gets or sets whether to send HTML emails. |
| `Username` | Gets or sets the username for SMTP authentication. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForGmail(String,String,String[])` | Creates a Gmail configuration. |
| `ForOutlook(String,String,String[])` | Creates an Outlook/Office365 configuration. |
| `IsValid` | Validates the configuration. |

