---
title: "LicenseRequiredException"
description: "Exception thrown when a model persistence operation (save or load) is attempted after the free trial period has expired and no valid license key is configured."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when a model persistence operation (save or load) is attempted
after the free trial period has expired and no valid license key is configured.

## For Beginners

AiDotNet offers a free trial for model save/load operations.
During the trial you can save and load models without restriction. Once the trial expires
(after 30 days or 10 save/load operations, whichever comes first), you need to register
for a free community license or purchase a commercial license at https://aidotnet.dev.

## How It Works

Training and inference are never restricted — this exception only applies to
`SaveModel()` and `LoadModel()` operations.

**How to resolve:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LicenseRequiredException` | Creates a new `LicenseRequiredException` with a default message. |
| `LicenseRequiredException(String)` | Creates a new `LicenseRequiredException` with a specified error message. |
| `LicenseRequiredException(String,Exception)` | Creates a new `LicenseRequiredException` with a specified error message and a reference to the inner exception that is the cause of this exception. |
| `LicenseRequiredException(TrialExpirationReason,Nullable<Int32>,Nullable<Int32>)` | Creates a new `LicenseRequiredException` with trial expiration details. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpirationReason` | Gets the reason the trial expired. |
| `OperationsPerformed` | Gets the number of save/load operations performed during the trial. |
| `TrialDaysElapsed` | Gets the number of days the trial was active before expiring. |

