---
title: "ModelMetadataExemptAttribute"
description: "Marks a class that implements IFullModel as exempt from model metadata validation diagnostics."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Marks a class that implements IFullModel as exempt from model metadata validation diagnostics.

## For Beginners

Some classes implement IFullModel because they need to wrap or
contain models, but they aren't models themselves. This attribute tells the build system
to skip validation checks (like requiring [ModelDomain], [ModelCategory], etc.) on these
classes.

## How It Works

Apply this attribute to classes that implement IFullModel but are not user-facing models
intended for discovery. For example, result wrappers, adapted model containers, or
internal infrastructure classes that should not appear in the model discovery API.

**Usage:**

This attribute is intentionally public so that external consumers extending the library
can exempt their own IFullModel implementations from metadata validation diagnostics.

