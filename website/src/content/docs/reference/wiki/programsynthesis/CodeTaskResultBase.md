---
title: "CodeTaskResultBase"
description: "Base type for structured results returned from code tasks."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ProgramSynthesis.Results`

Base type for structured results returned from code tasks.

## For Beginners

This is the common "envelope" for any code task result.
It tells you whether the task succeeded, and it includes useful metadata about what happened.

## How It Works

All task results include the task identity, a success/error envelope, and telemetry that can be
tier-redacted by Serving.

