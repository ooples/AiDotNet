---
title: "TelemetryChatMiddleware"
description: "An `IChatMiddleware` that emits OpenTelemetry GenAI telemetry for each chat call: a client span tagged with the operation, response model, finish reason, and token usage, plus operation-count and token-usage metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IChatMiddleware` that emits OpenTelemetry GenAI telemetry for each chat call: a client span
tagged with the operation, response model, finish reason, and token usage, plus operation-count and
token-usage metrics. Drop it into the middleware pipeline to make every model call observable.

## For Beginners

Add this filter and each model call automatically reports how long it took, how
many tokens it used, and why it stopped — to whatever monitoring tool you've connected, with no other code.

## How It Works

Spans and metrics are emitted on `Source`/`Meter`;
when no collector is listening, the overhead is negligible (the span is not even created). Tag names follow
the OpenTelemetry GenAI semantic conventions so standard dashboards understand them.

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ChatRequestContext,ChatPipelineDelegate,CancellationToken)` |  |

