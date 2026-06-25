---
title: "GpuDiagnosticLevel"
description: "Verbosity level for GPU backend diagnostic output."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Verbosity level for GPU backend diagnostic output.

## For Beginners

Think of this like the log level in any
logging framework: `Silent` is OFF, `Minimal`
is "just the important stuff", `Verbose` is "tell me
everything".

## How It Works

Addresses github.com/ooples/AiDotNet#1122. AiDotNet's GPU backends
(OpenCL, HIP, CUDA) emit status messages during device discovery,
kernel compilation, and availability checks. Historically these were
always written to `WriteLine`, producing
~40 lines of output on every `AiModelBuilder.BuildAsync()`. This
enum gives applications fine-grained control over that output.

## Fields

| Field | Summary |
|:-----|:--------|
| `Minimal` | Only critical GPU backend output — device selected, compilation failures, OpenCL DLL-not-found errors. |
| `Silent` | No GPU backend output written to Console or any sink. |
| `Verbose` | All GPU backend diagnostic output — device discovery, every kernel compilation step, OpenCL platform queries, GEMM tuning progress. |

