---
title: "IDisposableGaussianProcess<T>"
description: "A Gaussian Process whose implementation owns native or unmanaged resources (e.g., cached Cholesky factors, GPU memory) and must be released deterministically."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

A Gaussian Process whose implementation owns native or unmanaged resources
(e.g., cached Cholesky factors, GPU memory) and must be released deterministically.

## How It Works

Implemented separately from `IGaussianProcess` so the base contract
stays a pure-data interface. External implementers of `IGaussianProcess`
are not forced to add a Dispose method just because the framework's own
implementations need one.

