---
title: "RdpPrivacyAccountant"
description: "A simple RĂ©nyi Differential Privacy (RDP) accountant for repeated Gaussian mechanisms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy.Accounting`

A simple RĂ©nyi Differential Privacy (RDP) accountant for repeated Gaussian mechanisms.

## How It Works

**For Beginners:** RDP accounting often gives tighter (less pessimistic) privacy reporting
than basic composition. This implementation uses a conservative Gaussian-mechanism RDP bound
and converts it back to an (epsilon, delta) style report.

