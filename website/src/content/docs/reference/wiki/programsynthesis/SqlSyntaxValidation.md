---
title: "SqlSyntaxValidation"
description: "Global registration point for the optional precise SQL syntax validator used by `NeuralProgramSynthesizer`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.ProgramSynthesis.Engines`

Global registration point for the optional precise SQL syntax validator used by
`NeuralProgramSynthesizer`.

## How It Works

**For Beginners:** the program synthesizer validates candidate SQL programs. By
default it uses a lightweight generic structural check. If you want precise SQL
parsing, reference the opt-in `AiDotNet.Storage.Sqlite` package and set
`Validator` to a `SqliteSqlSyntaxValidator` instance — the
synthesizer will then use real SQLite parsing. This keeps the native SQLite
dependency out of the core package (audit-2026-05 finding #14).

## Properties

| Property | Summary |
|:-----|:--------|
| `Validator` | The active SQL syntax validator, or `null` to use generic structural validation. |

