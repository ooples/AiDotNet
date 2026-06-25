---
title: "ISqlSyntaxValidator"
description: "Validates whether a string is syntactically valid SQL, used by the program synthesizer to reject malformed candidate SQL programs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ProgramSynthesis.Interfaces`

Validates whether a string is syntactically valid SQL, used by the program
synthesizer to reject malformed candidate SQL programs.

## How It Works

The precise SQLite-backed implementation (`SqliteSqlSyntaxValidator`) ships
in the opt-in `AiDotNet.Storage.Sqlite` package (audit-2026-05 finding #14),
keeping the native SQLite dependency out of the core package. Register it via
`Validator`.

## Methods

| Method | Summary |
|:-----|:--------|
| `IsValidSql(String)` | Returns `true` if `sql` is syntactically valid SQL. |

