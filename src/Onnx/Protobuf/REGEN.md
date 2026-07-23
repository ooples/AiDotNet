# Regenerating ONNX Protobuf C# Classes

The `Generated/Onnx.cs` file in this directory is **vendored** — produced by `protoc` from `onnx.proto3` and committed to the repo so AiDotNet's build does not need a `protoc` toolchain.

## When to regenerate

- Updating to a newer ONNX schema release (currently pinned to **v1.17.0**).
- Adding a new field or message to support a layer type the current schema doesn't cover. (This should be rare — the ONNX schema is stable across operator additions; new layer types usually only need new operator *names*, not new schema messages.)

## How to regenerate

```bash
# 1. Re-fetch the .proto from a pinned ONNX tag (replace v1.17.0 with the new tag).
curl -sSL -o src/Onnx/Protobuf/onnx.proto3 \
  https://raw.githubusercontent.com/onnx/onnx/v1.17.0/onnx/onnx.proto3

# 2. Re-apply the AiDotNet namespace override.
#    Add this block after the `package onnx;` line in the freshly-fetched file:
#
#        option csharp_namespace = "AiDotNet.Onnx.Protobuf";
#
#    (See the existing onnx.proto3 in this directory for the exact comment text.)

# 3. Generate C# classes. Requires protoc >= 28 with built-in C# generator support.
#    grpcio-tools' protoc does NOT include the C# generator; use the official protoc release.
protoc -I src/Onnx/Protobuf --csharp_out=src/Onnx/Protobuf/Generated src/Onnx/Protobuf/onnx.proto3

# 4. Build to confirm no regressions.
dotnet build src/AiDotNet.csproj

# 5. Run the ONNX round-trip tests.
dotnet test tests/AiDotNet.Tests --filter "FullyQualifiedName~Onnx"
```

## Why vendored rather than NuGet

Two reasons:

1. **Multi-target compatibility.** AiDotNet targets both `net10.0` and `net471`. The most popular community NuGet (`Onnx.Net`) only supports `netstandard2.1`, which is not compatible with `net471`. Vendoring the generated classes works on both targets because `Google.Protobuf` (the only runtime dep) supports both.

2. **License clarity.** AiDotNet is BSL 1.1. Pulling another package adds a license surface to audit on every update. The ONNX `.proto3` is Apache-2.0 (compatible) and the generated C# is a transformation we own — no external package version to track during BSL audits.

## Files in this directory

| File | Source | Purpose |
|---|---|---|
| `onnx.proto3` | github.com/onnx/onnx tag v1.17.0 + AiDotNet namespace override | Schema definition. Edit to bump version or adjust namespace. |
| `Generated/Onnx.cs` | `protoc` output | Generated C# protobuf classes. **Do not edit by hand** — re-run protoc. |
| `REGEN.md` | This file | Instructions. |
