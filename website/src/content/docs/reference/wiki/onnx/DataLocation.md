---
title: "DataLocation"
description: "Location of the data for this tensor."
section: "API Reference"
---

`Enums` · `AiDotNet.Onnx.Protobuf`

Location of the data for this tensor. MUST be one of:

- DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
- EXTERNAL - data stored in an external location as described by external_data field.

