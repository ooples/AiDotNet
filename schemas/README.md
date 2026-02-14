# AiDotNet YAML Configuration Schema

This directory contains the JSON Schema for AiDotNet YAML configuration files.

## Generating the Schema

The schema is generated at runtime from reflection over the actual configuration types:

```csharp
using AiDotNet.Configuration;

var schema = YamlJsonSchema.Generate();
File.WriteAllText("schemas/aidotnet-config.schema.json", schema);
```

## Using the Schema

Add this directive at the top of your YAML configuration file for IntelliSense support in VS Code (requires the [YAML Language Server](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) extension):

```yaml
# yaml-language-server: $schema=./schemas/aidotnet-config.schema.json
```

The schema provides:
- Property name autocomplete
- Type validation
- Enum value suggestions
- Hover descriptions for each section
