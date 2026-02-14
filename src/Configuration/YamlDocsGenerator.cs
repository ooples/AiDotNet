using System.Reflection;

namespace AiDotNet.Configuration;

/// <summary>
/// Generates markdown reference documentation for the AiDotNet YAML configuration system.
/// Outputs a complete reference of all available sections, their types, properties,
/// and available implementation choices.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Run this to generate a markdown file documenting every
/// YAML configuration option:</para>
/// <code>
/// var docs = YamlDocsGenerator.Generate();
/// File.WriteAllText("docs/yaml-config-reference.md", docs);
/// </code>
/// </remarks>
public static class YamlDocsGenerator
{
    /// <summary>
    /// Generates markdown reference documentation for the YAML configuration system.
    /// </summary>
    /// <returns>Markdown string with the full configuration reference.</returns>
    public static string Generate()
    {
        var sb = new StringBuilder();

        sb.AppendLine("# AiDotNet YAML Configuration Reference");
        sb.AppendLine();
        sb.AppendLine("This document describes all available YAML configuration sections for `AiModelBuilder`.");
        sb.AppendLine("Every section maps 1:1 to a `Configure*()` method on the builder.");
        sb.AppendLine();
        sb.AppendLine("## Quick Start");
        sb.AppendLine();
        sb.AppendLine("```csharp");
        sb.AppendLine("var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>(\"config.yaml\");");
        sb.AppendLine("var result = await builder.BuildAsync();");
        sb.AppendLine("```");
        sb.AppendLine();
        sb.AppendLine("## Table of Contents");
        sb.AppendLine();

        // Build TOC from metadata
        var sections = YamlSchemaMetadata.Sections;
        var categories = new Dictionary<string, List<YamlSectionMeta>>
        {
            { "Enum-Based Selection", new List<YamlSectionMeta>() },
            { "Deployment & Infrastructure (POCO)", new List<YamlSectionMeta>() },
            { "Interface-Based (type + params)", new List<YamlSectionMeta>() },
            { "Generic Configuration (type + params)", new List<YamlSectionMeta>() },
            { "Pipeline (steps)", new List<YamlSectionMeta>() },
        };

        // Add enum-based manually (they're in the hand-written file)
        sb.AppendLine("- [optimizer](#optimizer)");
        sb.AppendLine("- [timeSeriesModel](#timeseriesmodel)");

        foreach (var section in sections)
        {
            if (section.SectionName == "Model") continue; // covered by timeSeriesModel

            var cat = section.Category switch
            {
                "Poco" when section.IsHandWritten => "Deployment & Infrastructure (POCO)",
                "Poco" => "Deployment & Infrastructure (POCO)",
                "Interface" => "Interface-Based (type + params)",
                "AbstractGeneric" => "Interface-Based (type + params)",
                "ConcreteGeneric" => "Generic Configuration (type + params)",
                "Pipeline" => "Pipeline (steps)",
                _ => "Interface-Based (type + params)",
            };

            if (categories.ContainsKey(cat))
            {
                categories[cat].Add(section);
            }

            var anchor = ToCamelCase(section.PropertyName).ToLowerInvariant();
            sb.AppendLine($"- [{ToCamelCase(section.PropertyName)}](#{anchor})");
        }

        sb.AppendLine();

        // Enum-based sections
        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine("## Enum-Based Selection");
        sb.AppendLine();

        GenerateOptimizerDocs(sb);
        GenerateTimeSeriesModelDocs(sb);

        // POCO config sections
        sb.AppendLine("## Deployment & Infrastructure Configuration");
        sb.AppendLine();

        var pocoSections = new (string yamlKey, string description, Type configType)[]
        {
            ("quantization", "Model quantization configuration for lower precision inference.", typeof(Deployment.Configuration.QuantizationConfig)),
            ("compression", "Model compression configuration for reducing model size.", typeof(Deployment.Configuration.CompressionConfig)),
            ("caching", "Model caching configuration for storing loaded models.", typeof(Deployment.Configuration.CacheConfig)),
            ("versioning", "Model versioning configuration for managing model versions.", typeof(Deployment.Configuration.VersioningConfig)),
            ("abTesting", "A/B testing configuration for comparing model versions.", typeof(Deployment.Configuration.ABTestingConfig)),
            ("telemetry", "Telemetry configuration for tracking inference metrics.", typeof(Deployment.Configuration.TelemetryConfig)),
            ("export", "Export configuration for converting models to different formats.", typeof(Deployment.Configuration.ExportConfig)),
            ("gpuAcceleration", "GPU acceleration configuration for hardware-accelerated computation.", typeof(Engines.GpuAccelerationConfig)),
            ("profiling", "Performance profiling configuration.", typeof(Deployment.Configuration.ProfilingConfig)),
            ("jitCompilation", "JIT compilation configuration for accelerated inference.", typeof(JitCompilationConfig)),
            ("mixedPrecision", "Mixed precision training configuration.", typeof(MixedPrecision.MixedPrecisionConfig)),
            ("reasoning", "Reasoning strategy configuration.", typeof(Reasoning.Models.ReasoningConfig)),
            ("benchmarking", "Benchmarking configuration for standardized benchmark suites.", typeof(BenchmarkingOptions)),
            ("inferenceOptimizations", "Inference optimization configuration (KV caching, batching, speculative decoding).", typeof(InferenceOptimizationConfig)),
            ("interpretability", "Interpretability configuration for model explainability.", typeof(Models.Options.InterpretabilityOptions)),
            ("memoryManagement", "Training memory management configuration.", typeof(Training.Memory.TrainingMemoryConfig)),
        };

        foreach (var (yamlKey, description, configType) in pocoSections)
        {
            GeneratePocoSectionDocs(sb, yamlKey, description, configType);
        }

        // Interface-based sections
        sb.AppendLine("## Interface-Based Sections");
        sb.AppendLine();
        sb.AppendLine("These sections use `type` to select a concrete implementation and `params` to configure it.");
        sb.AppendLine();

        foreach (var section in sections)
        {
            if (section.SectionName == "Model") continue;
            if (section.IsHandWritten) continue;

            switch (section.Category)
            {
                case "Interface":
                case "AbstractGeneric":
                    GenerateInterfaceSectionDocs(sb, section);
                    break;
                case "ConcreteGeneric":
                    GenerateGenericPocoDocs(sb, section);
                    break;
                case "Pipeline":
                    GeneratePipelineDocs(sb, section);
                    break;
                case "Poco" when !section.IsHandWritten:
                    GeneratePocoFromMetadataDocs(sb, section);
                    break;
            }
        }

        return sb.ToString();
    }

    private static void GenerateOptimizerDocs(StringBuilder sb)
    {
        sb.AppendLine("### optimizer");
        sb.AppendLine();
        sb.AppendLine("Select the optimizer algorithm for model training.");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine("optimizer:");
        sb.AppendLine("  type: \"Adam\"");
        sb.AppendLine("```");
        sb.AppendLine();
        sb.AppendLine("**Available types:**");
        sb.AppendLine();

        foreach (var name in Enum.GetNames(typeof(Enums.OptimizerType)))
        {
            sb.AppendLine($"- `{name}`");
        }

        sb.AppendLine();
    }

    private static void GenerateTimeSeriesModelDocs(StringBuilder sb)
    {
        sb.AppendLine("### timeSeriesModel");
        sb.AppendLine();
        sb.AppendLine("Select the time series model type.");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine("timeSeriesModel:");
        sb.AppendLine("  type: \"ARIMA\"");
        sb.AppendLine("```");
        sb.AppendLine();
        sb.AppendLine("**Available types:**");
        sb.AppendLine();

        foreach (var name in Enum.GetNames(typeof(Enums.TimeSeriesModelType)))
        {
            sb.AppendLine($"- `{name}`");
        }

        sb.AppendLine();
    }

    private static void GeneratePocoSectionDocs(StringBuilder sb, string yamlKey, string description, Type configType)
    {
        sb.AppendLine($"### {yamlKey}");
        sb.AppendLine();
        sb.AppendLine(description);
        sb.AppendLine();
        sb.AppendLine("| Property | Type | Description |");
        sb.AppendLine("|----------|------|-------------|");

        foreach (var prop in configType.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (prop.GetGetMethod() is null || prop.GetSetMethod() is null) continue;

            var propName = ToCamelCase(prop.Name);
            var typeName = GetFriendlyTypeName(prop.PropertyType);
            var enumValues = "";

            var underlying = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;
            if (underlying.IsEnum)
            {
                var names = Enum.GetNames(underlying);
                enumValues = $" Values: {string.Join(", ", names.Take(5))}";
                if (names.Length > 5)
                {
                    enumValues += $" ... ({names.Length} total)";
                }
            }

            sb.AppendLine($"| `{propName}` | {typeName} | {enumValues} |");
        }

        sb.AppendLine();
        sb.AppendLine("**Example:**");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine($"{yamlKey}:");

        var exampleProps = configType.GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.GetGetMethod() is not null && p.GetSetMethod() is not null)
            .Take(3);

        foreach (var prop in exampleProps)
        {
            var propName = ToCamelCase(prop.Name);
            var exampleValue = GetExampleValue(prop.PropertyType);
            sb.AppendLine($"  {propName}: {exampleValue}");
        }

        sb.AppendLine("```");
        sb.AppendLine();
    }

    private static void GenerateInterfaceSectionDocs(StringBuilder sb, YamlSectionMeta section)
    {
        var yamlKey = ToCamelCase(section.PropertyName);
        sb.AppendLine($"### {yamlKey}");
        sb.AppendLine();
        sb.AppendLine($"Configuration for {section.SectionName}. Select a concrete implementation by type name.");
        sb.AppendLine();

        if (YamlRegisteredTypeNames.SectionTypes.TryGetValue(section.SectionName, out var typeNames) && typeNames.Length > 0)
        {
            sb.AppendLine("**Available types:**");
            sb.AppendLine();
            foreach (var name in typeNames)
            {
                sb.AppendLine($"- `{name}`");
            }
            sb.AppendLine();
        }

        sb.AppendLine("**Example:**");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine($"{yamlKey}:");

        var exampleType = typeNames is { Length: > 0 } ? typeNames[0] : "TypeName";
        sb.AppendLine($"  type: \"{exampleType}\"");
        sb.AppendLine("  params:");
        sb.AppendLine("    # Set implementation-specific properties here");
        sb.AppendLine("```");
        sb.AppendLine();
    }

    private static void GenerateGenericPocoDocs(StringBuilder sb, YamlSectionMeta section)
    {
        var yamlKey = ToCamelCase(section.PropertyName);
        sb.AppendLine($"### {yamlKey}");
        sb.AppendLine();
        sb.AppendLine($"Configuration for {section.SectionName}. Set properties via the params dictionary.");
        sb.AppendLine();
        sb.AppendLine("**Example:**");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine($"{yamlKey}:");
        sb.AppendLine("  params:");
        sb.AppendLine("    # Set configuration properties here");
        sb.AppendLine("```");
        sb.AppendLine();
    }

    private static void GeneratePipelineDocs(StringBuilder sb, YamlSectionMeta section)
    {
        var yamlKey = ToCamelCase(section.PropertyName);
        sb.AppendLine($"### {yamlKey}");
        sb.AppendLine();
        sb.AppendLine($"Pipeline configuration for {section.SectionName}. Define ordered processing steps.");
        sb.AppendLine();
        sb.AppendLine("**Example:**");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine($"{yamlKey}:");
        sb.AppendLine("  steps:");
        sb.AppendLine("    - type: \"StepType1\"");
        sb.AppendLine("    - type: \"StepType2\"");
        sb.AppendLine("      params:");
        sb.AppendLine("        setting: value");
        sb.AppendLine("```");
        sb.AppendLine();
    }

    private static void GeneratePocoFromMetadataDocs(StringBuilder sb, YamlSectionMeta section)
    {
        var yamlKey = ToCamelCase(section.PropertyName);
        sb.AppendLine($"### {yamlKey}");
        sb.AppendLine();
        sb.AppendLine($"Configuration for {section.SectionName}.");
        sb.AppendLine();

        if (section.PocoProperties.Length > 0)
        {
            sb.AppendLine("| Property | Type |");
            sb.AppendLine("|----------|------|");

            foreach (var propEntry in section.PocoProperties)
            {
                var parts = propEntry.Split(':');
                if (parts.Length == 2)
                {
                    sb.AppendLine($"| `{parts[0]}` | {parts[1]} |");
                }
            }

            sb.AppendLine();
        }

        sb.AppendLine("**Example:**");
        sb.AppendLine();
        sb.AppendLine("```yaml");
        sb.AppendLine($"{yamlKey}:");

        foreach (var propEntry in section.PocoProperties.Take(3))
        {
            var parts = propEntry.Split(':');
            if (parts.Length == 2)
            {
                var exampleValue = parts[1] switch
                {
                    "boolean" => "true",
                    "integer" => "10",
                    "number" => "0.01",
                    "string" => "\"value\"",
                    _ => "{}",
                };
                sb.AppendLine($"  {parts[0]}: {exampleValue}");
            }
        }

        sb.AppendLine("```");
        sb.AppendLine();
    }

    private static string GetFriendlyTypeName(Type type)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;
        var isNullable = Nullable.GetUnderlyingType(type) is not null;
        var suffix = isNullable ? "?" : "";

        if (underlying == typeof(bool)) return $"boolean{suffix}";
        if (underlying == typeof(int)) return $"integer{suffix}";
        if (underlying == typeof(long)) return $"long{suffix}";
        if (underlying == typeof(float)) return $"float{suffix}";
        if (underlying == typeof(double)) return $"double{suffix}";
        if (underlying == typeof(string)) return "string";
        if (underlying.IsEnum) return $"enum ({underlying.Name})";
        if (underlying.IsGenericType && underlying.GetGenericTypeDefinition() == typeof(Dictionary<,>)) return "object";
        if (underlying.IsGenericType && underlying.GetGenericTypeDefinition() == typeof(List<>)) return "array";
        return underlying.Name;
    }

    private static string GetExampleValue(Type type)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;

        if (underlying == typeof(bool)) return "true";
        if (underlying == typeof(int) || underlying == typeof(long)) return "10";
        if (underlying == typeof(float) || underlying == typeof(double)) return "0.01";
        if (underlying == typeof(string)) return "\"value\"";
        if (underlying.IsEnum)
        {
            var names = Enum.GetNames(underlying);
            return names.Length > 0 ? $"\"{names[0]}\"" : "\"Unknown\"";
        }

        return "{}";
    }

    private static string ToCamelCase(string name)
    {
        if (string.IsNullOrEmpty(name)) return name;
        return char.ToLowerInvariant(name[0]) + name.Substring(1);
    }
}
