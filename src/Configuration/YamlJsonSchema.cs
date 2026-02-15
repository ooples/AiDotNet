using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Configuration;

/// <summary>
/// Generates a JSON Schema for the AiDotNet YAML configuration system.
/// The schema provides IntelliSense/autocomplete when editing YAML files in VS Code
/// (with the YAML Language Server extension) and validates configuration structure.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> JSON Schema tells your editor what properties are valid
/// in a YAML file, what values they accept, and provides descriptions on hover.
/// To use it, add this comment at the top of your YAML file:</para>
/// <code>
/// # yaml-language-server: $schema=./aidotnet-config.schema.json
/// </code>
/// <para>Then generate the schema file by calling:</para>
/// <code>
/// var schema = YamlJsonSchema.Generate();
/// File.WriteAllText("aidotnet-config.schema.json", schema);
/// </code>
/// </remarks>
public static class YamlJsonSchema
{
    /// <summary>
    /// Generates a complete JSON Schema for the AiDotNet YAML configuration.
    /// </summary>
    /// <returns>A JSON string containing the full schema.</returns>
    public static string Generate()
    {
        var schema = new JObject
        {
            ["$schema"] = "https://json-schema.org/draft/2020-12/schema",
            ["title"] = "AiDotNet YAML Configuration",
            ["description"] = "Configuration schema for AiModelBuilder YAML files. Defines model training, deployment, and inference settings.",
            ["type"] = "object",
            ["additionalProperties"] = false,
        };

        var properties = new JObject();
        schema["properties"] = properties;

        // Add hand-written sections from YamlModelConfig.
        AddOptimizerSection(properties);
        AddTimeSeriesModelSection(properties);
        AddPocoConfigSections(properties);

        // Add all sections discovered by the source generator.
        AddGeneratedSections(properties);

        return schema.ToString(Formatting.Indented);
    }

    private static void AddOptimizerSection(JObject properties)
    {
        var optimizerTypes = GetEnumNames(typeof(Enums.OptimizerType));

        properties["optimizer"] = new JObject
        {
            ["type"] = "object",
            ["description"] = "Select the optimizer algorithm for model training.",
            ["additionalProperties"] = false,
            ["properties"] = new JObject
            {
                ["type"] = new JObject
                {
                    ["type"] = "string",
                    ["description"] = "The optimizer type name (case-insensitive).",
                    ["enum"] = new JArray(optimizerTypes.Select(n => (object)n).ToArray()),
                },
            },
        };
    }

    private static void AddTimeSeriesModelSection(JObject properties)
    {
        var modelTypes = GetEnumNames(typeof(Enums.TimeSeriesModelType));

        properties["timeSeriesModel"] = new JObject
        {
            ["type"] = "object",
            ["description"] = "Select the time series model type.",
            ["additionalProperties"] = false,
            ["properties"] = new JObject
            {
                ["type"] = new JObject
                {
                    ["type"] = "string",
                    ["description"] = "The time series model type name (case-insensitive).",
                    ["enum"] = new JArray(modelTypes.Select(n => (object)n).ToArray()),
                },
            },
        };
    }

    private static void AddPocoConfigSections(JObject properties)
    {
        // Hand-written POCO config sections in YamlModelConfig.cs.
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
            ("interpretability", "Interpretability configuration for model explainability (SHAP, LIME, etc.).", typeof(Models.Options.InterpretabilityOptions)),
            ("memoryManagement", "Training memory management configuration.", typeof(Training.Memory.TrainingMemoryConfig)),
        };

        foreach (var (yamlKey, description, configType) in pocoSections)
        {
            properties[yamlKey] = BuildPocoSchema(configType, description);
        }
    }

    private static void AddGeneratedSections(JObject properties)
    {
        foreach (var section in YamlSchemaMetadata.Sections)
        {
            // Skip hand-written sections (already added above).
            if (section.IsHandWritten) continue;

            // Skip the IFullModel "Model" section (handled by timeSeriesModel).
            if (section.SectionName == "Model") continue;

            var yamlKey = ToCamelCase(section.PropertyName);

            switch (section.Category)
            {
                case "Interface":
                case "AbstractGeneric":
                    properties[yamlKey] = BuildTypeSectionSchema(section);
                    break;

                case "Poco":
                    properties[yamlKey] = BuildPocoFromMetadata(section);
                    break;

                case "ConcreteGeneric":
                    // Generic POCOs use YamlTypeSection in YAML (type + params).
                    properties[yamlKey] = BuildGenericPocoSchema(section);
                    break;

                case "Pipeline":
                    properties[yamlKey] = BuildPipelineSchema(section);
                    break;

                default:
                    properties[yamlKey] = BuildTypeSectionSchema(section);
                    break;
            }
        }
    }

    private static JObject BuildTypeSectionSchema(YamlSectionMeta section)
    {
        var schema = new JObject
        {
            ["type"] = "object",
            ["description"] = $"Configuration for {section.SectionName}. Specify a type name and optional parameters.",
            ["additionalProperties"] = false,
            ["properties"] = new JObject
            {
                ["type"] = BuildTypePropertyWithEnum(section.SectionName),
                ["params"] = new JObject
                {
                    ["type"] = "object",
                    ["description"] = "Optional parameters to set on the created instance (property names are case-insensitive).",
                    ["additionalProperties"] = true,
                },
            },
        };

        return schema;
    }

    private static JObject BuildTypePropertyWithEnum(string sectionName)
    {
        var typeProp = new JObject
        {
            ["type"] = "string",
            ["description"] = $"The concrete implementation type name for {sectionName} (case-insensitive).",
        };

        if (YamlRegisteredTypeNames.SectionTypes.TryGetValue(sectionName, out var typeNames) && typeNames.Length > 0)
        {
            typeProp["enum"] = new JArray(typeNames.Select(n => (object)n).ToArray());
        }

        return typeProp;
    }

    private static JObject BuildGenericPocoSchema(YamlSectionMeta section)
    {
        return new JObject
        {
            ["type"] = "object",
            ["description"] = $"Configuration for {section.SectionName}. Set properties via the params dictionary.",
            ["additionalProperties"] = false,
            ["properties"] = new JObject
            {
                ["type"] = new JObject
                {
                    ["type"] = "string",
                    ["description"] = $"Optional type override for {section.SectionName}.",
                },
                ["params"] = new JObject
                {
                    ["type"] = "object",
                    ["description"] = "Properties to set on the configuration object (case-insensitive).",
                    ["additionalProperties"] = true,
                },
            },
        };
    }

    private static JObject BuildPipelineSchema(YamlSectionMeta section)
    {
        return new JObject
        {
            ["type"] = "object",
            ["description"] = $"Pipeline configuration for {section.SectionName}. Define ordered processing steps.",
            ["additionalProperties"] = false,
            ["properties"] = new JObject
            {
                ["steps"] = new JObject
                {
                    ["type"] = "array",
                    ["description"] = "Ordered list of processing steps.",
                    ["items"] = new JObject
                    {
                        ["type"] = "object",
                        ["properties"] = new JObject
                        {
                            ["type"] = new JObject
                            {
                                ["type"] = "string",
                                ["description"] = "The step implementation type name.",
                            },
                            ["params"] = new JObject
                            {
                                ["type"] = "object",
                                ["description"] = "Optional parameters for this step.",
                                ["additionalProperties"] = true,
                            },
                        },
                    },
                },
            },
        };
    }

    private static JObject BuildPocoFromMetadata(YamlSectionMeta section)
    {
        var props = new JObject();

        foreach (var propEntry in section.PocoProperties)
        {
            var parts = propEntry.Split(':');
            if (parts.Length == 2)
            {
                props[parts[0]] = new JObject
                {
                    ["type"] = parts[1],
                };
            }
        }

        return new JObject
        {
            ["type"] = "object",
            ["description"] = $"Configuration for {section.SectionName}.",
            ["additionalProperties"] = false,
            ["properties"] = props,
        };
    }

    private static JObject BuildPocoSchema(Type configType, string description)
    {
        var props = new JObject();

        foreach (var prop in configType.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (prop.GetGetMethod() is null || prop.GetSetMethod() is null) continue;

            var propSchema = new JObject
            {
                ["type"] = MapClrTypeToJsonSchema(prop.PropertyType),
            };

            // Add enum values for enum properties.
            var underlying = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;
            if (underlying.IsEnum)
            {
                propSchema["enum"] = new JArray(Enum.GetNames(underlying).Select(n => (object)n).ToArray());
            }

            props[ToCamelCase(prop.Name)] = propSchema;
        }

        return new JObject
        {
            ["type"] = "object",
            ["description"] = description,
            ["additionalProperties"] = false,
            ["properties"] = props,
        };
    }

    private static string MapClrTypeToJsonSchema(Type type)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;

        if (underlying == typeof(bool)) return "boolean";
        if (underlying == typeof(int) || underlying == typeof(long) || underlying == typeof(short)) return "integer";
        if (underlying == typeof(float) || underlying == typeof(double) || underlying == typeof(decimal)) return "number";
        if (underlying == typeof(string)) return "string";
        if (underlying.IsEnum) return "string";
        if (underlying.IsArray || (underlying.IsGenericType && underlying.GetGenericTypeDefinition() == typeof(List<>))) return "array";
        if (underlying.IsGenericType && underlying.GetGenericTypeDefinition() == typeof(Dictionary<,>)) return "object";
        return "object";
    }

    private static string[] GetEnumNames(Type enumType)
    {
        return Enum.GetNames(enumType);
    }

    private static string ToCamelCase(string name)
    {
        if (string.IsNullOrEmpty(name)) return name;
        return char.ToLowerInvariant(name[0]) + name.Substring(1);
    }
}
