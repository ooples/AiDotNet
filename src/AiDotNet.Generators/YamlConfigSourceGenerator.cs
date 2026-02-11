using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that discovers all Configure* methods on AiModelBuilder
/// and generates complete YAML configuration mapping code.
/// </summary>
/// <remarks>
/// <para>
/// This generator scans the compilation for the <c>AiModelBuilder</c> class, finds every
/// <c>Configure*()</c> method, and produces:
/// </para>
/// <list type="bullet">
/// <item><description>A partial <c>YamlModelConfig</c> with properties for every uncovered section.</description></item>
/// <item><description>A partial <c>YamlConfigApplier</c> that wires each new section to its <c>Configure*()</c> call.</description></item>
/// <item><description>A <c>YamlTypeRegistry</c> that maps type-name strings to concrete implementations for interface-based sections.</description></item>
/// <item><description>A <c>YamlParamsHelper</c> that applies <c>params</c> dictionaries to option objects via reflection.</description></item>
/// </list>
/// </remarks>
[Generator]
public class YamlConfigSourceGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Find class declarations named "AiModelBuilder" as a fast syntax filter.
        var builderDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => node is ClassDeclarationSyntax cds
                && cds.Identifier.Text == "AiModelBuilder",
            transform: static (ctx, _) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol)
            .Where(static s => s is not null)
            .Collect();

        // Combine with compilation so we can resolve types globally.
        var combined = builderDeclarations.Combine(context.CompilationProvider);

        context.RegisterSourceOutput(combined, static (spc, source) =>
        {
            var (builders, compilation) = source;
            Execute(spc, builders, compilation);
        });
    }

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> builders,
        Compilation compilation)
    {
        // Find the AiModelBuilder type (there should be exactly one).
        var builderType = builders.FirstOrDefault(b => b is not null
            && b.ContainingNamespace.ToDisplayString() == "AiDotNet");

        if (builderType is null) return;

        // Discover all Configure* methods.
        var configureMethods = DiscoverConfigureMethods(builderType);
        if (configureMethods.Count == 0) return;

        // Discover all concrete implementations for interface parameters.
        var sections = AnalyzeSections(configureMethods, compilation);

        // Emit helper types.
        EmitYamlTypeSection(context);
        EmitYamlPipelineSection(context);
        EmitYamlParamsHelper(context);

        // Emit the partial YamlModelConfig with generated properties.
        EmitYamlModelConfig(context, sections);

        // Emit the partial YamlConfigApplier with generated wiring.
        EmitYamlConfigApplier(context, sections);

        // Emit the type registry that maps string names to concrete types.
        EmitYamlTypeRegistry(context, sections);
    }

    // ───────────────────────────────────────────────────────────────
    // Discovery
    // ───────────────────────────────────────────────────────────────

    private static List<ConfigureMethodInfo> DiscoverConfigureMethods(INamedTypeSymbol builderType)
    {
        var methods = new List<ConfigureMethodInfo>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var member in builderType.GetMembers())
        {
            if (member is not IMethodSymbol method) continue;
            if (!method.Name.StartsWith("Configure", StringComparison.Ordinal)) continue;
            if (method.DeclaredAccessibility != Accessibility.Public) continue;
            if (method.IsStatic) continue;

            // Extract the section name from the method name (e.g., "ConfigureOptimizer" → "Optimizer").
            var sectionName = method.Name.Substring("Configure".Length);
            if (string.IsNullOrEmpty(sectionName)) continue;

            // Skip async overloads (e.g., ConfigureTokenizerFromPretrainedAsync).
            if (sectionName.EndsWith("Async", StringComparison.Ordinal)) continue;

            // Deduplicate overloads — take the first (simplest) overload per section name.
            if (!seen.Add(sectionName)) continue;

            // Get the first parameter (the main config/interface parameter).
            if (method.Parameters.Length == 0) continue;
            var firstParam = method.Parameters[0];
            var paramType = firstParam.Type;

            var info = new ConfigureMethodInfo
            {
                MethodName = method.Name,
                SectionName = sectionName,
                ParameterType = paramType,
                ParameterTypeName = paramType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                IsNullable = firstParam.NullableAnnotation == NullableAnnotation.Annotated
                    || (paramType is INamedTypeSymbol nts && nts.Name == "Nullable"),
            };

            // Categorize the parameter type.
            if (paramType.TypeKind == TypeKind.Interface)
            {
                info.Category = SectionCategory.Interface;
            }
            else if (paramType is INamedTypeSymbol named && named.Name == "Action" && named.TypeArguments.Length > 0)
            {
                info.Category = SectionCategory.ActionBuilder;
                info.ActionInnerType = named.TypeArguments[0];
            }
            else if (paramType.TypeKind == TypeKind.Class || paramType.TypeKind == TypeKind.Struct)
            {
                info.Category = SectionCategory.PocoConfig;
                info.IsAbstract = paramType.IsAbstract;

                // Check for parameterless constructor availability.
                if (paramType is INamedTypeSymbol namedParamType)
                {
                    info.HasParameterlessCtor = namedParamType.Constructors.Any(c =>
                        c.Parameters.Length == 0 && c.DeclaredAccessibility == Accessibility.Public) ||
                        !namedParamType.Constructors.Any(c =>
                            c.DeclaredAccessibility == Accessibility.Public && !c.IsImplicitlyDeclared);
                }
            }
            else
            {
                info.Category = SectionCategory.Unknown;
            }

            methods.Add(info);
        }

        return methods;
    }

    private static List<SectionInfo> AnalyzeSections(
        List<ConfigureMethodInfo> methods,
        Compilation compilation)
    {
        var sections = new List<SectionInfo>();

        foreach (var method in methods)
        {
            var section = new SectionInfo
            {
                Method = method,
                YamlPropertyName = ToCamelCase(method.SectionName),
            };

            if (method.Category == SectionCategory.Interface)
            {
                // Find all non-abstract, non-interface types that implement this interface.
                section.ConcreteImplementations = FindImplementations(method.ParameterType, compilation);
            }
            else if (method.Category == SectionCategory.PocoConfig && method.IsAbstract && ContainsTypeParameters(method.ParameterType))
            {
                // Abstract classes with generic type params need implementation lookup (subclass discovery).
                section.ConcreteImplementations = FindImplementations(method.ParameterType, compilation);
            }

            sections.Add(section);
        }

        return sections;
    }

    private static List<ImplementationInfo> FindImplementations(ITypeSymbol interfaceType, Compilation compilation)
    {
        var implementations = new List<ImplementationInfo>();
        var visitor = new ImplementationFinder(interfaceType, implementations);
        visitor.Visit(compilation.GlobalNamespace);
        return implementations;
    }

    // ───────────────────────────────────────────────────────────────
    // Emitters
    // ───────────────────────────────────────────────────────────────

    private static void EmitYamlTypeSection(SourceProductionContext context)
    {
        const string source = @"// <auto-generated/>
#nullable enable

using System.Collections.Generic;

namespace AiDotNet.Configuration;

/// <summary>
/// Generic YAML section for any type-based configuration.
/// Specifies the concrete type name and optional parameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Use this in YAML to pick a specific implementation
/// and pass its settings. For example:
/// <code>
/// optimizer:
///   type: ""Adam""
///   params:
///     initialLearningRate: 0.001
///     beta1: 0.9
/// </code>
/// </para>
/// </remarks>
public class YamlTypeSection
{
    /// <summary>
    /// The concrete type name (case-insensitive). Must match a registered implementation.
    /// </summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Optional dictionary of parameters to set on the created instance's options.
    /// Keys are property names (case-insensitive), values are the property values.
    /// </summary>
    public Dictionary<string, object> Params { get; set; } = new Dictionary<string, object>();
}
";
        context.AddSource("YamlTypeSection.g.cs", source);
    }

    private static void EmitYamlPipelineSection(SourceProductionContext context)
    {
        const string source = @"// <auto-generated/>
#nullable enable

using System.Collections.Generic;

namespace AiDotNet.Configuration;

/// <summary>
/// YAML section for pipeline-style configurations (preprocessing, postprocessing, etc.).
/// Contains an ordered list of steps, each with a type and optional parameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Use this in YAML to define a chain of processing steps:
/// <code>
/// preprocessing:
///   steps:
///     - type: ""StandardScaler""
///     - type: ""SimpleImputer""
///       params:
///         strategy: ""Mean""
/// </code>
/// </para>
/// </remarks>
public class YamlPipelineSection
{
    /// <summary>
    /// Ordered list of pipeline steps. Each step has a type and optional parameters.
    /// </summary>
    public List<YamlTypeSection> Steps { get; set; } = new List<YamlTypeSection>();
}
";
        context.AddSource("YamlPipelineSection.g.cs", source);
    }

    private static void EmitYamlParamsHelper(SourceProductionContext context)
    {
        const string source = @"// <auto-generated/>
#nullable enable

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Reflection;

namespace AiDotNet.Configuration;

/// <summary>
/// Applies a dictionary of YAML parameters to an object's properties via reflection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This helper takes the <c>params:</c> section from YAML
/// and sets the matching properties on the target object. Property names are matched
/// case-insensitively, and values are converted to the correct types automatically.</para>
/// </remarks>
internal static class YamlParamsHelper
{
    /// <summary>
    /// Sets properties on <paramref name=""target""/> from the key/value pairs in <paramref name=""parameters""/>.
    /// </summary>
    /// <param name=""target"">The object whose properties will be set.</param>
    /// <param name=""parameters"">Dictionary of property names to values.</param>
    internal static void ApplyParams(object target, Dictionary<string, object>? parameters)
    {
        if (target is null || parameters is null || parameters.Count == 0) return;

        var targetType = target.GetType();

        foreach (var kvp in parameters)
        {
            var prop = targetType.GetProperty(kvp.Key,
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);

            if (prop is null || !prop.CanWrite) continue;

            try
            {
                var value = ConvertValue(kvp.Value, prop.PropertyType);
                prop.SetValue(target, value);
            }
            catch (Exception)
            {
                // Skip properties that can't be converted — user may have a typo.
                // In debug builds this could log a warning.
            }
        }
    }

    private static object? ConvertValue(object? value, Type targetType)
    {
        if (value is null) return null;

        // Unwrap Nullable<T>
        var underlyingType = Nullable.GetUnderlyingType(targetType) ?? targetType;

        // Handle enums from string
        if (underlyingType.IsEnum && value is string enumString)
        {
            return Enum.Parse(underlyingType, enumString, ignoreCase: true);
        }

        // Handle TypeConverter for complex types
        var converter = TypeDescriptor.GetConverter(underlyingType);
        if (converter.CanConvertFrom(value.GetType()))
        {
            return converter.ConvertFrom(null, CultureInfo.InvariantCulture, value);
        }

        // Fallback to Convert.ChangeType
        return Convert.ChangeType(value, underlyingType, CultureInfo.InvariantCulture);
    }
}
";
        context.AddSource("YamlParamsHelper.g.cs", source);
    }

    private static void EmitYamlModelConfig(SourceProductionContext context, List<SectionInfo> sections)
    {
        // These sections are already hand-written in YamlModelConfig.cs — skip them.
        var existingSections = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "Optimizer", "Quantization", "Compression", "Caching", "Versioning",
            "ABTesting", "Telemetry", "Export", "GpuAcceleration", "Profiling",
            "JitCompilation", "MixedPrecision", "Reasoning", "Benchmarking",
            "InferenceOptimizations", "Interpretability", "MemoryManagement",
        };

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Configuration;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated partial class adding YAML properties for all Configure* methods.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("public partial class YamlModelConfig");
        sb.AppendLine("{");

        foreach (var section in sections)
        {
            if (existingSections.Contains(section.Method.SectionName)) continue;

            // Skip the TimeSeriesModel-specific one (already handled via YamlTimeSeriesModelSection).
            if (section.Method.SectionName == "Model" &&
                section.Method.ParameterTypeName.Contains("IFullModel")) continue;

            var propType = GetYamlPropertyType(section);
            var propName = ToPascalCase(section.Method.SectionName);

            sb.AppendLine($"    /// <summary>");
            sb.AppendLine($"    /// YAML configuration for {section.Method.MethodName}().");
            sb.AppendLine($"    /// </summary>");
            sb.AppendLine($"    public {propType}? {propName} {{ get; set; }}");
            sb.AppendLine();
        }

        sb.AppendLine("}");

        context.AddSource("YamlModelConfig.g.cs", sb.ToString());
    }

    private static void EmitYamlConfigApplier(SourceProductionContext context, List<SectionInfo> sections)
    {
        // Same skip list as above — these are already hand-written.
        var existingSections = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "Optimizer", "Quantization", "Compression", "Caching", "Versioning",
            "ABTesting", "Telemetry", "Export", "GpuAcceleration", "Profiling",
            "JitCompilation", "MixedPrecision", "Reasoning", "Benchmarking",
            "InferenceOptimizations", "Interpretability", "MemoryManagement",
        };

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Configuration;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated partial class with Apply wiring for all discovered Configure* methods.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static partial class YamlConfigApplier<T, TInput, TOutput>");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Applies auto-generated YAML sections to the builder.");
        sb.AppendLine("    /// Called from the hand-written Apply() method.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    internal static void ApplyGenerated(YamlModelConfig config, AiDotNet.AiModelBuilder<T, TInput, TOutput> builder)");
        sb.AppendLine("    {");

        foreach (var section in sections)
        {
            if (existingSections.Contains(section.Method.SectionName)) continue;
            if (section.Method.SectionName == "Model" &&
                section.Method.ParameterTypeName.Contains("IFullModel")) continue;

            var propName = ToPascalCase(section.Method.SectionName);

            var hasGenericParams = ContainsTypeParameters(section.Method.ParameterType);

            switch (section.Method.Category)
            {
                case SectionCategory.PocoConfig when !hasGenericParams:
                    // Non-generic POCO — pass directly from config.
                    sb.AppendLine($"        if (config.{propName} is not null)");
                    sb.AppendLine($"        {{");
                    sb.AppendLine($"            builder.{section.Method.MethodName}(config.{propName});");
                    sb.AppendLine($"        }}");
                    sb.AppendLine();
                    break;

                case SectionCategory.PocoConfig when hasGenericParams && section.Method.IsAbstract:
                    // Abstract generic class — use registry to find concrete subclass.
                    sb.AppendLine($"        if (config.{propName} is not null && !string.IsNullOrWhiteSpace(config.{propName}.Type))");
                    sb.AppendLine($"        {{");
                    sb.AppendLine($"            var instance = YamlTypeRegistry<T, TInput, TOutput>.CreateInstance<{section.Method.ParameterTypeName}>(");
                    sb.AppendLine($"                \"{section.Method.SectionName}\", config.{propName}.Type, config.{propName}.Params);");
                    sb.AppendLine($"            builder.{section.Method.MethodName}(instance);");
                    sb.AppendLine($"        }}");
                    sb.AppendLine();
                    break;

                case SectionCategory.PocoConfig when hasGenericParams && !section.Method.IsAbstract && section.Method.HasParameterlessCtor:
                {
                    // Concrete generic POCO with parameterless ctor — create instance directly, apply params.
                    var typeName = section.Method.ParameterType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                    sb.AppendLine($"        if (config.{propName} is not null)");
                    sb.AppendLine($"        {{");
                    sb.AppendLine($"            var instance = new {typeName}();");
                    sb.AppendLine($"            YamlParamsHelper.ApplyParams(instance, config.{propName}.Params);");
                    sb.AppendLine($"            builder.{section.Method.MethodName}(instance);");
                    sb.AppendLine($"        }}");
                    sb.AppendLine();
                    break;
                }

                case SectionCategory.PocoConfig when hasGenericParams && !section.Method.IsAbstract && !section.Method.HasParameterlessCtor:
                    // Concrete generic POCO without parameterless ctor — requires constructor args, skip for now.
                    sb.AppendLine($"        // TODO: {section.Method.MethodName} requires constructor parameters — manual YAML wiring needed.");
                    sb.AppendLine();
                    break;

                case SectionCategory.Interface:
                    sb.AppendLine($"        if (config.{propName} is not null && !string.IsNullOrWhiteSpace(config.{propName}.Type))");
                    sb.AppendLine($"        {{");
                    sb.AppendLine($"            var instance = YamlTypeRegistry<T, TInput, TOutput>.CreateInstance<{section.Method.ParameterTypeName}>(");
                    sb.AppendLine($"                \"{section.Method.SectionName}\", config.{propName}.Type, config.{propName}.Params);");
                    sb.AppendLine($"            builder.{section.Method.MethodName}(instance);");
                    sb.AppendLine($"        }}");
                    sb.AppendLine();
                    break;

                case SectionCategory.ActionBuilder:
                    // For Action<Pipeline> methods, we skip for now — they need step-by-step factory support.
                    sb.AppendLine($"        // TODO: {section.Method.MethodName} uses Action<> builder pattern — requires pipeline step factories.");
                    sb.AppendLine();
                    break;

                default:
                    sb.AppendLine($"        // TODO: {section.Method.MethodName} — unsupported parameter category.");
                    sb.AppendLine();
                    break;
            }
        }

        sb.AppendLine("    }");
        sb.AppendLine("}");

        context.AddSource("YamlConfigApplier.g.cs", sb.ToString());
    }

    private static void EmitYamlTypeRegistry(SourceProductionContext context, List<SectionInfo> sections)
    {
        // Include both interface sections and abstract generic POCO sections that have implementations.
        var interfaceSections = sections
            .Where(s => s.ConcreteImplementations.Count > 0 &&
                (s.Method.Category == SectionCategory.Interface ||
                 (s.Method.Category == SectionCategory.PocoConfig && s.Method.IsAbstract && ContainsTypeParameters(s.Method.ParameterType))))
            .ToList();

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Configuration;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated registry of all concrete implementations discovered for each interface-based Configure* method.");
        sb.AppendLine("/// Maps string type names (from YAML) to concrete Type objects.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class YamlTypeRegistry<T, TInput, TOutput>");
        sb.AppendLine("{");

        // Emit a static dictionary per interface section.
        foreach (var section in interfaceSections)
        {
            var dictName = $"_{ToCamelCase(section.Method.SectionName)}Types";
            sb.AppendLine($"    /// <summary>Registered implementations for {section.Method.SectionName}.</summary>");
            sb.AppendLine($"    private static readonly Dictionary<string, Type> {dictName} = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase)");
            sb.AppendLine($"    {{");

            foreach (var impl in section.ConcreteImplementations)
            {
                sb.AppendLine($"        {{ \"{impl.ShortName}\", typeof({impl.FullyQualifiedName}) }},");
            }

            sb.AppendLine($"    }};");
            sb.AppendLine();
        }

        // Emit a lookup dictionary that maps section names to their type dictionaries.
        sb.AppendLine("    private static readonly Dictionary<string, Dictionary<string, Type>> _sectionRegistries = new Dictionary<string, Dictionary<string, Type>>(StringComparer.OrdinalIgnoreCase)");
        sb.AppendLine("    {");
        foreach (var section in interfaceSections)
        {
            var dictName = $"_{ToCamelCase(section.Method.SectionName)}Types";
            sb.AppendLine($"        {{ \"{section.Method.SectionName}\", {dictName} }},");
        }
        sb.AppendLine("    };");
        sb.AppendLine();

        // Emit a CreateInstance method.
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Creates an instance of the specified type for the given YAML section.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    /// <typeparam name=\"TInterface\">The interface type expected by the Configure method.</typeparam>");
        sb.AppendLine("    /// <param name=\"sectionName\">The YAML section name (e.g., \"Regularization\").</param>");
        sb.AppendLine("    /// <param name=\"typeName\">The concrete type name from YAML (e.g., \"L2Regularization\").</param>");
        sb.AppendLine("    /// <param name=\"parameters\">Optional parameters to apply to the instance.</param>");
        sb.AppendLine("    /// <returns>The created and configured instance.</returns>");
        sb.AppendLine("    internal static TInterface CreateInstance<TInterface>(string sectionName, string typeName, Dictionary<string, object>? parameters = null)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (!_sectionRegistries.TryGetValue(sectionName, out var registry))");
        sb.AppendLine("        {");
        sb.AppendLine("            throw new ArgumentException($\"Unknown YAML section: '{sectionName}'.\");");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        if (!registry.TryGetValue(typeName, out var concreteType))");
        sb.AppendLine("        {");
        sb.AppendLine("            var available = string.Join(\", \", registry.Keys);");
        sb.AppendLine("            throw new ArgumentException($\"Unknown type '{typeName}' for section '{sectionName}'. Available types: {available}\");");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        var instance = Activator.CreateInstance(concreteType);");
        sb.AppendLine("        if (instance is null)");
        sb.AppendLine("        {");
        sb.AppendLine("            throw new InvalidOperationException($\"Failed to create instance of type '{concreteType.FullName}'.\");");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        YamlParamsHelper.ApplyParams(instance, parameters);");
        sb.AppendLine();
        sb.AppendLine("        if (instance is TInterface typed)");
        sb.AppendLine("        {");
        sb.AppendLine("            return typed;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        throw new InvalidCastException($\"Type '{concreteType.FullName}' does not implement '{typeof(TInterface).Name}'.\");");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Emit a GetRegisteredTypes method for schema/docs generation.
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Gets all registered type names for a given section. Used for JSON Schema and documentation generation.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    internal static IReadOnlyDictionary<string, Dictionary<string, Type>> GetAllRegistries() => _sectionRegistries;");

        sb.AppendLine("}");

        context.AddSource("YamlTypeRegistry.g.cs", sb.ToString());
    }

    // ───────────────────────────────────────────────────────────────
    // Helpers
    // ───────────────────────────────────────────────────────────────

    private static string GetYamlPropertyType(SectionInfo section)
    {
        return section.Method.Category switch
        {
            SectionCategory.Interface => "YamlTypeSection",
            SectionCategory.ActionBuilder => "YamlPipelineSection",
            SectionCategory.PocoConfig when ContainsTypeParameters(section.Method.ParameterType) => "YamlTypeSection",
            SectionCategory.PocoConfig => section.Method.ParameterType.ToDisplayString(
                SymbolDisplayFormat.FullyQualifiedFormat).Replace("global::", ""),
            _ => "YamlTypeSection",
        };
    }

    /// <summary>
    /// Checks if a type symbol contains unresolved generic type parameters (T, TInput, TOutput, etc.).
    /// These can't be used in the non-generic YamlModelConfig class.
    /// </summary>
    private static bool ContainsTypeParameters(ITypeSymbol type)
    {
        if (type.TypeKind == TypeKind.TypeParameter)
            return true;

        if (type is INamedTypeSymbol named && named.TypeArguments.Length > 0)
        {
            foreach (var arg in named.TypeArguments)
            {
                if (ContainsTypeParameters(arg))
                    return true;
            }
        }

        return false;
    }

    private static string ToCamelCase(string name)
    {
        if (string.IsNullOrEmpty(name)) return name;
        return char.ToLowerInvariant(name[0]) + name.Substring(1);
    }

    private static string ToPascalCase(string name)
    {
        if (string.IsNullOrEmpty(name)) return name;
        return char.ToUpperInvariant(name[0]) + name.Substring(1);
    }

    // ───────────────────────────────────────────────────────────────
    // Data models
    // ───────────────────────────────────────────────────────────────

    private enum SectionCategory
    {
        Unknown,
        Interface,
        PocoConfig,
        ActionBuilder,
    }

    private class ConfigureMethodInfo
    {
        public string MethodName { get; set; } = "";
        public string SectionName { get; set; } = "";
        public ITypeSymbol ParameterType { get; set; } = null!;
        public string ParameterTypeName { get; set; } = "";
        public bool IsNullable { get; set; }
        public bool IsAbstract { get; set; }
        public bool HasParameterlessCtor { get; set; }
        public SectionCategory Category { get; set; }
        public ITypeSymbol? ActionInnerType { get; set; }
    }

    private class SectionInfo
    {
        public ConfigureMethodInfo Method { get; set; } = null!;
        public string YamlPropertyName { get; set; } = "";
        public List<ImplementationInfo> ConcreteImplementations { get; set; } = new();
    }

    private class ImplementationInfo
    {
        public string ShortName { get; set; } = "";
        public string FullyQualifiedName { get; set; } = "";
    }

    // ───────────────────────────────────────────────────────────────
    // Implementation finder — walks namespaces to find concrete types
    // ───────────────────────────────────────────────────────────────

    private class ImplementationFinder : SymbolVisitor
    {
        private readonly ITypeSymbol _interfaceType;
        private readonly List<ImplementationInfo> _results;

        public ImplementationFinder(ITypeSymbol interfaceType, List<ImplementationInfo> results)
        {
            _interfaceType = interfaceType;
            _results = results;
        }

        public override void VisitNamespace(INamespaceSymbol symbol)
        {
            foreach (var member in symbol.GetMembers())
            {
                member.Accept(this);
            }
        }

        public override void VisitNamedType(INamedTypeSymbol symbol)
        {
            // Skip abstract, static, and interface types.
            if (symbol.IsAbstract || symbol.IsStatic || symbol.TypeKind == TypeKind.Interface)
            {
                // Still visit nested types even for abstract/static types.
                foreach (var nestedType in symbol.GetTypeMembers())
                {
                    nestedType.Accept(this);
                }
                return;
            }

            var interfaceOriginal = _interfaceType.OriginalDefinition;
            bool isMatch = false;

            // Check if this type implements the target interface.
            if (_interfaceType.TypeKind == TypeKind.Interface)
            {
                foreach (var iface in symbol.AllInterfaces)
                {
                    if (SymbolEqualityComparer.Default.Equals(iface.OriginalDefinition, interfaceOriginal))
                    {
                        isMatch = true;
                        break;
                    }
                }
            }

            // Check base class hierarchy (for abstract base class targets).
            if (!isMatch && _interfaceType.TypeKind == TypeKind.Class)
            {
                var baseType = symbol.BaseType;
                while (baseType is not null)
                {
                    if (SymbolEqualityComparer.Default.Equals(baseType.OriginalDefinition, interfaceOriginal))
                    {
                        isMatch = true;
                        break;
                    }
                    baseType = baseType.BaseType;
                }
            }

            if (isMatch)
            {
                // Only include types with a parameterless constructor (or no explicit constructors).
                var hasParameterlessCtor = symbol.Constructors.Any(c =>
                    c.Parameters.Length == 0 &&
                    c.DeclaredAccessibility == Accessibility.Public);

                // Also allow types with no explicitly declared constructors (default ctor).
                var hasDefaultCtor = !symbol.Constructors.Any(c =>
                    c.DeclaredAccessibility == Accessibility.Public &&
                    !c.IsImplicitlyDeclared);

                if (hasParameterlessCtor || hasDefaultCtor)
                {
                    _results.Add(new ImplementationInfo
                    {
                        ShortName = symbol.Name,
                        FullyQualifiedName = symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                            .Replace("global::", ""),
                    });
                }
            }

            // Visit nested types.
            foreach (var nestedType in symbol.GetTypeMembers())
            {
                nestedType.Accept(this);
            }
        }
    }
}
