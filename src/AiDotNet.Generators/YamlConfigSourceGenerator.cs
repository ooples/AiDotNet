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

        // Discover additional types marked with [YamlConfigurable] attribute.
        var attributeSections = DiscoverAttributeMarkedTypes(compilation, sections, builderType);
        sections.AddRange(attributeSections);

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

        // Emit the non-generic type names for JSON Schema generation.
        EmitYamlRegisteredTypeNames(context, sections);

        // Emit JSON Schema metadata for POCO sections.
        EmitYamlSchemaMetadata(context, sections, compilation);
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
                ParameterName = firstParam.Name,
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
            else if (method.Category == SectionCategory.PocoConfig && method.IsAbstract)
            {
                // Abstract classes (generic or non-generic) need implementation lookup (subclass discovery).
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

    /// <summary>
    /// Discovers interfaces and abstract classes marked with [YamlConfigurable("SectionName")]
    /// and returns SectionInfo objects for each, skipping sections already covered by Configure methods.
    /// </summary>
    private static List<SectionInfo> DiscoverAttributeMarkedTypes(
        Compilation compilation,
        List<SectionInfo> existingSections,
        INamedTypeSymbol builderType)
    {
        var results = new List<SectionInfo>();
        var existingNames = new HashSet<string>(
            existingSections.Select(s => s.Method.SectionName),
            StringComparer.OrdinalIgnoreCase);

        // Find the YamlConfigurableAttribute type in the compilation.
        var attrType = compilation.GetTypeByMetadataName("AiDotNet.Configuration.YamlConfigurableAttribute");
        if (attrType is null) return results;

        // Walk all types looking for the attribute.
        var visitor = new AttributeTypeFinder(attrType);
        visitor.Visit(compilation.GlobalNamespace);

        foreach (var (markedType, sectionName) in visitor.DiscoveredTypes)
        {
            // Skip if a Configure method already covers this section name.
            if (existingNames.Contains(sectionName))
            {
                // Collision case: a Configure-method section already claims this name. If that
                // section is a POCO/options section (no registry entries of its own) and the marked
                // type is an INTERFACE with concrete implementations, merge those implementations
                // onto the existing section so the type registry exposes them under this name — while
                // leaving the section's strongly-typed POCO config property intact. This is what lets
                // ConfigureAutoML(AutoMLOptions) (the "AutoML" POCO section) and IAutoMLModel
                // (marked [YamlConfigurable("AutoML")]) share one "AutoML" section that BOTH configures
                // via options AND resolves concrete IAutoMLModel implementations from YAML.
                if (markedType.TypeKind == TypeKind.Interface)
                {
                    var existing = existingSections.FirstOrDefault(s =>
                        string.Equals(s.Method.SectionName, sectionName, StringComparison.OrdinalIgnoreCase));
                    if (existing != null && existing.ConcreteImplementations.Count == 0)
                    {
                        var mergedImpls = FindImplementations(markedType, compilation);
                        if (mergedImpls.Count > 0)
                        {
                            existing.ConcreteImplementations = mergedImpls;
                            existing.Method.RegistryMerged = true;

                            // Only enable the applier's YAML `type:` -> CreateInstance branch when the
                            // builder actually exposes a Configure<Section>(interface) overload to receive
                            // the resolved implementation. Without it (e.g. a section that only has a POCO
                            // options overload), the merge stays registry-only: the impls are discoverable
                            // in schema/docs but the applier keeps using the options path, so we don't emit
                            // a call to a non-existent overload.
                            if (HasConfigureOverloadAccepting(builderType, sectionName, markedType))
                            {
                                existing.Method.MergedInterfaceTypeName =
                                    markedType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                            }
                        }
                    }
                }
                continue;
            }

            var info = new ConfigureMethodInfo
            {
                MethodName = "",
                SectionName = sectionName,
                ParameterType = markedType,
                ParameterTypeName = markedType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                IsNullable = false,
                IsAbstract = markedType.TypeKind == TypeKind.Interface || markedType.IsAbstract,
                IsAttributeDiscovered = true,
            };

            if (markedType.TypeKind == TypeKind.Interface)
            {
                info.Category = SectionCategory.Interface;
            }
            else if (markedType.TypeKind == TypeKind.Class)
            {
                info.Category = SectionCategory.PocoConfig;
            }
            else
            {
                info.Category = SectionCategory.Unknown;
            }

            var section = new SectionInfo
            {
                Method = info,
                YamlPropertyName = ToCamelCase(sectionName),
                ConcreteImplementations = markedType.TypeKind == TypeKind.Class && !markedType.IsAbstract
                    ? GetSelfImplementationIfRegisterable(markedType)
                    : FindImplementations(markedType, compilation),
            };

            // Only add if implementations were found.
            if (section.ConcreteImplementations.Count > 0)
            {
                existingNames.Add(sectionName);
                results.Add(section);
            }
        }

        return results;
    }

    /// <summary>
    /// Returns true if <paramref name="builderType"/> exposes a public instance method
    /// <c>Configure&lt;sectionName&gt;</c> whose first parameter can receive an instance of
    /// <paramref name="interfaceType"/> (the parameter is that interface, or implements it). Scans the
    /// RAW member list (not the deduped Configure-method set) so an interface overload that shares a
    /// section name with a POCO options overload is still found.
    /// </summary>
    private static bool HasConfigureOverloadAccepting(
        INamedTypeSymbol builderType, string sectionName, ITypeSymbol interfaceType)
    {
        var methodName = "Configure" + sectionName;
        foreach (var member in builderType.GetMembers())
        {
            if (member is not IMethodSymbol method) continue;
            if (method.IsStatic || method.DeclaredAccessibility != Accessibility.Public) continue;
            if (!string.Equals(method.Name, methodName, StringComparison.Ordinal)) continue;
            if (method.Parameters.Length == 0) continue;

            // The applier emits `builder.Configure<Section>(instance)` where `instance` is typed
            // as `interfaceType`. That single-argument call only binds when every trailing parameter
            // is optional (or params), so reject overloads with required trailing parameters.
            if (HasRequiredTrailingParameters(method)) continue;

            // For `builder.Configure<Section>(instance)` to compile, the first parameter must be
            // assignable FROM `interfaceType` — i.e. it is `interfaceType` itself or a base interface
            // that `interfaceType` implements. (Checking `paramType.AllInterfaces` would be the wrong
            // direction: it would also accept a MORE-derived concrete parameter that an
            // `interfaceType`-typed argument cannot satisfy.)
            var paramType = method.Parameters[0].Type;
            if (SymbolEqualityComparer.Default.Equals(paramType.OriginalDefinition, interfaceType.OriginalDefinition))
                return true;
            if (interfaceType.AllInterfaces.Any(i =>
                    SymbolEqualityComparer.Default.Equals(i.OriginalDefinition, paramType.OriginalDefinition)))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Returns true if <paramref name="method"/> has any parameter after the first that is neither
    /// optional nor a <c>params</c> array — i.e. a single-argument invocation of it would not compile.
    /// </summary>
    private static bool HasRequiredTrailingParameters(IMethodSymbol method)
    {
        for (var i = 1; i < method.Parameters.Length; i++)
        {
            var p = method.Parameters[i];
            if (!p.IsOptional && !p.IsParams)
                return true;
        }
        return false;
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
            "TimeSeriesModel",
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

            var docText = section.Method.IsAttributeDiscovered
                ? $"YAML configuration for the {section.Method.SectionName} section."
                : $"YAML configuration for {section.Method.MethodName}().";

            sb.AppendLine($"    /// <summary>");
            sb.AppendLine($"    /// {docText}");
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
            "TimeSeriesModel",
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

            // Attribute-discovered sections have no Configure method — registry-only.
            if (section.Method.IsAttributeDiscovered) continue;

            var propName = ToPascalCase(section.Method.SectionName);

            var hasGenericParams = ContainsTypeParameters(section.Method.ParameterType);

            // Merged POCO+interface section (e.g. AutoML): a YAML `type:` selects a concrete
            // implementation of the merged interface through the type registry and passes it to the
            // interface overload of the Configure method; otherwise the POCO options object is built
            // from `params:` and passed to the options overload. This makes the registry entries the
            // merge exposes genuinely instantiable from YAML, not just discoverable in schema/docs.
            if (section.Method.RegistryMerged && section.Method.MergedInterfaceTypeName is not null)
            {
                var mergedPocoType = section.Method.ParameterType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                sb.AppendLine($"        if (config.{propName} is not null)");
                sb.AppendLine($"        {{");
                sb.AppendLine($"            if (!string.IsNullOrWhiteSpace(config.{propName}.Type))");
                sb.AppendLine($"            {{");
                sb.AppendLine($"                var instance = YamlTypeRegistry<T, TInput, TOutput>.CreateInstance<{section.Method.MergedInterfaceTypeName}>(");
                sb.AppendLine($"                    \"{section.Method.SectionName}\", config.{propName}.Type, config.{propName}.Params);");
                sb.AppendLine($"                builder.{section.Method.MethodName}(instance);");
                sb.AppendLine($"            }}");
                sb.AppendLine($"            else");
                sb.AppendLine($"            {{");
                sb.AppendLine($"                var options = new {mergedPocoType}();");
                sb.AppendLine($"                YamlParamsHelper.ApplyParams(options, config.{propName}.Params);");
                sb.AppendLine($"                builder.{section.Method.MethodName}(options);");
                sb.AppendLine($"            }}");
                sb.AppendLine($"        }}");
                sb.AppendLine();
                continue;
            }

            switch (section.Method.Category)
            {
                case SectionCategory.PocoConfig when !hasGenericParams && section.Method.IsAbstract:
                    // Non-generic abstract class — use registry to find concrete subclass (e.g., ModelOptions).
                    sb.AppendLine($"        if (config.{propName} is not null && !string.IsNullOrWhiteSpace(config.{propName}.Type))");
                    sb.AppendLine($"        {{");
                    sb.AppendLine($"            var instance = YamlTypeRegistry<T, TInput, TOutput>.CreateInstance<{section.Method.ParameterTypeName}>(");
                    sb.AppendLine($"                \"{section.Method.SectionName}\", config.{propName}.Type, config.{propName}.Params);");
                    sb.AppendLine($"            builder.{section.Method.MethodName}(instance);");
                    sb.AppendLine($"        }}");
                    sb.AppendLine();
                    break;

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
        // Include both interface sections and abstract POCO sections (generic or non-generic) that have implementations.
        var interfaceSections = sections
            .Where(s => s.ConcreteImplementations.Count > 0 &&
                (s.Method.Category == SectionCategory.Interface ||
                 (s.Method.Category == SectionCategory.PocoConfig && s.Method.IsAbstract) ||
                 (s.Method.Category == SectionCategory.PocoConfig && s.Method.IsAttributeDiscovered) ||
                 s.Method.RegistryMerged))
            .ToList();

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine("using System.Globalization;");
        sb.AppendLine("using System.Linq;");
        sb.AppendLine("using System.Reflection;");
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

            var seenNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var impl in section.ConcreteImplementations)
            {
                if (seenNames.Add(impl.ShortName))
                {
                    sb.AppendLine($"        {{ \"{impl.ShortName}\", typeof({impl.FullyQualifiedName}) }},");
                }
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
        sb.AppendLine("        var instance = CreateWithBestConstructor(concreteType, parameters);");
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
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Creates an instance using the best matching constructor. Tries parameterless first,");
        sb.AppendLine("    /// then falls back to constructors whose parameters can be resolved from YAML params");
        sb.AppendLine("    /// or created with default values.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    private static object? CreateWithBestConstructor(Type concreteType, Dictionary<string, object>? parameters)");
        sb.AppendLine("    {");
        sb.AppendLine("        var constructors = concreteType.GetConstructors(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);");
        sb.AppendLine();
        sb.AppendLine("        // Try parameterless constructor first.");
        sb.AppendLine("        foreach (var ctor in constructors)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (ctor.GetParameters().Length == 0)");
        sb.AppendLine("            {");
        sb.AppendLine("                return ctor.Invoke(Array.Empty<object>());");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        // Sort constructors: fewest parameters first for simplest match.");
        sb.AppendLine("        var sortedCtors = constructors.OrderBy(c => c.GetParameters().Length).ToArray();");
        sb.AppendLine();
        sb.AppendLine("        foreach (var ctor in sortedCtors)");
        sb.AppendLine("        {");
        sb.AppendLine("            var ctorParams = ctor.GetParameters();");
        sb.AppendLine("            var args = new object?[ctorParams.Length];");
        sb.AppendLine("            var canResolve = true;");
        sb.AppendLine();
        sb.AppendLine("            for (int i = 0; i < ctorParams.Length; i++)");
        sb.AppendLine("            {");
        sb.AppendLine("                var param = ctorParams[i];");
        sb.AppendLine("                var paramType = param.ParameterType;");
        sb.AppendLine();
        sb.AppendLine("                // Check if YAML params has a matching value by name.");
        sb.AppendLine("                if (parameters is not null && TryResolveFromParams(parameters, param.Name ?? \"\", paramType, out var resolved))");
        sb.AppendLine("                {");
        sb.AppendLine("                    args[i] = resolved;");
        sb.AppendLine("                }");
        sb.AppendLine("                else if (param.HasDefaultValue)");
        sb.AppendLine("                {");
        sb.AppendLine("                    args[i] = param.DefaultValue;");
        sb.AppendLine("                }");
        sb.AppendLine("                else if (paramType.IsValueType)");
        sb.AppendLine("                {");
        sb.AppendLine("                    args[i] = Activator.CreateInstance(paramType);");
        sb.AppendLine("                }");
        sb.AppendLine("                else if (IsNullableParam(param))");
        sb.AppendLine("                {");
        sb.AppendLine("                    args[i] = null;");
        sb.AppendLine("                }");
        sb.AppendLine("                else");
        sb.AppendLine("                {");
        sb.AppendLine("                    // Try to create the parameter type (e.g., an Options object).");
        sb.AppendLine("                    try");
        sb.AppendLine("                    {");
        sb.AppendLine("                        var paramInstance = Activator.CreateInstance(paramType);");
        sb.AppendLine("                        if (paramInstance is not null)");
        sb.AppendLine("                        {");
        sb.AppendLine("                            YamlParamsHelper.ApplyParams(paramInstance, parameters);");
        sb.AppendLine("                            args[i] = paramInstance;");
        sb.AppendLine("                        }");
        sb.AppendLine("                        else");
        sb.AppendLine("                        {");
        sb.AppendLine("                            canResolve = false;");
        sb.AppendLine("                            break;");
        sb.AppendLine("                        }");
        sb.AppendLine("                    }");
        sb.AppendLine("                    catch");
        sb.AppendLine("                    {");
        sb.AppendLine("                        canResolve = false;");
        sb.AppendLine("                        break;");
        sb.AppendLine("                    }");
        sb.AppendLine("                }");
        sb.AppendLine("            }");
        sb.AppendLine();
        sb.AppendLine("            if (canResolve)");
        sb.AppendLine("            {");
        sb.AppendLine("                try");
        sb.AppendLine("                {");
        sb.AppendLine("                    return ctor.Invoke(args);");
        sb.AppendLine("                }");
        sb.AppendLine("                catch");
        sb.AppendLine("                {");
        sb.AppendLine("                    // Try next constructor.");
        sb.AppendLine("                }");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        // Last resort: try Activator with no args (may throw).");
        sb.AppendLine("        return Activator.CreateInstance(concreteType);");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    private static bool TryResolveFromParams(Dictionary<string, object> parameters, string paramName, Type paramType, out object? result)");
        sb.AppendLine("    {");
        sb.AppendLine("        result = null;");
        sb.AppendLine("        foreach (var kvp in parameters)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (string.Equals(kvp.Key, paramName, StringComparison.OrdinalIgnoreCase))");
        sb.AppendLine("            {");
        sb.AppendLine("                try");
        sb.AppendLine("                {");
        sb.AppendLine("                    var underlyingType = Nullable.GetUnderlyingType(paramType) ?? paramType;");
        sb.AppendLine("                    if (underlyingType.IsEnum && kvp.Value is string enumStr)");
        sb.AppendLine("                    {");
        sb.AppendLine("                        result = Enum.Parse(underlyingType, enumStr, ignoreCase: true);");
        sb.AppendLine("                        return true;");
        sb.AppendLine("                    }");
        sb.AppendLine("                    result = Convert.ChangeType(kvp.Value, underlyingType, System.Globalization.CultureInfo.InvariantCulture);");
        sb.AppendLine("                    return true;");
        sb.AppendLine("                }");
        sb.AppendLine("                catch");
        sb.AppendLine("                {");
        sb.AppendLine("                    return false;");
        sb.AppendLine("                }");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return false;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    private static bool IsNullableParam(System.Reflection.ParameterInfo param)");
        sb.AppendLine("    {");
        sb.AppendLine("        // Check for nullable reference type or Nullable<T>.");
        sb.AppendLine("        if (Nullable.GetUnderlyingType(param.ParameterType) is not null) return true;");
        sb.AppendLine("        var nullableAttr = param.GetCustomAttributes(true)");
        sb.AppendLine("            .FirstOrDefault(a => a.GetType().FullName == \"System.Runtime.CompilerServices.NullableAttribute\");");
        sb.AppendLine("        if (nullableAttr is not null) return true;");
        sb.AppendLine("        // If the parameter has a nullable annotation in metadata.");
        sb.AppendLine("        return !param.ParameterType.IsValueType;");
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

    private static void EmitYamlRegisteredTypeNames(SourceProductionContext context, List<SectionInfo> sections)
    {
        // Collect interface sections and abstract POCOs (generic or non-generic) that have implementations.
        var registrySections = sections
            .Where(s => s.ConcreteImplementations.Count > 0 &&
                (s.Method.Category == SectionCategory.Interface ||
                 (s.Method.Category == SectionCategory.PocoConfig && s.Method.IsAbstract) ||
                 (s.Method.Category == SectionCategory.PocoConfig && s.Method.IsAttributeDiscovered) ||
                 s.Method.RegistryMerged))
            .ToList();

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Configuration;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Non-generic registry of type name strings per YAML section.");
        sb.AppendLine("/// Used for JSON Schema generation and documentation without requiring generic type arguments.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class YamlRegisteredTypeNames");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Maps section names to arrays of available type names for that section.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    internal static readonly Dictionary<string, string[]> SectionTypes = new Dictionary<string, string[]>()");
        sb.AppendLine("    {");

        foreach (var section in registrySections)
        {
            var uniqueNames = section.ConcreteImplementations
                .Select(i => i.ShortName)
                .Distinct(StringComparer.OrdinalIgnoreCase);
            var names = string.Join(", ", uniqueNames.Select(n => $"\"{n}\""));
            sb.AppendLine($"        {{ \"{section.Method.SectionName}\", new[] {{ {names} }} }},");
        }

        sb.AppendLine("    };");
        sb.AppendLine("}");

        context.AddSource("YamlRegisteredTypeNames.g.cs", sb.ToString());
    }

    private static void EmitYamlSchemaMetadata(SourceProductionContext context, List<SectionInfo> sections, Compilation compilation)
    {
        // These sections are handled by hand in YamlModelConfig.cs
        var existingSections = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "Optimizer", "Quantization", "Compression", "Caching", "Versioning",
            "ABTesting", "Telemetry", "Export", "GpuAcceleration", "Profiling",
            "JitCompilation", "MixedPrecision", "Reasoning", "Benchmarking",
            "InferenceOptimizations", "Interpretability", "MemoryManagement",
            "TimeSeriesModel",
        };

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Configuration;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Compile-time metadata about each YAML section, used for JSON Schema and docs generation.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class YamlSchemaMetadata");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Metadata for each discovered YAML section.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    internal static readonly List<YamlSectionMeta> Sections = new List<YamlSectionMeta>()");
        sb.AppendLine("    {");

        foreach (var section in sections)
        {
            if (section.Method.SectionName == "Model" &&
                section.Method.ParameterTypeName.Contains("IFullModel")) continue;

            var category = section.Method.Category switch
            {
                SectionCategory.Interface => "Interface",
                SectionCategory.PocoConfig when section.Method.IsAttributeDiscovered && section.ConcreteImplementations.Count > 0 => "Interface",
                SectionCategory.PocoConfig when section.Method.IsAbstract && !ContainsTypeParameters(section.Method.ParameterType) => "AbstractNonGeneric",
                SectionCategory.PocoConfig when section.Method.IsAbstract && ContainsTypeParameters(section.Method.ParameterType) => "AbstractGeneric",
                SectionCategory.PocoConfig when ContainsTypeParameters(section.Method.ParameterType) => "ConcreteGeneric",
                SectionCategory.PocoConfig => "Poco",
                SectionCategory.ActionBuilder => "Pipeline",
                _ => "Unknown",
            };

            var propName = ToPascalCase(section.Method.SectionName);
            var isExisting = existingSections.Contains(section.Method.SectionName);

            // For POCO types (non-abstract), extract public property names and types.
            var pocoProps = new List<string>();
            if (section.Method.Category == SectionCategory.PocoConfig && !ContainsTypeParameters(section.Method.ParameterType) && !section.Method.IsAbstract)
            {
                if (IsScalarYamlParameter(section.Method.ParameterType))
                {
                    var paramName = string.IsNullOrWhiteSpace(section.Method.ParameterName)
                        ? ToCamelCase(section.Method.SectionName)
                        : ToCamelCase(section.Method.ParameterName);
                    var jsonType = GetJsonSchemaType(section.Method.ParameterType);
                    pocoProps.Add($"\"{paramName}:{jsonType}\"");
                }
                else if (section.Method.ParameterType is INamedTypeSymbol namedType)
                {
                    foreach (var member in namedType.GetMembers())
                    {
                        if (member is IPropertySymbol prop &&
                            prop.DeclaredAccessibility == Accessibility.Public &&
                            prop.GetMethod is not null &&
                            prop.SetMethod is not null &&
                            !prop.IsStatic)
                        {
                            var jsonType = GetJsonSchemaType(prop.Type);
                            pocoProps.Add($"\"{ToCamelCase(prop.Name)}:{jsonType}\"");
                        }
                    }
                }
            }

            var propsArray = pocoProps.Count > 0
                ? $"new[] {{ {string.Join(", ", pocoProps)} }}"
                : "System.Array.Empty<string>()";

            sb.AppendLine($"        new YamlSectionMeta(\"{section.Method.SectionName}\", \"{propName}\", \"{category}\", {(isExisting ? "true" : "false")}, {propsArray}),");
        }

        sb.AppendLine("    };");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Metadata about a single YAML configuration section.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class YamlSectionMeta");
        sb.AppendLine("{");
        sb.AppendLine("    internal YamlSectionMeta(string sectionName, string propertyName, string category, bool isHandWritten, string[] pocoProperties)");
        sb.AppendLine("    {");
        sb.AppendLine("        SectionName = sectionName;");
        sb.AppendLine("        PropertyName = propertyName;");
        sb.AppendLine("        Category = category;");
        sb.AppendLine("        IsHandWritten = isHandWritten;");
        sb.AppendLine("        PocoProperties = pocoProperties;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>The section name (e.g., \"Optimizer\").</summary>");
        sb.AppendLine("    internal string SectionName { get; }");
        sb.AppendLine("    /// <summary>The property name in YamlModelConfig (e.g., \"Optimizer\").</summary>");
        sb.AppendLine("    internal string PropertyName { get; }");
        sb.AppendLine("    /// <summary>Category: Interface, Poco, ConcreteGeneric, AbstractGeneric, Pipeline, Unknown.</summary>");
        sb.AppendLine("    internal string Category { get; }");
        sb.AppendLine("    /// <summary>Whether this section is hand-written in YamlModelConfig.cs.</summary>");
        sb.AppendLine("    internal bool IsHandWritten { get; }");
        sb.AppendLine("    /// <summary>For Poco sections: \"propName:jsonType\" entries for each public property.</summary>");
        sb.AppendLine("    internal string[] PocoProperties { get; }");
        sb.AppendLine("}");

        context.AddSource("YamlSchemaMetadata.g.cs", sb.ToString());
    }

    /// <summary>
    /// Maps a Roslyn type symbol to a JSON Schema type string.
    /// </summary>
    private static string GetJsonSchemaType(ITypeSymbol type)
    {
        // Unwrap nullable
        if (type is INamedTypeSymbol { Name: "Nullable", TypeArguments.Length: 1 } nullable)
        {
            type = nullable.TypeArguments[0];
        }

        var name = type.SpecialType switch
        {
            SpecialType.System_Boolean => "boolean",
            SpecialType.System_Int32 => "integer",
            SpecialType.System_Int64 => "integer",
            SpecialType.System_Single => "number",
            SpecialType.System_Double => "number",
            SpecialType.System_String => "string",
            _ => null,
        };

        if (name is not null) return name;

        // Check for enum
        if (type.TypeKind == TypeKind.Enum) return "string";

        // Check for Dictionary
        if (type is INamedTypeSymbol { Name: "Dictionary" }) return "object";

        // Check for List/Array
        if (type is IArrayTypeSymbol || (type is INamedTypeSymbol { Name: "List" })) return "array";

        return "object";
    }

    private static bool IsScalarYamlParameter(ITypeSymbol type)
    {
        if (type is INamedTypeSymbol { Name: "Nullable", TypeArguments.Length: 1 } nullable)
        {
            type = nullable.TypeArguments[0];
        }

        if (type.TypeKind == TypeKind.Enum) return true;

        return type.SpecialType is SpecialType.System_Boolean
            or SpecialType.System_Int32
            or SpecialType.System_Int64
            or SpecialType.System_Single
            or SpecialType.System_Double
            or SpecialType.System_String;
    }

    private static List<ImplementationInfo> GetSelfImplementationIfRegisterable(INamedTypeSymbol symbol)
    {
        if (!IsEffectivelyPublicForGeneratedCode(symbol) ||
            !HasOnlyResolvableTypeParametersForRegistry(symbol))
        {
            return new List<ImplementationInfo>();
        }

        var hasPublicCtor = symbol.Constructors.Any(c =>
            c.DeclaredAccessibility == Accessibility.Public &&
            !c.IsImplicitlyDeclared);
        var hasImplicitDefaultCtor = symbol.Constructors.Any(c =>
            c.IsImplicitlyDeclared &&
            c.DeclaredAccessibility == Accessibility.Public);

        if (!hasPublicCtor && !hasImplicitDefaultCtor)
        {
            return new List<ImplementationInfo>();
        }

        return new List<ImplementationInfo>
        {
            new ImplementationInfo
            {
                ShortName = symbol.Name,
                FullyQualifiedName = symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                    .Replace("global::", ""),
            },
        };
    }

    private static bool IsEffectivelyPublicForGeneratedCode(INamedTypeSymbol symbol)
    {
        for (INamedTypeSymbol? current = symbol; current is not null; current = current.ContainingType)
        {
            if (current.DeclaredAccessibility != Accessibility.Public)
            {
                return false;
            }
        }

        return true;
    }

    private static bool HasOnlyResolvableTypeParametersForRegistry(INamedTypeSymbol symbol)
    {
        var allowed = new HashSet<string>(StringComparer.Ordinal) { "T", "TInput", "TOutput" };
        var allTypeParams = new List<ITypeParameterSymbol>();
        var current = symbol;
        while (current is not null)
        {
            allTypeParams.AddRange(current.TypeParameters);
            current = current.ContainingType;
        }

        foreach (var tp in allTypeParams)
        {
            if (!allowed.Contains(tp.Name))
            {
                return false;
            }
        }

        return true;
    }

    // ───────────────────────────────────────────────────────────────
    // Helpers
    // ───────────────────────────────────────────────────────────────

    private static string GetYamlPropertyType(SectionInfo section)
    {
        // A merged POCO+interface section whose builder exposes the interface overload emits a
        // `type:`/`params:` applier branch (see EmitYamlConfigApplier), so the YAML property MUST be
        // a YamlTypeSection regardless of the POCO's own shape. Otherwise a non-generic merged POCO
        // would surface its concrete type here and the applier's `config.<Section>.Type` access
        // would not compile. This mirrors the applier's `RegistryMerged && MergedInterfaceTypeName`
        // guard exactly; registry-only merges (no interface overload) keep the POCO path below.
        if (section.Method.RegistryMerged && section.Method.MergedInterfaceTypeName is not null)
        {
            return "YamlTypeSection";
        }

        return section.Method.Category switch
        {
            SectionCategory.Interface => "YamlTypeSection",
            SectionCategory.ActionBuilder => "YamlPipelineSection",
            SectionCategory.PocoConfig when section.Method.IsAbstract => "YamlTypeSection",
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
        public ITypeSymbol? ParameterType { get; set; }
        public string ParameterName { get; set; } = "";
        public string ParameterTypeName { get; set; } = "";
        public bool IsNullable { get; set; }
        public bool IsAbstract { get; set; }
        public bool HasParameterlessCtor { get; set; }
        public SectionCategory Category { get; set; }
        public ITypeSymbol? ActionInnerType { get; set; }
        public bool IsAttributeDiscovered { get; set; }

        /// <summary>
        /// Set when a [YamlConfigurable]-marked INTERFACE's section name collides with an existing
        /// POCO/options Configure section (e.g. ConfigureAutoML(AutoMLOptions) claims "AutoML" while
        /// IAutoMLModel is marked [YamlConfigurable("AutoML")]). The interface's concrete
        /// implementations are merged onto the existing section so the TYPE REGISTRY exposes them
        /// under this name — WITHOUT flipping <see cref="IsAttributeDiscovered"/>, which would drop
        /// the section's strongly-typed POCO config property from YamlModelConfig (see EmitYamlModelConfig).
        /// </summary>
        public bool RegistryMerged { get; set; }

        /// <summary>
        /// For a <see cref="RegistryMerged"/> section, the fully-qualified name of the merged
        /// [YamlConfigurable] INTERFACE (e.g. <c>global::AiDotNet.Interfaces.IAutoMLModel&lt;T,TInput,TOutput&gt;</c>).
        /// The applier uses it to build a <c>CreateInstance&lt;interface&gt;()</c> branch so a YAML
        /// <c>type:</c> on this section resolves a concrete implementation, alongside the POCO options path.
        /// </summary>
        public string? MergedInterfaceTypeName { get; set; }
    }

    private class SectionInfo
    {
        public ConfigureMethodInfo? Method { get; set; }
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
                // Register all concrete implementations that have at least one public constructor.
                var hasPublicCtor = symbol.Constructors.Any(c =>
                    c.DeclaredAccessibility == Accessibility.Public &&
                    !c.IsImplicitlyDeclared);

                // Also allow types with a compiler-generated public default constructor
                // (i.e., no explicitly declared constructors at all).
                var hasImplicitDefaultCtor = symbol.Constructors.Any(c =>
                    c.IsImplicitlyDeclared &&
                    c.DeclaredAccessibility == Accessibility.Public);

                // Skip types that aren't publicly accessible from generated code. This must check the
                // FULL containment chain, not just nested types: top-level INTERNAL types in referenced
                // assemblies (e.g. FormattedLogValues : IReadOnlyList<...> in Microsoft.Extensions.Logging)
                // match broad BCL interfaces like IReadOnlyList<T> and would be emitted as typeof(...)
                // references the generated registry cannot compile against (CS0122).
                if (!IsEffectivelyPublic(symbol))
                {
                    hasPublicCtor = false;
                    hasImplicitDefaultCtor = false;
                }

                // Skip types with generic parameters that can't be resolved from <T, TInput, TOutput>.
                if (hasPublicCtor || hasImplicitDefaultCtor)
                {
                    if (!HasOnlyResolvableTypeParameters(symbol))
                    {
                        hasPublicCtor = false;
                        hasImplicitDefaultCtor = false;
                    }
                }

                if (hasPublicCtor || hasImplicitDefaultCtor)
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

        /// <summary>
        /// True when the type and every containing type is declared public — i.e. generated code in
        /// another assembly can legally reference it via <c>typeof(...)</c>.
        /// </summary>
        private static bool IsEffectivelyPublic(INamedTypeSymbol symbol)
        {
            for (INamedTypeSymbol? current = symbol; current is not null; current = current.ContainingType)
            {
                if (current.DeclaredAccessibility != Accessibility.Public)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Checks that all type parameters on the given symbol can be resolved from the
        /// registry's available type parameters: T, TInput, TOutput.
        /// Types like <c>StratifiedKFoldCrossValidator&lt;T, TInput, TOutput, TMetadata&gt;</c>
        /// have extra type parameters that can't be resolved and must be excluded.
        /// </summary>
        private static bool HasOnlyResolvableTypeParameters(INamedTypeSymbol symbol)
        {
            // The allowed type parameter names that the YamlTypeRegistry<T, TInput, TOutput> provides.
            var allowed = new HashSet<string>(StringComparer.Ordinal) { "T", "TInput", "TOutput" };

            // Collect all type parameters from the symbol and its containing types.
            var allTypeParams = new List<ITypeParameterSymbol>();
            var current = symbol;
            while (current is not null)
            {
                allTypeParams.AddRange(current.TypeParameters);
                current = current.ContainingType;
            }

            foreach (var tp in allTypeParams)
            {
                if (!allowed.Contains(tp.Name))
                {
                    return false;
                }
            }

            return true;
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Attribute finder — discovers [YamlConfigurable] marked types
    // ───────────────────────────────────────────────────────────────

    private class AttributeTypeFinder : SymbolVisitor
    {
        private readonly INamedTypeSymbol _attributeType;
        public List<(INamedTypeSymbol Type, string SectionName)> DiscoveredTypes { get; } = new();

        public AttributeTypeFinder(INamedTypeSymbol attributeType)
        {
            _attributeType = attributeType;
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
            foreach (var attr in symbol.GetAttributes())
            {
                if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, _attributeType) ||
                    SymbolEqualityComparer.Default.Equals(attr.AttributeClass?.OriginalDefinition, _attributeType))
                {
                    if (attr.ConstructorArguments.Length > 0 &&
                        attr.ConstructorArguments[0].Value is string sectionName &&
                        !string.IsNullOrEmpty(sectionName))
                    {
                        DiscoveredTypes.Add((symbol, sectionName));
                    }
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
