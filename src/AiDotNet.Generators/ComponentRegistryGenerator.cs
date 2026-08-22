using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that auto-generates a static <c>ComponentMetadataRegistry</c>
/// class at compile time, collecting all Tier 2 component metadata from attributes on concrete classes
/// annotated with <c>[ComponentType]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Automatically discovers all non-abstract classes that have at least one
/// <c>[ComponentType]</c> attribute and reads their <c>[ComponentType]</c>,
/// <c>[PipelineStage]</c>, <c>[ComponentDependency]</c>, and <c>[ResearchPaper]</c>
/// attributes to build a zero-reflection static registry.
/// </para>
/// </remarks>
[Generator]
public class ComponentRegistryGenerator : IIncrementalGenerator
{
    // Fully-qualified attribute names
    private const string ComponentTypeAttr = "AiDotNet.Attributes.ComponentTypeAttribute";
    private const string PipelineStageAttr = "AiDotNet.Attributes.PipelineStageAttribute";
    private const string ComponentDependencyAttr = "AiDotNet.Attributes.ComponentDependencyAttribute";
    private const string ResearchPaperAttr = "AiDotNet.Attributes.ResearchPaperAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Syntax-first filter: non-abstract class declarations (any class can have [ComponentType])
        // Values, not symbols -- see DiscoveryApiGenerator for the full rationale.
        var entries = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => Analyze(ctx))
            .Where(static e => e is not null)
            .Select(static (e, _) => e ?? ComponentEntryData.Empty);

        context.RegisterSourceOutput(entries.Collect(), static (spc, collected) => Emit(spc, collected));
    }

    /// <summary>
    /// Resolves one candidate class into a value-equatable entry, or null when it carries no
    /// [ComponentType]. All symbol access is confined to this method.
    /// </summary>
    private static ComponentEntryData? Analyze(GeneratorSyntaxContext ctx)
    {
        if (GetComponentClassOrNull(ctx) is not INamedTypeSymbol componentClass)
            return null;

        var compilation = ctx.SemanticModel.Compilation;
        var componentTypeAttrSymbol = compilation.GetTypeByMetadataName(ComponentTypeAttr);
        var pipelineStageAttrSymbol = compilation.GetTypeByMetadataName(PipelineStageAttr);
        var componentDependencyAttrSymbol = compilation.GetTypeByMetadataName(ComponentDependencyAttr);
        var researchPaperAttrSymbol = compilation.GetTypeByMetadataName(ResearchPaperAttr);

        if (componentTypeAttrSymbol is null)
            return null;

        var fullName = componentClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        var entry = ExtractMetadata(
            componentClass, fullName,
            componentTypeAttrSymbol, pipelineStageAttrSymbol,
            componentDependencyAttrSymbol, researchPaperAttrSymbol);

        return entry.ComponentTypes.Length > 0 ? entry : null;
    }

    private static bool IsCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;

        // Unlike ModelRegistryGenerator, we don't require a BaseList since any class
        // can have [ComponentType]. But we still skip abstract classes.
        foreach (var modifier in cds.Modifiers)
        {
            if (modifier.Text == "abstract")
                return false;
        }

        return true;
    }

    private static INamedTypeSymbol? GetComponentClassOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        // Check if the class has at least one [ComponentType] attribute
        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is not null &&
                attr.AttributeClass.ToDisplayString() == ComponentTypeAttr)
            {
                return symbol;
            }
        }

        return null;
    }

    private static void Emit(SourceProductionContext context, ImmutableArray<ComponentEntryData> candidates)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitEmptyRegistry(context);
            return;
        }

        var entries = new List<ComponentEntryData>();
        var seen = new HashSet<string>();

        foreach (var entry in candidates)
        {
            if (entry.FullyQualifiedName.Length == 0)
                continue;

            // Deduplicate (same class can appear from multiple syntax trees for partial classes)
            if (!seen.Add(entry.FullyQualifiedName))
                continue;

            entries.Add(entry);
        }

        if (entries.Count == 0)
        {
            EmitEmptyRegistry(context);
            return;
        }

        // Sort entries by fully-qualified name for deterministic output
        entries.Sort((a, b) => string.Compare(a.FullyQualifiedName, b.FullyQualifiedName, System.StringComparison.Ordinal));

        EmitRegistry(context, entries);
    }

    private static ComponentEntryData ExtractMetadata(
        INamedTypeSymbol componentClass,
        string fullyQualifiedName,
        INamedTypeSymbol componentTypeAttrSymbol,
        INamedTypeSymbol? pipelineStageAttrSymbol,
        INamedTypeSymbol? componentDependencyAttrSymbol,
        INamedTypeSymbol? researchPaperAttrSymbol)
    {
        var attributes = componentClass.GetAttributes();
        var componentTypes = new List<int>();
        var pipelineStages = new List<int>();
        var dependencies = new List<DependencyData>();
        var papers = new List<PaperData>();
        string summary = string.Empty;

        foreach (var attr in attributes)
        {
            if (attr.AttributeClass is null)
                continue;

            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, componentTypeAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var val = attr.ConstructorArguments[0].Value;
                    if (val is int intVal)
                    {
                        componentTypes.Add(intVal);
                    }
                }
            }
            else if (pipelineStageAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, pipelineStageAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var val = attr.ConstructorArguments[0].Value;
                    if (val is int intVal)
                    {
                        pipelineStages.Add(intVal);
                    }
                }
            }
            else if (componentDependencyAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, componentDependencyAttrSymbol))
            {
                var dep = new DependencyData();

                // First constructor argument is typeof(DependencyType)
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var typeArg = attr.ConstructorArguments[0].Value as INamedTypeSymbol;
                    if (typeArg is not null)
                    {
                        dep.DependencyTypeName = typeArg.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                    }
                }

                // Second constructor argument (if present) is description string
                if (attr.ConstructorArguments.Length >= 2)
                {
                    dep.Description = attr.ConstructorArguments[1].Value as string ?? string.Empty;
                }

                // Check named arguments for Description and Required
                foreach (var named in attr.NamedArguments)
                {
                    if (named.Key == "Description" && named.Value.Value is string desc)
                    {
                        dep.Description = desc;
                    }
                    else if (named.Key == "Required" && named.Value.Value is bool required)
                    {
                        dep.Required = required;
                    }
                }

                dependencies.Add(dep);
            }
            else if (researchPaperAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, researchPaperAttrSymbol))
            {
                var paper = new PaperData();
                if (attr.ConstructorArguments.Length >= 2)
                {
                    paper.Title = attr.ConstructorArguments[0].Value as string ?? string.Empty;
                    paper.Url = attr.ConstructorArguments[1].Value as string ?? string.Empty;
                }
                // Check named arguments for Year and Authors
                foreach (var named in attr.NamedArguments)
                {
                    if (named.Key == "Year" && named.Value.Value is int year)
                    {
                        paper.Year = year;
                    }
                    else if (named.Key == "Authors" && named.Value.Value is string authors)
                    {
                        paper.Authors = authors;
                    }
                }
                papers.Add(paper);
            }
        }

        // Extract XML documentation summary
        var xmlDoc = componentClass.GetDocumentationCommentXml();
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            summary = ExtractXmlElement(xmlDoc, "summary");
        }

        return new ComponentEntryData(
            fullyQualifiedName,
            componentClass.Name,
            componentClass.TypeParameters.Length,
            componentTypes.ToImmutableArray(),
            pipelineStages.ToImmutableArray(),
            dependencies.ToImmutableArray(),
            papers.ToImmutableArray(),
            summary);
    }

    private static string ExtractXmlElement(string xml, string elementName)
    {
        var startTag = "<" + elementName + ">";
        var endTag = "</" + elementName + ">";
        var startIdx = xml.IndexOf(startTag, System.StringComparison.Ordinal);
        if (startIdx < 0)
            return string.Empty;

        startIdx += startTag.Length;
        var endIdx = xml.IndexOf(endTag, startIdx, System.StringComparison.Ordinal);
        if (endIdx < 0)
            return string.Empty;

        return CleanXmlText(xml.Substring(startIdx, endIdx - startIdx));
    }

    private static string CleanXmlText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        // Remove XML tags, normalize whitespace
        var sb = new StringBuilder(text.Length);
        var inTag = false;
        foreach (var c in text)
        {
            if (c == '<')
            {
                inTag = true;
                continue;
            }
            if (c == '>')
            {
                inTag = false;
                continue;
            }
            if (!inTag)
            {
                sb.Append(c);
            }
        }

        // Single-pass whitespace normalization: collapse all runs of whitespace to a single space
        var raw = sb.ToString();
        var normalized = new StringBuilder(raw.Length);
        bool prevWasSpace = false;
        foreach (char c in raw)
        {
            if (c == ' ' || c == '\r' || c == '\n' || c == '\t')
            {
                if (!prevWasSpace)
                {
                    normalized.Append(' ');
                    prevWasSpace = true;
                }
            }
            else
            {
                normalized.Append(c);
                prevWasSpace = false;
            }
        }

        return normalized.ToString().Trim();
    }

    private static void EmitEmptyRegistry(SourceProductionContext context)
    {
        EmitRegistry(context, new List<ComponentEntryData>());
    }

    private static void EmitRegistry(SourceProductionContext context, List<ComponentEntryData> entries)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine("using AiDotNet.Enums;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Generated;");
        sb.AppendLine();

        // ComponentDependencyEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Represents a dependency that a component requires from another component or interface.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class ComponentDependencyEntry");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Gets the fully-qualified type name of the dependency.</summary>");
        sb.AppendLine("    public string DependencyTypeName { get; }");
        sb.AppendLine("    /// <summary>Gets the description of why this dependency is needed.</summary>");
        sb.AppendLine("    public string Description { get; }");
        sb.AppendLine("    /// <summary>Gets whether this dependency is required (true) or optional (false).</summary>");
        sb.AppendLine("    public bool Required { get; }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Initializes a new dependency entry.</summary>");
        sb.AppendLine("    public ComponentDependencyEntry(string dependencyTypeName, string description, bool required)");
        sb.AppendLine("    {");
        sb.AppendLine("        DependencyTypeName = dependencyTypeName;");
        sb.AppendLine("        Description = description;");
        sb.AppendLine("        Required = required;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ComponentPaperEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Represents an academic paper reference for a component.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class ComponentPaperEntry");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Gets the paper title.</summary>");
        sb.AppendLine("    public string Title { get; }");
        sb.AppendLine("    /// <summary>Gets the paper URL.</summary>");
        sb.AppendLine("    public string Url { get; }");
        sb.AppendLine("    /// <summary>Gets the publication year (0 if unknown).</summary>");
        sb.AppendLine("    public int Year { get; }");
        sb.AppendLine("    /// <summary>Gets the paper authors.</summary>");
        sb.AppendLine("    public string Authors { get; }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Initializes a new paper entry.</summary>");
        sb.AppendLine("    public ComponentPaperEntry(string title, string url, int year, string authors)");
        sb.AppendLine("    {");
        sb.AppendLine("        Title = title;");
        sb.AppendLine("        Url = url;");
        sb.AppendLine("        Year = year;");
        sb.AppendLine("        Authors = authors;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ComponentMetadataEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Contains all metadata for a single component class, collected from attributes at compile time.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class ComponentMetadataEntry");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Gets the fully-qualified type name of the component class.</summary>");
        sb.AppendLine("    public string TypeName { get; }");
        sb.AppendLine("    /// <summary>Gets the short class name of the component.</summary>");
        sb.AppendLine("    public string ClassName { get; }");
        sb.AppendLine("    /// <summary>Gets the number of generic type parameters.</summary>");
        sb.AppendLine("    public int TypeParameterCount { get; }");
        sb.AppendLine("    /// <summary>Gets the component types this class serves as.</summary>");
        sb.AppendLine("    public IReadOnlyList<ComponentType> ComponentTypes { get; }");
        sb.AppendLine("    /// <summary>Gets the pipeline stages this component operates in.</summary>");
        sb.AppendLine("    public IReadOnlyList<PipelineStage> PipelineStages { get; }");
        sb.AppendLine("    /// <summary>Gets the dependencies this component requires.</summary>");
        sb.AppendLine("    public IReadOnlyList<ComponentDependencyEntry> Dependencies { get; }");
        sb.AppendLine("    /// <summary>Gets the academic paper references.</summary>");
        sb.AppendLine("    public IReadOnlyList<ComponentPaperEntry> Papers { get; }");
        sb.AppendLine("    /// <summary>Gets the XML doc summary text.</summary>");
        sb.AppendLine("    public string Summary { get; }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Initializes a new metadata entry.</summary>");
        sb.AppendLine("    public ComponentMetadataEntry(");
        sb.AppendLine("        string typeName, string className, int typeParameterCount,");
        sb.AppendLine("        ComponentType[] componentTypes, PipelineStage[] pipelineStages,");
        sb.AppendLine("        ComponentDependencyEntry[] dependencies, ComponentPaperEntry[] papers,");
        sb.AppendLine("        string summary)");
        sb.AppendLine("    {");
        sb.AppendLine("        TypeName = typeName;");
        sb.AppendLine("        ClassName = className;");
        sb.AppendLine("        TypeParameterCount = typeParameterCount;");
        sb.AppendLine("        ComponentTypes = componentTypes;");
        sb.AppendLine("        PipelineStages = pipelineStages;");
        sb.AppendLine("        Dependencies = dependencies;");
        sb.AppendLine("        Papers = papers;");
        sb.AppendLine("        Summary = summary;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ComponentMetadataRegistry static class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated static registry of all Tier 2 component metadata. Zero runtime reflection.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class ComponentMetadataRegistry");
        sb.AppendLine("{");
        sb.AppendLine($"    /// <summary>Total number of annotated component classes.</summary>");
        sb.AppendLine($"    public const int ComponentCount = {entries.Count};");
        sb.AppendLine();

        // All entries array
        sb.AppendLine("    /// <summary>Gets all component metadata entries.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ComponentMetadataEntry> All { get; } = new ComponentMetadataEntry[]");
        sb.AppendLine("    {");

        foreach (var entry in entries)
        {
            EmitEntry(sb, entry);
        }

        sb.AppendLine("    };");
        sb.AppendLine();

        // Lookup dictionaries (lazily built from All, thread-safe via Lazy<T>)
        sb.AppendLine("    private static readonly Lazy<Dictionary<ComponentType, List<ComponentMetadataEntry>>> _byComponentType =");
        sb.AppendLine("        new Lazy<Dictionary<ComponentType, List<ComponentMetadataEntry>>>(BuildByComponentType);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<PipelineStage, List<ComponentMetadataEntry>>> _byPipelineStage =");
        sb.AppendLine("        new Lazy<Dictionary<PipelineStage, List<ComponentMetadataEntry>>>(BuildByPipelineStage);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<string, ComponentMetadataEntry>> _byTypeName =");
        sb.AppendLine("        new Lazy<Dictionary<string, ComponentMetadataEntry>>(BuildByTypeName);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<string, List<ComponentMetadataEntry>>> _byClassName =");
        sb.AppendLine("        new Lazy<Dictionary<string, List<ComponentMetadataEntry>>>(BuildByClassName);");
        sb.AppendLine();

        // BuildByComponentType
        sb.AppendLine("    private static Dictionary<ComponentType, List<ComponentMetadataEntry>> BuildByComponentType()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<ComponentType, List<ComponentMetadataEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var componentType in entry.ComponentTypes)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (!dict.TryGetValue(componentType, out var list))");
        sb.AppendLine("                {");
        sb.AppendLine("                    list = new List<ComponentMetadataEntry>();");
        sb.AppendLine("                    dict[componentType] = list;");
        sb.AppendLine("                }");
        sb.AppendLine("                list.Add(entry);");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByPipelineStage
        sb.AppendLine("    private static Dictionary<PipelineStage, List<ComponentMetadataEntry>> BuildByPipelineStage()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<PipelineStage, List<ComponentMetadataEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var stage in entry.PipelineStages)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (!dict.TryGetValue(stage, out var list))");
        sb.AppendLine("                {");
        sb.AppendLine("                    list = new List<ComponentMetadataEntry>();");
        sb.AppendLine("                    dict[stage] = list;");
        sb.AppendLine("                }");
        sb.AppendLine("                list.Add(entry);");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByTypeName
        sb.AppendLine("    private static Dictionary<string, ComponentMetadataEntry> BuildByTypeName()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<string, ComponentMetadataEntry>(StringComparer.Ordinal);");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            dict[entry.TypeName] = entry;");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByClassName
        sb.AppendLine("    private static Dictionary<string, List<ComponentMetadataEntry>> BuildByClassName()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<string, List<ComponentMetadataEntry>>(StringComparer.Ordinal);");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (!dict.TryGetValue(entry.ClassName, out var list))");
        sb.AppendLine("            {");
        sb.AppendLine("                list = new List<ComponentMetadataEntry>();");
        sb.AppendLine("                dict[entry.ClassName] = list;");
        sb.AppendLine("            }");
        sb.AppendLine("            list.Add(entry);");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Query methods
        EmitLazyQueryMethod(sb, "GetByComponentType", "ComponentType", "componentType", "_byComponentType");
        EmitLazyQueryMethod(sb, "GetByPipelineStage", "PipelineStage", "pipelineStage", "_byPipelineStage");

        // GetByTypeName
        sb.AppendLine("    /// <summary>Gets the metadata entry for a specific component type name.</summary>");
        sb.AppendLine("    public static ComponentMetadataEntry? GetByTypeName(string typeName)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_byTypeName.Value.TryGetValue(typeName, out var entry))");
        sb.AppendLine("            return entry;");
        sb.AppendLine("        return null;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetByClassName
        sb.AppendLine("    /// <summary>Gets all metadata entries matching a short class name.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ComponentMetadataEntry> GetByClassName(string className)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_byClassName.Value.TryGetValue(className, out var list))");
        sb.AppendLine("            return list;");
        sb.AppendLine("        return Array.Empty<ComponentMetadataEntry>();");
        sb.AppendLine("    }");

        sb.AppendLine("}");

        context.AddSource("ComponentMetadataRegistry.g.cs", sb.ToString());
    }

    private static void EmitLazyQueryMethod(StringBuilder sb, string methodName, string enumType, string paramName, string fieldName)
    {
        sb.AppendLine($"    /// <summary>Gets all component entries for the specified {paramName}.</summary>");
        sb.AppendLine($"    public static IReadOnlyList<ComponentMetadataEntry> {methodName}({enumType} {paramName})");
        sb.AppendLine("    {");
        sb.AppendLine($"        if ({fieldName}.Value.TryGetValue({paramName}, out var list))");
        sb.AppendLine("            return list;");
        sb.AppendLine($"        return System.Array.Empty<ComponentMetadataEntry>();");
        sb.AppendLine("    }");
        sb.AppendLine();
    }

    private static void EmitEntry(StringBuilder sb, ComponentEntryData entry)
    {
        sb.AppendLine("        new ComponentMetadataEntry(");

        // TypeName, ClassName, TypeParameterCount
        sb.AppendLine($"            {EscapeString(entry.FullyQualifiedName)},");
        sb.AppendLine($"            {EscapeString(entry.ClassName)},");
        sb.AppendLine($"            {entry.TypeParameterCount},");

        // ComponentTypes array
        if (entry.ComponentTypes.Length == 0)
        {
            sb.AppendLine("            System.Array.Empty<ComponentType>(),");
        }
        else
        {
            sb.Append("            new ComponentType[] { ");
            sb.Append(string.Join(", ", entry.ComponentTypes.Select(ct => $"(ComponentType){ct}")));
            sb.AppendLine(" },");
        }

        // PipelineStages array
        if (entry.PipelineStages.Length == 0)
        {
            sb.AppendLine("            System.Array.Empty<PipelineStage>(),");
        }
        else
        {
            sb.Append("            new PipelineStage[] { ");
            sb.Append(string.Join(", ", entry.PipelineStages.Select(ps => $"(PipelineStage){ps}")));
            sb.AppendLine(" },");
        }

        // Dependencies array
        if (entry.Dependencies.Length == 0)
        {
            sb.AppendLine("            System.Array.Empty<ComponentDependencyEntry>(),");
        }
        else
        {
            sb.AppendLine("            new ComponentDependencyEntry[]");
            sb.AppendLine("            {");
            foreach (var dep in entry.Dependencies)
            {
                sb.AppendLine($"                new ComponentDependencyEntry({EscapeString(dep.DependencyTypeName)}, {EscapeString(dep.Description)}, {(dep.Required ? "true" : "false")}),");
            }
            sb.AppendLine("            },");
        }

        // Papers array
        if (entry.Papers.Length == 0)
        {
            sb.AppendLine("            System.Array.Empty<ComponentPaperEntry>(),");
        }
        else
        {
            sb.AppendLine("            new ComponentPaperEntry[]");
            sb.AppendLine("            {");
            foreach (var paper in entry.Papers)
            {
                sb.AppendLine($"                new ComponentPaperEntry({EscapeString(paper.Title)}, {EscapeString(paper.Url)}, {paper.Year}, {EscapeString(paper.Authors)}),");
            }
            sb.AppendLine("            },");
        }

        // Summary
        sb.AppendLine($"            {EscapeString(entry.Summary)}");
        sb.AppendLine("        ),");
    }

    private static string EscapeString(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "\"\"";

        return "\"" + value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\n", "\\n")
            .Replace("\r", "\\r")
            .Replace("\t", "\\t") + "\"";
    }

    /// <summary>
    /// One registered component, as plain values.
    /// </summary>
    /// <remarks>
    /// Structural equality has to reach THROUGH the nested collections, not stop at the top-level
    /// type: this entry owns Dependencies and Papers, so DependencyData and PaperData are equatable
    /// too. Comparing the outer object while its children compared by reference would leave the
    /// pipeline exactly as uncacheable as before.
    /// </remarks>
    private sealed class ComponentEntryData : System.IEquatable<ComponentEntryData>
    {
        public static readonly ComponentEntryData Empty = new(
            string.Empty, string.Empty, 0,
            ImmutableArray<int>.Empty, ImmutableArray<int>.Empty,
            ImmutableArray<DependencyData>.Empty, ImmutableArray<PaperData>.Empty, string.Empty);

        public ComponentEntryData(
            string fullyQualifiedName,
            string className,
            int typeParameterCount,
            ImmutableArray<int> componentTypes,
            ImmutableArray<int> pipelineStages,
            ImmutableArray<DependencyData> dependencies,
            ImmutableArray<PaperData> papers,
            string summary)
        {
            FullyQualifiedName = fullyQualifiedName;
            ClassName = className;
            TypeParameterCount = typeParameterCount;
            ComponentTypes = componentTypes.IsDefault ? ImmutableArray<int>.Empty : componentTypes;
            PipelineStages = pipelineStages.IsDefault ? ImmutableArray<int>.Empty : pipelineStages;
            Dependencies = dependencies.IsDefault ? ImmutableArray<DependencyData>.Empty : dependencies;
            Papers = papers.IsDefault ? ImmutableArray<PaperData>.Empty : papers;
            Summary = summary;
        }

        public string FullyQualifiedName { get; }
        public string ClassName { get; }
        public int TypeParameterCount { get; }
        public ImmutableArray<int> ComponentTypes { get; }
        public ImmutableArray<int> PipelineStages { get; }
        public ImmutableArray<DependencyData> Dependencies { get; }
        public ImmutableArray<PaperData> Papers { get; }
        public string Summary { get; }

        public bool Equals(ComponentEntryData? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            return string.Equals(FullyQualifiedName, other.FullyQualifiedName, System.StringComparison.Ordinal)
                && string.Equals(ClassName, other.ClassName, System.StringComparison.Ordinal)
                && TypeParameterCount == other.TypeParameterCount
                && string.Equals(Summary, other.Summary, System.StringComparison.Ordinal)
                && IntsEqual(ComponentTypes, other.ComponentTypes)
                && IntsEqual(PipelineStages, other.PipelineStages)
                && ItemsEqual(Dependencies, other.Dependencies)
                && ItemsEqual(Papers, other.Papers);
        }

        public override bool Equals(object? obj) => Equals(obj as ComponentEntryData);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 31) + FullyQualifiedName.GetHashCode();
                hash = (hash * 31) + ClassName.GetHashCode();
                hash = (hash * 31) + TypeParameterCount;
                hash = (hash * 31) + Summary.GetHashCode();
                hash = (hash * 31) + ComponentTypes.Length;
                hash = (hash * 31) + PipelineStages.Length;
                hash = (hash * 31) + Dependencies.Length;
                hash = (hash * 31) + Papers.Length;
                return hash;
            }
        }

        private static bool IntsEqual(ImmutableArray<int> left, ImmutableArray<int> right)
        {
            if (left.Length != right.Length) return false;
            for (int i = 0; i < left.Length; i++)
            {
                if (left[i] != right[i]) return false;
            }
            return true;
        }

        private static bool ItemsEqual<TItem>(ImmutableArray<TItem> left, ImmutableArray<TItem> right)
            where TItem : System.IEquatable<TItem>
        {
            if (left.Length != right.Length) return false;
            for (int i = 0; i < left.Length; i++)
            {
                if (!left[i].Equals(right[i])) return false;
            }
            return true;
        }
    }

    private sealed class DependencyData : System.IEquatable<DependencyData>
    {
        public string DependencyTypeName { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public bool Required { get; set; } = true;

        public bool Equals(DependencyData? other)
            => other is not null
            && string.Equals(DependencyTypeName, other.DependencyTypeName, System.StringComparison.Ordinal)
            && string.Equals(Description, other.Description, System.StringComparison.Ordinal)
            && Required == other.Required;

        public override bool Equals(object? obj) => Equals(obj as DependencyData);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 31) + DependencyTypeName.GetHashCode();
                hash = (hash * 31) + Description.GetHashCode();
                hash = (hash * 31) + (Required ? 1 : 0);
                return hash;
            }
        }
    }

    private sealed class PaperData : System.IEquatable<PaperData>
    {
        public string Title { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public int Year { get; set; }
        public string Authors { get; set; } = string.Empty;

        public bool Equals(PaperData? other)
            => other is not null
            && string.Equals(Title, other.Title, System.StringComparison.Ordinal)
            && string.Equals(Url, other.Url, System.StringComparison.Ordinal)
            && Year == other.Year
            && string.Equals(Authors, other.Authors, System.StringComparison.Ordinal);

        public override bool Equals(object? obj) => Equals(obj as PaperData);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 31) + Title.GetHashCode();
                hash = (hash * 31) + Url.GetHashCode();
                hash = (hash * 31) + Year;
                hash = (hash * 31) + Authors.GetHashCode();
                return hash;
            }
        }
    }
}
