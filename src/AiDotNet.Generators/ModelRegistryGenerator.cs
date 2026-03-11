using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that auto-generates a static <c>ModelMetadataRegistry</c>
/// class at compile time, collecting all model metadata from attributes on concrete IFullModel
/// implementations.
/// </summary>
/// <remarks>
/// <para>
/// Automatically discovers all non-abstract classes implementing IFullModel anywhere in their
/// inheritance chain (via Roslyn's AllInterfaces — no hardcoded type list required) and reads
/// their [ModelDomain], [ModelCategory], [ModelTask], [ModelComplexity], [ModelInput], and
/// [ModelPaper] attributes to build a zero-reflection static registry.
/// </para>
/// </remarks>
[Generator]
public class ModelRegistryGenerator : IIncrementalGenerator
{
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";

    // Fully-qualified attribute names
    private const string ModelDomainAttr = "AiDotNet.Attributes.ModelDomainAttribute";
    private const string ModelCategoryAttr = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttr = "AiDotNet.Attributes.ModelTaskAttribute";
    private const string ModelComplexityAttr = "AiDotNet.Attributes.ModelComplexityAttribute";
    private const string ModelInputAttr = "AiDotNet.Attributes.ModelInputAttribute";
    private const string ModelPaperAttr = "AiDotNet.Attributes.ModelPaperAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Syntax-first filter: non-abstract class declarations with base types
        var classDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => GetModelClassOrNull(ctx))
            .Where(static s => s is not null);

        var collected = classDeclarations.Collect().Combine(context.CompilationProvider);

        context.RegisterSourceOutput(collected, static (spc, source) =>
        {
            var (candidates, compilation) = source;
            Execute(spc, candidates, compilation);
        });
    }

    private static bool IsCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;

        if (cds.BaseList is null || cds.BaseList.Types.Count == 0)
            return false;

        foreach (var modifier in cds.Modifiers)
        {
            if (modifier.Text == "abstract")
                return false;
        }

        return true;
    }

    private static INamedTypeSymbol? GetModelClassOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        if (ImplementsIFullModel(symbol))
            return symbol;

        return null;
    }

    private static bool ImplementsIFullModel(INamedTypeSymbol type)
    {
        foreach (var iface in type.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(IFullModelName, System.StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> candidates,
        Compilation compilation)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitEmptyRegistry(context);
            return;
        }

        // Resolve attribute type symbols
        var domainAttrSymbol = compilation.GetTypeByMetadataName(ModelDomainAttr);
        var categoryAttrSymbol = compilation.GetTypeByMetadataName(ModelCategoryAttr);
        var taskAttrSymbol = compilation.GetTypeByMetadataName(ModelTaskAttr);
        var complexityAttrSymbol = compilation.GetTypeByMetadataName(ModelComplexityAttr);
        var inputAttrSymbol = compilation.GetTypeByMetadataName(ModelInputAttr);
        var paperAttrSymbol = compilation.GetTypeByMetadataName(ModelPaperAttr);

        // If core attributes don't exist, emit empty registry
        if (domainAttrSymbol is null || categoryAttrSymbol is null ||
            taskAttrSymbol is null || complexityAttrSymbol is null ||
            inputAttrSymbol is null)
        {
            EmitEmptyRegistry(context);
            return;
        }

        var entries = new List<ModelEntryData>();
        var seen = new HashSet<string>();
        var annotatedNames = new HashSet<string>();

        foreach (var modelClass in candidates)
        {
            if (modelClass is null)
                continue;

            var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);

            // Deduplicate (same class can appear from multiple syntax trees for partial classes)
            if (!seen.Add(fullName))
                continue;

            var entry = ExtractMetadata(
                modelClass, fullName,
                domainAttrSymbol, categoryAttrSymbol, taskAttrSymbol,
                complexityAttrSymbol, inputAttrSymbol, paperAttrSymbol);

            // Only include fully-annotated models to avoid default enum values
            if (entry.HasAllRequiredMetadata)
            {
                entries.Add(entry);
                annotatedNames.Add(fullName);
            }
        }

        // Sort entries by fully-qualified name for deterministic output
        entries.Sort((a, b) => string.Compare(a.FullyQualifiedName, b.FullyQualifiedName, System.StringComparison.Ordinal));

        EmitRegistry(context, entries);

        // Emit discovery manifest listing ALL concrete IFullModel classes with file paths
        EmitDiscoveryManifest(context, candidates, annotatedNames);
    }

    private static ModelEntryData ExtractMetadata(
        INamedTypeSymbol modelClass,
        string fullyQualifiedName,
        INamedTypeSymbol domainAttrSymbol,
        INamedTypeSymbol categoryAttrSymbol,
        INamedTypeSymbol taskAttrSymbol,
        INamedTypeSymbol complexityAttrSymbol,
        INamedTypeSymbol inputAttrSymbol,
        INamedTypeSymbol? paperAttrSymbol)
    {
        var attributes = modelClass.GetAttributes();
        var entry = new ModelEntryData
        {
            FullyQualifiedName = fullyQualifiedName,
            ClassName = modelClass.Name,
            TypeParameterCount = modelClass.TypeParameters.Length
        };

        foreach (var attr in attributes)
        {
            if (attr.AttributeClass is null)
                continue;

            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, domainAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var val = attr.ConstructorArguments[0].Value;
                    if (val is int intVal)
                    {
                        entry.Domains.Add(intVal);
                    }
                }
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, categoryAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var val = attr.ConstructorArguments[0].Value;
                    if (val is int intVal)
                    {
                        entry.Categories.Add(intVal);
                    }
                }
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, taskAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var val = attr.ConstructorArguments[0].Value;
                    if (val is int intVal)
                    {
                        entry.Tasks.Add(intVal);
                    }
                }
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, complexityAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1)
                {
                    var val = attr.ConstructorArguments[0].Value;
                    if (val is int intVal)
                    {
                        entry.Complexity = intVal;
                        entry.HasComplexity = true;
                    }
                }
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, inputAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 2)
                {
                    var inputType = attr.ConstructorArguments[0].Value as INamedTypeSymbol;
                    var outputType = attr.ConstructorArguments[1].Value as INamedTypeSymbol;
                    if (inputType is not null)
                    {
                        entry.InputTypeName = inputType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                    }
                    if (outputType is not null)
                    {
                        entry.OutputTypeName = outputType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                    }
                }
            }
            else if (paperAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, paperAttrSymbol))
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
                entry.Papers.Add(paper);
            }
        }

        // Extract XML documentation
        var xmlDoc = modelClass.GetDocumentationCommentXml();
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            entry.Summary = ExtractXmlElement(xmlDoc, "summary");
            entry.BeginnerGuide = ExtractBeginnerRemarks(xmlDoc);
        }

        return entry;
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

    private static string ExtractBeginnerRemarks(string xml)
    {
        // Look for "For Beginners" content within <remarks>
        var remarksContent = ExtractXmlElement(xml, "remarks");
        if (string.IsNullOrWhiteSpace(remarksContent))
            return string.Empty;

        var beginnerIdx = remarksContent.IndexOf("For Beginners", System.StringComparison.OrdinalIgnoreCase);
        if (beginnerIdx < 0)
            return string.Empty;

        // Take text after "For Beginners:" or "For Beginners</b>" marker
        var colonIdx = remarksContent.IndexOf(":", beginnerIdx, System.StringComparison.Ordinal);
        var closeBIdx = remarksContent.IndexOf("</b>", beginnerIdx, System.StringComparison.Ordinal);

        int contentStart;
        if (closeBIdx >= 0 && (colonIdx < 0 || closeBIdx < colonIdx))
        {
            contentStart = closeBIdx + 4;
        }
        else if (colonIdx >= 0)
        {
            contentStart = colonIdx + 1;
        }
        else
        {
            contentStart = beginnerIdx + "For Beginners".Length;
        }

        // Take until end of remarks or next </para>
        var endIdx = remarksContent.IndexOf("</para>", contentStart, System.StringComparison.Ordinal);
        if (endIdx < 0)
            endIdx = remarksContent.Length;

        var text = remarksContent.Substring(contentStart, endIdx - contentStart);
        return CleanXmlText(text);
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

        // Normalize whitespace
        var result = sb.ToString().Trim();
        while (result.Contains("  "))
        {
            result = result.Replace("  ", " ");
        }
        result = result.Replace("\r\n", " ").Replace("\n", " ").Replace("\r", " ");
        while (result.Contains("  "))
        {
            result = result.Replace("  ", " ");
        }

        return result.Trim();
    }

    private static void EmitEmptyRegistry(SourceProductionContext context)
    {
        EmitRegistry(context, new List<ModelEntryData>());
    }

    private static void EmitRegistry(SourceProductionContext context, List<ModelEntryData> entries)
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

        // ModelPaperEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Represents an academic paper reference for a model.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("public sealed class ModelPaperEntry");
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
        sb.AppendLine("    public ModelPaperEntry(string title, string url, int year, string authors)");
        sb.AppendLine("    {");
        sb.AppendLine("        Title = title;");
        sb.AppendLine("        Url = url;");
        sb.AppendLine("        Year = year;");
        sb.AppendLine("        Authors = authors;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ModelMetadataEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Contains all metadata for a single model class, collected from attributes at compile time.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("public sealed class ModelMetadataEntry");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Gets the fully-qualified type name of the model class.</summary>");
        sb.AppendLine("    public string TypeName { get; }");
        sb.AppendLine("    /// <summary>Gets the short class name of the model.</summary>");
        sb.AppendLine("    public string ClassName { get; }");
        sb.AppendLine("    /// <summary>Gets the number of generic type parameters.</summary>");
        sb.AppendLine("    public int TypeParameterCount { get; }");
        sb.AppendLine("    /// <summary>Gets the application domains this model belongs to.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelDomain> Domains { get; }");
        sb.AppendLine("    /// <summary>Gets the algorithm categories this model belongs to.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelCategory> Categories { get; }");
        sb.AppendLine("    /// <summary>Gets the tasks this model performs.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelTask> Tasks { get; }");
        sb.AppendLine("    /// <summary>Gets the computational complexity.</summary>");
        sb.AppendLine("    public ModelComplexity Complexity { get; }");
        sb.AppendLine("    /// <summary>Gets the expected input type name.</summary>");
        sb.AppendLine("    public string InputTypeName { get; }");
        sb.AppendLine("    /// <summary>Gets the expected output type name.</summary>");
        sb.AppendLine("    public string OutputTypeName { get; }");
        sb.AppendLine("    /// <summary>Gets the academic paper references.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelPaperEntry> Papers { get; }");
        sb.AppendLine("    /// <summary>Gets the XML doc summary text.</summary>");
        sb.AppendLine("    public string Summary { get; }");
        sb.AppendLine("    /// <summary>Gets the beginner-friendly guide text.</summary>");
        sb.AppendLine("    public string BeginnerGuide { get; }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Initializes a new metadata entry.</summary>");
        sb.AppendLine("    public ModelMetadataEntry(");
        sb.AppendLine("        string typeName, string className, int typeParameterCount,");
        sb.AppendLine("        ModelDomain[] domains, ModelCategory[] categories, ModelTask[] tasks,");
        sb.AppendLine("        ModelComplexity complexity, string inputTypeName, string outputTypeName,");
        sb.AppendLine("        ModelPaperEntry[] papers, string summary, string beginnerGuide)");
        sb.AppendLine("    {");
        sb.AppendLine("        TypeName = typeName;");
        sb.AppendLine("        ClassName = className;");
        sb.AppendLine("        TypeParameterCount = typeParameterCount;");
        sb.AppendLine("        Domains = domains;");
        sb.AppendLine("        Categories = categories;");
        sb.AppendLine("        Tasks = tasks;");
        sb.AppendLine("        Complexity = complexity;");
        sb.AppendLine("        InputTypeName = inputTypeName;");
        sb.AppendLine("        OutputTypeName = outputTypeName;");
        sb.AppendLine("        Papers = papers;");
        sb.AppendLine("        Summary = summary;");
        sb.AppendLine("        BeginnerGuide = beginnerGuide;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ModelMetadataRegistry static class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated static registry of all model metadata. Zero runtime reflection.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("public static class ModelMetadataRegistry");
        sb.AppendLine("{");
        sb.AppendLine($"    /// <summary>Total number of annotated model classes.</summary>");
        sb.AppendLine($"    public const int ModelCount = {entries.Count};");
        sb.AppendLine();

        // All entries array
        sb.AppendLine("    /// <summary>Gets all model metadata entries.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ModelMetadataEntry> All { get; } = new ModelMetadataEntry[]");
        sb.AppendLine("    {");

        foreach (var entry in entries)
        {
            EmitEntry(sb, entry);
        }

        sb.AppendLine("    };");
        sb.AppendLine();

        // Lookup dictionaries (lazily built from All, thread-safe via Lazy<T>)
        sb.AppendLine("    private static readonly Lazy<Dictionary<ModelDomain, List<ModelMetadataEntry>>> _byDomain =");
        sb.AppendLine("        new Lazy<Dictionary<ModelDomain, List<ModelMetadataEntry>>>(BuildByDomain);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<ModelCategory, List<ModelMetadataEntry>>> _byCategory =");
        sb.AppendLine("        new Lazy<Dictionary<ModelCategory, List<ModelMetadataEntry>>>(BuildByCategory);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<ModelTask, List<ModelMetadataEntry>>> _byTask =");
        sb.AppendLine("        new Lazy<Dictionary<ModelTask, List<ModelMetadataEntry>>>(BuildByTask);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<ModelComplexity, List<ModelMetadataEntry>>> _byComplexity =");
        sb.AppendLine("        new Lazy<Dictionary<ModelComplexity, List<ModelMetadataEntry>>>(BuildByComplexity);");
        sb.AppendLine("    private static readonly Lazy<Dictionary<string, ModelMetadataEntry>> _byTypeName =");
        sb.AppendLine("        new Lazy<Dictionary<string, ModelMetadataEntry>>(BuildByTypeName);");
        sb.AppendLine();

        // BuildByDomain
        sb.AppendLine("    private static Dictionary<ModelDomain, List<ModelMetadataEntry>> BuildByDomain()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<ModelDomain, List<ModelMetadataEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var domain in entry.Domains)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (!dict.TryGetValue(domain, out var list))");
        sb.AppendLine("                {");
        sb.AppendLine("                    list = new List<ModelMetadataEntry>();");
        sb.AppendLine("                    dict[domain] = list;");
        sb.AppendLine("                }");
        sb.AppendLine("                list.Add(entry);");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByCategory
        sb.AppendLine("    private static Dictionary<ModelCategory, List<ModelMetadataEntry>> BuildByCategory()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<ModelCategory, List<ModelMetadataEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var category in entry.Categories)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (!dict.TryGetValue(category, out var list))");
        sb.AppendLine("                {");
        sb.AppendLine("                    list = new List<ModelMetadataEntry>();");
        sb.AppendLine("                    dict[category] = list;");
        sb.AppendLine("                }");
        sb.AppendLine("                list.Add(entry);");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByTask
        sb.AppendLine("    private static Dictionary<ModelTask, List<ModelMetadataEntry>> BuildByTask()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<ModelTask, List<ModelMetadataEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var task in entry.Tasks)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (!dict.TryGetValue(task, out var list))");
        sb.AppendLine("                {");
        sb.AppendLine("                    list = new List<ModelMetadataEntry>();");
        sb.AppendLine("                    dict[task] = list;");
        sb.AppendLine("                }");
        sb.AppendLine("                list.Add(entry);");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByComplexity
        sb.AppendLine("    private static Dictionary<ModelComplexity, List<ModelMetadataEntry>> BuildByComplexity()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<ModelComplexity, List<ModelMetadataEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (!dict.TryGetValue(entry.Complexity, out var list))");
        sb.AppendLine("            {");
        sb.AppendLine("                list = new List<ModelMetadataEntry>();");
        sb.AppendLine("                dict[entry.Complexity] = list;");
        sb.AppendLine("            }");
        sb.AppendLine("            list.Add(entry);");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildByTypeName
        sb.AppendLine("    private static Dictionary<string, ModelMetadataEntry> BuildByTypeName()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<string, ModelMetadataEntry>(StringComparer.Ordinal);");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            dict[entry.TypeName] = entry;");
        sb.AppendLine("        }");
        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Query methods
        EmitLazyQueryMethod(sb, "GetByDomain", "ModelDomain", "domain", "_byDomain");
        EmitLazyQueryMethod(sb, "GetByCategory", "ModelCategory", "category", "_byCategory");
        EmitLazyQueryMethod(sb, "GetByTask", "ModelTask", "task", "_byTask");
        EmitLazyQueryMethod(sb, "GetByComplexity", "ModelComplexity", "complexity", "_byComplexity");

        // GetByTypeName
        sb.AppendLine("    /// <summary>Gets the metadata entry for a specific model type name.</summary>");
        sb.AppendLine("    public static ModelMetadataEntry? GetByTypeName(string typeName)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_byTypeName.Value.TryGetValue(typeName, out var entry))");
        sb.AppendLine("            return entry;");
        sb.AppendLine("        return null;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetByClassName
        sb.AppendLine("    /// <summary>Gets all metadata entries matching a short class name.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ModelMetadataEntry> GetByClassName(string className)");
        sb.AppendLine("    {");
        sb.AppendLine("        var results = new List<ModelMetadataEntry>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (string.Equals(entry.ClassName, className, StringComparison.Ordinal))");
        sb.AppendLine("                results.Add(entry);");
        sb.AppendLine("        }");
        sb.AppendLine("        return results;");
        sb.AppendLine("    }");

        sb.AppendLine("}");

        context.AddSource("ModelMetadataRegistry.g.cs", sb.ToString());
    }

    private static void EmitLazyQueryMethod(StringBuilder sb, string methodName, string enumType, string paramName, string fieldName)
    {
        sb.AppendLine($"    /// <summary>Gets all model entries for the specified {paramName}.</summary>");
        sb.AppendLine($"    public static IReadOnlyList<ModelMetadataEntry> {methodName}({enumType} {paramName})");
        sb.AppendLine("    {");
        sb.AppendLine($"        if ({fieldName}.Value.TryGetValue({paramName}, out var list))");
        sb.AppendLine("            return list;");
        sb.AppendLine($"        return System.Array.Empty<ModelMetadataEntry>();");
        sb.AppendLine("    }");
        sb.AppendLine();
    }

    private static void EmitEntry(StringBuilder sb, ModelEntryData entry)
    {
        sb.AppendLine("        new ModelMetadataEntry(");

        // TypeName, ClassName, TypeParameterCount
        sb.AppendLine($"            {EscapeString(entry.FullyQualifiedName)},");
        sb.AppendLine($"            {EscapeString(entry.ClassName)},");
        sb.AppendLine($"            {entry.TypeParameterCount},");

        // Domains array
        if (entry.Domains.Count == 0)
        {
            sb.AppendLine("            System.Array.Empty<ModelDomain>(),");
        }
        else
        {
            sb.Append("            new ModelDomain[] { ");
            sb.Append(string.Join(", ", entry.Domains.Select(d => $"(ModelDomain){d}")));
            sb.AppendLine(" },");
        }

        // Categories array
        if (entry.Categories.Count == 0)
        {
            sb.AppendLine("            System.Array.Empty<ModelCategory>(),");
        }
        else
        {
            sb.Append("            new ModelCategory[] { ");
            sb.Append(string.Join(", ", entry.Categories.Select(c => $"(ModelCategory){c}")));
            sb.AppendLine(" },");
        }

        // Tasks array
        if (entry.Tasks.Count == 0)
        {
            sb.AppendLine("            System.Array.Empty<ModelTask>(),");
        }
        else
        {
            sb.Append("            new ModelTask[] { ");
            sb.Append(string.Join(", ", entry.Tasks.Select(t => $"(ModelTask){t}")));
            sb.AppendLine(" },");
        }

        // Complexity
        sb.AppendLine($"            (ModelComplexity){entry.Complexity},");

        // InputTypeName, OutputTypeName
        sb.AppendLine($"            {EscapeString(entry.InputTypeName)},");
        sb.AppendLine($"            {EscapeString(entry.OutputTypeName)},");

        // Papers array
        if (entry.Papers.Count == 0)
        {
            sb.AppendLine("            System.Array.Empty<ModelPaperEntry>(),");
        }
        else
        {
            sb.AppendLine("            new ModelPaperEntry[]");
            sb.AppendLine("            {");
            foreach (var paper in entry.Papers)
            {
                sb.AppendLine($"                new ModelPaperEntry({EscapeString(paper.Title)}, {EscapeString(paper.Url)}, {paper.Year}, {EscapeString(paper.Authors)}),");
            }
            sb.AppendLine("            },");
        }

        // Summary, BeginnerGuide
        sb.AppendLine($"            {EscapeString(entry.Summary)},");
        sb.AppendLine($"            {EscapeString(entry.BeginnerGuide)}");
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

    private class ModelEntryData
    {
        public string FullyQualifiedName { get; set; } = string.Empty;
        public string ClassName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Domains { get; } = new List<int>();
        public List<int> Categories { get; } = new List<int>();
        public List<int> Tasks { get; } = new List<int>();
        public int Complexity { get; set; }
        public bool HasComplexity { get; set; }
        public string InputTypeName { get; set; } = string.Empty;
        public string OutputTypeName { get; set; } = string.Empty;
        public List<PaperData> Papers { get; } = new List<PaperData>();
        public string Summary { get; set; } = string.Empty;
        public string BeginnerGuide { get; set; } = string.Empty;

        public bool HasAnyMetadata =>
            Domains.Count > 0 || Categories.Count > 0 || Tasks.Count > 0 || HasComplexity ||
            !string.IsNullOrEmpty(InputTypeName) || Papers.Count > 0;

        public bool HasAllRequiredMetadata =>
            Domains.Count > 0 && Categories.Count > 0 && Tasks.Count > 0 && HasComplexity &&
            !string.IsNullOrEmpty(InputTypeName);
    }

    private class PaperData
    {
        public string Title { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public int Year { get; set; }
        public string Authors { get; set; } = string.Empty;
    }

    /// <summary>
    /// Emits a discovery manifest listing ALL concrete IFullModel implementations
    /// with their source file paths, class names, and annotation status.
    /// </summary>
    private static void EmitDiscoveryManifest(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> candidates,
        HashSet<string> annotatedFullNames)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Generated;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated manifest of all discovered concrete IFullModel implementations.");
        sb.AppendLine("/// Use this to track annotation progress for issue #958.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("public static class ModelDiscoveryManifest");
        sb.AppendLine("{");

        var manifestEntries = new List<(string className, string fullName, string filePath, bool hasAttributes)>();
        var seen = new HashSet<string>();

        foreach (var modelClass in candidates)
        {
            if (modelClass is null)
                continue;

            var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName))
                continue;

            var location = modelClass.Locations.FirstOrDefault();
            var filePath = string.Empty;
            if (location is not null && location.SourceTree is not null)
            {
                filePath = location.SourceTree.FilePath;
                // Normalize to forward slashes and make relative to src/
                filePath = filePath.Replace("\\", "/");
                var srcIdx = filePath.IndexOf("/src/", System.StringComparison.OrdinalIgnoreCase);
                if (srcIdx >= 0)
                {
                    filePath = filePath.Substring(srcIdx + 1); // Keep "src/..."
                }
            }

            var hasAttributes = annotatedFullNames.Contains(fullName);
            manifestEntries.Add((modelClass.Name, fullName, filePath, hasAttributes));
        }

        // Sort by file path for deterministic output and easy batching
        manifestEntries.Sort((a, b) => string.Compare(a.filePath, b.filePath, System.StringComparison.OrdinalIgnoreCase));

        var totalCount = manifestEntries.Count;
        var annotatedCount = 0;
        foreach (var e in manifestEntries)
        {
            if (e.hasAttributes) annotatedCount++;
        }

        sb.AppendLine($"    /// <summary>Total concrete IFullModel implementations discovered.</summary>");
        sb.AppendLine($"    public const int TotalModels = {totalCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Models with at least one metadata attribute.</summary>");
        sb.AppendLine($"    public const int AnnotatedModels = {annotatedCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Models still needing annotation.</summary>");
        sb.AppendLine($"    public const int UnannotatedModels = {totalCount - annotatedCount};");
        sb.AppendLine();

        // Emit all entries as a string array for programmatic access
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// All discovered model entries as \"FilePath|ClassName|FullyQualifiedName|IsAnnotated\" strings.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    public static readonly string[] AllEntries = new string[]");
        sb.AppendLine("    {");

        foreach (var entry in manifestEntries)
        {
            var annotated = entry.hasAttributes ? "true" : "false";
            sb.AppendLine($"        \"{EscapeString(entry.filePath).Trim('"')}|{EscapeString(entry.className).Trim('"')}|{EscapeString(entry.fullName).Trim('"')}|{annotated}\",");
        }

        sb.AppendLine("    };");
        sb.AppendLine("}");

        context.AddSource("ModelDiscoveryManifest.g.cs", sb.ToString());
    }
}
