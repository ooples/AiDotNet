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
/// [ResearchPaper] attributes to build a zero-reflection static registry.
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
    private const string ResearchPaperAttr = "AiDotNet.Attributes.ResearchPaperAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Values, not symbols -- see DiscoveryApiGenerator. Note Analyze deliberately does NOT
        // filter: the discovery manifest lists EVERY concrete IFullModel, annotated or not, so the
        // admission rules (HasAnyMetadata / HasAllRequiredMetadata) have to stay in Emit where both
        // consumers can apply them. The relative file path the manifest needs is derived here too,
        // because a Location roots its SyntaxTree and must not enter the pipeline.
        var modelEntries = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => Analyze(ctx))
            .Where(static e => e is not null)
            .Select(static (e, _) => e ?? ModelEntryData.Empty);

        context.RegisterSourceOutput(modelEntries.Collect(), static (spc, collected) => Emit(spc, collected));
    }

    /// <summary>
    /// Resolves one candidate class into a value-equatable entry. Returns an entry for every
    /// concrete IFullModel, annotated or not, because the discovery manifest needs them all.
    /// All symbol access is confined to this method.
    /// </summary>
    private static ModelEntryData? Analyze(GeneratorSyntaxContext ctx)
    {
        if (GetModelClassOrNull(ctx) is not INamedTypeSymbol modelClass)
            return null;

        var compilation = ctx.SemanticModel.Compilation;
        var domainAttrSymbol = compilation.GetTypeByMetadataName(ModelDomainAttr);
        var categoryAttrSymbol = compilation.GetTypeByMetadataName(ModelCategoryAttr);
        var taskAttrSymbol = compilation.GetTypeByMetadataName(ModelTaskAttr);
        var complexityAttrSymbol = compilation.GetTypeByMetadataName(ModelComplexityAttr);
        var inputAttrSymbol = compilation.GetTypeByMetadataName(ModelInputAttr);
        var paperAttrSymbol = compilation.GetTypeByMetadataName(ResearchPaperAttr);

        var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);

        // Derive the manifest's relative file path here; a Location cannot live in the pipeline.
        var filePath = string.Empty;
        var location = modelClass.Locations.FirstOrDefault();
        if (location is not null && location.SourceTree is not null)
        {
            filePath = location.SourceTree.FilePath.Replace("\\", "/");
            var srcIdx = filePath.IndexOf("/src/", System.StringComparison.OrdinalIgnoreCase);
            if (srcIdx >= 0)
            {
                filePath = filePath.Substring(srcIdx + 1);
            }
        }

        // When the core attributes are absent the registry is empty, but the manifest still lists
        // the class -- so return a metadata-free entry rather than null.
        if (domainAttrSymbol is null || categoryAttrSymbol is null ||
            taskAttrSymbol is null || complexityAttrSymbol is null ||
            inputAttrSymbol is null)
        {
            return ModelEntryData.MetadataFree(fullName, modelClass.Name, modelClass.TypeParameters.Length, filePath);
        }

        return ExtractMetadata(
            modelClass, fullName, filePath,
            domainAttrSymbol, categoryAttrSymbol, taskAttrSymbol,
            complexityAttrSymbol, inputAttrSymbol, paperAttrSymbol);
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

    private static void Emit(SourceProductionContext context, ImmutableArray<ModelEntryData> candidates)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitEmptyRegistry(context);
            EmitDiscoveryManifest(context, ImmutableArray<ModelEntryData>.Empty, new HashSet<string>());
            return;
        }

        var entries = new List<ModelEntryData>();
        var deduped = new List<ModelEntryData>();
        var seen = new HashSet<string>();
        var manifestAnnotatedNames = new HashSet<string>();

        foreach (var entry in candidates)
        {
            if (entry.FullyQualifiedName.Length == 0)
                continue;

            // Deduplicate (same class can appear from multiple syntax trees for partial classes)
            if (!seen.Add(entry.FullyQualifiedName))
                continue;

            deduped.Add(entry);

            // Track models with at least one metadata attribute for manifest progress
            if (entry.HasAnyMetadata)
            {
                manifestAnnotatedNames.Add(entry.FullyQualifiedName);
            }

            // Only include fully-annotated models in the registry to avoid default enum values
            if (entry.HasAllRequiredMetadata)
            {
                entries.Add(entry);
            }
        }

        // Sort entries by fully-qualified name for deterministic output
        entries.Sort((a, b) => string.Compare(a.FullyQualifiedName, b.FullyQualifiedName, System.StringComparison.Ordinal));

        EmitRegistry(context, entries);

        // Emit discovery manifest listing ALL concrete IFullModel classes with file paths
        EmitDiscoveryManifest(context, deduped.ToImmutableArray(), manifestAnnotatedNames);
    }

    private static ModelEntryData ExtractMetadata(
        INamedTypeSymbol modelClass,
        string fullyQualifiedName,
        string relativeFilePath,
        INamedTypeSymbol domainAttrSymbol,
        INamedTypeSymbol categoryAttrSymbol,
        INamedTypeSymbol taskAttrSymbol,
        INamedTypeSymbol complexityAttrSymbol,
        INamedTypeSymbol inputAttrSymbol,
        INamedTypeSymbol? paperAttrSymbol)
    {
        var attributes = modelClass.GetAttributes();
        var domains = new List<int>();
        var categories = new List<int>();
        var tasks = new List<int>();
        var papers = new List<PaperData>();
        int complexity = 0;
        bool hasComplexity = false;
        string inputTypeName = string.Empty;
        string outputTypeName = string.Empty;
        string summary = string.Empty;
        string beginnerGuide = string.Empty;

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
                        domains.Add(intVal);
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
                        categories.Add(intVal);
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
                        tasks.Add(intVal);
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
                        complexity = intVal;
                        hasComplexity = true;
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
                        inputTypeName = inputType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                    }
                    if (outputType is not null)
                    {
                        outputTypeName = outputType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                    }
                }
            }
            else if (paperAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, paperAttrSymbol))
            {
                string paperTitle = string.Empty;
                string paperUrl = string.Empty;
                int paperYear = 0;
                string paperAuthors = string.Empty;
                if (attr.ConstructorArguments.Length >= 2)
                {
                    paperTitle = attr.ConstructorArguments[0].Value as string ?? string.Empty;
                    paperUrl = attr.ConstructorArguments[1].Value as string ?? string.Empty;
                }
                // Check named arguments for Year and Authors
                foreach (var named in attr.NamedArguments)
                {
                    if (named.Key == "Year" && named.Value.Value is int year)
                    {
                        paperYear = year;
                    }
                    else if (named.Key == "Authors" && named.Value.Value is string authors)
                    {
                        paperAuthors = authors;
                    }
                }
                papers.Add(new PaperData(paperTitle, paperUrl, paperYear, paperAuthors));
            }
        }

        // Extract XML documentation
        var xmlDoc = modelClass.GetDocumentationCommentXml();
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            summary = ExtractXmlElement(xmlDoc, "summary");
            beginnerGuide = ExtractBeginnerRemarks(xmlDoc);
        }

        return new ModelEntryData(
            fullyQualifiedName,
            modelClass.Name,
            modelClass.TypeParameters.Length,
            relativeFilePath,
            domains.ToImmutableArray(),
            categories.ToImmutableArray(),
            tasks.ToImmutableArray(),
            complexity,
            hasComplexity,
            inputTypeName,
            outputTypeName,
            papers.ToImmutableArray(),
            summary,
            beginnerGuide);
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
        // Extract raw remarks content (before CleanXmlText) to preserve XML tags for parsing
        var remarksContent = ExtractRawXmlElement(xml, "remarks");
        if (string.IsNullOrWhiteSpace(remarksContent))
            return string.Empty;

        var beginnerIdx = remarksContent.IndexOf("For Beginners", System.StringComparison.OrdinalIgnoreCase);
        if (beginnerIdx < 0)
            return string.Empty;

        // Take text after "For Beginners:</b>" or "For Beginners:" marker
        var closeBIdx = remarksContent.IndexOf("</b>", beginnerIdx, System.StringComparison.Ordinal);
        var colonIdx = remarksContent.IndexOf(":", beginnerIdx, System.StringComparison.Ordinal);

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

        // Take until the closing </para> of the beginner paragraph only
        var endIdx = remarksContent.IndexOf("</para>", contentStart, System.StringComparison.Ordinal);
        if (endIdx < 0)
            endIdx = remarksContent.Length;

        var text = remarksContent.Substring(contentStart, endIdx - contentStart);
        return CleanXmlText(text);
    }

    private static string ExtractRawXmlElement(string xml, string elementName)
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

        return xml.Substring(startIdx, endIdx - startIdx);
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

        // ResearchPaperEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Represents an academic paper reference for a model.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class ResearchPaperEntry");
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
        sb.AppendLine("    public ResearchPaperEntry(string title, string url, int year, string authors)");
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
        sb.AppendLine("internal sealed class ModelMetadataEntry");
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
        sb.AppendLine("    public IReadOnlyList<ResearchPaperEntry> Papers { get; }");
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
        sb.AppendLine("        ResearchPaperEntry[] papers, string summary, string beginnerGuide)");
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
        sb.AppendLine("internal static class ModelMetadataRegistry");
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
        sb.AppendLine("    private static readonly Lazy<Dictionary<string, List<ModelMetadataEntry>>> _byClassName =");
        sb.AppendLine("        new Lazy<Dictionary<string, List<ModelMetadataEntry>>>(BuildByClassName);");
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

        // BuildByClassName
        sb.AppendLine("    private static Dictionary<string, List<ModelMetadataEntry>> BuildByClassName()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<string, List<ModelMetadataEntry>>(StringComparer.Ordinal);");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (!dict.TryGetValue(entry.ClassName, out var list))");
        sb.AppendLine("            {");
        sb.AppendLine("                list = new List<ModelMetadataEntry>();");
        sb.AppendLine("                dict[entry.ClassName] = list;");
        sb.AppendLine("            }");
        sb.AppendLine("            list.Add(entry);");
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
        sb.AppendLine("        if (_byClassName.Value.TryGetValue(className, out var list))");
        sb.AppendLine("            return list;");
        sb.AppendLine("        return Array.Empty<ModelMetadataEntry>();");
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
        if (entry.Domains.Length == 0)
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
        if (entry.Categories.Length == 0)
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
        if (entry.Tasks.Length == 0)
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
        if (entry.Papers.Length == 0)
        {
            sb.AppendLine("            System.Array.Empty<ResearchPaperEntry>(),");
        }
        else
        {
            sb.AppendLine("            new ResearchPaperEntry[]");
            sb.AppendLine("            {");
            foreach (var paper in entry.Papers)
            {
                sb.AppendLine($"                new ResearchPaperEntry({EscapeString(paper.Title)}, {EscapeString(paper.Url)}, {paper.Year}, {EscapeString(paper.Authors)}),");
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

    /// <summary>
    /// One discovered model, as plain values. RelativeFilePath is carried instead of a Location:
    /// a Location holds its SyntaxTree and would root the whole Compilation in the pipeline.
    /// </summary>
    private sealed class ModelEntryData : System.IEquatable<ModelEntryData>
    {
        public static readonly ModelEntryData Empty = MetadataFree(string.Empty, string.Empty, 0, string.Empty);

        /// <summary>An entry with no metadata -- still listed by the discovery manifest.</summary>
        public static ModelEntryData MetadataFree(string fullyQualifiedName, string className, int typeParameterCount, string relativeFilePath)
            => new(fullyQualifiedName, className, typeParameterCount, relativeFilePath,
                ImmutableArray<int>.Empty, ImmutableArray<int>.Empty, ImmutableArray<int>.Empty,
                0, false, string.Empty, string.Empty, ImmutableArray<PaperData>.Empty,
                string.Empty, string.Empty);

        public ModelEntryData(
            string fullyQualifiedName,
            string className,
            int typeParameterCount,
            string relativeFilePath,
            ImmutableArray<int> domains,
            ImmutableArray<int> categories,
            ImmutableArray<int> tasks,
            int complexity,
            bool hasComplexity,
            string inputTypeName,
            string outputTypeName,
            ImmutableArray<PaperData> papers,
            string summary,
            string beginnerGuide)
        {
            FullyQualifiedName = fullyQualifiedName;
            ClassName = className;
            TypeParameterCount = typeParameterCount;
            RelativeFilePath = relativeFilePath;
            Domains = domains.IsDefault ? ImmutableArray<int>.Empty : domains;
            Categories = categories.IsDefault ? ImmutableArray<int>.Empty : categories;
            Tasks = tasks.IsDefault ? ImmutableArray<int>.Empty : tasks;
            Complexity = complexity;
            HasComplexity = hasComplexity;
            InputTypeName = inputTypeName;
            OutputTypeName = outputTypeName;
            Papers = papers.IsDefault ? ImmutableArray<PaperData>.Empty : papers;
            Summary = summary;
            BeginnerGuide = beginnerGuide;
        }

        public string FullyQualifiedName { get; }
        public string ClassName { get; }
        public int TypeParameterCount { get; }
        public string RelativeFilePath { get; }
        public ImmutableArray<int> Domains { get; }
        public ImmutableArray<int> Categories { get; }
        public ImmutableArray<int> Tasks { get; }
        public int Complexity { get; }
        public bool HasComplexity { get; }
        public string InputTypeName { get; }
        public string OutputTypeName { get; }
        public ImmutableArray<PaperData> Papers { get; }
        public string Summary { get; }
        public string BeginnerGuide { get; }

        public bool HasAnyMetadata =>
            Domains.Length > 0 || Categories.Length > 0 || Tasks.Length > 0 || HasComplexity ||
            !string.IsNullOrEmpty(InputTypeName) || Papers.Length > 0;

        public bool HasAllRequiredMetadata =>
            Domains.Length > 0 && Categories.Length > 0 && Tasks.Length > 0 && HasComplexity &&
            !string.IsNullOrEmpty(InputTypeName);

        public bool Equals(ModelEntryData? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            if (!string.Equals(FullyQualifiedName, other.FullyQualifiedName, System.StringComparison.Ordinal)) return false;
            if (!string.Equals(ClassName, other.ClassName, System.StringComparison.Ordinal)) return false;
            if (TypeParameterCount != other.TypeParameterCount) return false;
            if (!string.Equals(RelativeFilePath, other.RelativeFilePath, System.StringComparison.Ordinal)) return false;
            if (Complexity != other.Complexity || HasComplexity != other.HasComplexity) return false;
            if (!string.Equals(InputTypeName, other.InputTypeName, System.StringComparison.Ordinal)) return false;
            if (!string.Equals(OutputTypeName, other.OutputTypeName, System.StringComparison.Ordinal)) return false;
            if (!string.Equals(Summary, other.Summary, System.StringComparison.Ordinal)) return false;
            if (!string.Equals(BeginnerGuide, other.BeginnerGuide, System.StringComparison.Ordinal)) return false;
            if (!IntsEqual(Domains, other.Domains)) return false;
            if (!IntsEqual(Categories, other.Categories)) return false;
            if (!IntsEqual(Tasks, other.Tasks)) return false;
            if (Papers.Length != other.Papers.Length) return false;
            for (int i = 0; i < Papers.Length; i++)
            {
                if (!Papers[i].Equals(other.Papers[i])) return false;
            }
            return true;
        }

        public override bool Equals(object? obj) => Equals(obj as ModelEntryData);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 31) + FullyQualifiedName.GetHashCode();
                hash = (hash * 31) + ClassName.GetHashCode();
                hash = (hash * 31) + TypeParameterCount;
                hash = (hash * 31) + RelativeFilePath.GetHashCode();
                hash = (hash * 31) + Complexity;
                hash = (hash * 31) + (HasComplexity ? 1 : 0);
                hash = (hash * 31) + Domains.Length;
                hash = (hash * 31) + Categories.Length;
                hash = (hash * 31) + Tasks.Length;
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
    }

    /// <remarks>
    /// CONSTRUCTOR-INITIALISED AND GET-ONLY ON PURPOSE. This type is an ELEMENT of an
    /// ImmutableArray held by a cached pipeline entry, and ImmutableArray freezes the SEQUENCE, not
    /// the elements. While these had settable properties, an element could still be mutated after
    /// Roslyn had compared the entry, which makes the entry's Equals and GetHashCode unstable for
    /// exactly the cached state this refactor is trying to make comparable.
    /// </remarks>
    private sealed class PaperData : System.IEquatable<PaperData>
    {
        public PaperData(string title, string url, int year, string authors)
        {
            Title = title;
            Url = url;
            Year = year;
            Authors = authors;
        }

        public string Title { get; }
        public string Url { get; }
        public int Year { get; }
        public string Authors { get; }

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

    /// <summary>
    /// Emits a discovery manifest listing ALL concrete IFullModel implementations
    /// with their source file paths, class names, and annotation status.
    /// </summary>
    private static void EmitDiscoveryManifest(
        SourceProductionContext context,
        ImmutableArray<ModelEntryData> candidates,
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
        sb.AppendLine("internal static class ModelDiscoveryManifest");
        sb.AppendLine("{");

        var manifestEntries = new List<(string className, string fullName, string filePath, bool hasAttributes)>();
        var seen = new HashSet<string>();

        foreach (var entry in candidates)
        {
            if (entry.FullyQualifiedName.Length == 0)
                continue;
            if (!seen.Add(entry.FullyQualifiedName))
                continue;

            // The relative path was derived in Analyze, where the Location was still available.
            var hasAttributes = annotatedFullNames.Contains(entry.FullyQualifiedName);
            manifestEntries.Add((entry.ClassName, entry.FullyQualifiedName, entry.RelativeFilePath, hasAttributes));
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
