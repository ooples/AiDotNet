using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that auto-generates documentation data from model
/// metadata attributes and XML doc comments for building model catalogs and selection guides.
/// </summary>
/// <remarks>
/// <para>
/// Produces a static <c>ModelDocumentation</c> class with pre-computed documentation data
/// grouped by domain, task, and complexity. Consumers can use this data to render markdown
/// catalogs, selection guides, or web-based model browsers at runtime without reflection.
/// </para>
/// </remarks>
[Generator]
public class DocumentationGenerator : IIncrementalGenerator
{
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";
    private const string ModelDomainAttr = "AiDotNet.Attributes.ModelDomainAttribute";
    private const string ModelCategoryAttr = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttr = "AiDotNet.Attributes.ModelTaskAttribute";
    private const string ModelComplexityAttr = "AiDotNet.Attributes.ModelComplexityAttribute";
    private const string ResearchPaperAttr = "AiDotNet.Attributes.ResearchPaperAttribute";
    private const string ModelInputAttr = "AiDotNet.Attributes.ModelInputAttribute";
    private const string ModelMetadataExemptAttr = "AiDotNet.Attributes.ModelMetadataExemptAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Values, not symbols -- see DiscoveryApiGenerator for the full rationale.
        var docEntries = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => Analyze(ctx))
            .Where(static e => e is not null)
            .Select(static (e, _) => e ?? DocEntry.Empty);

        context.RegisterSourceOutput(docEntries.Collect(), static (spc, collected) => Emit(spc, collected));
    }

    /// <summary>
    /// Resolves one candidate class into a value-equatable entry, or null when it is not
    /// documentable. All symbol access is confined to this method.
    /// </summary>
    private static DocEntry? Analyze(GeneratorSyntaxContext ctx)
    {
        if (GetModelClassOrNull(ctx) is not INamedTypeSymbol modelClass)
            return null;

        var compilation = ctx.SemanticModel.Compilation;
        var domainAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ModelDomainAttr);
        var categoryAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ModelCategoryAttr);
        var taskAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ModelTaskAttr);
        var complexityAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ModelComplexityAttr);
        var paperAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ResearchPaperAttr);
        var inputAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ModelInputAttr);
        var exemptAttrSymbol = GeneratorHelpers.ResolveSourceType(compilation, ModelMetadataExemptAttr);

        if (domainAttrSymbol is null || categoryAttrSymbol is null ||
            taskAttrSymbol is null || complexityAttrSymbol is null)
        {
            return null;
        }

        if (exemptAttrSymbol is not null && HasAttribute(modelClass.GetAttributes(), exemptAttrSymbol))
            return null;

        var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        var entry = ExtractDocEntry(modelClass, fullName,
            domainAttrSymbol, categoryAttrSymbol, taskAttrSymbol,
            complexityAttrSymbol, paperAttrSymbol, inputAttrSymbol);

        return entry.Domains.Length > 0 && entry.Tasks.Length > 0 && entry.HasComplexity ? entry : null;
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
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(IFullModelName, System.StringComparison.Ordinal))
            {
                return symbol;
            }
        }
        return null;
    }

    /// <summary>
    /// Emits from already-resolved entries; symbol work and the admission rules moved into
    /// <see cref="Analyze"/>, while dedupe and ordering stay here where the whole set is visible.
    /// </summary>
    private static void Emit(SourceProductionContext context, ImmutableArray<DocEntry> candidates)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitDocumentationClass(context, new List<DocEntry>());
            return;
        }

        var entries = new List<DocEntry>();
        var seen = new HashSet<string>();

        foreach (var entry in candidates)
        {
            if (entry.FullyQualifiedName.Length == 0) continue;
            if (!seen.Add(entry.FullyQualifiedName)) continue;

            entries.Add(entry);
        }

        entries.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        EmitDocumentationClass(context, entries);
    }

    private static DocEntry ExtractDocEntry(
        INamedTypeSymbol modelClass,
        string fullyQualifiedName,
        INamedTypeSymbol domainAttrSymbol,
        INamedTypeSymbol categoryAttrSymbol,
        INamedTypeSymbol taskAttrSymbol,
        INamedTypeSymbol complexityAttrSymbol,
        INamedTypeSymbol? paperAttrSymbol,
        INamedTypeSymbol? inputAttrSymbol)
    {
        var domains = new List<int>();
        var categories = new List<int>();
        var tasks = new List<int>();
        var papers = new List<PaperInfo>();
        int complexity = 0;
        bool hasComplexity = false;
        string inputTypeName = string.Empty;
        string outputTypeName = string.Empty;
        string summary = string.Empty;
        string beginnerGuide = string.Empty;

        foreach (var attr in modelClass.GetAttributes())
        {
            if (attr.AttributeClass is null) continue;

            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, domainAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int d)
                    domains.Add(d);
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, categoryAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int c)
                    categories.Add(c);
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, taskAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int t)
                    tasks.Add(t);
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, complexityAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int cx)
                {
                    complexity = cx;
                    hasComplexity = true;
                }
            }
            else if (paperAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, paperAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 2)
                {
                    var title = attr.ConstructorArguments[0].Value as string ?? string.Empty;
                    var url = attr.ConstructorArguments[1].Value as string ?? string.Empty;
                    int year = 0;
                    var authors = string.Empty;
                    foreach (var named in attr.NamedArguments)
                    {
                        if (named.Key == "Year" && named.Value.Value is int y) year = y;
                        else if (named.Key == "Authors" && named.Value.Value is string a) authors = a;
                    }
                    papers.Add(new PaperInfo(title, url, year, authors));
                }
            }
            else if (inputAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, inputAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 2)
                {
                    var inputType = attr.ConstructorArguments[0].Value as INamedTypeSymbol;
                    var outputType = attr.ConstructorArguments[1].Value as INamedTypeSymbol;
                    if (inputType is not null)
                        inputTypeName = inputType.Name;
                    if (outputType is not null)
                        outputTypeName = outputType.Name;
                }
            }
        }

        // Extract XML docs (with syntax tree fallback for CI/net471)
        var xmlDoc = modelClass.GetDocumentationCommentXml();
        if (string.IsNullOrWhiteSpace(xmlDoc))
        {
            xmlDoc = GeneratorHelpers.ExtractXmlDocFromSyntax(modelClass);
        }
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            summary = ExtractXmlElement(xmlDoc, "summary");
            beginnerGuide = ExtractBeginnerRemarks(xmlDoc);
        }

        return new DocEntry(
            modelClass.Name,
            fullyQualifiedName,
            modelClass.TypeParameters.Length,
            domains.ToImmutableArray(),
            categories.ToImmutableArray(),
            tasks.ToImmutableArray(),
            complexity,
            hasComplexity,
            papers.ToImmutableArray(),
            inputTypeName,
            outputTypeName,
            summary,
            beginnerGuide);
    }

    private static void EmitDocumentationClass(SourceProductionContext context, List<DocEntry> entries)
    {
        var domainNames = GeneratorHelpers.DomainNames;

        var taskNames = GeneratorHelpers.TaskNames;
        var complexityNames = GeneratorHelpers.ComplexityNames;
        var categoryNames = GeneratorHelpers.CategoryNames;

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

        // ModelDocEntry class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Documentation entry for a single model, used to generate catalogs and guides.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class ModelDocEntry");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Gets the model class name.</summary>");
        sb.AppendLine("    public string ClassName { get; }");
        sb.AppendLine("    /// <summary>Gets the fully-qualified type name.</summary>");
        sb.AppendLine("    public string TypeName { get; }");
        sb.AppendLine("    /// <summary>Gets the model type.</summary>");
        sb.AppendLine("    public Type ModelType { get; }");
        sb.AppendLine("    /// <summary>Gets the complexity level.</summary>");
        sb.AppendLine("    public ModelComplexity Complexity { get; }");
        sb.AppendLine("    /// <summary>Gets the domains this model belongs to.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelDomain> Domains { get; }");
        sb.AppendLine("    /// <summary>Gets the categories this model belongs to.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelCategory> Categories { get; }");
        sb.AppendLine("    /// <summary>Gets the tasks this model performs.</summary>");
        sb.AppendLine("    public IReadOnlyList<ModelTask> Tasks { get; }");
        sb.AppendLine("    /// <summary>Gets the summary description.</summary>");
        sb.AppendLine("    public string Summary { get; }");
        sb.AppendLine("    /// <summary>Gets the beginner-friendly guide.</summary>");
        sb.AppendLine("    public string BeginnerGuide { get; }");
        sb.AppendLine("    /// <summary>Gets the paper title (first paper, if any).</summary>");
        sb.AppendLine("    public string PaperTitle { get; }");
        sb.AppendLine("    /// <summary>Gets the paper URL (first paper, if any).</summary>");
        sb.AppendLine("    public string PaperUrl { get; }");
        sb.AppendLine("    /// <summary>Gets the expected input type name.</summary>");
        sb.AppendLine("    public string InputType { get; }");
        sb.AppendLine("    /// <summary>Gets the expected output type name.</summary>");
        sb.AppendLine("    public string OutputType { get; }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Initializes a new documentation entry.</summary>");
        sb.AppendLine("    public ModelDocEntry(");
        sb.AppendLine("        string className, string typeName, Type modelType,");
        sb.AppendLine("        ModelComplexity complexity,");
        sb.AppendLine("        ModelDomain[] domains, ModelCategory[] categories, ModelTask[] tasks,");
        sb.AppendLine("        string summary, string beginnerGuide,");
        sb.AppendLine("        string paperTitle, string paperUrl,");
        sb.AppendLine("        string inputType, string outputType)");
        sb.AppendLine("    {");
        sb.AppendLine("        ClassName = className;");
        sb.AppendLine("        TypeName = typeName;");
        sb.AppendLine("        ModelType = modelType;");
        sb.AppendLine("        Complexity = complexity;");
        sb.AppendLine("        Domains = domains;");
        sb.AppendLine("        Categories = categories;");
        sb.AppendLine("        Tasks = tasks;");
        sb.AppendLine("        Summary = summary;");
        sb.AppendLine("        BeginnerGuide = beginnerGuide;");
        sb.AppendLine("        PaperTitle = paperTitle;");
        sb.AppendLine("        PaperUrl = paperUrl;");
        sb.AppendLine("        InputType = inputType;");
        sb.AppendLine("        OutputType = outputType;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ModelDocumentation static class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated documentation data for all annotated models.");
        sb.AppendLine("/// Use this to build model catalogs, selection guides, and comparison tables.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class ModelDocumentation");
        sb.AppendLine("{");

        sb.AppendLine($"    /// <summary>Total documented models.</summary>");
        sb.AppendLine($"    public const int ModelCount = {entries.Count};");
        sb.AppendLine();

        // All entries
        sb.AppendLine("    /// <summary>All documented model entries.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ModelDocEntry> All { get; } = new ModelDocEntry[]");
        sb.AppendLine("    {");

        foreach (var entry in entries)
        {
            var typeExpr = BuildTypeOfExpression(entry);
            var complexityEnum = complexityNames.TryGetValue(entry.Complexity, out var cxName) ? cxName : "Low";

            sb.AppendLine("        new ModelDocEntry(");
            sb.AppendLine($"            {EscapeString(StripGenericArity(entry.ClassName))},");
            sb.AppendLine($"            {EscapeString(entry.FullyQualifiedName)},");
            sb.AppendLine($"            {typeExpr},");
            sb.AppendLine($"            ModelComplexity.{complexityEnum},");

            // Domains
            if (entry.Domains.Length == 0)
                sb.AppendLine("            Array.Empty<ModelDomain>(),");
            else
            {
                sb.Append("            new ModelDomain[] { ");
                sb.Append(string.Join(", ", entry.Domains.Select(d =>
                    domainNames.TryGetValue(d, out var n) ? $"ModelDomain.{n}" : $"(ModelDomain){d}")));
                sb.AppendLine(" },");
            }

            // Categories
            if (entry.Categories.Length == 0)
                sb.AppendLine("            Array.Empty<ModelCategory>(),");
            else
            {
                sb.Append("            new ModelCategory[] { ");
                sb.Append(string.Join(", ", entry.Categories.Select(c =>
                    categoryNames.TryGetValue(c, out var n) ? $"ModelCategory.{n}" : $"(ModelCategory){c}")));
                sb.AppendLine(" },");
            }

            // Tasks
            if (entry.Tasks.Length == 0)
                sb.AppendLine("            Array.Empty<ModelTask>(),");
            else
            {
                sb.Append("            new ModelTask[] { ");
                sb.Append(string.Join(", ", entry.Tasks.Select(t =>
                    taskNames.TryGetValue(t, out var n) ? $"ModelTask.{n}" : $"(ModelTask){t}")));
                sb.AppendLine(" },");
            }

            sb.AppendLine($"            {EscapeString(entry.Summary)},");
            sb.AppendLine($"            {EscapeString(entry.BeginnerGuide)},");

            var firstPaper = entry.Papers.Length > 0 ? entry.Papers[0] : null;
            sb.AppendLine($"            {EscapeString(firstPaper?.Title ?? string.Empty)},");
            sb.AppendLine($"            {EscapeString(firstPaper?.Url ?? string.Empty)},");
            sb.AppendLine($"            {EscapeString(entry.InputTypeName)},");
            sb.AppendLine($"            {EscapeString(entry.OutputTypeName)}");
            sb.AppendLine("        ),");
        }

        sb.AppendLine("    };");
        sb.AppendLine();

        // Query methods
        sb.AppendLine("    /// <summary>Gets all documented models for a specific domain.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ModelDocEntry> GetByDomain(ModelDomain domain)");
        sb.AppendLine("    {");
        sb.AppendLine("        var result = new List<ModelDocEntry>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var d in entry.Domains)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (d == domain) { result.Add(entry); break; }");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
        sb.AppendLine();

        sb.AppendLine("    /// <summary>Gets all documented models for a specific task.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ModelDocEntry> GetByTask(ModelTask task)");
        sb.AppendLine("    {");
        sb.AppendLine("        var result = new List<ModelDocEntry>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var t in entry.Tasks)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (t == task) { result.Add(entry); break; }");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
        sb.AppendLine();

        sb.AppendLine("    /// <summary>Gets all documented models with a specific complexity.</summary>");
        sb.AppendLine("    public static IReadOnlyList<ModelDocEntry> GetByComplexity(ModelComplexity complexity)");
        sb.AppendLine("    {");
        sb.AppendLine("        var result = new List<ModelDocEntry>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (entry.Complexity == complexity)");
        sb.AppendLine("                result.Add(entry);");
        sb.AppendLine("        }");
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Selection guide helper
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Gets model recommendations for a task, grouped by complexity level.");
        sb.AppendLine("    /// Useful for building selection guides.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    public static Dictionary<ModelComplexity, List<ModelDocEntry>> GetSelectionGuide(ModelTask task)");
        sb.AppendLine("    {");
        sb.AppendLine("        var guide = new Dictionary<ModelComplexity, List<ModelDocEntry>>();");
        sb.AppendLine("        foreach (var entry in All)");
        sb.AppendLine("        {");
        sb.AppendLine("            bool matchesTask = false;");
        sb.AppendLine("            foreach (var t in entry.Tasks)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (t == task) { matchesTask = true; break; }");
        sb.AppendLine("            }");
        sb.AppendLine("            if (!matchesTask) continue;");
        sb.AppendLine("            if (!guide.TryGetValue(entry.Complexity, out var list))");
        sb.AppendLine("            {");
        sb.AppendLine("                list = new List<ModelDocEntry>();");
        sb.AppendLine("                guide[entry.Complexity] = list;");
        sb.AppendLine("            }");
        sb.AppendLine("            list.Add(entry);");
        sb.AppendLine("        }");
        sb.AppendLine("        return guide;");
        sb.AppendLine("    }");

        sb.AppendLine("}");

        context.AddSource("ModelDocumentation.g.cs", sb.ToString());
    }

    private static string BuildTypeOfExpression(DocEntry entry)
    {
        var typeName = StripGenericSuffix(entry.FullyQualifiedName);

        if (entry.TypeParameterCount > 0)
        {
            var commas = new string(',', entry.TypeParameterCount - 1);
            return $"typeof({typeName}<{commas}>)";
        }
        return $"typeof({typeName})";
    }

    private static string StripGenericSuffix(string fullyQualifiedName)
    {
        var name = fullyQualifiedName;
        if (name.StartsWith("global::", System.StringComparison.Ordinal))
            name = name.Substring("global::".Length);
        var angleBracketIdx = name.IndexOf('<');
        if (angleBracketIdx >= 0)
            name = name.Substring(0, angleBracketIdx);
        return name;
    }

    private static string StripGenericArity(string className)
    {
        var idx = className.IndexOf('`');
        return idx >= 0 ? className.Substring(0, idx) : className;
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

    private static string ExtractXmlElement(string xml, string elementName)
    {
        var startTag = "<" + elementName + ">";
        var endTag = "</" + elementName + ">";
        var startIdx = xml.IndexOf(startTag, System.StringComparison.Ordinal);
        if (startIdx < 0) return string.Empty;
        startIdx += startTag.Length;
        var endIdx = xml.IndexOf(endTag, startIdx, System.StringComparison.Ordinal);
        if (endIdx < 0) return string.Empty;
        return CleanXmlText(xml.Substring(startIdx, endIdx - startIdx));
    }

    private static string ExtractBeginnerRemarks(string xml)
    {
        var remarksContent = ExtractRawXmlElement(xml, "remarks");
        if (string.IsNullOrWhiteSpace(remarksContent))
            return string.Empty;

        var beginnerIdx = remarksContent.IndexOf("For Beginners", System.StringComparison.OrdinalIgnoreCase);
        if (beginnerIdx < 0)
            return string.Empty;

        var closeBIdx = remarksContent.IndexOf("</b>", beginnerIdx, System.StringComparison.Ordinal);
        var colonIdx = remarksContent.IndexOf(":", beginnerIdx, System.StringComparison.Ordinal);

        int contentStart;
        if (closeBIdx >= 0 && (colonIdx < 0 || closeBIdx < colonIdx))
            contentStart = closeBIdx + 4;
        else if (colonIdx >= 0)
            contentStart = colonIdx + 1;
        else
            contentStart = beginnerIdx + "For Beginners".Length;

        var endIdx = remarksContent.IndexOf("</para>", contentStart, System.StringComparison.Ordinal);
        if (endIdx < 0) endIdx = remarksContent.Length;

        return CleanXmlText(remarksContent.Substring(contentStart, endIdx - contentStart));
    }

    private static string ExtractRawXmlElement(string xml, string elementName)
    {
        var startTag = "<" + elementName + ">";
        var endTag = "</" + elementName + ">";
        var startIdx = xml.IndexOf(startTag, System.StringComparison.Ordinal);
        if (startIdx < 0) return string.Empty;
        startIdx += startTag.Length;
        var endIdx = xml.IndexOf(endTag, startIdx, System.StringComparison.Ordinal);
        if (endIdx < 0) return string.Empty;
        return xml.Substring(startIdx, endIdx - startIdx);
    }

    private static string CleanXmlText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        var sb = new StringBuilder(text.Length);
        var inTag = false;
        foreach (var c in text)
        {
            if (c == '<') { inTag = true; continue; }
            if (c == '>') { inTag = false; continue; }
            if (!inTag) sb.Append(c);
        }

        var raw = sb.ToString();
        var normalized = new StringBuilder(raw.Length);
        bool prevSpace = false;
        foreach (char c in raw)
        {
            if (c == ' ' || c == '\r' || c == '\n' || c == '\t')
            {
                if (!prevSpace) { normalized.Append(' '); prevSpace = true; }
            }
            else { normalized.Append(c); prevSpace = false; }
        }

        return normalized.ToString().Trim();
    }

    private static bool HasAttribute(ImmutableArray<AttributeData> attributes, INamedTypeSymbol attributeType)
    {
        foreach (var attr in attributes)
        {
            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, attributeType))
                return true;
        }
        return false;
    }

    /// <summary>
    /// One documented model, as plain values. Immutable and structurally equal so the incremental
    /// pipeline can actually compare it; Papers is nested, so PaperInfo is equatable too.
    /// </summary>
    private sealed class DocEntry : System.IEquatable<DocEntry>
    {
        public static readonly DocEntry Empty = new(
            string.Empty, string.Empty, 0,
            ImmutableArray<int>.Empty, ImmutableArray<int>.Empty, ImmutableArray<int>.Empty,
            0, false, ImmutableArray<PaperInfo>.Empty,
            string.Empty, string.Empty, string.Empty, string.Empty);

        public DocEntry(
            string className,
            string fullyQualifiedName,
            int typeParameterCount,
            ImmutableArray<int> domains,
            ImmutableArray<int> categories,
            ImmutableArray<int> tasks,
            int complexity,
            bool hasComplexity,
            ImmutableArray<PaperInfo> papers,
            string inputTypeName,
            string outputTypeName,
            string summary,
            string beginnerGuide)
        {
            ClassName = className;
            FullyQualifiedName = fullyQualifiedName;
            TypeParameterCount = typeParameterCount;
            Domains = domains.IsDefault ? ImmutableArray<int>.Empty : domains;
            Categories = categories.IsDefault ? ImmutableArray<int>.Empty : categories;
            Tasks = tasks.IsDefault ? ImmutableArray<int>.Empty : tasks;
            Complexity = complexity;
            HasComplexity = hasComplexity;
            Papers = papers.IsDefault ? ImmutableArray<PaperInfo>.Empty : papers;
            InputTypeName = inputTypeName;
            OutputTypeName = outputTypeName;
            Summary = summary;
            BeginnerGuide = beginnerGuide;
        }

        public string ClassName { get; }
        public string FullyQualifiedName { get; }
        public int TypeParameterCount { get; }
        public ImmutableArray<int> Domains { get; }
        public ImmutableArray<int> Categories { get; }
        public ImmutableArray<int> Tasks { get; }
        public int Complexity { get; }
        public bool HasComplexity { get; }
        public ImmutableArray<PaperInfo> Papers { get; }
        public string InputTypeName { get; }
        public string OutputTypeName { get; }
        public string Summary { get; }
        public string BeginnerGuide { get; }

        public bool Equals(DocEntry? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            return string.Equals(ClassName, other.ClassName, System.StringComparison.Ordinal)
                && string.Equals(FullyQualifiedName, other.FullyQualifiedName, System.StringComparison.Ordinal)
                && TypeParameterCount == other.TypeParameterCount
                && Complexity == other.Complexity
                && HasComplexity == other.HasComplexity
                && string.Equals(InputTypeName, other.InputTypeName, System.StringComparison.Ordinal)
                && string.Equals(OutputTypeName, other.OutputTypeName, System.StringComparison.Ordinal)
                && string.Equals(Summary, other.Summary, System.StringComparison.Ordinal)
                && string.Equals(BeginnerGuide, other.BeginnerGuide, System.StringComparison.Ordinal)
                && IntsEqual(Domains, other.Domains)
                && IntsEqual(Categories, other.Categories)
                && IntsEqual(Tasks, other.Tasks)
                && ItemsEqual(Papers, other.Papers);
        }

        public override bool Equals(object? obj) => Equals(obj as DocEntry);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 31) + ClassName.GetHashCode();
                hash = (hash * 31) + FullyQualifiedName.GetHashCode();
                hash = (hash * 31) + TypeParameterCount;
                hash = (hash * 31) + Complexity;
                hash = (hash * 31) + (HasComplexity ? 1 : 0);
                hash = (hash * 31) + InputTypeName.GetHashCode();
                hash = (hash * 31) + OutputTypeName.GetHashCode();
                hash = (hash * 31) + Summary.GetHashCode();
                hash = (hash * 31) + BeginnerGuide.GetHashCode();
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

    /// <remarks>
    /// CONSTRUCTOR-INITIALISED AND GET-ONLY ON PURPOSE. This type is an ELEMENT of an
    /// ImmutableArray held by a cached pipeline entry, and ImmutableArray freezes the SEQUENCE, not
    /// the elements. While these had settable properties, an element could still be mutated after
    /// Roslyn had compared the entry, which makes the entry's Equals and GetHashCode unstable for
    /// exactly the cached state this refactor is trying to make comparable.
    /// </remarks>
    private sealed class PaperInfo : System.IEquatable<PaperInfo>
    {
        public PaperInfo(string title, string url, int year, string authors)
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

        public bool Equals(PaperInfo? other)
            => other is not null
            && string.Equals(Title, other.Title, System.StringComparison.Ordinal)
            && string.Equals(Url, other.Url, System.StringComparison.Ordinal)
            && Year == other.Year
            && string.Equals(Authors, other.Authors, System.StringComparison.Ordinal);

        public override bool Equals(object? obj) => Equals(obj as PaperInfo);

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
