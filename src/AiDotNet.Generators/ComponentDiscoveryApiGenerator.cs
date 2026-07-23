using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that produces a strongly-typed, IntelliSense-friendly
/// discovery API from [ComponentType] and [PipelineStage] attributes on component classes.
/// </summary>
/// <remarks>
/// <para>
/// Generates a static <c>Components</c> class with nested classes per <c>ComponentType</c>,
/// each containing properties that return <c>Type</c> references to component classes.
/// Also generates query methods for looking up components by <c>ComponentType</c> and <c>PipelineStage</c>.
/// </para>
/// <para>
/// Usage examples:
/// <code>
/// // Browse by component type via IntelliSense
/// Type retriever = Components.Retrievers.HybridRetriever;
///
/// // Query all retrievers
/// IReadOnlyList&lt;Type&gt; retrievers = Components.ByComponentType(ComponentType.Retriever);
///
/// // Query by pipeline stage
/// IReadOnlyList&lt;Type&gt; trainingComponents = Components.ByPipelineStage(PipelineStage.Training);
/// </code>
/// </para>
/// </remarks>
[Generator]
public class ComponentDiscoveryApiGenerator : IIncrementalGenerator
{
    private const string ComponentTypeAttr = "AiDotNet.Attributes.ComponentTypeAttribute";
    private const string PipelineStageAttr = "AiDotNet.Attributes.PipelineStageAttribute";
    private const string ResearchPaperAttr = "AiDotNet.Attributes.ResearchPaperAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var classDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => GetComponentClassOrNull(ctx))
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

        // Skip abstract classes
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

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> candidates,
        Compilation compilation)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitEmpty(context);
            return;
        }

        var componentTypeAttrSymbol = compilation.GetTypeByMetadataName(ComponentTypeAttr);
        var pipelineStageAttrSymbol = compilation.GetTypeByMetadataName(PipelineStageAttr);
        var paperAttrSymbol = compilation.GetTypeByMetadataName(ResearchPaperAttr);

        if (componentTypeAttrSymbol is null)
        {
            EmitEmpty(context);
            return;
        }

        var entries = new List<ComponentDiscoveryEntry>();
        var seen = new HashSet<string>();

        foreach (var componentClass in candidates)
        {
            if (componentClass is null)
                continue;

            var fullName = componentClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName))
                continue;

            var entry = ExtractEntry(componentClass, fullName,
                componentTypeAttrSymbol, pipelineStageAttrSymbol, paperAttrSymbol);

            if (entry.ComponentTypes.Count > 0)
            {
                entries.Add(entry);
            }
        }

        entries.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        EmitComponentsClass(context, entries);
    }

    private static ComponentDiscoveryEntry ExtractEntry(
        INamedTypeSymbol componentClass,
        string fullyQualifiedName,
        INamedTypeSymbol componentTypeAttrSymbol,
        INamedTypeSymbol? pipelineStageAttrSymbol,
        INamedTypeSymbol? paperAttrSymbol)
    {
        var entry = new ComponentDiscoveryEntry
        {
            ClassName = componentClass.Name,
            FullyQualifiedName = fullyQualifiedName,
            TypeParameterCount = componentClass.TypeParameters.Length
        };

        foreach (var attr in componentClass.GetAttributes())
        {
            if (attr.AttributeClass is null)
                continue;

            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, componentTypeAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int ct)
                    entry.ComponentTypes.Add(ct);
            }
            else if (pipelineStageAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, pipelineStageAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int ps)
                    entry.PipelineStages.Add(ps);
            }
            else if (paperAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, paperAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 2)
                {
                    entry.PaperTitle = attr.ConstructorArguments[0].Value as string ?? string.Empty;
                }
            }
        }

        // Extract XML doc summary
        var xmlDoc = componentClass.GetDocumentationCommentXml();
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            entry.Summary = ExtractSummary(xmlDoc);
        }

        return entry;
    }

    private static string ExtractSummary(string xml)
    {
        var startTag = "<summary>";
        var endTag = "</summary>";
        var startIdx = xml.IndexOf(startTag, System.StringComparison.Ordinal);
        if (startIdx < 0) return string.Empty;
        startIdx += startTag.Length;
        var endIdx = xml.IndexOf(endTag, startIdx, System.StringComparison.Ordinal);
        if (endIdx < 0) return string.Empty;
        var raw = xml.Substring(startIdx, endIdx - startIdx);
        // Strip XML tags and normalize whitespace
        var sb = new StringBuilder(raw.Length);
        var inTag = false;
        foreach (var c in raw)
        {
            if (c == '<') { inTag = true; continue; }
            if (c == '>') { inTag = false; continue; }
            if (!inTag) sb.Append(c);
        }
        var text = sb.ToString();
        var normalized = new StringBuilder(text.Length);
        bool prevSpace = false;
        foreach (char c in text)
        {
            if (c == ' ' || c == '\r' || c == '\n' || c == '\t')
            {
                if (!prevSpace) { normalized.Append(' '); prevSpace = true; }
            }
            else { normalized.Append(c); prevSpace = false; }
        }
        return normalized.ToString().Trim();
    }

    private static void EmitEmpty(SourceProductionContext context)
    {
        EmitComponentsClass(context, new List<ComponentDiscoveryEntry>());
    }

    private static void EmitComponentsClass(SourceProductionContext context, List<ComponentDiscoveryEntry> entries)
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

        // ComponentType enum name map (must match AiDotNet.Enums.ComponentType)
        var componentTypeNames = new Dictionary<int, string>
        {
            {0, "Retriever"}, {1, "Reranker"}, {2, "Chunker"}, {3, "QueryProcessor"},
            {4, "Generator"}, {5, "ContextCompressor"}, {6, "QueryExpander"},
            {7, "DocumentStore"}, {8, "VectorIndex"}, {9, "EntityRecognizer"},
            {10, "MetaLearner"}, {11, "ActiveLearner"}, {12, "ContinualLearner"},
            {13, "DistillationStrategy"}, {14, "FederatedAggregator"}, {15, "FederatedTrainer"},
            {16, "PrivacyMechanism"}, {17, "PSIProtocol"}, {18, "PersonalizationStrategy"},
            {19, "FederatedUnlearner"}, {20, "DataLoader"}, {21, "VerificationScheme"},
            {22, "CryptoPrimitive"}, {23, "GraphPartitioner"}, {24, "BenchmarkUtility"},
            {25, "TransferAlgorithm"}, {26, "DomainAdapter"}, {27, "Scaler"},
            {28, "Encoder"}, {29, "DimensionReducer"}, {30, "FeatureSelector"},
            {31, "FeatureGenerator"}, {32, "Optimizer"}, {33, "Scheduler"},
            {34, "Regularizer"}, {35, "Evaluator"}
        };

        // Plural forms for nested class names
        var componentTypePluralNames = new Dictionary<int, string>
        {
            {0, "Retrievers"}, {1, "Rerankers"}, {2, "Chunkers"}, {3, "QueryProcessors"},
            {4, "Generators"}, {5, "ContextCompressors"}, {6, "QueryExpanders"},
            {7, "DocumentStores"}, {8, "VectorIndexes"}, {9, "EntityRecognizers"},
            {10, "MetaLearners"}, {11, "ActiveLearners"}, {12, "ContinualLearners"},
            {13, "DistillationStrategies"}, {14, "FederatedAggregators"}, {15, "FederatedTrainers"},
            {16, "PrivacyMechanisms"}, {17, "PSIProtocols"}, {18, "PersonalizationStrategies"},
            {19, "FederatedUnlearners"}, {20, "DataLoaders"}, {21, "VerificationSchemes"},
            {22, "CryptoPrimitives"}, {23, "GraphPartitioners"}, {24, "BenchmarkUtilities"},
            {25, "TransferAlgorithms"}, {26, "DomainAdapters"}, {27, "Scalers"},
            {28, "Encoders"}, {29, "DimensionReducers"}, {30, "FeatureSelectors"},
            {31, "FeatureGenerators"}, {32, "Optimizers"}, {33, "Schedulers"},
            {34, "Regularizers"}, {35, "Evaluators"}
        };

        // PipelineStage enum name map (must match AiDotNet.Enums.PipelineStage)
        var pipelineStageNames = new Dictionary<int, string>
        {
            {0, "DataIngestion"}, {1, "Indexing"}, {2, "Retrieval"},
            {3, "PostRetrieval"}, {4, "Generation"}, {5, "Preprocessing"},
            {6, "Training"}, {7, "Evaluation"}, {8, "QueryProcessing"}
        };

        // Build componentType→entries hierarchy
        var byComponentType = new Dictionary<int, List<ComponentDiscoveryEntry>>();
        foreach (var entry in entries)
        {
            foreach (var ct in entry.ComponentTypes)
            {
                if (!byComponentType.TryGetValue(ct, out var list))
                {
                    list = new List<ComponentDiscoveryEntry>();
                    byComponentType[ct] = list;
                }
                // Avoid duplicates
                bool exists = false;
                foreach (var e in list)
                {
                    if (e.FullyQualifiedName == entry.FullyQualifiedName)
                    {
                        exists = true;
                        break;
                    }
                }
                if (!exists)
                    list.Add(entry);
            }
        }

        // Emit Components class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated strongly-typed component discovery API.");
        sb.AppendLine("/// Provides IntelliSense-friendly access to all annotated components grouped by component type and pipeline stage.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static partial class Components");
        sb.AppendLine("{");

        // Emit nested classes per ComponentType
        var sortedComponentTypes = byComponentType.Keys.OrderBy(k => k).ToList();
        foreach (var ctInt in sortedComponentTypes)
        {
            if (!componentTypePluralNames.TryGetValue(ctInt, out var pluralName))
                continue;
            if (!componentTypeNames.TryGetValue(ctInt, out var singularName))
                continue;

            var componentList = byComponentType[ctInt];
            componentList.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

            sb.AppendLine($"    /// <summary>{singularName} components.</summary>");
            sb.AppendLine($"    public static class {pluralName}");
            sb.AppendLine("    {");

            var emittedInGroup = new HashSet<string>();
            foreach (var component in componentList)
            {
                var propName = SanitizePropertyName(component.ClassName, emittedInGroup);
                if (propName is null) continue;

                var typeOfExpr = BuildTypeOfExpression(component);
                var summary = !string.IsNullOrEmpty(component.Summary)
                    ? EscapeXmlComment(component.Summary)
                    : component.ClassName;

                sb.AppendLine($"        /// <summary>{summary}</summary>");
                if (!string.IsNullOrEmpty(component.PaperTitle))
                {
                    sb.AppendLine($"        /// <remarks>Paper: {EscapeXmlComment(component.PaperTitle)}</remarks>");
                }
                sb.AppendLine($"        public static Type {propName} => {typeOfExpr};");
            }

            sb.AppendLine("    }");
            sb.AppendLine();
        }

        // Emit query methods
        EmitQueryMethods(sb, entries, componentTypeNames, pipelineStageNames);

        sb.AppendLine("}");

        context.AddSource("Components.g.cs", sb.ToString());
    }

    private static void EmitQueryMethods(
        StringBuilder sb,
        List<ComponentDiscoveryEntry> entries,
        Dictionary<int, string> componentTypeNames,
        Dictionary<int, string> pipelineStageNames)
    {
        sb.AppendLine("    private static readonly Dictionary<ComponentType, Type[]> _componentTypeLookup = BuildComponentTypeLookup();");
        sb.AppendLine("    private static readonly Dictionary<PipelineStage, Type[]> _pipelineStageLookup = BuildPipelineStageLookup();");
        sb.AppendLine();

        // ByComponentType method
        sb.AppendLine("    /// <summary>Gets all component types for the specified component type.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> ByComponentType(ComponentType type)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_componentTypeLookup.TryGetValue(type, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // ByPipelineStage method
        sb.AppendLine("    /// <summary>Gets all component types for the specified pipeline stage.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> ByPipelineStage(PipelineStage stage)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_pipelineStageLookup.TryGetValue(stage, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // ByComponentTypeAndPipelineStage method
        sb.AppendLine("    /// <summary>Gets all component types matching both component type and pipeline stage.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> ByComponentTypeAndPipelineStage(ComponentType type, PipelineStage stage)");
        sb.AppendLine("    {");
        sb.AppendLine("        var byType = ByComponentType(type);");
        sb.AppendLine("        var byStage = ByPipelineStage(stage);");
        sb.AppendLine("        if (byType.Count == 0 || byStage.Count == 0)");
        sb.AppendLine("            return Array.Empty<Type>();");
        sb.AppendLine("        var stageSet = new HashSet<Type>(byStage);");
        sb.AppendLine("        var result = new List<Type>();");
        sb.AppendLine("        foreach (var t in byType)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (stageSet.Contains(t))");
        sb.AppendLine("                result.Add(t);");
        sb.AppendLine("        }");
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Build lookup methods
        EmitBuildLookup(sb, "BuildComponentTypeLookup", "ComponentType", entries,
            e => e.ComponentTypes, componentTypeNames);
        EmitBuildLookup(sb, "BuildPipelineStageLookup", "PipelineStage", entries,
            e => e.PipelineStages, pipelineStageNames);
    }

    private static void EmitBuildLookup(
        StringBuilder sb,
        string methodName,
        string enumType,
        List<ComponentDiscoveryEntry> entries,
        System.Func<ComponentDiscoveryEntry, List<int>> selector,
        Dictionary<int, string> nameMap)
    {
        // Group entries by their enum values
        var grouped = new Dictionary<int, List<ComponentDiscoveryEntry>>();
        foreach (var entry in entries)
        {
            foreach (var val in selector(entry))
            {
                if (!grouped.TryGetValue(val, out var list))
                {
                    list = new List<ComponentDiscoveryEntry>();
                    grouped[val] = list;
                }
                // Avoid duplicates
                bool exists = false;
                foreach (var e in list)
                {
                    if (e.FullyQualifiedName == entry.FullyQualifiedName)
                    {
                        exists = true;
                        break;
                    }
                }
                if (!exists)
                    list.Add(entry);
            }
        }

        sb.AppendLine($"    private static Dictionary<{enumType}, Type[]> {methodName}()");
        sb.AppendLine("    {");
        sb.AppendLine($"        var dict = new Dictionary<{enumType}, Type[]>();");

        foreach (var kvp in grouped.OrderBy(k => k.Key))
        {
            if (!nameMap.TryGetValue(kvp.Key, out var enumName))
                continue;

            sb.Append($"        dict[{enumType}.{enumName}] = new Type[] {{ ");
            var typesList = kvp.Value.OrderBy(e => e.ClassName).ToList();
            for (int i = 0; i < typesList.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(BuildTypeOfExpression(typesList[i]));
            }
            sb.AppendLine(" };");
        }

        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");
        sb.AppendLine();
    }

    private static string BuildTypeOfExpression(ComponentDiscoveryEntry entry)
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

    private static string? SanitizePropertyName(string className, HashSet<string> emittedNames)
    {
        // Remove generic arity suffix like `1
        var idx = className.IndexOf('`');
        var name = idx >= 0 ? className.Substring(0, idx) : className;

        // Ensure starts with letter
        if (name.Length == 0 || !char.IsLetter(name[0]))
            name = "_" + name;

        // Replace invalid chars
        var sb = new StringBuilder(name.Length);
        foreach (var c in name)
        {
            if (char.IsLetterOrDigit(c) || c == '_')
                sb.Append(c);
        }
        name = sb.ToString();

        // Handle duplicates by appending a number
        var baseName = name;
        int counter = 2;
        while (!emittedNames.Add(name))
        {
            name = baseName + counter;
            counter++;
        }

        return name;
    }

    private static string EscapeXmlComment(string text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        return text
            .Replace("&", "&amp;")
            .Replace("<", "&lt;")
            .Replace(">", "&gt;")
            .Replace("\"", "&quot;");
    }

    private class ComponentDiscoveryEntry
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> ComponentTypes { get; } = new List<int>();
        public List<int> PipelineStages { get; } = new List<int>();
        public string PaperTitle { get; set; } = string.Empty;
        public string Summary { get; set; } = string.Empty;
    }
}
