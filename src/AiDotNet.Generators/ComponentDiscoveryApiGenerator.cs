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
        // Values, not symbols. See DiscoveryApiGenerator for the full rationale: an ISymbol in the
        // pipeline is not value-equatable (so nothing ever caches) and roots the whole Compilation
        // (so the cache pins compilations in memory). Analyze does the semantic work and returns a
        // value-equatable entry; the Compilation is read transiently and never escapes.
        var entries = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => Analyze(ctx))
            .Where(static e => e is not null)
            .Select(static (e, _) => e ?? ComponentDiscoveryEntry.Empty);

        context.RegisterSourceOutput(entries.Collect(), static (spc, collected) => Emit(spc, collected));
    }

    /// <summary>
    /// Resolves one candidate class into a value-equatable entry, or null when it is not a
    /// discoverable component. All symbol access is confined to this method.
    /// </summary>
    private static ComponentDiscoveryEntry? Analyze(GeneratorSyntaxContext ctx)
    {
        if (GetComponentClassOrNull(ctx) is not INamedTypeSymbol componentClass)
            return null;

        var compilation = ctx.SemanticModel.Compilation;
        var componentTypeAttrSymbol = compilation.GetTypeByMetadataName(ComponentTypeAttr);
        var pipelineStageAttrSymbol = compilation.GetTypeByMetadataName(PipelineStageAttr);
        var paperAttrSymbol = compilation.GetTypeByMetadataName(ResearchPaperAttr);

        if (componentTypeAttrSymbol is null)
            return null;

        var fullName = componentClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        var entry = ExtractEntry(componentClass, fullName,
            componentTypeAttrSymbol, pipelineStageAttrSymbol, paperAttrSymbol);

        // Same admission rule as before: no component type means not discoverable.
        return entry.ComponentTypes.Length > 0 ? entry : null;
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

    /// <summary>
    /// Emits from already-resolved entries; dedupe and ordering stay here where the whole set is
    /// visible, while symbol work lives in <see cref="Analyze"/>.
    /// </summary>
    private static void Emit(SourceProductionContext context, ImmutableArray<ComponentDiscoveryEntry> candidates)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitEmpty(context);
            return;
        }

        var entries = new List<ComponentDiscoveryEntry>();
        var seen = new HashSet<string>();

        foreach (var entry in candidates)
        {
            if (entry.FullyQualifiedName.Length == 0)
                continue;
            if (!seen.Add(entry.FullyQualifiedName))
                continue;

            entries.Add(entry);
        }

        if (entries.Count == 0)
        {
            EmitEmpty(context);
            return;
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
        var componentTypes = new List<int>();
        var pipelineStages = new List<int>();
        string paperTitle = string.Empty;
        string summary = string.Empty;

        foreach (var attr in componentClass.GetAttributes())
        {
            if (attr.AttributeClass is null)
                continue;

            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, componentTypeAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int ct)
                    componentTypes.Add(ct);
            }
            else if (pipelineStageAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, pipelineStageAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int ps)
                    pipelineStages.Add(ps);
            }
            else if (paperAttrSymbol is not null &&
                     SymbolEqualityComparer.Default.Equals(attr.AttributeClass, paperAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 2)
                {
                    paperTitle = attr.ConstructorArguments[0].Value as string ?? string.Empty;
                }
            }
        }

        // Extract XML doc summary
        var xmlDoc = componentClass.GetDocumentationCommentXml();
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            summary = ExtractSummary(xmlDoc);
        }

        return new ComponentDiscoveryEntry(
            componentClass.Name,
            fullyQualifiedName,
            componentClass.TypeParameters.Length,
            componentTypes.ToImmutableArray(),
            pipelineStages.ToImmutableArray(),
            paperTitle,
            summary);
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
        System.Func<ComponentDiscoveryEntry, ImmutableArray<int>> selector,
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

    /// <summary>
    /// One discoverable component, as plain values.
    /// </summary>
    /// <remarks>
    /// Immutable and structurally equal because this type travels through the incremental pipeline;
    /// the previous mutable List&lt;int&gt; shape compared by reference, so identical data never
    /// matched and nothing downstream could be skipped.
    /// </remarks>
    private sealed class ComponentDiscoveryEntry : System.IEquatable<ComponentDiscoveryEntry>
    {
        public static readonly ComponentDiscoveryEntry Empty = new(
            string.Empty, string.Empty, 0,
            ImmutableArray<int>.Empty, ImmutableArray<int>.Empty, string.Empty, string.Empty);

        public ComponentDiscoveryEntry(
            string className,
            string fullyQualifiedName,
            int typeParameterCount,
            ImmutableArray<int> componentTypes,
            ImmutableArray<int> pipelineStages,
            string paperTitle,
            string summary)
        {
            ClassName = className;
            FullyQualifiedName = fullyQualifiedName;
            TypeParameterCount = typeParameterCount;
            ComponentTypes = componentTypes.IsDefault ? ImmutableArray<int>.Empty : componentTypes;
            PipelineStages = pipelineStages.IsDefault ? ImmutableArray<int>.Empty : pipelineStages;
            PaperTitle = paperTitle;
            Summary = summary;
        }

        public string ClassName { get; }
        public string FullyQualifiedName { get; }
        public int TypeParameterCount { get; }
        public ImmutableArray<int> ComponentTypes { get; }
        public ImmutableArray<int> PipelineStages { get; }
        public string PaperTitle { get; }
        public string Summary { get; }

        public bool Equals(ComponentDiscoveryEntry? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            return string.Equals(ClassName, other.ClassName, System.StringComparison.Ordinal)
                && string.Equals(FullyQualifiedName, other.FullyQualifiedName, System.StringComparison.Ordinal)
                && TypeParameterCount == other.TypeParameterCount
                && string.Equals(PaperTitle, other.PaperTitle, System.StringComparison.Ordinal)
                && string.Equals(Summary, other.Summary, System.StringComparison.Ordinal)
                && SequenceEqual(ComponentTypes, other.ComponentTypes)
                && SequenceEqual(PipelineStages, other.PipelineStages);
        }

        public override bool Equals(object? obj) => Equals(obj as ComponentDiscoveryEntry);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 31) + ClassName.GetHashCode();
                hash = (hash * 31) + FullyQualifiedName.GetHashCode();
                hash = (hash * 31) + TypeParameterCount;
                hash = (hash * 31) + PaperTitle.GetHashCode();
                hash = (hash * 31) + Summary.GetHashCode();
                hash = (hash * 31) + ComponentTypes.Length;
                hash = (hash * 31) + PipelineStages.Length;
                return hash;
            }
        }

        private static bool SequenceEqual(ImmutableArray<int> left, ImmutableArray<int> right)
        {
            if (left.Length != right.Length) return false;
            for (int i = 0; i < left.Length; i++)
            {
                if (left[i] != right[i]) return false;
            }
            return true;
        }
    }
}
