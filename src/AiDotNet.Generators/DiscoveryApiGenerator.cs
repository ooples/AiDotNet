using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that produces a strongly-typed, IntelliSense-friendly
/// discovery API from [ModelDomain], [ModelCategory], and [ModelTask] attributes on model classes.
/// </summary>
/// <remarks>
/// <para>
/// Generates a static <c>Models</c> class with nested classes per domain/task,
/// each containing properties that return <c>Type</c> references to model classes.
/// Also generates query methods for looking up models by domain, task, complexity, and category.
/// </para>
/// </remarks>
[Generator]
public class DiscoveryApiGenerator : IIncrementalGenerator
{
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";
    private const string ModelDomainAttr = "AiDotNet.Attributes.ModelDomainAttribute";
    private const string ModelCategoryAttr = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttr = "AiDotNet.Attributes.ModelTaskAttribute";
    private const string ModelComplexityAttr = "AiDotNet.Attributes.ModelComplexityAttribute";
    private const string ModelPaperAttr = "AiDotNet.Attributes.ModelPaperAttribute";
    private const string ModelMetadataExemptAttr = "AiDotNet.Attributes.ModelMetadataExemptAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
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
            EmitEmpty(context);
            return;
        }

        var domainAttrSymbol = compilation.GetTypeByMetadataName(ModelDomainAttr);
        var categoryAttrSymbol = compilation.GetTypeByMetadataName(ModelCategoryAttr);
        var taskAttrSymbol = compilation.GetTypeByMetadataName(ModelTaskAttr);
        var complexityAttrSymbol = compilation.GetTypeByMetadataName(ModelComplexityAttr);
        var paperAttrSymbol = compilation.GetTypeByMetadataName(ModelPaperAttr);
        var exemptAttrSymbol = compilation.GetTypeByMetadataName(ModelMetadataExemptAttr);

        if (domainAttrSymbol is null || categoryAttrSymbol is null ||
            taskAttrSymbol is null || complexityAttrSymbol is null)
        {
            EmitEmpty(context);
            return;
        }

        var entries = new List<DiscoveryEntry>();
        var seen = new HashSet<string>();

        foreach (var modelClass in candidates)
        {
            if (modelClass is null)
                continue;

            var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName))
                continue;

            // Skip classes marked with [ModelMetadataExempt]
            if (exemptAttrSymbol is not null && HasAttribute(modelClass.GetAttributes(), exemptAttrSymbol))
                continue;

            var entry = ExtractEntry(modelClass, fullName, domainAttrSymbol, categoryAttrSymbol,
                taskAttrSymbol, complexityAttrSymbol, paperAttrSymbol);

            if (entry.Domains.Count > 0 && entry.Tasks.Count > 0)
            {
                entries.Add(entry);
            }
        }

        entries.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        EmitModelsClass(context, entries);
    }

    private static DiscoveryEntry ExtractEntry(
        INamedTypeSymbol modelClass,
        string fullyQualifiedName,
        INamedTypeSymbol domainAttrSymbol,
        INamedTypeSymbol categoryAttrSymbol,
        INamedTypeSymbol taskAttrSymbol,
        INamedTypeSymbol complexityAttrSymbol,
        INamedTypeSymbol? paperAttrSymbol)
    {
        var entry = new DiscoveryEntry
        {
            ClassName = modelClass.Name,
            FullyQualifiedName = fullyQualifiedName,
            TypeParameterCount = modelClass.TypeParameters.Length
        };

        foreach (var attr in modelClass.GetAttributes())
        {
            if (attr.AttributeClass is null)
                continue;

            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, domainAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int d)
                    entry.Domains.Add(d);
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, categoryAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int c)
                    entry.Categories.Add(c);
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, taskAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int t)
                    entry.Tasks.Add(t);
            }
            else if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, complexityAttrSymbol))
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int cx)
                {
                    entry.Complexity = cx;
                    entry.HasComplexity = true;
                }
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
        var xmlDoc = modelClass.GetDocumentationCommentXml();
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
        EmitModelsClass(context, new List<DiscoveryEntry>());
    }

    private static void EmitModelsClass(SourceProductionContext context, List<DiscoveryEntry> entries)
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

        // --- Domain/Task enum name maps (needed for valid C# identifiers) ---
        var domainNames = new Dictionary<int, string>
        {
            {0, "General"}, {1, "Vision"}, {2, "Language"}, {3, "Audio"},
            {4, "Video"}, {5, "Multimodal"}, {6, "Healthcare"}, {7, "Finance"},
            {8, "Science"}, {9, "Robotics"}, {10, "GraphAnalysis"}, {11, "ThreeD"},
            {12, "Tabular"}, {13, "TimeSeries"}, {14, "Generative"},
            {15, "ReinforcementLearning"}, {16, "Causal"}, {17, "MachineLearning"}
        };

        var taskNames = new Dictionary<int, string>
        {
            {0, "Classification"}, {1, "Regression"}, {2, "Generation"}, {3, "Segmentation"},
            {4, "Detection"}, {5, "Embedding"}, {6, "Translation"}, {7, "Forecasting"},
            {8, "Clustering"}, {9, "Denoising"}, {10, "SuperResolution"}, {11, "StyleTransfer"},
            {12, "Inpainting"}, {13, "SpeechRecognition"}, {14, "TextToSpeech"},
            {15, "SourceSeparation"}, {16, "AnomalyDetection"}, {17, "Recommendation"},
            {18, "Ranking"}, {19, "DepthEstimation"}, {20, "OpticalFlow"}, {21, "Tracking"},
            {22, "ActionRecognition"}, {23, "ImageEditing"}, {24, "Editing"},
            {25, "TextToImage"}, {26, "TextToVideo"}, {27, "ThreeDGeneration"},
            {28, "MotionGeneration"}, {29, "SurvivalAnalysis"}, {30, "CausalInference"},
            {31, "Synthesis"}, {32, "FeatureExtraction"}, {33, "Restoration"},
            {34, "Compression"}, {35, "FrameInterpolation"}, {36, "Enhancement"},
            {37, "SignalProcessing"}, {38, "DimensionalityReduction"},
            {39, "BinaryClassification"}, {40, "MultiClassClassification"},
            {41, "VideoGeneration"}, {42, "ImageToVideo"}, {43, "VideoToVideo"}
        };

        var complexityNames = new Dictionary<int, string>
        {
            {0, "Low"}, {1, "Medium"}, {2, "High"}, {3, "VeryHigh"}
        };

        var categoryNames = new Dictionary<int, string>
        {
            {0, "NeuralNetwork"}, {1, "Regression"}, {2, "Classifier"}, {3, "Clustering"},
            {4, "GAN"}, {5, "Diffusion"}, {6, "Transformer"}, {7, "ReinforcementLearningAgent"},
            {8, "GaussianProcess"}, {9, "Ensemble"}, {10, "Bayesian"}, {11, "SurvivalModel"},
            {12, "CausalModel"}, {13, "TimeSeriesModel"}, {14, "Autoencoder"},
            {15, "RecurrentNetwork"}, {16, "ConvolutionalNetwork"}, {17, "GraphNetwork"},
            {18, "EmbeddingModel"}, {19, "FoundationModel"}, {20, "MetaLearning"},
            {21, "TabularModel"}, {22, "SyntheticDataGenerator"}, {23, "PhysicsInformed"},
            {24, "NeuralOperator"}, {25, "Agent"}, {26, "SignalProcessing"},
            {27, "SVM"}, {28, "Kernel"}, {29, "InstanceBased"}, {30, "Linear"},
            {31, "DecisionTree"}, {32, "Statistical"}, {33, "Regularization"},
            {34, "Interpretable"}, {35, "Optimization"}, {36, "AnomalyDetection"}
        };

        // Build domain→task→models hierarchy
        var hierarchy = new Dictionary<int, Dictionary<int, List<DiscoveryEntry>>>();
        foreach (var entry in entries)
        {
            foreach (var domain in entry.Domains)
            {
                if (!hierarchy.TryGetValue(domain, out var taskMap))
                {
                    taskMap = new Dictionary<int, List<DiscoveryEntry>>();
                    hierarchy[domain] = taskMap;
                }
                foreach (var task in entry.Tasks)
                {
                    if (!taskMap.TryGetValue(task, out var list))
                    {
                        list = new List<DiscoveryEntry>();
                        taskMap[task] = list;
                    }
                    list.Add(entry);
                }
            }
        }

        // Emit Models class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated strongly-typed model discovery API.");
        sb.AppendLine("/// Provides IntelliSense-friendly access to all annotated models grouped by domain and task.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static partial class Models");
        sb.AppendLine("{");

        // Emit nested domain classes
        var sortedDomains = hierarchy.Keys.OrderBy(k => k).ToList();
        foreach (var domainInt in sortedDomains)
        {
            if (!domainNames.TryGetValue(domainInt, out var domainName))
                continue;

            var taskMap = hierarchy[domainInt];
            sb.AppendLine($"    /// <summary>Models for the {domainName} domain.</summary>");
            sb.AppendLine($"    public static class {domainName}");
            sb.AppendLine("    {");

            var sortedTasks = taskMap.Keys.OrderBy(k => k).ToList();
            foreach (var taskInt in sortedTasks)
            {
                if (!taskNames.TryGetValue(taskInt, out var taskName))
                    continue;

                var models = taskMap[taskInt];
                models.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

                sb.AppendLine($"        /// <summary>Models for {taskName} tasks in {domainName}.</summary>");
                sb.AppendLine($"        public static class {taskName}");
                sb.AppendLine("        {");

                var emittedInTask = new HashSet<string>();
                foreach (var model in models)
                {
                    var propName = SanitizePropertyName(model.ClassName, emittedInTask);
                    if (propName is null) continue;

                    var typeOfExpr = BuildTypeOfExpression(model);
                    var summary = !string.IsNullOrEmpty(model.Summary)
                        ? EscapeXmlComment(model.Summary)
                        : model.ClassName;

                    sb.AppendLine($"            /// <summary>{summary}</summary>");
                    if (!string.IsNullOrEmpty(model.PaperTitle))
                    {
                        sb.AppendLine($"            /// <remarks>Paper: {EscapeXmlComment(model.PaperTitle)}</remarks>");
                    }
                    sb.AppendLine($"            public static Type {propName} => {typeOfExpr};");
                }

                sb.AppendLine("        }");
                sb.AppendLine();
            }

            sb.AppendLine("    }");
            sb.AppendLine();
        }

        // Emit query methods using pre-built dictionaries
        EmitQueryMethods(sb, entries, domainNames, taskNames, complexityNames, categoryNames);

        sb.AppendLine("}");

        context.AddSource("Models.g.cs", sb.ToString());
    }

    private static void EmitQueryMethods(
        StringBuilder sb,
        List<DiscoveryEntry> entries,
        Dictionary<int, string> domainNames,
        Dictionary<int, string> taskNames,
        Dictionary<int, string> complexityNames,
        Dictionary<int, string> categoryNames)
    {
        // GetByDomain
        sb.AppendLine("    private static readonly Dictionary<ModelDomain, Type[]> _domainLookup = BuildDomainLookup();");
        sb.AppendLine("    private static readonly Dictionary<ModelTask, Type[]> _taskLookup = BuildTaskLookup();");
        sb.AppendLine("    private static readonly Dictionary<ModelComplexity, Type[]> _complexityLookup = BuildComplexityLookup();");
        sb.AppendLine("    private static readonly Dictionary<ModelCategory, Type[]> _categoryLookup = BuildCategoryLookup();");
        sb.AppendLine();

        // GetByDomain method
        sb.AppendLine("    /// <summary>Gets all model types for the specified domain.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> GetByDomain(ModelDomain domain)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_domainLookup.TryGetValue(domain, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetByTask method
        sb.AppendLine("    /// <summary>Gets all model types for the specified task.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> GetByTask(ModelTask task)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_taskLookup.TryGetValue(task, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetByComplexity method
        sb.AppendLine("    /// <summary>Gets all model types with the specified complexity.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> GetByComplexity(ModelComplexity complexity)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_complexityLookup.TryGetValue(complexity, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetByCategory method
        sb.AppendLine("    /// <summary>Gets all model types for the specified category.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> GetByCategory(ModelCategory category)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_categoryLookup.TryGetValue(category, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetByDomainAndTask method
        sb.AppendLine("    /// <summary>Gets all model types matching both domain and task.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> GetByDomainAndTask(ModelDomain domain, ModelTask task)");
        sb.AppendLine("    {");
        sb.AppendLine("        var domainModels = GetByDomain(domain);");
        sb.AppendLine("        var taskModels = GetByTask(task);");
        sb.AppendLine("        if (domainModels.Count == 0 || taskModels.Count == 0)");
        sb.AppendLine("            return Array.Empty<Type>();");
        sb.AppendLine("        var taskSet = new HashSet<Type>(taskModels);");
        sb.AppendLine("        var result = new List<Type>();");
        sb.AppendLine("        foreach (var t in domainModels)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (taskSet.Contains(t))");
        sb.AppendLine("                result.Add(t);");
        sb.AppendLine("        }");
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Build lookup methods
        EmitBuildLookup(sb, "BuildDomainLookup", "ModelDomain", entries, e => e.Domains, domainNames);
        EmitBuildLookup(sb, "BuildTaskLookup", "ModelTask", entries, e => e.Tasks, taskNames);
        EmitBuildComplexityLookup(sb, entries, complexityNames);
        EmitBuildLookup(sb, "BuildCategoryLookup", "ModelCategory", entries, e => e.Categories, categoryNames);
    }

    private static void EmitBuildLookup(
        StringBuilder sb,
        string methodName,
        string enumType,
        List<DiscoveryEntry> entries,
        System.Func<DiscoveryEntry, List<int>> selector,
        Dictionary<int, string> nameMap)
    {
        // Group entries by their enum values
        var grouped = new Dictionary<int, List<DiscoveryEntry>>();
        foreach (var entry in entries)
        {
            foreach (var val in selector(entry))
            {
                if (!grouped.TryGetValue(val, out var list))
                {
                    list = new List<DiscoveryEntry>();
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

    private static void EmitBuildComplexityLookup(
        StringBuilder sb,
        List<DiscoveryEntry> entries,
        Dictionary<int, string> complexityNames)
    {
        var grouped = new Dictionary<int, List<DiscoveryEntry>>();
        foreach (var entry in entries)
        {
            if (!entry.HasComplexity) continue;
            if (!grouped.TryGetValue(entry.Complexity, out var list))
            {
                list = new List<DiscoveryEntry>();
                grouped[entry.Complexity] = list;
            }
            list.Add(entry);
        }

        sb.AppendLine("    private static Dictionary<ModelComplexity, Type[]> BuildComplexityLookup()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<ModelComplexity, Type[]>();");

        foreach (var kvp in grouped.OrderBy(k => k.Key))
        {
            if (!complexityNames.TryGetValue(kvp.Key, out var enumName))
                continue;

            sb.Append($"        dict[ModelComplexity.{enumName}] = new Type[] {{ ");
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

    private static string BuildTypeOfExpression(DiscoveryEntry entry)
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

    private static bool HasAttribute(ImmutableArray<AttributeData> attributes, INamedTypeSymbol attributeType)
    {
        foreach (var attr in attributes)
        {
            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, attributeType))
                return true;
        }
        return false;
    }

    private class DiscoveryEntry
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Domains { get; } = new List<int>();
        public List<int> Categories { get; } = new List<int>();
        public List<int> Tasks { get; } = new List<int>();
        public int Complexity { get; set; }
        public bool HasComplexity { get; set; }
        public string PaperTitle { get; set; } = string.Empty;
        public string Summary { get; set; } = string.Empty;
    }
}
