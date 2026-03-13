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
    private const string ModelPaperAttr = "AiDotNet.Attributes.ModelPaperAttribute";
    private const string ModelInputAttr = "AiDotNet.Attributes.ModelInputAttribute";
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

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> candidates,
        Compilation compilation)
    {
        if (candidates.IsDefaultOrEmpty)
        {
            EmitDocumentationClass(context, new List<DocEntry>());
            return;
        }

        var domainAttrSymbol = compilation.GetTypeByMetadataName(ModelDomainAttr);
        var categoryAttrSymbol = compilation.GetTypeByMetadataName(ModelCategoryAttr);
        var taskAttrSymbol = compilation.GetTypeByMetadataName(ModelTaskAttr);
        var complexityAttrSymbol = compilation.GetTypeByMetadataName(ModelComplexityAttr);
        var paperAttrSymbol = compilation.GetTypeByMetadataName(ModelPaperAttr);
        var inputAttrSymbol = compilation.GetTypeByMetadataName(ModelInputAttr);
        var exemptAttrSymbol = compilation.GetTypeByMetadataName(ModelMetadataExemptAttr);

        if (domainAttrSymbol is null || categoryAttrSymbol is null ||
            taskAttrSymbol is null || complexityAttrSymbol is null)
        {
            EmitDocumentationClass(context, new List<DocEntry>());
            return;
        }

        var entries = new List<DocEntry>();
        var seen = new HashSet<string>();

        foreach (var modelClass in candidates)
        {
            if (modelClass is null) continue;

            var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName)) continue;

            // Skip classes marked with [ModelMetadataExempt]
            if (exemptAttrSymbol is not null && HasAttribute(modelClass.GetAttributes(), exemptAttrSymbol))
                continue;

            var entry = ExtractDocEntry(modelClass, fullName,
                domainAttrSymbol, categoryAttrSymbol, taskAttrSymbol,
                complexityAttrSymbol, paperAttrSymbol, inputAttrSymbol);

            if (entry.Domains.Count > 0 && entry.Tasks.Count > 0 && entry.HasComplexity)
            {
                entries.Add(entry);
            }
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
        var entry = new DocEntry
        {
            ClassName = modelClass.Name,
            FullyQualifiedName = fullyQualifiedName,
            TypeParameterCount = modelClass.TypeParameters.Length
        };

        foreach (var attr in modelClass.GetAttributes())
        {
            if (attr.AttributeClass is null) continue;

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
                    var title = attr.ConstructorArguments[0].Value as string ?? string.Empty;
                    var url = attr.ConstructorArguments[1].Value as string ?? string.Empty;
                    int year = 0;
                    var authors = string.Empty;
                    foreach (var named in attr.NamedArguments)
                    {
                        if (named.Key == "Year" && named.Value.Value is int y) year = y;
                        else if (named.Key == "Authors" && named.Value.Value is string a) authors = a;
                    }
                    entry.Papers.Add(new PaperInfo { Title = title, Url = url, Year = year, Authors = authors });
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
                        entry.InputTypeName = inputType.Name;
                    if (outputType is not null)
                        entry.OutputTypeName = outputType.Name;
                }
            }
        }

        // Extract XML docs
        var xmlDoc = modelClass.GetDocumentationCommentXml();
        if (!string.IsNullOrWhiteSpace(xmlDoc))
        {
            entry.Summary = ExtractXmlElement(xmlDoc, "summary");
            entry.BeginnerGuide = ExtractBeginnerRemarks(xmlDoc);
        }

        return entry;
    }

    private static void EmitDocumentationClass(SourceProductionContext context, List<DocEntry> entries)
    {
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
            if (entry.Domains.Count == 0)
                sb.AppendLine("            Array.Empty<ModelDomain>(),");
            else
            {
                sb.Append("            new ModelDomain[] { ");
                sb.Append(string.Join(", ", entry.Domains.Select(d =>
                    domainNames.TryGetValue(d, out var n) ? $"ModelDomain.{n}" : $"(ModelDomain){d}")));
                sb.AppendLine(" },");
            }

            // Categories
            if (entry.Categories.Count == 0)
                sb.AppendLine("            Array.Empty<ModelCategory>(),");
            else
            {
                sb.Append("            new ModelCategory[] { ");
                sb.Append(string.Join(", ", entry.Categories.Select(c =>
                    categoryNames.TryGetValue(c, out var n) ? $"ModelCategory.{n}" : $"(ModelCategory){c}")));
                sb.AppendLine(" },");
            }

            // Tasks
            if (entry.Tasks.Count == 0)
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

            var firstPaper = entry.Papers.Count > 0 ? entry.Papers[0] : null;
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

    private class DocEntry
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Domains { get; } = new List<int>();
        public List<int> Categories { get; } = new List<int>();
        public List<int> Tasks { get; } = new List<int>();
        public int Complexity { get; set; }
        public bool HasComplexity { get; set; }
        public List<PaperInfo> Papers { get; } = new List<PaperInfo>();
        public string InputTypeName { get; set; } = string.Empty;
        public string OutputTypeName { get; set; } = string.Empty;
        public string Summary { get; set; } = string.Empty;
        public string BeginnerGuide { get; set; } = string.Empty;
    }

    private class PaperInfo
    {
        public string Title { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public int Year { get; set; }
        public string Authors { get; set; } = string.Empty;
    }
}
