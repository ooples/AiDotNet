using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that produces a compile-time compatibility matrix
/// showing which models work with which optimizers, loss functions, and preprocessors.
/// </summary>
/// <remarks>
/// <para>
/// Analyzes [ModelCategory] attributes to determine compatible loss functions and optimizers
/// based on predefined compatibility rules. Emits a static <c>ModelCompatibility</c> class
/// with query methods and diagnostic warnings for suspicious combinations.
/// </para>
/// </remarks>
[Generator]
public class CompatibilityMatrixGenerator : IIncrementalGenerator
{
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";
    private const string ModelCategoryAttr = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttr = "AiDotNet.Attributes.ModelTaskAttribute";

    // Diagnostic for suspicious model/optimizer combinations
    private static readonly DiagnosticDescriptor SuspiciousOptimizer = new(
        id: "AIDN030",
        title: "Suspicious model-optimizer combination",
        messageFormat: "Model '{0}' uses category '{1}' which may not work well with SGD optimizer. Consider Adam or RMSProp.",
        category: "AiDotNet.Compatibility",
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true);

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
        var categoryAttrSymbol = compilation.GetTypeByMetadataName(ModelCategoryAttr);

        if (categoryAttrSymbol is null || candidates.IsDefaultOrEmpty)
        {
            EmitCompatibilityClass(context, new List<CompatEntry>());
            return;
        }

        var entries = new List<CompatEntry>();
        var seen = new HashSet<string>();

        foreach (var modelClass in candidates)
        {
            if (modelClass is null)
                continue;

            var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName))
                continue;

            var categories = new List<int>();
            foreach (var attr in modelClass.GetAttributes())
            {
                if (attr.AttributeClass is not null &&
                    SymbolEqualityComparer.Default.Equals(attr.AttributeClass, categoryAttrSymbol))
                {
                    if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int c)
                        categories.Add(c);
                }
            }

            if (categories.Count > 0)
            {
                entries.Add(new CompatEntry
                {
                    ClassName = modelClass.Name,
                    FullyQualifiedName = fullName,
                    TypeParameterCount = modelClass.TypeParameters.Length,
                    Categories = categories
                });
            }
        }

        entries.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        EmitCompatibilityClass(context, entries);
    }

    private static void EmitCompatibilityClass(SourceProductionContext context, List<CompatEntry> entries)
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

        // ModelCompatibilityInfo class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Contains compatibility information for a model type.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal sealed class ModelCompatibilityInfo");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Gets the model type.</summary>");
        sb.AppendLine("    public Type ModelType { get; }");
        sb.AppendLine("    /// <summary>Gets compatible optimizer categories.</summary>");
        sb.AppendLine("    public IReadOnlyList<string> CompatibleOptimizers { get; }");
        sb.AppendLine("    /// <summary>Gets compatible loss function categories.</summary>");
        sb.AppendLine("    public IReadOnlyList<string> CompatibleLossFunctions { get; }");
        sb.AppendLine("    /// <summary>Gets recommended preprocessors.</summary>");
        sb.AppendLine("    public IReadOnlyList<string> RecommendedPreprocessors { get; }");
        sb.AppendLine("    /// <summary>Gets any compatibility warnings.</summary>");
        sb.AppendLine("    public IReadOnlyList<string> Warnings { get; }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Initializes a new compatibility info entry.</summary>");
        sb.AppendLine("    public ModelCompatibilityInfo(");
        sb.AppendLine("        Type modelType,");
        sb.AppendLine("        string[] compatibleOptimizers,");
        sb.AppendLine("        string[] compatibleLossFunctions,");
        sb.AppendLine("        string[] recommendedPreprocessors,");
        sb.AppendLine("        string[] warnings)");
        sb.AppendLine("    {");
        sb.AppendLine("        ModelType = modelType;");
        sb.AppendLine("        CompatibleOptimizers = compatibleOptimizers;");
        sb.AppendLine("        CompatibleLossFunctions = compatibleLossFunctions;");
        sb.AppendLine("        RecommendedPreprocessors = recommendedPreprocessors;");
        sb.AppendLine("        Warnings = warnings;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // ModelCompatibility static class
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated compile-time compatibility matrix for models, optimizers, and loss functions.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class ModelCompatibility");
        sb.AppendLine("{");

        // Build the lookup dictionary
        sb.AppendLine("    private static readonly Dictionary<Type, ModelCompatibilityInfo> _lookup = BuildLookup();");
        sb.AppendLine();

        // GetInfo method
        sb.AppendLine("    /// <summary>Gets compatibility information for a model type.</summary>");
        sb.AppendLine("    public static ModelCompatibilityInfo? GetInfo(Type modelType)");
        sb.AppendLine("    {");
        sb.AppendLine("        var key = modelType.IsGenericType ? modelType.GetGenericTypeDefinition() : modelType;");
        sb.AppendLine("        if (_lookup.TryGetValue(key, out var info))");
        sb.AppendLine("            return info;");
        sb.AppendLine("        return null;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetCompatibleOptimizers
        sb.AppendLine("    /// <summary>Gets optimizer categories compatible with the given model type.</summary>");
        sb.AppendLine("    public static IReadOnlyList<string> GetCompatibleOptimizers(Type modelType)");
        sb.AppendLine("    {");
        sb.AppendLine("        var info = GetInfo(modelType);");
        sb.AppendLine("        return info is not null ? info.CompatibleOptimizers : Array.Empty<string>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetCompatibleLossFunctions
        sb.AppendLine("    /// <summary>Gets loss function categories compatible with the given model type.</summary>");
        sb.AppendLine("    public static IReadOnlyList<string> GetCompatibleLossFunctions(Type modelType)");
        sb.AppendLine("    {");
        sb.AppendLine("        var info = GetInfo(modelType);");
        sb.AppendLine("        return info is not null ? info.CompatibleLossFunctions : Array.Empty<string>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // GetRecommendedPreprocessors
        sb.AppendLine("    /// <summary>Gets recommended preprocessors for the given model type.</summary>");
        sb.AppendLine("    public static IReadOnlyList<string> GetRecommendedPreprocessors(Type modelType)");
        sb.AppendLine("    {");
        sb.AppendLine("        var info = GetInfo(modelType);");
        sb.AppendLine("        return info is not null ? info.RecommendedPreprocessors : Array.Empty<string>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // BuildLookup
        sb.AppendLine("    private static Dictionary<Type, ModelCompatibilityInfo> BuildLookup()");
        sb.AppendLine("    {");
        sb.AppendLine("        var dict = new Dictionary<Type, ModelCompatibilityInfo>();");

        foreach (var entry in entries)
        {
            var typeExpr = BuildTypeOfExpression(entry);
            var (optimizers, lossFunctions, preprocessors, warnings) = GetCompatibilityRules(entry.Categories);

            sb.AppendLine($"        dict[{typeExpr}] = new ModelCompatibilityInfo(");
            sb.AppendLine($"            {typeExpr},");
            sb.AppendLine($"            new string[] {{ {FormatStringArray(optimizers)} }},");
            sb.AppendLine($"            new string[] {{ {FormatStringArray(lossFunctions)} }},");
            sb.AppendLine($"            new string[] {{ {FormatStringArray(preprocessors)} }},");
            sb.AppendLine($"            new string[] {{ {FormatStringArray(warnings)} }});");
        }

        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");

        sb.AppendLine("}");

        context.AddSource("ModelCompatibility.g.cs", sb.ToString());
    }

    private static (List<string> optimizers, List<string> lossFunctions, List<string> preprocessors, List<string> warnings) GetCompatibilityRules(List<int> categories)
    {
        var optimizers = new HashSet<string>();
        var lossFunctions = new HashSet<string>();
        var preprocessors = new HashSet<string>();
        var warnings = new List<string>();

        foreach (var cat in categories)
        {
            switch (cat)
            {
                case 2: // Classifier
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("BinaryCrossEntropy");
                    lossFunctions.Add("Hinge");
                    lossFunctions.Add("FocalLoss");
                    AddAllGradientOptimizers(optimizers);
                    preprocessors.Add("StandardScaler");
                    preprocessors.Add("OneHotEncoder");
                    break;

                case 1: // Regression
                case 30: // Linear
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    lossFunctions.Add("Huber");
                    lossFunctions.Add("LogCosh");
                    AddAllGradientOptimizers(optimizers);
                    preprocessors.Add("StandardScaler");
                    preprocessors.Add("MinMaxScaler");
                    break;

                case 4: // GAN
                    lossFunctions.Add("Adversarial");
                    lossFunctions.Add("Wasserstein");
                    lossFunctions.Add("HingeLoss");
                    optimizers.Add("Adam");
                    optimizers.Add("RMSProp");
                    optimizers.Add("AdamW");
                    warnings.Add("SGD not recommended for GAN training");
                    preprocessors.Add("MinMaxScaler");
                    break;

                case 5: // Diffusion
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("L1");
                    lossFunctions.Add("VLB");
                    optimizers.Add("Adam");
                    optimizers.Add("AdamW");
                    preprocessors.Add("StandardScaler");
                    break;

                case 6: // Transformer
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("MSE");
                    optimizers.Add("Adam");
                    optimizers.Add("AdamW");
                    optimizers.Add("LAMB");
                    preprocessors.Add("TokenizerPreprocessor");
                    preprocessors.Add("StandardScaler");
                    break;

                case 14: // Autoencoder
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("BCE");
                    lossFunctions.Add("KLDivergence");
                    AddAllGradientOptimizers(optimizers);
                    preprocessors.Add("MinMaxScaler");
                    break;

                case 11: // SurvivalModel
                    lossFunctions.Add("CoxPartialLikelihood");
                    lossFunctions.Add("LogRankLoss");
                    optimizers.Add("Adam");
                    optimizers.Add("LBFGS");
                    preprocessors.Add("StandardScaler");
                    break;

                case 13: // TimeSeriesModel
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    lossFunctions.Add("QuantileLoss");
                    optimizers.Add("Adam");
                    optimizers.Add("AdaGrad");
                    optimizers.Add("RMSProp");
                    preprocessors.Add("TimeSeriesScaler");
                    preprocessors.Add("StandardScaler");
                    break;

                case 0: // NeuralNetwork (general)
                case 15: // RecurrentNetwork
                case 16: // ConvolutionalNetwork
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("MAE");
                    AddAllGradientOptimizers(optimizers);
                    preprocessors.Add("StandardScaler");
                    break;

                case 17: // GraphNetwork
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("MSE");
                    optimizers.Add("Adam");
                    optimizers.Add("AdamW");
                    preprocessors.Add("GraphNormalizer");
                    preprocessors.Add("StandardScaler");
                    break;

                case 22: // SyntheticDataGenerator
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("KLDivergence");
                    optimizers.Add("Adam");
                    preprocessors.Add("MinMaxScaler");
                    break;

                case 9: // Ensemble
                case 31: // DecisionTree
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    lossFunctions.Add("CrossEntropy");
                    // Tree models don't use gradient optimizers
                    optimizers.Add("BuiltIn");
                    preprocessors.Add("StandardScaler");
                    break;

                case 27: // SVM
                case 28: // Kernel
                    lossFunctions.Add("Hinge");
                    lossFunctions.Add("MSE");
                    optimizers.Add("SMO");
                    optimizers.Add("LBFGS");
                    preprocessors.Add("StandardScaler");
                    break;

                case 29: // InstanceBased
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    optimizers.Add("BuiltIn");
                    preprocessors.Add("StandardScaler");
                    preprocessors.Add("MinMaxScaler");
                    break;

                default:
                    // General defaults for unrecognized categories
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("CrossEntropy");
                    AddAllGradientOptimizers(optimizers);
                    preprocessors.Add("StandardScaler");
                    break;
            }
        }

        return (optimizers.OrderBy(o => o).ToList(),
                lossFunctions.OrderBy(l => l).ToList(),
                preprocessors.OrderBy(p => p).ToList(),
                warnings);
    }

    private static void AddAllGradientOptimizers(HashSet<string> optimizers)
    {
        optimizers.Add("SGD");
        optimizers.Add("Adam");
        optimizers.Add("AdamW");
        optimizers.Add("RMSProp");
        optimizers.Add("AdaGrad");
        optimizers.Add("Adadelta");
    }

    private static string BuildTypeOfExpression(CompatEntry entry)
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
        // Remove <T>, <T, U>, etc. from the end
        var angleBracketIdx = name.IndexOf('<');
        if (angleBracketIdx >= 0)
            name = name.Substring(0, angleBracketIdx);
        return name;
    }

    private static string FormatStringArray(IEnumerable<string> values)
    {
        var items = values.ToList();
        if (items.Count == 0)
            return string.Empty;

        return string.Join(", ", items.Select(v => $"\"{v}\""));
    }

    private class CompatEntry
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Categories { get; set; } = new List<int>();
    }
}
