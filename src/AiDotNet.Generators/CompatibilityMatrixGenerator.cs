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
    private const string ModelMetadataExemptAttr = "AiDotNet.Attributes.ModelMetadataExemptAttribute";

    // Named constants for ModelCategory enum values (must match AiDotNet.Enums.ModelCategory)
    private const int CatNeuralNetwork = 0;
    private const int CatRegression = 1;
    private const int CatClassifier = 2;
    private const int CatGAN = 4;
    private const int CatDiffusion = 5;
    private const int CatTransformer = 6;
    private const int CatEnsemble = 9;
    private const int CatSurvivalModel = 11;
    private const int CatTimeSeriesModel = 13;
    private const int CatAutoencoder = 14;
    private const int CatRecurrentNetwork = 15;
    private const int CatConvolutionalNetwork = 16;
    private const int CatGraphNetwork = 17;
    private const int CatSyntheticDataGenerator = 22;
    private const int CatSVM = 27;
    private const int CatKernel = 28;
    private const int CatInstanceBased = 29;
    private const int CatLinear = 30;
    private const int CatDecisionTree = 31;

    // Diagnostic for suspicious model/optimizer combinations
    private static readonly DiagnosticDescriptor SuspiciousOptimizer = new(
        id: "AIDN030",
        title: "Conflicting optimizer requirements across model categories",
        messageFormat: "Model '{0}' has categories '{1}' with conflicting optimizer requirements. Ensure the model's default optimizer is from the intersection of compatible optimizers for all its categories.",
        category: "AiDotNet.Compatibility",
        defaultSeverity: DiagnosticSeverity.Warning,
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
        var exemptAttrSymbol = compilation.GetTypeByMetadataName(ModelMetadataExemptAttr);
        var categoryEnumType = compilation.GetTypeByMetadataName("AiDotNet.Enums.ModelCategory");

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

            // Skip classes marked with [ModelMetadataExempt]
            if (exemptAttrSymbol is not null && HasAttribute(modelClass.GetAttributes(), exemptAttrSymbol))
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
                    Categories = categories,
                    Location = modelClass.Locations.Length > 0 ? modelClass.Locations[0] : null
                });
            }
        }

        entries.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        // Emit AIDN030 diagnostics for models whose categories have conflicting optimizer
        // requirements. For example, a model with [NeuralNetwork, GAN] categories: NeuralNetwork
        // allows all gradient optimizers (including SGD), but GAN restricts to Adam/RMSProp/AdamW.
        // The intersection removes SGD, and AIDN030 fires to flag the conflict so developers
        // ensure the model's default optimizer is from the safe intersection set.
        foreach (var entry in entries)
        {
            var (_, _, _, warnings) = GetCompatibilityRules(entry.Categories);
            if (warnings.Count > 0 && entry.Location is not null)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    SuspiciousOptimizer,
                    entry.Location,
                    entry.ClassName,
                    string.Join(", ", entry.Categories.Select(c => GetCategoryName(c, categoryEnumType)))));
            }
        }

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

        // IsCompatible method
        sb.AppendLine("    /// <summary>Checks whether the given model type is compatible with the given optimizer type.</summary>");
        sb.AppendLine("    /// <param name=\"modelType\">The model type to check.</param>");
        sb.AppendLine("    /// <param name=\"optimizerType\">The optimizer type to check against.</param>");
        sb.AppendLine("    /// <returns><c>true</c> if the combination is compatible or the model is unknown; otherwise <c>false</c>.</returns>");
        sb.AppendLine("    public static bool IsCompatible(Type modelType, Type optimizerType)");
        sb.AppendLine("    {");
        sb.AppendLine("        var info = GetInfo(modelType);");
        sb.AppendLine("        if (info is null)");
        sb.AppendLine("            return true; // Unknown models are assumed compatible");
        sb.AppendLine();
        sb.AppendLine("        var optimizerName = optimizerType.Name;");
        sb.AppendLine("        var backtick = optimizerName.IndexOf('`');");
        sb.AppendLine("        if (backtick >= 0)");
        sb.AppendLine("            optimizerName = optimizerName.Substring(0, backtick);");
        sb.AppendLine();
        sb.AppendLine("        foreach (var compat in info.CompatibleOptimizers)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (string.Equals(compat, optimizerName, StringComparison.OrdinalIgnoreCase))");
        sb.AppendLine("                return true;");
        sb.AppendLine("            if (optimizerName.IndexOf(compat, StringComparison.OrdinalIgnoreCase) >= 0)");
        sb.AppendLine("                return true;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        return false;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // IsCompatibleLossFunction method
        sb.AppendLine("    /// <summary>Checks whether the given model type is compatible with the given loss function type.</summary>");
        sb.AppendLine("    /// <param name=\"modelType\">The model type to check.</param>");
        sb.AppendLine("    /// <param name=\"lossFunctionType\">The loss function type to check against.</param>");
        sb.AppendLine("    /// <returns><c>true</c> if the combination is compatible or the model is unknown; otherwise <c>false</c>.</returns>");
        sb.AppendLine("    public static bool IsCompatibleLossFunction(Type modelType, Type lossFunctionType)");
        sb.AppendLine("    {");
        sb.AppendLine("        var info = GetInfo(modelType);");
        sb.AppendLine("        if (info is null)");
        sb.AppendLine("            return true; // Unknown models are assumed compatible");
        sb.AppendLine();
        sb.AppendLine("        var lossName = lossFunctionType.Name;");
        sb.AppendLine("        var backtick = lossName.IndexOf('`');");
        sb.AppendLine("        if (backtick >= 0)");
        sb.AppendLine("            lossName = lossName.Substring(0, backtick);");
        sb.AppendLine();
        sb.AppendLine("        foreach (var compat in info.CompatibleLossFunctions)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (string.Equals(compat, lossName, StringComparison.OrdinalIgnoreCase))");
        sb.AppendLine("                return true;");
        sb.AppendLine("            if (lossName.IndexOf(compat, StringComparison.OrdinalIgnoreCase) >= 0)");
        sb.AppendLine("                return true;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        return false;");
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
            sb.AppendLine($"            optimizers: new string[] {{ {FormatStringArray(optimizers)} }},");
            sb.AppendLine($"            lossFunctions: new string[] {{ {FormatStringArray(lossFunctions)} }},");
            sb.AppendLine($"            preprocessors: new string[] {{ {FormatStringArray(preprocessors)} }},");
            sb.AppendLine($"            warnings: new string[] {{ {FormatStringArray(warnings)} }});");
        }

        sb.AppendLine("        return dict;");
        sb.AppendLine("    }");

        sb.AppendLine("}");

        context.AddSource("ModelCompatibility.g.cs", sb.ToString());
    }

    private static (List<string> optimizers, List<string> lossFunctions, List<string> preprocessors, List<string> warnings) GetCompatibilityRules(List<int> categories)
    {
        var lossFunctions = new HashSet<string>();
        var preprocessors = new HashSet<string>();
        var warnings = new List<string>();

        // Collect each category's optimizer set separately so we can intersect them.
        // Union is used for loss functions and preprocessors (additive),
        // but intersection is used for optimizers (restrictive — every category must agree).
        var perCategoryOptimizers = new List<HashSet<string>>();

        foreach (var cat in categories)
        {
            var catOptimizers = new HashSet<string>();
            switch (cat)
            {
                case CatClassifier:
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("BinaryCrossEntropy");
                    lossFunctions.Add("Hinge");
                    lossFunctions.Add("FocalLoss");
                    AddAllGradientOptimizers(catOptimizers);
                    preprocessors.Add("StandardScaler");
                    preprocessors.Add("OneHotEncoder");
                    break;

                case CatRegression:
                case CatLinear:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    lossFunctions.Add("Huber");
                    lossFunctions.Add("LogCosh");
                    AddAllGradientOptimizers(catOptimizers);
                    preprocessors.Add("StandardScaler");
                    preprocessors.Add("MinMaxScaler");
                    break;

                case CatGAN:
                    lossFunctions.Add("Adversarial");
                    lossFunctions.Add("Wasserstein");
                    lossFunctions.Add("HingeLoss");
                    catOptimizers.Add("Adam");
                    catOptimizers.Add("RMSProp");
                    catOptimizers.Add("AdamW");
                    preprocessors.Add("MinMaxScaler");
                    break;

                case CatDiffusion:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("L1");
                    lossFunctions.Add("VLB");
                    catOptimizers.Add("Adam");
                    catOptimizers.Add("AdamW");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatTransformer:
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("MSE");
                    catOptimizers.Add("Adam");
                    catOptimizers.Add("AdamW");
                    catOptimizers.Add("LAMB");
                    preprocessors.Add("TokenizerPreprocessor");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatAutoencoder:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("BCE");
                    lossFunctions.Add("KLDivergence");
                    AddAllGradientOptimizers(catOptimizers);
                    preprocessors.Add("MinMaxScaler");
                    break;

                case CatSurvivalModel:
                    lossFunctions.Add("CoxPartialLikelihood");
                    lossFunctions.Add("LogRankLoss");
                    catOptimizers.Add("Adam");
                    catOptimizers.Add("LBFGS");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatTimeSeriesModel:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    lossFunctions.Add("QuantileLoss");
                    catOptimizers.Add("Adam");
                    catOptimizers.Add("AdaGrad");
                    catOptimizers.Add("RMSProp");
                    preprocessors.Add("TimeSeriesScaler");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatNeuralNetwork:
                case CatRecurrentNetwork:
                case CatConvolutionalNetwork:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("MAE");
                    AddAllGradientOptimizers(catOptimizers);
                    preprocessors.Add("StandardScaler");
                    break;

                case CatGraphNetwork:
                    lossFunctions.Add("CrossEntropy");
                    lossFunctions.Add("MSE");
                    catOptimizers.Add("Adam");
                    catOptimizers.Add("AdamW");
                    preprocessors.Add("GraphNormalizer");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatSyntheticDataGenerator:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("KLDivergence");
                    catOptimizers.Add("Adam");
                    preprocessors.Add("MinMaxScaler");
                    break;

                case CatEnsemble:
                case CatDecisionTree:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    lossFunctions.Add("CrossEntropy");
                    // Tree models don't use gradient optimizers
                    catOptimizers.Add("BuiltIn");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatSVM:
                case CatKernel:
                    lossFunctions.Add("Hinge");
                    lossFunctions.Add("MSE");
                    catOptimizers.Add("SMO");
                    catOptimizers.Add("LBFGS");
                    preprocessors.Add("StandardScaler");
                    break;

                case CatInstanceBased:
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("MAE");
                    catOptimizers.Add("BuiltIn");
                    preprocessors.Add("StandardScaler");
                    preprocessors.Add("MinMaxScaler");
                    break;

                default:
                    // General defaults for unrecognized categories
                    lossFunctions.Add("MSE");
                    lossFunctions.Add("CrossEntropy");
                    AddAllGradientOptimizers(catOptimizers);
                    preprocessors.Add("StandardScaler");
                    break;
            }

            if (catOptimizers.Count > 0)
                perCategoryOptimizers.Add(catOptimizers);
        }

        // Build the final optimizer set. Use union across categories for the compatibility
        // matrix (what CAN work), but check for conflicts: if a restrictive category (GAN,
        // Diffusion, Transformer, etc.) is combined with a permissive one (NeuralNetwork),
        // the restrictive category's set should be the final answer. When this happens, we
        // emit AIDN030 so developers verify their model uses an optimizer from the safe set.
        var optimizers = new HashSet<string>();
        foreach (var set in perCategoryOptimizers)
            optimizers.UnionWith(set);

        if (optimizers.Count == 0)
            AddAllGradientOptimizers(optimizers);

        // Detect conflicts: if we have multiple category optimizer sets and they disagree,
        // compute the intersection to find the safe set. If the safe set is smaller than
        // the union, there's a conflict worth flagging.
        if (perCategoryOptimizers.Count > 1)
        {
            var safeSet = new HashSet<string>(perCategoryOptimizers[0]);
            for (int i = 1; i < perCategoryOptimizers.Count; i++)
                safeSet.IntersectWith(perCategoryOptimizers[i]);

            // Only warn if the intersection actually removed optimizers that are known to be
            // problematic for certain model types (e.g., SGD removed because GAN category
            // doesn't include it). We use the intersection as the final optimizer set to
            // enforce that all categories agree.
            if (safeSet.Count > 0 && safeSet.Count < optimizers.Count)
            {
                var removed = new HashSet<string>(optimizers);
                removed.ExceptWith(safeSet);

                // Use the safe intersection as the actual optimizer set
                optimizers = safeSet;
                warnings.Add($"Incompatible optimizers removed: {string.Join(", ", removed.OrderBy(r => r))}");
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
        => GeneratorHelpers.BuildTypeOfExpression(entry.FullyQualifiedName, entry.TypeParameterCount);

    private static string FormatStringArray(IEnumerable<string> values)
        => GeneratorHelpers.FormatStringArray(values);

    private static string GetCategoryName(int categoryValue, INamedTypeSymbol? categoryEnumType)
        => GeneratorHelpers.GetEnumName(categoryValue, GeneratorHelpers.CategoryNames, categoryEnumType);

    private static bool HasAttribute(ImmutableArray<AttributeData> attributes, INamedTypeSymbol attributeType)
    {
        foreach (var attr in attributes)
        {
            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, attributeType))
                return true;
        }
        return false;
    }

    private class CompatEntry
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Categories { get; set; } = new List<int>();
        public Location? Location { get; set; }
    }
}
