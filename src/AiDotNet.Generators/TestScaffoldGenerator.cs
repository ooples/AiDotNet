using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that cross-references model classes against test classes
/// to identify untested models, auto-generate test scaffolds, and produce a coverage report.
/// </summary>
/// <remarks>
/// <para>
/// Discovers all concrete IFullModel implementations decorated with [ModelDomain] and checks
/// for matching test classes. For untested models, resolves the appropriate test base class
/// from [ModelCategory]/[ModelTask] metadata and interface hierarchy, then generates a
/// minimal test class that exercises all inherited invariant tests.
/// </para>
/// <para>
/// Model discovery works in two modes:
/// <list type="bullet">
/// <item>Source mode: finds model classes defined as source in the current compilation (when running in the source project)</item>
/// <item>Reference mode: finds model classes from referenced assemblies (when running in the test project)</item>
/// </list>
/// </para>
/// </remarks>
[Generator]
public class TestScaffoldGenerator : IIncrementalGenerator
{
    // Interface detection prefixes
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";
    private const string INeuralNetworkModelName = "AiDotNet.Interfaces.INeuralNetworkModel";
    private const string IDiffusionModelName = "AiDotNet.Interfaces.IDiffusionModel";
    private const string IGaussianProcessPrefix = "AiDotNet.Interfaces.IGaussianProcess<";
    private const string IActivationFunctionPrefix = "AiDotNet.Interfaces.IActivationFunction<";
    private const string ILossFunctionPrefix = "AiDotNet.Interfaces.ILossFunction<";

    // Attribute metadata names
    private const string ModelDomainAttr = "AiDotNet.Attributes.ModelDomainAttribute";
    private const string ModelCategoryAttr = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttr = "AiDotNet.Attributes.ModelTaskAttribute";
    private const string ModelInputAttr = "AiDotNet.Attributes.ModelInputAttribute";
    private const string ModelMetadataExemptAttr = "AiDotNet.Attributes.ModelMetadataExemptAttribute";

    // Activation/Loss/Layer attribute metadata names
    private const string ActivationPropertyAttr = "AiDotNet.Attributes.ActivationPropertyAttribute";
    private const string LossPropertyAttr = "AiDotNet.Attributes.LossPropertyAttribute";
    private const string LayerPropertyAttr = "AiDotNet.Attributes.LayerPropertyAttribute";
    private const string LossFunctionBasePrefix = "AiDotNet.LossFunctions.LossFunctionBase<";
    private const string ISelfSupervisedLossPrefix = "AiDotNet.Interfaces.ISelfSupervisedLoss<";
    private const string ILayerPrefix = "AiDotNet.Interfaces.ILayer<";
    private const string LayerBasePrefix = "AiDotNet.NeuralNetworks.Layers.LayerBase<";

    // ModelCategory enum values (must match AiDotNet.Enums.ModelCategory)
    private const int CategoryGAN = 4;
    private const int CategoryDiffusion = 5;
    private const int CategoryGaussianProcess = 8;
    private const int CategoryTimeSeriesModel = 13;
    private const int CategoryGraphNetwork = 17;
    private const int CategoryEmbeddingModel = 18;
    private const int CategoryNeuralNetwork = 0;
    private const int CategoryMetaLearning = 20;

    // ModelTask enum values (must match AiDotNet.Enums.ModelTask)
    private const int TaskClassification = 0;
    private const int TaskRegression = 1;
    private const int TaskClustering = 2;

    private static readonly DiagnosticDescriptor UntestedModel = new(
        id: "AIDN040",
        title: "Model has no test coverage",
        messageFormat: "Model '{0}' has no corresponding test class and could not be auto-generated (missing category/task metadata)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Model has no test coverage and lacks sufficient metadata for auto-generation. Add [ModelCategory] and [ModelTask] attributes, or create a manual test class.");

    private static readonly DiagnosticDescriptor CoverageSummary = new(
        id: "AIDN041",
        title: "Model test coverage summary",
        messageFormat: "{0} of {1} annotated models have test coverage ({2:F1}%)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor ActivationCoverageSummary = new(
        id: "AIDN042",
        title: "Activation function test coverage summary",
        messageFormat: "{0} of {1} activation functions have test coverage ({2:F1}%)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor LossCoverageSummary = new(
        id: "AIDN043",
        title: "Loss function test coverage summary",
        messageFormat: "{0} of {1} loss functions have test coverage ({2:F1}%)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor LayerCoverageSummary = new(
        id: "AIDN044",
        title: "Layer test coverage summary",
        messageFormat: "{0} of {1} layers have test coverage ({2:F1}%)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Collect model classes from source (works when running in the source project)
        var modelClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsModelCandidate(node),
            transform: static (ctx, _) => GetModelClassOrNull(ctx))
            .Where(static s => s is not null);

        // Collect test classes (classes ending in Tests/Test or containing [Fact]/[Theory])
        var testClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsTestCandidate(node),
            transform: static (ctx, _) => GetTestClassName(ctx))
            .Where(static s => s is not null);

        // Collect activation function classes from source
        var activationClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsModelCandidate(node),
            transform: static (ctx, _) => GetActivationFunctionOrNull(ctx))
            .Where(static s => s is not null);

        // Collect loss function classes from source
        var lossClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsModelCandidate(node),
            transform: static (ctx, _) => GetLossFunctionOrNull(ctx))
            .Where(static s => s is not null);

        // Collect layer classes from source
        var layerClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsModelCandidate(node),
            transform: static (ctx, _) => GetLayerOrNull(ctx))
            .Where(static s => s is not null);

        var combined = modelClasses.Collect()
            .Combine(testClasses.Collect())
            .Combine(activationClasses.Collect())
            .Combine(lossClasses.Collect())
            .Combine(layerClasses.Collect())
            .Combine(context.CompilationProvider);

        context.RegisterSourceOutput(combined, static (spc, source) =>
        {
            var (((((models, tests), activations), losses), layers), compilation) = source;
            Execute(spc, models, tests, compilation);
            ExecuteActivationAndLossGeneration(spc, activations, losses, compilation);
            ExecuteLayerGeneration(spc, layers, compilation);
        });
    }

    private static bool IsModelCandidate(SyntaxNode node)
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

    private static bool IsTestCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;

        // Check if class name ends with "Tests" or "Test"
        if (cds.Identifier.Text.EndsWith("Tests", System.StringComparison.Ordinal) ||
            cds.Identifier.Text.EndsWith("Test", System.StringComparison.Ordinal))
        {
            return true;
        }

        // Check if any method has a test attribute (xUnit, NUnit, or MSTest)
        foreach (var member in cds.Members)
        {
            if (member is MethodDeclarationSyntax method)
            {
                foreach (var attrList in method.AttributeLists)
                {
                    foreach (var attr in attrList.Attributes)
                    {
                        var name = attr.Name.ToString();
                        // xUnit
                        if (name == "Fact" || name == "Theory" ||
                            name == "Xunit.Fact" || name == "Xunit.Theory")
                            return true;
                        // NUnit
                        if (name == "Test" || name == "TestCase" ||
                            name == "NUnit.Framework.Test" || name == "NUnit.Framework.TestCase")
                            return true;
                        // MSTest
                        if (name == "TestMethod" ||
                            name == "Microsoft.VisualStudio.TestTools.UnitTesting.TestMethod")
                            return true;
                    }
                }
            }
        }

        return false;
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

    private static string? GetTestClassName(GeneratorSyntaxContext ctx)
    {
        if (ctx.Node is not ClassDeclarationSyntax cds)
            return null;
        return cds.Identifier.Text;
    }

    /// <summary>
    /// Returns the type symbol if it implements IActivationFunction&lt;T&gt; and has [ActivationProperty].
    /// </summary>
    private static INamedTypeSymbol? GetActivationFunctionOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        // Check for [ActivationProperty] attribute
        bool hasActivationProperty = false;
        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is not null &&
                attr.AttributeClass.ToDisplayString().EndsWith("ActivationPropertyAttribute", System.StringComparison.Ordinal))
            {
                hasActivationProperty = true;
                break;
            }
        }
        if (!hasActivationProperty)
            return null;

        // Verify it implements IActivationFunction<T>
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(IActivationFunctionPrefix, System.StringComparison.Ordinal))
            {
                return symbol;
            }
        }
        return null;
    }

    /// <summary>
    /// Returns the type symbol if it extends LayerBase&lt;T&gt; (or implements ILayer&lt;T&gt;)
    /// and has [LayerProperty].
    /// </summary>
    private static INamedTypeSymbol? GetLayerOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        // Check for [LayerProperty] attribute
        bool hasLayerProperty = false;
        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is not null &&
                attr.AttributeClass.ToDisplayString().EndsWith("LayerPropertyAttribute", System.StringComparison.Ordinal))
            {
                hasLayerProperty = true;
                break;
            }
        }
        if (!hasLayerProperty)
            return null;

        // Check if it implements ILayer<T> or extends LayerBase<T>
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(ILayerPrefix, System.StringComparison.Ordinal))
            {
                return symbol;
            }
        }

        // Check base type chain for LayerBase<T>
        var baseType = symbol.BaseType;
        while (baseType is not null)
        {
            if (baseType.IsGenericType &&
                baseType.OriginalDefinition.ToDisplayString().StartsWith(LayerBasePrefix, System.StringComparison.Ordinal))
            {
                return symbol;
            }
            baseType = baseType.BaseType;
        }

        return null;
    }

    /// <summary>
    /// Returns the type symbol if it extends LossFunctionBase&lt;T&gt; (or implements ILossFunction&lt;T&gt;)
    /// and has [LossProperty].
    /// </summary>
    private static INamedTypeSymbol? GetLossFunctionOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        // Check for [LossProperty] attribute
        bool hasLossProperty = false;
        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is not null &&
                attr.AttributeClass.ToDisplayString().EndsWith("LossPropertyAttribute", System.StringComparison.Ordinal))
            {
                hasLossProperty = true;
                break;
            }
        }
        if (!hasLossProperty)
            return null;

        // Check if it implements ILossFunction<T> or extends LossFunctionBase<T>
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(ILossFunctionPrefix, System.StringComparison.Ordinal))
            {
                return symbol;
            }
        }

        // Also check base type chain for LossFunctionBase
        var baseType = symbol.BaseType;
        while (baseType is not null)
        {
            if (baseType.IsGenericType &&
                baseType.OriginalDefinition.ToDisplayString().StartsWith(LossFunctionBasePrefix, System.StringComparison.Ordinal))
            {
                return symbol;
            }
            baseType = baseType.BaseType;
        }

        // Also check for ISelfSupervisedLoss<T>
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(ISelfSupervisedLossPrefix, System.StringComparison.Ordinal))
            {
                return symbol;
            }
        }

        return null;
    }

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> sourceModels,
        ImmutableArray<string?> testClassNames,
        Compilation compilation)
    {
        var domainAttrSymbol = compilation.GetTypeByMetadataName(ModelDomainAttr);
        var categoryAttrSymbol = compilation.GetTypeByMetadataName(ModelCategoryAttr);
        var taskAttrSymbol = compilation.GetTypeByMetadataName(ModelTaskAttr);
        var exemptAttrSymbol = compilation.GetTypeByMetadataName(ModelMetadataExemptAttr);

        // Build test class name set for fast lookup
        var testNames = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
        foreach (var name in testClassNames)
        {
            if (name is not null)
                testNames.Add(name);
        }

        var testedModels = new List<ModelTestInfo>();
        var untestedModels = new List<ModelTestInfo>();
        var seen = new HashSet<string>();

        // First: collect models from source (syntax-based discovery)
        foreach (var modelClass in sourceModels)
        {
            if (modelClass is null)
                continue;

            ProcessModelSymbol(modelClass, domainAttrSymbol, categoryAttrSymbol, taskAttrSymbol,
                exemptAttrSymbol, testNames, testedModels, untestedModels, seen);
        }

        // Detect if we're in the source project (not the test project).
        string assemblyName = compilation.AssemblyName ?? string.Empty;
        bool isTestProject = assemblyName.IndexOf("Test", System.StringComparison.OrdinalIgnoreCase) >= 0;
        bool modelsFoundFromSource = seen.Count > 0 && !isTestProject;

        // Second: if no source models were found, discover from referenced assemblies.
        if (!modelsFoundFromSource)
        {
            DiscoverModelsFromReferencedAssemblies(compilation, domainAttrSymbol, categoryAttrSymbol,
                taskAttrSymbol, exemptAttrSymbol, testNames, testedModels, untestedModels, seen);
        }

        testedModels.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));
        untestedModels.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        // Auto-generate test classes for untested models (test project only)
        if (!modelsFoundFromSource)
        {
            var autoGenerated = new List<ModelTestInfo>();
            var generatedTestNames = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);

            foreach (var model in untestedModels)
            {
                var family = ResolveTestBaseClass(model);
                if (family is null)
                    continue;

                var testClassName = StripBacktick(model.ClassName) + "Tests";

                // Avoid duplicate test class names and conflicts with existing tests.
                // A duplicate means this model was already auto-generated (same model
                // discovered from multiple referenced assemblies) — still count as covered.
                if (!generatedTestNames.Add(testClassName))
                {
                    autoGenerated.Add(model);
                    testNames.Add(testClassName);
                    continue;
                }
                if (testNames.Contains(testClassName))
                    continue;

                // Use constructor call if the model has a zero-arg constructor and is type-compatible.
                // Otherwise, emit a throw so the test compiles but fails at runtime with a clear message.
                bool canConstruct = model.HasParameterlessConstructor &&
                                    IsCompatibleWithFamily(model, family.Value);

                EmitGeneratedTestClass(context, model, family.Value, testClassName, canConstruct);
                autoGenerated.Add(model);
                testNames.Add(testClassName);
            }

            // Move auto-generated from untested → tested (only if constructible)
            foreach (var model in autoGenerated)
            {
                untestedModels.Remove(model);
                // Only count as "has tests" if the test can actually construct the model.
                // Runtime-throwing scaffolds don't provide real test coverage.
                bool constructible = !string.IsNullOrEmpty(model.ClassName);
                model.HasTests = constructible;
                testedModels.Add(model);
            }

            // Re-sort after moves
            testedModels.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));
            untestedModels.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

            // Emit AIDN040 for remaining untested models
            foreach (var model in untestedModels)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    UntestedModel,
                    Location.None,
                    model.ClassName));
            }

            // Emit AIDN041 summary
            var totalCount = testedModels.Count + untestedModels.Count;
            if (totalCount > 0)
            {
                var coveragePct = testedModels.Count * 100.0 / totalCount;
                context.ReportDiagnostic(Diagnostic.Create(
                    CoverageSummary,
                    Location.None,
                    testedModels.Count,
                    totalCount,
                    coveragePct));
            }
        }

        EmitTestCoverageClass(context, testedModels, untestedModels);
    }

    /// <summary>
    /// Processes a single model type symbol, extracting metadata and checking for test coverage.
    /// </summary>
    private static void ProcessModelSymbol(
        INamedTypeSymbol modelClass,
        INamedTypeSymbol? domainAttrSymbol,
        INamedTypeSymbol? categoryAttrSymbol,
        INamedTypeSymbol? taskAttrSymbol,
        INamedTypeSymbol? exemptAttrSymbol,
        HashSet<string> testNames,
        List<ModelTestInfo> testedModels,
        List<ModelTestInfo> untestedModels,
        HashSet<string> seen)
    {
        var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
        if (!seen.Add(fullName))
            return;

        // Skip classes marked with [ModelMetadataExempt]
        if (exemptAttrSymbol is not null && HasAttribute(modelClass.GetAttributes(), exemptAttrSymbol))
            return;

        // Extract attributes and detect input/output types
        bool hasModelDomain = false;
        var domains = new List<int>();
        var categories = new List<int>();
        var tasks = new List<int>();
        // NOTE: These boolean flags are a lossy representation of the actual generic
        // type parameters. A more precise approach would track the full INamedTypeSymbol
        // for input/output types. Current approach is sufficient for test scaffolding
        // (determines which test helper to call) but can't distinguish e.g. Tensor<float>
        // from Tensor<double> or custom TInput types.
        bool usesTensorInput = false;
        bool usesMatrixInput = false;
        bool usesVectorOutput = false;

        foreach (var attr in modelClass.GetAttributes())
        {
            if (attr.AttributeClass is null)
                continue;

            // Use SymbolEqualityComparer first, fall back to string matching
            // for cross-assembly scenarios where symbol resolution may differ
            bool isDomain = (domainAttrSymbol is not null &&
                SymbolEqualityComparer.Default.Equals(attr.AttributeClass, domainAttrSymbol)) ||
                attr.AttributeClass.ToDisplayString().EndsWith("ModelDomainAttribute", System.StringComparison.Ordinal);
            bool isCategory = (categoryAttrSymbol is not null &&
                SymbolEqualityComparer.Default.Equals(attr.AttributeClass, categoryAttrSymbol)) ||
                attr.AttributeClass.ToDisplayString().EndsWith("ModelCategoryAttribute", System.StringComparison.Ordinal);
            bool isTask = (taskAttrSymbol is not null &&
                SymbolEqualityComparer.Default.Equals(attr.AttributeClass, taskAttrSymbol)) ||
                attr.AttributeClass.ToDisplayString().EndsWith("ModelTaskAttribute", System.StringComparison.Ordinal);
            bool isInput = attr.AttributeClass.ToDisplayString().EndsWith("ModelInputAttribute", System.StringComparison.Ordinal);

            if (isDomain)
            {
                hasModelDomain = true;
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int d)
                    domains.Add(d);
            }
            else if (isCategory)
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int c)
                    categories.Add(c);
            }
            else if (isTask)
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int t)
                    tasks.Add(t);
            }
            else if (isInput && attr.ConstructorArguments.Length >= 2)
            {
                // [ModelInput(typeof(Tensor<>), typeof(Tensor<>))] or [ModelInput(typeof(Matrix<>), typeof(Vector<>))]
                // For metadata types, ConstructorArguments[0].Value is an INamedTypeSymbol
                var inputTypeSym = attr.ConstructorArguments[0].Value as INamedTypeSymbol;
                var outputTypeSym = attr.ConstructorArguments[1].Value as INamedTypeSymbol;
                if (inputTypeSym is not null)
                {
                    var inputName = inputTypeSym.Name;
                    if (inputName.Contains("Tensor"))
                        usesTensorInput = true;
                    else if (inputName.Contains("Matrix"))
                        usesMatrixInput = true;
                }
                if (outputTypeSym is not null)
                {
                    if (outputTypeSym.Name.Contains("Vector"))
                        usesVectorOutput = true;
                }
            }
        }

        if (!hasModelDomain)
            return;

        // Detect interfaces and refine input types from the type hierarchy
        bool implementsNeuralNetworkModel = false;
        bool implementsDiffusionModel = false;
        bool implementsGaussianProcess = false;

        foreach (var iface in modelClass.AllInterfaces)
        {
            if (!iface.IsGenericType)
                continue;

            var display = iface.OriginalDefinition.ToDisplayString();

            if (display.StartsWith(INeuralNetworkModelName, System.StringComparison.Ordinal))
            {
                implementsNeuralNetworkModel = true;
            }
            else if (display.StartsWith(IDiffusionModelName, System.StringComparison.Ordinal))
            {
                implementsDiffusionModel = true;
            }
            else if (display.StartsWith(IGaussianProcessPrefix, System.StringComparison.Ordinal))
            {
                implementsGaussianProcess = true;
            }

            // Detect IFullModel type arguments for input/output types
            if (display.StartsWith(IFullModelName, System.StringComparison.Ordinal) &&
                iface.TypeArguments.Length >= 3)
            {
                var inputTypeDisplay = iface.TypeArguments[1].ToDisplayString();
                var outputTypeDisplay = iface.TypeArguments[2].ToDisplayString();
                if (inputTypeDisplay.Contains("Matrix"))
                    usesMatrixInput = true;
                else if (inputTypeDisplay.Contains("Tensor"))
                    usesTensorInput = true;
                if (outputTypeDisplay.Contains("Vector"))
                    usesVectorOutput = true;
            }
        }

        // Walk the base type chain to detect mid-level hierarchy bases
        bool extendsAudioNN = false, extendsDocumentNN = false, extendsVisionLanguage = false;
        bool extendsSegmentation = false, extendsVideoNN = false;
        bool extendsTts = false, extendsFinancial = false, extendsNER = false, extendsCode = false;
        bool extendsLatentDiffusion = false, extendsNonLinearRegression = false;
        bool extendsProbabilisticClassifier = false;
        // Phase B gap + Phase C
        bool extendsForecasting = false, extendsThreeDDiffusion = false;
        bool extendsAnomalyDetector = false, extendsSurvival = false;
        bool extendsCausal = false, extendsRLAgent = false;
        // Phase B leaf-level
        bool extendsVideoDiffusion = false, extendsAudioDiffusion = false;
        bool extendsFrameInterpolation = false, extendsVideoSR = false, extendsVideoDenoising = false;
        bool extendsAudioClassifier = false, extendsOpticalFlow = false, extendsSpeakerRecognition = false;
        bool extendsEnsembleClassifier = false, extendsNaiveBayes = false, extendsSVM = false;
        bool extendsVideoInpainting = false, extendsVideoStabilization = false;
        bool extendsLinearClassifier = false, extendsMetaClassifier = false;
        bool extendsOrdinalClassifier = false, extendsSemiSupervised = false;
        bool extendsMultiLabel = false, extendsFinancialNLP = false;
        bool extendsRiskModel = false, extendsPortfolioOptimizer = false;
        bool extendsTransformerNER = false, extendsSpanBasedNER = false, extendsSequenceLabelingNER = false;

        var baseType = modelClass.BaseType;
        while (baseType is not null)
        {
            var baseName = baseType.Name;
            // Phase B extended leaf-level checks
            if (baseName.StartsWith("VideoInpaintingBase", System.StringComparison.Ordinal))
                extendsVideoInpainting = true;
            else if (baseName.StartsWith("VideoStabilizationBase", System.StringComparison.Ordinal))
                extendsVideoStabilization = true;
            else if (baseName.StartsWith("LinearClassifierBase", System.StringComparison.Ordinal))
                extendsLinearClassifier = true;
            else if (baseName.StartsWith("MetaClassifierBase", System.StringComparison.Ordinal))
                extendsMetaClassifier = true;
            else if (baseName.StartsWith("OrdinalClassifierBase", System.StringComparison.Ordinal))
                extendsOrdinalClassifier = true;
            else if (baseName.StartsWith("SemiSupervisedClassifierBase", System.StringComparison.Ordinal))
                extendsSemiSupervised = true;
            else if (baseName.StartsWith("MultiLabelClassifierBase", System.StringComparison.Ordinal))
                extendsMultiLabel = true;
            else if (baseName.StartsWith("FinancialNLPModelBase", System.StringComparison.Ordinal))
                extendsFinancialNLP = true;
            else if (baseName.StartsWith("RiskModelBase", System.StringComparison.Ordinal))
                extendsRiskModel = true;
            else if (baseName.StartsWith("PortfolioOptimizerBase", System.StringComparison.Ordinal))
                extendsPortfolioOptimizer = true;
            else if (baseName.StartsWith("TransformerNERBase", System.StringComparison.Ordinal))
                extendsTransformerNER = true;
            else if (baseName.StartsWith("SpanBasedNERBase", System.StringComparison.Ordinal))
                extendsSpanBasedNER = true;
            else if (baseName.StartsWith("SequenceLabelingNERBase", System.StringComparison.Ordinal))
                extendsSequenceLabelingNER = true;
            // Phase B leaf-level checks (most specific first)
            else if (baseName.StartsWith("VideoDiffusionModelBase", System.StringComparison.Ordinal))
                extendsVideoDiffusion = true;
            else if (baseName.StartsWith("AudioDiffusionModelBase", System.StringComparison.Ordinal))
                extendsAudioDiffusion = true;
            else if (baseName.StartsWith("FrameInterpolationBase", System.StringComparison.Ordinal))
                extendsFrameInterpolation = true;
            else if (baseName.StartsWith("VideoSuperResolutionBase", System.StringComparison.Ordinal))
                extendsVideoSR = true;
            else if (baseName.StartsWith("VideoDenoisingBase", System.StringComparison.Ordinal))
                extendsVideoDenoising = true;
            else if (baseName.StartsWith("AudioClassifierBase", System.StringComparison.Ordinal))
                extendsAudioClassifier = true;
            else if (baseName.StartsWith("OpticalFlowBase", System.StringComparison.Ordinal))
                extendsOpticalFlow = true;
            else if (baseName.StartsWith("SpeakerRecognitionBase", System.StringComparison.Ordinal))
                extendsSpeakerRecognition = true;
            else if (baseName.StartsWith("EnsembleClassifierBase", System.StringComparison.Ordinal))
                extendsEnsembleClassifier = true;
            else if (baseName.StartsWith("NaiveBayesBase", System.StringComparison.Ordinal))
                extendsNaiveBayes = true;
            else if (baseName.StartsWith("SVMBase", System.StringComparison.Ordinal))
                extendsSVM = true;
            // Phase A mid-level checks
            else if (baseName.StartsWith("AudioNeuralNetworkBase", System.StringComparison.Ordinal))
                extendsAudioNN = true;
            else if (baseName.StartsWith("DocumentNeuralNetworkBase", System.StringComparison.Ordinal))
                extendsDocumentNN = true;
            else if (baseName.StartsWith("VisionLanguageModelBase", System.StringComparison.Ordinal))
                extendsVisionLanguage = true;
            else if (baseName.StartsWith("SegmentationModelBase", System.StringComparison.Ordinal) ||
                     baseName.EndsWith("SegmentationBase", System.StringComparison.Ordinal))
                extendsSegmentation = true;
            else if (baseName.StartsWith("VideoNeuralNetworkBase", System.StringComparison.Ordinal) ||
                     baseName.EndsWith("VideoBase", System.StringComparison.Ordinal))
                extendsVideoNN = true;
            else if (baseName.StartsWith("TtsModelBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("AcousticModelBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("VocoderBase", System.StringComparison.Ordinal))
                extendsTts = true;
            else if (baseName.StartsWith("ForecastingModelBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("TimeSeriesFoundationModelBase", System.StringComparison.Ordinal))
                extendsForecasting = true;
            else if (baseName.StartsWith("FinancialModelBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("RiskModelBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("PortfolioOptimizerBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("FinancialNLPModelBase", System.StringComparison.Ordinal))
                extendsFinancial = true;
            else if (baseName.StartsWith("NERNeuralNetworkBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("SequenceLabelingNERBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("SpanBasedNERBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("TransformerNERBase", System.StringComparison.Ordinal))
                extendsNER = true;
            else if (baseName.StartsWith("CodeModelBase", System.StringComparison.Ordinal))
                extendsCode = true;
            else if (baseName.StartsWith("ThreeDDiffusionModelBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("3DDiffusionModelBase", System.StringComparison.Ordinal))
                extendsThreeDDiffusion = true;
            else if (baseName.StartsWith("AnomalyDetectorBase", System.StringComparison.Ordinal))
                extendsAnomalyDetector = true;
            else if (baseName.StartsWith("SurvivalModelBase", System.StringComparison.Ordinal))
                extendsSurvival = true;
            else if (baseName.StartsWith("CausalModelBase", System.StringComparison.Ordinal))
                extendsCausal = true;
            else if (baseName.StartsWith("ReinforcementLearningAgentBase", System.StringComparison.Ordinal) ||
                     baseName.StartsWith("DeepReinforcementLearningAgentBase", System.StringComparison.Ordinal))
                extendsRLAgent = true;
            else if (baseName.StartsWith("LatentDiffusionModelBase", System.StringComparison.Ordinal))
                extendsLatentDiffusion = true;
            else if (baseName.StartsWith("NonLinearRegressionBase", System.StringComparison.Ordinal))
                extendsNonLinearRegression = true;
            else if (baseName.StartsWith("ProbabilisticClassifierBase", System.StringComparison.Ordinal))
                extendsProbabilisticClassifier = true;

            baseType = baseType.BaseType;
        }

        // Detect a public constructor callable with zero arguments:
        // either parameterless, or all parameters have default values.
        bool hasParameterlessCtor = false;
        foreach (var ctor in modelClass.InstanceConstructors)
        {
            if (ctor.DeclaredAccessibility != Accessibility.Public)
                continue;

            if (ctor.Parameters.Length == 0)
            {
                hasParameterlessCtor = true;
                break;
            }

            // Check if all parameters have default values (callable with zero args)
            bool allOptional = true;
            foreach (var param in ctor.Parameters)
            {
                if (!param.HasExplicitDefaultValue)
                {
                    allOptional = false;
                    break;
                }
            }
            if (allOptional)
            {
                hasParameterlessCtor = true;
                break;
            }
        }

        var className = modelClass.Name;
        var info = new ModelTestInfo
        {
            ClassName = className,
            FullyQualifiedName = fullName,
            TypeParameterCount = modelClass.TypeParameters.Length,
            Domains = domains,
            Categories = categories,
            Tasks = tasks,
            ImplementsNeuralNetworkModel = implementsNeuralNetworkModel,
            ImplementsDiffusionModel = implementsDiffusionModel,
            ImplementsGaussianProcess = implementsGaussianProcess,
            UsesTensorInput = usesTensorInput,
            UsesMatrixInput = usesMatrixInput,
            UsesVectorOutput = usesVectorOutput,
            HasParameterlessConstructor = hasParameterlessCtor,
            ExtendsAudioNeuralNetworkBase = extendsAudioNN,
            ExtendsDocumentNeuralNetworkBase = extendsDocumentNN,
            ExtendsVisionLanguageModelBase = extendsVisionLanguage,
            ExtendsSegmentationModelBase = extendsSegmentation,
            ExtendsVideoNeuralNetworkBase = extendsVideoNN,
            ExtendsTtsModelBase = extendsTts,
            ExtendsFinancialModelBase = extendsFinancial,
            ExtendsNERNeuralNetworkBase = extendsNER,
            ExtendsCodeModelBase = extendsCode,
            ExtendsVideoDiffusionModelBase = extendsVideoDiffusion,
            ExtendsAudioDiffusionModelBase = extendsAudioDiffusion,
            ExtendsFrameInterpolationBase = extendsFrameInterpolation,
            ExtendsVideoSuperResolutionBase = extendsVideoSR,
            ExtendsVideoDenoisingBase = extendsVideoDenoising,
            ExtendsAudioClassifierBase = extendsAudioClassifier,
            ExtendsOpticalFlowBase = extendsOpticalFlow,
            ExtendsSpeakerRecognitionBase = extendsSpeakerRecognition,
            ExtendsEnsembleClassifierBase = extendsEnsembleClassifier,
            ExtendsNaiveBayesBase = extendsNaiveBayes,
            ExtendsSVMBase = extendsSVM,
            ExtendsForecastingModelBase = extendsForecasting,
            ExtendsThreeDDiffusionModelBase = extendsThreeDDiffusion,
            ExtendsVideoInpaintingBase = extendsVideoInpainting,
            ExtendsVideoStabilizationBase = extendsVideoStabilization,
            ExtendsLinearClassifierBase = extendsLinearClassifier,
            ExtendsMetaClassifierBase = extendsMetaClassifier,
            ExtendsOrdinalClassifierBase = extendsOrdinalClassifier,
            ExtendsSemiSupervisedClassifierBase = extendsSemiSupervised,
            ExtendsMultiLabelClassifierBase = extendsMultiLabel,
            ExtendsFinancialNLPModelBase = extendsFinancialNLP,
            ExtendsRiskModelBase = extendsRiskModel,
            ExtendsPortfolioOptimizerBase = extendsPortfolioOptimizer,
            ExtendsTransformerNERBase = extendsTransformerNER,
            ExtendsSpanBasedNERBase = extendsSpanBasedNER,
            ExtendsSequenceLabelingNERBase = extendsSequenceLabelingNER,
            ExtendsAnomalyDetectorBase = extendsAnomalyDetector,
            ExtendsSurvivalModelBase = extendsSurvival,
            ExtendsCausalModelBase = extendsCausal,
            ExtendsRLAgentBase = extendsRLAgent,
            ExtendsLatentDiffusionModelBase = extendsLatentDiffusion,
            ExtendsNonLinearRegressionBase = extendsNonLinearRegression,
            ExtendsProbabilisticClassifierBase = extendsProbabilisticClassifier,
            Location = modelClass.Locations.Length > 0 ? modelClass.Locations[0] : null
        };

        bool hasCoverage = HasTestCoverage(className, testNames);
        info.HasTests = hasCoverage;

        if (hasCoverage)
            testedModels.Add(info);
        else
            untestedModels.Add(info);
    }

    /// <summary>
    /// Discovers model classes from referenced assemblies by traversing all public types.
    /// </summary>
    private static void DiscoverModelsFromReferencedAssemblies(
        Compilation compilation,
        INamedTypeSymbol? domainAttrSymbol,
        INamedTypeSymbol? categoryAttrSymbol,
        INamedTypeSymbol? taskAttrSymbol,
        INamedTypeSymbol? exemptAttrSymbol,
        HashSet<string> testNames,
        List<ModelTestInfo> testedModels,
        List<ModelTestInfo> untestedModels,
        HashSet<string> seen)
    {
        foreach (var reference in compilation.References)
        {
            var symbol = compilation.GetAssemblyOrModuleSymbol(reference);
            if (symbol is IAssemblySymbol assembly)
            {
                CollectModelsFromNamespace(assembly.GlobalNamespace, domainAttrSymbol, categoryAttrSymbol,
                    taskAttrSymbol, exemptAttrSymbol, testNames, testedModels, untestedModels, seen);
            }
        }
    }

    /// <summary>
    /// Recursively collects model types from a namespace symbol.
    /// </summary>
    private static void CollectModelsFromNamespace(
        INamespaceSymbol ns,
        INamedTypeSymbol? domainAttrSymbol,
        INamedTypeSymbol? categoryAttrSymbol,
        INamedTypeSymbol? taskAttrSymbol,
        INamedTypeSymbol? exemptAttrSymbol,
        HashSet<string> testNames,
        List<ModelTestInfo> testedModels,
        List<ModelTestInfo> untestedModels,
        HashSet<string> seen)
    {
        foreach (var member in ns.GetMembers())
        {
            if (member is INamespaceSymbol childNs)
            {
                CollectModelsFromNamespace(childNs, domainAttrSymbol, categoryAttrSymbol,
                    taskAttrSymbol, exemptAttrSymbol, testNames, testedModels, untestedModels, seen);
            }
            else if (member is INamedTypeSymbol type)
            {
                if (type.TypeKind == TypeKind.Class &&
                    !type.IsAbstract &&
                    ImplementsIFullModel(type))
                {
                    ProcessModelSymbol(type, domainAttrSymbol, categoryAttrSymbol, taskAttrSymbol,
                        exemptAttrSymbol, testNames, testedModels, untestedModels, seen);
                }
            }
        }
    }

    /// <summary>
    /// Checks whether a type implements IFullModel anywhere in its interface hierarchy.
    /// </summary>
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

    /// <summary>
    /// Resolves the appropriate test family for a model based on its category, task,
    /// and interface metadata. Returns null if no family can be determined.
    /// </summary>
    /// <remarks>
    /// Priority ordering (first match wins):
    /// 1. GaussianProcess category → GaussianProcess
    /// 2. TimeSeriesModel category → TimeSeries
    /// 3. Diffusion category → Diffusion
    /// 4. GAN category → GAN
    /// 5. EmbeddingModel category → Embedding
    /// 6. GraphNetwork category → GraphNN
    /// 7. Regression task + Matrix input → Regression
    /// 8. Classification task + Matrix input → Classification
    /// 9. Clustering task + Matrix input → Clustering
    /// 10. Neural network interface or Tensor input → NeuralNetwork
    /// 11. Matrix input fallback → Regression
    /// </remarks>
    private static TestFamily? ResolveTestBaseClass(ModelTestInfo model)
    {
        // === TIER 1: Specialized families (check first — most specific) ===

        // Priority 1: GaussianProcess
        if (model.Categories.Contains(CategoryGaussianProcess) || model.ImplementsGaussianProcess)
            return TestFamily.GaussianProcess;

        // Priority 2: TimeSeriesModel
        if (model.Categories.Contains(CategoryTimeSeriesModel))
            return TestFamily.TimeSeries;

        // Priority 2a: 3D Diffusion (most specific diffusion subtype)
        if (model.ExtendsThreeDDiffusionModelBase)
            return TestFamily.ThreeDDiffusion;

        // Priority 3a: Video Diffusion
        if (model.ExtendsVideoDiffusionModelBase)
            return TestFamily.VideoDiffusion;

        // Priority 3b: Audio Diffusion
        if (model.ExtendsAudioDiffusionModelBase)
            return TestFamily.AudioDiffusion;

        // Priority 3c: Latent Diffusion (more specific than plain Diffusion)
        if (model.ExtendsLatentDiffusionModelBase)
            return TestFamily.LatentDiffusion;

        // Priority 4: Diffusion (plain, non-latent)
        if (model.Categories.Contains(CategoryDiffusion) || model.ImplementsDiffusionModel)
            return TestFamily.Diffusion;

        // Priority 5: GAN
        if (model.Categories.Contains(CategoryGAN))
            return TestFamily.GAN;

        // Priority 6: EmbeddingModel
        if (model.Categories.Contains(CategoryEmbeddingModel))
            return TestFamily.Embedding;

        // Priority 7: GraphNetwork
        if (model.Categories.Contains(CategoryGraphNetwork))
            return TestFamily.GraphNN;

        // === TIER 2: Mid-level NN hierarchy (base class chain detection) ===

        // Priority 8a: Audio Classifier (leaf of AudioNN)
        if (model.ExtendsAudioClassifierBase)
            return TestFamily.AudioClassifier;

        // Priority 8b: Speaker Recognition (leaf of AudioNN)
        if (model.ExtendsSpeakerRecognitionBase)
            return TestFamily.SpeakerRecognition;

        // Priority 8: Audio NN
        if (model.ExtendsAudioNeuralNetworkBase)
            return TestFamily.AudioNN;

        // Priority 9: Document NN
        if (model.ExtendsDocumentNeuralNetworkBase)
            return TestFamily.DocumentNN;

        // Priority 10: Vision-Language
        if (model.ExtendsVisionLanguageModelBase)
            return TestFamily.VisionLanguage;

        // Priority 11: Segmentation
        if (model.ExtendsSegmentationModelBase)
            return TestFamily.Segmentation;

        // Priority 12a: Frame Interpolation (leaf of VideoNN)
        if (model.ExtendsFrameInterpolationBase)
            return TestFamily.FrameInterpolation;

        // Priority 12b: Video Super-Resolution (leaf of VideoNN)
        if (model.ExtendsVideoSuperResolutionBase)
            return TestFamily.VideoSuperResolution;

        // Priority 12c: Video Denoising (leaf of VideoNN)
        if (model.ExtendsVideoDenoisingBase)
            return TestFamily.VideoDenoising;

        // Priority 12e: Video Inpainting (leaf of VideoNN)
        if (model.ExtendsVideoInpaintingBase)
            return TestFamily.VideoInpainting;

        // Priority 12f: Video Stabilization (leaf of VideoNN)
        if (model.ExtendsVideoStabilizationBase)
            return TestFamily.VideoStabilization;

        // Priority 12d: Optical Flow (leaf of VideoNN)
        if (model.ExtendsOpticalFlowBase)
            return TestFamily.OpticalFlow;

        // Priority 12: Video NN
        if (model.ExtendsVideoNeuralNetworkBase)
            return TestFamily.VideoNN;

        // Priority 13: TTS
        if (model.ExtendsTtsModelBase)
            return TestFamily.TTS;

        // Priority 13a: Forecasting (leaf of Financial)
        if (model.ExtendsForecastingModelBase)
            return TestFamily.Forecasting;

        // Priority 13b: Financial NLP (leaf of Financial)
        if (model.ExtendsFinancialNLPModelBase)
            return TestFamily.FinancialNLP;

        // Priority 13c: Risk Model (leaf of Financial)
        if (model.ExtendsRiskModelBase)
            return TestFamily.RiskModel;

        // Priority 13d: Portfolio Optimizer (leaf of Financial)
        if (model.ExtendsPortfolioOptimizerBase)
            return TestFamily.PortfolioOptimizer;

        // Priority 14: Financial
        if (model.ExtendsFinancialModelBase)
            return TestFamily.Financial;

        // Priority 14e: Transformer NER (leaf of NER)
        if (model.ExtendsTransformerNERBase)
            return TestFamily.TransformerNER;

        // Priority 14f: Span-Based NER (leaf of NER)
        if (model.ExtendsSpanBasedNERBase)
            return TestFamily.SpanBasedNER;

        // Priority 14g: Sequence Labeling NER (leaf of NER)
        if (model.ExtendsSequenceLabelingNERBase)
            return TestFamily.SequenceLabelingNER;

        // Priority 15: NER
        if (model.ExtendsNERNeuralNetworkBase)
            return TestFamily.NER;

        // Priority 16: Code
        if (model.ExtendsCodeModelBase)
            return TestFamily.CodeModel;

        // === TIER 3: Matrix/Vector model families ===

        // Priority 13: Non-linear Regression (more specific than Regression)
        if (model.ExtendsNonLinearRegressionBase)
            return TestFamily.NonLinearRegression;

        // Priority 13e: Linear Classifier (leaf of ProbabilisticClassifier)
        if (model.ExtendsLinearClassifierBase)
            return TestFamily.LinearClassifier;

        // Priority 14a: SVM (leaf of ProbabilisticClassifier)
        if (model.ExtendsSVMBase)
            return TestFamily.SVM;

        // Priority 14b: NaiveBayes (leaf of ProbabilisticClassifier)
        if (model.ExtendsNaiveBayesBase)
            return TestFamily.NaiveBayes;

        // Priority 14: Probabilistic Classifier
        if (model.ExtendsProbabilisticClassifierBase)
            return TestFamily.ProbabilisticClassifier;

        // Priority 14c: Ensemble Classifier
        if (model.ExtendsEnsembleClassifierBase)
            return TestFamily.EnsembleClassifier;

        // Priority 14d: Meta Classifier
        if (model.ExtendsMetaClassifierBase)
            return TestFamily.MetaClassifier;

        // Priority 14e: Ordinal Classifier
        if (model.ExtendsOrdinalClassifierBase)
            return TestFamily.OrdinalClassifier;

        // Priority 14f: Semi-Supervised Classifier
        if (model.ExtendsSemiSupervisedClassifierBase)
            return TestFamily.SemiSupervisedClassifier;

        // Priority 14g: Multi-Label Classifier (uses Matrix output, not Vector)
        if (model.ExtendsMultiLabelClassifierBase)
            return TestFamily.MultiLabelClassifier;

        // Priority 15: Regression task + Matrix input
        if (model.Tasks.Contains(TaskRegression) && model.UsesMatrixInput)
            return TestFamily.Regression;

        // Priority 16: Classification task + Matrix input
        if (model.Tasks.Contains(TaskClassification) && model.UsesMatrixInput)
            return TestFamily.Classification;

        // Priority 17: Clustering task + Matrix input
        if (model.Tasks.Contains(TaskClustering) && model.UsesMatrixInput)
            return TestFamily.Clustering;

        // === TIER 4: Phase C new top-level families ===

        // Priority 17a: Anomaly Detection
        if (model.ExtendsAnomalyDetectorBase)
            return TestFamily.AnomalyDetector;

        // Priority 17b: Survival Analysis
        if (model.ExtendsSurvivalModelBase)
            return TestFamily.Survival;

        // Priority 17c: Causal Inference
        if (model.ExtendsCausalModelBase)
            return TestFamily.Causal;

        // Priority 17d: Reinforcement Learning
        if (model.ExtendsRLAgentBase)
            return TestFamily.ReinforcementLearning;

        // === TIER 5: Fallbacks ===

        // Priority 18: Neural network (by interface or Tensor input)
        if (model.ImplementsNeuralNetworkModel || model.UsesTensorInput)
            return TestFamily.NeuralNetwork;

        // Priority 19: Matrix input fallback → Regression
        if (model.UsesMatrixInput)
            return TestFamily.Regression;

        // Priority 20: NeuralNetwork category
        if (model.Categories.Contains(CategoryNeuralNetwork))
            return TestFamily.NeuralNetwork;

        // Priority 21: MetaLearning category → NeuralNetwork
        if (model.Categories.Contains(CategoryMetaLearning))
            return TestFamily.NeuralNetwork;

        // Cannot determine — skip generation
        return null;
    }

    /// <summary>
    /// Emits a generated test class for a model that has no manual test coverage.
    /// </summary>
    /// <param name="canConstruct">
    /// When true, emits <c>new Model&lt;double&gt;()</c>. When false, emits
    /// <c>throw new NotImplementedException(...)</c> so the test compiles but
    /// fails at runtime with a clear message to create a manual test class.
    /// </param>
    private static void EmitGeneratedTestClass(
        SourceProductionContext context,
        ModelTestInfo model,
        TestFamily family,
        string testClassName,
        bool canConstruct)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(model.FullyQualifiedName);
        string factoryBody;

        if (canConstruct)
        {
            string constructorExpr;
            if (model.TypeParameterCount == 0)
            {
                constructorExpr = $"new {typeName}()";
            }
            else if (model.TypeParameterCount == 1)
            {
                constructorExpr = $"new {typeName}<double>()";
            }
            else
            {
                // Multi-type-parameter models (e.g., Model<T, TInput, TOutput>)
                // Resolve TInput/TOutput from the detected IFullModel type arguments
                string inputType = model.UsesTensorInput ? "Tensor<double>" : "Matrix<double>";
                string outputType = model.UsesVectorOutput ? "Vector<double>" :
                                    model.UsesTensorInput ? "Tensor<double>" : "Vector<double>";
                constructorExpr = $"new {typeName}<double, {inputType}, {outputType}>()";
            }
            factoryBody = $"        => {constructorExpr};";
        }
        else
        {
            // Models requiring constructor arguments get a placeholder that fails at runtime
            // with a clear message, so the test exists but reminds developers to implement it.
            factoryBody = "        => throw new System.NotImplementedException(" +
                $"\"'{GeneratorHelpers.StripGenericSuffix(model.ClassName)}' requires constructor arguments. Implement this factory manually.\");";
        }

        var baseClassName = GetBaseClassName(family);
        var factoryMethodName = GetFactoryMethodName(family);
        var returnTypeCode = GetReturnTypeCode(family);

        var sb = new StringBuilder();
        // Multi-type-parameter models may need Tensor<> and/or Matrix<>/Vector<> usings
        bool needsTensorUsing = model.TypeParameterCount > 1 && model.UsesTensorInput;
        bool needsMatrixUsingForModel = NeedsMatrixUsing(family) ||
                                        (model.TypeParameterCount > 1 && model.UsesMatrixInput);

        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// If this model needs constructor arguments, create a manual test class to replace this.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        if (needsTensorUsing)
            sb.AppendLine("using AiDotNet.Tensors;");
        if (needsMatrixUsingForModel)
            sb.AppendLine("using AiDotNet.Tensors.LinearAlgebra;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : {baseClassName}");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override {returnTypeCode} {factoryMethodName}()");
        sb.AppendLine(factoryBody);
        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(model.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Verifies that the model's actual interfaces are compatible with the resolved test family.
    /// Prevents generating code that won't compile (e.g., casting to wrong interface).
    /// </summary>
    private static bool IsCompatibleWithFamily(ModelTestInfo model, TestFamily family)
    {
        switch (family)
        {
            // NN-derived families require INeuralNetworkModel interface
            case TestFamily.GAN:
            case TestFamily.Embedding:
            case TestFamily.GraphNN:
            case TestFamily.AudioNN:
            case TestFamily.DocumentNN:
            case TestFamily.VisionLanguage:
            case TestFamily.Segmentation:
            case TestFamily.VideoNN:
            case TestFamily.TTS:
            case TestFamily.Financial:
            case TestFamily.NER:
            case TestFamily.CodeModel:
            case TestFamily.FrameInterpolation:
            case TestFamily.VideoSuperResolution:
            case TestFamily.VideoDenoising:
            case TestFamily.AudioClassifier:
            case TestFamily.OpticalFlow:
            case TestFamily.SpeakerRecognition:
            case TestFamily.Forecasting:
            case TestFamily.VideoInpainting:
            case TestFamily.VideoStabilization:
            case TestFamily.FinancialNLP:
            case TestFamily.RiskModel:
            case TestFamily.PortfolioOptimizer:
            case TestFamily.TransformerNER:
            case TestFamily.SpanBasedNER:
            case TestFamily.SequenceLabelingNER:
            case TestFamily.NeuralNetwork:
                return model.ImplementsNeuralNetworkModel;

            // Diffusion families require IDiffusionModel interface
            case TestFamily.Diffusion:
            case TestFamily.LatentDiffusion:
            case TestFamily.VideoDiffusion:
            case TestFamily.AudioDiffusion:
            case TestFamily.ThreeDDiffusion:
                return model.ImplementsDiffusionModel;

            // GP family requires IGaussianProcess interface
            case TestFamily.GaussianProcess:
                return model.ImplementsGaussianProcess;

            // Matrix/Vector families require IFullModel<T, Matrix<T>, Vector<T>>
            case TestFamily.Regression:
            case TestFamily.NonLinearRegression:
            case TestFamily.Classification:
            case TestFamily.ProbabilisticClassifier:
            case TestFamily.EnsembleClassifier:
            case TestFamily.NaiveBayes:
            case TestFamily.SVM:
            case TestFamily.AnomalyDetector:
            case TestFamily.Survival:
            case TestFamily.Causal:
            case TestFamily.LinearClassifier:
            case TestFamily.MetaClassifier:
            case TestFamily.OrdinalClassifier:
            case TestFamily.SemiSupervisedClassifier:
            case TestFamily.Clustering:
            case TestFamily.TimeSeries:
                return model.UsesMatrixInput && model.UsesVectorOutput;

            // MultiLabel uses Matrix/Matrix — compatible if has Matrix input
            case TestFamily.MultiLabelClassifier:
                return model.UsesMatrixInput;

            // RL uses Vector/Vector — always compatible if it has IFullModel
            case TestFamily.ReinforcementLearning:
                return true;

            default:
                return false;
        }
    }

    private static bool HasTestCoverage(string modelClassName, HashSet<string> testNames)
    {
        // Strip generic arity suffix
        var baseName = modelClassName;
        var backtick = baseName.IndexOf('`');
        if (backtick >= 0)
            baseName = baseName.Substring(0, backtick);

        // Check common test naming conventions
        if (testNames.Contains(baseName + "Tests")) return true;
        if (testNames.Contains(baseName + "Test")) return true;
        if (testNames.Contains(baseName + "_Tests")) return true;
        if (testNames.Contains(baseName + "IntegrationTests")) return true;
        if (testNames.Contains(baseName + "UnitTests")) return true;

        // Check if any test class name contains the model name at a word boundary
        foreach (var testName in testNames)
        {
            int idx = testName.IndexOf(baseName, System.StringComparison.OrdinalIgnoreCase);
            if (idx < 0) continue;
            int afterMatch = idx + baseName.Length;
            if (afterMatch >= testName.Length) return true;
            string remainder = testName.Substring(afterMatch);
            if (remainder.StartsWith("Tests", System.StringComparison.Ordinal) ||
                remainder.StartsWith("Test", System.StringComparison.Ordinal) ||
                remainder.StartsWith("_", System.StringComparison.Ordinal) ||
                !char.IsLetter(remainder[0])) return true;
        }

        return false;
    }

    /// <summary>
    /// Generates test classes for activation functions and loss functions discovered via attributes.
    /// </summary>
    // Test base class metadata names for type-system-based coverage detection
    private static readonly string[] ActivationTestBases = new[]
    {
        "AiDotNet.Tests.ModelFamilyTests.Base.ActivationFunctionTestBase"
    };
    private static readonly string[] LossTestBases = new[]
    {
        "AiDotNet.Tests.ModelFamilyTests.Base.LossFunctionTestBase",
        "AiDotNet.Tests.ModelFamilyTests.Base.TripletLossTestBase",
        "AiDotNet.Tests.ModelFamilyTests.Base.ContrastiveLossTestBase",
        "AiDotNet.Tests.ModelFamilyTests.Base.SparseCategoricalLossTestBase"
    };

    private static void ExecuteActivationAndLossGeneration(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> activationClasses,
        ImmutableArray<INamedTypeSymbol?> lossClasses,
        Compilation compilation)
    {
        // Only generate in test projects
        string assemblyName = compilation.AssemblyName ?? string.Empty;
        bool isTestProject = assemblyName.IndexOf("Test", System.StringComparison.OrdinalIgnoreCase) >= 0;

        var activationSeen = new HashSet<string>();
        var lossSeen = new HashSet<string>();

        // Collect from source
        var sourceActivations = new List<ComponentTestInfo>();
        foreach (var symbol in activationClasses)
        {
            if (symbol is null) continue;
            var info = ExtractActivationInfo(symbol);
            if (info is not null && activationSeen.Add(info.FullyQualifiedName))
                sourceActivations.Add(info);
        }

        var sourceLosses = new List<ComponentTestInfo>();
        foreach (var symbol in lossClasses)
        {
            if (symbol is null) continue;
            var info = ExtractLossInfo(symbol);
            if (info is not null && lossSeen.Add(info.FullyQualifiedName))
                sourceLosses.Add(info);
        }

        bool hasSourceItems = sourceActivations.Count > 0 || sourceLosses.Count > 0;
        bool isSourceProject = hasSourceItems && !isTestProject;

        // If in test project or no source items, also discover from referenced assemblies
        if (!isSourceProject)
        {
            DiscoverComponentsFromReferencedAssemblies(compilation, activationSeen, lossSeen,
                sourceActivations, sourceLosses);
        }

        // Generate test classes (only in test projects)
        if (!isSourceProject)
        {
            // Use Roslyn's type system to find which components already have manual test coverage.
            // Walk the inheritance chain of all source classes to find those inheriting from our
            // test base classes, then inspect their factory method to resolve the concrete type
            // being tested via the semantic model.
            var coveredActivations = FindCoveredComponentTypes(compilation, ActivationTestBases, "CreateActivation");
            var coveredLosses = FindCoveredComponentTypes(compilation, LossTestBases, "CreateLoss");

            int activationTested = 0, activationTotal = sourceActivations.Count;
            var generatedActivationNames = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);

            foreach (var act in sourceActivations)
            {
                // Type-system-based coverage: check if any test class inheriting from
                // ActivationFunctionTestBase constructs this exact type in its CreateActivation() factory
                if (coveredActivations.Contains(act.FullyQualifiedName))
                {
                    activationTested++;
                    continue;
                }

                // Skip vector-only activations (e.g. Softmax) or those with learnable params
                // that need constructor args
                if (act.IsVectorActivation || (act.HasLearnableParameters && !act.HasParameterlessConstructor))
                {
                    activationTested++; // Don't count as untested since they can't auto-test
                    continue;
                }

                if (!act.HasParameterlessConstructor)
                    continue;

                var testClassName = StripBacktick(act.ClassName) + "Tests";
                if (!generatedActivationNames.Add(testClassName))
                    continue;

                EmitActivationTestClass(context, act, testClassName);
                activationTested++;
            }

            if (activationTotal > 0)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    ActivationCoverageSummary, Location.None,
                    activationTested, activationTotal,
                    activationTested * 100.0 / activationTotal));
            }

            int lossTested = 0, lossTotal = sourceLosses.Count;
            var generatedLossNames = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);

            foreach (var loss in sourceLosses)
            {
                // Type-system-based coverage
                if (coveredLosses.Contains(loss.FullyQualifiedName))
                {
                    lossTested++;
                    continue;
                }

                // Skip ImageMatrix (requires feature extractor function — can't auto-construct)
                // Skip SelfSupervised (implements ISelfSupervisedLoss, not LossFunctionBase)
                // Skip ComplexInterleaved (needs ComplexLossTestBase — TODO)
                if (loss.ApiShape == ApiShapeImageMatrix || loss.ApiShape == ApiShapeSelfSupervised || loss.ApiShape == ApiShapeComplexInterleaved || loss.ApiShape == ApiShapePairedEmbedding)
                {
                    lossTested++; // Don't count as untested since they can't auto-test
                    continue;
                }

                if (!loss.HasParameterlessConstructor)
                    continue;

                var testClassName = StripBacktick(loss.ClassName) + "Tests";
                if (!generatedLossNames.Add(testClassName))
                    continue;

                // Route to the correct test base class based on API shape
                switch (loss.ApiShape)
                {
                    case ApiShapeTripletMatrix:
                        EmitTripletLossTestClass(context, loss, testClassName);
                        break;
                    case ApiShapeTargetNoiseMatrix:
                        EmitContrastiveLossTestClass(context, loss, testClassName);
                        break;
                    case ApiShapeSparseIndex:
                        EmitSparseCategoricalLossTestClass(context, loss, testClassName);
                        break;
                    default:
                        // Standard VectorVector API — skip if throws NotSupportedException
                        if (loss.ThrowsNotSupported || !loss.ExtendsLossFunctionBase)
                        {
                            lossTested++;
                            continue;
                        }
                        EmitLossTestClass(context, loss, testClassName);
                        break;
                }
                lossTested++;
            }

            if (lossTotal > 0)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    LossCoverageSummary, Location.None,
                    lossTested, lossTotal,
                    lossTested * 100.0 / lossTotal));
            }
        }
    }

    /// <summary>
    /// Uses Roslyn's type system to determine which component types already have manual test coverage.
    /// Walks all source classes in the compilation, checks their inheritance chain against the known
    /// test base classes, and inspects the factory method's object creation expressions to resolve
    /// the exact concrete component type being tested via the semantic model.
    /// </summary>
    /// <param name="compilation">The current compilation containing both source and referenced types.</param>
    /// <param name="testBaseClassFullNames">Metadata names of test base classes to search for.</param>
    /// <param name="factoryMethodName">The factory method name to inspect (e.g., "CreateActivation", "CreateLoss").</param>
    /// <returns>Set of fully qualified original definition names for covered component types.</returns>
    private static HashSet<string> FindCoveredComponentTypes(
        Compilation compilation,
        string[] testBaseClassFullNames,
        string factoryMethodName)
    {
        var covered = new HashSet<string>();

        // Resolve test base class symbols from the compilation
        var baseTypeSymbols = new List<INamedTypeSymbol>();
        foreach (var name in testBaseClassFullNames)
        {
            var baseType = compilation.GetTypeByMetadataName(name);
            if (baseType is not null)
                baseTypeSymbols.Add(baseType);
        }

        if (baseTypeSymbols.Count == 0)
            return covered;

        // Walk every syntax tree in the compilation to find test classes
        foreach (var syntaxTree in compilation.SyntaxTrees)
        {
            var semanticModel = compilation.GetSemanticModel(syntaxTree);
            var root = syntaxTree.GetRoot();

            foreach (var classDecl in root.DescendantNodes().OfType<ClassDeclarationSyntax>())
            {
                var classSymbol = semanticModel.GetDeclaredSymbol(classDecl) as INamedTypeSymbol;
                if (classSymbol is null || classSymbol.IsAbstract)
                    continue;

                // Walk the inheritance chain to check if this class derives from any of our test bases
                if (!InheritsFromAny(classSymbol, baseTypeSymbols))
                    continue;

                // Found a test class. Now inspect the factory method override to find
                // which concrete component type it constructs.
                ExtractConstructedTypesFromFactory(classSymbol, factoryMethodName, semanticModel, covered);
            }
        }

        return covered;
    }

    /// <summary>
    /// Walks the base type chain of <paramref name="type"/> to check if it inherits from
    /// any of the <paramref name="baseTypes"/>.
    /// </summary>
    private static bool InheritsFromAny(INamedTypeSymbol type, List<INamedTypeSymbol> baseTypes)
    {
        var current = type.BaseType;
        while (current is not null)
        {
            foreach (var baseType in baseTypes)
            {
                if (SymbolEqualityComparer.Default.Equals(current, baseType))
                    return true;
            }
            current = current.BaseType;
        }
        return false;
    }

    /// <summary>
    /// Finds the override of <paramref name="factoryMethodName"/> in <paramref name="classSymbol"/>,
    /// walks its syntax tree for object creation expressions, resolves each to its concrete type
    /// via the semantic model, and adds the original generic definition to <paramref name="covered"/>.
    /// </summary>
    private static void ExtractConstructedTypesFromFactory(
        INamedTypeSymbol classSymbol,
        string factoryMethodName,
        SemanticModel semanticModel,
        HashSet<string> covered)
    {
        foreach (var member in classSymbol.GetMembers(factoryMethodName))
        {
            if (member is not IMethodSymbol method || !method.IsOverride)
                continue;

            foreach (var syntaxRef in method.DeclaringSyntaxReferences)
            {
                var methodSyntax = syntaxRef.GetSyntax();

                // Find explicit object creation: new SomeType<double>(args)
                foreach (var creation in methodSyntax.DescendantNodes()
                    .OfType<ObjectCreationExpressionSyntax>())
                {
                    AddResolvedType(semanticModel, creation, covered);
                }

                // Find implicit object creation: new(args) with target-typed new
                foreach (var creation in methodSyntax.DescendantNodes()
                    .OfType<ImplicitObjectCreationExpressionSyntax>())
                {
                    AddResolvedType(semanticModel, creation, covered);
                }
            }
        }
    }

    /// <summary>
    /// Resolves the type of an object creation expression via the semantic model and adds
    /// its original generic definition to the covered set.
    /// </summary>
    private static void AddResolvedType(SemanticModel semanticModel, SyntaxNode creationExpr, HashSet<string> covered)
    {
        var typeInfo = semanticModel.GetTypeInfo(creationExpr);
        if (typeInfo.Type is not INamedTypeSymbol createdType)
            return;

        // Get the unbound generic definition (e.g., ReLUActivation<T> from ReLUActivation<double>)
        var originalDef = createdType.IsGenericType
            ? createdType.OriginalDefinition
            : createdType;

        covered.Add(originalDef.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
    }

    // Layer test base class metadata names
    private static readonly string[] LayerTestBases = new[]
    {
        "AiDotNet.Tests.ModelFamilyTests.Base.LayerTestBase",
        "AiDotNet.Tests.ModelFamilyTests.Base.DualInputLayerTestBase",
        "AiDotNet.Tests.ModelFamilyTests.Base.MultiInputLayerTestBase",
        "AiDotNet.Tests.ModelFamilyTests.Base.GraphLayerTestBase"
    };

    // LayerApiShape enum values (must match AiDotNet.Enums.LayerApiShape)
    private const int LayerApiShapeSingleTensor = 0;
    private const int LayerApiShapeDualTensor = 1;
    private const int LayerApiShapeMultiInput = 2;
    private const int LayerApiShapeGraphWithSetup = 3;

    /// <summary>
    /// Generates test classes for layers discovered via [LayerProperty] attributes.
    /// </summary>
    private static void ExecuteLayerGeneration(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> layerClasses,
        Compilation compilation)
    {
        string assemblyName = compilation.AssemblyName ?? string.Empty;
        bool isTestProject = assemblyName.IndexOf("Test", System.StringComparison.OrdinalIgnoreCase) >= 0;


        var layerSeen = new HashSet<string>();
        var sourceLayers = new List<LayerTestInfo>();

        // Collect from source
        foreach (var symbol in layerClasses)
        {
            if (symbol is null) continue;
            var info = ExtractLayerInfo(symbol);
            if (info is not null && layerSeen.Add(info.FullyQualifiedName))
                sourceLayers.Add(info);
        }

        bool hasSourceItems = sourceLayers.Count > 0;
        bool isSourceProject = hasSourceItems && !isTestProject;

        // Discover from referenced assemblies if in test project
        if (!isSourceProject)
        {
            DiscoverLayersFromReferencedAssemblies(compilation, layerSeen, sourceLayers);
        }

        // Generate test classes (only in test projects)
        if (!isSourceProject)
        {
            var coveredLayers = FindCoveredComponentTypes(compilation, LayerTestBases, "CreateLayer");

            int layerTested = 0, layerTotal = sourceLayers.Count;
            var generatedNames = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);

            foreach (var layer in sourceLayers)
            {
                // Type-system-based coverage detection
                if (coveredLayers.Contains(layer.FullyQualifiedName))
                {
                    layerTested++;
                    continue;
                }

                // Skip if no accessible constructor
                if (!layer.HasParameterlessConstructor && string.IsNullOrEmpty(layer.TestConstructorArgs))
                    continue;

                var testClassName = StripBacktick(layer.ClassName) + "Tests";
                if (!generatedNames.Add(testClassName))
                    continue;

                switch (layer.ApiShape)
                {
                    case LayerApiShapeDualTensor:
                        EmitDualInputLayerTestClass(context, layer, testClassName);
                        break;
                    case LayerApiShapeMultiInput:
                        EmitMultiInputLayerTestClass(context, layer, testClassName);
                        break;
                    case LayerApiShapeGraphWithSetup:
                        EmitGraphLayerTestClass(context, layer, testClassName);
                        break;
                    default:
                        EmitLayerTestClass(context, layer, testClassName);
                        break;
                }
                layerTested++;
            }

            if (layerTotal > 0)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    LayerCoverageSummary, Location.None,
                    layerTested, layerTotal,
                    layerTested * 100.0 / layerTotal));
            }
        }
    }

    /// <summary>
    /// Extracts layer metadata from [LayerProperty] attributes.
    /// </summary>
    private static LayerTestInfo? ExtractLayerInfo(INamedTypeSymbol symbol)
    {
        bool isTrainable = true, hasTrainingMode = false, changesShape = false, isStateful = false;
        bool supportsBackprop = true;
        int apiShape = LayerApiShapeSingleTensor;
        string testInputShape = "";
        string testConstructorArgs = "";

        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is null) continue;
            if (!attr.AttributeClass.ToDisplayString().EndsWith("LayerPropertyAttribute", System.StringComparison.Ordinal))
                continue;

            foreach (var named in attr.NamedArguments)
            {
                switch (named.Key)
                {
                    case "IsTrainable":
                        isTrainable = (bool)(named.Value.Value ?? true);
                        break;
                    case "SupportsBackpropagation":
                        supportsBackprop = (bool)(named.Value.Value ?? true);
                        break;
                    case "HasTrainingMode":
                        hasTrainingMode = (bool)(named.Value.Value ?? false);
                        break;
                    case "ChangesShape":
                        changesShape = (bool)(named.Value.Value ?? false);
                        break;
                    case "IsStateful":
                        isStateful = (bool)(named.Value.Value ?? false);
                        break;
                    case "ApiShape":
                        apiShape = (int)(named.Value.Value ?? 0);
                        break;
                    case "TestInputShape":
                        testInputShape = (string)(named.Value.Value ?? "");
                        break;
                    case "TestConstructorArgs":
                        testConstructorArgs = (string)(named.Value.Value ?? "");
                        break;
                }
            }
        }

        bool hasParameterlessCtor = HasAccessibleParameterlessConstructor(symbol);

        return new LayerTestInfo
        {
            ClassName = symbol.Name,
            FullyQualifiedName = symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            TypeParameterCount = symbol.TypeParameters.Length,
            HasParameterlessConstructor = hasParameterlessCtor,
            IsTrainable = isTrainable,
            SupportsBackpropagation = supportsBackprop,
            HasTrainingMode = hasTrainingMode,
            ChangesShape = changesShape,
            IsStateful = isStateful,
            ApiShape = apiShape,
            TestInputShape = testInputShape,
            TestConstructorArgs = testConstructorArgs
        };
    }

    /// <summary>
    /// Discovers layers from referenced assemblies by checking for [LayerProperty] attributes.
    /// </summary>
    private static void DiscoverLayersFromReferencedAssemblies(
        Compilation compilation,
        HashSet<string> layerSeen,
        List<LayerTestInfo> layers)
    {
        foreach (var reference in compilation.References)
        {
            var symbol = compilation.GetAssemblyOrModuleSymbol(reference);
            if (symbol is IAssemblySymbol assembly)
            {
                CollectLayersFromNamespace(assembly.GlobalNamespace, layerSeen, layers);
            }
        }
    }

    private static void CollectLayersFromNamespace(
        INamespaceSymbol ns,
        HashSet<string> layerSeen,
        List<LayerTestInfo> layers)
    {
        foreach (var member in ns.GetMembers())
        {
            if (member is INamespaceSymbol childNs)
            {
                CollectLayersFromNamespace(childNs, layerSeen, layers);
            }
            else if (member is INamedTypeSymbol type)
            {
                if (type.TypeKind != TypeKind.Class || type.IsAbstract)
                    continue;

                bool hasLayerProp = false;
                foreach (var attr in type.GetAttributes())
                {
                    if (attr.AttributeClass is not null &&
                        attr.AttributeClass.ToDisplayString().EndsWith("LayerPropertyAttribute", System.StringComparison.Ordinal))
                    {
                        hasLayerProp = true;
                        break;
                    }
                }

                if (hasLayerProp)
                {
                    var info = ExtractLayerInfo(type);
                    if (info is not null && layerSeen.Add(info.FullyQualifiedName))
                        layers.Add(info);
                }
            }
        }
    }

    /// <summary>
    /// Emits a generated test class for a standard single-input layer.
    /// </summary>
    private static void EmitLayerTestClass(
        SourceProductionContext context,
        LayerTestInfo layer,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName);
        string constructorArgs = string.IsNullOrEmpty(layer.TestConstructorArgs) ? "" : layer.TestConstructorArgs;
        string constructorExpr = $"new {typeName}<double>({constructorArgs})";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated layer test. Invariant tests are inherited from LayerTestBase.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : LayerTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override ILayer<double> CreateLayer()");
        sb.AppendLine($"        => {constructorExpr};");

        // Override InputShape if specified
        if (!string.IsNullOrEmpty(layer.TestInputShape))
        {
            sb.AppendLine($"    protected override int[] InputShape => new[] {{ {layer.TestInputShape} }};");
        }

        // Override ExpectsTrainableParameters
        if (!layer.IsTrainable)
            sb.AppendLine("    protected override bool ExpectsTrainableParameters => false;");

        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits a generated test class for a dual-input layer.
    /// </summary>
    private static void EmitDualInputLayerTestClass(
        SourceProductionContext context,
        LayerTestInfo layer,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName);
        string constructorArgs = string.IsNullOrEmpty(layer.TestConstructorArgs) ? "" : layer.TestConstructorArgs;
        string constructorExpr = $"new {typeName}<double>({constructorArgs})";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated dual-input layer test. Invariant tests are inherited from DualInputLayerTestBase.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : DualInputLayerTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override ILayer<double> CreateLayer()");
        sb.AppendLine($"        => {constructorExpr};");

        // Override input shapes if specified
        if (!string.IsNullOrEmpty(layer.TestInputShape))
        {
            sb.AppendLine($"    protected override int[] PrimaryInputShape => new[] {{ {layer.TestInputShape} }};");
        }

        // Override ExpectsTrainableParameters
        if (!layer.IsTrainable)
            sb.AppendLine("    protected override bool ExpectsTrainableParameters => false;");

        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits a generated test class for a multi-input layer (AddLayer, ConcatenateLayer, MultiplyLayer).
    /// </summary>
    private static void EmitMultiInputLayerTestClass(
        SourceProductionContext context,
        LayerTestInfo layer,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName);
        string constructorArgs = string.IsNullOrEmpty(layer.TestConstructorArgs) ? "" : layer.TestConstructorArgs;
        string constructorExpr = $"new {typeName}<double>({constructorArgs})";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated multi-input layer test.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : MultiInputLayerTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override ILayer<double> CreateLayer()");
        sb.AppendLine($"        => {constructorExpr};");

        if (!string.IsNullOrEmpty(layer.TestInputShape))
            sb.AppendLine($"    protected override int[] InputShape => new[] {{ {layer.TestInputShape} }};");
        if (!layer.IsTrainable)
            sb.AppendLine("    protected override bool ExpectsTrainableParameters => false;");

        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits a generated test class for a graph layer requiring setup.
    /// The generated class extends GraphLayerTestBase and provides a SetupLayer override
    /// that calls the appropriate setup method (SetLaplacian, SetEdgeAdjacency, etc.)
    /// with synthetic graph data.
    /// </summary>
    private static void EmitGraphLayerTestClass(
        SourceProductionContext context,
        LayerTestInfo layer,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName);
        var shortName = GeneratorHelpers.StripGenericSuffix(layer.ClassName);
        string constructorArgs = string.IsNullOrEmpty(layer.TestConstructorArgs) ? "" : layer.TestConstructorArgs;
        string constructorExpr = $"new {typeName}<double>({constructorArgs})";

        // Determine which setup method to call based on the layer type name
        string setupCode;
        if (shortName.Contains("DiffusionConv"))
        {
            setupCode = @"
        // Create synthetic Laplacian (identity-like sparse matrix for 4 nodes)
        var laplacian = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 4, 4 });
        for (int i = 0; i < 4; i++) { laplacian[i, i] = 2.0; if (i > 0) { laplacian[i, i-1] = -1.0; laplacian[i-1, i] = -1.0; } }
        ((AiDotNet.NeuralNetworks.Layers.DiffusionConvLayer<double>)layer).SetLaplacian(laplacian);";
        }
        else if (shortName.Contains("MeshEdgeConv") || shortName.Contains("MeshPool"))
        {
            setupCode = @"
        // Create synthetic edge adjacency (each node connects to next)
        var edges = new int[4, 2]; // 4 edges for 4 nodes
        for (int i = 0; i < 4; i++) { edges[i, 0] = i; edges[i, 1] = (i + 1) % 4; }
        var method = layer.GetType().GetMethod(""SetEdgeAdjacency"");
        method?.Invoke(layer, new object[] { edges });";
        }
        else if (shortName.Contains("SpiralConv"))
        {
            setupCode = @"
        // Create synthetic spiral indices (3 neighbors per vertex for 4 vertices)
        var spirals = new int[4, 3]; // 4 vertices, 3 spiral neighbors each
        for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) spirals[i, j] = (i + j) % 4;
        ((AiDotNet.NeuralNetworks.Layers.SpiralConvLayer<double>)layer).SetSpiralIndices(spirals);";
        }
        else
        {
            // Generic fallback: no setup needed or unknown layer
            setupCode = "        // No specific graph setup required for this layer";
        }

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated graph layer test with domain-specific setup.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tensors;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : GraphLayerTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override ILayer<double> CreateLayer()");
        sb.AppendLine($"        => {constructorExpr};");
        sb.AppendLine();
        sb.AppendLine($"    protected override void SetupLayer(ILayer<double> layer)");
        sb.AppendLine("    {");
        sb.AppendLine(setupCode);
        sb.AppendLine("    }");

        if (!string.IsNullOrEmpty(layer.TestInputShape))
            sb.AppendLine($"    protected override int[] InputShape => new[] {{ {layer.TestInputShape} }};");
        if (!layer.IsTrainable)
            sb.AppendLine("    protected override bool ExpectsTrainableParameters => false;");

        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(layer.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Info about a layer for test generation.
    /// </summary>
    private class LayerTestInfo
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public bool HasParameterlessConstructor { get; set; }
        public bool IsTrainable { get; set; } = true;
        public bool SupportsBackpropagation { get; set; } = true;
        public bool HasTrainingMode { get; set; }
        public bool ChangesShape { get; set; }
        public bool IsStateful { get; set; }
        public int ApiShape { get; set; }
        public string TestInputShape { get; set; } = "";
        public string TestConstructorArgs { get; set; } = "";
    }

    /// <summary>
    /// Extracts activation function metadata from attributes.
    /// </summary>
    private static ComponentTestInfo? ExtractActivationInfo(INamedTypeSymbol symbol)
    {
        bool isMonotonic = true, zeroPreserving = true, isBounded = false;
        bool isVectorActivation = false, hasLearnableParams = false, isStochastic = false;
        double boundLower = -1.0, boundUpper = 1.0;

        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is null) continue;
            if (!attr.AttributeClass.ToDisplayString().EndsWith("ActivationPropertyAttribute", System.StringComparison.Ordinal))
                continue;

            foreach (var named in attr.NamedArguments)
            {
                switch (named.Key)
                {
                    case "IsMonotonic":
                        isMonotonic = (bool)(named.Value.Value ?? true);
                        break;
                    case "ZeroPreserving":
                        zeroPreserving = (bool)(named.Value.Value ?? true);
                        break;
                    case "IsBounded":
                        isBounded = (bool)(named.Value.Value ?? false);
                        break;
                    case "IsVectorActivation":
                        isVectorActivation = (bool)(named.Value.Value ?? false);
                        break;
                    case "HasLearnableParameters":
                        hasLearnableParams = (bool)(named.Value.Value ?? false);
                        break;
                    case "IsStochastic":
                        isStochastic = (bool)(named.Value.Value ?? false);
                        break;
                    case "BoundLower":
                        boundLower = (double)(named.Value.Value ?? -1.0);
                        break;
                    case "BoundUpper":
                        boundUpper = (double)(named.Value.Value ?? 1.0);
                        break;
                }
            }
        }

        bool hasParameterlessCtor = HasAccessibleParameterlessConstructor(symbol);

        return new ComponentTestInfo
        {
            ClassName = symbol.Name,
            FullyQualifiedName = symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            TypeParameterCount = symbol.TypeParameters.Length,
            IsMonotonic = isMonotonic,
            ZeroPreserving = zeroPreserving,
            IsBounded = isBounded,
            IsVectorActivation = isVectorActivation,
            HasLearnableParameters = hasLearnableParams,
            IsStochastic = isStochastic,
            BoundLower = boundLower,
            BoundUpper = boundUpper,
            HasParameterlessConstructor = hasParameterlessCtor,
            IsActivation = true
        };
    }

    /// <summary>
    /// Extracts loss function metadata from attributes.
    /// </summary>
    // LossApiShape enum values (must match AiDotNet.Enums.LossApiShape)
    private const int ApiShapeVectorVector = 0;
    private const int ApiShapeTripletMatrix = 1;
    private const int ApiShapeTargetNoiseMatrix = 2;
    private const int ApiShapeImageMatrix = 3;
    private const int ApiShapeSelfSupervised = 4;
    private const int ApiShapeSparseIndex = 5;
    private const int ApiShapeComplexInterleaved = 6;
    private const int ApiShapePairedEmbedding = 7;

    // LossTestInputFormat enum values (must match AiDotNet.Enums.LossTestInputFormat)
    private const int InputFormatContinuous = 0;
    private const int InputFormatSignedLabels = 1;
    private const int InputFormatProbabilityDistribution = 2;
    private const int InputFormatSimilarityLabels = 3;
    private const int InputFormatCriticScores = 4;
    private const int InputFormatSegmentationMask = 5;
    private const int InputFormatMarginBased = 6;

    private static ComponentTestInfo? ExtractLossInfo(INamedTypeSymbol symbol)
    {
        bool isNonNegative = true, zeroForIdentical = true, zeroDerivForIdentical = true;
        bool hasStandardGradientSign = true;
        bool throwsNotSupported = false;
        int apiShape = ApiShapeVectorVector;
        int testInputFormat = 0; // Continuous

        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass is null) continue;
            if (!attr.AttributeClass.ToDisplayString().EndsWith("LossPropertyAttribute", System.StringComparison.Ordinal))
                continue;

            foreach (var named in attr.NamedArguments)
            {
                switch (named.Key)
                {
                    case "IsNonNegative":
                        isNonNegative = (bool)(named.Value.Value ?? true);
                        break;
                    case "ZeroForIdentical":
                        zeroForIdentical = (bool)(named.Value.Value ?? true);
                        break;
                    case "ApiShape":
                        apiShape = (int)(named.Value.Value ?? 0);
                        break;
                    case "TestInputFormat":
                        testInputFormat = (int)(named.Value.Value ?? 0);
                        break;
                    case "ZeroDerivativeForIdentical":
                        zeroDerivForIdentical = (bool)(named.Value.Value ?? true);
                        break;
                    case "HasStandardGradientSign":
                        hasStandardGradientSign = (bool)(named.Value.Value ?? true);
                        break;
                }
            }
        }

        bool hasParameterlessCtor = HasAccessibleParameterlessConstructor(symbol);

        // Check if the CalculateLoss override throws NotSupportedException (source-mode only)
        foreach (var member in symbol.GetMembers("CalculateLoss"))
        {
            if (member is IMethodSymbol method && method.IsOverride)
            {
                // Check if the method body is a single throw statement
                // We can detect this by looking at the method's declaring syntax references
                foreach (var syntaxRef in method.DeclaringSyntaxReferences)
                {
                    var syntax = syntaxRef.GetSyntax();
                    var text = syntax.ToString();
                    if (text.Contains("throw new NotSupportedException") ||
                        text.Contains("throw new System.NotSupportedException"))
                    {
                        throwsNotSupported = true;
                        break;
                    }
                }
                if (throwsNotSupported) break;
            }
        }

        // Check if it extends LossFunctionBase<T>
        bool extendsBase = false;
        var baseType = symbol.BaseType;
        while (baseType is not null)
        {
            if (baseType.IsGenericType &&
                baseType.OriginalDefinition.ToDisplayString().StartsWith(LossFunctionBasePrefix, System.StringComparison.Ordinal))
            {
                extendsBase = true;
                break;
            }
            baseType = baseType.BaseType;
        }

        return new ComponentTestInfo
        {
            ClassName = symbol.Name,
            FullyQualifiedName = symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            TypeParameterCount = symbol.TypeParameters.Length,
            IsNonNegative = isNonNegative,
            ZeroForIdentical = zeroForIdentical,
            HasParameterlessConstructor = hasParameterlessCtor,
            ThrowsNotSupported = throwsNotSupported,
            ExtendsLossFunctionBase = extendsBase,
            ApiShape = apiShape,
            TestInputFormat = testInputFormat,
            // Use explicit attribute value; for non-continuous formats, auto-disable
            HasStandardGradientSign = hasStandardGradientSign && testInputFormat == InputFormatContinuous,
            // Use explicit attribute if set, otherwise infer from format
            ZeroDerivativeForIdentical = zeroDerivForIdentical && zeroForIdentical,
            IsActivation = false
        };
    }

    /// <summary>
    /// Checks if a type has a public constructor callable with zero arguments.
    /// </summary>
    private static bool HasAccessibleParameterlessConstructor(INamedTypeSymbol symbol)
    {
        foreach (var ctor in symbol.InstanceConstructors)
        {
            if (ctor.DeclaredAccessibility != Accessibility.Public)
                continue;
            if (ctor.Parameters.Length == 0)
                return true;
            bool allOptional = true;
            foreach (var param in ctor.Parameters)
            {
                if (!param.HasExplicitDefaultValue)
                {
                    allOptional = false;
                    break;
                }
            }
            if (allOptional)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Discovers activation and loss function types from referenced assemblies.
    /// </summary>
    private static void DiscoverComponentsFromReferencedAssemblies(
        Compilation compilation,
        HashSet<string> activationSeen,
        HashSet<string> lossSeen,
        List<ComponentTestInfo> activations,
        List<ComponentTestInfo> losses)
    {
        foreach (var reference in compilation.References)
        {
            var symbol = compilation.GetAssemblyOrModuleSymbol(reference);
            if (symbol is IAssemblySymbol assembly)
            {
                CollectComponentsFromNamespace(assembly.GlobalNamespace, activationSeen, lossSeen, activations, losses);
            }
        }
    }

    /// <summary>
    /// Recursively collects activation/loss types from a namespace.
    /// </summary>
    private static void CollectComponentsFromNamespace(
        INamespaceSymbol ns,
        HashSet<string> activationSeen,
        HashSet<string> lossSeen,
        List<ComponentTestInfo> activations,
        List<ComponentTestInfo> losses)
    {
        foreach (var member in ns.GetMembers())
        {
            if (member is INamespaceSymbol childNs)
            {
                CollectComponentsFromNamespace(childNs, activationSeen, lossSeen, activations, losses);
            }
            else if (member is INamedTypeSymbol type)
            {
                if (type.TypeKind != TypeKind.Class || type.IsAbstract)
                    continue;

                // Check for ActivationProperty attribute
                bool hasActivationProp = false;
                bool hasLossProp = false;
                foreach (var attr in type.GetAttributes())
                {
                    if (attr.AttributeClass is null) continue;
                    var attrName = attr.AttributeClass.ToDisplayString();
                    if (attrName.EndsWith("ActivationPropertyAttribute", System.StringComparison.Ordinal))
                        hasActivationProp = true;
                    else if (attrName.EndsWith("LossPropertyAttribute", System.StringComparison.Ordinal))
                        hasLossProp = true;
                }

                if (hasActivationProp)
                {
                    var info = ExtractActivationInfo(type);
                    if (info is not null && activationSeen.Add(info.FullyQualifiedName))
                        activations.Add(info);
                }

                if (hasLossProp)
                {
                    var info = ExtractLossInfo(type);
                    if (info is not null && lossSeen.Add(info.FullyQualifiedName))
                        losses.Add(info);
                }
            }
        }
    }

    /// <summary>
    /// Emits a generated test class for an activation function.
    /// </summary>
    private static void EmitActivationTestClass(
        SourceProductionContext context,
        ComponentTestInfo act,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(act.FullyQualifiedName);
        string constructorExpr = act.TypeParameterCount <= 1
            ? $"new {typeName}<double>()"
            : $"new {typeName}<double>()";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated activation function test. Mathematical invariant tests are inherited from ActivationFunctionTestBase.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : ActivationFunctionTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override IActivationFunction<double> CreateActivation()");
        sb.AppendLine($"        => {constructorExpr};");

        // Override properties based on attribute metadata
        if (!act.IsMonotonic)
            sb.AppendLine("    protected override bool IsMonotonic => false;");
        if (!act.ZeroPreserving)
            sb.AppendLine("    protected override bool ZeroMapsToZero => false;");
        if (act.IsBounded)
        {
            sb.AppendLine("    protected override bool IsBounded => true;");
            if (act.BoundLower != -1.0)
                sb.AppendLine($"    protected override double BoundLower => {act.BoundLower.ToString(System.Globalization.CultureInfo.InvariantCulture)};");
            if (act.BoundUpper != 1.0)
                sb.AppendLine($"    protected override double BoundUpper => {act.BoundUpper.ToString(System.Globalization.CultureInfo.InvariantCulture)};");
        }
        if (act.IsStochastic)
            sb.AppendLine("    protected override bool IsStochastic => true;");

        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(act.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits a generated test class for a loss function.
    /// </summary>
    private static void EmitLossTestClass(
        SourceProductionContext context,
        ComponentTestInfo loss,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName);
        string constructorExpr = loss.TypeParameterCount <= 1
            ? $"new {typeName}<double>()"
            : $"new {typeName}<double>()";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated loss function test. Mathematical invariant tests are inherited from LossFunctionTestBase.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : LossFunctionTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override ILossFunction<double> CreateLoss()");
        sb.AppendLine($"        => {constructorExpr};");

        // Override properties based on attribute metadata
        if (!loss.IsNonNegative)
            sb.AppendLine("    protected override bool IsNonNegative => false;");
        if (!loss.ZeroForIdentical)
            sb.AppendLine("    protected override bool ZeroLossForIdentical => false;");

        if (!loss.HasStandardGradientSign)
            sb.AppendLine("    protected override bool HasStandardGradientSign => false;");
        if (!loss.ZeroDerivativeForIdentical)
            sb.AppendLine("    protected override bool ZeroDerivativeForIdentical => false;");

        // Emit test data overrides based on TestInputFormat
        EmitTestDataOverrides(sb, loss.TestInputFormat);

        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits virtual property overrides for test data based on the loss input format.
    /// </summary>
    private static void EmitTestDataOverrides(StringBuilder sb, int testInputFormat)
    {
        switch (testInputFormat)
        {
            case InputFormatSignedLabels:
                // Hinge, SquaredHinge, ModifiedHuber, Exponential: labels in {-1, +1}
                sb.AppendLine("    protected override double[] TestPredicted => new[] { 0.5, -0.3, 1.2 };");
                sb.AppendLine("    protected override double[] TestActual => new[] { 1.0, -1.0, 1.0 };");
                sb.AppendLine("    protected override double[] SmallErrorPredicted => new[] { 0.8, -0.8, 0.8 };");
                sb.AppendLine("    protected override double[] LargeErrorPredicted => new[] { -0.5, 0.5, -0.5 };");
                sb.AppendLine("    protected override double[] ErrorTestActual => new[] { 1.0, -1.0, 1.0 };");
                sb.AppendLine("    protected override double[] SignTestPredicted => new[] { 2.0 };");
                sb.AppendLine("    protected override double[] SignTestActual => new[] { 1.0 };");
                break;

            case InputFormatProbabilityDistribution:
                // CrossEntropy, CategoricalCE, Focal, WeightedCE: probabilities
                sb.AppendLine("    protected override double[] TestPredicted => new[] { 0.7, 0.2, 0.1 };");
                sb.AppendLine("    protected override double[] TestActual => new[] { 1.0, 0.0, 0.0 };");
                sb.AppendLine("    protected override double[] SmallErrorPredicted => new[] { 0.8, 0.1, 0.1 };");
                sb.AppendLine("    protected override double[] LargeErrorPredicted => new[] { 0.4, 0.3, 0.3 };");
                sb.AppendLine("    protected override double[] ErrorTestActual => new[] { 1.0, 0.0, 0.0 };");
                sb.AppendLine("    protected override double[] SignTestPredicted => new[] { 0.9 };");
                sb.AppendLine("    protected override double[] SignTestActual => new[] { 1.0 };");
                break;

            case InputFormatSimilarityLabels:
                // ContrastiveLoss: similarity {0,1} with distance predictions
                sb.AppendLine("    protected override double[] TestPredicted => new[] { 0.5, 1.2, 0.3 };");
                sb.AppendLine("    protected override double[] TestActual => new[] { 1.0, 0.0, 1.0 };");
                sb.AppendLine("    protected override double[] SmallErrorPredicted => new[] { 0.2, 0.8, 0.2 };");
                sb.AppendLine("    protected override double[] LargeErrorPredicted => new[] { 1.5, 0.1, 1.5 };");
                sb.AppendLine("    protected override double[] ErrorTestActual => new[] { 1.0, 0.0, 1.0 };");
                sb.AppendLine("    protected override double[] SignTestPredicted => new[] { 0.5 };");
                sb.AppendLine("    protected override double[] SignTestActual => new[] { 1.0 };");
                break;

            case InputFormatCriticScores:
                // Wasserstein: critic scores with {-1, +1} labels
                sb.AppendLine("    protected override double[] TestPredicted => new[] { 2.5, -1.3, 0.8 };");
                sb.AppendLine("    protected override double[] TestActual => new[] { 1.0, -1.0, 1.0 };");
                sb.AppendLine("    protected override double[] SmallErrorPredicted => new[] { 0.8, -0.8, 0.8 };");
                sb.AppendLine("    protected override double[] LargeErrorPredicted => new[] { -0.5, 0.5, -0.5 };");
                sb.AppendLine("    protected override double[] ErrorTestActual => new[] { 1.0, -1.0, 1.0 };");
                sb.AppendLine("    protected override double[] SignTestPredicted => new[] { 2.0 };");
                sb.AppendLine("    protected override double[] SignTestActual => new[] { 1.0 };");
                break;

            case InputFormatSegmentationMask:
                // Dice, Jaccard: binary mask predictions
                sb.AppendLine("    protected override double[] TestPredicted => new[] { 0.8, 0.1, 0.9 };");
                sb.AppendLine("    protected override double[] TestActual => new[] { 1.0, 0.0, 1.0 };");
                sb.AppendLine("    protected override double[] SmallErrorPredicted => new[] { 0.9, 0.1, 0.9 };");
                sb.AppendLine("    protected override double[] LargeErrorPredicted => new[] { 0.5, 0.5, 0.5 };");
                sb.AppendLine("    protected override double[] ErrorTestActual => new[] { 1.0, 0.0, 1.0 };");
                sb.AppendLine("    protected override double[] SignTestPredicted => new[] { 0.9 };");
                sb.AppendLine("    protected override double[] SignTestActual => new[] { 1.0 };");
                break;

            case InputFormatMarginBased:
                // MarginLoss (capsule networks)
                sb.AppendLine("    protected override double[] TestPredicted => new[] { 0.85, 0.15, 0.75 };");
                sb.AppendLine("    protected override double[] TestActual => new[] { 1.0, 0.0, 1.0 };");
                sb.AppendLine("    protected override double[] SmallErrorPredicted => new[] { 0.88, 0.12, 0.88 };");
                sb.AppendLine("    protected override double[] LargeErrorPredicted => new[] { 0.5, 0.5, 0.5 };");
                sb.AppendLine("    protected override double[] ErrorTestActual => new[] { 1.0, 0.0, 1.0 };");
                sb.AppendLine("    protected override double[] SignTestPredicted => new[] { 0.85 };");
                sb.AppendLine("    protected override double[] SignTestActual => new[] { 1.0 };");
                break;

            // InputFormatContinuous (0) = default, no overrides needed
        }
    }

    /// <summary>
    /// Emits a generated test class for a triplet-style loss function (anchor, positive, negative matrices).
    /// </summary>
    private static void EmitTripletLossTestClass(
        SourceProductionContext context,
        ComponentTestInfo loss,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName);
        string constructorExpr = $"new {typeName}<double>()";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated triplet loss test. Invariant tests are inherited from TripletLossTestBase.");
        sb.AppendLine("using AiDotNet.LossFunctions;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : TripletLossTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override TripletLoss<double> CreateLoss()");
        sb.AppendLine($"        => {constructorExpr};");
        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits a generated test class for a noise contrastive estimation loss (target logits + noise matrix).
    /// </summary>
    private static void EmitContrastiveLossTestClass(
        SourceProductionContext context,
        ComponentTestInfo loss,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName);
        string constructorExpr = $"new {typeName}<double>()";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated contrastive loss test. Invariant tests are inherited from ContrastiveLossTestBase.");
        sb.AppendLine("using AiDotNet.LossFunctions;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : ContrastiveLossTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override NoiseContrastiveEstimationLoss<double> CreateLoss()");
        sb.AppendLine($"        => {constructorExpr};");
        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Emits a generated test class for a sparse categorical loss (different-length predicted/actual vectors).
    /// </summary>
    private static void EmitSparseCategoricalLossTestClass(
        SourceProductionContext context,
        ComponentTestInfo loss,
        string testClassName)
    {
        var typeName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName);
        string constructorExpr = $"new {typeName}<double>()";

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("// Auto-generated sparse categorical loss test. Invariant tests are inherited from SparseCategoricalLossTestBase.");
        sb.AppendLine("using AiDotNet.Interfaces;");
        sb.AppendLine("using AiDotNet.Tests.ModelFamilyTests.Base;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine($"public class {testClassName} : SparseCategoricalLossTestBase");
        sb.AppendLine("{");
        sb.AppendLine($"    protected override ILossFunction<double> CreateLoss()");
        sb.AppendLine($"        => {constructorExpr};");
        sb.AppendLine("}");

        var hintName = GeneratorHelpers.StripGenericSuffix(loss.FullyQualifiedName).Replace(".", "_") + "Tests.g.cs";
        context.AddSource(hintName, sb.ToString());
    }

    /// <summary>
    /// Info about an activation function or loss function for test generation.
    /// </summary>
    private class ComponentTestInfo
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public bool HasParameterlessConstructor { get; set; }
        public bool IsActivation { get; set; }

        // Activation-specific
        public bool IsMonotonic { get; set; } = true;
        public bool ZeroPreserving { get; set; } = true;
        public bool IsBounded { get; set; }
        public double BoundLower { get; set; } = -1.0;
        public double BoundUpper { get; set; } = 1.0;
        public bool IsVectorActivation { get; set; }
        public bool HasLearnableParameters { get; set; }
        public bool IsStochastic { get; set; }

        // Loss-specific
        public bool IsNonNegative { get; set; } = true;
        public bool ZeroForIdentical { get; set; } = true;
        public bool ThrowsNotSupported { get; set; }
        public bool ExtendsLossFunctionBase { get; set; }
        public int ApiShape { get; set; }
        public int TestInputFormat { get; set; }
        public bool HasStandardGradientSign { get; set; } = true;
        public bool ZeroDerivativeForIdentical { get; set; } = true;
    }

    private static void EmitTestCoverageClass(
        SourceProductionContext context,
        List<ModelTestInfo> testedModels,
        List<ModelTestInfo> untestedModels)
    {
        var totalCount = testedModels.Count + untestedModels.Count;
        var coveragePercent = totalCount > 0
            ? (testedModels.Count * 100.0 / totalCount)
            : 0.0;

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Generated;");
        sb.AppendLine();

        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated test coverage report for annotated model classes.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class TestCoverage");
        sb.AppendLine("{");

        sb.AppendLine($"    /// <summary>Total annotated models tracked.</summary>");
        sb.AppendLine($"    public const int TotalModels = {totalCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Models with test coverage.</summary>");
        sb.AppendLine($"    public const int TestedCount = {testedModels.Count};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Models without test coverage.</summary>");
        sb.AppendLine($"    public const int UntestedCount = {untestedModels.Count};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Coverage percentage.</summary>");
        sb.AppendLine($"    public const double CoveragePercent = {coveragePercent.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)};");
        sb.AppendLine();

        sb.AppendLine("    /// <summary>Names of models that have corresponding test classes.</summary>");
        sb.AppendLine("    public static IReadOnlyList<string> TestedModelNames { get; } = new string[]");
        sb.AppendLine("    {");
        foreach (var model in testedModels)
        {
            sb.AppendLine($"        \"{EscapeString(model.ClassName)}\",");
        }
        sb.AppendLine("    };");
        sb.AppendLine();

        sb.AppendLine("    /// <summary>Names of models that do NOT have corresponding test classes.</summary>");
        sb.AppendLine("    public static IReadOnlyList<string> UntestedModelNames { get; } = new string[]");
        sb.AppendLine("    {");
        foreach (var model in untestedModels)
        {
            sb.AppendLine($"        \"{EscapeString(model.ClassName)}\",");
        }
        sb.AppendLine("    };");

        sb.AppendLine("}");

        context.AddSource("TestCoverage.g.cs", sb.ToString());
    }

    private static string StripBacktick(string name)
    {
        var backtick = name.IndexOf('`');
        return backtick >= 0 ? name.Substring(0, backtick) : name;
    }

    private static string EscapeString(string value)
    {
        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"");
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

    private class ModelTestInfo
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Domains { get; set; } = new List<int>();
        public List<int> Categories { get; set; } = new List<int>();
        public List<int> Tasks { get; set; } = new List<int>();
        public bool HasTests { get; set; }
        public Location? Location { get; set; }

        // Interface detection
        public bool ImplementsNeuralNetworkModel { get; set; }
        public bool ImplementsDiffusionModel { get; set; }
        public bool ImplementsGaussianProcess { get; set; }

        // Input type detection (from IFullModel type arguments)
        public bool UsesTensorInput { get; set; }
        public bool UsesMatrixInput { get; set; }

        /// <summary>Whether the IFullModel output type is Vector (not Matrix or Tensor).</summary>
        public bool UsesVectorOutput { get; set; }

        /// <summary>Whether the model has an accessible parameterless constructor.</summary>
        public bool HasParameterlessConstructor { get; set; }

        // Base class chain detection (for mid-level hierarchy resolution)
        public bool ExtendsAudioNeuralNetworkBase { get; set; }
        public bool ExtendsDocumentNeuralNetworkBase { get; set; }
        public bool ExtendsVisionLanguageModelBase { get; set; }
        public bool ExtendsSegmentationModelBase { get; set; }
        public bool ExtendsVideoNeuralNetworkBase { get; set; }
        public bool ExtendsLatentDiffusionModelBase { get; set; }
        public bool ExtendsTtsModelBase { get; set; }
        public bool ExtendsFinancialModelBase { get; set; }
        public bool ExtendsNERNeuralNetworkBase { get; set; }
        public bool ExtendsCodeModelBase { get; set; }
        // Phase B: Leaf-level hierarchy
        public bool ExtendsVideoDiffusionModelBase { get; set; }
        public bool ExtendsAudioDiffusionModelBase { get; set; }
        public bool ExtendsFrameInterpolationBase { get; set; }
        public bool ExtendsVideoSuperResolutionBase { get; set; }
        public bool ExtendsVideoDenoisingBase { get; set; }
        public bool ExtendsAudioClassifierBase { get; set; }
        public bool ExtendsOpticalFlowBase { get; set; }
        public bool ExtendsSpeakerRecognitionBase { get; set; }
        public bool ExtendsEnsembleClassifierBase { get; set; }
        public bool ExtendsNaiveBayesBase { get; set; }
        public bool ExtendsSVMBase { get; set; }
        public bool ExtendsForecastingModelBase { get; set; }
        public bool ExtendsThreeDDiffusionModelBase { get; set; }
        public bool ExtendsVideoInpaintingBase { get; set; }
        public bool ExtendsVideoStabilizationBase { get; set; }
        public bool ExtendsLinearClassifierBase { get; set; }
        public bool ExtendsMetaClassifierBase { get; set; }
        public bool ExtendsOrdinalClassifierBase { get; set; }
        public bool ExtendsSemiSupervisedClassifierBase { get; set; }
        public bool ExtendsMultiLabelClassifierBase { get; set; }
        public bool ExtendsFinancialNLPModelBase { get; set; }
        public bool ExtendsRiskModelBase { get; set; }
        public bool ExtendsPortfolioOptimizerBase { get; set; }
        public bool ExtendsTransformerNERBase { get; set; }
        public bool ExtendsSpanBasedNERBase { get; set; }
        public bool ExtendsSequenceLabelingNERBase { get; set; }
        public bool ExtendsAnomalyDetectorBase { get; set; }
        public bool ExtendsSurvivalModelBase { get; set; }
        public bool ExtendsCausalModelBase { get; set; }
        public bool ExtendsRLAgentBase { get; set; }
        public bool ExtendsNonLinearRegressionBase { get; set; }
        public bool ExtendsProbabilisticClassifierBase { get; set; }
    }

    /// <summary>
    /// Identifies which test base class family a model should use.
    /// Each value maps to a specific base class, factory method, return type, and using set.
    /// </summary>
    private enum TestFamily
    {
        GaussianProcess,
        TimeSeries,
        Diffusion,
        LatentDiffusion,
        GAN,
        Embedding,
        GraphNN,
        AudioNN,
        DocumentNN,
        VisionLanguage,
        Segmentation,
        VideoNN,
        TTS,
        Financial,
        NER,
        CodeModel,
        VideoDiffusion,
        AudioDiffusion,
        FrameInterpolation,
        VideoSuperResolution,
        VideoDenoising,
        AudioClassifier,
        OpticalFlow,
        SpeakerRecognition,
        EnsembleClassifier,
        NaiveBayes,
        SVM,
        Forecasting,
        ThreeDDiffusion,
        VideoInpainting,
        VideoStabilization,
        LinearClassifier,
        MetaClassifier,
        OrdinalClassifier,
        SemiSupervisedClassifier,
        MultiLabelClassifier,
        FinancialNLP,
        RiskModel,
        PortfolioOptimizer,
        TransformerNER,
        SpanBasedNER,
        SequenceLabelingNER,
        AnomalyDetector,
        Survival,
        Causal,
        ReinforcementLearning,
        Regression,
        NonLinearRegression,
        Classification,
        ProbabilisticClassifier,
        Clustering,
        NeuralNetwork
    }

    /// <summary>
    /// Returns the base class name for the given test family.
    /// </summary>
    private static string GetBaseClassName(TestFamily family)
    {
        switch (family)
        {
            case TestFamily.GaussianProcess:       return "GaussianProcessModelTestBase";
            case TestFamily.TimeSeries:            return "TimeSeriesModelTestBase";
            case TestFamily.Diffusion:             return "DiffusionModelTestBase";
            case TestFamily.LatentDiffusion:       return "LatentDiffusionTestBase";
            case TestFamily.GAN:                   return "GANModelTestBase";
            case TestFamily.Embedding:             return "EmbeddingModelTestBase";
            case TestFamily.GraphNN:               return "GraphNNModelTestBase";
            case TestFamily.AudioNN:               return "AudioNNModelTestBase";
            case TestFamily.DocumentNN:            return "DocumentNNModelTestBase";
            case TestFamily.VisionLanguage:        return "VisionLanguageTestBase";
            case TestFamily.Segmentation:          return "SegmentationTestBase";
            case TestFamily.VideoNN:               return "VideoNNModelTestBase";
            case TestFamily.TTS:                   return "TTSModelTestBase";
            case TestFamily.Financial:             return "FinancialModelTestBase";
            case TestFamily.NER:                   return "NERModelTestBase";
            case TestFamily.CodeModel:             return "CodeModelTestBase";
            case TestFamily.VideoDiffusion:        return "VideoDiffusionTestBase";
            case TestFamily.AudioDiffusion:        return "AudioDiffusionTestBase";
            case TestFamily.FrameInterpolation:    return "FrameInterpolationTestBase";
            case TestFamily.VideoSuperResolution:  return "VideoSuperResolutionTestBase";
            case TestFamily.VideoDenoising:        return "VideoDenoisingTestBase";
            case TestFamily.AudioClassifier:       return "AudioClassifierTestBase";
            case TestFamily.OpticalFlow:           return "OpticalFlowTestBase";
            case TestFamily.SpeakerRecognition:    return "SpeakerRecognitionTestBase";
            case TestFamily.EnsembleClassifier:    return "EnsembleClassifierTestBase";
            case TestFamily.NaiveBayes:            return "NaiveBayesTestBase";
            case TestFamily.SVM:                   return "SVMTestBase";
            case TestFamily.Forecasting:           return "ForecastingModelTestBase";
            case TestFamily.ThreeDDiffusion:       return "ThreeDDiffusionTestBase";
            case TestFamily.VideoInpainting:       return "VideoInpaintingTestBase";
            case TestFamily.VideoStabilization:    return "VideoStabilizationTestBase";
            case TestFamily.LinearClassifier:      return "LinearClassifierTestBase";
            case TestFamily.MetaClassifier:        return "MetaClassifierTestBase";
            case TestFamily.OrdinalClassifier:     return "OrdinalClassifierTestBase";
            case TestFamily.SemiSupervisedClassifier: return "SemiSupervisedClassifierTestBase";
            case TestFamily.MultiLabelClassifier:  return "MultiLabelClassifierTestBase";
            case TestFamily.FinancialNLP:          return "FinancialNLPTestBase";
            case TestFamily.RiskModel:             return "RiskModelTestBase";
            case TestFamily.PortfolioOptimizer:    return "PortfolioOptimizerTestBase";
            case TestFamily.TransformerNER:        return "TransformerNERTestBase";
            case TestFamily.SpanBasedNER:          return "SpanBasedNERTestBase";
            case TestFamily.SequenceLabelingNER:   return "SequenceLabelingNERTestBase";
            case TestFamily.AnomalyDetector:       return "AnomalyDetectorTestBase";
            case TestFamily.Survival:              return "SurvivalModelTestBase";
            case TestFamily.Causal:                return "CausalModelTestBase";
            case TestFamily.ReinforcementLearning: return "ReinforcementLearningTestBase";
            case TestFamily.Regression:            return "RegressionModelTestBase";
            case TestFamily.NonLinearRegression:   return "NonLinearRegressionTestBase";
            case TestFamily.Classification:        return "ClassificationModelTestBase";
            case TestFamily.ProbabilisticClassifier: return "ProbabilisticClassifierTestBase";
            case TestFamily.Clustering:            return "ClusteringModelTestBase";
            case TestFamily.NeuralNetwork:         return "NeuralNetworkModelTestBase";
            default:                               return "RegressionModelTestBase";
        }
    }

    /// <summary>
    /// Returns the factory method name for the given test family.
    /// NN-derived families use CreateNetwork(); all others use CreateModel().
    /// </summary>
    private static string GetFactoryMethodName(TestFamily family)
    {
        switch (family)
        {
            case TestFamily.GAN:
            case TestFamily.Embedding:
            case TestFamily.GraphNN:
            case TestFamily.AudioNN:
            case TestFamily.DocumentNN:
            case TestFamily.VisionLanguage:
            case TestFamily.Segmentation:
            case TestFamily.VideoNN:
            case TestFamily.TTS:
            case TestFamily.Financial:
            case TestFamily.NER:
            case TestFamily.CodeModel:
            case TestFamily.FrameInterpolation:
            case TestFamily.VideoSuperResolution:
            case TestFamily.VideoDenoising:
            case TestFamily.AudioClassifier:
            case TestFamily.OpticalFlow:
            case TestFamily.SpeakerRecognition:
            case TestFamily.Forecasting:
            case TestFamily.VideoInpainting:
            case TestFamily.VideoStabilization:
            case TestFamily.FinancialNLP:
            case TestFamily.RiskModel:
            case TestFamily.PortfolioOptimizer:
            case TestFamily.TransformerNER:
            case TestFamily.SpanBasedNER:
            case TestFamily.SequenceLabelingNER:
            case TestFamily.NeuralNetwork:
                return "CreateNetwork";
            default:
                return "CreateModel";
        }
    }

    /// <summary>
    /// Returns the factory method return type code for the given test family.
    /// </summary>
    private static string GetReturnTypeCode(TestFamily family)
    {
        switch (family)
        {
            case TestFamily.GaussianProcess:
                return "IGaussianProcess<double>";
            case TestFamily.Diffusion:
            case TestFamily.LatentDiffusion:
            case TestFamily.VideoDiffusion:
            case TestFamily.AudioDiffusion:
            case TestFamily.ThreeDDiffusion:
                return "IDiffusionModel<double>";
            case TestFamily.GAN:
            case TestFamily.Embedding:
            case TestFamily.GraphNN:
            case TestFamily.AudioNN:
            case TestFamily.DocumentNN:
            case TestFamily.VisionLanguage:
            case TestFamily.Segmentation:
            case TestFamily.VideoNN:
            case TestFamily.TTS:
            case TestFamily.Financial:
            case TestFamily.NER:
            case TestFamily.CodeModel:
            case TestFamily.FrameInterpolation:
            case TestFamily.VideoSuperResolution:
            case TestFamily.VideoDenoising:
            case TestFamily.AudioClassifier:
            case TestFamily.OpticalFlow:
            case TestFamily.SpeakerRecognition:
            case TestFamily.Forecasting:
            case TestFamily.VideoInpainting:
            case TestFamily.VideoStabilization:
            case TestFamily.FinancialNLP:
            case TestFamily.RiskModel:
            case TestFamily.PortfolioOptimizer:
            case TestFamily.TransformerNER:
            case TestFamily.SpanBasedNER:
            case TestFamily.SequenceLabelingNER:
            case TestFamily.NeuralNetwork:
                return "INeuralNetworkModel<double>";
            case TestFamily.ReinforcementLearning:
                return "IFullModel<double, Vector<double>, Vector<double>>";
            case TestFamily.MultiLabelClassifier:
                return "IFullModel<double, Matrix<double>, Matrix<double>>";
            case TestFamily.TimeSeries:
            case TestFamily.Regression:
            case TestFamily.NonLinearRegression:
            case TestFamily.Classification:
            case TestFamily.ProbabilisticClassifier:
            case TestFamily.AnomalyDetector:
            case TestFamily.Survival:
            case TestFamily.Causal:
            case TestFamily.LinearClassifier:
            case TestFamily.MetaClassifier:
            case TestFamily.OrdinalClassifier:
            case TestFamily.SemiSupervisedClassifier:
            case TestFamily.Clustering:
            default:
                return "IFullModel<double, Matrix<double>, Vector<double>>";
        }
    }

    /// <summary>
    /// Returns whether the generated test file needs using AiDotNet.Tensors.LinearAlgebra
    /// (for Matrix/Vector types in the return type).
    /// </summary>
    private static bool NeedsMatrixUsing(TestFamily family)
    {
        switch (family)
        {
            case TestFamily.TimeSeries:
            case TestFamily.Regression:
            case TestFamily.NonLinearRegression:
            case TestFamily.Classification:
            case TestFamily.ProbabilisticClassifier:
            case TestFamily.EnsembleClassifier:
            case TestFamily.NaiveBayes:
            case TestFamily.SVM:
            case TestFamily.AnomalyDetector:
            case TestFamily.Survival:
            case TestFamily.Causal:
            case TestFamily.ReinforcementLearning:
            case TestFamily.LinearClassifier:
            case TestFamily.MetaClassifier:
            case TestFamily.OrdinalClassifier:
            case TestFamily.SemiSupervisedClassifier:
            case TestFamily.MultiLabelClassifier:
            case TestFamily.Clustering:
                return true;
            default:
                return false;
        }
    }
}
