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

    // Attribute metadata names
    private const string ModelDomainAttr = "AiDotNet.Attributes.ModelDomainAttribute";
    private const string ModelCategoryAttr = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttr = "AiDotNet.Attributes.ModelTaskAttribute";
    private const string ModelInputAttr = "AiDotNet.Attributes.ModelInputAttribute";
    private const string ModelMetadataExemptAttr = "AiDotNet.Attributes.ModelMetadataExemptAttribute";

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
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Model has no test coverage and lacks sufficient metadata for auto-generation. Add [ModelCategory] and [ModelTask] attributes, or create a manual test class.");

    private static readonly DiagnosticDescriptor CoverageSummary = new(
        id: "AIDN041",
        title: "Model test coverage summary",
        messageFormat: "{0} of {1} annotated models have test coverage ({2:F1}%)",
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

        var combined = modelClasses.Collect()
            .Combine(testClasses.Collect())
            .Combine(context.CompilationProvider);

        context.RegisterSourceOutput(combined, static (spc, source) =>
        {
            var ((models, tests), compilation) = source;
            Execute(spc, models, tests, compilation);
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

            // Move auto-generated from untested → tested
            foreach (var model in autoGenerated)
            {
                untestedModels.Remove(model);
                model.HasTests = true;
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
        bool extendsLatentDiffusion = false, extendsNonLinearRegression = false;
        bool extendsProbabilisticClassifier = false;

        var baseType = modelClass.BaseType;
        while (baseType is not null)
        {
            var baseName = baseType.Name;
            if (baseName.StartsWith("AudioNeuralNetworkBase", System.StringComparison.Ordinal))
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

        // Priority 3: Latent Diffusion (more specific than plain Diffusion)
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

        // Priority 12: Video NN
        if (model.ExtendsVideoNeuralNetworkBase)
            return TestFamily.VideoNN;

        // === TIER 3: Matrix/Vector model families ===

        // Priority 13: Non-linear Regression (more specific than Regression)
        if (model.ExtendsNonLinearRegressionBase)
            return TestFamily.NonLinearRegression;

        // Priority 14: Probabilistic Classifier (more specific than Classification)
        if (model.ExtendsProbabilisticClassifierBase)
            return TestFamily.ProbabilisticClassifier;

        // Priority 15: Regression task + Matrix input
        if (model.Tasks.Contains(TaskRegression) && model.UsesMatrixInput)
            return TestFamily.Regression;

        // Priority 16: Classification task + Matrix input
        if (model.Tasks.Contains(TaskClassification) && model.UsesMatrixInput)
            return TestFamily.Classification;

        // Priority 17: Clustering task + Matrix input
        if (model.Tasks.Contains(TaskClustering) && model.UsesMatrixInput)
            return TestFamily.Clustering;

        // === TIER 4: Fallbacks ===

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
            factoryBody = $"        => throw new System.NotImplementedException(" +
                          $"\"Model '{EscapeString(model.ClassName)}' requires constructor arguments. " +
                          $"Create a manual test class in ModelFamilyTests/ to replace this auto-generated stub.\");";
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
            case TestFamily.NeuralNetwork:
                return model.ImplementsNeuralNetworkModel;

            // Diffusion families require IDiffusionModel interface
            case TestFamily.Diffusion:
            case TestFamily.LatentDiffusion:
                return model.ImplementsDiffusionModel;

            // GP family requires IGaussianProcess interface
            case TestFamily.GaussianProcess:
                return model.ImplementsGaussianProcess;

            // Matrix/Vector families require IFullModel<T, Matrix<T>, Vector<T>>
            case TestFamily.Regression:
            case TestFamily.NonLinearRegression:
            case TestFamily.Classification:
            case TestFamily.ProbabilisticClassifier:
            case TestFamily.Clustering:
            case TestFamily.TimeSeries:
                return model.UsesMatrixInput && model.UsesVectorOutput;

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
                return "IDiffusionModel<double>";
            case TestFamily.GAN:
            case TestFamily.Embedding:
            case TestFamily.GraphNN:
            case TestFamily.AudioNN:
            case TestFamily.DocumentNN:
            case TestFamily.VisionLanguage:
            case TestFamily.Segmentation:
            case TestFamily.VideoNN:
            case TestFamily.NeuralNetwork:
                return "INeuralNetworkModel<double>";
            case TestFamily.TimeSeries:
            case TestFamily.Regression:
            case TestFamily.NonLinearRegression:
            case TestFamily.Classification:
            case TestFamily.ProbabilisticClassifier:
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
            case TestFamily.Clustering:
                return true;
            default:
                return false;
        }
    }
}
