using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// The paper-fidelity half of the test scaffold: checks that ask whether a model is the model its
/// paper describes, not merely whether it runs.
/// </summary>
/// <remarks>
/// <para>
/// The family invariants prove a model produces finite, deterministic, clonable output. They passed for
/// models whose options were never read (Dreamer, World Models, Decision Transformer) and for an agent
/// whose objective applied softmax twice (A3C). Four checks close that gap:
/// </para>
/// <list type="number">
/// <item><b>Option sensitivity</b> - a generated theory, one case per perturbable options property, that
/// requires perturbing the property to change observable behaviour (<see cref="EmitOptionSensitivityTheory"/>),
/// backed statically by ADNOPT001 in <see cref="OptionsFidelityAnalyzer"/>.</item>
/// <item><b>Learns a toy task</b> - inherited facts on the RL, classification and regression family bases.</item>
/// <item><b>Paper test required</b> - every model needs a hand-written <c>&lt;Model&gt;PaperTests</c> class
/// (ADNGEN003, <see cref="ReportMissingPaperTests"/>).</item>
/// <item><b>Paper defaults declared</b> - <c>[PaperDefault]</c> values checked against the options
/// constructor by a generated test (<see cref="EmitPaperDefaultsTests"/>), with ADNOPT002/003 at build time.</item>
/// </list>
/// </remarks>
public partial class TestScaffoldGenerator
{
    private const string ModelOptionsDisplayName = "AiDotNet.Models.Options.ModelOptions";
    private const string OptionSensitivityHarnessMetadataName = "AiDotNet.Tests.ModelFamilyTests.Base.OptionSensitivityHarness";
    private const string PaperDefaultsHarnessMetadataName = "AiDotNet.Tests.ModelFamilyTests.Base.PaperDefaultsHarness";

    // The xUnit category every paper-fidelity test carries. The PR shards exclude it (Invoke-Shard.ps1); the
    // paper-fidelity lanes select it (tools/PaperFidelity/Invoke-PaperFidelitySlice.ps1).
    private const string PaperFidelityCategory = "PaperFidelity";

    private static readonly DiagnosticDescriptor MissingPaperTests = new(
        id: "ADNGEN003",
        title: "Model has no hand-written paper test",
        messageFormat: "Model '{0}' has no hand-written test class named {1} with at least one [Fact] or [Theory]. Generated invariants prove a model runs; only an equation-level or reference-value check against the paper proves it is the paper's model. Add {1} in tests/AiDotNet.Tests.",
        category: "AiDotNet.PaperFidelity",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Every model must have a hand-written <ModelName>PaperTests class checking at least one equation or published reference value. Existing violations are listed in DefectBaselines.ModelsWithoutPaperTests, which may only shrink.");

    private static readonly DiagnosticDescriptor StalePaperTestsBaselineEntry = new(
        id: "ADNGEN004",
        title: "Missing-paper-test baseline entry is stale",
        messageFormat: "'{0}' is listed in DefectBaselines.ModelsWithoutPaperTests but now has a paper test class or no longer exists; remove its line so the baseline stays an exact inventory",
        category: "AiDotNet.PaperFidelity",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor UnconstructiblePaperDefaults = new(
        id: "ADNGEN005",
        title: "Options class with [PaperDefault] cannot be constructed by the generated test",
        messageFormat: "'{0}' declares [PaperDefault] values but {1}, so the generated paper-defaults test cannot construct it to check them",
        category: "AiDotNet.PaperFidelity",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A declared paper value that no test can check is an unverified claim. Give the options class a public parameterless constructor.");

    /// <summary>
    /// Families whose base declares <c>AssertOptionChangesObservableBehaviour</c>: every family on the
    /// NeuralNetworkModelTestBase, ClassificationModelTestBase and RegressionModelTestBase chains, and RL.
    /// </summary>
    private static readonly HashSet<TestFamily> OptionSensitivityFamilies = new()
    {
        // NeuralNetworkModelTestBase chain.
        TestFamily.NeuralNetwork, TestFamily.GAN, TestFamily.Embedding, TestFamily.GraphNN, TestFamily.AudioNN,
        TestFamily.DocumentNN, TestFamily.VisionLanguage, TestFamily.Segmentation, TestFamily.VideoNN, TestFamily.TTS,
        TestFamily.Financial, TestFamily.NER, TestFamily.CodeModel, TestFamily.FrameInterpolation,
        TestFamily.VideoSuperResolution, TestFamily.VideoDenoising, TestFamily.AudioClassifier, TestFamily.OpticalFlow,
        TestFamily.SpeakerRecognition, TestFamily.Forecasting, TestFamily.VideoInpainting, TestFamily.VideoStabilization,
        TestFamily.FinancialNLP, TestFamily.RiskModel, TestFamily.PortfolioOptimizer, TestFamily.TransformerNER,
        TestFamily.SpanBasedNER, TestFamily.SequenceLabelingNER,
        // ClassificationModelTestBase chain.
        TestFamily.Classification, TestFamily.EnsembleClassifier, TestFamily.MetaClassifier, TestFamily.OrdinalClassifier,
        TestFamily.ProbabilisticClassifier, TestFamily.SemiSupervisedClassifier, TestFamily.LinearClassifier,
        TestFamily.NaiveBayes, TestFamily.SVM,
        // RegressionModelTestBase chain.
        TestFamily.Regression, TestFamily.NonLinearRegression,
        // ReinforcementLearningTestBase.
        TestFamily.ReinforcementLearning,
    };

    /// <summary>One generated option-sensitivity case: an options property, and why it is skipped if it is.</summary>
    private sealed class OptionSensitivityCase
    {
        public OptionSensitivityCase(string property, string? skipReason)
        {
            Property = property;
            SkipReason = skipReason;
        }

        public string Property { get; }

        public string? SkipReason { get; }
    }

    // ---------------------------------------------------------------------------------------------
    // Paper tests (ADNGEN003 / ADNGEN004)
    // ---------------------------------------------------------------------------------------------

    /// <summary>
    /// The class name of a <c>*PaperTests</c> class that can actually run a test (see
    /// <see cref="PaperTestDetection.HasRunnableTest"/>); null otherwise, so it does not clear ADNGEN003.
    /// </summary>
    private static string? GetRunnablePaperTestClassName(GeneratorSyntaxContext ctx) =>
        ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) is INamedTypeSymbol type && PaperTestDetection.HasRunnableTest(type)
            ? type.Name
            : null;

    /// <summary>A class named <c>*PaperTests</c> with at least one xUnit test method.</summary>
    private static bool IsPaperTestCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax declaration) return false;
        if (!declaration.Identifier.ValueText.EndsWith("PaperTests", System.StringComparison.Ordinal)) return false;
        foreach (var method in declaration.Members.OfType<MethodDeclarationSyntax>())
        {
            foreach (var attribute in method.AttributeLists.SelectMany(list => list.Attributes))
            {
                string name = attribute.Name.ToString();
                int dot = name.LastIndexOf('.');
                if (dot >= 0) name = name.Substring(dot + 1);
                if (name.EndsWith("Attribute", System.StringComparison.Ordinal))
                    name = name.Substring(0, name.Length - "Attribute".Length);
                if (name is "Fact" or "Theory" or "SkippableFact" or "SkippableTheory") return true;
            }
        }

        return false;
    }

    /// <summary>
    /// The paper test class a model must have: <c>&lt;SimpleName&gt;PaperTests</c>, qualified by the
    /// model's namespace leaf exactly where its generated fixture is (a namesake that does not own its
    /// simple name, or a simple name several models share without an owner).
    /// </summary>
    private static string ExpectedPaperTestClassName(ModelTestInfo model, IReadOnlyDictionary<string, int> simpleNameCounts)
    {
        string simple = StripBacktick(model.ClassName);
        bool namesakeWithoutTheName =
            CollisionOwners.TryGetValue(simple, out var ownerNamespace)
            && model.FullyQualifiedName.IndexOf(ownerNamespace, System.StringComparison.Ordinal) < 0;
        bool unownedShared = !CollisionOwners.ContainsKey(simple)
            && simpleNameCounts.TryGetValue(simple, out int count) && count > 1;
        string prefix = namesakeWithoutTheName || unownedShared ? NamespaceLeafPrefix(model.FullyQualifiedName) : string.Empty;
        return prefix + simple + "PaperTests";
    }

    /// <summary>
    /// Reports every model without its hand-written paper test (ADNGEN003) outside the baseline, and
    /// every baseline entry that is no longer missing one (ADNGEN004).
    /// </summary>
    private static void ReportMissingPaperTests(
        SourceProductionContext context,
        IEnumerable<ModelTestInfo> census,
        ImmutableArray<string?> paperTestClassNames)
    {
        var paperTests = new HashSet<string>(paperTestClassNames.Where(n => n is not null).Select(n => n ?? string.Empty),
            System.StringComparer.Ordinal);
        var models = census
            .GroupBy(m => m.FullyQualifiedName, System.StringComparer.Ordinal)
            .Select(g => g.First())
            .OrderBy(m => m.FullyQualifiedName, System.StringComparer.Ordinal)
            .ToList();
        var simpleNameCounts = models
            .GroupBy(m => StripBacktick(m.ClassName), System.StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Count(), System.StringComparer.Ordinal);

        var missing = new HashSet<string>(System.StringComparer.Ordinal);
        foreach (var model in models)
        {
            string expected = ExpectedPaperTestClassName(model, simpleNameCounts);
            if (paperTests.Contains(expected)) continue;
            missing.Add(model.FullyQualifiedName);
            if (DefectBaselines.ModelsWithoutPaperTests.Contains(model.FullyQualifiedName)) continue;
            context.ReportDiagnostic(Diagnostic.Create(MissingPaperTests, Location.None, model.FullyQualifiedName, expected));
        }

        foreach (var stale in DefectBaselines.ModelsWithoutPaperTests.Where(entry => !missing.Contains(entry))
                     .OrderBy(entry => entry, System.StringComparer.Ordinal))
        {
            context.ReportDiagnostic(Diagnostic.Create(StalePaperTestsBaselineEntry, Location.None, stale));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Option sensitivity
    // ---------------------------------------------------------------------------------------------

    /// <summary>
    /// The generated option-sensitivity cases for a model: every public settable/init property of its
    /// options type (and that type's bases up to <c>ModelOptions</c>) whose type has a generic valid
    /// perturbation. A property exempted by <c>[OptionBehaviourExempt]</c> or listed in the ADNOPT001
    /// baseline is still emitted, as a skipped case carrying the reason, so the census stays visible.
    /// </summary>
    private static List<OptionSensitivityCase> ComputeOptionSensitivityCases(INamedTypeSymbol modelClass)
    {
        var cases = new List<OptionSensitivityCase>();
        var options = FindDeclaredModelOptionsType(modelClass);
        if (options is null) return cases;

        var seen = new HashSet<string>(System.StringComparer.Ordinal);
        for (INamedTypeSymbol? walk = options.OriginalDefinition; walk is not null; walk = walk.BaseType?.OriginalDefinition)
        {
            foreach (var property in walk.GetMembers().OfType<IPropertySymbol>().OrderBy(p => p.Name, System.StringComparer.Ordinal))
            {
                if (property.IsStatic || property.IsIndexer || property.IsOverride) continue;
                if (property.DeclaredAccessibility != Accessibility.Public) continue;
                if (property.SetMethod is not { DeclaredAccessibility: Accessibility.Public }) continue;
                if (!seen.Add(property.Name)) continue;
                if (!HasGenericPerturbation(property.Type)) continue;

                string? skip = null;
                var exemption = property.GetAttributes().FirstOrDefault(a =>
                    a.AttributeClass?.ToDisplayString() == OptionsFidelityAnalyzer.OptionBehaviourExemptAttributeName);
                if (exemption is { ConstructorArguments.Length: 1 } && exemption.ConstructorArguments[0].Value is string reason
                    && !string.IsNullOrWhiteSpace(reason))
                {
                    skip = "[OptionBehaviourExempt] " + reason;
                }
                else if (OptionsFidelityBaselines.UnreadOptions.Contains(OptionsFidelityAnalyzer.PropertyKey(property)))
                {
                    skip = "ADNOPT001 (baselined in OptionsFidelityBaselines.UnreadOptions): nothing in the library reads "
                        + OptionsFidelityAnalyzer.PropertyKey(property) + ", so perturbing it cannot change behaviour.";
                }

                cases.Add(new OptionSensitivityCase(property.Name, skip));
            }

            if (walk.ToDisplayString() == ModelOptionsDisplayName) break;
        }

        return cases;
    }

    /// <summary>Whether the runtime harness has a generic valid perturbation for a property of this type.</summary>
    private static bool HasGenericPerturbation(ITypeSymbol type)
    {
        if (type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } nullable)
            type = nullable.TypeArguments[0];
        if (IsScalarPerturbable(type)) return true;
        var element = OptionsFidelityAnalyzer.ListElementType(type);
        if (element is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } nullableElement)
            element = nullableElement.TypeArguments[0];
        // Only concrete arrays and List<T> can be rebuilt with a different length.
        bool rebuildable = type is IArrayTypeSymbol
            || type is INamedTypeSymbol { Name: "List" or "IList" or "IReadOnlyList" or "ICollection" or "IReadOnlyCollection" or "IEnumerable" };
        return element is not null && rebuildable && IsScalarPerturbable(element);
    }

    private static bool IsScalarPerturbable(ITypeSymbol type)
        => type.TypeKind is TypeKind.Enum or TypeKind.TypeParameter
            || type.SpecialType is SpecialType.System_Boolean
                or SpecialType.System_Byte or SpecialType.System_SByte or SpecialType.System_Int16 or SpecialType.System_UInt16
                or SpecialType.System_Int32 or SpecialType.System_UInt32 or SpecialType.System_Int64 or SpecialType.System_UInt64
                or SpecialType.System_Single or SpecialType.System_Double or SpecialType.System_Decimal;

    /// <summary>
    /// Appends the per-option theory to a generated fixture whose family base supports it. One case per
    /// property, so CI reports each ignored option by name.
    /// </summary>
    private static void EmitOptionSensitivityTheory(StringBuilder sb, ModelTestInfo model, TestFamily family)
    {
        if (!OptionSensitivityFamilies.Contains(family) || model.OptionSensitivityCases.Count == 0) return;

        sb.AppendLine();
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Paper fidelity: perturbing each options property must change observable behaviour.");
        sb.AppendLine("    /// Skipped cases are ADNOPT001-baselined (never read) or carry an [OptionBehaviourExempt] reason.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    [Xunit.SkippableTheory(Timeout = 300000)]");
        sb.AppendLine($"    [Xunit.Trait(\"Category\", \"{PaperFidelityCategory}\")]");
        foreach (var optionCase in model.OptionSensitivityCases)
        {
            sb.Append("    [Xunit.InlineData(\"").Append(optionCase.Property).Append('"');
            if (optionCase.SkipReason is not null)
                sb.Append(", Skip = \"").Append(EscapeString(optionCase.SkipReason)).Append('"');
            sb.AppendLine(")]");
        }

        sb.AppendLine("    public async System.Threading.Tasks.Task Options_EachShouldChangeObservableBehaviour(string option)");
        sb.AppendLine("    {");
        sb.AppendLine("        await System.Threading.Tasks.Task.Yield();");
        sb.AppendLine("        AssertOptionChangesObservableBehaviour(option);");
        sb.AppendLine("    }");
    }

    // ---------------------------------------------------------------------------------------------
    // Paper defaults
    // ---------------------------------------------------------------------------------------------

    /// <summary>
    /// Emits one test per options class that declares any <c>[PaperDefault]</c>, asserting its
    /// parameterless constructor ships every declared value.
    /// </summary>
    private static void EmitPaperDefaultsTests(SourceProductionContext context, Compilation compilation)
    {
        if (compilation.GetTypeByMetadataName(PaperDefaultsHarnessMetadataName) is null) return;

        var declaring = new List<INamedTypeSymbol>();
        foreach (var reference in compilation.References)
        {
            if (compilation.GetAssemblyOrModuleSymbol(reference) is not IAssemblySymbol { Name: "AiDotNet" } assembly) continue;
            CollectPaperDefaultOptions(assembly.GlobalNamespace, declaring);
        }

        if (declaring.Count == 0) return;

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("namespace AiDotNet.Tests.ModelFamilyTests.Generated;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>Every options class's constructor must ship the values its [PaperDefault] declarations cite.</summary>");
        sb.AppendLine($"[Xunit.Trait(\"Category\", \"{PaperFidelityCategory}\")]");
        sb.AppendLine("public class GeneratedPaperDefaultsTests");
        sb.AppendLine("{");
        foreach (var options in declaring.OrderBy(t => t.ToDisplayString(), System.StringComparer.Ordinal))
        {
            // An abstract base cannot be constructed, and the concrete classes that inherit its declarations are
            // enumerated on their own, whose tests check its values; reporting it would be an error no edit can fix.
            if (options.IsAbstract) continue;

            string? problem = null;
            if (!HasPublicParameterlessConstructor(options)) problem = "it has no public parameterless constructor";
            else if (options.IsGenericType && !CanCloseOptionsWithDouble(options))
                problem = "it is generic in a way the generated test cannot close over double";

            if (problem is not null)
            {
                context.ReportDiagnostic(Diagnostic.Create(UnconstructiblePaperDefaults, Location.None, options.ToDisplayString(), problem));
                continue;
            }

            string closed = RenderClosedOptionsType(options);
            sb.AppendLine();
            sb.AppendLine("    [Xunit.Fact]");
            sb.AppendLine($"    public void {ToGeneratedIdentifier(options.ToDisplayString())}_ShipsPaperDefaults()");
            sb.AppendLine($"        => global::AiDotNet.Tests.ModelFamilyTests.Base.PaperDefaultsHarness.AssertConstructorDefaultsMatchPaper(typeof(GeneratedPaperDefaultsTests), typeof({closed}));");
        }

        sb.AppendLine("}");
        context.AddSource("GeneratedPaperDefaultsTests.g.cs", sb.ToString());
    }

    private static void CollectPaperDefaultOptions(INamespaceSymbol ns, List<INamedTypeSymbol> declaring)
    {
        foreach (var member in ns.GetMembers())
        {
            if (member is INamespaceSymbol child)
            {
                CollectPaperDefaultOptions(child, declaring);
                continue;
            }

            if (member is not INamedTypeSymbol { TypeKind: TypeKind.Class, DeclaredAccessibility: Accessibility.Public } type) continue;
            if (!DerivesFromModelOptions(type)) continue;
            bool declares = false;
            for (INamedTypeSymbol? walk = type; walk is not null && !declares; walk = walk.BaseType)
            {
                declares = walk.GetMembers().OfType<IPropertySymbol>().Any(p => p.GetAttributes().Any(a =>
                    a.AttributeClass?.ToDisplayString() == OptionsFidelityAnalyzer.PaperDefaultAttributeName));
            }

            if (declares) declaring.Add(type);
        }
    }

    private static bool DerivesFromModelOptions(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? walk = type; walk is not null; walk = walk.BaseType)
        {
            if (walk.ToDisplayString() == ModelOptionsDisplayName) return true;
        }

        return false;
    }
}
