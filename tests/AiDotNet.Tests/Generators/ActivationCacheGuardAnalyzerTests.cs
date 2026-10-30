using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Generators;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Locks ADN0065 (an activation cache written in a forward without a training-pass guard), its
/// <c>[KeepsActivation]</c> exemption, ADN0068 (an exemption with no reason) and the field baseline.
/// </summary>
public sealed class ActivationCacheGuardAnalyzerTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Field)] public sealed class ScratchAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class KeepsActivationAttribute : Attribute
    {
        public KeepsActivationAttribute(string reason) { Reason = reason; }
        public string Reason { get; }
    }
}
public sealed class Tensor<T>
{
    public Tensor<T> Add(Tensor<T> other) => this;
}
internal static class InferenceMode
{
    public static bool IsActive => false;
}
public abstract class LayerBase
{
    protected bool IsTrainingMode { get; set; }
    protected bool UseAuxiliaryLoss { get; set; }
    protected bool ShouldCacheForBackward => IsTrainingMode && !InferenceMode.IsActive;
    protected void SaveForBackward<TValue>(ref TValue field, TValue value) where TValue : class
    {
        field = ShouldCacheForBackward ? value : null;
    }
}";

    private static ImmutableArray<MetadataReference> BaseReferences()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location))
                continue;
            references.Add(global::AiDotNet.Tests.Generators.CachedMetadataReference.FromFile(assembly.Location));
        }
        return references.ToImmutableArray();
    }

    private static async Task<ImmutableArray<Diagnostic>> RunAsync(
        string source, IEnumerable<string>? baseline = null, string assemblyName = "AiDotNet")
    {
        var compilation = CSharpCompilation.Create(
            assemblyName,
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        var analyzer = new ActivationCacheGuardAnalyzer(baseline ?? Array.Empty<string>(), "TestBaseline");
        return await compilation.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(analyzer))
            .GetAnalyzerDiagnosticsAsync();
    }

    private static string Layer(string members, string forwardBody) => $@"
using System.Collections.Generic;
using AiDotNet.Attributes;
public sealed class Subject : LayerBase
{{
{members}
    public Tensor<double> Forward(Tensor<double> input)
    {{
        var weights = input;
{forwardBody}
        return input;
    }}
}}";

    private static IEnumerable<Diagnostic> Unguarded(ImmutableArray<Diagnostic> diagnostics)
        => diagnostics.Where(d => d.Id == "ADN0065");

    // Mutation: return false from IsBackwardCacheField, or drop the ReportDiagnostic call.
    [Fact]
    public async Task UnguardedField_IsReportedAsError()
    {
        var diagnostics = await RunAsync(Layer(
            "    private Tensor<double> _lastInput;",
            "        _lastInput = input;"));

        var diagnostic = Assert.Single(Unguarded(diagnostics));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("_lastInput", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    // Mutation: delete the "KeepsActivation(field) is { } keep && HasReason(keep)" skip in AnalyzeMethod.
    [Fact]
    public async Task KeepsActivationWithReason_SuppressesTheField()
    {
        var diagnostics = await RunAsync(Layer(
            "    [KeepsActivation(\"Recurrent memory carried to the next frame.\")] private Tensor<double> _lastHidden;",
            "        _lastHidden = input;"));

        Assert.Empty(Unguarded(diagnostics));
        Assert.DoesNotContain(diagnostics, d => d.Id == "ADN0068");
    }

    // Mutation: drop the IsNullOrWhiteSpace test from HasReason (a blank reason then exempts the field
    // and ADN0065 disappears), or remove the SymbolKind.Field action (ADN0068 disappears).
    [Theory]
    [InlineData("\"\"")]
    [InlineData("\"   \"")]
    [InlineData("null")]
    public async Task KeepsActivationWithoutReason_IsReportedAndDoesNotExempt(string reason)
    {
        var diagnostics = await RunAsync(Layer(
            $"    [KeepsActivation({reason})] private Tensor<double> _lastHidden;",
            "        _lastHidden = input;"));

        var misuse = Assert.Single(diagnostics.Where(d => d.Id == "ADN0068"));
        Assert.Contains("_lastHidden", misuse.GetMessage(), StringComparison.Ordinal);
        Assert.Single(Unguarded(diagnostics));
    }

    // Verifies the existing behaviour: a SaveForBackward call is not a raw write. Mutation: treat a field
    // passed by ref to any invocation in a forward as a write.
    [Fact]
    public async Task SaveForBackward_IsNotReported()
    {
        var diagnostics = await RunAsync(Layer(
            "    private Tensor<double> _lastInput;",
            "        SaveForBackward(ref _lastInput, input);"));

        Assert.Empty(Unguarded(diagnostics));
    }

    // Verifies the existing guards. Mutation: return false from IsGuarded, or drop the local-initializer walk
    // (the second case).
    [Theory]
    [InlineData("        if (ShouldCacheForBackward) _lastInput = input;")]
    [InlineData("        bool cacheBwd = ShouldCacheForBackward;\n        if (cacheBwd) { _lastInput = input; }")]
    public async Task ShouldCacheForBackwardIf_IsNotReported(string body)
    {
        Assert.Empty(Unguarded(await RunAsync(Layer("    private Tensor<double> _lastInput;", body))));
    }

    // The else-branch of a guard runs exactly when no backward is coming. Mutation: drop the
    // "child != ifStatement.Statement" test in IsGuarded, so any enclosing guard if counts.
    [Fact]
    public async Task WriteInElseBranchOfGuard_IsReported()
    {
        var diagnostics = await RunAsync(Layer(
            "    private Tensor<double> _lastInput;",
            "        if (ShouldCacheForBackward) { _lastInput = input; }\n        else { _lastInput = weights; }"));

        Assert.Single(Unguarded(diagnostics));
    }

    // A negated guard runs exactly when no backward is coming. Mutation: drop the IsNegatedWithin check in
    // IsGuardCondition, so any mention of a ShouldCache* name counts as a guard.
    [Theory]
    [InlineData("        if (!ShouldCacheForBackward) _lastInput = input;")]
    [InlineData("        if (!this.ShouldCacheForBackward) _lastInput = input;")]
    [InlineData("        if (ShouldCacheForBackward == false) _lastInput = input;")]
    [InlineData("        if (!(IsTrainingMode && ShouldCacheForBackward)) _lastInput = input;")]
    public async Task NegatedGuard_IsReported(string body)
    {
        Assert.Single(Unguarded(await RunAsync(Layer("    private Tensor<double> _lastInput;", body))));
    }

    // Mutation: make IsNegatedWithin treat every comparison as a negation.
    [Fact]
    public async Task GuardComparedWithTrue_IsNotReported()
    {
        Assert.Empty(Unguarded(await RunAsync(Layer(
            "    private Tensor<double> _lastInput;",
            "        if (ShouldCacheForBackward == true) _lastInput = input;"))));
    }

    // Mutation: delete the IsGuardedValue check in AnalyzeMethod.
    [Fact]
    public async Task ShouldCacheForBackwardTernary_IsNotReported()
    {
        var diagnostics = await RunAsync(Layer(
            "    private Tensor<double> _lastInput;",
            "        _lastInput = ShouldCacheForBackward ? input : null;"));

        Assert.Empty(Unguarded(diagnostics));
    }

    // Mutation: drop the IsNullLike(conditional.WhenFalse) requirement in IsGuardedValue; the ternary then
    // counts as a guard although it stores a tensor on both branches.
    [Fact]
    public async Task TernaryThatStoresOnBothBranches_IsReported()
    {
        var diagnostics = await RunAsync(Layer(
            "    private Tensor<double> _lastInput;",
            "        _lastInput = ShouldCacheForBackward ? input : weights;"));

        Assert.Single(Unguarded(diagnostics));
    }

    // Mutation: delete the "requiresTraining && excludesInference" return in IsGuardCondition.
    [Theory]
    [InlineData("        _lastAttentionScores = UseAuxiliaryLoss && IsTrainingMode && !InferenceMode.IsActive\n            ? weights\n            : null;")]
    [InlineData("        if (UseAuxiliaryLoss && IsTrainingMode && !InferenceMode.IsActive) { _lastAttentionScores = weights; }")]
    [InlineData("        if (IsTrainingMode && !InferenceMode.IsActive)\n        {\n            _lastAttentionScores = weights;\n        }")]
    [InlineData("        bool keep = UseAuxiliaryLoss && (this.IsTrainingMode && !(InferenceMode.IsActive));\n        if (keep) _lastAttentionScores = weights;")]
    public async Task AuxiliaryLossTrainingGuard_IsNotReported(string body)
    {
        Assert.Empty(Unguarded(await RunAsync(Layer("    private Tensor<double> _lastAttentionScores;", body))));
    }

    // A training-pass guard must both require training AND exclude an inference scope, conjunctively.
    // Mutation: accept IsTrainingMode alone, or collect conjuncts through || as well as &&.
    [Theory]
    [InlineData("        if (UseAuxiliaryLoss && IsTrainingMode) _lastAttentionScores = weights;")]
    [InlineData("        if (!InferenceMode.IsActive) _lastAttentionScores = weights;")]
    [InlineData("        if (IsTrainingMode || !InferenceMode.IsActive) _lastAttentionScores = weights;")]
    public async Task IncompleteTrainingGuard_IsReported(string body)
    {
        Assert.Single(Unguarded(await RunAsync(Layer("    private Tensor<double> _lastAttentionScores;", body))));
    }

    // Mutation: remove the EndsWith("Out") name test from IsBackwardCacheField.
    [Fact]
    public async Task OutSuffixField_IsReportedUnlessGuarded()
    {
        Assert.Single(Unguarded(await RunAsync(Layer(
            "    private Tensor<double> _attnOut;",
            "        _attnOut = weights;"))));
        Assert.Empty(Unguarded(await RunAsync(Layer(
            "    private Tensor<double> _attnOut;",
            "        if (ShouldCacheForBackward) _attnOut = weights;"))));
    }

    // Mutation: delete the InvocationExpressionSyntax ("Add") branch in AnalyzeMethod.
    [Fact]
    public async Task AddIntoListField_IsReportedUnlessGuarded()
    {
        const string members = "    private readonly List<Tensor<double>> _layerOutputs = new List<Tensor<double>>();";
        var unguarded = Assert.Single(Unguarded(await RunAsync(Layer(members, "        _layerOutputs.Add(weights);"))));
        Assert.Contains("_layerOutputs", unguarded.GetMessage(), StringComparison.Ordinal);

        Assert.Empty(Unguarded(await RunAsync(Layer(
            members, "        if (ShouldCacheForBackward) _layerOutputs.Add(weights);"))));
    }

    // Tensor<T>.Add is arithmetic, not storage. Mutation: remove the IsCollection filter on Add.
    [Fact]
    public async Task TensorArithmeticAddOnCacheField_IsNotReported()
    {
        var diagnostics = await RunAsync(Layer(
            "    private Tensor<double> _lastOutput = new Tensor<double>();",
            "        var sum = _lastOutput.Add(input);"));

        Assert.Empty(Unguarded(diagnostics));
    }

    // A value cannot pin a tensor. Mutation: remove the IsValueType / TypeParameter filter.
    [Fact]
    public async Task ValueTypedCacheNamedField_IsNotReported()
    {
        var diagnostics = await RunAsync(Layer(
            "    private int _lastBatchSize;",
            "        _lastBatchSize = 3;"));

        Assert.Empty(Unguarded(diagnostics));
    }

    // Mutation: delete the "_baseline.Contains(key)" skip (the baselined field is then reported).
    [Fact]
    public async Task BaselinedField_IsNotReported_ButANewFieldIs()
    {
        var diagnostics = await RunAsync(
            Layer(
                "    private Tensor<double> _lastInput;\n    private Tensor<double> _lastHidden;",
                "        _lastInput = input;\n        _lastHidden = input;"),
            baseline: new[] { "Subject._lastInput" });

        var diagnostic = Assert.Single(Unguarded(diagnostics));
        Assert.Contains("_lastHidden", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(diagnostics, d => d.Id == "ADNDEF004");
    }

    // The ratchet: a baselined field that no longer violates fails the build until its line is removed.
    // Mutation: delete the compilation-end action. (BaselinedField_IsNotReported_ButANewFieldIs catches the
    // converse, an over-eager ratchet: recording the key only after the baseline skip, or dropping the
    // "!violating.ContainsKey(e)" filter, makes a baselined violation look stale.)
    [Fact]
    public async Task StaleBaselineEntry_IsReported()
    {
        var diagnostics = await RunAsync(
            Layer("    private Tensor<double> _lastInput;", "        if (ShouldCacheForBackward) _lastInput = input;"),
            baseline: new[] { "Subject._lastInput" });

        var stale = Assert.Single(diagnostics.Where(d => d.Id == "ADNDEF004"));
        Assert.Equal(DiagnosticSeverity.Error, stale.Severity);
        Assert.Contains("Subject._lastInput", stale.GetMessage(), StringComparison.Ordinal);
        Assert.Contains("TestBaseline", stale.GetMessage(), StringComparison.Ordinal);
    }

    // The rule describes the library only. Mutation: remove the AssemblyName check.
    [Fact]
    public async Task OtherAssemblies_AreNotAnalyzed()
    {
        var diagnostics = await RunAsync(
            Layer("    private Tensor<double> _lastInput;", "        _lastInput = input;"),
            assemblyName: "AiDotNetTests");

        Assert.Empty(diagnostics);
    }
}
