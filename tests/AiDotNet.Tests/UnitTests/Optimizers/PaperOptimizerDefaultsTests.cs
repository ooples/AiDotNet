using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests for the #1928 mechanism: a model that constructs its optimizer with no options should
/// train at its paper's settings rather than the optimizer class's generic defaults.
/// </summary>
/// <remarks>
/// These exercise <see cref="PaperOptimizerDefaults.Resolve"/> directly, which is the exact call
/// the four optimizer constructors now make (<c>: base(model, PaperOptimizerDefaults.Resolve(model,
/// options, OptimizerKind.X))</c>), so they cover the real path without standing up a full model.
/// </remarks>
public class PaperOptimizerDefaultsTests
{
    private const double PaperLearningRate = 1.6e-4;
    private const double PaperWeightDecay = 0.05;

    private sealed class Undeclared { }

    [PaperOptimizer(OptimizerKind.AdamW,
                    LearningRate = PaperLearningRate, WeightDecay = PaperWeightDecay,
                    Source = "Synthetic fixture, not a real paper")]
    private sealed class DeclaresAdamW { }

    // A paper specifying plain Adam with NO weight decay. AdamW would otherwise contribute its own
    // decoupled 0.01 to every parameter on every step -- the exact defect commit 1972a510a fixed in
    // SpanBasedNERBase. Declaring zero has to be distinguishable from declaring nothing.
    [PaperOptimizer(OptimizerKind.AdamW, WeightDecay = 0.0, Source = "Synthetic fixture")]
    private sealed class DeclaresZeroWeightDecay { }

    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, Source = "Synthetic fixture")]
    private sealed class DeclaresAdamOnly { }

    private sealed class SizedModel : IPaperOptimizerVariant
    {
        public SizedModel(string? variant) => PaperOptimizerVariant = variant;
        public string? PaperOptimizerVariant { get; }
    }

    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 9e-9, Source = "fixture: default row")]
    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Variant = "Tiny", Source = "fixture: Table 8")]
    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Variant = "Huge", Source = "fixture: Table 8")]
    private sealed class VariantModel : IPaperOptimizerVariant
    {
        public VariantModel(string? variant) => PaperOptimizerVariant = variant;
        public string? PaperOptimizerVariant { get; }
    }

    private static AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>> ResolveAdamW(
        object? model, AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>? caller = null)
        => PaperOptimizerDefaults.Resolve(model, caller, OptimizerKind.AdamW);

    [Fact]
    public void ModelWithNoDeclaration_KeepsTheLibraryDefaults()
    {
        // Control arm. Without this, every assertion below could pass simply because the mechanism
        // overwrites everything it touches.
        var untouched = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        var resolved = ResolveAdamW(new Undeclared());

        Assert.Equal(untouched.InitialLearningRate, resolved.InitialLearningRate);
        Assert.Equal(untouched.WeightDecay, resolved.WeightDecay);
    }

    [Fact]
    public void DeclaredHyperparameters_AreAppliedWhenTheCallerSuppliesNoOptions()
    {
        var resolved = ResolveAdamW(new DeclaresAdamW());

        Assert.Equal(PaperLearningRate, resolved.InitialLearningRate);
        Assert.Equal(PaperWeightDecay, resolved.WeightDecay);

        // And they genuinely differ from the defaults, or the assertion proves nothing.
        var untouched = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        Assert.NotEqual(untouched.InitialLearningRate, resolved.InitialLearningRate);
        Assert.NotEqual(untouched.WeightDecay, resolved.WeightDecay);
    }

    [Fact]
    public void CallerSuppliedOptions_AlwaysWin()
    {
        // ConfigureOptimizer must never be overridden by a paper default; the paper is the default,
        // not the law.
        var caller = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.42,
            WeightDecay = 0.99,
        };

        var resolved = ResolveAdamW(new DeclaresAdamW(), caller);

        Assert.Same(caller, resolved);
        Assert.Equal(0.42, resolved.InitialLearningRate);
        Assert.Equal(0.99, resolved.WeightDecay);
    }

    [Fact]
    public void ADeclarationForADifferentOptimizer_IsNotApplied()
    {
        // A learning rate chosen for Adam is not a learning rate for AdamW. Transplanting values
        // across optimizers would be worse than the default it replaced.
        var untouched = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        var resolved = ResolveAdamW(new DeclaresAdamOnly());

        Assert.Equal(untouched.InitialLearningRate, resolved.InitialLearningRate);
    }

    [Fact]
    public void ExplicitZeroWeightDecay_IsAppliedRatherThanTreatedAsUnset()
    {
        // The case that matters most. AdamW's own default is 0.01, so a paper specifying plain Adam
        // needs to say zero and have it stick. If "unset" were encoded as 0 instead of NaN, this
        // declaration would be silently ignored and the defect would survive the fix.
        var untouched = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        Assert.Equal(0.01, untouched.WeightDecay);

        var resolved = ResolveAdamW(new DeclaresZeroWeightDecay());

        Assert.Equal(0.0, resolved.WeightDecay);
        // Unstated values are left alone -- this declaration says nothing about the learning rate.
        Assert.Equal(untouched.InitialLearningRate, resolved.InitialLearningRate);
    }

    [Theory]
    [InlineData("Tiny", 1e-4)]
    [InlineData("Huge", 5e-5)]
    public void VariantKeyedDeclaration_SelectsTheMatchingRow(string variant, double expected)
    {
        var resolved = ResolveAdamW(new VariantModel(variant));
        Assert.Equal(expected, resolved.InitialLearningRate);
    }

    [Fact]
    public void AVariantWithNoRowOfItsOwn_FallsBackToTheUnkeyedDeclaration()
    {
        // Partial population is the expected steady state: sizes get filled in as papers are read.
        var resolved = ResolveAdamW(new VariantModel("SomeSizeNobodyDeclared"));
        Assert.Equal(9e-9, resolved.InitialLearningRate);
    }

    [Fact]
    public void AModelExposingNoVariant_UsesTheUnkeyedDeclaration()
    {
        var resolved = ResolveAdamW(new VariantModel(null));
        Assert.Equal(9e-9, resolved.InitialLearningRate);
    }

    [Fact]
    public void ANullModel_IsSafeAndChangesNothing()
    {
        // Optimizers accept a null model and have it set later, so resolution must tolerate it.
        var untouched = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        var resolved = ResolveAdamW(model: null);

        Assert.Equal(untouched.InitialLearningRate, resolved.InitialLearningRate);
        Assert.Equal(untouched.WeightDecay, resolved.WeightDecay);
    }

    [Fact]
    public void AnOptionsTypeWithoutTheDeclaredKnob_SkipsItRatherThanThrowing()
    {
        // AdamOptimizerOptions has no WeightDecay. A paper that states one is telling the reader
        // something true that this optimizer cannot express -- information, not a crash.
        var resolved = PaperOptimizerDefaults.Resolve(
            new DeclaresAdamW(),
            (AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>?)null,
            OptimizerKind.AdamW);

        Assert.Equal(PaperLearningRate, resolved.InitialLearningRate);
    }
}
