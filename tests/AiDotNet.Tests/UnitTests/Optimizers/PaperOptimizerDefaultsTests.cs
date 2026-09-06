using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests for how a model's <c>[PaperOptimizer]</c> recipe is selected (#1928).
/// </summary>
/// <remarks>
/// <para>
/// These cover selection, which is the part with branching. That the selected recipe actually
/// produces the paper's optimizer is asserted against REAL model types in the population batch,
/// because a declaration can only be attached to a type at compile time — a synthetic fixture here
/// could not prove that a shipped model is wired.
/// </para>
/// </remarks>
public class PaperOptimizerDefaultsTests
{
    private sealed class Undeclared { }

    [PaperOptimizer(OptimizerKind.SgdMomentum, LearningRate = 0.1, Momentum = 0.9, WeightDecay = 1e-4,
                    Source = "Synthetic fixture, not a real paper")]
    private sealed class DeclaresSgdMomentum { }

    // A declaration carrying no values at all still identifies the optimizer, which is itself the
    // most consequential part of the recipe -- it decides which algorithm runs.
    [PaperOptimizer(OptimizerKind.Adam, Source = "Synthetic fixture")]
    private sealed class DeclaresOptimizerOnly { }

    [PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 0.5, Source = "Synthetic fixture")]
    private sealed class DeclaresUnspecified { }

    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 9e-9, Source = "fixture: default row")]
    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Variant = "Tiny", Source = "fixture: Table 8")]
    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Variant = "Huge", Source = "fixture: Table 8")]
    private sealed class VariantModel : IPaperOptimizerVariant
    {
        public VariantModel(string? variant) => PaperOptimizerVariant = variant;
        public string? PaperOptimizerVariant { get; }
    }

    [Fact]
    public void AModelDeclaringNothing_ResolvesToNoRecipe()
    {
        // The behaviour-neutrality guarantee: with no declaration the factory returns nothing and
        // the call site keeps the optimizer it already constructed. Without this, migrating 592
        // call sites would silently change how every model trains.
        Assert.Null(PaperOptimizerFactory.Find(new Undeclared()));
    }

    [Fact]
    public void ANullModel_IsSafe()
    {
        Assert.Null(PaperOptimizerFactory.Find(null));
    }

    [Fact]
    public void TheDeclaredOptimizerAndItsScalarsAreResolved()
    {
        var recipe = PaperOptimizerFactory.Find(new DeclaresSgdMomentum());

        Assert.NotNull(recipe);
        // The optimizer identity is the point: this model's paper trains with SGD-momentum, and an
        // earlier design would have discarded the whole recipe because the model builds Adam.
        Assert.Equal(OptimizerKind.SgdMomentum, recipe!.Optimizer);
        Assert.Equal(0.1, recipe.LearningRate, precision: 12);
        Assert.Equal(0.9, recipe.Momentum, precision: 12);
        Assert.Equal(1e-4, recipe.WeightDecay, precision: 12);
    }

    [Fact]
    public void ARecipeNamingOnlyTheOptimizer_IsStillResolved()
    {
        // Knowing the paper uses Adam rather than AdamW matters even with no numbers attached:
        // AdamW's decoupled decay is applied on every step and is not the same operation as Adam's
        // L2. So a declaration with no scalars is still worth honouring.
        var recipe = PaperOptimizerFactory.Find(new DeclaresOptimizerOnly());

        Assert.NotNull(recipe);
        Assert.Equal(OptimizerKind.Adam, recipe!.Optimizer);
        Assert.False(recipe.DeclaresAnyHyperparameter);
    }

    [Fact]
    public void ARecipeLeftUnspecified_IsIgnored()
    {
        // Unspecified names no algorithm, so there is nothing to build; falling through to the
        // caller's default beats guessing.
        Assert.Null(PaperOptimizerFactory.Find(new DeclaresUnspecified()));
    }

    [Theory]
    [InlineData("Tiny", 1e-4)]
    [InlineData("Huge", 5e-5)]
    public void AVariantKeyedRecipe_SelectsTheMatchingRow(string variant, double expected)
    {
        var recipe = PaperOptimizerFactory.Find(new VariantModel(variant));
        Assert.NotNull(recipe);
        Assert.Equal(expected, recipe!.LearningRate, precision: 12);
    }

    [Fact]
    public void AVariantWithNoRowOfItsOwn_FallsBackToTheUnkeyedRecipe()
    {
        // Partial population is the expected steady state as sizes get filled in one at a time.
        var recipe = PaperOptimizerFactory.Find(new VariantModel("SomeSizeNobodyDeclared"));
        Assert.Equal(9e-9, recipe!.LearningRate, precision: 12);
    }

    [Fact]
    public void AModelExposingNoVariant_UsesTheUnkeyedRecipe()
    {
        var recipe = PaperOptimizerFactory.Find(new VariantModel(null));
        Assert.Equal(9e-9, recipe!.LearningRate, precision: 12);
    }

    [Fact]
    public void UnsetIsNaN_SoAnExplicitZeroIsDistinguishable()
    {
        // The case that matters most. A paper specifying plain Adam declares WeightDecay = 0 and it
        // must stick, because AdamW's own default is 0.01 applied to every parameter on every step.
        // Encoding unset as 0 would silently drop exactly that declaration.
        var zero = new PaperOptimizerAttribute(OptimizerKind.Adam) { WeightDecay = 0.0 };
        var unset = new PaperOptimizerAttribute(OptimizerKind.Adam);

        Assert.True(zero.DeclaresAnyHyperparameter);
        Assert.False(unset.DeclaresAnyHyperparameter);
        Assert.True(double.IsNaN(unset.WeightDecay));
    }

    [Fact]
    public void ScheduleAndClippingArePartOfTheRecipe()
    {
        // The schedule is not an implementation detail: a post-LN transformer without warmup
        // diverges at the same learning rate that works with it. Declaring the rate while dropping
        // the schedule reproduces neither.
        var recipe = new PaperOptimizerAttribute(OptimizerKind.Adam)
        {
            Schedule = LearningRateSchedulerType.LinearWarmup,
            WarmupSteps = 4000,
            MaxGradientNorm = 1.0,
        };

        Assert.True(recipe.DeclaresAnyHyperparameter);
        Assert.Equal(LearningRateSchedulerType.LinearWarmup, recipe.Schedule);
        Assert.Equal(4000, recipe.WarmupSteps);
        Assert.Equal(1.0, recipe.MaxGradientNorm, precision: 12);
    }
}
