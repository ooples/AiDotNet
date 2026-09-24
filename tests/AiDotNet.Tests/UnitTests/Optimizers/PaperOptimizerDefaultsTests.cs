using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Moq;
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

    [PaperOptimizer(OptimizerKind.Adam, Source = "Synthetic inherited fixture")]
    private abstract class DeclaringBase { }

    private sealed class InheritingModel : DeclaringBase { }

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

        var resolved = Assert.IsType<PaperOptimizerAttribute>(recipe);
        // The optimizer identity is the point: this model's paper trains with SGD-momentum, and an
        // earlier design would have discarded the whole recipe because the model builds Adam.
        Assert.Equal(OptimizerKind.SgdMomentum, resolved.Optimizer);
        Assert.Equal(0.1, resolved.LearningRate, precision: 12);
        Assert.Equal(0.9, resolved.Momentum, precision: 12);
        Assert.Equal(1e-4, resolved.WeightDecay, precision: 12);
    }

    [Fact]
    public void ARecipeNamingOnlyTheOptimizer_IsStillResolved()
    {
        // Knowing the paper uses Adam rather than AdamW matters even with no numbers attached:
        // AdamW's decoupled decay is applied on every step and is not the same operation as Adam's
        // L2. So a declaration with no scalars is still worth honouring.
        var recipe = PaperOptimizerFactory.Find(new DeclaresOptimizerOnly());

        var resolved = Assert.IsType<PaperOptimizerAttribute>(recipe);
        Assert.Equal(OptimizerKind.Adam, resolved.Optimizer);
        Assert.False(resolved.DeclaresAnyHyperparameter);
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
        var resolved = Assert.IsType<PaperOptimizerAttribute>(recipe);
        Assert.Equal(expected, resolved.LearningRate, precision: 12);
    }

    [Fact]
    public void AVariantWithNoRowOfItsOwn_FallsBackToTheUnkeyedRecipe()
    {
        // Partial population is the expected steady state as sizes get filled in one at a time.
        var recipe = PaperOptimizerFactory.Find(new VariantModel("SomeSizeNobodyDeclared"));
        var resolved = Assert.IsType<PaperOptimizerAttribute>(recipe);
        Assert.Equal(9e-9, resolved.LearningRate, precision: 12);
    }

    [Fact]
    public void AModelExposingNoVariant_UsesTheUnkeyedRecipe()
    {
        var recipe = PaperOptimizerFactory.Find(new VariantModel(null));
        var resolved = Assert.IsType<PaperOptimizerAttribute>(recipe);
        Assert.Equal(9e-9, resolved.LearningRate, precision: 12);
    }

    [Fact]
    public void InheritedDeclaration_IsResolvedAtRuntime()
    {
        var recipe = PaperOptimizerFactory.Find(new InheritingModel());
        var resolved = Assert.IsType<PaperOptimizerAttribute>(recipe);
        Assert.Equal(OptimizerKind.Adam, resolved.Optimizer);
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

    [Theory]
    [InlineData(OptimizerKind.Adam)]
    [InlineData(OptimizerKind.AdamW)]
    [InlineData(OptimizerKind.Sgd)]
    [InlineData(OptimizerKind.SgdMomentum)]
    [InlineData(OptimizerKind.Adam8Bit)]
    [InlineData(OptimizerKind.RmsProp)]
    [InlineData(OptimizerKind.Adagrad)]
    [InlineData(OptimizerKind.Adadelta)]
    [InlineData(OptimizerKind.Adamax)]
    [InlineData(OptimizerKind.Nadam)]
    [InlineData(OptimizerKind.Lamb)]
    [InlineData(OptimizerKind.Lion)]
    [InlineData(OptimizerKind.LBfgs)]
    public void EveryDeclaredOptimizerKind_ConstructsItsRealImplementation(OptimizerKind kind)
    {
        var recipe = new PaperOptimizerAttribute(kind) { Source = "Synthetic fixture" };
        var model = new Mock<IFullModel<double, object, object>>().Object;
        Type expectedType = kind switch
        {
            OptimizerKind.Adam => typeof(AdamOptimizer<double, object, object>),
            OptimizerKind.AdamW => typeof(AdamWOptimizer<double, object, object>),
            OptimizerKind.Sgd => typeof(StochasticGradientDescentOptimizer<double, object, object>),
            OptimizerKind.SgdMomentum => typeof(MomentumOptimizer<double, object, object>),
            OptimizerKind.Adam8Bit => typeof(Adam8BitOptimizer<double, object, object>),
            OptimizerKind.RmsProp => typeof(RootMeanSquarePropagationOptimizer<double, object, object>),
            OptimizerKind.Adagrad => typeof(AdagradOptimizer<double, object, object>),
            OptimizerKind.Adadelta => typeof(AdaDeltaOptimizer<double, object, object>),
            OptimizerKind.Adamax => typeof(AdaMaxOptimizer<double, object, object>),
            OptimizerKind.Nadam => typeof(NadamOptimizer<double, object, object>),
            OptimizerKind.Lamb => typeof(LAMBOptimizer<double, object, object>),
            OptimizerKind.Lion => typeof(LionOptimizer<double, object, object>),
            OptimizerKind.LBfgs => typeof(LBFGSOptimizer<double, object, object>),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
        };

        var optimizer = PaperOptimizerFactory.CreateFromRecipe(model, recipe);

        Assert.IsType(expectedType, optimizer);
    }

    [Fact]
    public void NesterovRecipe_ConstructsNesterovImplementation()
    {
        var recipe = new PaperOptimizerAttribute(OptimizerKind.SgdMomentum)
        {
            Source = "Synthetic fixture",
            UseNesterov = true,
        };
        var model = new Mock<IFullModel<double, object, object>>().Object;

        var optimizer = PaperOptimizerFactory.CreateFromRecipe(model, recipe);

        Assert.IsType<NesterovAcceleratedGradientOptimizer<double, object, object>>(optimizer);
    }

    [Fact]
    public void ExponentialSchedule_PreservesDeclaredMinimumLearningRate()
    {
        var recipe = new PaperOptimizerAttribute(OptimizerKind.Adam)
        {
            Source = "Synthetic fixture",
            LearningRate = 0.1,
            Schedule = LearningRateSchedulerType.Exponential,
            DecayRate = 0.5,
            MinLearningRate = 0.02,
        };
        var model = new Mock<IFullModel<double, object, object>>().Object;

        var optimizer = Assert.IsType<AdamOptimizer<double, object, object>>(
            PaperOptimizerFactory.CreateFromRecipe(model, recipe));
        var options = Assert.IsType<AdamOptimizerOptions<double, object, object>>(optimizer.GetOptions());
        var scheduler = Assert.IsType<ExponentialLRScheduler>(options.LearningRateScheduler);

        Assert.Equal(0.02, scheduler.GetLearningRateAtStep(100), precision: 12);
    }

    [Fact]
    public void WarmupAndExponentialSchedule_AreComposedRatherThanDroppingWarmup()
    {
        var recipe = new PaperOptimizerAttribute(OptimizerKind.Adam)
        {
            Source = "Synthetic fixture",
            LearningRate = 0.1,
            Schedule = LearningRateSchedulerType.Exponential,
            WarmupSteps = 5,
            DecayRate = 0.9,
        };

        var scheduler = PaperOptimizerFactory.BuildScheduler(recipe, recipe.LearningRate);
        var sequential = Assert.IsType<SequentialLRScheduler>(scheduler);

        Assert.IsType<LinearWarmupScheduler>(sequential.Schedulers[0]);
        Assert.IsType<ExponentialLRScheduler>(sequential.Schedulers[1]);
    }

    [Fact]
    public void InvalidOrUnsupportedSchedule_IsSurfaced()
    {
        var missingDecay = new PaperOptimizerAttribute(OptimizerKind.Adam)
        {
            Source = "Synthetic fixture",
            Schedule = LearningRateSchedulerType.Exponential,
        };
        var unsupported = new PaperOptimizerAttribute(OptimizerKind.Adam)
        {
            Source = "Synthetic fixture",
            Schedule = LearningRateSchedulerType.Lambda,
        };

        Assert.Throws<ArgumentException>(() => PaperOptimizerFactory.BuildScheduler(missingDecay, 0.1));
        Assert.Throws<NotSupportedException>(() => PaperOptimizerFactory.BuildScheduler(unsupported, 0.1));
    }

    [Fact]
    public void PaperControlledAdaptiveValues_DoNotDriftAfterConstruction()
    {
        var adamRecipe = new PaperOptimizerAttribute(OptimizerKind.Adam)
        {
            Source = "Synthetic fixture",
            Beta1 = 0.8,
        };
        var adadeltaRecipe = new PaperOptimizerAttribute(OptimizerKind.Adadelta)
        {
            Source = "Synthetic fixture",
            Rho = 0.91,
        };
        var model = new Mock<IFullModel<double, object, object>>().Object;

        var adam = Assert.IsType<AdamOptimizer<double, object, object>>(
            PaperOptimizerFactory.CreateFromRecipe(model, adamRecipe));
        var adadelta = Assert.IsType<AdaDeltaOptimizer<double, object, object>>(
            PaperOptimizerFactory.CreateFromRecipe(model, adadeltaRecipe));
        var adamOptions = Assert.IsType<AdamOptimizerOptions<double, object, object>>(adam.GetOptions());
        var adadeltaOptions = Assert.IsType<AdaDeltaOptimizerOptions<double, object, object>>(adadelta.GetOptions());

        Assert.False(adamOptions.UseAdaptiveBetas);
        Assert.False(adadeltaOptions.UseAdaptiveRho);
    }
}
