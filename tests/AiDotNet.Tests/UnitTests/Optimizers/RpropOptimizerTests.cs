using System;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins <see cref="RpropOptimizer{T, TInput, TOutput}"/> to the pseudocode of Riedmiller and Braun (1993).
/// </summary>
/// <remarks>
/// <para>
/// Rprop's three cases are easy to implement in a way that looks right and trains fine but is not Rprop. The two
/// specific traps are (a) still taking a step on the reversal branch, and (b) forgetting to zero the stored
/// gradient there, which makes one overshoot count as two and halves the step twice. Both are targeted directly
/// below with exact expected values, because both leave a working optimizer behind when they are wrong.
/// </para>
/// </remarks>
public class RpropOptimizerTests
{
    private const double Delta0 = 0.1;
    private const double EtaPlus = 1.2;
    private const double EtaMinus = 0.5;
    private const double MinStep = 1e-6;
    private const double MaxStep = 50.0;

    private static RpropOptimizerOptions<double, Matrix<double>, Vector<double>> Options()
        => new()
        {
            InitialStepSize = Delta0,
            EtaPlus = EtaPlus,
            EtaMinus = EtaMinus,
            MinStepSize = MinStep,
            MaxStepSize = MaxStep,
        };

    private static RpropOptimizer<double, Matrix<double>, Vector<double>> CreateOptimizer()
        => new(null, Options());

    /// <summary>
    /// The defining property: the size of the step depends only on Delta, never on the size of the gradient.
    /// </summary>
    /// <remarks>
    /// A gradient of 1e-6 and a gradient of 1e+6 must produce exactly the same movement. This is what makes
    /// Rprop immune to vanishing and exploding gradients, and it is the single assertion that separates it from
    /// every other optimizer in the library.
    /// </remarks>
    [Theory]
    [InlineData(1e-6)]
    [InlineData(1.0)]
    [InlineData(1e6)]
    public void FirstStepMovesByDeltaZero_RegardlessOfGradientMagnitude(double gradientMagnitude)
    {
        var optimizer = CreateOptimizer();
        var updated = optimizer.UpdateParameters(
            new Vector<double>(new[] { 0.0 }), new Vector<double>(new[] { gradientMagnitude }));

        Assert.Equal(-Delta0, updated[0], 12);
    }

    /// <summary>
    /// A gradient that keeps its sign grows the step by eta+ each time, so the moves are a geometric series.
    /// </summary>
    [Fact]
    public void ConsistentGradientSignGrowsTheStepByEtaPlus()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 0.0 });
        var gradient = new Vector<double>(new[] { 2.0 });

        double expectedDelta = Delta0;
        for (int t = 1; t <= 6; t++)
        {
            var next = optimizer.UpdateParameters(parameters, gradient);
            // Step 1 hits the "either gradient was zero" branch (prev is 0), which moves but does not grow.
            double delta = t == 1 ? Delta0 : (expectedDelta *= EtaPlus);
            Assert.Equal(parameters[0] - delta, next[0], 12);
            parameters = next;
        }
    }

    /// <summary>
    /// On a sign reversal the paper shrinks the step and takes NO move at all. An implementation that shrinks and
    /// then still steps would look almost identical in aggregate but is a different algorithm.
    /// </summary>
    [Fact]
    public void SignReversalShrinksTheStepAndMovesNothing()
    {
        var optimizer = CreateOptimizer();

        var afterFirst = optimizer.UpdateParameters(
            new Vector<double>(new[] { 0.0 }), new Vector<double>(new[] { 1.0 }));
        Assert.Equal(-Delta0, afterFirst[0], 12);

        var afterReversal = optimizer.UpdateParameters(afterFirst, new Vector<double>(new[] { -1.0 }));

        Assert.Equal(afterFirst[0], afterReversal[0], 12);

        var stepSizes = optimizer.GetStepSizes();
        Assert.NotNull(stepSizes);
        Assert.Equal(Delta0 * EtaMinus, stepSizes![0], 12);
    }

    /// <summary>
    /// After a reversal the stored gradient is zeroed, so the NEXT step is treated as a fresh sign rather than as
    /// a continuation. This is the subtlest line in the paper and the easiest to drop.
    /// </summary>
    /// <remarks>
    /// Gradients +1, -1, -1. With the zeroing, step 3 sees a product of 0, takes the "held" branch, and moves by
    /// the already-halved 0.05 without changing it. Without the zeroing, step 3 would see (-1)*(-1) &gt; 0, grow to
    /// 0.06, and move by that instead. The two differ by exactly one factor of eta+, so the assertion separates
    /// them cleanly.
    /// </remarks>
    [Fact]
    public void ReversalZeroesTheStoredGradient_SoTheNextStepIsNotTreatedAsAContinuation()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 0.0 });

        parameters = optimizer.UpdateParameters(parameters, new Vector<double>(new[] { 1.0 }));
        var beforeThird = optimizer.UpdateParameters(parameters, new Vector<double>(new[] { -1.0 }));
        var afterThird = optimizer.UpdateParameters(beforeThird, new Vector<double>(new[] { -1.0 }));

        double halved = Delta0 * EtaMinus;                 // 0.05
        Assert.Equal(beforeThird[0] + halved, afterThird[0], 12);

        var stepSizes = optimizer.GetStepSizes();
        Assert.NotNull(stepSizes);
        Assert.Equal(halved, stepSizes![0], 12);           // NOT halved * EtaPlus

        // Guard the assertion itself: the two candidate answers must actually be distinguishable here.
        Assert.NotEqual(halved, halved * EtaPlus, 6);
    }

    /// <summary>
    /// The step size saturates at Delta_max rather than compounding without bound.
    /// </summary>
    [Fact]
    public void StepSizeSaturatesAtTheMaximum()
    {
        var optimizer = new RpropOptimizer<double, Matrix<double>, Vector<double>>(
            null, new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialStepSize = Delta0,
                EtaPlus = EtaPlus,
                EtaMinus = EtaMinus,
                MinStepSize = MinStep,
                MaxStepSize = 1.0,
            });

        var parameters = new Vector<double>(new[] { 0.0 });
        var gradient = new Vector<double>(new[] { 1.0 });
        for (int t = 0; t < 40; t++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        var stepSizes = optimizer.GetStepSizes();
        Assert.NotNull(stepSizes);
        Assert.Equal(1.0, stepSizes![0], 12);
    }

    /// <summary>
    /// A gradient whose sign flips every step drives the step size to Delta_min. This is exactly the failure mode
    /// that makes Rprop unusable on mini-batches, so it is pinned rather than left as folklore.
    /// </summary>
    [Fact]
    public void AlternatingGradientSignsCollapseTheStepToTheMinimum()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 0.0 });

        for (int t = 0; t < 200; t++)
        {
            double g = (t % 2 == 0) ? 1.0 : -1.0;
            parameters = optimizer.UpdateParameters(parameters, new Vector<double>(new[] { g }));
        }

        var stepSizes = optimizer.GetStepSizes();
        Assert.NotNull(stepSizes);
        Assert.Equal(MinStep, stepSizes![0], 12);
    }

    /// <summary>
    /// Independent weights adapt independently — one weight's oscillation must not shrink another's step.
    /// </summary>
    [Fact]
    public void StepSizesAdaptPerWeightIndependently()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 0.0, 0.0 });

        // Element 0 keeps its sign; element 1 alternates.
        for (int t = 0; t < 6; t++)
        {
            double flip = (t % 2 == 0) ? 1.0 : -1.0;
            parameters = optimizer.UpdateParameters(parameters, new Vector<double>(new[] { 1.0, flip }));
        }

        var stepSizes = optimizer.GetStepSizes();
        Assert.NotNull(stepSizes);
        Assert.True(stepSizes![0] > Delta0, $"the consistent weight's step should have grown, was {stepSizes[0]}");
        Assert.True(stepSizes[1] < Delta0, $"the oscillating weight's step should have shrunk, was {stepSizes[1]}");
    }

    /// <summary>
    /// All five hyperparameters must be set explicitly on Extras. In particular Delta_0 must be the paper's 0.1,
    /// not the extras' own 0.01 default — leaving it unset would start the fused path from a different step size
    /// than the eager path, which is the exact class of silent divergence this PR exists to close.
    /// </summary>
    [Fact]
    public void FusedSpecSetsEveryHyperparameterExplicitly()
    {
        var optimizer = CreateOptimizer();

        Assert.True(((IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.Rprop, config.Type);

        Assert.NotNull(config.Extras);
        Assert.Equal((float)EtaPlus, config.Extras!.RpropEtaPlus, 6);
        Assert.Equal((float)EtaMinus, config.Extras.RpropEtaMinus, 6);
        Assert.Equal((float)MinStep, config.Extras.RpropStepMin, 9);
        Assert.Equal((float)MaxStep, config.Extras.RpropStepMax, 3);
        Assert.Equal((float)Delta0, config.Extras.RpropInitialStep, 6);
        Assert.NotEqual(0.01f, config.Extras.RpropInitialStep);
    }

    /// <summary>
    /// Rprop has no learning rate, and the fused kernel takes no lr argument, so a configured schedule would be
    /// silently ignored on the compiled path. Declining is the honest response.
    /// </summary>
    [Fact]
    public void FusedSpecDeclines_WhenALearningRateScheduleIsConfigured()
    {
        var options = Options();
        options.LearningRateScheduler = new ExponentialLRScheduler(0.1, 0.95);
        var optimizer = new RpropOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.False(((IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out _),
            "Rprop fused with a learning-rate schedule the kernel has no learning rate to apply it to.");
    }

    [Theory]
    [InlineData(1.0, EtaMinus, Delta0)]      // EtaPlus must exceed 1
    [InlineData(EtaPlus, 1.0, Delta0)]       // EtaMinus must be below 1
    [InlineData(EtaPlus, EtaMinus, 1e-9)]    // Delta_0 below MinStepSize
    [InlineData(EtaPlus, EtaMinus, 1e9)]     // Delta_0 above MaxStepSize
    public void ConstructorRejectsHyperparametersTheAlgorithmIsNotDefinedFor(
        double etaPlus, double etaMinus, double initialStep)
    {
        var options = new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            EtaPlus = etaPlus,
            EtaMinus = etaMinus,
            InitialStepSize = initialStep,
            MinStepSize = MinStep,
            MaxStepSize = MaxStep,
        };

        Assert.Throws<ArgumentOutOfRangeException>(
            () => new RpropOptimizer<double, Matrix<double>, Vector<double>>(null, options));
    }

    /// <summary>
    /// The move is -sign(g_effective)*Delta and the optimizer holds both terms afterwards, so the inverse is
    /// exact — including on weights that reversed and therefore did not move at all.
    /// </summary>
    [Fact]
    public void ReverseUpdateExactlyInvertsTheStep_IncludingWeightsThatDidNotMove()
    {
        var optimizer = CreateOptimizer();
        var start = new Vector<double>(new[] { 1.0, -2.0, 0.5 });

        // Prime the stored gradient so the second step reverses on element 1 only.
        var primed = optimizer.UpdateParameters(start, new Vector<double>(new[] { 1.0, 1.0, 1.0 }));

        var gradient = new Vector<double>(new[] { 1.0, -1.0, 1.0 });
        var updated = optimizer.UpdateParameters(primed, gradient);
        var restored = optimizer.ReverseUpdate(updated, gradient);

        for (int i = 0; i < primed.Length; i++)
        {
            Assert.Equal(primed[i], restored[i], 12);
        }

        // Element 1 must genuinely have been the no-move case, or this test proves less than it looks.
        Assert.Equal(primed[1], updated[1], 12);
    }

    /// <summary>
    /// The step sizes and the remembered gradients ARE Rprop's learned state, so a round trip must carry both.
    /// </summary>
    [Fact]
    public void SerializationRoundTripPreservesStepSizesAndRememberedGradients()
    {
        var original = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 1.0, -1.0 });

        for (int t = 0; t < 5; t++)
        {
            double flip = (t % 3 == 0) ? -1.0 : 1.0;
            parameters = original.UpdateParameters(parameters, new Vector<double>(new[] { 1.0, flip }));
        }

        var restored = CreateOptimizer();
        restored.Deserialize(original.Serialize());

        var originalSteps = original.GetStepSizes();
        var restoredSteps = restored.GetStepSizes();
        Assert.NotNull(originalSteps);
        Assert.NotNull(restoredSteps);
        for (int i = 0; i < originalSteps!.Length; i++)
        {
            Assert.Equal(originalSteps[i], restoredSteps![i], 12);
        }

        // The next step depends on the remembered gradient too, so this checks both halves of the state.
        var nextGradient = new Vector<double>(new[] { 1.0, -1.0 });
        var fromOriginal = original.UpdateParameters(parameters, nextGradient);
        var fromRestored = restored.UpdateParameters(parameters, nextGradient);
        for (int i = 0; i < fromOriginal.Length; i++)
        {
            Assert.Equal(fromOriginal[i], fromRestored[i], 12);
        }
    }
}
