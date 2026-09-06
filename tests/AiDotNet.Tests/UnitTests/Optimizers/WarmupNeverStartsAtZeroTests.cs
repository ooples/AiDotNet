using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Guards the defect that declaring a paper's warmup introduced: a first step of exactly zero (#1928).
/// </summary>
/// <remarks>
/// <para>
/// A ramp computed as <c>step / warmup</c> returns zero on step 0, so the first optimizer step moves
/// nothing. Over a paper-length run that is one wasted step. Over a short one it can be most of
/// training, and the model then appears not to learn at all — which is exactly how this surfaced:
/// wav2vec 2.0's generated Training_ShouldChangeParameters began failing with "Parameters did not
/// change after training. Gradients may be zero or learning rate is 0."
/// </para>
/// <para>
/// The failure was real and the declaration caused it, confirmed by reverting the declaration alone
/// and watching the test pass again. The fix was to the code, not the test: the reference
/// implementations index the ramp from one, so its first step is one step's worth of the peak.
/// </para>
/// </remarks>
public class WarmupNeverStartsAtZeroTests
{
    /// <summary>The schedule attached to a built optimizer, read the way the factory writes it.</summary>
    /// <remarks>
    /// By reflection because the scheduler lives on the concrete options type rather than the shared
    /// base the accessor returns — the same reason the factory sets it by reflection.
    /// </remarks>
    private static ILearningRateScheduler? SchedulerOf(object? optimizer)
    {
        var typed = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);
        object options = typed.GetOptions();
        var property = options.GetType().GetProperty("LearningRateScheduler");
        return property?.GetValue(options) as ILearningRateScheduler;
    }

    private static Wav2Vec2<double> Model()
        => new(new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 32));

    [Fact]
    public void ADeclaredWarmupHasAPositiveRateOnItsFirstStep()
    {
        // wav2vec 2.0 declares warmup over the first 8% of updates followed by linear decay, so it
        // exercises the ramp the whole way through the recipe path rather than in isolation.
        var built = PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(Model());

        var scheduler = SchedulerOf(built);

        Assert.NotNull(scheduler);
        Assert.True(scheduler.GetLearningRateAtStep(0) > 0,
            "the first step of a declared warmup must move the parameters; a rate of zero there makes "
            + "a short run look like a model that cannot learn");
    }

    [Fact]
    public void TheRampStillClimbsToThePaperRate()
    {
        // Starting non-zero must not flatten the ramp into a constant: the shape is still a ramp.
        var built = PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(Model());
        var scheduler = SchedulerOf(built);

        Assert.NotNull(scheduler);
        Assert.True(scheduler.GetLearningRateAtStep(0) < scheduler.GetLearningRateAtStep(4));

        var typed = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(built);

        // And it is the paper's BASE rate of 5e-4 it climbs towards, not a library default.
        Assert.Equal(5e-4, typed.GetOptions().InitialLearningRate, precision: 12);
    }

    [Fact]
    public void ARunWithNoRoomToDecayHoldsThePeakAndSaysSo()
    {
        // A decay needs a horizon. Without one, decaySteps goes negative and every post-warmup step
        // returns the floor — a silent zero. Holding the peak is the safe half of the schedule, but
        // it is not the whole schedule, so it must be reported rather than passed off as Exact.
        var recipe = new AiDotNet.Attributes.PaperOptimizerAttribute(Enums.OptimizerKind.Adam)
        {
            LearningRate = 1e-3,
            Schedule = LearningRateSchedulerType.LinearWarmup,
            WarmupSteps = 50,
            PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
            Source = "fixture: warmup longer than the run",
        };

        Assert.Equal(LinearWarmupScheduler.DecayMode.Linear, recipe.PostWarmupDecay);
        Assert.True(recipe.DeclaresAnyHyperparameter);
    }
}
