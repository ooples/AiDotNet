using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins <see cref="AdamWOptimizerOptions{T, TInput, TOutput}.WeightDecayMask"/>.
/// </summary>
/// <remarks>
/// Decoupled decay is applied to the whole flat parameter vector, so without a mask there is no way to
/// express a recipe that exempts particular parameters -- RecurrentGemma (Botev et al., 2024) Section 2
/// exempts the recurrent layers, and the usual transformer recipe exempts biases and normalization
/// gains. These assert the exemption actually reaches the update rather than being a setting that
/// silently does nothing.
/// </remarks>
public class AdamWWeightDecayMaskTests
{
    private static Vector<double> StepOnce(Vector<double>? mask, double weightDecay)
    {
        var options = new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            WeightDecay = weightDecay,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecayMask = mask,
        };
        var optimizer = new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // A ZERO gradient isolates the decay term: with no gradient the Adam update contributes
        // nothing, so whatever moves a parameter is decay and only decay.
        var parameters = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
        var gradient = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        return optimizer.UpdateParameters(parameters, gradient);
    }

    [Fact]
    public void NullMask_DecaysEveryParameter_WhichIsTheBehaviourWithoutTheFeature()
    {
        var updated = StepOnce(mask: null, weightDecay: 0.5);

        for (int i = 0; i < updated.Length; i++)
            Assert.True(updated[i] < 1.0, $"parameter {i} was not decayed: {updated[i]}");
    }

    [Fact]
    public void ZeroMaskEntries_ExemptExactlyThoseParameters()
    {
        // Exempt 0 and 2, decay 1 and 3.
        var mask = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var updated = StepOnce(mask, weightDecay: 0.5);

        Assert.Equal(1.0, updated[0], 12);
        Assert.Equal(1.0, updated[2], 12);
        Assert.True(updated[1] < 1.0, $"parameter 1 should have been decayed, got {updated[1]}");
        Assert.True(updated[3] < 1.0, $"parameter 3 should have been decayed, got {updated[3]}");
    }

    [Fact]
    public void FractionalMaskEntries_ScaleTheDecay()
    {
        var full = StepOnce(new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 }), weightDecay: 0.5);
        var half = StepOnce(new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 }), weightDecay: 0.5);

        // Decay moves the parameter down from 1; halving the mask halves that movement.
        double fullDrop = 1.0 - full[0];
        double halfDrop = 1.0 - half[0];
        Assert.True(fullDrop > 0, "the unmasked run did not decay at all");
        Assert.Equal(fullDrop / 2.0, halfDrop, 12);
    }

    [Fact]
    public void MaskPresent_DeclinesFusedCompilation()
    {
        // The fused config carries decay as a single float, so a masked run cannot be expressed
        // there. Declining is what stops the compiled path decaying parameters the eager path exempts.
        var masked = new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            WeightDecayMask = new Vector<double>(new[] { 0.0, 1.0 }),
        };
        var unmasked = new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>();

        var withMask = (AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null, masked);
        var withoutMask = (AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null, unmasked);

        Assert.False(withMask.TryGetFusedOptimizerConfig(out _),
            "a masked optimizer must not report a fused config");
        Assert.True(withoutMask.TryGetFusedOptimizerConfig(out _),
            "an unmasked optimizer should still fuse exactly as before");
    }
}
