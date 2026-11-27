using System.Linq;
using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Helpers;
using AiDotNet.Models.Parameters;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// #2447: the nightly's ControlNetFluxModel_Clone_CreatesIndependentCopy ran past the 10-minute hang
/// limit and was reported as a crashed test host. The ControlNet encoder registered as a plain parameter
/// source, which the manifest takes to be fully materialized, while its down blocks are still lazy. The
/// copy-on-write clone saw a parameter surface wider than the layer state it could share, declined, and
/// fell back to materializing and copying every weight of the 12 B-parameter model.
/// </summary>
public sealed class ControlNetEncoderParameterLayoutTests
{
    [Fact]
    public void Layout_ReportsOnlyTheConvolutionsThatHoldValues()
    {
        var encoder = new ControlNetEncoder<float>(inputChannels: 3, baseChannels: 8, channelMultipliers: new[] { 1, 2 });

        var fresh = Assert.Single(((IParameterLayoutSource)encoder).GetParameterLayout());
        Assert.Equal(encoder.ParameterCount, fresh.ParameterCount);
        Assert.Equal(ParameterReadiness.ShapeResolvedUnmaterialized, fresh.Readiness);
        Assert.InRange(fresh.MaterializedParameterCount, 1, encoder.ParameterCount - 1); // the zero convs only

        encoder.GetParameters();

        var read = Assert.Single(((IParameterLayoutSource)encoder).GetParameterLayout());
        Assert.Equal(ParameterReadiness.Materialized, read.Readiness);
        Assert.Equal(encoder.ParameterCount, read.MaterializedParameterCount);
    }

    [Fact]
    public void Clone_OfAFreshControlNetFluxModel_SharesInsteadOfCopyingEveryWeight()
    {
        var model = new ControlNetFluxModel<float>(
            predictor: new FluxDoubleStreamPredictor<float>(
                inputChannels: 16, hiddenSize: 64, numJointLayers: 1, numSingleLayers: 1,
                numHeads: 4, contextDim: 32, seed: 1),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 8,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 1),
            seed: 1);
        // A fresh model's only values are the zero convolutions, and a fresh copy's are zero too: counts and
        // zeros agree whether or not anything was shared. Give the source values a copy can only have by
        // taking them from it.
        var sourceConv = model.ControlEncoder.ZeroConvolutions[0];
        var marked = Ramp(checked((int)sourceConv.ParameterCount), offset: 0.25f);
        sourceConv.SetParameters(marked);

        var copy = (ControlNetFluxModel<float>)AiDotNet.Models.CloneEngine.CopyConfiguration(model);
        var copyConv = copy.ControlEncoder.ZeroConvolutions[0];
        Assert.NotSame(sourceConv, copyConv);
        Assert.All(copyConv.GetParameters().ToArray(), value => Assert.Equal(0f, value));

        bool shared = CopyOnWriteCloneHelper.TryShareTrainableParameters<float>(
            model, copy, out var status, out string mismatch);

        Assert.True(shared, $"Copy-on-write share declined ({status}): {mismatch}");
        Assert.Equal(CopyOnWriteShareStatus.Shared, status);
        Assert.Equal(model.ParameterCount, copy.ParameterCount);
        Assert.Equal(model.ParameterLayout.MaterializedParameterCount, copy.ParameterLayout.MaterializedParameterCount);

        // The copy holds the source's values...
        Assert.Equal(marked.ToArray(), copyConv.GetParameters().ToArray());

        // ...and neither side's writes reach the other.
        var copyValues = Ramp(marked.Length, offset: -0.75f);
        copyConv.SetParameters(copyValues);
        Assert.Equal(marked.ToArray(), sourceConv.GetParameters().ToArray());

        var sourceValues = Ramp(marked.Length, offset: 1.5f);
        sourceConv.SetParameters(sourceValues);
        Assert.Equal(copyValues.ToArray(), copyConv.GetParameters().ToArray());
        Assert.Equal(sourceValues.ToArray(), sourceConv.GetParameters().ToArray());
    }

    [Fact]
    public void Clone_OfAControlNetFluxModel_CarriesTheSourceValues_AndIsIndependent()
    {
        var model = new ControlNetFluxModel<float>(
            predictor: new FluxDoubleStreamPredictor<float>(
                inputChannels: 16, hiddenSize: 64, numJointLayers: 1, numSingleLayers: 1,
                numHeads: 4, contextDim: 32, seed: 1),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 8,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 1),
            seed: 1);
        var sourceConv = model.ControlEncoder.ZeroConvolutions[0];
        var marked = Ramp(checked((int)sourceConv.ParameterCount), offset: 0.25f);
        sourceConv.SetParameters(marked);

        var clone = Assert.IsType<ControlNetFluxModel<float>>(model.Clone());
        var cloneConv = clone.ControlEncoder.ZeroConvolutions[0];

        Assert.NotSame(sourceConv, cloneConv);
        Assert.Equal(marked.ToArray(), cloneConv.GetParameters().ToArray());

        var cloneValues = Ramp(marked.Length, offset: -0.75f);
        cloneConv.SetParameters(cloneValues);
        Assert.Equal(marked.ToArray(), sourceConv.GetParameters().ToArray());

        var sourceValues = Ramp(marked.Length, offset: 1.5f);
        sourceConv.SetParameters(sourceValues);
        Assert.Equal(cloneValues.ToArray(), cloneConv.GetParameters().ToArray());

    }

    [Fact]
    public void Clone_OfAMaterializedControlNetFluxModel_HoldsEveryValueOfTheSource()
    {
        var model = NewSmallModel(seed: 1);
        var expected = model.GetParameters();   // every lazy weight now holds its values

        var clone = Assert.IsType<ControlNetFluxModel<float>>(model.Clone());
        var actual = clone.GetParameters();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i]) Assert.Fail($"Parameter {i}: source {expected[i]}, clone {actual[i]}.");
        }

        // And a write to the clone's whole surface does not reach the source.
        var written = new AiDotNet.Tensors.LinearAlgebra.Vector<float>(actual.Length);
        for (int i = 0; i < written.Length; i++) written[i] = 2f + (i % 251) * 0.003f;
        clone.SetParameters(written);
        var after = model.GetParameters();
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != after[i]) Assert.Fail($"Parameter {i} of the source changed from {expected[i]} to {after[i]}.");
        }
    }

    private static ControlNetFluxModel<float> NewSmallModel(int seed) => new(
        predictor: new FluxDoubleStreamPredictor<float>(
            inputChannels: 16, hiddenSize: 64, numJointLayers: 1, numSingleLayers: 1,
            numHeads: 4, contextDim: 32, seed: seed),
        vae: new StandardVAE<float>(
            inputChannels: 3, latentChannels: 16, baseChannels: 8,
            channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: seed),
        seed: seed);

    private static AiDotNet.Tensors.LinearAlgebra.Vector<float> Ramp(int length, float offset)
    {
        var values = new AiDotNet.Tensors.LinearAlgebra.Vector<float>(length);
        for (int i = 0; i < length; i++) values[i] = offset + (i % 251) * 0.003f;
        return values;
    }
}
