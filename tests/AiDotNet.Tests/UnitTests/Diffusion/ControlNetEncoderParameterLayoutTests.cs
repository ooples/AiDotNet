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
        var copy = (AiDotNet.Diffusion.DiffusionModelBase<float>)AiDotNet.Models.CloneEngine.CopyConfiguration(model);

        bool shared = CopyOnWriteCloneHelper.TryShareTrainableParameters<float>(
            model, copy, out var status, out string mismatch);

        Assert.True(shared, $"Copy-on-write share declined ({status}): {mismatch}");
        Assert.Equal(CopyOnWriteShareStatus.Shared, status);
        Assert.Equal(model.ParameterCount, copy.ParameterCount);
        Assert.Equal(model.ParameterLayout.MaterializedParameterCount, copy.ParameterLayout.MaterializedParameterCount);
    }
}
