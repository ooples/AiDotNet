using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.Guidance;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 3 guidance methods and control models.
/// </summary>
public class ControlModelContractTests : DiffusionUnitTestBase
{
    #region Guidance Method Tests

    [Fact(Timeout = 120000)]
    public async Task PerturbedAttentionGuidance_DefaultConstructor_CreatesValid()
    {
        var guidance = new PerturbedAttentionGuidance<float>();

        Assert.NotNull(guidance);
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionGuidance_DefaultConstructor_CreatesValid()
    {
        var guidance = new SelfAttentionGuidance<float>();

        Assert.NotNull(guidance);
    }

    [Fact(Timeout = 120000)]
    public async Task DynamicCFGScheduler_DefaultConstructor_CreatesValid()
    {
        var scheduler = new DynamicCFGScheduler<float>();

        Assert.NotNull(scheduler);
    }

    [Fact(Timeout = 120000)]
    public async Task RescaledCFG_DefaultConstructor_CreatesValid()
    {
        var cfg = new RescaledCFG<float>();

        Assert.NotNull(cfg);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptiveProjectedGuidance_DefaultConstructor_CreatesValid()
    {
        var guidance = new AdaptiveProjectedGuidance<float>();

        Assert.NotNull(guidance);
    }

    #endregion

    #region New Control Model Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task ControlNetPlusPlusModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetPlusPlusModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    // HeavyTimeout (#1706): Flux backbone (~12B params ~= 48 GB fp32) — constructing it alone exceeds
    // the 16 GB PR runner, which SIGTERM-cancelled the 03b shard. Runs in the nightly HeavyTimeout lane.
    [Trait("Category", "HeavyTimeout")]
    public async Task ControlNetFluxModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetFluxModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    // HeavyTimeout (#1706): SD3 MMDiT backbone (billions of params) — constructing it alone exceeds the
    // 16 GB PR runner, which SIGTERM-cancelled the 03b shard. Runs in the nightly HeavyTimeout lane.
    [Trait("Category", "HeavyTimeout")]
    public async Task ControlNetSD3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetSD3Model<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetUnionProModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetUnionProModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterPlusModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterPlusModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterFaceIDPlusModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterFaceIDPlusModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetLiteModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetLiteModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetQRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetQRModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ReferenceOnlyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ReferenceOnlyModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task StyleAlignedModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StyleAlignedModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetTileModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetTileModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetInpaintingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetInpaintingModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNeXtModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNeXtModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlARModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlARModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    // HeavyTimeout (#1706): Flux backbone (~12B params ~= 48 GB fp32) — constructing it alone exceeds
    // the 16 GB PR runner, which SIGTERM-cancelled the 03b shard. Runs in the nightly HeavyTimeout lane.
    [Trait("Category", "HeavyTimeout")]
    public async Task ControlNetPlusPlusFluxModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetPlusPlusFluxModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion

    #region Clone Contract Tests

    [Fact(Timeout = 120000)]
    public async Task ControlNetPlusPlusModel_Clone_CreatesIndependentCopy()
    {
        var model = new ControlNetPlusPlusModel<float>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, (int)clone.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    // HeavyTimeout (#1706): Clone constructs a second ~12B-param Flux model — exceeds the 16 GB PR
    // runner, which SIGTERM-cancelled the 03b shard. Runs in the nightly HeavyTimeout lane.
    [Trait("Category", "HeavyTimeout")]
    public async Task ControlNetFluxModel_Clone_CreatesIndependentCopy()
    {
        var model = new ControlNetFluxModel<float>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, (int)clone.ParameterCount);
    }

    #endregion

    #region Missing Model Coverage Tests

    [Fact(Timeout = 120000)]
    public async Task IPAdapterModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterModel<float>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion
}
