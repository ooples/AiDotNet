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
        var guidance = new PerturbedAttentionGuidance<double>();

        Assert.NotNull(guidance);
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionGuidance_DefaultConstructor_CreatesValid()
    {
        var guidance = new SelfAttentionGuidance<double>();

        Assert.NotNull(guidance);
    }

    [Fact(Timeout = 120000)]
    public async Task DynamicCFGScheduler_DefaultConstructor_CreatesValid()
    {
        var scheduler = new DynamicCFGScheduler<double>();

        Assert.NotNull(scheduler);
    }

    [Fact(Timeout = 120000)]
    public async Task RescaledCFG_DefaultConstructor_CreatesValid()
    {
        var cfg = new RescaledCFG<double>();

        Assert.NotNull(cfg);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptiveProjectedGuidance_DefaultConstructor_CreatesValid()
    {
        var guidance = new AdaptiveProjectedGuidance<double>();

        Assert.NotNull(guidance);
    }

    #endregion

    #region New Control Model Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task ControlNetPlusPlusModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetPlusPlusModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetFluxModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetFluxModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetSD3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetSD3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetUnionProModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetUnionProModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterPlusModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterPlusModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterFaceIDPlusModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterFaceIDPlusModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetLiteModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetLiteModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetQRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetQRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ReferenceOnlyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ReferenceOnlyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task StyleAlignedModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StyleAlignedModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetTileModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetTileModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetInpaintingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetInpaintingModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNeXtModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNeXtModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlARModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlARModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetPlusPlusFluxModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetPlusPlusFluxModel<double>();

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
        var model = new ControlNetPlusPlusModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetFluxModel_Clone_CreatesIndependentCopy()
    {
        var model = new ControlNetFluxModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    #endregion

    #region Missing Model Coverage Tests

    [Fact(Timeout = 120000)]
    public async Task IPAdapterModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion
}
