using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phases 4, 6, and 7: T2I models, fast generation, schedulers, and VAEs.
/// </summary>
public class FastGenContractTests
{
    #region Phase 4 — New T2I Models

    [Fact]
    public void Flux2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Flux2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact]
    public void SANAModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SANAModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(32, model.LatentChannels);
    }

    [Fact]
    public void StableDiffusion35Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableDiffusion35Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact]
    public void LuminaImage2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new LuminaImage2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void HiDreamModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HiDreamModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void CogView4Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new CogView4Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void PlaygroundV3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new PlaygroundV3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void Imagen3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Imagen3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void MeissonicModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MeissonicModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void RecraftV3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new RecraftV3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void Ideogram3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Ideogram3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void MidJourneyV7Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new MidJourneyV7Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6c — New VAEs

    [Fact]
    public void DeepCompressionVAE_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new DeepCompressionVAE<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
        Assert.True(vae.DownsampleFactor >= 32);
    }

    [Fact]
    public void EQVAEModel_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new EQVAEModel<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
    }

    [Fact]
    public void LiteVAEModel_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new LiteVAEModel<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
    }

    [Fact]
    public void SDXLVAEModel_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new SDXLVAEModel<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
    }

    #endregion

    #region Phase 6d — Consistency Models

    [Fact]
    public void ImprovedConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ImprovedConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void EasyConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new EasyConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void MultiStepConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MultiStepConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void MultistepLCModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MultistepLCModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void TrainingEfficientLCM_DefaultConstructor_CreatesValidModel()
    {
        var model = new TrainingEfficientLCM<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6e — Adversarial Distillation

    [Fact]
    public void SDXLTurboModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDXLTurboModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void SDXLLightningModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDXLLightningModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void HyperSDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HyperSDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void DMD2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new DMD2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SenseFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SenseFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SANASprintModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SANASprintModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SwiftBrushModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SwiftBrushModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void FlashDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FlashDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6f — Trajectory/Score

    [Fact]
    public void PCMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PCMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void TCDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TCDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void InstaFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new InstaFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void FlowMapModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FlowMapModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SCottModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SCottModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void PeRFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PeRFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SiDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SiDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SiDDiTModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SiDDiTModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 7a — Architecture-Specific Fast

    [Fact]
    public void SD3TurboModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SD3TurboModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact]
    public void FluxSchnellModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FluxSchnellModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact]
    public void SD3FlashModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SD3FlashModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact]
    public void Flux2SchnellModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Flux2SchnellModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact]
    public void PixArtDeltaLCMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PixArtDeltaLCMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 7b — Hybrid/Emerging

    [Fact]
    public void TransfusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TransfusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void MARModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MARModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void ARDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ARDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void AutoRegressiveMaskedDiffusion_DefaultConstructor_CreatesValidModel()
    {
        var model = new AutoRegressiveMaskedDiffusion<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6a — New Schedulers

    [Fact]
    public void RectifiedFlowScheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new RectifiedFlowScheduler<double>(config);

        Assert.NotNull(scheduler);
        Assert.NotNull(scheduler.Config);
    }

    [Fact]
    public void FlowDPMSolverScheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowDPMSolverScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DPMSolverV3Scheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverV3Scheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void SASolverScheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new SASolverScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    #endregion

    #region Clone Tests - Representative Fast Gen Models

    [Fact]
    public void SDXLTurboModel_Clone_CreatesIndependentCopy()
    {
        var model = new SDXLTurboModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact]
    public void FluxSchnellModel_Clone_CreatesIndependentCopy()
    {
        var model = new FluxSchnellModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact]
    public void ImprovedConsistencyModel_Clone_CreatesIndependentCopy()
    {
        var model = new ImprovedConsistencyModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    #endregion

    #region GetParameters/SetParameters Round-trip

    [Fact]
    public void SDXLTurboModel_GetSetParameters_RoundTrips()
    {
        var model = new SDXLTurboModel<double>();

        var parameters = model.GetParameters();
        Assert.True(parameters.Length > 0);

        model.SetParameters(parameters);
        var retrieved = model.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    [Fact]
    public void Flux2Model_GetSetParameters_RoundTrips()
    {
        var model = new Flux2Model<double>();

        var parameters = model.GetParameters();
        Assert.True(parameters.Length > 0);

        model.SetParameters(parameters);
        var retrieved = model.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    #endregion

    #region GetModelMetadata Tests

    [Fact]
    public void SDXLTurboModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new SDXLTurboModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    [Fact]
    public void FluxSchnellModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new FluxSchnellModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    [Fact]
    public void SANAModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new SANAModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    #endregion
}
