using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phases 4, 6, and 7: T2I models, fast generation, schedulers, and VAEs.
/// </summary>
// Shares one xUnit collection with NewConditionerContractTests so the two classes' tests run
// sequentially (tests within a collection never run in parallel). The foundation-scale FLUX/MMDiT
// round-trip + clone tests materialize multi-GB FP32 models; serializing keeps at most one resident
// at a time so the 16 GB CI runner does not OOM under parallel execution.
[Collection("FoundationScaleSerial")]
public class FastGenContractTests : DiffusionUnitTestBase
{
    #region Phase 4 — New T2I Models

    [Fact(Timeout = 120000)]
    public async Task Flux2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Flux2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task SANAModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SANAModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(32, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task StableDiffusion35Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableDiffusion35Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task LuminaImage2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new LuminaImage2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task HiDreamModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HiDreamModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task CogView4Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new CogView4Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task PlaygroundV3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new PlaygroundV3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task Imagen3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Imagen3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task MeissonicModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MeissonicModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task RecraftV3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new RecraftV3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task Ideogram3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Ideogram3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task MidJourneyV7Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new MidJourneyV7Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6c — New VAEs

    [Fact(Timeout = 120000)]
    public async Task DeepCompressionVAE_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new DeepCompressionVAE<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
        Assert.True(vae.DownsampleFactor >= 32);
    }

    [Fact(Timeout = 120000)]
    public async Task EQVAEModel_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new EQVAEModel<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task LiteVAEModel_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new LiteVAEModel<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SDXLVAEModel_DefaultConstructor_CreatesValidVAE()
    {
        var vae = new SDXLVAEModel<double>();

        Assert.NotNull(vae);
        Assert.True(vae.ParameterCount > 0);
    }

    #endregion

    #region Phase 6d — Consistency Models

    [Fact(Timeout = 120000)]
    public async Task ImprovedConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ImprovedConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task EasyConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new EasyConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiStepConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MultiStepConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task MultistepLCModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MultistepLCModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task TrainingEfficientLCM_DefaultConstructor_CreatesValidModel()
    {
        var model = new TrainingEfficientLCM<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6e — Adversarial Distillation

    [Fact(Timeout = 120000)]
    public async Task SDXLTurboModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDXLTurboModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SDXLLightningModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDXLLightningModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task HyperSDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HyperSDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task DMD2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new DMD2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SenseFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SenseFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SANASprintModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SANASprintModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SwiftBrushModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SwiftBrushModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task FlashDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FlashDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6f — Trajectory/Score

    [Fact(Timeout = 120000)]
    public async Task PCMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PCMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task TCDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TCDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task InstaFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new InstaFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task FlowMapModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FlowMapModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SCottModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SCottModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task PeRFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PeRFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SiDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SiDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SiDDiTModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SiDDiTModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 7a — Architecture-Specific Fast

    [Fact(Timeout = 120000)]
    public async Task SD3TurboModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SD3TurboModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task FluxSchnellModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FluxSchnellModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task SD3FlashModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SD3FlashModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task Flux2SchnellModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Flux2SchnellModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtDeltaLCMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PixArtDeltaLCMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 7b — Hybrid/Emerging

    [Fact(Timeout = 120000)]
    public async Task TransfusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TransfusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task MARModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MARModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task ARDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ARDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task AutoRegressiveMaskedDiffusion_DefaultConstructor_CreatesValidModel()
    {
        var model = new AutoRegressiveMaskedDiffusion<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Phase 6a — New Schedulers

    [Fact(Timeout = 120000)]
    public async Task RectifiedFlowScheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new RectifiedFlowScheduler<double>(config);

        Assert.NotNull(scheduler);
        Assert.NotNull(scheduler.Config);
    }

    [Fact(Timeout = 120000)]
    public async Task FlowDPMSolverScheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowDPMSolverScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact(Timeout = 120000)]
    public async Task DPMSolverV3Scheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverV3Scheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact(Timeout = 120000)]
    public async Task SASolverScheduler_DefaultConstructor_CreatesValidScheduler()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new SASolverScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    #endregion

    #region Clone Tests - Representative Fast Gen Models

    [Fact(Timeout = 120000)]
    public async Task SDXLTurboModel_Clone_CreatesIndependentCopy()
    {
        // FP32: SDXLTurbo's ~348M params at FP64 require ~2.7 GB for a single
        // Vector<T>, plus 2× that during Clone (source + clone). CI hit OOM
        // at the contiguous Vector allocation. FP32 halves both costs and
        // matches the production-canonical precision for SDXL Turbo weights.
        var model = new SDXLTurboModel<float>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, (int)clone.ParameterCount);
    }

    // FP32 (production-canonical for FLUX): a materialized ~12B-param FluxSchnell at FP64 would dwarf
    // the 16 GB CI runner. Clone is lazy-preserving (a never-forwarded model materializes nothing), so
    // this stays cheap; the timeout is a generous foundation-scale ceiling.
    [Fact(Timeout = 600000)]
    public async Task FluxSchnellModel_Clone_CreatesIndependentCopy()
    {
        await Task.Yield(); // async suspension point so [Fact(Timeout)] can actually enforce the ceiling
        var model = new FluxSchnellModel<float>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        // Compare as long: FluxSchnell exceeds int.MaxValue parameters, so an (int) cast would overflow.
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task ImprovedConsistencyModel_Clone_CreatesIndependentCopy()
    {
        var model = new ImprovedConsistencyModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, (int)clone.ParameterCount);
    }

    #endregion

    #region GetParameters/SetParameters Round-trip

    [Fact(Timeout = 120000)]
    public async Task SDXLTurboModel_GetSetParameters_RoundTrips()
    {
        // FP32: same rationale as SDXLTurboModel_Clone_CreatesIndependentCopy.
        // GetParameters allocates a single contiguous Vector<T> sized to all
        // ~348M params; FP64 requires 2.7 GB which OOMs at the LOH boundary.
        var model = new SDXLTurboModel<float>();

        var parameters = model.GetParameters();
        Assert.True(parameters.Length > 0);

        model.SetParameters(parameters);
        var retrieved = model.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    // FP32 (production-canonical) so a materialized foundation-scale model fits the 16 GB CI runner —
    // FP64 (~34 GB) would OOM; serialized via the collection so only one foundation-scale model is
    // resident at a time. Timeout sized for a streaming round-trip over billions of parameters.
    // HeavyTimeout (#1715): Flux 2 is foundation-scale (~6.5 B params / ~25 GB fp32). The OOM is fixed —
    // GetParameterChunks now streams to disk with a bounded resident set + lossless write-back, so the
    // round-trip is memory-safe (measured: ~8 GB resident peak vs the prior OOM). But round-tripping
    // ~25 GB across disk three times inherently exceeds the per-test budget on the 16 GB runner.
    // Excluded from the default gate, tracked in the nightly HeavyTimeout lane; graduates back when
    // streaming IO is fast enough. (Inherent-runtime bucket, not an OOM — that's resolved.)
    [Trait("Category", "HeavyTimeout")]
    [Fact(Timeout = 600000)]
    public async Task Flux2Model_GetSetParameters_RoundTrips()
    {
        await Task.Yield();
        var model = new Flux2Model<float>();

        // Flux 2 is foundation-scale: a flat GetParameters()/SetParameters() round-trip overflows the
        // int-bounded Vector<T>/List<T> length and OOMs. The allocation-free streaming chunk API (#1624)
        // yields the resident weight tensors by reference, so the round-trip never builds a flat
        // aggregate; AssertParameterChunksRoundTrip writes/reads them in place.
        AssertParameterChunksRoundTrip(model.GetParameterChunks, model.SetParameterChunks);
    }

    #endregion

    #region GetModelMetadata Tests

    [Fact(Timeout = 120000)]
    public async Task SDXLTurboModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new SDXLTurboModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task FluxSchnellModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new FluxSchnellModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task SANAModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new SANAModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    #endregion

    #region Gap Analysis — OSDS Model

    [Fact(Timeout = 120000)]
    public async Task OSDSModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new OSDSModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion

    #region Gap Analysis — 3D Score Distillation Methods

    [Fact(Timeout = 120000)]
    public async Task ScoreDistillationSampling_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var sds = new AiDotNet.Diffusion.Distillation.ScoreDistillationSampling<double>(teacher);

        Assert.NotNull(sds);
        Assert.Equal(100.0, sds.GuidanceScale);
    }

    [Fact(Timeout = 120000)]
    public async Task VariationalScoreDistillation_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var vsd = new AiDotNet.Diffusion.Distillation.VariationalScoreDistillation<double>(teacher);

        Assert.NotNull(vsd);
        Assert.Equal(7.5, vsd.GuidanceScale);
        Assert.Equal(4, vsd.LoRARank);
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyDistillationSampling_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var csd = new AiDotNet.Diffusion.Distillation.ConsistencyDistillationSampling<double>(teacher);

        Assert.NotNull(csd);
        Assert.Equal(50.0, csd.GuidanceScale);
        Assert.Equal(1.0, csd.ConsistencyWeight);
    }

    [Fact(Timeout = 120000)]
    public async Task IntervalScoreMatching_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var ism = new AiDotNet.Diffusion.Distillation.IntervalScoreMatching<double>(teacher);

        Assert.NotNull(ism);
        Assert.Equal(7.5, ism.GuidanceScale);
        Assert.Equal(1, ism.IntervalSteps);
    }

    [Fact(Timeout = 120000)]
    public async Task DenoisedScoreDistillation_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var dsd = new AiDotNet.Diffusion.Distillation.DenoisedScoreDistillation<double>(teacher);

        Assert.NotNull(dsd);
        Assert.Equal(7.5, dsd.GuidanceScale);
        Assert.Equal(50, dsd.DenoisingSteps);
    }

    [Fact(Timeout = 120000)]
    public async Task UnifiedDistillationSampling_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var uds = new AiDotNet.Diffusion.Distillation.UnifiedDistillationSampling<double>(teacher);

        Assert.NotNull(uds);
        Assert.Equal(1.0, uds.PretrainedWeight);
        Assert.Equal(0.0, uds.ParticleWeight);
        Assert.Equal(1.0, uds.NoiseWeight);
    }

    [Fact(Timeout = 120000)]
    public async Task RewardScoreDistillation_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var rsd = new AiDotNet.Diffusion.Distillation.RewardScoreDistillation<double>(teacher);

        Assert.NotNull(rsd);
        Assert.Equal(100.0, rsd.GuidanceScale);
        Assert.Equal(10.0, rsd.RewardWeight);
    }

    [Fact(Timeout = 120000)]
    public async Task SemanticScoreDistillation_Constructor_CreatesValid()
    {
        var teacher = new StableDiffusion15Model<double>();
        var ssd = new AiDotNet.Diffusion.Distillation.SemanticScoreDistillation<double>(teacher);

        Assert.NotNull(ssd);
        Assert.Equal(100.0, ssd.GuidanceScale);
        Assert.Equal(1.0, ssd.SemanticWeight);
        Assert.Equal(1.0, ssd.AppearanceWeight);
    }

    #endregion

    #region Missing Model Coverage Tests

    [Fact(Timeout = 120000)]
    public async Task ConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion
}
