using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.Memory;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Video;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Extended integration tests for Diffusion module covering model construction,
/// DiffusionModelOptions, VAE components, noise predictors, conditioning modules,
/// and all model categories (TextToImage, Video, Audio, 3D, Control, etc.).
/// </summary>
public class DiffusionExtendedIntegrationTests
{
    #region DiffusionModelOptions

    [Fact(Timeout = 120000)]
    public async Task DiffusionModelOptions_DefaultValues()
    {
        var opts = new DiffusionModelOptions<double>();

        Assert.Equal(0.001, opts.LearningRate);
        Assert.Equal(1000, opts.TrainTimesteps);
        Assert.Equal(0.0001, opts.BetaStart);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionModelOptions_SetCustomValues()
    {
        var opts = new DiffusionModelOptions<double>
        {
            LearningRate = 0.0002,
            TrainTimesteps = 500,
            BetaStart = 0.001,
            BetaEnd = 0.02,
            BetaSchedule = BetaSchedule.SquaredCosine
        };

        Assert.Equal(0.0002, opts.LearningRate);
        Assert.Equal(500, opts.TrainTimesteps);
        Assert.Equal(0.001, opts.BetaStart);
        Assert.Equal(0.02, opts.BetaEnd);
        Assert.Equal(BetaSchedule.SquaredCosine, opts.BetaSchedule);
    }

    #endregion

    #region TextToImage Models

    [Fact(Timeout = 120000)]
    public async Task StableDiffusion15Model_Construction()
    {
        // FP32: see KandinskyModel_Construction below for the rationale and
        // commit 1c21d67f9 for the upstream ModelFamily migration of the same
        // model. SD15 paper-faithful defaults OOM standalone at FP64 on a
        // 16 GB CI host.
        var model = new StableDiffusion15Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableDiffusion2Model_Construction()
    {
        var model = new StableDiffusion2Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableDiffusion3Model_Construction()
    {
        var model = new StableDiffusion3Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SDXLModel_Construction()
    {
        var model = new SDXLModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableCascadeModel_Construction()
    {
        // FP32: same rationale as SD15 above (commit 1c21d67f9). StableCascade
        // (Stage C prior + Stage B decoder + Stage A VQGAN) is on the
        // standalone-OOM list at FP64 and additionally surfaced as a local
        // OOM once KandinskyModel's FP32 fix freed up its share of memory
        // (memory-pressure shift).
        var model = new StableCascadeModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DallE2Model_Construction()
    {
        // FP32: same rationale (commit 1c21d67f9). DALL·E 2's paper-faithful
        // ~3.5B-param decoder is on the standalone-OOM-at-FP64 list.
        var model = new DallE2Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DallE3Model_Construction()
    {
        var model = new DallE3Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DeepFloydIFModel_Construction()
    {
        // FP32: same rationale (commit 1c21d67f9). DeepFloyd IF's 3-stage
        // cascade (~4B params total) is on the standalone-OOM-at-FP64 list.
        var model = new DeepFloydIFModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ImagenModel_Construction()
    {
        // FP32: same rationale (commit 1c21d67f9). Imagen's base UNet +
        // super-resolution UNet is on the standalone-OOM-at-FP64 list and
        // additionally surfaced as a local OOM after Kandinsky's FP32 fix
        // freed its memory share (memory-pressure shift).
        var model = new ImagenModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Imagen2Model_Construction()
    {
        var model = new Imagen2Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task KandinskyModel_Construction()
    {
        // FP32 (not FP64) for the same reason as the ModelFamily KandinskyModelTests
        // migrated in commit 1c21d67f9 (`test(diffusion): migrate 12 paper-scale
        // diffusion tests to FP32`): Kandinsky 3.0's paper-faithful defaults
        // (prior + decoder UNets + MoVQ-GAN VAE) instantiate ~16 GB of weight
        // tensors at FP64, which OOMs the CI runner — confirmed locally:
        // `new KandinskyModel<double>()` throws OutOfMemoryException at
        // ConvolutionalLayer.EnsureInitialized while constructing the decoder
        // UNet's level-3 ResBlocks, while `new KandinskyModel<float>()` succeeds
        // in ~45s on the same machine. FP32 is the production-canonical
        // precision for SD/Kandinsky paper checkpoints (FP32 master / FP16
        // working). The other Construction smoke tests in this file are also
        // migrated to <float> to reduce cumulative shard memory pressure and
        // avoid future OOM as the test suite grows.
        var model = new KandinskyModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtModel_Construction()
    {
        var model = new PixArtModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtSigmaModel_Construction()
    {
        var model = new PixArtSigmaModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtDeltaModel_Construction()
    {
        var model = new PixArtDeltaModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Flux1Model_Construction()
    {
        var model = new Flux1Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task HunyuanDiTModel_Construction()
    {
        var model = new HunyuanDiTModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task KolorsModel_Construction()
    {
        var model = new KolorsModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PlaygroundV25Model_Construction()
    {
        var model = new PlaygroundV25Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task RAPHAELModel_Construction()
    {
        var model = new RAPHAELModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task EDiffIModel_Construction()
    {
        var model = new EDiffIModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task OmniGenModel_Construction()
    {
        var model = new OmniGenModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region Video Diffusion Models

    [Fact(Timeout = 120000)]
    public async Task AnimateDiffModel_Construction()
    {
        var model = new AnimateDiffModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task CogVideoModel_Construction()
    {
        var model = new CogVideoModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableVideoDiffusionModel_Construction()
    {
        var model = new StableVideoDiffusion<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenSoraModel_Construction()
    {
        var model = new OpenSoraModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SoraModel_Construction()
    {
        var model = new SoraModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MakeAVideoModel_Construction()
    {
        var model = new MakeAVideoModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task RunwayGenModel_Construction()
    {
        var model = new RunwayGenModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoCrafterModel_Construction()
    {
        var model = new VideoCrafterModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LatteModel_Construction()
    {
        var model = new LatteModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ModelScopeT2VModel_Construction()
    {
        var model = new ModelScopeT2VModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LTXVideoModel_Construction()
    {
        var model = new LTXVideoModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LuminaT2XModel_Construction()
    {
        var model = new AiDotNet.Diffusion.TextToImage.LuminaT2XModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Mochi1Model_Construction()
    {
        var model = new Mochi1Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task HunyuanVideoModel_Construction()
    {
        var model = new HunyuanVideoModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task VeoModel_Construction()
    {
        var model = new VeoModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task KlingModel_Construction()
    {
        var model = new KlingModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task WanVideoModel_Construction()
    {
        var model = new WanVideoModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region Audio Diffusion Models

    [Fact(Timeout = 120000)]
    public async Task AudioLDMModel_Construction()
    {
        var model = new AudioLDMModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task AudioLDM2Model_Construction()
    {
        var model = new AudioLDM2Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffWaveModel_Construction()
    {
        var model = new DiffWaveModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MusicGenModel_Construction()
    {
        // FP32: same rationale as the diffusion family above. MusicGen wasn't
        // on commit 1c21d67f9's standalone-OOM list — that probe covered only
        // the diffusion / VAE families — but the local sweep after Kandinsky
        // FP32 surfaced it as an OOM in the shared diffusion-shard process at
        // FP64. Migrating to FP32 brings it under the same memory budget as
        // the other paper-scale models in this file.
        var model = new MusicGenModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task RiffusionModel_Construction()
    {
        var model = new RiffusionModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableAudioModel_Construction()
    {
        var model = new StableAudioModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task BarkModel_Construction()
    {
        var model = new BarkModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SoundStormModel_Construction()
    {
        var model = new SoundStormModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task JEN1Model_Construction()
    {
        var model = new JEN1Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UdioModel_Construction()
    {
        var model = new UdioModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task VoiceCraftModel_Construction()
    {
        var model = new VoiceCraftModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region 3D Diffusion Models

    [Fact(Timeout = 120000)]
    public async Task ShapEModel_Construction()
    {
        var model = new ShapEModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PointEModel_Construction()
    {
        var model = new PointEModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Zero123Model_Construction()
    {
        var model = new Zero123Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Magic3DModel_Construction()
    {
        var model = new Magic3DModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task TripoSRModel_Construction()
    {
        var model = new TripoSRModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task One2345Model_Construction()
    {
        var model = new One2345Model<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MVDreamModel_Construction()
    {
        var model = new MVDreamModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SyncDreamerModel_Construction()
    {
        var model = new SyncDreamerModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Wonder3DModel_Construction()
    {
        var model = new Wonder3DModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MeshyModel_Construction()
    {
        var model = new MeshyModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LGMModel_Construction()
    {
        var model = new LGMModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Instant3DModel_Construction()
    {
        var model = new Instant3DModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region Control Models

    [Fact(Timeout = 120000)]
    public async Task ControlNetModel_Construction()
    {
        var model = new ControlNetModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetXSModel_Construction()
    {
        var model = new ControlNetXSModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetUnionModel_Construction()
    {
        var model = new ControlNetUnionModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task T2IAdapterModel_Construction()
    {
        var model = new T2IAdapterModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterModel_Construction()
    {
        var model = new IPAdapterModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterFaceIDModel_Construction()
    {
        var model = new IPAdapterFaceIDModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task InstantIDModel_Construction()
    {
        var model = new InstantIDModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PhotoMakerModel_Construction()
    {
        var model = new PhotoMakerModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UniControlNetModel_Construction()
    {
        var model = new UniControlNetModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region FastGeneration Models

    [Fact(Timeout = 120000)]
    public async Task ConsistencyModel_Construction()
    {
        var model = new ConsistencyModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LatentConsistencyModel_Construction()
    {
        var model = new LatentConsistencyModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SDTurboModel_Construction()
    {
        var model = new SDTurboModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task AuraFlowModel_Construction()
    {
        var model = new AuraFlowModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region SuperResolution Models

    [Fact(Timeout = 120000)]
    public async Task SDUpscalerModel_Construction()
    {
        var model = new SDUpscalerModel<float>();
        Assert.NotNull(model);
    }

    #endregion

    #region ImageEditing Models

    [Fact(Timeout = 120000)]
    public async Task BlendedDiffusionModel_Construction()
    {
        var model = new BlendedDiffusionModel<float>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffEditModel_Construction()
    {
        var model = new DiffEditModel<float>();
        Assert.NotNull(model);
    }

    #endregion
}
