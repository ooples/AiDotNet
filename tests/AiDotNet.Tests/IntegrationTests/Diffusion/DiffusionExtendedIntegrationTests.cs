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
        var model = new StableDiffusion15Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableDiffusion2Model_Construction()
    {
        var model = new StableDiffusion2Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableDiffusion3Model_Construction()
    {
        var model = new StableDiffusion3Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SDXLModel_Construction()
    {
        var model = new SDXLModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableCascadeModel_Construction()
    {
        var model = new StableCascadeModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DallE2Model_Construction()
    {
        var model = new DallE2Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DallE3Model_Construction()
    {
        var model = new DallE3Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DeepFloydIFModel_Construction()
    {
        var model = new DeepFloydIFModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ImagenModel_Construction()
    {
        var model = new ImagenModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Imagen2Model_Construction()
    {
        var model = new Imagen2Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task KandinskyModel_Construction()
    {
        var model = new KandinskyModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtModel_Construction()
    {
        var model = new PixArtModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtSigmaModel_Construction()
    {
        var model = new PixArtSigmaModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PixArtDeltaModel_Construction()
    {
        var model = new PixArtDeltaModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Flux1Model_Construction()
    {
        var model = new Flux1Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task HunyuanDiTModel_Construction()
    {
        var model = new HunyuanDiTModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task KolorsModel_Construction()
    {
        var model = new KolorsModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PlaygroundV25Model_Construction()
    {
        var model = new PlaygroundV25Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task RAPHAELModel_Construction()
    {
        var model = new RAPHAELModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task EDiffIModel_Construction()
    {
        var model = new EDiffIModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task OmniGenModel_Construction()
    {
        var model = new OmniGenModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region Video Diffusion Models

    [Fact(Timeout = 120000)]
    public async Task AnimateDiffModel_Construction()
    {
        var model = new AnimateDiffModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task CogVideoModel_Construction()
    {
        var model = new CogVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableVideoDiffusionModel_Construction()
    {
        var model = new StableVideoDiffusion<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenSoraModel_Construction()
    {
        var model = new OpenSoraModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SoraModel_Construction()
    {
        var model = new SoraModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MakeAVideoModel_Construction()
    {
        var model = new MakeAVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task RunwayGenModel_Construction()
    {
        var model = new RunwayGenModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoCrafterModel_Construction()
    {
        var model = new VideoCrafterModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LatteModel_Construction()
    {
        var model = new LatteModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ModelScopeT2VModel_Construction()
    {
        var model = new ModelScopeT2VModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LTXVideoModel_Construction()
    {
        var model = new LTXVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LuminaT2XModel_Construction()
    {
        var model = new AiDotNet.Diffusion.TextToImage.LuminaT2XModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Mochi1Model_Construction()
    {
        var model = new Mochi1Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task HunyuanVideoModel_Construction()
    {
        var model = new HunyuanVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task VeoModel_Construction()
    {
        var model = new VeoModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task KlingModel_Construction()
    {
        var model = new KlingModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task WanVideoModel_Construction()
    {
        var model = new WanVideoModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region Audio Diffusion Models

    [Fact(Timeout = 120000)]
    public async Task AudioLDMModel_Construction()
    {
        var model = new AudioLDMModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task AudioLDM2Model_Construction()
    {
        var model = new AudioLDM2Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffWaveModel_Construction()
    {
        var model = new DiffWaveModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MusicGenModel_Construction()
    {
        var model = new MusicGenModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task RiffusionModel_Construction()
    {
        var model = new RiffusionModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task StableAudioModel_Construction()
    {
        var model = new StableAudioModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task BarkModel_Construction()
    {
        var model = new BarkModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SoundStormModel_Construction()
    {
        var model = new SoundStormModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task JEN1Model_Construction()
    {
        var model = new JEN1Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UdioModel_Construction()
    {
        var model = new UdioModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task VoiceCraftModel_Construction()
    {
        var model = new VoiceCraftModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region 3D Diffusion Models

    [Fact(Timeout = 120000)]
    public async Task ShapEModel_Construction()
    {
        var model = new ShapEModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PointEModel_Construction()
    {
        var model = new PointEModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Zero123Model_Construction()
    {
        var model = new Zero123Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Magic3DModel_Construction()
    {
        var model = new Magic3DModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task TripoSRModel_Construction()
    {
        var model = new TripoSRModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task One2345Model_Construction()
    {
        var model = new One2345Model<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MVDreamModel_Construction()
    {
        var model = new MVDreamModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SyncDreamerModel_Construction()
    {
        var model = new SyncDreamerModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Wonder3DModel_Construction()
    {
        var model = new Wonder3DModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MeshyModel_Construction()
    {
        var model = new MeshyModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LGMModel_Construction()
    {
        var model = new LGMModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Instant3DModel_Construction()
    {
        var model = new Instant3DModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region Control Models

    [Fact(Timeout = 120000)]
    public async Task ControlNetModel_Construction()
    {
        var model = new ControlNetModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetXSModel_Construction()
    {
        var model = new ControlNetXSModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ControlNetUnionModel_Construction()
    {
        var model = new ControlNetUnionModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task T2IAdapterModel_Construction()
    {
        var model = new T2IAdapterModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterModel_Construction()
    {
        var model = new IPAdapterModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task IPAdapterFaceIDModel_Construction()
    {
        var model = new IPAdapterFaceIDModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task InstantIDModel_Construction()
    {
        var model = new InstantIDModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PhotoMakerModel_Construction()
    {
        var model = new PhotoMakerModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UniControlNetModel_Construction()
    {
        var model = new UniControlNetModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region FastGeneration Models

    [Fact(Timeout = 120000)]
    public async Task ConsistencyModel_Construction()
    {
        var model = new ConsistencyModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LatentConsistencyModel_Construction()
    {
        var model = new LatentConsistencyModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SDTurboModel_Construction()
    {
        var model = new SDTurboModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task AuraFlowModel_Construction()
    {
        var model = new AuraFlowModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region SuperResolution Models

    [Fact(Timeout = 120000)]
    public async Task SDUpscalerModel_Construction()
    {
        var model = new SDUpscalerModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region ImageEditing Models

    [Fact(Timeout = 120000)]
    public async Task BlendedDiffusionModel_Construction()
    {
        var model = new BlendedDiffusionModel<double>();
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffEditModel_Construction()
    {
        var model = new DiffEditModel<double>();
        Assert.NotNull(model);
    }

    #endregion
}
