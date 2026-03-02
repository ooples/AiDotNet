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

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Extended integration tests for Diffusion module covering model construction,
/// DiffusionModelOptions, VAE components, noise predictors, conditioning modules,
/// and all model categories (TextToImage, Video, Audio, 3D, Control, etc.).
/// </summary>
public class DiffusionExtendedIntegrationTests
{
    #region DiffusionModelOptions

    [Fact]
    public void DiffusionModelOptions_DefaultValues()
    {
        var opts = new DiffusionModelOptions<double>();

        Assert.Equal(0.001, opts.LearningRate);
        Assert.Equal(1000, opts.TrainTimesteps);
        Assert.Equal(0.0001, opts.BetaStart);
    }

    [Fact]
    public void DiffusionModelOptions_SetCustomValues()
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

    [Fact]
    public void StableDiffusion15Model_Construction()
    {
        var model = new StableDiffusion15Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void StableDiffusion2Model_Construction()
    {
        var model = new StableDiffusion2Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void StableDiffusion3Model_Construction()
    {
        var model = new StableDiffusion3Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void SDXLModel_Construction()
    {
        var model = new SDXLModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void StableCascadeModel_Construction()
    {
        var model = new StableCascadeModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void DallE2Model_Construction()
    {
        var model = new DallE2Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void DallE3Model_Construction()
    {
        var model = new DallE3Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void DeepFloydIFModel_Construction()
    {
        var model = new DeepFloydIFModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void ImagenModel_Construction()
    {
        var model = new ImagenModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Imagen2Model_Construction()
    {
        var model = new Imagen2Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void KandinskyModel_Construction()
    {
        var model = new KandinskyModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void PixArtModel_Construction()
    {
        var model = new PixArtModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void PixArtSigmaModel_Construction()
    {
        var model = new PixArtSigmaModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void PixArtDeltaModel_Construction()
    {
        var model = new PixArtDeltaModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Flux1Model_Construction()
    {
        var model = new Flux1Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void HunyuanDiTModel_Construction()
    {
        var model = new HunyuanDiTModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void KolorsModel_Construction()
    {
        var model = new KolorsModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void PlaygroundV25Model_Construction()
    {
        var model = new PlaygroundV25Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void RAPHAELModel_Construction()
    {
        var model = new RAPHAELModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void EDiffIModel_Construction()
    {
        var model = new EDiffIModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void OmniGenModel_Construction()
    {
        var model = new OmniGenModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region Video Diffusion Models

    [Fact]
    public void AnimateDiffModel_Construction()
    {
        var model = new AnimateDiffModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void CogVideoModel_Construction()
    {
        var model = new CogVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void StableVideoDiffusionModel_Construction()
    {
        var model = new StableVideoDiffusion<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void OpenSoraModel_Construction()
    {
        var model = new OpenSoraModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void SoraModel_Construction()
    {
        var model = new SoraModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void MakeAVideoModel_Construction()
    {
        var model = new MakeAVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void RunwayGenModel_Construction()
    {
        var model = new RunwayGenModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void VideoCrafterModel_Construction()
    {
        var model = new VideoCrafterModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void LatteModel_Construction()
    {
        var model = new LatteModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void ModelScopeT2VModel_Construction()
    {
        var model = new ModelScopeT2VModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void LTXVideoModel_Construction()
    {
        var model = new LTXVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void LuminaT2XModel_Construction()
    {
        var model = new LuminaT2XModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Mochi1Model_Construction()
    {
        var model = new Mochi1Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void HunyuanVideoModel_Construction()
    {
        var model = new HunyuanVideoModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void VeoModel_Construction()
    {
        var model = new VeoModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void KlingModel_Construction()
    {
        var model = new KlingModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void WanVideoModel_Construction()
    {
        var model = new WanVideoModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region Audio Diffusion Models

    [Fact]
    public void AudioLDMModel_Construction()
    {
        var model = new AudioLDMModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void AudioLDM2Model_Construction()
    {
        var model = new AudioLDM2Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void DiffWaveModel_Construction()
    {
        var model = new DiffWaveModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void MusicGenModel_Construction()
    {
        var model = new MusicGenModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void RiffusionModel_Construction()
    {
        var model = new RiffusionModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void StableAudioModel_Construction()
    {
        var model = new StableAudioModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void BarkModel_Construction()
    {
        var model = new BarkModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void SoundStormModel_Construction()
    {
        var model = new SoundStormModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void JEN1Model_Construction()
    {
        var model = new JEN1Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void UdioModel_Construction()
    {
        var model = new UdioModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void VoiceCraftModel_Construction()
    {
        var model = new VoiceCraftModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region 3D Diffusion Models

    [Fact]
    public void ShapEModel_Construction()
    {
        var model = new ShapEModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void PointEModel_Construction()
    {
        var model = new PointEModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Zero123Model_Construction()
    {
        var model = new Zero123Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Magic3DModel_Construction()
    {
        var model = new Magic3DModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void TripoSRModel_Construction()
    {
        var model = new TripoSRModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void One2345Model_Construction()
    {
        var model = new One2345Model<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void MVDreamModel_Construction()
    {
        var model = new MVDreamModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void SyncDreamerModel_Construction()
    {
        var model = new SyncDreamerModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Wonder3DModel_Construction()
    {
        var model = new Wonder3DModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void MeshyModel_Construction()
    {
        var model = new MeshyModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void LGMModel_Construction()
    {
        var model = new LGMModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void Instant3DModel_Construction()
    {
        var model = new Instant3DModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region Control Models

    [Fact]
    public void ControlNetModel_Construction()
    {
        var model = new ControlNetModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void ControlNetXSModel_Construction()
    {
        var model = new ControlNetXSModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void ControlNetUnionModel_Construction()
    {
        var model = new ControlNetUnionModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void T2IAdapterModel_Construction()
    {
        var model = new T2IAdapterModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void IPAdapterModel_Construction()
    {
        var model = new IPAdapterModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void IPAdapterFaceIDModel_Construction()
    {
        var model = new IPAdapterFaceIDModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void InstantIDModel_Construction()
    {
        var model = new InstantIDModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void PhotoMakerModel_Construction()
    {
        var model = new PhotoMakerModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void UniControlNetModel_Construction()
    {
        var model = new UniControlNetModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region FastGeneration Models

    [Fact]
    public void ConsistencyModel_Construction()
    {
        var model = new ConsistencyModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void LatentConsistencyModel_Construction()
    {
        var model = new LatentConsistencyModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void SDTurboModel_Construction()
    {
        var model = new SDTurboModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void AuraFlowModel_Construction()
    {
        var model = new AuraFlowModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region SuperResolution Models

    [Fact]
    public void SDUpscalerModel_Construction()
    {
        var model = new SDUpscalerModel<double>();
        Assert.NotNull(model);
    }

    #endregion

    #region ImageEditing Models

    [Fact]
    public void BlendedDiffusionModel_Construction()
    {
        var model = new BlendedDiffusionModel<double>();
        Assert.NotNull(model);
    }

    [Fact]
    public void DiffEditModel_Construction()
    {
        var model = new DiffEditModel<double>();
        Assert.NotNull(model);
    }

    #endregion
}
