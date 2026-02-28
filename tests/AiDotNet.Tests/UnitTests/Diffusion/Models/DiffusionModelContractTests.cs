using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests verifying that all diffusion models follow the golden constructor pattern
/// and implement the required interfaces correctly.
/// </summary>
public class DiffusionModelContractTests
{
    #region Golden Constructor Pattern Tests - New Models

    [Fact]
    public void StableDiffusion15Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableDiffusion15Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void StableDiffusion2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableDiffusion2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void StableDiffusion3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableDiffusion3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels); // SD3 uses 16 latent channels
    }

    [Fact]
    public void Flux1Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Flux1Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels); // FLUX uses 16 latent channels
    }

    [Fact]
    public void DallE2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new DallE2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void KandinskyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new KandinskyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void ImagenModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ImagenModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void DeepFloydIFModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DeepFloydIFModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Golden Constructor Pattern Tests - Pre-existing Models

    [Fact]
    public void SDXLModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDXLModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void DallE3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new DallE3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void DreamFusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DreamFusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void ControlNetModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region IParameterizable Contract Tests

    [Fact]
    public void StableDiffusion15Model_GetParameters_ReturnsNonEmptyVector()
    {
        var model = new StableDiffusion15Model<double>();

        var parameters = model.GetParameters();

        Assert.True(parameters.Length > 0, "Parameters should not be empty");
    }

    [Fact]
    public void StableDiffusion15Model_ParameterCount_MatchesGetParametersLength()
    {
        var model = new StableDiffusion15Model<double>();

        var parameters = model.GetParameters();

        Assert.Equal(model.ParameterCount, parameters.Length);
    }

    [Fact]
    public void StableDiffusion15Model_SetParameters_DoesNotThrow()
    {
        var model = new StableDiffusion15Model<double>();

        var parameters = model.GetParameters();
        model.SetParameters(parameters);

        // Verify parameters were set by getting them again
        var retrievedParams = model.GetParameters();
        Assert.Equal(parameters.Length, retrievedParams.Length);
    }

    #endregion

    #region IDiffusionModel Contract Tests

    [Fact]
    public void StableDiffusion15Model_Clone_CreatesIndependentCopy()
    {
        var model = new StableDiffusion15Model<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact]
    public void StableDiffusion15Model_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new StableDiffusion15Model<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    #endregion

    #region Specialized Model Tests

    [Fact]
    public void CogVideoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CogVideoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void Magic3DModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Magic3DModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void JEN1Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new JEN1Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void StableCascadeModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableCascadeModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void T2IAdapterModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new T2IAdapterModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void SDTurboModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDTurboModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void LatentConsistencyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new LatentConsistencyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact]
    public void PlaygroundV25Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new PlaygroundV25Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Text-to-Image Models (Tasks #42-#52)

    [Fact]
    public void PixArtSigmaModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PixArtSigmaModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void PixArtDeltaModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PixArtDeltaModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void Imagen2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Imagen2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void RAPHAELModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new RAPHAELModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void EDiffIModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new EDiffIModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void HunyuanDiTModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HunyuanDiTModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void KolorsModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new KolorsModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void AuraFlowModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new AuraFlowModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void LuminaT2XModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new AiDotNet.Diffusion.TextToImage.LuminaT2XModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void OmniGenModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new OmniGenModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    #endregion

    #region Control/Adapter Models (Tasks #53-#58)

    [Fact]
    public void ControlNetXSModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetXSModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void InstantIDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new InstantIDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void PhotoMakerModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PhotoMakerModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void IPAdapterFaceIDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IPAdapterFaceIDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void ControlNetUnionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ControlNetUnionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void UniControlNetModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new UniControlNetModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    #endregion

    #region Image Editing Models (Tasks #59-#68)

    [Fact]
    public void InstructPix2PixModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new InstructPix2PixModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void PromptToPromptModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PromptToPromptModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void NullTextInversionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new NullTextInversionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void DiffEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DiffEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void LEDITSPPModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new LEDITSPPModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void MagicBrushModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MagicBrushModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void ImagicModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ImagicModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void SDEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void PaintByExampleModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PaintByExampleModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void BlendedDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new BlendedDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    #endregion

    #region Super-Resolution Models (Tasks #69-#74)

    [Fact]
    public void SDUpscalerModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDUpscalerModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void RealESRGANModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new RealESRGANModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void StableSRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableSRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void DiffBIRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DiffBIRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void SUPIRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SUPIRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact]
    public void UpscaleAVideoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new UpscaleAVideoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    #endregion

    #region Video Models (Tasks #75-#86)

    [Fact]
    public void SoraModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SoraModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void ModelScopeT2VModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ModelScopeT2VModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void LatteModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new LatteModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void OpenSoraModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new OpenSoraModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void RunwayGenModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new RunwayGenModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.SupportsTextToVideo);
        Assert.True(model.SupportsImageToVideo);
    }

    [Fact]
    public void MakeAVideoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MakeAVideoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void KlingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new KlingModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void VeoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new VeoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void Mochi1Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Mochi1Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void HunyuanVideoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HunyuanVideoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void LTXVideoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new LTXVideoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
        Assert.True(model.SupportsTextToVideo);
    }

    [Fact]
    public void WanVideoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new WanVideoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(16, model.LatentChannels);
        Assert.True(model.SupportsTextToVideo);
        Assert.True(model.SupportsImageToVideo);
        Assert.Equal("14B", model.Variant); // default variant
    }

    [Fact]
    public void WanVideoModel_Create1_3B_CreatesLightweightVariant()
    {
        var model = WanVideoModel<double>.Create1_3B();

        Assert.NotNull(model);
        Assert.Equal("1.3B", model.Variant);
    }

    [Fact]
    public void WanVideoModel_Create5B_CreatesMediumVariant()
    {
        var model = WanVideoModel<double>.Create5B();

        Assert.NotNull(model);
        Assert.Equal("5B", model.Variant);
    }

    #endregion

    #region 3D Models (Tasks #87-#94)

    [Fact]
    public void SyncDreamerModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SyncDreamerModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void Wonder3DModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Wonder3DModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void One2345Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new One2345Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void Instant3DModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Instant3DModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void DreamGaussianModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DreamGaussianModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void LGMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new LGMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void TripoSRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TripoSRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    [Fact]
    public void MeshyModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MeshyModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.LatentChannels > 0);
    }

    #endregion

    #region Audio Models (Tasks #95-#99)

    [Fact]
    public void StableAudioModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableAudioModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(64, model.LatentChannels);
        Assert.True(model.SupportsTextToAudio);
        Assert.True(model.SupportsTextToMusic);
    }

    [Fact]
    public void BarkModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new BarkModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(8, model.LatentChannels);
        Assert.True(model.SupportsTextToAudio);
        Assert.True(model.SupportsTextToSpeech);
    }

    [Fact]
    public void VoiceCraftModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new VoiceCraftModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(8, model.LatentChannels);
        Assert.True(model.SupportsTextToSpeech);
        Assert.True(model.SupportsAudioToAudio);
    }

    [Fact]
    public void SoundStormModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SoundStormModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(8, model.LatentChannels);
        Assert.True(model.SupportsTextToSpeech);
    }

    [Fact]
    public void UdioModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new UdioModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(64, model.LatentChannels);
        Assert.True(model.SupportsTextToMusic);
        Assert.True(model.SupportsAudioToAudio);
    }

    #endregion

    #region Clone Contract Tests - Representative Models

    [Fact]
    public void WanVideoModel_Clone_CreatesIndependentCopy()
    {
        var model = new WanVideoModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact]
    public void BarkModel_Clone_CreatesIndependentCopy()
    {
        var model = new BarkModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact]
    public void SDUpscalerModel_Clone_CreatesIndependentCopy()
    {
        var model = new SDUpscalerModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    [Fact]
    public void PixArtSigmaModel_Clone_CreatesIndependentCopy()
    {
        var model = new PixArtSigmaModel<double>();
        var clone = model.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(model, clone);
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
    }

    #endregion

    #region GetModelMetadata Contract Tests - Representative Models

    [Fact]
    public void WanVideoModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new WanVideoModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.Contains("Wan", metadata.Name);
    }

    [Fact]
    public void BarkModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new BarkModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.Contains("Bark", metadata.Name);
    }

    [Fact]
    public void SoraModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new SoraModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    [Fact]
    public void StableAudioModel_GetModelMetadata_ReturnsValidMetadata()
    {
        var model = new StableAudioModel<double>();

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Name);
    }

    #endregion

    #region IParameterizable Contract Tests - Representative New Models

    [Fact]
    public void InstructPix2PixModel_GetParameters_ReturnsNonEmptyVector()
    {
        var model = new InstructPix2PixModel<double>();

        var parameters = model.GetParameters();

        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void SoraModel_ParameterCount_MatchesGetParametersLength()
    {
        var model = new SoraModel<double>();

        var parameters = model.GetParameters();

        Assert.Equal(model.ParameterCount, parameters.Length);
    }

    [Fact]
    public void UdioModel_SetParameters_DoesNotThrow()
    {
        var model = new UdioModel<double>();

        var parameters = model.GetParameters();
        model.SetParameters(parameters);

        var retrievedParams = model.GetParameters();
        Assert.Equal(parameters.Length, retrievedParams.Length);
    }

    [Fact]
    public void TripoSRModel_ParameterCount_MatchesGetParametersLength()
    {
        var model = new TripoSRModel<double>();

        var parameters = model.GetParameters();

        Assert.Equal(model.ParameterCount, parameters.Length);
    }

    #endregion

    #region Missing Model Coverage Tests

    [Fact]
    public void PixArtModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PixArtModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void DiffWaveModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DiffWaveModel<double>();

        Assert.NotNull(model);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void RiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new RiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void Zero123Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new Zero123Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion
}
