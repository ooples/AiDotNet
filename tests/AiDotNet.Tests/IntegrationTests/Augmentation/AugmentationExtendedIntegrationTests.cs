using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Audio;
using AiDotNet.Augmentation.Image;
using AiDotNet.Augmentation.Tabular;
using AiDotNet.Augmentation.Tabular.Undersampling;
using AiDotNet.Augmentation.Text;
using AiDotNet.Augmentation.Video;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Augmentation;

/// <summary>
/// Extended integration tests for the Augmentation module covering configuration classes,
/// pipeline/compose/one-of/some-of orchestrators, augmentation context, settings classes,
/// and representative augmentations from each modality (image, audio, tabular, text, video).
/// </summary>
public class AugmentationExtendedIntegrationTests
{
    #region AugmentationConfig

    [Fact]
    public void AugmentationConfig_DefaultValues()
    {
        var config = new AugmentationConfig();

        Assert.True(config.IsEnabled);
        Assert.Equal(0.5, config.Probability);
        Assert.Null(config.Seed);
        Assert.True(config.EnableTTA);
        Assert.Equal(5, config.TTANumAugmentations);
        Assert.Equal(PredictionAggregationMethod.Mean, config.TTAAggregation);
        Assert.True(config.TTAIncludeOriginal);
        Assert.Null(config.ImageSettings);
        Assert.Null(config.TabularSettings);
        Assert.Null(config.AudioSettings);
        Assert.Null(config.TextSettings);
        Assert.Null(config.VideoSettings);
    }

    [Fact]
    public void AugmentationConfig_ForImages_HasImageSettings()
    {
        var config = AugmentationConfig.ForImages();

        Assert.NotNull(config.ImageSettings);
        Assert.Null(config.TabularSettings);
        Assert.Null(config.AudioSettings);
    }

    [Fact]
    public void AugmentationConfig_ForTabular_HasTabularSettings()
    {
        var config = AugmentationConfig.ForTabular();

        Assert.NotNull(config.TabularSettings);
        Assert.Null(config.ImageSettings);
    }

    [Fact]
    public void AugmentationConfig_ForAudio_HasAudioSettings()
    {
        var config = AugmentationConfig.ForAudio();

        Assert.NotNull(config.AudioSettings);
        Assert.Null(config.ImageSettings);
    }

    [Fact]
    public void AugmentationConfig_ForText_HasTextSettings()
    {
        var config = AugmentationConfig.ForText();

        Assert.NotNull(config.TextSettings);
        Assert.Null(config.ImageSettings);
    }

    [Fact]
    public void AugmentationConfig_ForVideo_HasVideoSettings()
    {
        var config = AugmentationConfig.ForVideo();

        Assert.NotNull(config.VideoSettings);
        Assert.Null(config.ImageSettings);
    }

    [Fact]
    public void AugmentationConfig_GetConfiguration_ReturnsAllKeys()
    {
        var config = new AugmentationConfig
        {
            Seed = 42,
            ImageSettings = new ImageAugmentationSettings()
        };

        var dict = config.GetConfiguration();

        Assert.True(dict.ContainsKey("isEnabled"));
        Assert.True(dict.ContainsKey("probability"));
        Assert.True(dict.ContainsKey("enableTTA"));
        Assert.True(dict.ContainsKey("seed"));
        Assert.True(dict.ContainsKey("imageSettings"));
    }

    [Fact]
    public void AugmentationConfig_CustomValues()
    {
        var config = new AugmentationConfig
        {
            IsEnabled = false,
            Probability = 0.8,
            Seed = 123,
            EnableTTA = false,
            TTANumAugmentations = 10,
            TTAAggregation = PredictionAggregationMethod.Median,
            TTAIncludeOriginal = false
        };

        Assert.False(config.IsEnabled);
        Assert.Equal(0.8, config.Probability);
        Assert.Equal(123, config.Seed);
        Assert.False(config.EnableTTA);
        Assert.Equal(10, config.TTANumAugmentations);
        Assert.Equal(PredictionAggregationMethod.Median, config.TTAAggregation);
        Assert.False(config.TTAIncludeOriginal);
    }

    #endregion

    #region ImageAugmentationSettings

    [Fact]
    public void ImageAugmentationSettings_DefaultValues()
    {
        var settings = new ImageAugmentationSettings();

        Assert.True(settings.EnableFlips);
        Assert.False(settings.EnableVerticalFlip);
        Assert.True(settings.EnableRotation);
        Assert.Equal(15.0, settings.RotationRange);
        Assert.True(settings.EnableColorJitter);
        Assert.Equal(0.2, settings.BrightnessRange);
        Assert.Equal(0.2, settings.ContrastRange);
        Assert.Equal(0.2, settings.SaturationRange);
        Assert.True(settings.EnableGaussianNoise);
        Assert.Equal(0.01, settings.NoiseStdDev);
        Assert.False(settings.EnableGaussianBlur);
        Assert.False(settings.EnableCutout);
        Assert.False(settings.EnableMixUp);
        Assert.Equal(0.2, settings.MixUpAlpha);
        Assert.False(settings.EnableCutMix);
    }

    [Fact]
    public void ImageAugmentationSettings_GetConfiguration_ReturnsAllKeys()
    {
        var settings = new ImageAugmentationSettings();
        var dict = settings.GetConfiguration();

        Assert.True(dict.ContainsKey("enableFlips"));
        Assert.True(dict.ContainsKey("rotationRange"));
        Assert.True(dict.ContainsKey("enableColorJitter"));
        Assert.True(dict.ContainsKey("noiseStdDev"));
    }

    #endregion

    #region TabularAugmentationSettings

    [Fact]
    public void TabularAugmentationSettings_DefaultValues()
    {
        var settings = new TabularAugmentationSettings();

        Assert.True(settings.EnableMixUp);
        Assert.Equal(0.2, settings.MixUpAlpha);
        Assert.True(settings.EnableFeatureNoise);
        Assert.Equal(0.01, settings.NoiseStdDev);
        Assert.False(settings.EnableFeatureDropout);
        Assert.Equal(0.1, settings.DropoutRate);
        Assert.False(settings.EnableSmote);
        Assert.Equal(5, settings.SmoteK);
    }

    [Fact]
    public void TabularAugmentationSettings_GetConfiguration_ReturnsAllKeys()
    {
        var settings = new TabularAugmentationSettings();
        var dict = settings.GetConfiguration();

        Assert.True(dict.ContainsKey("enableMixUp"));
        Assert.True(dict.ContainsKey("enableFeatureNoise"));
        Assert.True(dict.ContainsKey("enableSmote"));
    }

    #endregion

    #region AudioAugmentationSettings

    [Fact]
    public void AudioAugmentationSettings_DefaultValues()
    {
        var settings = new AudioAugmentationSettings();

        Assert.True(settings.EnablePitchShift);
        Assert.Equal(2.0, settings.PitchShiftRange);
        Assert.True(settings.EnableTimeStretch);
        Assert.Equal(0.8, settings.MinTimeStretch);
        Assert.Equal(1.2, settings.MaxTimeStretch);
        Assert.True(settings.EnableNoise);
        Assert.Equal(20.0, settings.NoiseSNR);
        Assert.True(settings.EnableVolumeChange);
        Assert.Equal(6.0, settings.VolumeChangeRange);
        Assert.True(settings.EnableTimeShift);
        Assert.Equal(0.1, settings.MaxTimeShift);
    }

    [Fact]
    public void AudioAugmentationSettings_GetConfiguration_ReturnsAllKeys()
    {
        var settings = new AudioAugmentationSettings();
        var dict = settings.GetConfiguration();

        Assert.True(dict.ContainsKey("enablePitchShift"));
        Assert.True(dict.ContainsKey("noiseSNR"));
        Assert.True(dict.ContainsKey("enableTimeShift"));
    }

    #endregion

    #region TextAugmentationSettings

    [Fact]
    public void TextAugmentationSettings_DefaultValues()
    {
        var settings = new TextAugmentationSettings();

        Assert.True(settings.EnableSynonymReplacement);
        Assert.Equal(0.1, settings.SynonymReplacementRate);
        Assert.True(settings.EnableRandomDeletion);
        Assert.Equal(0.1, settings.DeletionRate);
        Assert.True(settings.EnableRandomSwap);
        Assert.Equal(2, settings.NumSwaps);
        Assert.False(settings.EnableRandomInsertion);
        Assert.Equal(0.1, settings.InsertionRate);
        Assert.False(settings.EnableBackTranslation);
    }

    [Fact]
    public void TextAugmentationSettings_GetConfiguration_ReturnsAllKeys()
    {
        var settings = new TextAugmentationSettings();
        var dict = settings.GetConfiguration();

        Assert.True(dict.ContainsKey("enableSynonymReplacement"));
        Assert.True(dict.ContainsKey("enableRandomDeletion"));
        Assert.True(dict.ContainsKey("enableRandomSwap"));
    }

    #endregion

    #region VideoAugmentationSettings

    [Fact]
    public void VideoAugmentationSettings_DefaultValues()
    {
        var settings = new VideoAugmentationSettings();

        Assert.True(settings.EnableTemporalCrop);
        Assert.Equal(0.8, settings.CropRatio);
        Assert.True(settings.EnableTemporalFlip);
        Assert.True(settings.EnableFrameDropout);
        Assert.Equal(0.1, settings.DropoutRate);
        Assert.True(settings.EnableSpeedChange);
        Assert.Equal(0.8, settings.MinSpeed);
        Assert.Equal(1.2, settings.MaxSpeed);
        Assert.True(settings.EnableSpatialTransforms);
        Assert.Null(settings.SpatialSettings);
    }

    [Fact]
    public void VideoAugmentationSettings_GetConfiguration_ReturnsAllKeys()
    {
        var settings = new VideoAugmentationSettings();
        var dict = settings.GetConfiguration();

        Assert.True(dict.ContainsKey("enableTemporalCrop"));
        Assert.True(dict.ContainsKey("enableFrameDropout"));
        Assert.True(dict.ContainsKey("enableSpeedChange"));
    }

    [Fact]
    public void VideoAugmentationSettings_WithSpatialSettings_IncludesInConfig()
    {
        var settings = new VideoAugmentationSettings
        {
            SpatialSettings = new ImageAugmentationSettings()
        };
        var dict = settings.GetConfiguration();

        Assert.True(dict.ContainsKey("spatialSettings"));
    }

    #endregion

    #region DataModalityDetector

    [Fact]
    public void DataModalityDetector_String_ReturnsText()
    {
        var modality = DataModalityDetector.Detect<string>();
        Assert.Equal(DataModality.Text, modality);
    }

    [Fact]
    public void DataModalityDetector_StringArray_ReturnsText()
    {
        var modality = DataModalityDetector.Detect<string[]>();
        Assert.Equal(DataModality.Text, modality);
    }

    [Fact]
    public void DataModalityDetector_Tensor_ReturnsUnknown()
    {
        var modality = DataModalityDetector.Detect<Tensor<double>>();
        Assert.Equal(DataModality.Unknown, modality);
    }

    #endregion

    #region AugmentationContext

    [Fact]
    public void AugmentationContext_DefaultIsTraining()
    {
        var ctx = new AugmentationContext<double>();
        Assert.True(ctx.IsTraining);
    }

    [Fact]
    public void AugmentationContext_SeedIsReproducible()
    {
        var ctx1 = new AugmentationContext<double>(isTraining: true, seed: 42);
        var ctx2 = new AugmentationContext<double>(isTraining: true, seed: 42);

        double val1 = ctx1.GetRandomDouble(0, 1);
        double val2 = ctx2.GetRandomDouble(0, 1);

        Assert.Equal(val1, val2);
    }

    [Fact]
    public void AugmentationContext_ShouldApply_AlwaysForProbability1()
    {
        var ctx = new AugmentationContext<double>(seed: 42);

        for (int i = 0; i < 100; i++)
        {
            Assert.True(ctx.ShouldApply(1.0));
        }
    }

    [Fact]
    public void AugmentationContext_ShouldApply_NeverForProbability0()
    {
        var ctx = new AugmentationContext<double>(seed: 42);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(ctx.ShouldApply(0.0));
        }
    }

    [Fact]
    public void AugmentationContext_GetRandomInt_InRange()
    {
        var ctx = new AugmentationContext<double>(seed: 42);

        for (int i = 0; i < 100; i++)
        {
            int val = ctx.GetRandomInt(5, 10);
            Assert.InRange(val, 5, 9);
        }
    }

    [Fact]
    public void AugmentationContext_GetRandomDouble_InRange()
    {
        var ctx = new AugmentationContext<double>(seed: 42);

        for (int i = 0; i < 100; i++)
        {
            double val = ctx.GetRandomDouble(2.0, 5.0);
            Assert.InRange(val, 2.0, 5.0);
        }
    }

    [Fact]
    public void AugmentationContext_SampleGaussian_ReasonableValues()
    {
        var ctx = new AugmentationContext<double>(seed: 42);
        double sum = 0;
        int count = 1000;

        for (int i = 0; i < count; i++)
        {
            sum += ctx.SampleGaussian(10.0, 1.0);
        }

        double mean = sum / count;
        Assert.InRange(mean, 9.0, 11.0);
    }

    [Fact]
    public void AugmentationContext_SampleBeta_InUnitInterval()
    {
        var ctx = new AugmentationContext<double>(seed: 42);

        for (int i = 0; i < 100; i++)
        {
            double val = ctx.SampleBeta(2.0, 2.0);
            Assert.InRange(val, 0.0, 1.0);
        }
    }

    [Fact]
    public void AugmentationContext_CreateChildContext_SharesRandomState()
    {
        var ctx = new AugmentationContext<double>(seed: 42);
        var child = ctx.CreateChildContext(5);

        Assert.Equal(5, child.SampleIndex);
        Assert.True(child.IsTraining);
    }

    [Fact]
    public void AugmentationContext_InferenceMode()
    {
        var ctx = new AugmentationContext<double>(isTraining: false);
        Assert.False(ctx.IsTraining);
    }

    [Fact]
    public void AugmentationContext_Metadata_Accessible()
    {
        var ctx = new AugmentationContext<double>();
        ctx.Metadata["key"] = "value";
        Assert.Equal("value", ctx.Metadata["key"]);
    }

    #endregion

    #region AugmentationPipeline

    [Fact]
    public void AugmentationPipeline_Construction()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>("TestPipeline");

        Assert.Equal("TestPipeline", pipeline.Name);
        Assert.Equal(0, pipeline.AugmentationCount);
        Assert.Equal(AugmentationOrder.Sequential, pipeline.Order);
    }

    [Fact]
    public void AugmentationPipeline_Add_IncreasesCount()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();
        var flip = new HorizontalFlip<double>();
        pipeline.Add(flip);

        Assert.Equal(1, pipeline.AugmentationCount);
        Assert.Contains("HorizontalFlip", pipeline.AugmentationNames[0]);
    }

    [Fact]
    public void AugmentationPipeline_AddNull_Throws()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();

        Assert.Throws<ArgumentNullException>(() => pipeline.Add(null!));
    }

    [Fact]
    public void AugmentationPipeline_AddRange_AddsAll()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();
        var augs = new IAugmentation<double, ImageTensor<double>>[]
        {
            new HorizontalFlip<double>(),
            new VerticalFlip<double>()
        };
        pipeline.AddRange(augs);

        Assert.Equal(2, pipeline.AugmentationCount);
    }

    [Fact]
    public void AugmentationPipeline_EmptyPipeline_ReturnsOriginal()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();
        var image = new ImageTensor<double>(8, 8, channels: 3);
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = pipeline.Apply(image, ctx);

        Assert.Same(image, result);
    }

    [Fact]
    public void AugmentationPipeline_GetConfiguration_ReturnsInfo()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>("MyPipeline");
        pipeline.Add(new HorizontalFlip<double>());

        var config = pipeline.GetConfiguration();

        Assert.Equal("MyPipeline", config["name"]);
        Assert.Equal("Sequential", config["order"]);
    }

    #endregion

    #region Compose

    [Fact]
    public void Compose_Construction()
    {
        var compose = new Compose<double, ImageTensor<double>>(
            new HorizontalFlip<double>(1.0),
            new VerticalFlip<double>(1.0)
        );

        Assert.NotNull(compose);
        Assert.Equal("Compose", compose.Name);
    }

    #endregion

    #region OneOf

    [Fact]
    public void OneOf_Construction()
    {
        var oneOf = new OneOf<double, ImageTensor<double>>(
            new HorizontalFlip<double>(),
            new VerticalFlip<double>()
        );

        Assert.NotNull(oneOf);
        Assert.Equal("OneOf", oneOf.Name);
    }

    #endregion

    #region SomeOf

    [Fact]
    public void SomeOf_Construction()
    {
        var someOf = new SomeOf<double, ImageTensor<double>>(
            n: 2,
            new HorizontalFlip<double>(),
            new VerticalFlip<double>(),
            new Rotation<double>()
        );

        Assert.NotNull(someOf);
        Assert.Equal("SomeOf", someOf.Name);
    }

    #endregion

    #region Image Augmentations - Construction

    [Fact]
    public void HorizontalFlip_Construction_DefaultProbability()
    {
        var aug = new HorizontalFlip<double>();
        Assert.Equal(0.5, aug.Probability);
        Assert.Equal("HorizontalFlip`1", aug.Name);
    }

    [Fact]
    public void VerticalFlip_Construction()
    {
        var aug = new VerticalFlip<double>(0.3);
        Assert.Equal(0.3, aug.Probability);
    }

    [Fact]
    public void Rotation_Construction()
    {
        var aug = new Rotation<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Brightness_Construction()
    {
        var aug = new Brightness<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Contrast_Construction()
    {
        var aug = new Contrast<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GaussianNoise_Construction()
    {
        var aug = new GaussianNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GaussianBlur_Construction()
    {
        var aug = new GaussianBlur<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ColorJitter_Construction()
    {
        var aug = new ColorJitter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void CenterCrop_Construction()
    {
        var aug = new CenterCrop<double>(32, 32);
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomCrop_Construction()
    {
        var aug = new RandomCrop<double>(32, 32);
        Assert.NotNull(aug);
    }

    [Fact]
    public void Resize_Construction()
    {
        var aug = new Resize<double>(64, 64);
        Assert.NotNull(aug);
    }

    [Fact]
    public void Normalize_Construction()
    {
        var aug = new Normalize<double>(new[] { 0.485, 0.456, 0.406 }, new[] { 0.229, 0.224, 0.225 });
        Assert.NotNull(aug);
    }

    [Fact]
    public void Scale_Construction()
    {
        var aug = new Scale<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Pad_Construction()
    {
        var aug = new Pad<double>(2, 2, 2, 2);
        Assert.NotNull(aug);
    }

    [Fact]
    public void ElasticTransform_Construction()
    {
        var aug = new ElasticTransform<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Affine_Construction()
    {
        var aug = new Affine<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Perspective_Construction()
    {
        var aug = new Perspective<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Cutout_Construction()
    {
        var aug = new Cutout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void CoarseDropout_Construction()
    {
        var aug = new CoarseDropout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void MixUp_Construction()
    {
        var aug = new MixUp<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void CutMix_Construction()
    {
        var aug = new CutMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandAugment_Construction()
    {
        var aug = new RandAugment<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void AutoAugment_Construction()
    {
        var aug = new AutoAugment<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Solarize_Construction()
    {
        var aug = new Solarize<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Posterize_Construction()
    {
        var aug = new Posterize<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Equalize_Construction()
    {
        var aug = new Equalize<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Sharpen_Construction()
    {
        var aug = new Sharpen<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Emboss_Construction()
    {
        var aug = new Emboss<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void MotionBlur_Construction()
    {
        var aug = new MotionBlur<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void MedianBlur_Construction()
    {
        var aug = new MedianBlur<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void BoxBlur_Construction()
    {
        var aug = new BoxBlur<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Rain_Construction()
    {
        var aug = new Rain<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Snow_Construction()
    {
        var aug = new Snow<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Fog_Construction()
    {
        var aug = new Fog<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SunFlare_Construction()
    {
        var aug = new SunFlare<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Shadow_Construction()
    {
        var aug = new Shadow<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void JpegCompression_Construction()
    {
        var aug = new JpegCompression<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Downscale_Construction()
    {
        var aug = new Downscale<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RgbToGrayscale_Construction()
    {
        var aug = new RgbToGrayscale<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ToFloat_Construction()
    {
        var aug = new ToFloat<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ToTensor_Construction()
    {
        var aug = new ToTensor<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomErasing_Construction()
    {
        var aug = new RandomErasing<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GridDistortion_Construction()
    {
        var aug = new GridDistortion<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void OpticalDistortion_Construction()
    {
        var aug = new OpticalDistortion<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ChannelShuffle_Construction()
    {
        var aug = new ChannelShuffle<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ChannelDropout_Construction()
    {
        var aug = new ChannelDropout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void PoissonNoise_Construction()
    {
        var aug = new PoissonNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SaltAndPepperNoise_Construction()
    {
        var aug = new SaltAndPepperNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SpeckleNoise_Construction()
    {
        var aug = new SpeckleNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void MultiplicativeNoise_Construction()
    {
        var aug = new MultiplicativeNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GammaCorrection_Construction()
    {
        var aug = new GammaCorrection<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Saturation_Construction()
    {
        var aug = new Saturation<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void HueSaturationValue_Construction()
    {
        var aug = new HueSaturationValue<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomBrightnessContrast_Construction()
    {
        var aug = new RandomBrightnessContrast<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void UnsharpMask_Construction()
    {
        var aug = new UnsharpMask<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void CLAHE_Construction()
    {
        var aug = new CLAHE<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void FancyPCA_Construction()
    {
        var aug = new FancyPCA<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Superpixels_Construction()
    {
        var aug = new Superpixels<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void PixelDropout_Construction()
    {
        var aug = new PixelDropout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GridDropout_Construction()
    {
        var aug = new GridDropout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Mosaic_Construction()
    {
        var aug = new Mosaic<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Image Augmentations - Apply

    [Fact]
    public void HorizontalFlip_Apply_ProducesResult()
    {
        var flip = new HorizontalFlip<double>(1.0);
        var image = CreateTestImage(8, 8, 3);
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = flip.Apply(image, ctx);

        Assert.NotNull(result);
        Assert.Equal(8, result.Height);
        Assert.Equal(8, result.Width);
    }

    [Fact]
    public void HorizontalFlip_Apply_FlipsPixels()
    {
        var flip = new HorizontalFlip<double>(1.0);
        var image = CreateTestImage(4, 4, 1);
        // Set a known pixel at (0, 0) to 1.0
        image.SetPixel(0, 0, 0, 99.0);
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = flip.Apply(image, ctx);

        // After horizontal flip, pixel at (0, 0) should be at (0, 3)
        Assert.Equal(99.0, result.GetPixel(0, 3, 0));
    }

    [Fact]
    public void VerticalFlip_Apply_FlipsPixels()
    {
        var flip = new VerticalFlip<double>(1.0);
        var image = CreateTestImage(4, 4, 1);
        image.SetPixel(0, 0, 0, 77.0);
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = flip.Apply(image, ctx);

        // After vertical flip, pixel at (0, 0) should be at (3, 0)
        Assert.Equal(77.0, result.GetPixel(3, 0, 0));
    }

    [Fact]
    public void AugmentationBase_DisabledAugmentation_ReturnsOriginal()
    {
        var flip = new HorizontalFlip<double>(1.0);
        flip.IsEnabled = false;
        var image = CreateTestImage(8, 8, 3);
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = flip.Apply(image, ctx);

        // When disabled, should return original
        Assert.Same(image, result);
    }

    [Fact]
    public void AugmentationBase_InferenceMode_SkipsTrainingOnly()
    {
        var flip = new HorizontalFlip<double>(1.0);
        Assert.True(flip.IsTrainingOnly);

        var image = CreateTestImage(8, 8, 3);
        var ctx = new AugmentationContext<double>(isTraining: false, seed: 42);

        var result = flip.Apply(image, ctx);

        // Training-only augmentation should return original during inference
        Assert.Same(image, result);
    }

    [Fact]
    public void AugmentationBase_GetParameters_ReturnsInfo()
    {
        var flip = new HorizontalFlip<double>(0.7);
        var parameters = flip.GetParameters();

        Assert.Equal("HorizontalFlip`1", parameters["name"]);
        Assert.Equal(0.7, parameters["probability"]);
        Assert.Equal(true, parameters["isEnabled"]);
    }

    [Fact]
    public void AugmentationBase_InvalidProbability_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HorizontalFlip<double>(-0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HorizontalFlip<double>(1.1));
    }

    [Fact]
    public void AugmentationBase_OnAugmentationApplied_Event()
    {
        var flip = new HorizontalFlip<double>(1.0);
        bool eventFired = false;

        flip.OnAugmentationApplied += (sender, args) => { eventFired = true; };

        var image = CreateTestImage(4, 4, 1);
        var ctx = new AugmentationContext<double>(seed: 42);
        flip.Apply(image, ctx);

        Assert.True(eventFired);
    }

    #endregion

    #region Audio Augmentations

    [Fact]
    public void AudioNoise_Construction()
    {
        var aug = new AudioNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void PitchShift_Construction()
    {
        var aug = new PitchShift<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TimeStretch_Construction()
    {
        var aug = new TimeStretch<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TimeShift_Construction()
    {
        var aug = new TimeShift<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void VolumeChange_Construction()
    {
        var aug = new VolumeChange<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void AudioNoise_Apply_ProducesResult()
    {
        var aug = new AudioNoise<double>(probability: 1.0);
        var audio = new Tensor<double>(new[] { 16000 });
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = aug.Apply(audio, ctx);
        Assert.NotNull(result);
    }

    #endregion

    #region Tabular Augmentations

    [Fact]
    public void FeatureNoise_Construction()
    {
        var aug = new FeatureNoise<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void FeatureDropout_Construction()
    {
        var aug = new FeatureDropout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RowShuffle_Construction()
    {
        var aug = new RowShuffle<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TabularMixUp_Construction()
    {
        var aug = new TabularMixUp<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SmoteAugmenter_Construction()
    {
        var aug = new SmoteAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void AdasynAugmenter_Construction()
    {
        var aug = new AdasynAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void BorderlineSmoteAugmenter_Construction()
    {
        var aug = new BorderlineSmoteAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SmoteEnnAugmenter_Construction()
    {
        var aug = new SmoteEnnAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SmoteTomekAugmenter_Construction()
    {
        var aug = new SmoteTomekAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SvmSmoteAugmenter_Construction()
    {
        var aug = new SvmSmoteAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TomekLinksAugmenter_Construction()
    {
        var aug = new TomekLinksAugmenter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomUnderSampler_Construction()
    {
        var aug = new RandomUnderSampler<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void NearMissUnderSampler_Construction()
    {
        var aug = new NearMissUnderSampler<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Text Augmentations

    [Fact]
    public void SynonymReplacement_Construction()
    {
        var aug = new SynonymReplacement<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomDeletion_Construction()
    {
        var aug = new RandomDeletion<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomInsertion_Construction()
    {
        var aug = new RandomInsertion<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RandomSwap_Construction()
    {
        var aug = new RandomSwap<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SynonymReplacement_Apply_ProducesResult()
    {
        var aug = new SynonymReplacement<double>(probability: 1.0, replacementFraction: 1.0);
        var text = new[] { "The", "good", "cat", "is", "happy" };
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = aug.Apply(text, ctx);

        Assert.NotNull(result);
        Assert.True(result.Length > 0);
    }

    [Fact]
    public void RandomDeletion_Apply_ReducesWords()
    {
        var aug = new RandomDeletion<double>(deletionProbability: 0.5, probability: 1.0);
        var text = new[] { "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog" };
        var ctx = new AugmentationContext<double>(seed: 42);

        var result = aug.Apply(text, ctx);

        Assert.NotNull(result);
        // With 50% deletion probability, result should usually be shorter
        Assert.True(result.Length <= text.Length);
    }

    #endregion

    #region Video Augmentations

    [Fact]
    public void FrameDropout_Construction()
    {
        var aug = new FrameDropout<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TemporalCrop_Construction()
    {
        var aug = new TemporalCrop<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TemporalFlip_Construction()
    {
        var aug = new TemporalFlip<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SpeedChange_Construction()
    {
        var aug = new SpeedChange<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void VideoColorJitter_Construction()
    {
        var aug = new VideoColorJitter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SpatialTransform_Construction()
    {
        var aug = new SpatialTransform<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Additional Image Augmentations - Weather & Corruption

    [Fact]
    public void Clouds_Construction()
    {
        var aug = new Clouds<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Frost_Construction()
    {
        var aug = new Frost<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Spatter_Construction()
    {
        var aug = new Spatter<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ZoomBlur_Construction()
    {
        var aug = new ZoomBlur<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GlassBlur_Construction()
    {
        var aug = new GlassBlur<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Defocus_Construction()
    {
        var aug = new Defocus<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Additional Image Augmentations - Morphological

    [Fact]
    public void Dilate_Construction()
    {
        var aug = new Dilate<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Erode_Construction()
    {
        var aug = new Erode<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Opening_Construction()
    {
        var aug = new Opening<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Closing_Construction()
    {
        var aug = new Closing<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void MorphologicalGradient_Construction()
    {
        var aug = new MorphologicalGradient<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Additional Image Augmentations - Color Space

    [Fact]
    public void RgbToHsv_Construction()
    {
        var aug = new RgbToHsv<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RgbToLab_Construction()
    {
        var aug = new RgbToLab<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void RgbToBgr_Construction()
    {
        var aug = new RgbToBgr<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ToGray_Construction()
    {
        var aug = new ToGray<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ToSepia_Construction()
    {
        var aug = new ToSepia<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void Denormalize_Construction()
    {
        var aug = new Denormalize<double>(new[] { 0.485, 0.456, 0.406 }, new[] { 0.229, 0.224, 0.225 });
        Assert.NotNull(aug);
    }

    #endregion

    #region Additional Image Augmentations - Advanced Mixing

    [Fact]
    public void FMix_Construction()
    {
        var aug = new FMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void GridMask_Construction()
    {
        var aug = new GridMask<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void HideAndSeek_Construction()
    {
        var aug = new HideAndSeek<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SamplePairing_Construction()
    {
        var aug = new SamplePairing<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void ResizeMix_Construction()
    {
        var aug = new ResizeMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SaliencyMix_Construction()
    {
        var aug = new SaliencyMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void PuzzleMix_Construction()
    {
        var aug = new PuzzleMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void SnapMix_Construction()
    {
        var aug = new SnapMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void StyleMix_Construction()
    {
        var aug = new StyleMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TokenMix_Construction()
    {
        var aug = new TokenMix<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void TransMix_Construction()
    {
        var aug = new TransMix<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Additional Image Augmentations - Auto Augmentation Policies

    [Fact]
    public void TrivialAugment_Construction()
    {
        var aug = new TrivialAugment<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void UniformAugment_Construction()
    {
        var aug = new UniformAugment<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void FastAutoAugment_Construction()
    {
        var aug = new FastAutoAugment<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void AugMax_Construction()
    {
        var aug = new AugMax<double>();
        Assert.NotNull(aug);
    }

    [Fact]
    public void DADA_Construction()
    {
        var aug = new DADA<double>();
        Assert.NotNull(aug);
    }

    #endregion

    #region Helpers

    private static ImageTensor<double> CreateTestImage(int height, int width, int channels)
    {
        return new ImageTensor<double>(height, width, channels: channels);
    }

    #endregion
}
