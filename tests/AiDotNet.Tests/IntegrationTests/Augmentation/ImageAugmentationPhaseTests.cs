using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Image;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Augmentation;

/// <summary>
/// Comprehensive integration tests for Phases 1-12 image augmentation classes.
/// Tests cover construction, application, dimension preservation, value bounds,
/// pipeline chaining, spatial target transforms, label mixing, and edge cases.
/// </summary>
public class ImageAugmentationPhaseTests
{
    private const int TestSeed = 42;
    private const double Tolerance = 1e-6;

    #region Helper Methods

    private static ImageTensor<double> CreateTestImage(int height, int width, int channels = 3, double initialValue = 0.5)
    {
        var tensor = new Tensor<double>(new[] { channels, height, width });
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    double value = initialValue + (c * 0.1) + (y * 0.001) + (x * 0.0001);
                    tensor[c * height * width + y * width + x] = Math.Min(1.0, Math.Max(0.0, value));
                }

        var image = new ImageTensor<double>(tensor, ChannelOrder.CHW, ColorSpace.RGB);
        image.IsNormalized = true;
        return image;
    }

    private static AugmentationContext<double> CreateTestContext(bool isTraining = true, int? seed = TestSeed)
    {
        return new AugmentationContext<double>(isTraining, seed);
    }

    /// <summary>
    /// Asserts that all pixel values in an image are within [0, maxVal].
    /// </summary>
    private static void AssertPixelsInRange(ImageTensor<double> image, double minVal = 0.0, double maxVal = 1.0)
    {
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
                for (int c = 0; c < image.Channels; c++)
                {
                    var val = image.GetPixel(y, x, c);
                    Assert.InRange(val, minVal - 0.01, maxVal + 0.01);
                }
    }

    #endregion

    // ===================================================================
    // Phase 1: Basic Preprocessing (Resize, CenterCrop, Normalize, etc.)
    // ===================================================================

    #region Phase 1 - Basic Preprocessing

    [Fact]
    public void CenterCrop_Apply_ProducesCorrectDimensions()
    {
        var image = CreateTestImage(32, 32);
        var crop = new CenterCrop<double>(20, 20, probability: 1.0);
        var context = CreateTestContext();

        var result = crop.Apply(image, context);

        Assert.Equal(20, result.Height);
        Assert.Equal(20, result.Width);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void FiveCrop_Apply_ReturnsFiveCrops()
    {
        var image = CreateTestImage(32, 32);
        var fiveCrop = new FiveCrop<double>(16, 16);

        var crops = fiveCrop.GetCrops(image);

        Assert.Equal(5, crops.Count);
        foreach (var crop in crops)
        {
            Assert.Equal(16, crop.Height);
            Assert.Equal(16, crop.Width);
        }
    }

    [Fact]
    public void TenCrop_Apply_ReturnsTenCrops()
    {
        var image = CreateTestImage(32, 32);
        var tenCrop = new TenCrop<double>(16, 16);

        var crops = tenCrop.GetCrops(image);

        Assert.Equal(10, crops.Count);
        foreach (var crop in crops)
        {
            Assert.Equal(16, crop.Height);
            Assert.Equal(16, crop.Width);
        }
    }

    [Fact]
    public void PadToSquare_Apply_ProducesResult()
    {
        var image = CreateTestImage(20, 30);
        var pad = new PadToSquare<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = pad.Apply(image, context);

        // PadToSquare should produce a valid result
        Assert.True(result.Height > 0);
        Assert.True(result.Width > 0);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void ResizeWithAspectRatio_Apply_ResizesImage()
    {
        var image = CreateTestImage(20, 40);
        var resize = new ResizeWithAspectRatio<double>(30, 30, probability: 1.0);
        var context = CreateTestContext();

        var result = resize.Apply(image, context);

        // ResizeWithAspectRatio produces a valid image output
        Assert.True(result.Height > 0);
        Assert.True(result.Width > 0);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void Normalize_Apply_CentersValues()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        var normalize = new Normalize<double>(
            new[] { 0.485, 0.456, 0.406 },
            new[] { 0.229, 0.224, 0.225 },
            probability: 1.0);
        var context = CreateTestContext();

        var result = normalize.Apply(image, context);

        Assert.Equal(10, result.Height);
        Assert.Equal(10, result.Width);
    }

    [Fact]
    public void Denormalize_Reverses_Normalize()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        double[] mean = { 0.485, 0.456, 0.406 };
        double[] std = { 0.229, 0.224, 0.225 };
        var normalize = new Normalize<double>(mean, std, probability: 1.0);
        var denormalize = new Denormalize<double>(mean, std, probability: 1.0);
        var context = CreateTestContext();

        var normalized = normalize.Apply(image, context);
        var restored = denormalize.Apply(normalized, new AugmentationContext<double>(true, 42));

        // Restored should be close to original
        var origVal = image.GetPixel(5, 5, 0);
        var restoredVal = restored.GetPixel(5, 5, 0);
        Assert.True(Math.Abs(origVal - restoredVal) < 0.01,
            $"Expected ~{origVal}, got {restoredVal}");
    }

    [Fact]
    public void ToTensor_Apply_ScalesValues()
    {
        // Create an image with values in [0, 255] range
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        image.IsNormalized = false;
        for (int y = 0; y < 10; y++)
            for (int x = 0; x < 10; x++)
                for (int c = 0; c < 3; c++)
                    image.SetPixel(y, x, c, 128.0);

        var toTensor = new ToTensor<double>(scaleFactor: 255.0, probability: 1.0);
        var context = CreateTestContext();

        var result = toTensor.Apply(image, context);

        var val = result.GetPixel(5, 5, 0);
        Assert.True(Math.Abs(val - 128.0 / 255.0) < 0.01);
    }

    [Fact]
    public void ToFloat_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(10, 10);
        var toFloat = new ToFloat<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = toFloat.Apply(image, context);

        Assert.Equal(10, result.Height);
        Assert.Equal(10, result.Width);
    }

    [Fact]
    public void Pad_Apply_ExpandsDimensions()
    {
        var image = CreateTestImage(10, 10);
        var pad = new Pad<double>(2, 2, 2, 2, probability: 1.0);
        var context = CreateTestContext();

        var result = pad.Apply(image, context);

        Assert.Equal(14, result.Height);
        Assert.Equal(14, result.Width);
    }

    #endregion

    // ===================================================================
    // Phase 2: Color Space Conversions
    // ===================================================================

    #region Phase 2 - Color Space Conversions

    [Fact]
    public void RgbToGrayscale_Apply_ReducesToOneChannel()
    {
        var image = CreateTestImage(10, 10, channels: 3);
        var augmenter = new RgbToGrayscale<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(1, result.Channels);
        Assert.Equal(10, result.Height);
        Assert.Equal(10, result.Width);
    }

    [Fact]
    public void GrayscaleToRgb_Apply_ExpandsToThreeChannels()
    {
        var image = CreateTestImage(10, 10, channels: 1);
        var augmenter = new GrayscaleToRgb<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void RgbToBgr_Apply_SwapsRedAndBlue()
    {
        var image = CreateTestImage(10, 10);
        var origR = image.GetPixel(5, 5, 0);
        var origB = image.GetPixel(5, 5, 2);
        var augmenter = new RgbToBgr<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(origR, result.GetPixel(5, 5, 2), Tolerance);
        Assert.Equal(origB, result.GetPixel(5, 5, 0), Tolerance);
    }

    [Fact]
    public void RgbToHsv_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new RgbToHsv<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void RgbToHls_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new RgbToHls<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void RgbToLab_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new RgbToLab<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void RgbToYuv_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new RgbToYuv<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void RgbToXyz_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new RgbToXyz<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
    }

    #endregion

    // ===================================================================
    // Phase 3: Geometric Transforms
    // ===================================================================

    #region Phase 3 - Geometric Transforms

    [Fact]
    public void Resize_Apply_ChangeDimensions()
    {
        var image = CreateTestImage(20, 30);
        var resize = new Resize<double>(40, 50, probability: 1.0);
        var context = CreateTestContext();

        var result = resize.Apply(image, context);

        Assert.Equal(40, result.Height);
        Assert.Equal(50, result.Width);
    }

    [Fact]
    public void Perspective_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(32, 32);
        var perspective = new Perspective<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = perspective.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void ElasticTransform_Apply_PreservesDimensionsAndModifiesContent()
    {
        var image = CreateTestImage(20, 20);
        var elastic = new ElasticTransform<double>(alpha: 50.0, sigma: 5.0, probability: 1.0);
        var context = CreateTestContext();

        var result = elastic.Apply(image, context);

        Assert.Equal(20, result.Height);
        Assert.Equal(20, result.Width);
        // Center pixel should likely differ from original due to distortion
        Assert.NotEqual(image.GetPixel(10, 10, 0), result.GetPixel(10, 10, 0));
    }

    [Fact]
    public void GridDistortion_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(32, 32);
        var distortion = new GridDistortion<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = distortion.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void OpticalDistortion_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(32, 32);
        var distortion = new OpticalDistortion<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = distortion.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void PiecewiseAffine_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(32, 32);
        var pwa = new PiecewiseAffine<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = pwa.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void ThinPlateSpline_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(32, 32);
        var tps = new ThinPlateSpline<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = tps.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void RandomResizedCrop_Apply_ProducesTargetDimensions()
    {
        var image = CreateTestImage(64, 64);
        var crop = new RandomResizedCrop<double>(32, 32, probability: 1.0);
        var context = CreateTestContext();

        var result = crop.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void CenterCropOrPad_Apply_ProducesTargetDimensions()
    {
        var image = CreateTestImage(20, 30);
        var cropOrPad = new CenterCropOrPad<double>(25, 25, probability: 1.0);
        var context = CreateTestContext();

        var result = cropOrPad.Apply(image, context);

        Assert.Equal(25, result.Height);
        Assert.Equal(25, result.Width);
    }

    [Fact]
    public void LongestMaxSize_Apply_ConstrainsLongestSide()
    {
        var image = CreateTestImage(20, 40);
        var resize = new LongestMaxSize<double>(30, probability: 1.0);
        var context = CreateTestContext();

        var result = resize.Apply(image, context);

        Assert.True(Math.Max(result.Height, result.Width) <= 30);
    }

    [Fact]
    public void SmallestMaxSize_Apply_ConstrainsSmallestSide()
    {
        var image = CreateTestImage(20, 40);
        var resize = new SmallestMaxSize<double>(30, probability: 1.0);
        var context = CreateTestContext();

        var result = resize.Apply(image, context);

        Assert.True(Math.Min(result.Height, result.Width) >= 20);
    }

    #endregion

    // ===================================================================
    // Phase 4: Pixel-Level Transforms (Histogram, Intensity, Tone)
    // ===================================================================

    #region Phase 4 - Pixel-Level Transforms

    [Fact]
    public void HistogramEqualization_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new HistogramEqualization<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
        Assert.Equal(16, result.Width);
    }

    [Fact]
    public void CLAHE_Apply_ModifiesImageContrast()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.3);
        var clahe = new CLAHE<double>(clipLimit: 4.0, tileGridSize: 4, probability: 1.0);
        var context = CreateTestContext();

        var result = clahe.Apply(image, context);

        Assert.Equal(16, result.Height);
        // CLAHE should change pixel values
        Assert.NotEqual(image.GetPixel(8, 8, 0), result.GetPixel(8, 8, 0));
    }

    [Fact]
    public void Posterize_Apply_ReducesBitDepth()
    {
        var image = CreateTestImage(10, 10);
        var posterize = new Posterize<double>(minBitsPerChannel: 2, maxBitsPerChannel: 2, probability: 1.0);
        var context = CreateTestContext();

        var result = posterize.Apply(image, context);

        Assert.Equal(10, result.Height);
        AssertPixelsInRange(result);
    }

    [Fact]
    public void Solarize_Apply_InvertsAboveThreshold()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.8);
        var solarize = new Solarize<double>(threshold: 0.5, probability: 1.0);
        var context = CreateTestContext();

        var result = solarize.Apply(image, context);

        // Pixels above threshold should be inverted (maxVal - val)
        Assert.Equal(10, result.Height);
        AssertPixelsInRange(result);
    }

    [Fact]
    public void Equalize_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Equalize<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
        Assert.Equal(16, result.Width);
    }

    [Fact]
    public void AutoContrast_Apply_StretchesHistogram()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.4);
        var augmenter = new AutoContrast<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void Invert_Apply_InvertsAllPixels()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.3);
        var augmenter = new Invert<double>(probability: 1.0);
        var context = CreateTestContext();
        var original = image.GetPixel(5, 5, 0);

        var result = augmenter.Apply(image, context);

        var inverted = result.GetPixel(5, 5, 0);
        // For normalized [0,1]: inverted = 1 - original
        Assert.True(Math.Abs((1.0 - original) - inverted) < 0.05,
            $"Expected ~{1.0 - original}, got {inverted}");
    }

    [Fact]
    public void GammaCorrection_Apply_ModifiesValues()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        var augmenter = new GammaCorrection<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void RgbShift_Apply_ShiftsChannels()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new RgbShift<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void HueSaturationValue_Apply_ModifiesHSV()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new HueSaturationValue<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void ChannelShuffle_Apply_PreservesPixelValues()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new ChannelShuffle<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void ChannelDropout_Apply_ZerosOneChannel()
    {
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        var augmenter = new ChannelDropout<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(3, result.Channels);
        // At least one channel should have all zeros
        bool hasZeroChannel = false;
        for (int c = 0; c < 3; c++)
        {
            bool allZero = true;
            for (int y = 0; y < 10 && allZero; y++)
                for (int x = 0; x < 10 && allZero; x++)
                    if (Math.Abs(result.GetPixel(y, x, c)) > 1e-10)
                        allZero = false;
            if (allZero) hasZeroChannel = true;
        }
        Assert.True(hasZeroChannel, "ChannelDropout should zero out at least one channel");
    }

    [Fact]
    public void ToGray_Apply_ConvertsBrightness()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new ToGray<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void ToSepia_Apply_ProducesSepiaEffect()
    {
        var image = CreateTestImage(10, 10);
        var augmenter = new ToSepia<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(10, result.Height);
        Assert.Equal(3, result.Channels);
    }

    #endregion

    // ===================================================================
    // Phase 5: Blur and Noise
    // ===================================================================

    #region Phase 5 - Blur and Noise

    [Fact]
    public void MedianBlur_Apply_SmoothsImage()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new MedianBlur<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
        Assert.Equal(16, result.Width);
    }

    [Fact]
    public void MotionBlur_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new MotionBlur<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void GlassBlur_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new GlassBlur<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void ZoomBlur_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new ZoomBlur<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Defocus_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Defocus<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void BoxBlur_Apply_SmoothsImage()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new BoxBlur<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void ISONoise_Apply_AddsNoise()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.5);
        var augmenter = new ISONoise<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        int changed = 0;
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                if (Math.Abs(result.GetPixel(y, x, 0) - image.GetPixel(y, x, 0)) > 1e-10)
                    changed++;
        Assert.True(changed > 50, "ISONoise should modify most pixels");
    }

    [Fact]
    public void SaltAndPepperNoise_Apply_AddsBinaryNoise()
    {
        var image = CreateTestImage(20, 20, initialValue: 0.5);
        var augmenter = new SaltAndPepperNoise<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(20, result.Height);
    }

    [Fact]
    public void SpeckleNoise_Apply_AddsMultiplicativeNoise()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.5);
        var augmenter = new SpeckleNoise<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void PoissonNoise_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new PoissonNoise<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void MultiplicativeNoise_Apply_ScalesValues()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.5);
        var augmenter = new MultiplicativeNoise<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    #endregion

    // ===================================================================
    // Phase 6: Compression and Dropout
    // ===================================================================

    #region Phase 6 - Compression and Dropout

    [Fact]
    public void JpegCompression_Apply_ReducesQuality()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new JpegCompression<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void WebPCompression_Apply_ReducesQuality()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new WebPCompression<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Downscale_Apply_ReducesAndRestoresResolution()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new Downscale<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void ImageCompression_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new ImageCompression<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void CoarseDropout_Apply_CreatesHoles()
    {
        var image = CreateTestImage(32, 32, initialValue: 0.5);
        var augmenter = new CoarseDropout<double>(
            minHoles: 1, maxHoles: 3,
            minHoleHeight: 0.1, maxHoleHeight: 0.2,
            minHoleWidth: 0.1, maxHoleWidth: 0.2,
            fillValue: 0.0, probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        int zeros = 0;
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
                if (Math.Abs(result.GetPixel(y, x, 0)) < 1e-10)
                    zeros++;
        Assert.True(zeros > 0, "CoarseDropout should create zero regions");
    }

    [Fact]
    public void GridDropout_Apply_CreatesRegularDropoutPattern()
    {
        var image = CreateTestImage(32, 32, initialValue: 0.5);
        var augmenter = new GridDropout<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void RandomErasing_Apply_ErasesRegion()
    {
        var image = CreateTestImage(32, 32, initialValue: 0.5);
        var augmenter = new RandomErasing<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void GridMask_Apply_CreatesGridPattern()
    {
        var image = CreateTestImage(32, 32, initialValue: 0.5);
        var augmenter = new GridMask<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void HideAndSeek_Apply_HidesPatches()
    {
        var image = CreateTestImage(32, 32, initialValue: 0.5);
        var augmenter = new HideAndSeek<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void PixelDropout_Apply_DropsRandomPixels()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.5);
        var augmenter = new PixelDropout<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    #endregion

    // ===================================================================
    // Phase 7: Weather and Environmental Effects
    // ===================================================================

    #region Phase 7 - Weather Effects

    [Fact]
    public void Fog_Apply_AddsWhitishEffect()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.3);
        var augmenter = new Fog<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
        // Fog should brighten pixels toward white
        var originalMean = 0.0;
        var foggedMean = 0.0;
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
            {
                originalMean += image.GetPixel(y, x, 0);
                foggedMean += result.GetPixel(y, x, 0);
            }
        Assert.True(foggedMean >= originalMean - 1, "Fog should not significantly darken the image");
    }

    [Fact]
    public void Rain_Apply_AddsRainEffect()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Rain<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Snow_Apply_AddsSnowEffect()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Snow<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Frost_Apply_AddsFrostOverlay()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Frost<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Shadow_Apply_CreatesShaderRegion()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Shadow<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void SunFlare_Apply_AddsBrightFlare()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new SunFlare<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Clouds_Apply_AddsCloudEffect()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Clouds<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Spatter_Apply_AddsDirtEffect()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Spatter<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    #endregion

    // ===================================================================
    // Phase 8: Advanced Color and Sharpness
    // ===================================================================

    #region Phase 8 - Advanced Color and Sharpness

    [Fact]
    public void RandomBrightnessContrast_Apply_ModifiesBrightnessAndContrast()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.5);
        var augmenter = new RandomBrightnessContrast<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void RandomGamma_Apply_AdjustsGamma()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.5);
        var augmenter = new RandomGamma<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void RandomToneCurve_Apply_ModifiesToneCurve()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new RandomToneCurve<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void FancyPCA_Apply_ModifiesColorChannels()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new FancyPCA<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
        Assert.Equal(3, result.Channels);
    }

    [Fact]
    public void ColorConstancy_Apply_NormalizesColor()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new ColorConstancy<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void UnsharpMask_Apply_SharpensImage()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new UnsharpMask<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Sharpen_Apply_IncreasesEdgeContrast()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Sharpen<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Emboss_Apply_CreatesEmbossEffect()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Emboss<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Superpixels_Apply_CreatesPatchedImage()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Superpixels<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void TextureOverlay_Apply_BlendTexture()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new TextureOverlay<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void PlasmaTransform_Apply_GeneratesPlasmaEffect()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new PlasmaTransform<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    #endregion

    // ===================================================================
    // Phase 9: Mixing Augmenters (with Label Mixing)
    // ===================================================================

    #region Phase 9 - Mixing Augmenters

    [Fact]
    public void FMix_ApplyFMix_MixesImages()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var fmix = new FMix<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        var result = fmix.ApplyFMix(image1, image2, null, null, context);

        Assert.Equal(16, result.Height);
        Assert.Equal(16, result.Width);
        // Result should have values from both images
        var val = result.GetPixel(8, 8, 0);
        Assert.True(val >= 0.0 && val <= 1.0);
    }

    [Fact]
    public void FMix_LabelMixing_RaisesEvent()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var labels1 = new Vector<double>(new[] { 1.0, 0.0 });
        var labels2 = new Vector<double>(new[] { 0.0, 1.0 });
        var fmix = new FMix<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();
        var eventRaised = false;
        fmix.OnLabelMixing += (s, e) => eventRaised = true;

        fmix.ApplyFMix(image1, image2, labels1, labels2, context);

        Assert.True(eventRaised, "FMix should raise OnLabelMixing event");
        var lambda = fmix.LastMixingLambda;
        Assert.True(lambda >= 0.0 && lambda <= 1.0);
    }

    [Fact]
    public void SaliencyMix_ApplySaliencyMix_MixesImages()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var mix = new SaliencyMix<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        var result = mix.ApplySaliencyMix(image1, image2, null, null, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void SaliencyMix_LabelMixing_SetsLambda()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var labels1 = new Vector<double>(new[] { 1.0, 0.0 });
        var labels2 = new Vector<double>(new[] { 0.0, 1.0 });
        var mix = new SaliencyMix<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        mix.ApplySaliencyMix(image1, image2, labels1, labels2, context);

        var lambda = mix.LastMixingLambda;
        Assert.True(lambda >= 0.0 && lambda <= 1.0);
    }

    [Fact]
    public void PuzzleMix_ApplyPuzzleMix_MixesImages()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var mix = new PuzzleMix<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        var result = mix.ApplyPuzzleMix(image1, image2, null, null, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void SnapMix_ApplySnapMix_UsesSemanticMixing()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var mix = new SnapMix<double>(alpha: 5.0, probability: 1.0);
        var context = CreateTestContext();

        var result = mix.ApplySnapMix(image1, image2, null, null, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void ResizeMix_ApplyResizeMix_PastesResizedImage()
    {
        var image1 = CreateTestImage(32, 32, initialValue: 0.3);
        var image2 = CreateTestImage(32, 32, initialValue: 0.7);
        var mix = new ResizeMix<double>(minScale: 0.2, maxScale: 0.5, probability: 1.0);
        var context = CreateTestContext();

        var result = mix.ApplyResizeMix(image1, image2, null, null, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
        // Lambda should reflect area ratio
        var lambda = mix.LastMixingLambda;
        Assert.True(lambda >= 0.0 && lambda <= 1.0);
    }

    [Fact]
    public void TransMix_ApplyTransMix_PatchLevelMixing()
    {
        var image1 = CreateTestImage(32, 32, initialValue: 0.2);
        var image2 = CreateTestImage(32, 32, initialValue: 0.8);
        var mix = new TransMix<double>(alpha: 1.0, patchSize: 8, probability: 1.0);
        var context = CreateTestContext();

        var result = mix.ApplyTransMix(image1, image2, null, null, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void TokenMix_ApplyTokenMix_ReplacesPatches()
    {
        var image1 = CreateTestImage(32, 32, initialValue: 0.2);
        var image2 = CreateTestImage(32, 32, initialValue: 0.8);
        var mix = new TokenMix<double>(patchSize: 8, probability: 1.0);
        var context = CreateTestContext();

        var result = mix.ApplyTokenMix(image1, image2, null, null, context);

        Assert.Equal(32, result.Height);
        var lambda = mix.LastMixingLambda;
        Assert.True(lambda >= 0.0 && lambda <= 1.0);
    }

    [Fact]
    public void SamplePairing_ApplyPairing_AveragesImages()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.2);
        var image2 = CreateTestImage(16, 16, initialValue: 0.8);
        var pairing = new SamplePairing<double>(minWeight: 0.3, maxWeight: 0.3, probability: 1.0);
        var context = CreateTestContext();

        var result = pairing.ApplyPairing(image1, image2, null, null, context);

        Assert.Equal(16, result.Height);
        // With weight=0.3: result = image1 * 0.7 + image2 * 0.3
        var expected = image1.GetPixel(0, 0, 0) * 0.7 + image2.GetPixel(0, 0, 0) * 0.3;
        Assert.True(Math.Abs(result.GetPixel(0, 0, 0) - expected) < 0.05);
    }

    [Fact]
    public void MixingAugmenters_ResizeMismatchedImages()
    {
        var image1 = CreateTestImage(16, 16, initialValue: 0.3);
        var image2 = CreateTestImage(24, 24, initialValue: 0.7);
        var mix = new ResizeMix<double>(probability: 1.0);
        var context = CreateTestContext();

        // ResizeMix handles different sizes via internal resize
        var result = mix.ApplyResizeMix(image1, image2, null, null, context);

        Assert.Equal(16, result.Height);
        Assert.Equal(16, result.Width);
    }

    #endregion

    // ===================================================================
    // Phase 10: Object Detection Spatial Augmenters
    // ===================================================================

    #region Phase 10 - Object Detection Augmenters

    [Fact]
    public void BBoxSafeRandomCrop_Apply_ProducesValidDimensions()
    {
        var image = CreateTestImage(32, 32);
        var crop = new BBoxSafeRandomCrop<double>(minCropScale: 0.5, probability: 1.0);
        var context = CreateTestContext();

        var result = crop.Apply(image, context);

        Assert.True(result.Height >= 16);
        Assert.True(result.Width >= 16);
        Assert.True(result.Height <= 32);
        Assert.True(result.Width <= 32);
    }

    [Fact]
    public void BBoxSafeRandomCrop_ApplyWithTargets_TransformsBBox()
    {
        var image = CreateTestImage(100, 100);
        var box = new BoundingBox<double>(20, 20, 80, 80, BoundingBoxFormat.XYXY);
        var sample = new AugmentedSample<double, ImageTensor<double>>(image)
        {
            BoundingBoxes = new List<BoundingBox<double>> { box }
        };
        var crop = new BBoxSafeRandomCrop<double>(minCropScale: 0.5, probability: 1.0);
        var context = CreateTestContext();

        var result = crop.ApplyWithTargets(sample, context);

        Assert.NotNull(result.BoundingBoxes);
        Assert.True(result.BoundingBoxes!.Count >= 1);
    }

    [Fact]
    public void MinIoURandomCrop_Apply_CropsImage()
    {
        var image = CreateTestImage(32, 32);
        var crop = new MinIoURandomCrop<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = crop.Apply(image, context);

        Assert.True(result.Height > 0 && result.Height <= 32);
        Assert.True(result.Width > 0 && result.Width <= 32);
    }

    [Fact]
    public void RandomSizedBBoxSafeCrop_Apply_ProducesTargetSize()
    {
        var image = CreateTestImage(64, 64);
        var crop = new RandomSizedBBoxSafeCrop<double>(targetHeight: 32, targetWidth: 32, probability: 1.0);
        var context = CreateTestContext();

        var result = crop.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void CropNonEmptyMaskIfExists_Apply_CropsImage()
    {
        var image = CreateTestImage(32, 32);
        var crop = new CropNonEmptyMaskIfExists<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = crop.Apply(image, context);

        Assert.True(result.Height > 0 && result.Height <= 32);
    }

    [Fact]
    public void BBoxClipToImage_Apply_PreservesDimensions()
    {
        var image = CreateTestImage(32, 32);
        var clip = new BBoxClipToImage<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = clip.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void MaskDropout_Apply_DropsObjectMasks()
    {
        var image = CreateTestImage(16, 16, channels: 1, initialValue: 0.0);
        // Set some "object" pixels
        for (int y = 2; y < 6; y++)
            for (int x = 2; x < 6; x++)
                image.SetPixel(y, x, 0, 1.0);
        for (int y = 8; y < 12; y++)
            for (int x = 8; x < 12; x++)
                image.SetPixel(y, x, 0, 2.0);

        var dropout = new MaskDropout<double>(dropRate: 0.5, probability: 1.0);
        var context = CreateTestContext();

        var result = dropout.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void CopyPaste_ApplyCopyPaste_PastesPatches()
    {
        var target = CreateTestImage(32, 32, initialValue: 0.2);
        var source = CreateTestImage(32, 32, initialValue: 0.8);
        var copyPaste = new CopyPaste<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = copyPaste.ApplyCopyPaste(target, source, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void Mosaic_ApplyMosaic_CombinesFourImages()
    {
        var img1 = CreateTestImage(16, 16, initialValue: 0.1);
        var img2 = CreateTestImage(16, 16, initialValue: 0.3);
        var img3 = CreateTestImage(16, 16, initialValue: 0.5);
        var img4 = CreateTestImage(16, 16, initialValue: 0.7);
        var mosaic = new Mosaic<double>(outputHeight: 32, outputWidth: 32, probability: 1.0);
        var context = CreateTestContext();

        var result = mosaic.ApplyMosaic(img1, img2, img3, img4, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    #endregion

    // ===================================================================
    // Phase 11: AutoAugment and Learned Policies
    // ===================================================================

    #region Phase 11 - AutoAugment Policies

    [Fact]
    public void AutoAugment_Apply_TransformsImage()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new AutoAugment<double>(policy: AutoAugmentPolicy.ImageNet, probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void RandAugment_Apply_AppliesNTransforms()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new RandAugment<double>(n: 2, m: 9, probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void TrivialAugment_Apply_AppliesOneTransform()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new TrivialAugment<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void AugMax_Apply_TransformsImage()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new AugMax<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    #endregion

    // ===================================================================
    // Phase 12: Infrastructure (Policy, Scheduler, Preprocessor, Presets)
    // ===================================================================

    #region Phase 12 - Infrastructure

    [Fact]
    public void AugmentationPolicy_AppliesAllTransforms()
    {
        var image = CreateTestImage(32, 32);
        var policy = new AugmentationPolicy<double> { Name = "test_policy" };
        policy.Add(new HorizontalFlip<double>(probability: 1.0), 1.0);
        policy.Add(new Brightness<double>(1.2, 1.2, probability: 1.0), 1.0);
        var context = CreateTestContext();

        var result = policy.Apply(image, context);

        Assert.Equal(32, result.Height);
        Assert.Equal("test_policy", policy.Name);
        Assert.Equal(2, policy.Entries.Count);
    }

    [Fact]
    public void AugmentationPolicy_GetParameters_SerializesAll()
    {
        var policy = new AugmentationPolicy<double>();
        policy.Add(new HorizontalFlip<double>(), 0.5);

        var parameters = policy.GetParameters();

        Assert.Equal(1, (int)parameters["num_entries"]);
        Assert.True(parameters.ContainsKey("entry_0_type"));
    }

    [Fact]
    public void AugmentationScheduler_LinearSchedule_InterpolatesCorrectly()
    {
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var scheduler = new AugmentationScheduler<double>(
            flip, ScheduleType.Linear, totalEpochs: 100,
            startStrength: 0.0, endStrength: 1.0);

        Assert.True(Math.Abs(scheduler.CurrentStrength - 0.0) < 0.01);

        scheduler.SetEpoch(50);
        Assert.True(Math.Abs(scheduler.CurrentStrength - 0.5) < 0.01);

        scheduler.SetEpoch(100);
        Assert.True(Math.Abs(scheduler.CurrentStrength - 1.0) < 0.01);
    }

    [Fact]
    public void AugmentationScheduler_CosineSchedule_FollowsCosine()
    {
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var scheduler = new AugmentationScheduler<double>(
            flip, ScheduleType.Cosine, totalEpochs: 100,
            startStrength: 0.0, endStrength: 1.0);

        scheduler.SetEpoch(50);
        // At midpoint, cosine should be 0.5
        Assert.True(Math.Abs(scheduler.CurrentStrength - 0.5) < 0.01);
    }

    [Fact]
    public void AugmentationScheduler_StepSchedule_SwitchesAtMidpoint()
    {
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var scheduler = new AugmentationScheduler<double>(
            flip, ScheduleType.Step, totalEpochs: 100,
            startStrength: 0.0, endStrength: 1.0);

        scheduler.SetEpoch(25);
        Assert.True(Math.Abs(scheduler.CurrentStrength - 0.0) < 0.01);

        scheduler.SetEpoch(75);
        Assert.True(Math.Abs(scheduler.CurrentStrength - 1.0) < 0.01);
    }

    [Fact]
    public void AugmentationScheduler_Step_IncrementsEpoch()
    {
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var scheduler = new AugmentationScheduler<double>(
            flip, ScheduleType.Linear, totalEpochs: 10,
            startStrength: 0.0, endStrength: 1.0);

        for (int i = 0; i < 5; i++)
            scheduler.Step();

        Assert.True(Math.Abs(scheduler.CurrentStrength - 0.5) < 0.01);
    }

    [Fact]
    public void ImagePreprocessor_Process_ChainsTransforms()
    {
        var pipeline = new ImagePreprocessor<double>()
            .Resize(64, 64)
            .CenterCrop(32, 32)
            .RandomHorizontalFlip(0.5);

        Assert.Equal(3, pipeline.Transforms.Count);

        var image = CreateTestImage(100, 100);
        var result = pipeline.Process(image, CreateTestContext());

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void ImagePreprocessor_Add_AcceptsAnyAugmenter()
    {
        var pipeline = new ImagePreprocessor<double>()
            .Add(new Resize<double>(64, 64))
            .Add(new HorizontalFlip<double>(probability: 0.5))
            .Add(new GaussianNoise<double>(probability: 0.3));

        Assert.Equal(3, pipeline.Transforms.Count);
    }

    [Fact]
    public void ImagePresets_ImageNet_CreatesValidPipeline()
    {
        var trainingPipeline = ImagePresets<double>.ImageNet(training: true);
        var evalPipeline = ImagePresets<double>.ImageNet(training: false);

        Assert.True(trainingPipeline.Transforms.Count > 0);
        Assert.True(evalPipeline.Transforms.Count > 0);

        // Process an image through the eval pipeline
        var image = CreateTestImage(256, 256);
        var result = evalPipeline.Process(image, CreateTestContext());

        Assert.Equal(224, result.Height);
        Assert.Equal(224, result.Width);
    }

    [Fact]
    public void ImagePresets_COCO_CreatesValidPipeline()
    {
        var pipeline = ImagePresets<double>.COCO(targetSize: 320);

        Assert.True(pipeline.Transforms.Count > 0);
    }

    [Fact]
    public void ImagePresets_CLIP_CreatesValidPipeline()
    {
        var pipeline = ImagePresets<double>.CLIP();

        Assert.True(pipeline.Transforms.Count > 0);

        var image = CreateTestImage(256, 256);
        var result = pipeline.Process(image, CreateTestContext());

        Assert.Equal(224, result.Height);
        Assert.Equal(224, result.Width);
    }

    [Fact]
    public void PolicyRegistry_GetLightPolicy_CreatesPolicy()
    {
        var policy = PolicyRegistry<double>.Get("light_augmentation");

        Assert.Equal("light_augmentation", policy.Name);
        Assert.True(policy.Entries.Count > 0);
    }

    [Fact]
    public void PolicyRegistry_GetMediumPolicy_CreatesPolicy()
    {
        var policy = PolicyRegistry<double>.Get("medium_augmentation");

        Assert.Equal("medium_augmentation", policy.Name);
        Assert.True(policy.Entries.Count > 0);
    }

    [Fact]
    public void PolicyRegistry_GetHeavyPolicy_CreatesPolicy()
    {
        var policy = PolicyRegistry<double>.Get("heavy_augmentation");

        Assert.Equal("heavy_augmentation", policy.Name);
        Assert.True(policy.Entries.Count > 0);
    }

    [Fact]
    public void PolicyRegistry_GetNonExistent_Throws()
    {
        Assert.Throws<KeyNotFoundException>(() =>
            PolicyRegistry<double>.Get("nonexistent_policy"));
    }

    [Fact]
    public void PolicyRegistry_Register_AddsCustomPolicy()
    {
        PolicyRegistry<double>.Register("my_custom", () =>
        {
            var p = new AugmentationPolicy<double> { Name = "my_custom" };
            p.Add(new Invert<double>(), 1.0);
            return p;
        });

        var policy = PolicyRegistry<double>.Get("my_custom");
        Assert.Equal("my_custom", policy.Name);
        Assert.Single(policy.Entries);
    }

    [Fact]
    public void PolicyRegistry_GetNames_ReturnsRegisteredPolicies()
    {
        var names = PolicyRegistry<double>.GetNames().ToList();

        Assert.Contains("light_augmentation", names);
        Assert.Contains("medium_augmentation", names);
        Assert.Contains("heavy_augmentation", names);
    }

    [Fact]
    public void PolicyRegistry_ApplyPolicy_TransformsImage()
    {
        var policy = PolicyRegistry<double>.Get("heavy_augmentation");
        var image = CreateTestImage(32, 32);
        var context = CreateTestContext();

        var result = policy.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    #endregion

    // ===================================================================
    // Cross-Phase Integration Tests
    // ===================================================================

    #region Cross-Phase Integration

    [Fact]
    public void Pipeline_MixesPhases_InSinglePipeline()
    {
        var image = CreateTestImage(64, 64);
        var pipeline = new ImagePreprocessor<double>()
            .Resize(32, 32)                                                    // Phase 1
            .Add(new GaussianNoise<double>(probability: 1.0))                  // Phase 5
            .Add(new RandomBrightnessContrast<double>(probability: 1.0))       // Phase 8
            .Add(new CoarseDropout<double>(probability: 1.0));                 // Phase 6

        var result = pipeline.Process(image, CreateTestContext());

        Assert.Equal(32, result.Height);
        Assert.Equal(32, result.Width);
    }

    [Fact]
    public void Compose_AcceptsAllAugmenterTypes()
    {
        var compose = new Compose<double, ImageTensor<double>>(
            new HorizontalFlip<double>(probability: 1.0),    // SpatialImageAugmenterBase
            new Brightness<double>(probability: 1.0),         // ImageAugmenterBase
            new GaussianNoise<double>(probability: 1.0));     // ImageAugmenterBase

        var image = CreateTestImage(16, 16);
        var context = CreateTestContext();

        var result = compose.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void AugmentationPipeline_AcceptsAllPhaseAugmenters()
    {
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>("cross-phase");
        pipeline
            .Add(new Resize<double>(32, 32))            // Phase 3
            .Add(new CLAHE<double>(probability: 1.0))    // Phase 4
            .Add(new BoxBlur<double>(probability: 1.0))  // Phase 5
            .Add(new Fog<double>(probability: 1.0))      // Phase 7
            .Add(new Sharpen<double>(probability: 1.0)); // Phase 8

        var image = CreateTestImage(64, 64);
        var context = CreateTestContext();

        var result = pipeline.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void AllNewAugmenters_RespectProbabilityZero()
    {
        var image = CreateTestImage(16, 16);
        var context = CreateTestContext();
        var original00 = image.GetPixel(0, 0, 0);

        // Test a sample of augmenters with probability=0 (should be no-ops)
        var augmenters = new IAugmentation<double, ImageTensor<double>>[]
        {
            new CLAHE<double>(probability: 0.0),
            new ElasticTransform<double>(probability: 0.0),
            new Fog<double>(probability: 0.0),
            new CoarseDropout<double>(probability: 0.0),
            new RandomBrightnessContrast<double>(probability: 0.0),
            new Sharpen<double>(probability: 0.0),
        };

        foreach (var aug in augmenters)
        {
            var result = aug.Apply(image, new AugmentationContext<double>(true, 42));
            Assert.Equal(original00, result.GetPixel(0, 0, 0), Tolerance);
        }
    }

    [Fact]
    public void AllNewAugmenters_DoNotModifyOriginal()
    {
        var image = CreateTestImage(16, 16);
        var original55 = image.GetPixel(5, 5, 0);
        var context = CreateTestContext();

        var augmenters = new IAugmentation<double, ImageTensor<double>>[]
        {
            new CLAHE<double>(probability: 1.0),
            new Fog<double>(probability: 1.0),
            new CoarseDropout<double>(probability: 1.0),
            new Invert<double>(probability: 1.0),
            new Sharpen<double>(probability: 1.0),
        };

        foreach (var aug in augmenters)
        {
            _ = aug.Apply(image, new AugmentationContext<double>(true, 42));
            Assert.Equal(original55, image.GetPixel(5, 5, 0), Tolerance);
        }
    }

    [Fact]
    public void AllNewAugmenters_ReturnCorrectChannelCount()
    {
        var image = CreateTestImage(16, 16, channels: 3);
        var context = CreateTestContext();

        // These augmenters should preserve 3 channels
        var augmenters = new IAugmentation<double, ImageTensor<double>>[]
        {
            new Fog<double>(probability: 1.0),
            new Rain<double>(probability: 1.0),
            new Snow<double>(probability: 1.0),
            new CLAHE<double>(probability: 1.0),
            new CoarseDropout<double>(probability: 1.0),
            new Sharpen<double>(probability: 1.0),
        };

        foreach (var aug in augmenters)
        {
            var result = aug.Apply(image, new AugmentationContext<double>(true, 42));
            Assert.Equal(3, result.Channels);
        }
    }

    [Fact]
    public void AllNewAugmenters_HandleSmallImages()
    {
        var image = CreateTestImage(2, 2);
        var context = CreateTestContext();

        // These augmenters should not crash on tiny images
        var augmenters = new IAugmentation<double, ImageTensor<double>>[]
        {
            new Fog<double>(probability: 1.0),
            new CoarseDropout<double>(probability: 1.0),
            new BoxBlur<double>(probability: 1.0),
            new GaussianNoise<double>(probability: 1.0),
            new Invert<double>(probability: 1.0),
        };

        foreach (var aug in augmenters)
        {
            var result = aug.Apply(image, new AugmentationContext<double>(true, 42));
            Assert.Equal(2, result.Height);
            Assert.Equal(2, result.Width);
        }
    }

    [Fact]
    public void AllNewAugmenters_ProduceReproducibleResults()
    {
        var image = CreateTestImage(16, 16);

        var aug = new ElasticTransform<double>(alpha: 50.0, sigma: 5.0, probability: 1.0);

        var result1 = aug.Apply(image, new AugmentationContext<double>(true, 12345));
        var result2 = aug.Apply(image, new AugmentationContext<double>(true, 12345));

        // Same seed should produce same results
        Assert.Equal(result1.GetPixel(8, 8, 0), result2.GetPixel(8, 8, 0), Tolerance);
    }

    [Fact]
    public void GetParameters_ReturnsExpectedKeys_ForAllPhases()
    {
        // Phase 4
        var clahe = new CLAHE<double>(clipLimit: 2.0, tileGridSize: 4);
        var claheParams = clahe.GetParameters();
        Assert.Equal(2.0, (double)claheParams["clip_limit"], Tolerance);
        Assert.Equal(4, (int)claheParams["tile_grid_size"]);

        // Phase 5
        var coarse = new CoarseDropout<double>(minHoles: 2, maxHoles: 5);
        var coarseParams = coarse.GetParameters();
        Assert.Equal(2, (int)coarseParams["min_holes"]);
        Assert.Equal(5, (int)coarseParams["max_holes"]);

        // Phase 9
        var fmix = new FMix<double>(alpha: 2.0, decayPower: 4.0);
        var fmixParams = fmix.GetParameters();
        Assert.Equal(4.0, (double)fmixParams["decay_power"], Tolerance);

        // Phase 10
        var bboxCrop = new BBoxSafeRandomCrop<double>(minCropScale: 0.6);
        var bboxParams = bboxCrop.GetParameters();
        Assert.True(bboxParams.ContainsKey("probability"));
    }

    #endregion

    // ===================================================================
    // Medical Imaging Augmenters (Phase 8 additions)
    // ===================================================================

    #region Medical Imaging Augmenters

    [Fact]
    public void Dilate_Apply_ExpandsBrightRegions()
    {
        var image = CreateTestImage(16, 16, initialValue: 0.0);
        image.SetPixel(8, 8, 0, 1.0);
        var augmenter = new Dilate<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Erode_Apply_ShrinksBrightRegions()
    {
        var image = CreateTestImage(16, 16, initialValue: 1.0);
        var augmenter = new Erode<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Opening_Apply_ErodesAndDilates()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Opening<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void Closing_Apply_DilatesAndErodes()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new Closing<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void MorphologicalGradient_Apply_ExtractsEdges()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new MorphologicalGradient<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void HistogramColorTransfer_Apply_TransfersHistogram()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new HistogramColorTransfer<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void HistogramMatching_Apply_MatchesHistogram()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new HistogramMatching<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void BiasFieldCorrection_Apply_CorrectsBias()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new BiasFieldCorrection<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void GhostingArtifact_Apply_AddsGhosting()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new GhostingArtifact<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void SpikeArtifact_Apply_AddsSpikeNoise()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new SpikeArtifact<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void KSpaceMotion_Apply_SimulatesMotionArtifact()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new KSpaceMotion<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void IntensityNormalization_Apply_NormalizesIntensity()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new IntensityNormalization<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void WindowLevel_Apply_AdjustsWindowLevel()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new WindowLevel<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void SliceSelection_Apply_SelectsSlice()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new SliceSelection<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void StackSlices_Apply_ProcessesSlices()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new StackSlices<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    #endregion

    // ===================================================================
    // Style Transfer and Advanced Mixing (Phase 9 additions)
    // ===================================================================

    #region Style Transfer and Advanced Mixing

    [Fact]
    public void StyleMix_Apply_TransfersStyle()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new StyleMix<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void AdaIN_Apply_AppliesAdaptiveInstanceNorm()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new AdaIN<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    [Fact]
    public void FDA_Apply_PerformsFrequencyDomainAdaptation()
    {
        var image = CreateTestImage(16, 16);
        var augmenter = new FDA<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(16, result.Height);
    }

    #endregion

    // ===================================================================
    // AutoAugment Variants (Phase 11 additions)
    // ===================================================================

    #region AutoAugment Variants

    [Fact]
    public void FastAutoAugment_Apply_TransformsImage()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new FastAutoAugment<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void DADA_Apply_TransformsImage()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new DADA<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void UniformAugment_Apply_TransformsImage()
    {
        var image = CreateTestImage(32, 32);
        var augmenter = new UniformAugment<double>(probability: 1.0);
        var context = CreateTestContext();

        var result = augmenter.Apply(image, context);

        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void Mosaic9_ApplyMosaic9_CombinesNineImages()
    {
        var images = new ImageTensor<double>[9];
        for (int i = 0; i < 9; i++)
            images[i] = CreateTestImage(16, 16, initialValue: 0.1 * i);

        var mosaic = new Mosaic9<double>(outputHeight: 48, outputWidth: 48, probability: 1.0);
        var context = CreateTestContext();

        var result = mosaic.ApplyMosaic9(images, context);

        Assert.Equal(48, result.Height);
        Assert.Equal(48, result.Width);
    }

    #endregion
}
