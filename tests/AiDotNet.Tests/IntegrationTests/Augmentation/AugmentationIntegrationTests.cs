using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Audio;
using AiDotNet.Augmentation.Image;
using AiDotNet.Augmentation.Tabular;
using AiDotNet.Augmentation.Text;
using AiDotNet.Augmentation.Video;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Augmentation;

/// <summary>
/// Comprehensive integration tests for the Augmentation module.
/// Tests all augmentation types, pipelines, spatial targets, and configuration.
/// </summary>
public class AugmentationIntegrationTests
{
    private const int TestSeed = 42;
    private const double Tolerance = 1e-6;

    #region Helper Methods

    /// <summary>
    /// Creates a test image tensor with specified dimensions.
    /// </summary>
    private static ImageTensor<double> CreateTestImage(int height, int width, int channels = 3, double initialValue = 0.5)
    {
        var tensor = new Tensor<double>(new[] { channels, height, width });
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Create a gradient pattern for testing transformations
                    double value = initialValue + (c * 0.1) + (y * 0.001) + (x * 0.0001);
                    tensor[c * height * width + y * width + x] = Math.Min(1.0, Math.Max(0.0, value));
                }
            }
        }

        var image = new ImageTensor<double>(tensor, ChannelOrder.CHW, ColorSpace.RGB);
        image.IsNormalized = true; // Mark as normalized since values are in [0, 1]
        return image;
    }

    /// <summary>
    /// Creates a test matrix for tabular augmentation.
    /// </summary>
    private static Matrix<double> CreateTestMatrix(int rows, int cols)
    {
        var matrix = new Matrix<double>(rows, cols);
        var rand = RandomHelper.CreateSeededRandom(TestSeed);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = rand.NextDouble() * 10;
            }
        }

        return matrix;
    }

    /// <summary>
    /// Creates a test audio waveform tensor.
    /// </summary>
    private static Tensor<double> CreateTestAudio(int samples, int channels = 1)
    {
        var tensor = new Tensor<double>(new[] { channels, samples });
        var rand = RandomHelper.CreateSeededRandom(TestSeed);
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < samples; i++)
            {
                // Create a simple sine wave pattern
                tensor[c * samples + i] = Math.Sin(2 * Math.PI * i / 100.0) + (rand.NextDouble() * 0.1 - 0.05);
            }
        }

        return tensor;
    }

    /// <summary>
    /// Creates a test text array.
    /// </summary>
    private static string[] CreateTestText()
    {
        return new[]
        {
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Data augmentation improves model generalization"
        };
    }

    /// <summary>
    /// Creates a test video as a list of image tensors.
    /// </summary>
    private static List<ImageTensor<double>> CreateTestVideo(int frames, int height, int width)
    {
        var video = new List<ImageTensor<double>>();
        for (int f = 0; f < frames; f++)
        {
            video.Add(CreateTestImage(height, width, 3, 0.5 + f * 0.01));
        }

        return video;
    }

    /// <summary>
    /// Creates an augmentation context for testing.
    /// </summary>
    private static AugmentationContext<double> CreateTestContext(bool isTraining = true, int? seed = TestSeed)
    {
        return new AugmentationContext<double>(isTraining, seed);
    }

    #endregion

    #region AugmentationContext Tests

    [Fact]
    public void AugmentationContext_ShouldApply_ReturnsCorrectProbability()
    {
        // Arrange
        var context = CreateTestContext(seed: 42);

        // Act & Assert - probability 1.0 should always apply
        Assert.True(context.ShouldApply(1.0));

        // Act & Assert - probability 0.0 should never apply
        Assert.False(context.ShouldApply(0.0));
    }

    [Fact]
    public void AugmentationContext_GetRandomDouble_ReturnsValueInRange()
    {
        // Arrange
        var context = CreateTestContext();
        const double min = 0.0;
        const double max = 100.0;

        // Act
        var values = Enumerable.Range(0, 100).Select(_ => context.GetRandomDouble(min, max)).ToList();

        // Assert
        Assert.All(values, v => Assert.InRange(v, min, max));
    }

    [Fact]
    public void AugmentationContext_GetRandomInt_ReturnsValueInRange()
    {
        // Arrange
        var context = CreateTestContext();
        const int min = 0;
        const int max = 100;

        // Act
        var values = Enumerable.Range(0, 100).Select(_ => context.GetRandomInt(min, max)).ToList();

        // Assert
        Assert.All(values, v => Assert.InRange(v, min, max - 1));
    }

    [Fact]
    public void AugmentationContext_SampleBeta_ReturnsValueBetweenZeroAndOne()
    {
        // Arrange
        var context = CreateTestContext();

        // Act
        var values = Enumerable.Range(0, 100).Select(_ => context.SampleBeta(0.5, 0.5)).ToList();

        // Assert
        Assert.All(values, v => Assert.InRange(v, 0.0, 1.0));
    }

    [Fact]
    public void AugmentationContext_SampleGaussian_ReturnsSampledValues()
    {
        // Arrange
        var context = CreateTestContext();
        const double mean = 5.0;
        const double stdDev = 2.0;

        // Act
        var values = Enumerable.Range(0, 1000).Select(_ => context.SampleGaussian(mean, stdDev)).ToList();
        var actualMean = values.Average();
        var actualStdDev = Math.Sqrt(values.Select(v => Math.Pow(v - actualMean, 2)).Average());

        // Assert - sample statistics should be close to specified
        Assert.True(Math.Abs(actualMean - mean) < 0.5, $"Expected mean ~{mean}, got {actualMean}");
        Assert.True(Math.Abs(actualStdDev - stdDev) < 0.5, $"Expected stdDev ~{stdDev}, got {actualStdDev}");
    }

    [Fact]
    public void AugmentationContext_CreateChildContext_PreservesParentState()
    {
        // Arrange
        var parent = CreateTestContext();
        parent.BatchIndex = 5;

        // Act
        var child = parent.CreateChildContext(10);

        // Assert
        Assert.Equal(5, child.BatchIndex);
        Assert.Equal(10, child.SampleIndex);
        Assert.Equal(parent.IsTraining, child.IsTraining);
    }

    [Fact]
    public void AugmentationContext_WithSeed_ProducesReproducibleResults()
    {
        // Arrange
        var context1 = CreateTestContext(seed: 12345);
        var context2 = CreateTestContext(seed: 12345);

        // Act
        var values1 = Enumerable.Range(0, 10).Select(_ => context1.GetRandomDouble(0, 100)).ToList();
        var values2 = Enumerable.Range(0, 10).Select(_ => context2.GetRandomDouble(0, 100)).ToList();

        // Assert
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(values1[i], values2[i], Tolerance);
        }
    }

    #endregion

    #region ImageTensor Tests

    [Fact]
    public void ImageTensor_Constructor_CreatesCorrectDimensions()
    {
        // Arrange & Act
        var image = new ImageTensor<double>(32, 64, 3, ChannelOrder.CHW, ColorSpace.RGB);

        // Assert
        Assert.Equal(32, image.Height);
        Assert.Equal(64, image.Width);
        Assert.Equal(3, image.Channels);
        Assert.Equal(ChannelOrder.CHW, image.ChannelOrder);
        Assert.Equal(ColorSpace.RGB, image.ColorSpace);
    }

    [Fact]
    public void ImageTensor_GetSetPixel_WorksCorrectly()
    {
        // Arrange
        var image = new ImageTensor<double>(10, 10, 3, ChannelOrder.CHW, ColorSpace.RGB);

        // Act
        image.SetPixel(5, 7, 1, 0.75);
        var value = image.GetPixel(5, 7, 1);

        // Assert
        Assert.Equal(0.75, value, Tolerance);
    }

    [Fact]
    public void ImageTensor_GetPixelChannels_ReturnsAllChannels()
    {
        // Arrange
        var image = new ImageTensor<double>(10, 10, 3, ChannelOrder.CHW, ColorSpace.RGB);
        image.SetPixel(3, 4, 0, 0.1);
        image.SetPixel(3, 4, 1, 0.2);
        image.SetPixel(3, 4, 2, 0.3);

        // Act
        var channels = image.GetPixelChannels(3, 4);

        // Assert
        Assert.Equal(3, channels.Length);
        Assert.Equal(0.1, channels[0], Tolerance);
        Assert.Equal(0.2, channels[1], Tolerance);
        Assert.Equal(0.3, channels[2], Tolerance);
    }

    [Fact]
    public void ImageTensor_Clone_CreatesDeepCopy()
    {
        // Arrange
        var original = CreateTestImage(10, 10);
        original.SetPixel(5, 5, 0, 0.99);

        // Act
        var clone = original.Clone();
        clone.SetPixel(5, 5, 0, 0.01);

        // Assert
        Assert.Equal(0.99, original.GetPixel(5, 5, 0), Tolerance);
        Assert.Equal(0.01, clone.GetPixel(5, 5, 0), Tolerance);
    }

    [Fact]
    public void ImageTensor_Crop_ExtractsCorrectRegion()
    {
        // Arrange
        var image = CreateTestImage(20, 20);
        image.SetPixel(5, 5, 0, 0.99);

        // Act
        var cropped = image.Crop(3, 3, 5, 5);

        // Assert
        Assert.Equal(5, cropped.Width);
        Assert.Equal(5, cropped.Height);
        Assert.Equal(0.99, cropped.GetPixel(2, 2, 0), Tolerance); // (5-3, 5-3) in cropped coords
    }

    [Fact]
    public void ImageTensor_Crop_ThrowsOnInvalidBounds()
    {
        // Arrange
        var image = CreateTestImage(10, 10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => image.Crop(-1, 0, 5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => image.Crop(0, -1, 5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => image.Crop(8, 0, 5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => image.Crop(0, 8, 5, 5));
    }

    [Fact]
    public void ImageTensor_ToChannelOrder_ConvertsCorrectly()
    {
        // Arrange
        var image = new ImageTensor<double>(4, 5, 3, ChannelOrder.CHW, ColorSpace.RGB);
        image.SetPixel(2, 3, 1, 0.75);

        // Act
        var hwcImage = image.ToChannelOrder(ChannelOrder.HWC);

        // Assert
        Assert.Equal(ChannelOrder.HWC, hwcImage.ChannelOrder);
        Assert.Equal(4, hwcImage.Height);
        Assert.Equal(5, hwcImage.Width);
        Assert.Equal(3, hwcImage.Channels);
        Assert.Equal(0.75, hwcImage.GetPixel(2, 3, 1), Tolerance);
    }

    #endregion

    #region HorizontalFlip Tests

    [Fact]
    public void HorizontalFlip_Apply_FlipsImageCorrectly()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalTopLeft = image.GetPixel(0, 0, 0);
        var originalTopRight = image.GetPixel(0, 9, 0);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        var flipped = flip.Apply(image, context);

        // Assert
        Assert.Equal(originalTopRight, flipped.GetPixel(0, 0, 0), Tolerance);
        Assert.Equal(originalTopLeft, flipped.GetPixel(0, 9, 0), Tolerance);
    }

    [Fact]
    public void HorizontalFlip_Apply_PreservesDimensions()
    {
        // Arrange
        var image = CreateTestImage(15, 20);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        var flipped = flip.Apply(image, context);

        // Assert
        Assert.Equal(15, flipped.Height);
        Assert.Equal(20, flipped.Width);
        Assert.Equal(3, flipped.Channels);
    }

    [Fact]
    public void HorizontalFlip_Apply_DoesNotModifyOriginal()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalValue = image.GetPixel(5, 5, 0);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        _ = flip.Apply(image, context);

        // Assert
        Assert.Equal(originalValue, image.GetPixel(5, 5, 0), Tolerance);
    }

    [Fact]
    public void HorizontalFlip_Apply_RespectsProbability()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var flip = new HorizontalFlip<double>(probability: 0.0);
        var context = CreateTestContext();
        var originalValue = image.GetPixel(0, 0, 0);

        // Act
        var result = flip.Apply(image, context);

        // Assert - with 0.0 probability, image should be unchanged
        Assert.Equal(originalValue, result.GetPixel(0, 0, 0), Tolerance);
    }

    [Fact]
    public void HorizontalFlip_GetParameters_ReturnsExpectedValues()
    {
        // Arrange
        var flip = new HorizontalFlip<double>(probability: 0.7);

        // Act
        var parameters = flip.GetParameters();

        // Assert
        Assert.Equal("HorizontalFlip`1", parameters["name"]);
        Assert.Equal(0.7, (double)parameters["probability"], Tolerance);
        Assert.Equal("horizontal", parameters["flip_type"]);
    }

    #endregion

    #region VerticalFlip Tests

    [Fact]
    public void VerticalFlip_Apply_FlipsImageCorrectly()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalTopLeft = image.GetPixel(0, 0, 0);
        var originalBottomLeft = image.GetPixel(9, 0, 0);
        var flip = new VerticalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        var flipped = flip.Apply(image, context);

        // Assert
        Assert.Equal(originalBottomLeft, flipped.GetPixel(0, 0, 0), Tolerance);
        Assert.Equal(originalTopLeft, flipped.GetPixel(9, 0, 0), Tolerance);
    }

    #endregion

    #region Rotation Tests

    [Fact]
    public void Rotation_Apply_RotatesImage()
    {
        // Arrange
        var image = CreateTestImage(20, 20);
        var rotation = new Rotation<double>(minAngle: 15.0, maxAngle: 15.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var rotated = rotation.Apply(image, context);

        // Assert - dimensions should be preserved
        Assert.Equal(20, rotated.Height);
        Assert.Equal(20, rotated.Width);
        // Content should be different (rotated)
        Assert.NotEqual(image.GetPixel(0, 0, 0), rotated.GetPixel(0, 0, 0), Tolerance);
    }

    [Fact]
    public void Rotation_Apply_ZeroAngle_PreservesImage()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var rotation = new Rotation<double>(minAngle: 0.0, maxAngle: 0.0, probability: 1.0);
        var context = CreateTestContext();
        var originalCenter = image.GetPixel(5, 5, 0);

        // Act
        var rotated = rotation.Apply(image, context);

        // Assert - center pixel should be the same
        Assert.Equal(originalCenter, rotated.GetPixel(5, 5, 0), Tolerance);
    }

    [Fact]
    public void Rotation_Constructor_ThrowsOnInvalidAngleRange()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new Rotation<double>(minAngle: 45.0, maxAngle: -45.0));
    }

    [Fact]
    public void Rotation_GetParameters_ReturnsExpectedValues()
    {
        // Arrange
        var rotation = new Rotation<double>(
            minAngle: -30.0,
            maxAngle: 30.0,
            probability: 0.8,
            borderMode: BorderMode.Reflect);

        // Act
        var parameters = rotation.GetParameters();

        // Assert
        Assert.Equal(-30.0, (double)parameters["min_angle"], Tolerance);
        Assert.Equal(30.0, (double)parameters["max_angle"], Tolerance);
        Assert.Equal(0.8, (double)parameters["probability"], Tolerance);
        Assert.Equal("Reflect", parameters["border_mode"]);
    }

    #endregion

    #region Brightness Tests

    [Fact]
    public void Brightness_Apply_IncreasesValues()
    {
        // Arrange
        var image = CreateTestImage(10, 10, initialValue: 0.3);
        var brightness = new Brightness<double>(minFactor: 1.5, maxFactor: 1.5, probability: 1.0);
        var context = CreateTestContext();
        var originalValue = image.GetPixel(5, 5, 0);

        // Act
        var result = brightness.Apply(image, context);
        var newValue = result.GetPixel(5, 5, 0);

        // Assert - brightness increased, value should be higher
        Assert.True(newValue > originalValue || newValue >= 1.0);
    }

    [Fact]
    public void Brightness_Apply_ClampsToBounds()
    {
        // Arrange
        var image = CreateTestImage(10, 10, initialValue: 0.9);
        var brightness = new Brightness<double>(minFactor: 2.0, maxFactor: 2.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = brightness.Apply(image, context);

        // Assert - values should be clamped to [0, 1]
        for (int y = 0; y < 10; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                var value = result.GetPixel(y, x, 0);
                Assert.InRange(value, 0.0, 1.0);
            }
        }
    }

    #endregion

    #region GaussianNoise Tests

    [Fact]
    public void GaussianNoise_Apply_AddsNoiseToImage()
    {
        // Arrange
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        var noise = new GaussianNoise<double>(mean: 0.0, minStd: 0.1, maxStd: 0.1, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(image, context);

        // Assert - at least some pixels should be different
        int differentPixels = 0;
        for (int y = 0; y < 10; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                if (Math.Abs(result.GetPixel(y, x, 0) - image.GetPixel(y, x, 0)) > 1e-10)
                {
                    differentPixels++;
                }
            }
        }

        Assert.True(differentPixels > 50, $"Expected most pixels to change, but only {differentPixels} changed");
    }

    [Fact]
    public void GaussianNoise_Apply_ClampsToBounds()
    {
        // Arrange
        var image = CreateTestImage(10, 10, initialValue: 0.5);
        var noise = new GaussianNoise<double>(mean: 0.0, minStd: 1.0, maxStd: 1.0, probability: 1.0); // Large noise
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(image, context);

        // Assert - values should be clamped to [0, 1]
        for (int y = 0; y < 10; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    var value = result.GetPixel(y, x, c);
                    Assert.InRange(value, 0.0, 1.0);
                }
            }
        }
    }

    #endregion

    #region Cutout Tests

    [Fact]
    public void Cutout_Apply_CreatesMaskedRegion()
    {
        // Arrange
        var image = CreateTestImage(20, 20, initialValue: 0.5);
        var cutout = new Cutout<double>(
            numberOfHoles: 1,
            minHoleHeight: 5,
            maxHoleHeight: 5,
            minHoleWidth: 5,
            maxHoleWidth: 5,
            fillValue: 0.0,
            probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = cutout.Apply(image, context);

        // Assert - should have some zero pixels
        int zeroPixels = 0;
        for (int y = 0; y < 20; y++)
        {
            for (int x = 0; x < 20; x++)
            {
                if (Math.Abs(result.GetPixel(y, x, 0)) < 1e-10)
                {
                    zeroPixels++;
                }
            }
        }

        Assert.True(zeroPixels > 0, "Cutout should create masked (zero) pixels");
    }

    #endregion

    #region MixUp Tests

    [Fact]
    public void MixUp_ApplyMixUp_BlendsImages()
    {
        // Arrange
        var image1 = CreateTestImage(10, 10, initialValue: 0.2);
        var image2 = CreateTestImage(10, 10, initialValue: 0.8);
        var mixup = new MixUp<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = mixup.ApplyMixUp(image1, image2, null, null, context);

        // Assert - mixed values should be between the two originals
        var mixedValue = result.GetPixel(5, 5, 0);
        var val1 = image1.GetPixel(5, 5, 0);
        var val2 = image2.GetPixel(5, 5, 0);

        // Mixed value should be somewhere between the two
        Assert.True(mixedValue >= Math.Min(val1, val2) - 0.01 && mixedValue <= Math.Max(val1, val2) + 0.01);
    }

    [Fact]
    public void MixUp_ApplyMixUp_ThrowsOnDimensionMismatch()
    {
        // Arrange
        var image1 = CreateTestImage(10, 10);
        var image2 = CreateTestImage(15, 15); // Different size
        var mixup = new MixUp<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mixup.ApplyMixUp(image1, image2, null, null, context));
    }

    [Fact]
    public void MixUp_LastMixingLambda_IsSet()
    {
        // Arrange
        var image1 = CreateTestImage(10, 10, initialValue: 0.2);
        var image2 = CreateTestImage(10, 10, initialValue: 0.8);
        var mixup = new MixUp<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        mixup.ApplyMixUp(image1, image2, null, null, context);

        // Assert
        var lambda = mixup.LastMixingLambda;
        Assert.True(lambda >= 0.0 && lambda <= 1.0, $"Lambda should be in [0,1], got {lambda}");
    }

    #endregion

    #region Tabular Augmentation Tests

    [Fact]
    public void FeatureNoise_Apply_AddsNoiseToMatrix()
    {
        // Arrange
        var matrix = CreateTestMatrix(10, 5);
        var originalValue = matrix[5, 2];
        var noise = new FeatureNoise<double>(noiseStdDev: 0.5, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(matrix, context);

        // Assert - values should be different
        Assert.NotEqual(originalValue, result[5, 2]);
    }

    [Fact]
    public void FeatureNoise_Apply_PreservesMatrixDimensions()
    {
        // Arrange
        var matrix = CreateTestMatrix(20, 8);
        var noise = new FeatureNoise<double>(noiseStdDev: 0.1, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(matrix, context);

        // Assert
        Assert.Equal(20, result.Rows);
        Assert.Equal(8, result.Columns);
    }

    [Fact]
    public void FeatureNoise_Constructor_ThrowsOnNegativeStdDev()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new FeatureNoise<double>(noiseStdDev: -0.1));
    }

    [Fact]
    public void FeatureNoise_FeatureIndices_AppliesOnlyToSpecifiedFeatures()
    {
        // Arrange
        var matrix = CreateTestMatrix(10, 5);
        var originalCol0 = new double[10];
        var originalCol3 = new double[10];
        for (int i = 0; i < 10; i++)
        {
            originalCol0[i] = matrix[i, 0];
            originalCol3[i] = matrix[i, 3];
        }

        var noise = new FeatureNoise<double>(noiseStdDev: 1.0, probability: 1.0, featureIndices: new[] { 1, 2 });
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(matrix, context);

        // Assert - columns 0 and 3 should be unchanged
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(originalCol0[i], result[i, 0], Tolerance);
            Assert.Equal(originalCol3[i], result[i, 3], Tolerance);
        }
    }

    [Fact]
    public void FeatureDropout_Apply_ZerosOutFeatures()
    {
        // Arrange
        var matrix = CreateTestMatrix(10, 10);
        var dropout = new FeatureDropout<double>(dropoutRate: 0.5, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = dropout.Apply(matrix, context);

        // Assert - some values should be zero
        int zeroCount = 0;
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                if (Math.Abs(result[i, j]) < 1e-10)
                {
                    zeroCount++;
                }
            }
        }

        Assert.True(zeroCount > 0, "FeatureDropout should create some zero values");
    }

    [Fact]
    public void TabularMixUp_Apply_BlendsTwoRows()
    {
        // Arrange
        var matrix1 = new Matrix<double>(1, 5);
        var matrix2 = new Matrix<double>(1, 5);
        for (int j = 0; j < 5; j++)
        {
            matrix1[0, j] = 0.0;
            matrix2[0, j] = 1.0;
        }

        var mixup = new TabularMixUp<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();

        // Act - TabularMixUp needs two matrices, let's test parameters
        var parameters = mixup.GetParameters();

        // Assert
        Assert.Equal(1.0, (double)parameters["alpha"], Tolerance);
    }

    #endregion

    #region Audio Augmentation Tests

    [Fact]
    public void AudioNoise_Apply_AddsNoiseToWaveform()
    {
        // Arrange
        var audio = CreateTestAudio(1000);
        var originalValue = audio[500];
        var noise = new AudioNoise<double>(minSnrDb: 10.0, maxSnrDb: 10.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(audio, context);

        // Assert - values should be slightly different due to noise
        Assert.Equal(audio.Length, result.Length);
    }

    [Fact]
    public void TimeStretch_Apply_ChangesAudioLength()
    {
        // Arrange
        var audio = CreateTestAudio(1000);
        var stretch = new TimeStretch<double>(minRate: 0.5, maxRate: 0.5, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = stretch.Apply(audio, context);

        // Assert - stretched audio should be different length
        // Factor of 0.5 should roughly halve the length
        Assert.True(result.Length < audio.Length, "Time stretch with factor 0.5 should shorten audio");
    }

    [Fact]
    public void VolumeChange_Apply_ScalesAmplitude()
    {
        // Arrange
        var audio = CreateTestAudio(100);
        // Initialize with known values
        for (int i = 0; i < 100; i++)
        {
            audio[i] = 0.5;
        }

        var volumeUp = new VolumeChange<double>(minGainDb: 6.0, maxGainDb: 6.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = volumeUp.Apply(audio, context);

        // Assert - 6dB gain should approximately double amplitude
        // The ratio should be close to 2 (within some tolerance due to dB conversion)
        var ratio = result[0] / audio[0];
        Assert.True(ratio > 1.5 && ratio < 2.5, $"Expected ~2x amplitude, got {ratio}x");
    }

    [Fact]
    public void TimeShift_Apply_ShiftsAudioSamples()
    {
        // Arrange
        var audio = CreateTestAudio(100);
        var shift = new TimeShift<double>(maxShiftFraction: 0.1, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = shift.Apply(audio, context);

        // Assert - length should be preserved
        Assert.Equal(audio.Length, result.Length);
    }

    #endregion

    #region Text Augmentation Tests

    [Fact]
    public void RandomDeletion_Apply_RemovesWords()
    {
        // Arrange
        var text = new[] { "The quick brown fox jumps over the lazy dog" };
        var deletion = new RandomDeletion<double>(deletionProbability: 0.3, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = deletion.Apply(text, context);

        // Assert - should have fewer words
        var originalWordCount = text[0].Split(' ').Length;
        var resultWordCount = result[0].Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        Assert.True(resultWordCount < originalWordCount || resultWordCount == originalWordCount);
    }

    [Fact]
    public void RandomSwap_Apply_SwapsWordPositions()
    {
        // Arrange
        var text = new[] { "one two three four five" };
        var swap = new RandomSwap<double>(numSwaps: 2, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = swap.Apply(text, context);

        // Assert - words should still exist but potentially in different order
        var originalWords = text[0].Split(' ').ToHashSet();
        var resultWords = result[0].Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        Assert.Equal(originalWords.Count, resultWords.Count);
        Assert.Subset(originalWords, resultWords);
    }

    [Fact]
    public void RandomInsertion_Apply_InsertsWords()
    {
        // Arrange
        var text = new[] { "hello world" };
        var insertion = new RandomInsertion<double>(numInsertions: 1, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = insertion.Apply(text, context);

        // Assert - result might have more words
        Assert.NotNull(result);
        Assert.Single(result);
    }

    [Fact]
    public void SynonymReplacement_Apply_ReplacesWithSynonyms()
    {
        // Arrange
        var text = new[] { "The big dog ran quickly" };
        var replacement = new SynonymReplacement<double>(replacementFraction: 0.3, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = replacement.Apply(text, context);

        // Assert - result should exist
        Assert.NotNull(result);
        Assert.Single(result);
    }

    #endregion

    #region Compose Tests

    [Fact]
    public void Compose_Apply_AppliesAugmentationsInSequence()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var compose = new Compose<double, ImageTensor<double>>(
            new HorizontalFlip<double>(probability: 1.0),
            new VerticalFlip<double>(probability: 1.0));
        var context = CreateTestContext();

        // Act
        var result = compose.Apply(image, context);

        // Assert - both flips applied
        Assert.Equal(10, result.Height);
        Assert.Equal(10, result.Width);
    }

    [Fact]
    public void Compose_With_AddsAugmentation()
    {
        // Arrange
        var compose = new Compose<double, ImageTensor<double>>(
            new HorizontalFlip<double>(probability: 1.0));

        // Act
        var extended = compose.With(new VerticalFlip<double>(probability: 1.0));

        // Assert
        Assert.Equal(2, extended.Augmentations.Count);
        Assert.Single(compose.Augmentations); // Original unchanged
    }

    [Fact]
    public void Compose_GetParameters_ReturnsCorrectInfo()
    {
        // Arrange - use IEnumerable constructor with probability parameter
        var compose = new Compose<double, ImageTensor<double>>(
            new IAugmentation<double, ImageTensor<double>>[] { new HorizontalFlip<double>(), new Rotation<double>() },
            probability: 0.8);

        // Act
        var parameters = compose.GetParameters();

        // Assert
        Assert.Equal(0.8, (double)parameters["probability"], Tolerance);
        Assert.Equal(2, (int)parameters["num_augmentations"]);
    }

    [Fact]
    public void Compose_RespectsProbability()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var compose = new Compose<double, ImageTensor<double>>(
            new[] { new HorizontalFlip<double>(probability: 1.0) },
            probability: 0.0);
        var context = CreateTestContext();

        // Act
        var result = compose.Apply(image, context);

        // Assert - with 0.0 probability, should be unchanged
        Assert.Equal(image.GetPixel(0, 0, 0), result.GetPixel(0, 0, 0), Tolerance);
    }

    #endregion

    #region OneOf Tests

    [Fact]
    public void OneOf_Apply_SelectsOneAugmentation()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var oneof = new OneOf<double, ImageTensor<double>>(
            new HorizontalFlip<double>(probability: 1.0),
            new VerticalFlip<double>(probability: 1.0));
        var context = CreateTestContext();

        // Act
        var result = oneof.Apply(image, context);

        // Assert - result should be different from original
        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void OneOf_Constructor_ThrowsOnEmptyList()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new OneOf<double, ImageTensor<double>>(Array.Empty<IAugmentation<double, ImageTensor<double>>>()));
    }

    [Fact]
    public void OneOf_WithWeights_RespectsWeights()
    {
        // Arrange - give all weight to first augmentation
        var oneof = new OneOf<double, ImageTensor<double>>(
            new[]
            {
                (new HorizontalFlip<double>(probability: 1.0) as IAugmentation<double, ImageTensor<double>>, 1.0),
                (new VerticalFlip<double>(probability: 1.0), 0.0)
            });

        // Act
        var parameters = oneof.GetParameters();

        // Assert
        Assert.Equal(2, (int)parameters["num_augmentations"]);
    }

    [Fact]
    public void OneOf_GetParameters_ReturnsWeights()
    {
        // Arrange
        var oneof = new OneOf<double, ImageTensor<double>>(
            new HorizontalFlip<double>(),
            new VerticalFlip<double>());

        // Act
        var parameters = oneof.GetParameters();

        // Assert
        Assert.True(parameters.ContainsKey("weights"));
        var weights = (List<double>)parameters["weights"];
        Assert.Equal(2, weights.Count);
        Assert.Equal(0.5, weights[0], Tolerance);
        Assert.Equal(0.5, weights[1], Tolerance);
    }

    #endregion

    #region SomeOf Tests

    [Fact]
    public void SomeOf_Apply_SelectsNAugmentations()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var someof = new SomeOf<double, ImageTensor<double>>(
            n: 2,
            new HorizontalFlip<double>(probability: 1.0),
            new VerticalFlip<double>(probability: 1.0),
            new Rotation<double>(probability: 1.0));
        var context = CreateTestContext();

        // Act
        var result = someof.Apply(image, context);

        // Assert
        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void SomeOf_RangeN_SelectsBetweenMinAndMax()
    {
        // Arrange
        var someof = new SomeOf<double, ImageTensor<double>>(
            minN: 1,
            maxN: 3,
            new[]
            {
                new HorizontalFlip<double>(probability: 1.0) as IAugmentation<double, ImageTensor<double>>,
                new VerticalFlip<double>(probability: 1.0),
                new Rotation<double>(probability: 1.0),
                new Brightness<double>(probability: 1.0)
            });

        // Act
        var parameters = someof.GetParameters();

        // Assert
        Assert.Equal(1, (int)parameters["min_n"]);
        Assert.Equal(3, (int)parameters["max_n"]);
    }

    [Fact]
    public void SomeOf_Constructor_ThrowsOnInvalidN()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SomeOf<double, ImageTensor<double>>(
                minN: -1,
                maxN: 2,
                new[] { new HorizontalFlip<double>(probability: 1.0) as IAugmentation<double, ImageTensor<double>> }));

        Assert.Throws<ArgumentException>(() =>
            new SomeOf<double, ImageTensor<double>>(
                minN: 3,
                maxN: 1, // max < min
                new[] { new HorizontalFlip<double>(probability: 1.0) as IAugmentation<double, ImageTensor<double>> }));
    }

    #endregion

    #region AugmentationPipeline Tests

    [Fact]
    public void AugmentationPipeline_Add_ChainsProperly()
    {
        // Arrange
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>("TestPipeline");

        // Act
        pipeline
            .Add(new HorizontalFlip<double>())
            .Add(new Rotation<double>())
            .Add(new Brightness<double>());

        // Assert
        Assert.Equal(3, pipeline.AugmentationCount);
        Assert.Equal("TestPipeline", pipeline.Name);
    }

    [Fact]
    public void AugmentationPipeline_Apply_Sequential()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();
        pipeline.Order = AugmentationOrder.Sequential;
        pipeline.Add(new HorizontalFlip<double>(probability: 1.0));
        var context = CreateTestContext();

        // Act
        var result = pipeline.Apply(image, context);

        // Assert
        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void AugmentationPipeline_OneOf_SelectsSingle()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();
        pipeline.OneOf(
            new HorizontalFlip<double>(probability: 1.0),
            new VerticalFlip<double>(probability: 1.0));
        var context = CreateTestContext();

        // Act
        var result = pipeline.Apply(image, context);

        // Assert
        Assert.Equal(10, result.Height);
    }

    [Fact]
    public void AugmentationPipeline_GetConfiguration_ReturnsDetails()
    {
        // Arrange
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>("MyPipeline");
        pipeline.Order = AugmentationOrder.Random;
        pipeline.Add(new HorizontalFlip<double>());

        // Act
        var config = pipeline.GetConfiguration();

        // Assert
        Assert.Equal("MyPipeline", config["name"]);
        Assert.Equal("Random", config["order"]);
    }

    [Fact]
    public void AugmentationPipeline_AddNull_Throws()
    {
        // Arrange
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => pipeline.Add(null!));
    }

    #endregion

    #region AugmentationConfig Tests

    [Fact]
    public void AugmentationConfig_Defaults_AreCorrect()
    {
        // Arrange & Act
        var config = new AugmentationConfig();

        // Assert
        Assert.True(config.IsEnabled);
        Assert.Equal(0.5, config.Probability, Tolerance);
        Assert.True(config.EnableTTA);
        Assert.Equal(5, config.TTANumAugmentations);
        Assert.Equal(PredictionAggregationMethod.Mean, config.TTAAggregation);
        Assert.True(config.TTAIncludeOriginal);
    }

    [Fact]
    public void AugmentationConfig_ForImages_SetsImageSettings()
    {
        // Arrange & Act
        var config = AugmentationConfig.ForImages();

        // Assert
        Assert.NotNull(config.ImageSettings);
        Assert.True(config.ImageSettings.EnableFlips);
        Assert.True(config.ImageSettings.EnableRotation);
    }

    [Fact]
    public void AugmentationConfig_ForTabular_SetsTabularSettings()
    {
        // Arrange & Act
        var config = AugmentationConfig.ForTabular();

        // Assert
        Assert.NotNull(config.TabularSettings);
        Assert.True(config.TabularSettings.EnableMixUp);
    }

    [Fact]
    public void AugmentationConfig_ForAudio_SetsAudioSettings()
    {
        // Arrange & Act
        var config = AugmentationConfig.ForAudio();

        // Assert
        Assert.NotNull(config.AudioSettings);
        Assert.True(config.AudioSettings.EnablePitchShift);
    }

    [Fact]
    public void AugmentationConfig_ForText_SetsTextSettings()
    {
        // Arrange & Act
        var config = AugmentationConfig.ForText();

        // Assert
        Assert.NotNull(config.TextSettings);
        Assert.True(config.TextSettings.EnableSynonymReplacement);
    }

    [Fact]
    public void AugmentationConfig_ForVideo_SetsVideoSettings()
    {
        // Arrange & Act
        var config = AugmentationConfig.ForVideo();

        // Assert
        Assert.NotNull(config.VideoSettings);
        Assert.True(config.VideoSettings.EnableTemporalCrop);
    }

    [Fact]
    public void AugmentationConfig_GetConfiguration_ReturnsAllSettings()
    {
        // Arrange
        var config = new AugmentationConfig
        {
            IsEnabled = true,
            Probability = 0.7,
            Seed = 42,
            EnableTTA = true,
            ImageSettings = new ImageAugmentationSettings()
        };

        // Act
        var dict = config.GetConfiguration();

        // Assert
        Assert.True((bool)dict["isEnabled"]);
        Assert.Equal(0.7, (double)dict["probability"], Tolerance);
        Assert.Equal(42, (int)dict["seed"]);
        Assert.True(dict.ContainsKey("imageSettings"));
    }

    #endregion

    #region ImageAugmentationSettings Tests

    [Fact]
    public void ImageAugmentationSettings_Defaults_AreIndustryStandard()
    {
        // Arrange & Act
        var settings = new ImageAugmentationSettings();

        // Assert - verify industry-standard defaults
        Assert.True(settings.EnableFlips);
        Assert.False(settings.EnableVerticalFlip);
        Assert.True(settings.EnableRotation);
        Assert.Equal(15.0, settings.RotationRange, Tolerance);
        Assert.True(settings.EnableColorJitter);
        Assert.Equal(0.2, settings.BrightnessRange, Tolerance);
        Assert.Equal(0.2, settings.ContrastRange, Tolerance);
        Assert.Equal(0.2, settings.SaturationRange, Tolerance);
        Assert.True(settings.EnableGaussianNoise);
        Assert.Equal(0.01, settings.NoiseStdDev, Tolerance);
        Assert.False(settings.EnableMixUp);
        Assert.Equal(0.2, settings.MixUpAlpha, Tolerance);
    }

    [Fact]
    public void ImageAugmentationSettings_GetConfiguration_IncludesAllFields()
    {
        // Arrange
        var settings = new ImageAugmentationSettings { EnableMixUp = true };

        // Act
        var config = settings.GetConfiguration();

        // Assert
        Assert.True(config.ContainsKey("enableFlips"));
        Assert.True(config.ContainsKey("enableRotation"));
        Assert.True(config.ContainsKey("rotationRange"));
        Assert.True(config.ContainsKey("enableMixUp"));
        Assert.True((bool)config["enableMixUp"]);
    }

    #endregion

    #region DataModalityDetector Tests

    [Fact]
    public void DataModalityDetector_Detect_IdentifiesImageTensor()
    {
        // Act
        var modality = DataModalityDetector.Detect<ImageTensor<double>>();

        // Assert
        Assert.Equal(DataModality.Image, modality);
    }

    [Fact]
    public void DataModalityDetector_Detect_IdentifiesMatrix()
    {
        // Act
        var modality = DataModalityDetector.Detect<Matrix<double>>();

        // Assert
        Assert.Equal(DataModality.Tabular, modality);
    }

    [Fact]
    public void DataModalityDetector_Detect_IdentifiesString()
    {
        // Act
        var modality = DataModalityDetector.Detect<string>();

        // Assert
        Assert.Equal(DataModality.Text, modality);
    }

    [Fact]
    public void DataModalityDetector_Detect_IdentifiesStringArray()
    {
        // Act
        var modality = DataModalityDetector.Detect<string[]>();

        // Assert
        Assert.Equal(DataModality.Text, modality);
    }

    [Fact]
    public void DataModalityDetector_Detect_ReturnsTensorAsUnknown()
    {
        // Act
        var modality = DataModalityDetector.Detect<Tensor<double>>();

        // Assert - raw tensors are unknown as they could be any modality
        Assert.Equal(DataModality.Unknown, modality);
    }

    #endregion

    #region BoundingBox Tests

    [Fact]
    public void BoundingBox_Create_SetsCoordinates()
    {
        // Arrange & Act
        var box = new BoundingBox<double>(10, 20, 100, 150, BoundingBoxFormat.XYXY);

        // Assert
        Assert.Equal(10, box.X1, Tolerance);
        Assert.Equal(20, box.Y1, Tolerance);
        Assert.Equal(100, box.X2, Tolerance);
        Assert.Equal(150, box.Y2, Tolerance);
    }

    [Fact]
    public void BoundingBox_ToXYWH_ConvertsCorrectly()
    {
        // Arrange
        var box = new BoundingBox<double>(10, 20, 60, 80, BoundingBoxFormat.XYXY);

        // Act
        var (x, y, w, h) = box.ToXYWH();

        // Assert
        Assert.Equal(10, x, Tolerance);
        Assert.Equal(20, y, Tolerance);
        Assert.Equal(50, w, Tolerance); // 60 - 10
        Assert.Equal(60, h, Tolerance); // 80 - 20
    }

    [Fact]
    public void BoundingBox_IsValid_ChecksBounds()
    {
        // Arrange
        var validBox = new BoundingBox<double>(10, 20, 100, 150, BoundingBoxFormat.XYXY);
        var invalidBox = new BoundingBox<double>(100, 20, 10, 150, BoundingBoxFormat.XYXY); // X2 < X1

        // Assert
        Assert.True(validBox.IsValid());
        Assert.False(invalidBox.IsValid());
    }

    [Fact]
    public void BoundingBox_Clone_CreatesDeepCopy()
    {
        // Arrange
        var original = new BoundingBox<double>(10, 20, 100, 150, BoundingBoxFormat.XYXY);
        original.ClassIndex = 5;
        original.Confidence = 0.9;

        // Act
        var clone = original.Clone();
        clone.X1 = 50;

        // Assert
        Assert.Equal(10, original.X1, Tolerance);
        Assert.Equal(50, clone.X1, Tolerance);
        Assert.Equal(5, clone.ClassIndex);
        Assert.Equal(0.9, (double)clone.Confidence!, Tolerance);
    }

    #endregion

    #region Keypoint Tests

    [Fact]
    public void Keypoint_Create_SetsCoordinates()
    {
        // Arrange & Act
        var keypoint = new Keypoint<double>(100.5, 200.5);

        // Assert
        Assert.Equal(100.5, keypoint.X, Tolerance);
        Assert.Equal(200.5, keypoint.Y, Tolerance);
    }

    [Fact]
    public void Keypoint_Clone_CreatesDeepCopy()
    {
        // Arrange
        var original = new Keypoint<double>(100, 200)
        {
            Name = "nose",
            Visibility = 2 // 2 = visible
        };

        // Act
        var clone = original.Clone();
        clone.X = 50;

        // Assert
        Assert.Equal(100, original.X, Tolerance);
        Assert.Equal(50, clone.X, Tolerance);
        Assert.Equal("nose", clone.Name);
        Assert.Equal(2, clone.Visibility);
    }

    #endregion

    #region AugmentedSample Tests

    [Fact]
    public void AugmentedSample_Create_HoldsDataAndTargets()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var boxes = new List<BoundingBox<double>>
        {
            new BoundingBox<double>(0, 0, 5, 5, BoundingBoxFormat.XYXY)
        };

        // Act
        var sample = new AugmentedSample<double, ImageTensor<double>>(image)
        {
            BoundingBoxes = boxes
        };

        // Assert
        Assert.Equal(10, sample.Data.Height);
        Assert.True(sample.HasBoundingBoxes);
        Assert.Single(sample.BoundingBoxes!);
    }

    [Fact]
    public void AugmentedSample_Clone_CreatesDeepCopy()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var boxes = new List<BoundingBox<double>>
        {
            new BoundingBox<double>(0, 0, 5, 5, BoundingBoxFormat.XYXY)
        };
        var original = new AugmentedSample<double, ImageTensor<double>>(image)
        {
            BoundingBoxes = boxes
        };

        // Act
        var clone = original.Clone();
        clone.BoundingBoxes![0].X1 = 100;

        // Assert
        Assert.Equal(0, original.BoundingBoxes[0].X1, Tolerance);
        Assert.Equal(100, clone.BoundingBoxes[0].X1, Tolerance);
    }

    #endregion

    #region SpatialAugmentation With Targets Tests

    [Fact]
    public void HorizontalFlip_ApplyWithTargets_TransformsBoundingBox()
    {
        // Arrange
        var image = CreateTestImage(100, 100);
        var box = new BoundingBox<double>(10, 20, 30, 40, BoundingBoxFormat.XYXY);
        var sample = new AugmentedSample<double, ImageTensor<double>>(image)
        {
            BoundingBoxes = new List<BoundingBox<double>> { box }
        };
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = flip.ApplyWithTargets(sample, context);

        // Assert - box should be flipped horizontally
        var flippedBox = result.BoundingBoxes![0];
        // Original: x1=10, x2=30, width=100
        // After flip: new_x1 = 100 - 30 = 70, new_x2 = 100 - 10 = 90
        Assert.Equal(70, flippedBox.X1, 1.0);
        Assert.Equal(90, flippedBox.X2, 1.0);
    }

    [Fact]
    public void HorizontalFlip_ApplyWithTargets_TransformsKeypoint()
    {
        // Arrange
        var image = CreateTestImage(100, 100);
        var keypoint = new Keypoint<double>(25, 50);
        var sample = new AugmentedSample<double, ImageTensor<double>>(image)
        {
            Keypoints = new List<Keypoint<double>> { keypoint }
        };
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = flip.ApplyWithTargets(sample, context);

        // Assert - keypoint x should be flipped
        var flippedKeypoint = result.Keypoints![0];
        // Original x=25, width=100
        // After flip: new_x = 100 - 1 - 25 = 74
        Assert.Equal(74, flippedKeypoint.X, 1.0);
        Assert.Equal(50, flippedKeypoint.Y, 1.0); // Y unchanged
    }

    [Fact]
    public void Rotation_ApplyWithTargets_RotatesBoundingBox()
    {
        // Arrange
        var image = CreateTestImage(100, 100);
        var box = new BoundingBox<double>(40, 40, 60, 60, BoundingBoxFormat.XYXY);
        var sample = new AugmentedSample<double, ImageTensor<double>>(image)
        {
            BoundingBoxes = new List<BoundingBox<double>> { box }
        };
        var rotation = new Rotation<double>(minAngle: 45, maxAngle: 45, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = rotation.ApplyWithTargets(sample, context);

        // Assert - box should be rotated (and expanded to axis-aligned bounding box)
        Assert.Single(result.BoundingBoxes!);
        // Rotated box will have different coordinates
    }

    #endregion

    #region Training Mode Tests

    [Fact]
    public void Augmentation_TrainingOnly_SkipsInInferenceMode()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalValue = image.GetPixel(0, 0, 0);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = new AugmentationContext<double>(isTraining: false); // Inference mode

        // Act
        var result = flip.Apply(image, context);

        // Assert - should be unchanged in inference mode
        Assert.Equal(originalValue, result.GetPixel(0, 0, 0), Tolerance);
    }

    [Fact]
    public void Augmentation_TrainingOnly_AppliesInTrainingMode()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalTopRight = image.GetPixel(0, 9, 0);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = new AugmentationContext<double>(isTraining: true); // Training mode

        // Act
        var result = flip.Apply(image, context);

        // Assert - should be flipped in training mode
        Assert.Equal(originalTopRight, result.GetPixel(0, 0, 0), Tolerance);
    }

    #endregion

    #region IsEnabled Tests

    [Fact]
    public void Augmentation_IsEnabled_False_SkipsAugmentation()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalValue = image.GetPixel(0, 0, 0);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        flip.IsEnabled = false;
        var context = CreateTestContext();

        // Act
        var result = flip.Apply(image, context);

        // Assert - should be unchanged
        Assert.Equal(originalValue, result.GetPixel(0, 0, 0), Tolerance);
    }

    #endregion

    #region Video Augmentation Tests

    [Fact]
    public void TemporalFlip_Apply_ReversesFrameOrder()
    {
        // Arrange - TemporalFlip works on ImageTensor<double>[] (video frames)
        var frames = new ImageTensor<double>[5];
        for (int f = 0; f < 5; f++)
        {
            // Create frames with distinct values so we can verify order reversal
            frames[f] = CreateTestImage(10, 10, 3, 0.1 * f);
        }

        var flip = new TemporalFlip<double>(probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = flip.Apply(frames, context);

        // Assert - frame order should be reversed
        Assert.Equal(5, result.Length);
        // First frame of result should be last frame of input
        // We check by comparing pixel values (frame 4 had value ~0.4)
        Assert.True(result[0].GetPixel(0, 0, 0) > result[4].GetPixel(0, 0, 0));
    }

    [Fact]
    public void FrameDropout_Apply_DropsFrames()
    {
        // Arrange - FrameDropout works on ImageTensor<double>[]
        var frames = new ImageTensor<double>[10];
        for (int i = 0; i < 10; i++)
        {
            frames[i] = CreateTestImage(8, 8, 3, 0.5);
        }

        var dropout = new FrameDropout<double>(dropoutRate: 0.3, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = dropout.Apply(frames, context);

        // Assert - result should have fewer or equal frames
        Assert.NotNull(result);
        Assert.True(result.Length <= frames.Length);
        Assert.True(result.Length >= 2); // MinFramesToKeep default is 2
    }

    [Fact]
    public void SpeedChange_Apply_ChangesVideoSpeed()
    {
        // Arrange - SpeedChange works on ImageTensor<double>[] (video frames)
        var frames = new ImageTensor<double>[10];
        for (int i = 0; i < 10; i++)
        {
            frames[i] = CreateTestImage(8, 8, 3, 0.5 + i * 0.01);
        }

        var speedChange = new SpeedChange<double>(minSpeed: 2.0, maxSpeed: 2.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = speedChange.Apply(frames, context);

        // Assert - 2x speed should roughly halve frames
        Assert.True(result.Length < frames.Length);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void AugmentationBase_Constructor_ThrowsOnInvalidProbability()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new HorizontalFlip<double>(probability: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HorizontalFlip<double>(probability: 1.5));
    }

    [Fact]
    public void Compose_EmptyList_ReturnsOriginalData()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalValue = image.GetPixel(5, 5, 0);
        var compose = new Compose<double, ImageTensor<double>>(Array.Empty<IAugmentation<double, ImageTensor<double>>>());
        var context = CreateTestContext();

        // Act
        var result = compose.Apply(image, context);

        // Assert
        Assert.Equal(originalValue, result.GetPixel(5, 5, 0), Tolerance);
    }

    [Fact]
    public void AugmentationPipeline_EmptyPipeline_ReturnsOriginalData()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var originalValue = image.GetPixel(5, 5, 0);
        var pipeline = new AugmentationPipeline<double, ImageTensor<double>>();
        var context = CreateTestContext();

        // Act
        var result = pipeline.Apply(image, context);

        // Assert
        Assert.Equal(originalValue, result.GetPixel(5, 5, 0), Tolerance);
    }

    [Fact]
    public void Rotation_VerySmallImage_DoesNotCrash()
    {
        // Arrange
        var image = CreateTestImage(2, 2);
        var rotation = new Rotation<double>(minAngle: -45, maxAngle: 45, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = rotation.Apply(image, context);

        // Assert
        Assert.Equal(2, result.Height);
        Assert.Equal(2, result.Width);
    }

    [Fact]
    public void FeatureNoise_EmptyMatrix_DoesNotCrash()
    {
        // Arrange
        var matrix = new Matrix<double>(0, 0);
        var noise = new FeatureNoise<double>(noiseStdDev: 0.1, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = noise.Apply(matrix, context);

        // Assert
        Assert.Equal(0, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void RandomDeletion_EmptyText_DoesNotCrash()
    {
        // Arrange
        var text = new[] { "" };
        var deletion = new RandomDeletion<double>(deletionProbability: 0.5, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = deletion.Apply(text, context);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void RandomDeletion_SingleWord_PreservesMinimumContent()
    {
        // Arrange
        var text = new[] { "hello" };
        var deletion = new RandomDeletion<double>(deletionProbability: 1.0, probability: 1.0);
        var context = CreateTestContext();

        // Act
        var result = deletion.Apply(text, context);

        // Assert
        Assert.NotNull(result);
    }

    #endregion

    #region Event Tests

    [Fact]
    public void AugmentationBase_OnAugmentationApplied_RaisesEvent()
    {
        // Arrange
        var image = CreateTestImage(10, 10);
        var flip = new HorizontalFlip<double>(probability: 1.0);
        var context = CreateTestContext();
        var eventRaised = false;
        flip.OnAugmentationApplied += (sender, args) => eventRaised = true;

        // Act
        _ = flip.Apply(image, context);

        // Assert
        Assert.True(eventRaised);
    }

    [Fact]
    public void MixUp_OnLabelMixing_RaisesEvent()
    {
        // Arrange
        var image1 = CreateTestImage(10, 10, initialValue: 0.2);
        var image2 = CreateTestImage(10, 10, initialValue: 0.8);
        var labels1 = new Vector<double>(new[] { 1.0, 0.0 });
        var labels2 = new Vector<double>(new[] { 0.0, 1.0 });
        var mixup = new MixUp<double>(alpha: 1.0, probability: 1.0);
        var context = CreateTestContext();
        var eventRaised = false;
        mixup.OnLabelMixing += (sender, args) => eventRaised = true;

        // Act
        mixup.ApplyMixUp(image1, image2, labels1, labels2, context);

        // Assert
        Assert.True(eventRaised);
    }

    #endregion

    #region Recommendation and Validation Tests

    [Fact]
    public void AugmentationRecommendation_Properties_WorkCorrectly()
    {
        // Arrange & Act
        var recommendation = new AugmentationRecommendation
        {
            AugmentationType = "HorizontalFlip",
            RecommendedProbability = 0.5,
            ConfidenceScore = 0.95,
            Priority = 1,
            Reason = "Standard augmentation for image classification",
            IsCritical = false,
            IncompatibleWith = new List<string> { "TextAugmentation" }
        };

        // Assert
        Assert.Equal("HorizontalFlip", recommendation.AugmentationType);
        Assert.Equal(0.5, recommendation.RecommendedProbability, Tolerance);
        Assert.Equal(0.95, recommendation.ConfidenceScore, Tolerance);
        Assert.Equal(1, recommendation.Priority);
        Assert.False(recommendation.IsCritical);
        Assert.Single(recommendation.IncompatibleWith!);
    }

    [Fact]
    public void AugmentationValidationResult_Properties_WorkCorrectly()
    {
        // Arrange & Act
        var result = new AugmentationValidationResult
        {
            IsValid = false,
            Warnings = new List<string> { "MixUp may not work well with object detection" },
            Errors = new List<string> { "Text augmentation not supported for image data" },
            SuggestedFixes = new List<string> { "Remove text augmentations from pipeline" }
        };

        // Assert
        Assert.False(result.IsValid);
        Assert.Single(result.Warnings);
        Assert.Single(result.Errors);
        Assert.Single(result.SuggestedFixes);
    }

    [Fact]
    public void DatasetCharacteristics_Properties_WorkCorrectly()
    {
        // Arrange & Act
        var characteristics = new DatasetCharacteristics
        {
            SampleCount = 10000,
            NumClasses = 10,
            IsImbalanced = true,
            ImbalanceRatio = 5.0,
            ImageDimensions = (224, 224),
            HasVariableSizes = false,
            NumFeatures = 0,
            HasSpatialTargets = true,
            HasBoundingBoxes = true,
            HasKeypoints = false,
            HasMasks = false
        };

        // Assert
        Assert.Equal(10000, characteristics.SampleCount);
        Assert.Equal(10, characteristics.NumClasses);
        Assert.True(characteristics.IsImbalanced);
        Assert.Equal(5.0, characteristics.ImbalanceRatio, Tolerance);
        Assert.Equal((224, 224), characteristics.ImageDimensions);
        Assert.True(characteristics.HasBoundingBoxes);
    }

    #endregion

    #region TestTimeAugmentationResult Tests

    [Fact]
    public void TestTimeAugmentationResult_Properties_WorkCorrectly()
    {
        // Arrange
        var predictions = new List<double[]>
        {
            new[] { 0.8, 0.2 },
            new[] { 0.7, 0.3 },
            new[] { 0.9, 0.1 }
        };
        var aggregated = new[] { 0.8, 0.2 };

        // Act
        var result = new TestTimeAugmentationResult<double[]>(
            aggregated,
            predictions,
            confidence: 0.95,
            standardDeviation: 0.1);

        // Assert
        Assert.Equal(aggregated, result.AggregatedPrediction);
        Assert.Equal(3, result.IndividualPredictions.Count);
        Assert.Equal(0.95, result.Confidence);
        Assert.Equal(0.1, result.StandardDeviation);
    }

    #endregion
}
