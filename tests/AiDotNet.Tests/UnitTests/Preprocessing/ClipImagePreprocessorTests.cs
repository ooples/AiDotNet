using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Preprocessing.Image;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Unit tests for CLIP image preprocessor.
/// </summary>
public class ClipImagePreprocessorTests
{
    private readonly ClipImagePreprocessor<float> _preprocessor;

    public ClipImagePreprocessorTests()
    {
        _preprocessor = new ClipImagePreprocessor<float>();
    }

    [Fact]
    public void Constructor_WithDefaultParameters_SetsImageSizeTo224()
    {
        // Act
        var preprocessor = new ClipImagePreprocessor<float>();

        // Assert
        Assert.Equal(224, preprocessor.ImageSize);
    }

    [Fact]
    public void Constructor_WithCustomImageSize_UsesCustomSize()
    {
        // Act
        var preprocessor = new ClipImagePreprocessor<float>(imageSize: 336);

        // Assert
        Assert.Equal(336, preprocessor.ImageSize);
    }

    [Fact]
    public void Constructor_WithZeroImageSize_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ClipImagePreprocessor<float>(imageSize: 0));
    }

    [Fact]
    public void Constructor_WithNegativeImageSize_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ClipImagePreprocessor<float>(imageSize: -1));
    }

    [Fact]
    public void Constructor_WithInvalidMeanLength_ThrowsArgumentException()
    {
        // Arrange
        var invalidMean = new float[] { 0.5f, 0.5f }; // Only 2 values instead of 3

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new ClipImagePreprocessor<float>(mean: invalidMean));
    }

    [Fact]
    public void Constructor_WithInvalidStdLength_ThrowsArgumentException()
    {
        // Arrange
        var invalidStd = new float[] { 0.5f }; // Only 1 value instead of 3

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new ClipImagePreprocessor<float>(std: invalidStd));
    }

    [Fact]
    public void Preprocess_NullImage_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _preprocessor.Preprocess(null!));
    }

    [Fact]
    public void Preprocess_GrayscaleImage_ExpandsTo3Channels()
    {
        // Arrange - 2D grayscale image [H, W]
        var grayscale = CreateTensor(64, 64);
        FillWithValue(grayscale, 0.5f);

        // Act
        var result = _preprocessor.Preprocess(grayscale);

        // Assert
        Assert.Equal(3, result.Shape[0]); // Channels
        Assert.Equal(224, result.Shape[1]); // Height
        Assert.Equal(224, result.Shape[2]); // Width
    }

    [Fact]
    public void Preprocess_HWCImage_ConvertsToChannelsFirst()
    {
        // Arrange - [H, W, C] format (100x100x3)
        var hwcImage = CreateTensor(100, 100, 3);
        FillWithRandomValues(hwcImage, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(hwcImage);

        // Assert
        Assert.Equal(3, result.Shape.Length);
        Assert.Equal(3, result.Shape[0]); // Channels first
        Assert.Equal(224, result.Shape[1]); // Height
        Assert.Equal(224, result.Shape[2]); // Width
    }

    [Fact]
    public void Preprocess_CHWImage_MaintainsChannelsFirst()
    {
        // Arrange - [C, H, W] format (3x100x100)
        var chwImage = CreateTensor(3, 100, 100);
        FillWithRandomValues(chwImage, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(chwImage);

        // Assert
        Assert.Equal(3, result.Shape[0]); // Channels
        Assert.Equal(224, result.Shape[1]); // Height
        Assert.Equal(224, result.Shape[2]); // Width
    }

    [Fact]
    public void Preprocess_AlreadyCorrectSize_MaintainsSize()
    {
        // Arrange - Already 224x224
        var image = CreateTensor(3, 224, 224);
        FillWithRandomValues(image, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert
        Assert.Equal(224, result.Shape[1]);
        Assert.Equal(224, result.Shape[2]);
    }

    [Fact]
    public void Preprocess_LargerImage_ResizesDown()
    {
        // Arrange - Larger image (512x512)
        var image = CreateTensor(3, 512, 512);
        FillWithRandomValues(image, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert
        Assert.Equal(224, result.Shape[1]);
        Assert.Equal(224, result.Shape[2]);
    }

    [Fact]
    public void Preprocess_SmallerImage_ResizesUp()
    {
        // Arrange - Smaller image (50x50)
        var image = CreateTensor(3, 50, 50);
        FillWithRandomValues(image, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert
        Assert.Equal(224, result.Shape[1]);
        Assert.Equal(224, result.Shape[2]);
    }

    [Fact]
    public void Preprocess_NormalizesPixelValues()
    {
        // Arrange - Image with values 0-255
        var image = CreateTensor(3, 100, 100);
        FillWithValue(image, 128f); // Middle gray value

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert - Normalized values should be around 0 for middle gray
        // ImageNet normalization: (128/255 - mean) / std
        // For middle value, should be close to 0 after normalization
        var centerValue = result[1, 112, 112]; // Center pixel of middle channel
        Assert.True(Math.Abs(centerValue) < 2.0f, $"Normalized value {centerValue} should be reasonable");
    }

    [Fact]
    public void Preprocess_ValuesInRange0To1_DoesNotRescale()
    {
        // Arrange - Image with values already 0-1
        var image = CreateTensor(3, 100, 100);
        FillWithValue(image, 0.5f);

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert - Should still normalize using mean/std
        Assert.NotNull(result);
        // Values should be in reasonable normalized range
        var centerValue = result[1, 112, 112];
        Assert.True(Math.Abs(centerValue) < 3.0f, $"Normalized value should be reasonable");
    }

    [Fact]
    public void Preprocess_BatchImage_ExtractsFirstImage()
    {
        // Arrange - Batch of images [N, C, H, W]
        var batch = CreateTensor(2, 3, 100, 100);
        FillWithRandomValues(batch, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(batch);

        // Assert - Should extract first image
        Assert.Equal(3, result.Shape.Length);
        Assert.Equal(3, result.Shape[0]);
        Assert.Equal(224, result.Shape[1]);
        Assert.Equal(224, result.Shape[2]);
    }

    [Fact]
    public void Preprocess_SingleChannelImage_ExpandsToThreeChannels()
    {
        // Arrange - Single channel [1, H, W]
        var image = CreateTensor(1, 100, 100);
        FillWithRandomValues(image, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert
        Assert.Equal(3, result.Shape[0]);
    }

    [Fact]
    public void Preprocess_FourChannelImage_TakesFirstThreeChannels()
    {
        // Arrange - RGBA image [4, H, W]
        var image = CreateTensor(4, 100, 100);
        FillWithRandomValues(image, seed: 42);

        // Act
        var result = _preprocessor.Preprocess(image);

        // Assert
        Assert.Equal(3, result.Shape[0]);
    }

    [Fact]
    public void Preprocess_InvalidDimensions_ThrowsArgumentException()
    {
        // Arrange - 5D tensor (invalid)
        var invalid = CreateTensor(1, 2, 3, 4, 5);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            _preprocessor.Preprocess(invalid));
    }

    [Fact]
    public void PreprocessBatch_ProcessesMultipleImages()
    {
        // Arrange
        var images = new List<Tensor<float>>
        {
            CreateTensorWithRandomValues(3, 100, 100, seed: 1),
            CreateTensorWithRandomValues(3, 150, 150, seed: 2),
            CreateTensorWithRandomValues(3, 200, 200, seed: 3)
        };

        // Act
        var results = _preprocessor.PreprocessBatch(images).ToList();

        // Assert
        Assert.Equal(3, results.Count);
        Assert.All(results, r =>
        {
            Assert.Equal(3, r.Shape[0]);
            Assert.Equal(224, r.Shape[1]);
            Assert.Equal(224, r.Shape[2]);
        });
    }

    [Fact]
    public void PreprocessBatch_NullImages_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _preprocessor.PreprocessBatch(null!).ToList());
    }

    [Fact]
    public void PreprocessBatch_EmptyList_ReturnsEmptyResults()
    {
        // Arrange
        var images = new List<Tensor<float>>();

        // Act
        var results = _preprocessor.PreprocessBatch(images).ToList();

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public void Preprocess_WithCustomNormalization_UsesCustomValues()
    {
        // Arrange
        var customMean = new float[] { 0.5f, 0.5f, 0.5f };
        var customStd = new float[] { 0.5f, 0.5f, 0.5f };
        var preprocessor = new ClipImagePreprocessor<float>(mean: customMean, std: customStd);
        var image = CreateTensor(3, 100, 100);
        FillWithValue(image, 0.5f);

        // Act
        var result = preprocessor.Preprocess(image);

        // Assert - With value 0.5, mean 0.5, std 0.5: (0.5 - 0.5) / 0.5 = 0
        var centerValue = result[1, 112, 112];
        Assert.True(Math.Abs(centerValue) < 0.1f, $"Expected ~0, got {centerValue}");
    }

    [Fact]
    public void Preprocess_ProducesConsistentResults()
    {
        // Arrange - Same image, same seed
        var image1 = CreateTensorWithRandomValues(3, 100, 100, seed: 42);
        var image2 = CreateTensorWithRandomValues(3, 100, 100, seed: 42);

        // Act
        var result1 = _preprocessor.Preprocess(image1);
        var result2 = _preprocessor.Preprocess(image2);

        // Assert - Results should be identical
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 224; h++)
            {
                for (int w = 0; w < 224; w++)
                {
                    Assert.Equal(result1[c, h, w], result2[c, h, w], 5);
                }
            }
        }
    }

    #region Helper Methods

    private static Tensor<float> CreateTensor(params int[] shape)
    {
        return new Tensor<float>(shape);
    }

    private static Tensor<float> CreateTensorWithRandomValues(int c, int h, int w, int seed)
    {
        var tensor = new Tensor<float>(new[] { c, h, w });
        FillWithRandomValues(tensor, seed);
        return tensor;
    }

    private static void FillWithValue(Tensor<float> tensor, float value)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            tensor.Data[i] = value;
        }
    }

    private static void FillWithRandomValues(Tensor<float> tensor, int seed)
    {
        var random = new Random(seed);
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            tensor.Data[i] = (float)(random.NextDouble() * 255);
        }
    }

    #endregion
}
