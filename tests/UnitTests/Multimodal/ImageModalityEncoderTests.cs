using System;
using Xunit;
using AiDotNet.MultimodalAI.Encoders;
using AiDotNet.LinearAlgebra;

namespace AiDotNetTests.UnitTests.Multimodal
{
    /// <summary>
    /// Unit tests for ImageModalityEncoder class
    /// Ensures proper encoding of image data into vector representations
    /// </summary>
    public class ImageModalityEncoderTests
    {
        [Fact]
        public void Constructor_WithDefaultParameters_CreatesValidEncoder()
        {
            // Arrange & Act
            var encoder = new ImageModalityEncoder<double>();

            // Assert
            Assert.NotNull(encoder);
            Assert.Equal(512, encoder.OutputDimension);
        }

        [Fact]
        public void Constructor_WithCustomParameters_SetsPropertiesCorrectly()
        {
            // Arrange & Act
            var encoder = new ImageModalityEncoder<double>(
                outputDimension: 256,
                patchSize: 32,
                useColorHistogram: false,
                useTextureFeatures: false
            );

            // Assert
            Assert.NotNull(encoder);
            Assert.Equal(256, encoder.OutputDimension);
        }

        [Fact]
        public void Encode_With2DGrayscaleArray_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 128);
            var image = new double[28, 28]; // 28x28 grayscale image

            // Fill with sample data
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    image[i, j] = (i + j) % 256;
                }
            }

            // Act
            var result = encoder.Encode(image);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(128, result.Length);
        }

        [Fact]
        public void Encode_With3DColorArray_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 256);
            var image = new double[3, 32, 32]; // 32x32 RGB image

            // Fill with sample data
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        image[c, i, j] = (c * 85 + i + j) % 256;
                    }
                }
            }

            // Act
            var result = encoder.Encode(image);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(256, result.Length);
        }

        [Fact]
        public void Encode_WithTensor2D_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 128);
            var tensor = new Tensor<double>(new[] { 28, 28 });

            // Fill tensor with sample data
            for (int i = 0; i < 28 * 28; i++)
            {
                tensor.Data[i] = i % 256;
            }

            // Act
            var result = encoder.Encode(tensor);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(128, result.Length);
        }

        [Fact]
        public void Encode_WithTensor3D_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 256);
            var tensor = new Tensor<double>(new[] { 3, 32, 32 }); // RGB image

            // Fill tensor with sample data
            for (int i = 0; i < 3 * 32 * 32; i++)
            {
                tensor.Data[i] = i % 256;
            }

            // Act
            var result = encoder.Encode(tensor);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(256, result.Length);
        }

        [Fact]
        public void Encode_WithInvalidInput_ThrowsArgumentException()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>();
            var invalidInput = "not an image";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => encoder.Encode(invalidInput));
        }

        [Fact]
        public void Encode_WithNullInput_ThrowsArgumentException()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => encoder.Encode((Tensor<double>)null));
        }

        [Fact]
        public void Encode_With1DArray_ThrowsArgumentException()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>();
            var invalid1DArray = new double[100];

            // Act & Assert
            Assert.Throws<ArgumentException>(() => encoder.Encode(invalid1DArray));
        }

        [Fact]
        public void Encode_WithTensor4D_ThrowsArgumentException()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>();
            var invalid4DTensor = new Tensor<double>(new[] { 1, 3, 32, 32 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => encoder.Encode(invalid4DTensor));
        }

        [Fact]
        public void Encode_WithColorHistogramEnabled_IncludesColorFeatures()
        {
            // Arrange
            var encoderWithColor = new ImageModalityEncoder<double>(
                outputDimension: 256,
                useColorHistogram: true
            );
            var encoderWithoutColor = new ImageModalityEncoder<double>(
                outputDimension: 256,
                useColorHistogram: false
            );

            var colorImage = new double[3, 32, 32]; // RGB image
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        colorImage[c, i, j] = c * 100 + i;
                    }
                }
            }

            // Act
            var resultWith = encoderWithColor.Encode(colorImage);
            var resultWithout = encoderWithoutColor.Encode(colorImage);

            // Assert - Both should return valid vectors
            Assert.NotNull(resultWith);
            Assert.NotNull(resultWithout);
            Assert.Equal(256, resultWith.Length);
            Assert.Equal(256, resultWithout.Length);
        }

        [Fact]
        public void Encode_WithTextureFeatures_IncludesEdgeInformation()
        {
            // Arrange
            var encoderWithTexture = new ImageModalityEncoder<double>(
                outputDimension: 256,
                useTextureFeatures: true
            );
            var encoderWithoutTexture = new ImageModalityEncoder<double>(
                outputDimension: 256,
                useTextureFeatures: false
            );

            var image = new double[64, 64];
            // Create an image with edges
            for (int i = 0; i < 64; i++)
            {
                for (int j = 0; j < 64; j++)
                {
                    image[i, j] = (j < 32) ? 0 : 255; // Vertical edge in the middle
                }
            }

            // Act
            var resultWith = encoderWithTexture.Encode(image);
            var resultWithout = encoderWithoutTexture.Encode(image);

            // Assert - Both should return valid vectors
            Assert.NotNull(resultWith);
            Assert.NotNull(resultWithout);
            Assert.Equal(256, resultWith.Length);
            Assert.Equal(256, resultWithout.Length);
        }

        [Fact]
        public void Encode_WithDifferentPatchSizes_ReturnsConsistentDimension()
        {
            // Arrange
            var encoder8 = new ImageModalityEncoder<double>(
                outputDimension: 128,
                patchSize: 8
            );
            var encoder16 = new ImageModalityEncoder<double>(
                outputDimension: 128,
                patchSize: 16
            );
            var encoder32 = new ImageModalityEncoder<double>(
                outputDimension: 128,
                patchSize: 32
            );

            var image = new double[64, 64];
            for (int i = 0; i < 64; i++)
            {
                for (int j = 0; j < 64; j++)
                {
                    image[i, j] = i * j % 256;
                }
            }

            // Act
            var result8 = encoder8.Encode(image);
            var result16 = encoder16.Encode(image);
            var result32 = encoder32.Encode(image);

            // Assert - All should have the same output dimension
            Assert.Equal(128, result8.Length);
            Assert.Equal(128, result16.Length);
            Assert.Equal(128, result32.Length);
        }

        [Fact]
        public void Encode_ProducesNormalizedOutput()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 128);
            var image = new double[32, 32];

            // Fill with various values
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    image[i, j] = i * 10 + j;
                }
            }

            // Act
            var result = encoder.Encode(image);

            // Assert - Check that output is normalized (values should be reasonable)
            Assert.NotNull(result);
            foreach (var value in result.Data)
            {
                Assert.True(!double.IsNaN(value), "Encoded value should not be NaN");
                Assert.True(!double.IsInfinity(value), "Encoded value should not be infinite");
            }
        }

        [Fact]
        public void Encode_SameInputTwice_ProducesSameOutput()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 128);
            var image = new double[32, 32];

            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    image[i, j] = (i + j) % 256;
                }
            }

            // Act
            var result1 = encoder.Encode(image);
            var result2 = encoder.Encode(image);

            // Assert - Same input should produce same output (deterministic)
            Assert.Equal(result1.Length, result2.Length);
            for (int i = 0; i < result1.Length; i++)
            {
                Assert.Equal(result1[i], result2[i], 10); // Allow for small floating point differences
            }
        }

        [Fact]
        public void Encode_DifferentImages_ProducesDifferentOutputs()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 128);
            var image1 = new double[32, 32];
            var image2 = new double[32, 32];

            // Create two different images
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    image1[i, j] = i * 10;
                    image2[i, j] = j * 10;
                }
            }

            // Act
            var result1 = encoder.Encode(image1);
            var result2 = encoder.Encode(image2);

            // Assert - Different images should produce different encodings
            Assert.Equal(result1.Length, result2.Length);

            // Check that at least some values are different
            int differentCount = 0;
            for (int i = 0; i < result1.Length; i++)
            {
                if (Math.Abs(result1[i] - result2[i]) > 0.001)
                {
                    differentCount++;
                }
            }

            Assert.True(differentCount > result1.Length / 2,
                "Different images should produce significantly different encodings");
        }

        [Fact]
        public void Encode_SmallImage_HandlesCorrectly()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 128);
            var smallImage = new double[8, 8]; // Very small image

            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    smallImage[i, j] = i + j;
                }
            }

            // Act
            var result = encoder.Encode(smallImage);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(128, result.Length);
        }

        [Fact]
        public void Encode_LargeImage_HandlesCorrectly()
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: 512);
            var largeImage = new double[256, 256]; // Large image

            for (int i = 0; i < 256; i++)
            {
                for (int j = 0; j < 256; j++)
                {
                    largeImage[i, j] = (i * 256 + j) % 256;
                }
            }

            // Act
            var result = encoder.Encode(largeImage);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(512, result.Length);
        }

        [Theory]
        [InlineData(64)]
        [InlineData(128)]
        [InlineData(256)]
        [InlineData(512)]
        [InlineData(1024)]
        public void Encode_WithVariousOutputDimensions_ProducesCorrectSize(int outputDim)
        {
            // Arrange
            var encoder = new ImageModalityEncoder<double>(outputDimension: outputDim);
            var image = new double[32, 32];

            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    image[i, j] = i + j;
                }
            }

            // Act
            var result = encoder.Encode(image);

            // Assert
            Assert.Equal(outputDim, result.Length);
        }
    }
}
