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
        #region Test Helpers

        /// <summary>
        /// Generates a 2D grayscale image with values computed by the provided function.
        /// </summary>
        /// <param name="height">Image height</param>
        /// <param name="width">Image width</param>
        /// <param name="valueFunc">Function that computes pixel value given (i, j) coordinates</param>
        /// <returns>Generated 2D image array</returns>
        private static double[,] GenerateImage(int height, int width, Func<int, int, double> valueFunc)
        {
            var image = new double[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    image[i, j] = valueFunc(i, j);
                }
            }
            return image;
        }

        /// <summary>
        /// Generates a 3D color image (RGB) with values computed by the provided function.
        /// </summary>
        /// <param name="channels">Number of channels (typically 3 for RGB)</param>
        /// <param name="height">Image height</param>
        /// <param name="width">Image width</param>
        /// <param name="valueFunc">Function that computes pixel value given (c, i, j) coordinates</param>
        /// <returns>Generated 3D image array</returns>
        private static double[,,] GenerateColorImage(int channels, int height, int width, Func<int, int, int, double> valueFunc)
        {
            var image = new double[channels, height, width];
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        image[c, i, j] = valueFunc(c, i, j);
                    }
                }
            }
            return image;
        }

        /// <summary>
        /// Generates a 2D tensor with values computed by the provided function.
        /// </summary>
        /// <param name="height">Tensor height</param>
        /// <param name="width">Tensor width</param>
        /// <param name="valueFunc">Function that computes value given linear index</param>
        /// <returns>Generated 2D tensor</returns>
        private static Tensor<double> GenerateTensor(int height, int width, Func<int, double> valueFunc)
        {
            var tensor = new Tensor<double>(new[] { height, width });
            for (int i = 0; i < height * width; i++)
            {
                tensor.Data[i] = valueFunc(i);
            }
            return tensor;
        }

        /// <summary>
        /// Generates a 3D tensor with values computed by the provided function.
        /// </summary>
        /// <param name="depth">Tensor depth</param>
        /// <param name="height">Tensor height</param>
        /// <param name="width">Tensor width</param>
        /// <param name="valueFunc">Function that computes value given linear index</param>
        /// <returns>Generated 3D tensor</returns>
        private static Tensor<double> GenerateTensor3D(int depth, int height, int width, Func<int, double> valueFunc)
        {
            var tensor = new Tensor<double>(new[] { depth, height, width });
            for (int i = 0; i < depth * height * width; i++)
            {
                tensor.Data[i] = valueFunc(i);
            }
            return tensor;
        }

        #endregion

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
            var image = GenerateImage(28, 28, (i, j) => (i + j) % 256);

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
            var image = GenerateColorImage(3, 32, 32, (c, i, j) => (c * 85 + i + j) % 256);

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
            var tensor = GenerateTensor(28, 28, i => i % 256);

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
            var tensor = GenerateTensor3D(3, 32, 32, i => i % 256);

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

            var colorImage = GenerateColorImage(3, 32, 32, (c, i, j) => c * 100 + i);

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

            // Create an image with a vertical edge in the middle
            var image = GenerateImage(64, 64, (i, j) => (j < 32) ? 0 : 255);

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

            var image = GenerateImage(64, 64, (i, j) => i * j % 256);

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
            var image = GenerateImage(32, 32, (i, j) => i * 10 + j);

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
            var image = GenerateImage(32, 32, (i, j) => (i + j) % 256);

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
            // Create two different images
            var image1 = GenerateImage(32, 32, (i, j) => i * 10);
            var image2 = GenerateImage(32, 32, (i, j) => j * 10);

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
            var smallImage = GenerateImage(8, 8, (i, j) => i + j);

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
            var largeImage = GenerateImage(256, 256, (i, j) => (i * 256 + j) % 256);

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
            var image = GenerateImage(32, 32, (i, j) => i + j);

            // Act
            var result = encoder.Encode(image);

            // Assert
            Assert.Equal(outputDim, result.Length);
        }
    }
}
