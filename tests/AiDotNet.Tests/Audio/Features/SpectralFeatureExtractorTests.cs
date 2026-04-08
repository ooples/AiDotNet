using System;
using AiDotNet.Audio.Features;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Features
{
    public class SpectralFeatureExtractorTests
    {
        private static Tensor<float> CreateTestAudio(int sampleRate = 22050, double durationSeconds = 1.0)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);

            // Create a simple sine wave at 440 Hz (A4)
            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;
                audio[i] = (float)Math.Sin(2 * Math.PI * 440 * t);
            }

            return audio;
        }

        [Fact]
        public void SpectralFeatureExtractor_DefaultOptions_ExtractsBasicFeatures()
        {
            // Arrange
            var extractor = new SpectralFeatureExtractor<float>();
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert - Basic features = 5 (centroid, bandwidth, rolloff, flux, flatness)
            Assert.Equal(5, features.Shape[1]);
            Assert.True(features.Shape[0] > 0, "Should have at least one frame");
        }

        [Fact]
        public void SpectralFeatureExtractor_Name_IsSpectralFeatures()
        {
            // Arrange
            var extractor = new SpectralFeatureExtractor<float>();

            // Assert
            Assert.Equal("SpectralFeatures", extractor.Name);
        }

        [Fact]
        public void SpectralFeatureExtractor_CentroidOnly_ExtractsSingleFeature()
        {
            // Arrange
            var options = new SpectralFeatureOptions
            {
                FeatureTypes = SpectralFeatureType.Centroid
            };
            var extractor = new SpectralFeatureExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert
            Assert.Equal(1, features.Shape[1]);
        }

        [Fact]
        public void SpectralFeatureExtractor_AllFeatures_ExtractsAllFeatures()
        {
            // Arrange
            var options = new SpectralFeatureOptions
            {
                FeatureTypes = SpectralFeatureType.All
            };
            var extractor = new SpectralFeatureExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert - All = centroid + bandwidth + rolloff + flux + flatness + 6 contrast bands + ZCR = 12
            Assert.Equal(12, features.Shape[1]);
        }

        [Fact]
        public void SpectralFeatureExtractor_OutputValues_AreFinite()
        {
            // Arrange
            var extractor = new SpectralFeatureExtractor<float>();
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert
            for (int frame = 0; frame < features.Shape[0]; frame++)
            {
                for (int feat = 0; feat < features.Shape[1]; feat++)
                {
                    float value = features[frame, feat];
                    Assert.False(float.IsNaN(value), $"Feature[{frame},{feat}] is NaN");
                    Assert.False(float.IsInfinity(value), $"Feature[{frame},{feat}] is Infinity");
                }
            }
        }

        [Fact]
        public void SpectralFeatureExtractor_SpectralCentroid_IsPositive()
        {
            // Arrange
            var options = new SpectralFeatureOptions
            {
                FeatureTypes = SpectralFeatureType.Centroid
            };
            var extractor = new SpectralFeatureExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert - Spectral centroid should be positive for non-silent audio
            for (int frame = 0; frame < features.Shape[0]; frame++)
            {
                Assert.True(features[frame, 0] >= 0, $"Centroid[{frame}] should be >= 0");
            }
        }

        [Fact]
        public void SpectralFeatureExtractor_SpectralFlatness_IsBetweenZeroAndOne()
        {
            // Arrange
            var options = new SpectralFeatureOptions
            {
                FeatureTypes = SpectralFeatureType.Flatness
            };
            var extractor = new SpectralFeatureExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert - Flatness should be between 0 and 1
            for (int frame = 0; frame < features.Shape[0]; frame++)
            {
                float value = features[frame, 0];
                Assert.True(value >= 0 && value <= 1.0001f,
                    $"Flatness[{frame}] = {value} should be in [0, 1]");
            }
        }

        [Fact]
        public void SpectralFeatureExtractor_Contrast_ExtractsSixBands()
        {
            // Arrange
            var options = new SpectralFeatureOptions
            {
                FeatureTypes = SpectralFeatureType.Contrast
            };
            var extractor = new SpectralFeatureExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var features = extractor.Extract(audio);

            // Assert - Contrast has 6 frequency bands
            Assert.Equal(6, features.Shape[1]);
        }

        [Fact]
        public void SpectralFeatureType_BasicFlag_CombinesCorrectFeatures()
        {
            // Assert
            Assert.True(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.Centroid));
            Assert.True(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.Bandwidth));
            Assert.True(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.Rolloff));
            Assert.True(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.Flux));
            Assert.True(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.Flatness));
            Assert.False(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.Contrast));
            Assert.False(SpectralFeatureType.Basic.HasFlag(SpectralFeatureType.ZeroCrossingRate));
        }
    }
}
