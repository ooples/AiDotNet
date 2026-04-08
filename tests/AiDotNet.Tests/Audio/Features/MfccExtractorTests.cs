using System;
using AiDotNet.Audio.Features;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Features
{
    public class MfccExtractorTests
    {
        private static Tensor<float> CreateTestAudio(int sampleRate = 16000, double durationSeconds = 1.0)
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
        public void MfccExtractor_DefaultOptions_ExtractsCorrectDimension()
        {
            // Arrange
            var options = new MfccOptions
            {
                SampleRate = 16000,
                NumCoefficients = 13,
                NumMels = 40
            };
            var extractor = new MfccExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var mfccs = extractor.Extract(audio);

            // Assert
            Assert.Equal(13, mfccs.Shape[1]);
            Assert.True(mfccs.Shape[0] > 0, "Should have at least one frame");
        }

        [Fact]
        public void MfccExtractor_WithDelta_DoublesDimension()
        {
            // Arrange
            var options = new MfccOptions
            {
                SampleRate = 16000,
                NumCoefficients = 13,
                AppendDelta = true
            };
            var extractor = new MfccExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var mfccs = extractor.Extract(audio);

            // Assert
            Assert.Equal(26, mfccs.Shape[1]); // 13 MFCC + 13 delta
        }

        [Fact]
        public void MfccExtractor_WithDeltaDelta_TriplesDimension()
        {
            // Arrange
            var options = new MfccOptions
            {
                SampleRate = 16000,
                NumCoefficients = 13,
                AppendDelta = true,
                AppendDeltaDelta = true
            };
            var extractor = new MfccExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var mfccs = extractor.Extract(audio);

            // Assert
            Assert.Equal(39, mfccs.Shape[1]); // 13 MFCC + 13 delta + 13 delta-delta
        }

        [Fact]
        public void MfccExtractor_Name_IsMFCC()
        {
            // Arrange
            var extractor = new MfccExtractor<float>();

            // Assert
            Assert.Equal("MFCC", extractor.Name);
        }

        [Fact]
        public void MfccExtractor_FeatureDimension_MatchesCoefficients()
        {
            // Arrange
            var options = new MfccOptions { NumCoefficients = 20 };
            var extractor = new MfccExtractor<float>(options);

            // Assert
            Assert.Equal(20, extractor.FeatureDimension);
        }

        [Fact]
        public void MfccExtractor_OutputValues_AreFinite()
        {
            // Arrange
            var extractor = new MfccExtractor<float>();
            var audio = CreateTestAudio();

            // Act
            var mfccs = extractor.Extract(audio);

            // Assert
            for (int frame = 0; frame < mfccs.Shape[0]; frame++)
            {
                for (int coef = 0; coef < mfccs.Shape[1]; coef++)
                {
                    float value = mfccs[frame, coef];
                    Assert.False(float.IsNaN(value), $"MFCC[{frame},{coef}] is NaN");
                    Assert.False(float.IsInfinity(value), $"MFCC[{frame},{coef}] is Infinity");
                }
            }
        }

        [Fact]
        public void MfccExtractor_DifferentFrequencies_ProduceDifferentMFCCs()
        {
            // Arrange
            var extractor = new MfccExtractor<float>(new MfccOptions { SampleRate = 16000 });

            // Create two sine waves at different frequencies
            int numSamples = 16000;
            var audio1 = new Tensor<float>([numSamples]);
            var audio2 = new Tensor<float>([numSamples]);

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / 16000;
                audio1[i] = (float)Math.Sin(2 * Math.PI * 440 * t);  // 440 Hz
                audio2[i] = (float)Math.Sin(2 * Math.PI * 880 * t);  // 880 Hz
            }

            // Act
            var mfcc1 = extractor.Extract(audio1);
            var mfcc2 = extractor.Extract(audio2);

            // Assert - MFCCs should differ between different frequencies
            bool anyDifferent = false;
            for (int coef = 0; coef < Math.Min(mfcc1.Shape[1], 5); coef++)
            {
                if (Math.Abs(mfcc1[0, coef] - mfcc2[0, coef]) > 0.1)
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.True(anyDifferent, "MFCCs should differ for different frequencies");
        }
    }
}
