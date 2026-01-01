using System;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Fingerprinting
{
    public class FingerprintingTests
    {
        private static Tensor<float> CreateTestAudio(int sampleRate = 22050, double durationSeconds = 5.0)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);

            // Create a complex audio signal with multiple frequencies
            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;
                // Mix of frequencies to create more interesting fingerprint
                audio[i] = (float)(
                    0.5 * Math.Sin(2 * Math.PI * 440 * t) +
                    0.3 * Math.Sin(2 * Math.PI * 880 * t) +
                    0.2 * Math.Sin(2 * Math.PI * 1320 * t));
            }

            return audio;
        }

        [Fact]
        public void ChromaprintFingerprinter_Fingerprint_ReturnsNonEmptyHash()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();
            var audio = CreateTestAudio();

            // Act
            var fingerprint = fingerprinter.Fingerprint(audio);

            // Assert
            Assert.NotNull(fingerprint);
            Assert.NotNull(fingerprint.Hash);
            Assert.True(fingerprint.Hash.Length > 0, "Fingerprint hash should not be empty");
        }

        [Fact]
        public void ChromaprintFingerprinter_SameAudio_SameFingerprint()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();
            var audio = CreateTestAudio();

            // Act
            var fp1 = fingerprinter.Fingerprint(audio);
            var fp2 = fingerprinter.Fingerprint(audio);

            // Assert
            Assert.NotNull(fp1.Hash);
            Assert.NotNull(fp2.Hash);
            Assert.Equal(fp1.Hash, fp2.Hash);
        }

        [Fact]
        public void ChromaprintFingerprinter_DifferentAudio_DifferentFingerprint()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();

            int numSamples = 22050 * 5;
            var audio1 = new Tensor<float>([numSamples]);
            var audio2 = new Tensor<float>([numSamples]);

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / 22050;
                audio1[i] = (float)Math.Sin(2 * Math.PI * 440 * t);
                audio2[i] = (float)Math.Sin(2 * Math.PI * 880 * t);
            }

            // Act
            var fp1 = fingerprinter.Fingerprint(audio1);
            var fp2 = fingerprinter.Fingerprint(audio2);

            // Assert
            Assert.NotNull(fp1.Hash);
            Assert.NotNull(fp2.Hash);
            Assert.NotEqual(fp1.Hash, fp2.Hash);
        }

        [Fact]
        public void SpectrogramFingerprinter_Fingerprint_ReturnsNonEmptyHash()
        {
            // Arrange
            var fingerprinter = new SpectrogramFingerprinter<float>();
            var audio = CreateTestAudio();

            // Act
            var fingerprint = fingerprinter.Fingerprint(audio);

            // Assert
            Assert.NotNull(fingerprint);
            Assert.NotNull(fingerprint.Hash);
            Assert.True(fingerprint.Hash.Length > 0, "Fingerprint hash should not be empty");
        }

        [Fact]
        public void SpectrogramFingerprinter_SameAudio_SameFingerprint()
        {
            // Arrange
            var fingerprinter = new SpectrogramFingerprinter<float>();
            var audio = CreateTestAudio();

            // Act
            var fp1 = fingerprinter.Fingerprint(audio);
            var fp2 = fingerprinter.Fingerprint(audio);

            // Assert
            Assert.NotNull(fp1.Hash);
            Assert.NotNull(fp2.Hash);
            Assert.Equal(fp1.Hash, fp2.Hash);
        }

        [Fact]
        public void SpectrogramFingerprinter_HasFrameCount()
        {
            // Arrange
            var fingerprinter = new SpectrogramFingerprinter<float>();
            var audio = CreateTestAudio();

            // Act
            var fingerprint = fingerprinter.Fingerprint(audio);

            // Assert
            Assert.True(fingerprint.FrameCount > 0, "Should have frame count");
        }

        [Fact]
        public void ChromaprintFingerprinter_IdenticalFingerprints_HighSimilarity()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();
            var audio = CreateTestAudio();
            var fingerprint = fingerprinter.Fingerprint(audio);

            // Act
            double similarity = fingerprinter.ComputeSimilarity(fingerprint, fingerprint);

            // Assert
            Assert.Equal(1.0, similarity, 3); // Should be exactly 1.0
        }

        [Fact]
        public void ChromaprintFingerprinter_DifferentFingerprints_LowerSimilarity()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();

            int numSamples = 22050 * 5;
            var audio1 = new Tensor<float>([numSamples]);
            var audio2 = new Tensor<float>([numSamples]);

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / 22050;
                audio1[i] = (float)Math.Sin(2 * Math.PI * 440 * t);
                audio2[i] = (float)Math.Sin(2 * Math.PI * 880 * t);
            }

            var fp1 = fingerprinter.Fingerprint(audio1);
            var fp2 = fingerprinter.Fingerprint(audio2);

            // Act
            double similarity = fingerprinter.ComputeSimilarity(fp1, fp2);

            // Assert
            Assert.True(similarity < 1.0, "Different audio should have similarity < 1.0");
            Assert.True(similarity >= 0.0, "Similarity should be >= 0.0");
        }

        [Fact]
        public void ChromaprintFingerprinter_SimilaritySymmetric()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();

            int numSamples = 22050 * 3;
            var audio1 = new Tensor<float>([numSamples]);
            var audio2 = new Tensor<float>([numSamples]);

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / 22050;
                audio1[i] = (float)(Math.Sin(2 * Math.PI * 440 * t) + 0.5 * Math.Sin(2 * Math.PI * 550 * t));
                audio2[i] = (float)(Math.Sin(2 * Math.PI * 440 * t) + 0.5 * Math.Sin(2 * Math.PI * 660 * t));
            }

            var fp1 = fingerprinter.Fingerprint(audio1);
            var fp2 = fingerprinter.Fingerprint(audio2);

            // Act
            double sim1to2 = fingerprinter.ComputeSimilarity(fp1, fp2);
            double sim2to1 = fingerprinter.ComputeSimilarity(fp2, fp1);

            // Assert
            Assert.Equal(sim1to2, sim2to1, 6);
        }

        [Fact]
        public void AudioFingerprint_HasRequiredProperties()
        {
            // Arrange
            var fingerprinter = new ChromaprintFingerprinter<float>();
            var audio = CreateTestAudio();

            // Act
            var fingerprint = fingerprinter.Fingerprint(audio);

            // Assert
            Assert.NotNull(fingerprint.Data);
            Assert.True(fingerprint.Duration > 0, "Duration should be positive");
            Assert.True(fingerprint.SampleRate > 0, "SampleRate should be positive");
            Assert.False(string.IsNullOrEmpty(fingerprint.Algorithm), "Algorithm should be set");
        }
    }
}
