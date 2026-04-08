using System;
using AiDotNet.Audio.Features;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Features
{
    public class ChromaExtractorTests
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
        public void ChromaExtractor_DefaultOptions_Extracts12PitchClasses()
        {
            // Arrange
            var extractor = new ChromaExtractor<float>();
            var audio = CreateTestAudio();

            // Act
            var chroma = extractor.Extract(audio);

            // Assert
            Assert.Equal(12, chroma.Shape[1]); // 12 pitch classes
            Assert.True(chroma.Shape[0] > 0, "Should have at least one frame");
        }

        [Fact]
        public void ChromaExtractor_Name_IsChroma()
        {
            // Arrange
            var extractor = new ChromaExtractor<float>();

            // Assert
            Assert.Equal("Chroma", extractor.Name);
        }

        [Fact]
        public void ChromaExtractor_FeatureDimension_Is12()
        {
            // Arrange
            var extractor = new ChromaExtractor<float>();

            // Assert
            Assert.Equal(12, extractor.FeatureDimension);
        }

        [Fact]
        public void ChromaExtractor_OutputValues_AreFinite()
        {
            // Arrange
            var extractor = new ChromaExtractor<float>();
            var audio = CreateTestAudio();

            // Act
            var chroma = extractor.Extract(audio);

            // Assert
            for (int frame = 0; frame < chroma.Shape[0]; frame++)
            {
                for (int pitchClass = 0; pitchClass < 12; pitchClass++)
                {
                    float value = chroma[frame, pitchClass];
                    Assert.False(float.IsNaN(value), $"Chroma[{frame},{pitchClass}] is NaN");
                    Assert.False(float.IsInfinity(value), $"Chroma[{frame},{pitchClass}] is Infinity");
                }
            }
        }

        [Fact]
        public void ChromaExtractor_Normalized_ValuesInRange()
        {
            // Arrange
            var options = new ChromaOptions { Normalize = true };
            var extractor = new ChromaExtractor<float>(options);
            var audio = CreateTestAudio();

            // Act
            var chroma = extractor.Extract(audio);

            // Assert - normalized chroma values should be bounded
            for (int frame = 0; frame < chroma.Shape[0]; frame++)
            {
                for (int pitchClass = 0; pitchClass < 12; pitchClass++)
                {
                    float value = chroma[frame, pitchClass];
                    Assert.True(value >= 0, $"Chroma[{frame},{pitchClass}] should be >= 0, was {value}");
                    Assert.True(value <= 1.0001, $"Chroma[{frame},{pitchClass}] should be <= 1, was {value}");
                }
            }
        }

        [Fact]
        public void GetPitchClassName_ReturnsCorrectNames()
        {
            Assert.Equal("C", ChromaExtractor<float>.GetPitchClassName(0));
            Assert.Equal("C#", ChromaExtractor<float>.GetPitchClassName(1));
            Assert.Equal("D", ChromaExtractor<float>.GetPitchClassName(2));
            Assert.Equal("D#", ChromaExtractor<float>.GetPitchClassName(3));
            Assert.Equal("E", ChromaExtractor<float>.GetPitchClassName(4));
            Assert.Equal("F", ChromaExtractor<float>.GetPitchClassName(5));
            Assert.Equal("F#", ChromaExtractor<float>.GetPitchClassName(6));
            Assert.Equal("G", ChromaExtractor<float>.GetPitchClassName(7));
            Assert.Equal("G#", ChromaExtractor<float>.GetPitchClassName(8));
            Assert.Equal("A", ChromaExtractor<float>.GetPitchClassName(9));
            Assert.Equal("A#", ChromaExtractor<float>.GetPitchClassName(10));
            Assert.Equal("B", ChromaExtractor<float>.GetPitchClassName(11));
        }

        [Fact]
        public void GetPitchClassName_WrapsAround()
        {
            // Test that indices > 11 wrap around
            Assert.Equal("C", ChromaExtractor<float>.GetPitchClassName(12));
            Assert.Equal("D", ChromaExtractor<float>.GetPitchClassName(14));
        }

        [Fact]
        public void GetDominantPitchClass_ReturnsMaxIndex()
        {
            // Arrange
            var extractor = new ChromaExtractor<float>();
            var chromaFrame = new float[12];
            chromaFrame[9] = 1.0f; // A is dominant

            // Act
            var dominant = extractor.GetDominantPitchClass(chromaFrame);

            // Assert
            Assert.Equal(9, dominant); // A = index 9
        }

        [Fact]
        public void GetDominantPitchClass_ThrowsForWrongLength()
        {
            // Arrange
            var extractor = new ChromaExtractor<float>();
            var wrongLength = new float[10];

            // Act & Assert
            Assert.Throws<ArgumentException>(() => extractor.GetDominantPitchClass(wrongLength));
        }
    }
}
