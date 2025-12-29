using System;
using AiDotNet.Audio.Speaker;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Speaker
{
    public class SpeakerTests
    {
        private static Tensor<float> CreateSpeakerAudio(int sampleRate = 16000, double durationSeconds = 2.0, double fundamentalFreq = 150)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);

            // Create a simple voice-like signal with fundamental and harmonics
            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;
                double signal = 0;

                // Fundamental frequency (pitch)
                signal += 0.5 * Math.Sin(2 * Math.PI * fundamentalFreq * t);

                // Add harmonics (2nd, 3rd, 4th)
                signal += 0.3 * Math.Sin(2 * Math.PI * fundamentalFreq * 2 * t);
                signal += 0.15 * Math.Sin(2 * Math.PI * fundamentalFreq * 3 * t);
                signal += 0.05 * Math.Sin(2 * Math.PI * fundamentalFreq * 4 * t);

                audio[i] = (float)signal;
            }

            return audio;
        }

        // SpeakerEmbeddingExtractor Tests
        [Fact]
        public void SpeakerEmbeddingExtractor_Extract_ReturnsEmbedding()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var audio = CreateSpeakerAudio();

            // Act
            var embedding = extractor.Extract(audio);

            // Assert
            Assert.NotNull(embedding);
            Assert.NotNull(embedding.Vector);
            Assert.True(embedding.Vector.Length > 0, "Embedding should not be empty");
        }

        [Fact]
        public void SpeakerEmbeddingExtractor_SameAudio_SimilarEmbeddings()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var audio = CreateSpeakerAudio();

            // Act
            var emb1 = extractor.Extract(audio);
            var emb2 = extractor.Extract(audio);

            // Assert
            Assert.Equal(emb1.Vector.Length, emb2.Vector.Length);
            for (int i = 0; i < emb1.Vector.Length; i++)
            {
                Assert.Equal(emb1.Vector[i], emb2.Vector[i], 5);
            }
        }

        [Fact]
        public void SpeakerEmbeddingExtractor_DifferentSpeakers_DifferentEmbeddings()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();

            // Create audio with different fundamental frequencies (simulating different speakers)
            var audio1 = CreateSpeakerAudio(fundamentalFreq: 120); // Male-like
            var audio2 = CreateSpeakerAudio(fundamentalFreq: 220); // Female-like

            // Act
            var emb1 = extractor.Extract(audio1);
            var emb2 = extractor.Extract(audio2);

            // Assert - embeddings should differ
            double sumDiff = 0;
            for (int i = 0; i < emb1.Vector.Length; i++)
            {
                sumDiff += Math.Abs(emb1.Vector[i] - emb2.Vector[i]);
            }
            Assert.True(sumDiff > 0.1, "Embeddings for different speakers should differ");
        }

        [Fact]
        public void SpeakerEmbeddingExtractor_EmbeddingValues_AreFinite()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var audio = CreateSpeakerAudio();

            // Act
            var embedding = extractor.Extract(audio);

            // Assert
            for (int i = 0; i < embedding.Vector.Length; i++)
            {
                Assert.False(float.IsNaN(embedding.Vector[i]), $"Embedding[{i}] is NaN");
                Assert.False(float.IsInfinity(embedding.Vector[i]), $"Embedding[{i}] is Infinity");
            }
        }

        [Fact]
        public void SpeakerEmbeddingExtractor_EmbeddingDimension_MatchesOption()
        {
            // Arrange
            var options = new SpeakerEmbeddingOptions { EmbeddingDimension = 128 };
            var extractor = new SpeakerEmbeddingExtractor<float>(options);

            // Assert
            Assert.Equal(128, extractor.EmbeddingDimension);
        }

        [Fact]
        public void SpeakerEmbedding_CosineSimilarity_SameEmbedding_IsOne()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var audio = CreateSpeakerAudio();
            var embedding = extractor.Extract(audio);

            // Act
            double similarity = embedding.CosineSimilarity(embedding);

            // Assert
            Assert.Equal(1.0, similarity, 3);
        }

        [Fact]
        public void SpeakerEmbedding_CosineSimilarity_InRange()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var audio1 = CreateSpeakerAudio(fundamentalFreq: 150);
            var audio2 = CreateSpeakerAudio(fundamentalFreq: 200);
            var emb1 = extractor.Extract(audio1);
            var emb2 = extractor.Extract(audio2);

            // Act
            double similarity = emb1.CosineSimilarity(emb2);

            // Assert
            Assert.True(similarity >= -1.0 && similarity <= 1.0,
                $"Cosine similarity {similarity} should be in [-1, 1]");
        }

        // SpeakerVerifier Tests
        [Fact]
        public void SpeakerVerifier_EnrollAndVerify_Works()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var verifier = new SpeakerVerifier<float>();
            var audio = CreateSpeakerAudio();
            var embedding = extractor.Extract(audio);

            // Act
            verifier.Enroll("speaker1", embedding);
            var result = verifier.Verify("speaker1", embedding);

            // Assert
            Assert.True(result.IsVerified, "Same speaker should verify");
            Assert.Equal(1.0, result.Score, 2);
        }

        [Fact]
        public void SpeakerVerifier_EnrolledCount_IncreasesAfterEnroll()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var verifier = new SpeakerVerifier<float>();
            var audio = CreateSpeakerAudio();
            var embedding = extractor.Extract(audio);

            // Assert initial state
            Assert.Equal(0, verifier.EnrolledCount);

            // Act
            verifier.Enroll("speaker1", embedding);

            // Assert
            Assert.Equal(1, verifier.EnrolledCount);
        }

        [Fact]
        public void SpeakerVerifier_Unenroll_RemovesSpeaker()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var verifier = new SpeakerVerifier<float>();
            var audio = CreateSpeakerAudio();
            verifier.Enroll("speaker1", extractor.Extract(audio));

            // Act
            bool removed = verifier.Unenroll("speaker1");

            // Assert
            Assert.True(removed);
            Assert.Equal(0, verifier.EnrolledCount);
        }

        [Fact]
        public void SpeakerVerifier_IsEnrolled_ReturnsCorrectly()
        {
            // Arrange
            var extractor = new SpeakerEmbeddingExtractor<float>();
            var verifier = new SpeakerVerifier<float>();
            var audio = CreateSpeakerAudio();
            verifier.Enroll("speaker1", extractor.Extract(audio));

            // Assert
            Assert.True(verifier.IsEnrolled("speaker1"));
            Assert.False(verifier.IsEnrolled("speaker2"));
        }

        // SpeakerDiarizer Tests
        [Fact]
        public void SpeakerDiarizer_Diarize_ReturnsTurns()
        {
            // Arrange
            var diarizer = new SpeakerDiarizer<float>();
            var audio = CreateSpeakerAudio(durationSeconds: 3.0);

            // Act
            var result = diarizer.Diarize(audio);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Turns);
            Assert.True(result.Turns.Count >= 1, "Should have at least one speaker turn");
        }

        [Fact]
        public void SpeakerDiarizer_Turns_HaveValidTimestamps()
        {
            // Arrange
            var diarizer = new SpeakerDiarizer<float>();
            var audio = CreateSpeakerAudio(durationSeconds: 3.0);

            // Act
            var result = diarizer.Diarize(audio);

            // Assert
            foreach (var turn in result.Turns)
            {
                Assert.True(turn.StartTime >= 0, $"StartTime {turn.StartTime} should be >= 0");
                Assert.True(turn.EndTime > turn.StartTime,
                    $"EndTime {turn.EndTime} should be > StartTime {turn.StartTime}");
                Assert.False(string.IsNullOrEmpty(turn.SpeakerId), "SpeakerId should not be empty");
            }
        }

        [Fact]
        public void SpeakerDiarizer_NumSpeakers_IsPositive()
        {
            // Arrange
            var diarizer = new SpeakerDiarizer<float>();
            var audio = CreateSpeakerAudio(durationSeconds: 3.0);

            // Act
            var result = diarizer.Diarize(audio);

            // Assert
            Assert.True(result.NumSpeakers >= 1, "Should detect at least one speaker");
        }

        [Fact]
        public void SpeakerDiarizer_Duration_MatchesAudio()
        {
            // Arrange
            var diarizer = new SpeakerDiarizer<float>();
            var audio = CreateSpeakerAudio(durationSeconds: 3.0, sampleRate: 16000);

            // Act
            var result = diarizer.Diarize(audio);

            // Assert
            Assert.Equal(3.0, result.Duration, 1);
        }
    }
}
