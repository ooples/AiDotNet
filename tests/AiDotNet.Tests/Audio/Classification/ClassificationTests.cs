using System;
using System.Linq;
using AiDotNet.Audio.Classification;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Classification
{
    public class ClassificationTests
    {
        private static Tensor<float> CreateTestAudio(int sampleRate = 22050, double durationSeconds = 3.0,
            double frequency = 440, bool addNoise = false)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);
            var rand = new Random(42);

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;

                // Base signal
                double signal = 0.5 * Math.Sin(2 * Math.PI * frequency * t);

                // Add harmonics for richer content
                signal += 0.25 * Math.Sin(2 * Math.PI * frequency * 2 * t);
                signal += 0.125 * Math.Sin(2 * Math.PI * frequency * 3 * t);

                // Add noise if requested
                if (addNoise)
                {
                    signal += 0.1 * (rand.NextDouble() * 2 - 1);
                }

                audio[i] = (float)signal;
            }

            return audio;
        }

        private static Tensor<float> CreateMusicLikeAudio(int sampleRate = 22050, double durationSeconds = 3.0)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);
            var rand = new Random(42);

            // Create a more complex music-like signal with beat pattern
            double tempo = 120.0; // BPM
            double beatPeriod = 60.0 / tempo;

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;

                // Bass drum on beat
                double beat = Math.Sin(2 * Math.PI * t / beatPeriod);
                double kickEnvelope = Math.Max(0, beat) * Math.Exp(-5 * (t % beatPeriod));
                double kick = kickEnvelope * Math.Sin(2 * Math.PI * 60 * t);

                // Melody (simple sinusoid with vibrato)
                double vibrato = Math.Sin(2 * Math.PI * 5 * t) * 10;
                double melody = 0.3 * Math.Sin(2 * Math.PI * (440 + vibrato) * t);

                // Combine
                audio[i] = (float)(kick * 0.5 + melody);
            }

            return audio;
        }

        private static Tensor<float> CreateSpeechLikeAudio(int sampleRate = 16000, double durationSeconds = 2.0)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);

            // Create a speech-like signal with formants
            double fundamentalFreq = 150; // F0 for male voice

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;

                // Glottal pulse train
                double glottal = 0.5 * Math.Sin(2 * Math.PI * fundamentalFreq * t);

                // First formant (vowel-like)
                double f1 = 0.3 * Math.Sin(2 * Math.PI * 500 * t);

                // Second formant
                double f2 = 0.2 * Math.Sin(2 * Math.PI * 1500 * t);

                audio[i] = (float)(glottal + f1 + f2);
            }

            return audio;
        }

        // ==================== GenreClassifier Tests ====================

        [Fact]
        public void GenreClassifier_Classify_ReturnsResult()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();
            var audio = CreateMusicLikeAudio();

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result);
            Assert.False(string.IsNullOrEmpty(result.PredictedGenre));
            Assert.True(result.Confidence >= 0 && result.Confidence <= 1);
        }

        [Fact]
        public void GenreClassifier_Classify_HasAllProbabilities()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();
            var audio = CreateMusicLikeAudio();

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result.AllProbabilities);
            Assert.True(result.AllProbabilities.Count > 0);

            // Probabilities should sum to approximately 1
            double sum = result.AllProbabilities.Values.Sum();
            Assert.True(Math.Abs(sum - 1.0) < 0.01, $"Probabilities sum {sum} should be ~1");
        }

        [Fact]
        public void GenreClassifier_Classify_TopPredictionsAreOrdered()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();
            var audio = CreateMusicLikeAudio();

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result.TopPredictions);
            Assert.True(result.TopPredictions.Count > 0);

            for (int i = 1; i < result.TopPredictions.Count; i++)
            {
                Assert.True(result.TopPredictions[i - 1].Probability >= result.TopPredictions[i].Probability,
                    "Top predictions should be ordered by probability descending");
            }
        }

        [Fact]
        public void GenreClassifier_Genres_HasStandardGenres()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();

            // Assert
            Assert.NotNull(classifier.Genres);
            Assert.True(classifier.Genres.Length >= 10);
            Assert.Contains("rock", classifier.Genres);
            Assert.Contains("jazz", classifier.Genres);
            Assert.Contains("classical", classifier.Genres);
        }

        [Fact]
        public void GenreClassifier_Features_AreExtracted()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();
            var audio = CreateMusicLikeAudio();

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result.Features);
            Assert.NotNull(result.Features.MfccMean);
            Assert.True(result.Features.MfccMean.Length > 0);
            Assert.True(result.Features.Tempo > 0);
        }

        // ==================== AudioEventDetector Tests ====================

        [Fact]
        public void AudioEventDetector_Detect_ReturnsEvents()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();
            var audio = CreateSpeechLikeAudio();

            // Act
            var events = detector.Detect(audio);

            // Assert
            Assert.NotNull(events);
            // Should detect at least something (the threshold is 0.3 by default)
        }

        [Fact]
        public void AudioEventDetector_Events_HaveValidTimestamps()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();
            var audio = CreateSpeechLikeAudio(durationSeconds: 3.0);

            // Act
            var events = detector.Detect(audio);

            // Assert
            foreach (var evt in events)
            {
                Assert.True(evt.StartTime >= 0, $"StartTime {evt.StartTime} should be >= 0");
                Assert.True(evt.EndTime > evt.StartTime,
                    $"EndTime {evt.EndTime} should be > StartTime {evt.StartTime}");
                Assert.False(string.IsNullOrEmpty(evt.Label), "Label should not be empty");
                Assert.True(evt.Confidence >= 0 && evt.Confidence <= 1,
                    $"Confidence {evt.Confidence} should be in [0, 1]");
            }
        }

        [Fact]
        public void AudioEventDetector_DetectFrame_ReturnsScores()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();
            var audio = CreateSpeechLikeAudio(durationSeconds: 1.0);

            // Act
            var scores = detector.DetectFrame(audio);

            // Assert
            Assert.NotNull(scores);
            Assert.True(scores.Count > 0);

            foreach (var (label, score) in scores)
            {
                Assert.False(string.IsNullOrEmpty(label));
                Assert.True(score >= 0 && score <= 1, $"Score {score} for {label} should be in [0, 1]");
            }
        }

        [Fact]
        public void AudioEventDetector_DetectTopK_ReturnsCorrectCount()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();
            var audio = CreateSpeechLikeAudio(durationSeconds: 1.0);

            // Act
            var topK = detector.DetectTopK(audio, 5);

            // Assert
            Assert.NotNull(topK);
            Assert.True(topK.Count <= 5);
        }

        [Fact]
        public void AudioEventDetector_EventLabels_HasCommonEvents()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();

            // Assert
            Assert.NotNull(detector.EventLabels);
            Assert.Contains("Speech", detector.EventLabels);
            Assert.Contains("Music", detector.EventLabels);
        }

        // ==================== SceneClassifier Tests ====================

        [Fact]
        public void SceneClassifier_Classify_ReturnsResult()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();
            var audio = CreateTestAudio(addNoise: true);

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result);
            Assert.False(string.IsNullOrEmpty(result.PredictedScene));
            Assert.False(string.IsNullOrEmpty(result.Category));
            Assert.True(result.Confidence >= 0 && result.Confidence <= 1);
        }

        [Fact]
        public void SceneClassifier_Classify_HasAllProbabilities()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();
            var audio = CreateTestAudio(addNoise: true);

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result.AllProbabilities);
            Assert.True(result.AllProbabilities.Count > 0);

            // Probabilities should sum to approximately 1
            double sum = result.AllProbabilities.Values.Sum();
            Assert.True(Math.Abs(sum - 1.0) < 0.01, $"Probabilities sum {sum} should be ~1");
        }

        [Fact]
        public void SceneClassifier_ClassifyCategory_ReturnsCategory()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();
            var audio = CreateTestAudio(addNoise: true);

            // Act
            var (category, confidence) = classifier.ClassifyCategory(audio);

            // Assert
            Assert.False(string.IsNullOrEmpty(category));
            Assert.True(confidence >= 0 && confidence <= 1);
        }

        [Fact]
        public void SceneClassifier_Scenes_HasStandardScenes()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();

            // Assert
            Assert.NotNull(classifier.Scenes);
            Assert.True(classifier.Scenes.Length >= 10);
        }

        [Fact]
        public void SceneClassifier_Features_AreExtracted()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();
            var audio = CreateTestAudio(addNoise: true);

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result.Features);
            Assert.NotNull(result.Features.MfccMean);
            Assert.True(result.Features.MfccMean.Length > 0);
            Assert.NotNull(result.Features.BandEnergies);
        }

        [Fact]
        public void SceneClassifier_TopPredictions_AreOrdered()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();
            var audio = CreateTestAudio(addNoise: true);

            // Act
            var result = classifier.Classify(audio);

            // Assert
            Assert.NotNull(result.TopPredictions);
            Assert.True(result.TopPredictions.Count > 0);

            for (int i = 1; i < result.TopPredictions.Count; i++)
            {
                Assert.True(result.TopPredictions[i - 1].Probability >= result.TopPredictions[i].Probability,
                    "Top predictions should be ordered by probability descending");
            }
        }

        // ==================== Dispose Tests ====================

        [Fact]
        public void GenreClassifier_Dispose_DoesNotThrow()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();

            // Act & Assert
            classifier.Dispose();
            // Should not throw
        }

        [Fact]
        public void AudioEventDetector_Dispose_DoesNotThrow()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();

            // Act & Assert
            detector.Dispose();
            // Should not throw
        }

        [Fact]
        public void SceneClassifier_Dispose_DoesNotThrow()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();

            // Act & Assert
            classifier.Dispose();
            // Should not throw
        }

        [Fact]
        public void GenreClassifier_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var classifier = new GenreClassifier<float>();
            var audio = CreateMusicLikeAudio();
            classifier.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => classifier.Classify(audio));
        }

        [Fact]
        public void AudioEventDetector_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var detector = new AudioEventDetector<float>();
            var audio = CreateSpeechLikeAudio();
            detector.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => detector.Detect(audio));
        }

        [Fact]
        public void SceneClassifier_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var classifier = new SceneClassifier<float>();
            var audio = CreateTestAudio();
            classifier.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => classifier.Classify(audio));
        }
    }
}
