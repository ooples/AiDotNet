using System;
using AiDotNet.Audio.MusicAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.MusicAnalysis
{
    public class MusicAnalysisTests
    {
        private static Tensor<float> CreateRhythmicAudio(int sampleRate = 22050, double durationSeconds = 5.0, double bpm = 120)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);

            double beatsPerSecond = bpm / 60.0;
            double samplesPerBeat = sampleRate / beatsPerSecond;

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;
                double beatPhase = (i % samplesPerBeat) / samplesPerBeat;

                // Create a kick-like sound at each beat
                double envelope = Math.Exp(-beatPhase * 10);
                double tone = Math.Sin(2 * Math.PI * 100 * t) * envelope;

                // Add some harmonic content
                double harmonic = 0.3 * Math.Sin(2 * Math.PI * 440 * t);

                audio[i] = (float)(tone + harmonic);
            }

            return audio;
        }

        private static Tensor<float> CreateMajorChordAudio(int sampleRate = 22050, double durationSeconds = 2.0)
        {
            int numSamples = (int)(sampleRate * durationSeconds);
            var audio = new Tensor<float>([numSamples]);

            // C major chord: C4 (261.63), E4 (329.63), G4 (392.00)
            double c4 = 261.63;
            double e4 = 329.63;
            double g4 = 392.00;

            for (int i = 0; i < numSamples; i++)
            {
                double t = (double)i / sampleRate;
                audio[i] = (float)(
                    0.33 * Math.Sin(2 * Math.PI * c4 * t) +
                    0.33 * Math.Sin(2 * Math.PI * e4 * t) +
                    0.33 * Math.Sin(2 * Math.PI * g4 * t));
            }

            return audio;
        }

        // BeatTracker Tests
        [Fact]
        public void BeatTracker_Track_ReturnsResult()
        {
            // Arrange
            var tracker = new BeatTracker<float>();
            var audio = CreateRhythmicAudio(bpm: 120);

            // Act
            var result = tracker.Track(audio);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Tempo > 0, "Tempo should be positive");
        }

        [Fact]
        public void BeatTracker_DetectTempo_ReturnsReasonableTempo()
        {
            // Arrange
            var tracker = new BeatTracker<float>();
            var audio = CreateRhythmicAudio(bpm: 120);

            // Act
            var result = tracker.Track(audio);

            // Assert
            Assert.True(result.Tempo > 60 && result.Tempo < 200,
                $"Tempo {result.Tempo} should be in reasonable range");
        }

        [Fact]
        public void BeatTracker_DetectBeats_ReturnsMultipleBeats()
        {
            // Arrange
            var tracker = new BeatTracker<float>();
            var audio = CreateRhythmicAudio(durationSeconds: 5.0, bpm: 120);

            // Act
            var result = tracker.Track(audio);

            // Assert
            Assert.NotNull(result.BeatTimes);
            Assert.True(result.BeatTimes.Count > 0, "Should detect at least some beats");
        }

        [Fact]
        public void BeatTracker_ConfidenceScore_InRange()
        {
            // Arrange
            var tracker = new BeatTracker<float>();
            var audio = CreateRhythmicAudio();

            // Act
            var result = tracker.Track(audio);

            // Assert
            Assert.True(result.ConfidenceScore >= 0 && result.ConfidenceScore <= 1,
                $"Confidence {result.ConfidenceScore} should be in [0, 1]");
        }

        [Fact]
        public void BeatTracker_AverageBeatInterval_MatchesTempo()
        {
            // Arrange
            var tracker = new BeatTracker<float>();
            var audio = CreateRhythmicAudio();

            // Act
            var result = tracker.Track(audio);

            // Assert
            double expectedInterval = 60.0 / result.Tempo;
            Assert.Equal(expectedInterval, result.AverageBeatInterval, 3);
        }

        // ChordRecognizer Tests
        [Fact]
        public void ChordRecognizer_Recognize_ReturnsChords()
        {
            // Arrange
            var recognizer = new ChordRecognizer<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var chords = recognizer.Recognize(audio);

            // Assert
            Assert.NotNull(chords);
            Assert.True(chords.Count > 0, "Should recognize at least one chord");
        }

        [Fact]
        public void ChordRecognizer_ChordSegment_HasValidProperties()
        {
            // Arrange
            var recognizer = new ChordRecognizer<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var chords = recognizer.Recognize(audio);

            // Assert
            foreach (var chord in chords)
            {
                Assert.False(string.IsNullOrEmpty(chord.Name), "Chord name should not be empty");
                Assert.True(chord.StartTime >= 0, "StartTime should be non-negative");
                Assert.True(chord.EndTime >= chord.StartTime, "EndTime should be >= StartTime");
                Assert.True(chord.Duration >= 0, "Duration should be non-negative");
            }
        }

        [Fact]
        public void ChordRecognizer_ChordConfidence_InRange()
        {
            // Arrange
            var recognizer = new ChordRecognizer<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var chords = recognizer.Recognize(audio);

            // Assert
            foreach (var chord in chords)
            {
                Assert.True(chord.Confidence >= 0 && chord.Confidence <= 1,
                    $"Chord confidence {chord.Confidence} should be in [0, 1]");
            }
        }

        // KeyDetector Tests
        [Fact]
        public void KeyDetector_Detect_ReturnsKey()
        {
            // Arrange
            var detector = new KeyDetector<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var key = detector.Detect(audio);

            // Assert
            Assert.NotNull(key);
            Assert.True(key.KeyIndex >= 0 && key.KeyIndex < 12, $"Key index {key.KeyIndex} should be in [0, 11]");
        }

        [Fact]
        public void KeyDetector_KeyCorrelation_InRange()
        {
            // Arrange
            var detector = new KeyDetector<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var key = detector.Detect(audio);

            // Assert
            Assert.True(key.Correlation >= -1 && key.Correlation <= 1,
                $"Key correlation {key.Correlation} should be in [-1, 1]");
        }

        [Fact]
        public void KeyDetector_ModeIsValid()
        {
            // Arrange
            var detector = new KeyDetector<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var key = detector.Detect(audio);

            // Assert
            Assert.True(key.Mode == KeyMode.Major || key.Mode == KeyMode.Minor,
                "Mode should be Major or Minor");
        }

        [Fact]
        public void KeyDetector_NameIsNotEmpty()
        {
            // Arrange
            var detector = new KeyDetector<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var key = detector.Detect(audio);

            // Assert
            Assert.False(string.IsNullOrEmpty(key.Name), "Key name should not be empty");
        }

        [Fact]
        public void KeyDetector_RelativeKey_IsValid()
        {
            // Arrange
            var detector = new KeyDetector<float>();
            var audio = CreateMajorChordAudio();

            // Act
            var key = detector.Detect(audio);

            // Assert
            Assert.False(string.IsNullOrEmpty(key.RelativeKey),
                "Relative key should not be empty");
        }
    }
}
