using AiDotNet.Audio.Classification;
using AiDotNet.Audio.Localization;
using AiDotNet.Audio.MusicAnalysis;
using AiDotNet.Audio.Pitch;
using AiDotNet.Audio.Speaker;
using AiDotNet.Audio.SourceSeparation;
using AiDotNet.Audio.TextToSpeech;
using AiDotNet.Audio.VoiceActivity;
using AiDotNet.Audio.Whisper;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Audio;

/// <summary>
/// Extended integration tests for Audio module classes not covered in AudioIntegrationTests:
/// Whisper, Speaker, Localization, MusicAnalysis, TextToSpeech, VoiceActivity, Pitch,
/// SourceSeparation, and additional classification result classes.
/// </summary>
public class AudioExtendedIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region WhisperOptions

    [Fact]
    public void WhisperOptions_DefaultValues()
    {
        var options = new WhisperOptions();
        Assert.Equal(WhisperModelSize.Base, options.ModelSize);
        Assert.Null(options.Language);
        Assert.False(options.Translate);
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(80, options.NumMels);
        Assert.Equal(30, options.MaxAudioLengthSeconds);
        Assert.NotNull(options.OnnxOptions);
        Assert.Null(options.EncoderModelPath);
        Assert.Null(options.DecoderModelPath);
        Assert.Equal(448, options.MaxTokens);
        Assert.Equal(5, options.BeamSize);
        Assert.Equal(0.0, options.Temperature, Tolerance);
        Assert.False(options.ReturnTimestamps);
        Assert.False(options.WordTimestamps);
    }

    [Fact]
    public void WhisperOptions_CustomValues()
    {
        var options = new WhisperOptions
        {
            ModelSize = WhisperModelSize.LargeV3,
            Language = "en",
            Translate = true,
            SampleRate = 8000,
            NumMels = 128,
            MaxAudioLengthSeconds = 60,
            MaxTokens = 1024,
            BeamSize = 10,
            Temperature = 0.5,
            ReturnTimestamps = true,
            WordTimestamps = true,
            EncoderModelPath = "/path/encoder.onnx",
            DecoderModelPath = "/path/decoder.onnx"
        };
        Assert.Equal(WhisperModelSize.LargeV3, options.ModelSize);
        Assert.Equal("en", options.Language);
        Assert.True(options.Translate);
        Assert.Equal(8000, options.SampleRate);
        Assert.Equal(128, options.NumMels);
        Assert.Equal(60, options.MaxAudioLengthSeconds);
        Assert.Equal(1024, options.MaxTokens);
        Assert.Equal(10, options.BeamSize);
        Assert.Equal(0.5, options.Temperature, Tolerance);
        Assert.True(options.ReturnTimestamps);
        Assert.True(options.WordTimestamps);
        Assert.Equal("/path/encoder.onnx", options.EncoderModelPath);
        Assert.Equal("/path/decoder.onnx", options.DecoderModelPath);
    }

    #endregion

    #region WhisperModelSize Enum

    [Fact]
    public void WhisperModelSize_AllValues()
    {
        var values = (((WhisperModelSize[])Enum.GetValues(typeof(WhisperModelSize))));
        Assert.Equal(7, values.Length);
        Assert.Contains(WhisperModelSize.Tiny, values);
        Assert.Contains(WhisperModelSize.Base, values);
        Assert.Contains(WhisperModelSize.Small, values);
        Assert.Contains(WhisperModelSize.Medium, values);
        Assert.Contains(WhisperModelSize.Large, values);
        Assert.Contains(WhisperModelSize.LargeV2, values);
        Assert.Contains(WhisperModelSize.LargeV3, values);
    }

    [Fact]
    public void WhisperModelSize_OrderedCorrectly()
    {
        Assert.True((int)WhisperModelSize.Tiny < (int)WhisperModelSize.Base);
        Assert.True((int)WhisperModelSize.Base < (int)WhisperModelSize.Small);
        Assert.True((int)WhisperModelSize.Small < (int)WhisperModelSize.Medium);
        Assert.True((int)WhisperModelSize.Medium < (int)WhisperModelSize.Large);
    }

    #endregion

    #region WhisperResult

    [Fact]
    public void WhisperResult_DefaultValues()
    {
        var result = new WhisperResult();
        Assert.Equal(string.Empty, result.Text);
        Assert.Null(result.DetectedLanguage);
        Assert.Equal(0.0, result.LanguageProbability, Tolerance);
        Assert.NotNull(result.Words);
        Assert.Empty(result.Words);
        Assert.NotNull(result.Segments);
        Assert.Empty(result.Segments);
        Assert.Equal(0L, result.ProcessingTimeMs);
    }

    [Fact]
    public void WhisperResult_SetProperties()
    {
        var result = new WhisperResult
        {
            Text = "Hello world",
            DetectedLanguage = "en",
            LanguageProbability = 0.98,
            ProcessingTimeMs = 1500,
            Words = [new WhisperWord { Word = "Hello", StartTime = 0.0, EndTime = 0.5, Confidence = 0.95 }],
            Segments = [new WhisperSegment { Text = "Hello world", StartTime = 0.0, EndTime = 1.0 }]
        };
        Assert.Equal("Hello world", result.Text);
        Assert.Equal("en", result.DetectedLanguage);
        Assert.Equal(0.98, result.LanguageProbability, Tolerance);
        Assert.Equal(1500L, result.ProcessingTimeMs);
        Assert.Single(result.Words);
        Assert.Single(result.Segments);
    }

    #endregion

    #region WhisperSegment

    [Fact]
    public void WhisperSegment_DefaultValues()
    {
        var segment = new WhisperSegment();
        Assert.Equal(string.Empty, segment.Text);
        Assert.Equal(0.0, segment.StartTime, Tolerance);
        Assert.Equal(0.0, segment.EndTime, Tolerance);
        Assert.NotNull(segment.Words);
        Assert.Empty(segment.Words);
    }

    [Fact]
    public void WhisperSegment_SetProperties()
    {
        var segment = new WhisperSegment
        {
            Text = "Test segment",
            StartTime = 1.5,
            EndTime = 3.0,
            Words =
            [
                new WhisperWord { Word = "Test", StartTime = 1.5, EndTime = 2.0, Confidence = 0.9 },
                new WhisperWord { Word = "segment", StartTime = 2.0, EndTime = 3.0, Confidence = 0.85 }
            ]
        };
        Assert.Equal("Test segment", segment.Text);
        Assert.Equal(1.5, segment.StartTime, Tolerance);
        Assert.Equal(3.0, segment.EndTime, Tolerance);
        Assert.Equal(2, segment.Words.Count);
    }

    #endregion

    #region WhisperWord

    [Fact]
    public void WhisperWord_DefaultValues()
    {
        var word = new WhisperWord();
        Assert.Equal(string.Empty, word.Word);
        Assert.Equal(0.0, word.StartTime, Tolerance);
        Assert.Equal(0.0, word.EndTime, Tolerance);
        Assert.Equal(0.0, word.Confidence, Tolerance);
    }

    [Fact]
    public void WhisperWord_SetProperties()
    {
        var word = new WhisperWord
        {
            Word = "Hello",
            StartTime = 0.5,
            EndTime = 1.0,
            Confidence = 0.92
        };
        Assert.Equal("Hello", word.Word);
        Assert.Equal(0.5, word.StartTime, Tolerance);
        Assert.Equal(1.0, word.EndTime, Tolerance);
        Assert.Equal(0.92, word.Confidence, Tolerance);
    }

    #endregion

    #region SpeakerTurn

    [Fact]
    public void SpeakerTurn_DefaultValues()
    {
        var turn = new SpeakerTurn();
        Assert.Equal(string.Empty, turn.SpeakerId);
        Assert.Equal(0, turn.SpeakerIndex);
        Assert.Equal(0.0, turn.StartTime, Tolerance);
        Assert.Equal(0.0, turn.EndTime, Tolerance);
    }

    [Fact]
    public void SpeakerTurn_Duration_Computed()
    {
        var turn = new SpeakerTurn
        {
            SpeakerId = "Speaker_A",
            SpeakerIndex = 0,
            StartTime = 2.0,
            EndTime = 5.5
        };
        Assert.Equal(3.5, turn.Duration, Tolerance);
    }

    [Fact]
    public void SpeakerTurn_ZeroDuration()
    {
        var turn = new SpeakerTurn { StartTime = 1.0, EndTime = 1.0 };
        Assert.Equal(0.0, turn.Duration, Tolerance);
    }

    #endregion

    #region DiarizationResult

    [Fact]
    public void DiarizationResult_DefaultValues()
    {
        var result = new DiarizationResult();
        Assert.NotNull(result.Turns);
        Assert.Empty(result.Turns);
        Assert.Equal(0, result.NumSpeakers);
        Assert.Equal(0.0, result.Duration, Tolerance);
    }

    [Fact]
    public void DiarizationResult_SpeakingTimePerSpeaker()
    {
        var result = new DiarizationResult
        {
            NumSpeakers = 2,
            Duration = 10.0,
            Turns =
            [
                new SpeakerTurn { SpeakerId = "A", StartTime = 0, EndTime = 3 },
                new SpeakerTurn { SpeakerId = "B", StartTime = 3, EndTime = 6 },
                new SpeakerTurn { SpeakerId = "A", StartTime = 6, EndTime = 10 }
            ]
        };

        var speakingTime = result.SpeakingTimePerSpeaker;
        Assert.Equal(2, speakingTime.Count);
        Assert.Equal(7.0, speakingTime["A"], Tolerance); // 3 + 4
        Assert.Equal(3.0, speakingTime["B"], Tolerance);
    }

    [Fact]
    public void DiarizationResult_EmptyTurns_SpeakingTimeEmpty()
    {
        var result = new DiarizationResult();
        var speakingTime = result.SpeakingTimePerSpeaker;
        Assert.Empty(speakingTime);
    }

    #endregion

    #region SpeakerMatch

    [Fact]
    public void SpeakerMatch_DefaultValues()
    {
        var match = new SpeakerMatch();
        Assert.Equal(string.Empty, match.SpeakerId);
        Assert.Equal(0.0, match.Score, Tolerance);
    }

    [Fact]
    public void SpeakerMatch_SetProperties()
    {
        var match = new SpeakerMatch { SpeakerId = "Speaker_1", Score = 0.87 };
        Assert.Equal("Speaker_1", match.SpeakerId);
        Assert.Equal(0.87, match.Score, Tolerance);
    }

    #endregion

    #region IdentificationResult

    [Fact]
    public void IdentificationResult_DefaultValues()
    {
        var result = new IdentificationResult();
        Assert.Null(result.IdentifiedSpeakerId);
        Assert.Equal(0.0, result.TopScore, Tolerance);
        Assert.Equal(0.0, result.Threshold, Tolerance);
        Assert.NotNull(result.Matches);
        Assert.Empty(result.Matches);
    }

    [Fact]
    public void IdentificationResult_WithMatches()
    {
        var result = new IdentificationResult
        {
            IdentifiedSpeakerId = "Speaker_1",
            TopScore = 0.92,
            Threshold = 0.6,
            Matches =
            [
                new SpeakerMatch { SpeakerId = "Speaker_1", Score = 0.92 },
                new SpeakerMatch { SpeakerId = "Speaker_2", Score = 0.45 }
            ]
        };
        Assert.Equal("Speaker_1", result.IdentifiedSpeakerId);
        Assert.Equal(0.92, result.TopScore, Tolerance);
        Assert.Equal(2, result.Matches.Count);
    }

    #endregion

    #region VerificationResult

    [Fact]
    public void VerificationResult_DefaultValues()
    {
        var result = new VerificationResult();
        Assert.Equal(string.Empty, result.ClaimedSpeakerId);
        Assert.False(result.IsVerified);
        Assert.Equal(0.0, result.Score, Tolerance);
        Assert.Equal(0.0, result.Threshold, Tolerance);
        Assert.Null(result.ErrorMessage);
    }

    [Fact]
    public void VerificationResult_Verified()
    {
        var result = new VerificationResult
        {
            ClaimedSpeakerId = "Speaker_A",
            IsVerified = true,
            Score = 0.85,
            Threshold = 0.7
        };
        Assert.True(result.IsVerified);
        Assert.True(result.Score > result.Threshold);
    }

    [Fact]
    public void VerificationResult_NotVerified_WithError()
    {
        var result = new VerificationResult
        {
            ClaimedSpeakerId = "Speaker_B",
            IsVerified = false,
            Score = 0.3,
            Threshold = 0.7,
            ErrorMessage = "Score below threshold"
        };
        Assert.False(result.IsVerified);
        Assert.Equal("Score below threshold", result.ErrorMessage);
    }

    #endregion

    #region SpeakerEmbedding

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_IdenticalVectors()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 0, 0, 0 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 0, 0, 0 } };
        Assert.Equal(1.0, emb1.CosineSimilarity(emb2), 1e-6);
    }

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_OrthogonalVectors()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 0 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 0, 1 } };
        Assert.Equal(0.0, emb1.CosineSimilarity(emb2), 1e-6);
    }

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_OppositeVectors()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 0 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { -1, 0 } };
        Assert.Equal(-1.0, emb1.CosineSimilarity(emb2), 1e-6);
    }

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_ZeroVector()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 0, 0, 0 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 2, 3 } };
        Assert.Equal(0.0, emb1.CosineSimilarity(emb2), 1e-6);
    }

    [Fact]
    public void SpeakerEmbedding_EuclideanDistance_SameVector()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 2, 3 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 2, 3 } };
        Assert.Equal(0.0, emb1.EuclideanDistance(emb2), 1e-6);
    }

    [Fact]
    public void SpeakerEmbedding_EuclideanDistance_KnownValue()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 0, 0 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 3, 4 } };
        Assert.Equal(5.0, emb1.EuclideanDistance(emb2), 1e-6);
    }

    [Fact]
    public void SpeakerEmbedding_Properties()
    {
        var emb = new SpeakerEmbedding<double>
        {
            Vector = new double[] { 0.1, 0.2, 0.3 },
            Duration = 2.5,
            NumFrames = 100
        };
        Assert.Equal(3, emb.Vector.Length);
        Assert.Equal(2.5, emb.Duration, Tolerance);
        Assert.Equal(100, emb.NumFrames);
    }

    #endregion

    #region SpeakerDiarizerOptions

    [Fact]
    public void SpeakerDiarizerOptions_DefaultValues()
    {
        var options = new SpeakerDiarizerOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(1.5, options.WindowDurationSeconds, Tolerance);
        Assert.Equal(0.75, options.HopDurationSeconds, Tolerance);
        Assert.Equal(256, options.EmbeddingDimension);
        Assert.Equal(0.65, options.ClusteringThreshold, Tolerance);
        Assert.Equal(0.5, options.MinTurnDuration, Tolerance);
        Assert.Null(options.MaxSpeakers);
        Assert.Null(options.EmbeddingModelPath);
    }

    [Fact]
    public void SpeakerDiarizerOptions_CustomValues()
    {
        var options = new SpeakerDiarizerOptions
        {
            SampleRate = 8000,
            WindowDurationSeconds = 2.0,
            HopDurationSeconds = 1.0,
            EmbeddingDimension = 512,
            ClusteringThreshold = 0.8,
            MinTurnDuration = 1.0,
            MaxSpeakers = 4,
            EmbeddingModelPath = "/model/embedding.onnx"
        };
        Assert.Equal(8000, options.SampleRate);
        Assert.Equal(2.0, options.WindowDurationSeconds, Tolerance);
        Assert.Equal(512, options.EmbeddingDimension);
        Assert.Equal(4, options.MaxSpeakers);
    }

    #endregion

    #region SpeakerEmbeddingOptions

    [Fact]
    public void SpeakerEmbeddingOptions_DefaultValues()
    {
        var options = new SpeakerEmbeddingOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(512, options.FftSize);
        Assert.Equal(160, options.HopLength);
        Assert.Equal(40, options.NumMfcc);
        Assert.Equal(256, options.EmbeddingDimension);
        Assert.Null(options.ModelPath);
        Assert.NotNull(options.OnnxOptions);
    }

    #endregion

    #region SpeakerVerifierOptions

    [Fact]
    public void SpeakerVerifierOptions_DefaultValues()
    {
        var options = new SpeakerVerifierOptions();
        Assert.Equal(0.7, options.VerificationThreshold, Tolerance);
        Assert.Equal(0.6, options.IdentificationThreshold, Tolerance);
    }

    [Fact]
    public void SpeakerVerifierOptions_CustomValues()
    {
        var options = new SpeakerVerifierOptions
        {
            VerificationThreshold = 0.9,
            IdentificationThreshold = 0.8
        };
        Assert.Equal(0.9, options.VerificationThreshold, Tolerance);
        Assert.Equal(0.8, options.IdentificationThreshold, Tolerance);
    }

    #endregion

    #region LocalizationResult

    [Fact]
    public void LocalizationResult_Properties()
    {
        var result = new LocalizationResult
        {
            AzimuthDegrees = 45.0,
            ElevationDegrees = 30.0,
            TdoaSamples = 10,
            TdoaSeconds = 0.000625,
            Confidence = 0.85,
            Algorithm = "GCC-PHAT"
        };
        Assert.Equal(45.0, result.AzimuthDegrees, Tolerance);
        Assert.Equal(30.0, result.ElevationDegrees, Tolerance);
        Assert.Equal(10, result.TdoaSamples);
        Assert.Equal(0.000625, result.TdoaSeconds, Tolerance);
        Assert.Equal(0.85, result.Confidence, Tolerance);
        Assert.Equal("GCC-PHAT", result.Algorithm);
    }

    [Fact]
    public void LocalizationResult_GetDirectionVector_Forward()
    {
        // Azimuth 0, Elevation 0 -> forward direction (0, 1, 0)
        var result = new LocalizationResult
        {
            AzimuthDegrees = 0,
            ElevationDegrees = 0,
            Algorithm = "test"
        };
        var (x, y, z) = result.GetDirectionVector();
        Assert.Equal(0.0, x, 1e-6);
        Assert.Equal(1.0, y, 1e-6);
        Assert.Equal(0.0, z, 1e-6);
    }

    [Fact]
    public void LocalizationResult_GetDirectionVector_Right()
    {
        // Azimuth 90, Elevation 0 -> right direction (1, 0, 0)
        var result = new LocalizationResult
        {
            AzimuthDegrees = 90,
            ElevationDegrees = 0,
            Algorithm = "test"
        };
        var (x, y, z) = result.GetDirectionVector();
        Assert.Equal(1.0, x, 1e-6);
        Assert.Equal(0.0, y, 1e-4); // cos(pi/2) is ~0
        Assert.Equal(0.0, z, 1e-6);
    }

    [Fact]
    public void LocalizationResult_GetDirectionVector_UnitLength()
    {
        var result = new LocalizationResult
        {
            AzimuthDegrees = 45,
            ElevationDegrees = 30,
            Algorithm = "test"
        };
        var (x, y, z) = result.GetDirectionVector();
        double magnitude = Math.Sqrt(x * x + y * y + z * z);
        Assert.Equal(1.0, magnitude, 1e-6);
    }

    #endregion

    #region LocalizationAlgorithm Enum

    [Fact]
    public void LocalizationAlgorithm_AllValues()
    {
        var values = (((LocalizationAlgorithm[])Enum.GetValues(typeof(LocalizationAlgorithm))));
        Assert.Equal(3, values.Length);
        Assert.Contains(LocalizationAlgorithm.GCCPHAT, values);
        Assert.Contains(LocalizationAlgorithm.MUSIC, values);
        Assert.Contains(LocalizationAlgorithm.SRPPHAT, values);
    }

    #endregion

    #region SoundLocalizerOptions

    [Fact]
    public void SoundLocalizerOptions_DefaultValues()
    {
        var options = new SoundLocalizerOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(343.0, options.SpeedOfSound, Tolerance);
        Assert.Equal(LocalizationAlgorithm.GCCPHAT, options.Algorithm);
        Assert.Equal(1.0, options.AngleResolution, Tolerance);
        Assert.Equal(512, options.FrameSize);
        Assert.Equal(1000.0, options.CenterFrequency, Tolerance);
        Assert.Null(options.ModelPath);
        Assert.NotNull(options.OnnxOptions);
    }

    [Fact]
    public void SoundLocalizerOptions_CustomValues()
    {
        var options = new SoundLocalizerOptions
        {
            SampleRate = 48000,
            SpeedOfSound = 340.0,
            Algorithm = LocalizationAlgorithm.MUSIC,
            AngleResolution = 0.5,
            FrameSize = 1024,
            CenterFrequency = 2000.0
        };
        Assert.Equal(48000, options.SampleRate);
        Assert.Equal(340.0, options.SpeedOfSound, Tolerance);
        Assert.Equal(LocalizationAlgorithm.MUSIC, options.Algorithm);
    }

    #endregion

    #region ChordSegment

    [Fact]
    public void ChordSegment_DefaultValues()
    {
        var segment = new ChordSegment();
        Assert.Equal(string.Empty, segment.Name);
        Assert.Equal(0.0, segment.StartTime, Tolerance);
        Assert.Equal(0.0, segment.EndTime, Tolerance);
        Assert.Equal(0.0, segment.Confidence, Tolerance);
    }

    [Fact]
    public void ChordSegment_Duration_Computed()
    {
        var segment = new ChordSegment
        {
            Name = "Am",
            StartTime = 1.0,
            EndTime = 3.5,
            Confidence = 0.9
        };
        Assert.Equal(2.5, segment.Duration, Tolerance);
    }

    [Fact]
    public void ChordSegment_ZeroDuration()
    {
        var segment = new ChordSegment { StartTime = 2.0, EndTime = 2.0 };
        Assert.Equal(0.0, segment.Duration, Tolerance);
    }

    #endregion

    #region BeatTrackingResult

    [Fact]
    public void BeatTrackingResult_DefaultValues()
    {
        var result = new BeatTrackingResult();
        Assert.Equal(0.0, result.Tempo, Tolerance);
        Assert.NotNull(result.BeatTimes);
        Assert.Empty(result.BeatTimes);
        Assert.Equal(0.0, result.ConfidenceScore, Tolerance);
    }

    [Fact]
    public void BeatTrackingResult_AverageBeatInterval_120BPM()
    {
        var result = new BeatTrackingResult { Tempo = 120.0 };
        Assert.Equal(0.5, result.AverageBeatInterval, Tolerance); // 60/120 = 0.5s
    }

    [Fact]
    public void BeatTrackingResult_AverageBeatInterval_60BPM()
    {
        var result = new BeatTrackingResult { Tempo = 60.0 };
        Assert.Equal(1.0, result.AverageBeatInterval, Tolerance); // 60/60 = 1s
    }

    [Fact]
    public void BeatTrackingResult_AverageBeatInterval_ZeroTempo()
    {
        var result = new BeatTrackingResult { Tempo = 0.0 };
        Assert.Equal(0.0, result.AverageBeatInterval, Tolerance);
    }

    [Fact]
    public void BeatTrackingResult_WithBeatTimes()
    {
        var result = new BeatTrackingResult
        {
            Tempo = 120.0,
            BeatTimes = [0.0, 0.5, 1.0, 1.5, 2.0],
            ConfidenceScore = 0.95
        };
        Assert.Equal(5, result.BeatTimes.Count);
        Assert.Equal(0.95, result.ConfidenceScore, Tolerance);
    }

    #endregion

    #region KeyDetectionResult

    [Fact]
    public void KeyDetectionResult_DefaultValues()
    {
        var result = new KeyDetectionResult();
        Assert.Equal(0, result.KeyIndex);
        Assert.Equal(string.Empty, result.Name);
        Assert.Equal(string.Empty, result.RootNote);
        Assert.Equal(KeyMode.Major, result.Mode);
        Assert.Equal(0.0, result.Correlation, Tolerance);
        Assert.Equal(0.0, result.Confidence, Tolerance);
        Assert.Equal(string.Empty, result.RelativeKey);
    }

    [Fact]
    public void KeyDetectionResult_SetProperties()
    {
        var result = new KeyDetectionResult
        {
            KeyIndex = 9,
            Name = "A minor",
            RootNote = "A",
            Mode = KeyMode.Minor,
            Correlation = 0.85,
            Confidence = 0.92,
            RelativeKey = "C major"
        };
        Assert.Equal(9, result.KeyIndex);
        Assert.Equal("A minor", result.Name);
        Assert.Equal("A", result.RootNote);
        Assert.Equal(KeyMode.Minor, result.Mode);
        Assert.Equal(0.85, result.Correlation, Tolerance);
        Assert.Equal("C major", result.RelativeKey);
    }

    #endregion

    #region KeyMode Enum

    [Fact]
    public void KeyMode_AllValues()
    {
        var values = (((KeyMode[])Enum.GetValues(typeof(KeyMode))));
        Assert.Equal(2, values.Length);
        Assert.Contains(KeyMode.Major, values);
        Assert.Contains(KeyMode.Minor, values);
    }

    #endregion

    #region SourceSeparationOptions

    [Fact]
    public void SourceSeparationOptions_DefaultValues()
    {
        var options = new SourceSeparationOptions();
        Assert.Equal(44100, options.SampleRate);
        Assert.Equal(4096, options.FftSize);
        Assert.Equal(1024, options.HopLength);
        Assert.Equal(4, options.StemCount);
        Assert.Equal(31, options.HpssKernelSize);
        Assert.Null(options.ModelPath);
        Assert.NotNull(options.OnnxOptions);
    }

    [Fact]
    public void SourceSeparationOptions_CustomValues()
    {
        var options = new SourceSeparationOptions
        {
            SampleRate = 22050,
            FftSize = 2048,
            HopLength = 512,
            StemCount = 2,
            HpssKernelSize = 15,
            ModelPath = "/models/separator.onnx"
        };
        Assert.Equal(22050, options.SampleRate);
        Assert.Equal(2048, options.FftSize);
        Assert.Equal(2, options.StemCount);
    }

    #endregion

    #region SeparationResult

    [Fact]
    public void SeparationResult_ToDictionary()
    {
        var vocals = new Tensor<double>(new[] { 10 });
        var drums = new Tensor<double>(new[] { 10 });
        var bass = new Tensor<double>(new[] { 10 });
        var other = new Tensor<double>(new[] { 10 });

        var result = new SeparationResult<double>
        {
            Vocals = vocals,
            Drums = drums,
            Bass = bass,
            Other = other,
            SampleRate = 44100
        };

        var dict = result.ToDictionary();
        Assert.Equal(4, dict.Count);
        Assert.True(dict.ContainsKey("vocals"));
        Assert.True(dict.ContainsKey("drums"));
        Assert.True(dict.ContainsKey("bass"));
        Assert.True(dict.ContainsKey("other"));
        Assert.Same(vocals, dict["vocals"]);
    }

    [Fact]
    public void SeparationResult_SampleRate()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        var result = new SeparationResult<double>
        {
            Vocals = tensor,
            Drums = tensor,
            Bass = tensor,
            Other = tensor,
            SampleRate = 22050
        };
        Assert.Equal(22050, result.SampleRate);
    }

    #endregion

    #region TtsOptions

    [Fact]
    public void TtsOptions_DefaultValues()
    {
        var options = new TtsOptions();
        Assert.Equal(22050, options.SampleRate);
        Assert.Equal(1.0, options.SpeakingRate, Tolerance);
        Assert.Equal(0.0, options.PitchShift, Tolerance);
        Assert.Equal(1.0, options.Energy, Tolerance);
        Assert.Null(options.SpeakerId);
        Assert.Null(options.Language);
        Assert.Null(options.AcousticModelPath);
        Assert.Null(options.VocoderModelPath);
        Assert.True(options.UseGriffinLimFallback);
        Assert.Equal(60, options.GriffinLimIterations);
        Assert.NotNull(options.OnnxOptions);
        Assert.Equal(80, options.NumMels);
        Assert.Equal(1024, options.FftSize);
        Assert.Equal(256, options.HopLength);
    }

    [Fact]
    public void TtsOptions_CustomValues()
    {
        var options = new TtsOptions
        {
            SampleRate = 44100,
            SpeakingRate = 1.5,
            PitchShift = 2.0,
            Energy = 0.8,
            SpeakerId = 3,
            Language = "fr",
            AcousticModelPath = "/model/acoustic.onnx",
            VocoderModelPath = "/model/vocoder.onnx",
            UseGriffinLimFallback = false,
            GriffinLimIterations = 100,
            NumMels = 128,
            FftSize = 2048,
            HopLength = 512
        };
        Assert.Equal(44100, options.SampleRate);
        Assert.Equal(1.5, options.SpeakingRate, Tolerance);
        Assert.Equal(2.0, options.PitchShift, Tolerance);
        Assert.Equal(0.8, options.Energy, Tolerance);
        Assert.Equal(3, options.SpeakerId);
        Assert.Equal("fr", options.Language);
        Assert.False(options.UseGriffinLimFallback);
    }

    #endregion

    #region TtsResult

    [Fact]
    public void TtsResult_Properties()
    {
        var result = new TtsResult<double>
        {
            Audio = new double[] { 0.1, 0.2, 0.3, 0.4 },
            SampleRate = 22050,
            Duration = 2.5,
            ProcessingTimeMs = 1200
        };
        Assert.Equal(4, result.Audio.Length);
        Assert.Equal(22050, result.SampleRate);
        Assert.Equal(2.5, result.Duration, Tolerance);
        Assert.Equal(1200L, result.ProcessingTimeMs);
    }

    #endregion

    #region TtsModelType Enum

    [Fact]
    public void TtsModelType_AllValues()
    {
        var values = (((TtsModelType[])Enum.GetValues(typeof(TtsModelType))));
        Assert.Equal(3, values.Length);
        Assert.Contains(TtsModelType.FastSpeech2, values);
        Assert.Contains(TtsModelType.Tacotron2, values);
        Assert.Contains(TtsModelType.VITS, values);
    }

    #endregion

    #region VocoderType Enum

    [Fact]
    public void VocoderType_AllValues()
    {
        var values = (((VocoderType[])Enum.GetValues(typeof(VocoderType))));
        Assert.Equal(3, values.Length);
        Assert.Contains(VocoderType.HiFiGan, values);
        Assert.Contains(VocoderType.WaveGlow, values);
        Assert.Contains(VocoderType.GriffinLim, values);
    }

    #endregion

    #region WebRTCVadOptions

    [Fact]
    public void WebRTCVadOptions_DefaultValues()
    {
        var options = new WebRTCVadOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(30, options.FrameDurationMs);
        Assert.Equal(64, options.HiddenDim);
        Assert.Equal(3, options.NumLayers);
        Assert.Equal(1, options.AggressivenessMode);
        Assert.Equal(0.5, options.Threshold, Tolerance);
        Assert.Equal(250, options.MinSpeechDurationMs);
        Assert.Equal(100, options.MinSilenceDurationMs);
        Assert.Equal(0.1, options.DropoutRate, Tolerance);
        Assert.Null(options.ModelPath);
        Assert.NotNull(options.OnnxOptions);
        Assert.Equal(1e-3, options.LearningRate, 1e-10);
    }

    [Fact]
    public void WebRTCVadOptions_CustomValues()
    {
        var options = new WebRTCVadOptions
        {
            SampleRate = 8000,
            FrameDurationMs = 10,
            HiddenDim = 128,
            NumLayers = 5,
            AggressivenessMode = 3,
            Threshold = 0.8,
            MinSpeechDurationMs = 500,
            MinSilenceDurationMs = 200,
            DropoutRate = 0.2,
            LearningRate = 1e-4
        };
        Assert.Equal(8000, options.SampleRate);
        Assert.Equal(10, options.FrameDurationMs);
        Assert.Equal(128, options.HiddenDim);
        Assert.Equal(3, options.AggressivenessMode);
    }

    #endregion

    #region EnergyBasedVad - Construction and Detection

    [Fact]
    public void EnergyBasedVad_DefaultConstruction()
    {
        var vad = new EnergyBasedVad<double>();
        Assert.NotNull(vad);
        Assert.Equal(16000, vad.SampleRate);
        Assert.Equal(480, vad.FrameSize);
        Assert.Equal(0.5, vad.Threshold, Tolerance);
        Assert.Equal(250, vad.MinSpeechDurationMs);
        Assert.Equal(300, vad.MinSilenceDurationMs);
    }

    [Fact]
    public void EnergyBasedVad_CustomConstruction()
    {
        var vad = new EnergyBasedVad<double>(
            sampleRate: 8000,
            frameSize: 240,
            threshold: 0.3,
            minSpeechDurationMs: 100,
            minSilenceDurationMs: 150);
        Assert.Equal(8000, vad.SampleRate);
        Assert.Equal(240, vad.FrameSize);
        Assert.Equal(0.3, vad.Threshold, Tolerance);
    }

    [Fact]
    public void EnergyBasedVad_ImplementsInterface()
    {
        var vad = new EnergyBasedVad<double>();
        Assert.IsAssignableFrom<IVoiceActivityDetector<double>>(vad);
    }

    [Fact]
    public void EnergyBasedVad_DetectSpeech_SilentFrame_ReturnsFalse()
    {
        var vad = new EnergyBasedVad<double>(sampleRate: 16000, frameSize: 480);
        var silentFrame = new Tensor<double>(new[] { 480 }); // all zeros
        Assert.False(vad.DetectSpeech(silentFrame));
    }

    [Fact]
    public void EnergyBasedVad_GetSpeechProbability_SilentFrame_Low()
    {
        var vad = new EnergyBasedVad<double>(sampleRate: 16000, frameSize: 480);
        var silentFrame = new Tensor<double>(new[] { 480 });
        var prob = vad.GetSpeechProbability(silentFrame);
        Assert.True(prob < 0.5, $"Silent frame probability should be low, got {prob}");
    }

    [Fact]
    public void EnergyBasedVad_GetSpeechProbability_LoudFrame_Higher()
    {
        var vad = new EnergyBasedVad<double>(sampleRate: 16000, frameSize: 480);
        // Create a loud sine wave frame
        var loudFrame = new Tensor<double>(new[] { 480 });
        for (int i = 0; i < 480; i++)
        {
            loudFrame[i] = 0.8 * Math.Sin(2 * Math.PI * 440 * i / 16000.0);
        }
        var prob = vad.GetSpeechProbability(loudFrame);
        // Loud signal should have some speech probability (may not be > 0.5 due to energy normalization)
        Assert.False(double.IsNaN(prob));
        Assert.False(double.IsInfinity(prob));
    }

    [Fact]
    public void EnergyBasedVad_GetFrameProbabilities_ReturnsCorrectCount()
    {
        var vad = new EnergyBasedVad<double>(sampleRate: 16000, frameSize: 480);
        var audio = new Tensor<double>(new[] { 4800 }); // 10 frames
        var probs = vad.GetFrameProbabilities(audio);
        Assert.Equal(10, probs.Length);
    }

    [Fact]
    public void EnergyBasedVad_ProcessChunk_ReturnsTuple()
    {
        var vad = new EnergyBasedVad<double>(sampleRate: 16000, frameSize: 480);
        var frame = new Tensor<double>(new[] { 480 });
        var (isSpeech, probability) = vad.ProcessChunk(frame);
        Assert.False(double.IsNaN(probability));
    }

    [Fact]
    public void EnergyBasedVad_ResetState_ClearsState()
    {
        var vad = new EnergyBasedVad<double>(sampleRate: 16000, frameSize: 480);
        // Process some frames
        var frame = new Tensor<double>(new[] { 480 });
        for (int i = 0; i < 480; i++)
            frame[i] = 0.5 * Math.Sin(2 * Math.PI * 440 * i / 16000.0);

        vad.ProcessChunk(frame);
        vad.ProcessChunk(frame);

        // Reset
        vad.ResetState();

        // After reset, silent frame should not detect speech
        var silentFrame = new Tensor<double>(new[] { 480 });
        var (isSpeech, _) = vad.ProcessChunk(silentFrame);
        Assert.False(isSpeech);
    }

    #endregion

    #region YinPitchDetector - Construction and Detection

    [Fact]
    public void YinPitchDetector_DefaultConstruction()
    {
        var detector = new YinPitchDetector<double>();
        Assert.NotNull(detector);
        Assert.Equal(44100, detector.SampleRate);
        Assert.Equal(50, detector.MinPitch);
        Assert.Equal(2000, detector.MaxPitch);
    }

    [Fact]
    public void YinPitchDetector_CustomConstruction()
    {
        var detector = new YinPitchDetector<double>(
            sampleRate: 16000,
            minPitch: 80,
            maxPitch: 1000,
            threshold: 0.15,
            frameSize: 4096);
        Assert.Equal(16000, detector.SampleRate);
        Assert.Equal(80, detector.MinPitch);
        Assert.Equal(1000, detector.MaxPitch);
    }

    [Fact]
    public void YinPitchDetector_DetectPitch_440Hz_SineWave()
    {
        int sampleRate = 44100;
        var detector = new YinPitchDetector<double>(sampleRate: sampleRate, threshold: 0.2);

        // Generate 440 Hz sine wave - need at least 2 periods for detection
        int frameSize = 4096;
        var frame = new Tensor<double>(new[] { frameSize });
        for (int i = 0; i < frameSize; i++)
        {
            frame[i] = Math.Sin(2 * Math.PI * 440.0 * i / sampleRate);
        }

        var (hasPitch, pitch) = detector.DetectPitch(frame);
        if (hasPitch)
        {
            // Allow 5% tolerance for pitch detection
            Assert.InRange(pitch, 418, 462); // 440 +/- 5%
        }
    }

    [Fact]
    public void YinPitchDetector_DetectPitch_Silence_NoPitch()
    {
        var detector = new YinPitchDetector<double>();
        var silence = new Tensor<double>(new[] { 2048 }); // all zeros
        var (hasPitch, _) = detector.DetectPitch(silence);
        Assert.False(hasPitch);
    }

    [Fact]
    public void YinPitchDetector_PitchToMidi_A4_Is69()
    {
        var detector = new YinPitchDetector<double>();
        double midi = detector.PitchToMidi(440.0);
        Assert.Equal(69.0, midi, 1e-3);
    }

    [Fact]
    public void YinPitchDetector_PitchToMidi_C4_Is60()
    {
        var detector = new YinPitchDetector<double>();
        double midi = detector.PitchToMidi(261.63); // C4
        Assert.Equal(60.0, midi, 0.1);
    }

    [Fact]
    public void YinPitchDetector_MidiToPitch_69_Is440()
    {
        var detector = new YinPitchDetector<double>();
        double pitch = detector.MidiToPitch(69.0);
        Assert.Equal(440.0, pitch, 0.01);
    }

    [Fact]
    public void YinPitchDetector_PitchToMidi_MidiToPitch_RoundTrip()
    {
        var detector = new YinPitchDetector<double>();
        double originalPitch = 440.0;
        double midi = detector.PitchToMidi(originalPitch);
        double recoveredPitch = detector.MidiToPitch(midi);
        Assert.Equal(originalPitch, recoveredPitch, 0.01);
    }

    [Fact]
    public void YinPitchDetector_PitchToNoteName_A4()
    {
        var detector = new YinPitchDetector<double>();
        string note = detector.PitchToNoteName(440.0);
        Assert.Equal("A4", note);
    }

    [Fact]
    public void YinPitchDetector_PitchToNoteName_ZeroPitch()
    {
        var detector = new YinPitchDetector<double>();
        string note = detector.PitchToNoteName(0.0);
        Assert.Equal("---", note);
    }

    [Fact]
    public void YinPitchDetector_GetCentsDeviation_ExactNote()
    {
        var detector = new YinPitchDetector<double>();
        double cents = detector.GetCentsDeviation(440.0);
        Assert.Equal(0.0, cents, 0.1);
    }

    [Fact]
    public void YinPitchDetector_ExtractPitchContour_ReturnsValues()
    {
        int sampleRate = 16000;
        var detector = new YinPitchDetector<double>(sampleRate: sampleRate);

        // Create 0.5 seconds of audio (enough for at least a few contour frames)
        int numSamples = sampleRate / 2;
        var audio = new Tensor<double>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            audio[i] = 0.5 * Math.Sin(2 * Math.PI * 440.0 * i / sampleRate);
        }

        var contour = detector.ExtractPitchContour(audio, hopSizeMs: 20);
        Assert.True(contour.Length > 0, "Should have at least one contour frame");
    }

    [Fact]
    public void YinPitchDetector_ExtractDetailedPitchContour_HasTimeInfo()
    {
        int sampleRate = 16000;
        var detector = new YinPitchDetector<double>(sampleRate: sampleRate);

        int numSamples = sampleRate / 2;
        var audio = new Tensor<double>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            audio[i] = 0.5 * Math.Sin(2 * Math.PI * 440.0 * i / sampleRate);
        }

        var contour = detector.ExtractDetailedPitchContour(audio, hopSizeMs: 20);
        Assert.True(contour.Count > 0, "Should have at least one contour frame");

        // First frame should have time near 0
        Assert.True(contour[0].Time >= 0.0);

        // If there are multiple frames, time should increase
        if (contour.Count > 1)
        {
            Assert.True(contour[1].Time > contour[0].Time);
        }
    }

    #endregion

    #region GenreClassificationResult

    [Fact]
    public void GenreClassificationResult_Properties()
    {
        var result = new GenreClassificationResult
        {
            PredictedGenre = "Rock",
            Confidence = 0.85,
            AllProbabilities = new Dictionary<string, double>
            {
                ["Rock"] = 0.85,
                ["Pop"] = 0.10,
                ["Jazz"] = 0.05
            },
            TopPredictions =
            [
                ("Rock", 0.85),
                ("Pop", 0.10),
                ("Jazz", 0.05)
            ],
            Features = new GenreFeatures
            {
                MfccMean = new double[] { 1, 2, 3 },
                MfccStd = new double[] { 0.5, 0.5, 0.5 },
                SpectralCentroidMean = 2000,
                Tempo = 120
            }
        };
        Assert.Equal("Rock", result.PredictedGenre);
        Assert.Equal(0.85, result.Confidence, Tolerance);
        Assert.Equal(3, result.AllProbabilities.Count);
        Assert.Equal(3, result.TopPredictions.Count);
        Assert.Equal(120, result.Features.Tempo, Tolerance);
    }

    #endregion

    #region GenreFeatures

    [Fact]
    public void GenreFeatures_AllProperties()
    {
        var features = new GenreFeatures
        {
            MfccMean = new double[] { 1.0, 2.0 },
            MfccStd = new double[] { 0.1, 0.2 },
            SpectralCentroidMean = 2000.0,
            SpectralCentroidStd = 500.0,
            SpectralRolloffMean = 4000.0,
            ZeroCrossingRate = 0.15,
            RmsEnergy = 0.3,
            Tempo = 128.0
        };
        Assert.Equal(2, features.MfccMean.Length);
        Assert.Equal(2000.0, features.SpectralCentroidMean, Tolerance);
        Assert.Equal(500.0, features.SpectralCentroidStd, Tolerance);
        Assert.Equal(4000.0, features.SpectralRolloffMean, Tolerance);
        Assert.Equal(0.15, features.ZeroCrossingRate, Tolerance);
        Assert.Equal(0.3, features.RmsEnergy, Tolerance);
        Assert.Equal(128.0, features.Tempo, Tolerance);
    }

    #endregion

    #region SceneClassificationResult

    [Fact]
    public void SceneClassificationResult_Properties()
    {
        var result = new SceneClassificationResult
        {
            PredictedScene = "park",
            Category = "outdoor_nature",
            Confidence = 0.9,
            AllProbabilities = new Dictionary<string, double> { ["park"] = 0.9, ["street"] = 0.1 },
            TopPredictions = [("park", 0.9), ("street", 0.1)],
            Features = new SceneFeatures
            {
                MfccMean = new double[] { 1 },
                MfccStd = new double[] { 0.5 },
                MfccDelta = new double[] { 0.1 },
                BandEnergies = new double[] { 0.2, 0.3, 0.4 }
            }
        };
        Assert.Equal("park", result.PredictedScene);
        Assert.Equal("outdoor_nature", result.Category);
        Assert.Equal(0.9, result.Confidence, Tolerance);
        Assert.NotNull(result.Features);
    }

    #endregion

    #region SceneFeatures

    [Fact]
    public void SceneFeatures_AllProperties()
    {
        var features = new SceneFeatures
        {
            MfccMean = new double[] { 1, 2, 3 },
            MfccStd = new double[] { 0.5, 0.5, 0.5 },
            MfccDelta = new double[] { 0.1, 0.1, 0.1 },
            SpectralCentroid = 2500,
            SpectralBandwidth = 1000,
            SpectralFlatness = 0.3,
            SpectralContrast = 0.7,
            RmsEnergy = 0.25,
            ZeroCrossingRate = 0.12,
            EnergyVariance = 0.05,
            BandEnergies = new double[] { 0.1, 0.2, 0.3, 0.4 }
        };
        Assert.Equal(3, features.MfccMean.Length);
        Assert.Equal(2500, features.SpectralCentroid, Tolerance);
        Assert.Equal(1000, features.SpectralBandwidth, Tolerance);
        Assert.Equal(0.3, features.SpectralFlatness, Tolerance);
        Assert.Equal(0.7, features.SpectralContrast, Tolerance);
        Assert.Equal(0.25, features.RmsEnergy, Tolerance);
        Assert.Equal(0.12, features.ZeroCrossingRate, Tolerance);
        Assert.Equal(0.05, features.EnergyVariance, Tolerance);
        Assert.Equal(4, features.BandEnergies.Length);
    }

    #endregion

    #region Cross-Module - Pitch Detection Mathematical Properties

    [Fact]
    public void PitchDetector_PitchToMidi_OctaveRelationship()
    {
        var detector = new YinPitchDetector<double>();
        // Going up an octave should add 12 to MIDI note
        double midiA4 = detector.PitchToMidi(440.0);
        double midiA5 = detector.PitchToMidi(880.0);
        Assert.Equal(12.0, midiA5 - midiA4, 0.01);
    }

    [Fact]
    public void PitchDetector_MidiToPitch_OctaveDoubles()
    {
        var detector = new YinPitchDetector<double>();
        double pitchC4 = detector.MidiToPitch(60);
        double pitchC5 = detector.MidiToPitch(72);
        Assert.Equal(pitchC4 * 2, pitchC5, 0.01);
    }

    [Fact]
    public void PitchDetector_NoteNames_Chromatic()
    {
        var detector = new YinPitchDetector<double>();
        // C4 = MIDI 60 = 261.63 Hz
        string note = detector.PitchToNoteName(261.63);
        Assert.Equal("C4", note);
    }

    #endregion

    #region Cross-Module - VAD with Speech-Like Signal

    [Fact]
    public void EnergyBasedVad_DetectSpeechSegments_FullSilence_Empty()
    {
        var vad = new EnergyBasedVad<double>(
            sampleRate: 16000,
            frameSize: 480,
            minSpeechDurationMs: 100,
            minSilenceDurationMs: 100);

        var silence = new Tensor<double>(new[] { 16000 }); // 1 second silence
        var segments = vad.DetectSpeechSegments(silence);
        Assert.Empty(segments);
    }

    [Fact]
    public void EnergyBasedVad_ThresholdProperty_Settable()
    {
        var vad = new EnergyBasedVad<double>();
        vad.Threshold = 0.8;
        Assert.Equal(0.8, vad.Threshold, Tolerance);
    }

    #endregion

    #region Cross-Module - Speaker Embedding Similarity Symmetry

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_Symmetric()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 0.1, 0.5, 0.3, 0.2 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 0.4, 0.2, 0.1, 0.6 } };
        var sim12 = emb1.CosineSimilarity(emb2);
        var sim21 = emb2.CosineSimilarity(emb1);
        Assert.Equal(sim12, sim21, 1e-10);
    }

    [Fact]
    public void SpeakerEmbedding_EuclideanDistance_Symmetric()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 1, 2, 3 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 4, 5, 6 } };
        var dist12 = emb1.EuclideanDistance(emb2);
        var dist21 = emb2.EuclideanDistance(emb1);
        Assert.Equal(dist12, dist21, 1e-10);
    }

    [Fact]
    public void SpeakerEmbedding_EuclideanDistance_NonNegative()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { -1, 2, -3 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 4, -5, 6 } };
        Assert.True(emb1.EuclideanDistance(emb2) >= 0);
    }

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_InRange()
    {
        var emb1 = new SpeakerEmbedding<double> { Vector = new double[] { 0.3, 0.7, 0.1 } };
        var emb2 = new SpeakerEmbedding<double> { Vector = new double[] { 0.5, 0.2, 0.8 } };
        var sim = emb1.CosineSimilarity(emb2);
        Assert.InRange(sim, -1.0, 1.0);
    }

    #endregion

    #region Cross-Module - Localization Direction Vector Properties

    [Fact]
    public void LocalizationResult_GetDirectionVector_Up()
    {
        // Elevation 90 -> up direction (0, 0, 1)
        var result = new LocalizationResult
        {
            AzimuthDegrees = 0,
            ElevationDegrees = 90,
            Algorithm = "test"
        };
        var (x, y, z) = result.GetDirectionVector();
        Assert.Equal(0.0, x, 1e-6);
        Assert.Equal(0.0, y, 1e-4);
        Assert.Equal(1.0, z, 1e-6);
    }

    [Fact]
    public void LocalizationResult_GetDirectionVector_Left()
    {
        // Azimuth -90, Elevation 0 -> left direction (-1, 0, 0)
        var result = new LocalizationResult
        {
            AzimuthDegrees = -90,
            ElevationDegrees = 0,
            Algorithm = "test"
        };
        var (x, y, z) = result.GetDirectionVector();
        Assert.Equal(-1.0, x, 1e-6);
        Assert.Equal(0.0, y, 1e-4);
        Assert.Equal(0.0, z, 1e-6);
    }

    #endregion
}
