using AiDotNet.Audio.Emotion;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// <c>RecognizeEmotionTimeSeries</c> must score the AUDIO of each window.
/// </summary>
/// <remarks>
/// The window loop built a tensor and then assigned the samples into <c>windowTensor.ToVector()</c>,
/// which allocates and copies, so the write landed in a throwaway vector and every window handed to
/// the model was all zeros (the same defect <see cref="SileroVadFramePayloadTests"/> pins for the VAD).
/// Reference: each timed result must equal <c>RecognizeEmotion</c> on that window's samples.
/// </remarks>
public class SpeechEmotionRecognizerTimeSeriesPayloadTests
{
    // 0.56 s at 16 kHz: (8960 - 1024) / 256 + 1 = 32 frames, matching the 64 x 32 conv geometry.
    private const int WindowSamples = 8960;
    private const int WindowMs = 560;

    private static SpeechEmotionRecognizer<double> Create() =>
        new SpeechEmotionRecognizer<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4),
            options: new SpeechEmotionRecognizerOptions
            {
                SampleRate = 16000, NumMels = 64, NFft = 1024, HopLength = 256,
                InputDurationSeconds = 0.56, NumConvBlocks = 2, BaseFilters = 8, HiddenDim = 32,
                DropoutRate = 0.0,
                EmotionLabels = new[] { "neutral", "happy", "sad", "angry" },
                IncludeArousalValence = false
            });

    private static Tensor<double> Signal(int length, double amplitude, int seed)
    {
        var t = new Tensor<double>([length]);
        var span = t.Data.Span;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < span.Length; i++)
            span[i] = amplitude * ((rng.NextDouble() * 2.0) - 1.0);
        return t;
    }

    [Fact]
    public void TimeSeries_ScoresEachWindowsSamples()
    {
        var model = Create();
        var first = Signal(WindowSamples, 0.9, 11);
        var second = Signal(WindowSamples, 0.05, 23);
        var audio = new Tensor<double>([2 * WindowSamples]);
        first.Data.Span.CopyTo(audio.Data.Span.Slice(0, WindowSamples));
        second.Data.Span.CopyTo(audio.Data.Span.Slice(WindowSamples, WindowSamples));

        var timed = model.RecognizeEmotionTimeSeries(audio, WindowMs, WindowMs);

        Assert.Equal(2, timed.Count);
        var expectedFirst = model.RecognizeEmotion(first);
        var expectedSecond = model.RecognizeEmotion(second);
        Assert.Equal(expectedFirst.Emotion, timed[0].Emotion);
        Assert.Equal(expectedFirst.Confidence, timed[0].Confidence, 10);
        Assert.Equal(expectedSecond.Emotion, timed[1].Emotion);
        Assert.Equal(expectedSecond.Confidence, timed[1].Confidence, 10);
    }
}
