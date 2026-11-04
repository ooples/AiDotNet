using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class E3TTSOptionsTests
{
    [Fact]
    public void CopyPreservesInheritedTrainingAndSynthesisSettings()
    {
        var original = new E3TTSOptions
        {
            ConditionDim = 48, TimeEmbeddingDim = 32, NumDiffusionSteps = 37, NumFlowSteps = 7,
            InterChannels = 40, FilterChannels = 160, EncoderDim = 80, DecoderDim = 96,
            SampleRate = 24000, HiddenDim = 64, NumEncoderLayers = 3, NumDecoderLayers = 4,
            NumHeads = 2, VocabSize = 1234, MaxTextLength = 127, MaxMelLength = 513,
            MelChannels = 64, HopSize = 128, FftSize = 512, DropoutRate = 0.25,
            LearningRate = 0.003, WeightDecay = 0.02, ModelPath = "custom-e3tts.onnx",
        };

        var copy = new E3TTSOptions(original);

        Assert.Equal(Settings(original), Settings(copy));
        Assert.NotSame(original.OnnxOptions, copy.OnnxOptions);
    }

    private static object?[] Settings(E3TTSOptions options) =>
    [
        options.ConditionDim, options.TimeEmbeddingDim, options.NumDiffusionSteps, options.NumFlowSteps,
        options.InterChannels, options.FilterChannels, options.EncoderDim, options.DecoderDim,
        options.SampleRate, options.HiddenDim, options.NumEncoderLayers, options.NumDecoderLayers,
        options.NumHeads, options.VocabSize, options.MaxTextLength, options.MaxMelLength,
        options.MelChannels, options.HopSize, options.FftSize, options.DropoutRate,
        options.LearningRate, options.WeightDecay, options.ModelPath,
    ];
}
