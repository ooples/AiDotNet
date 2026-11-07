using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class GptSoVitsPromptStateTests
{
    [Theory]
    [InlineData(320)]
    [InlineData(640)]
    public void PairedGanUpsamplingControlsActualDecodedWaveformLength(int hop)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = GptSoVitsVersion.V2;
        options.HopSize = hop;
        options.SegmentSize = 4 * hop;
        options.UpsampleRates = hop == 320 ? new[] { 8, 5, 4, 2 } : new[] { 10, 8, 2, 2, 2 };
        options.UpsampleKernels = options.UpsampleRates.Select(rate => 2 * rate + rate % 2).ToArray();
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        var reference = new Tensor<float>(new[] { 1, options.FftSize / 2 + 1, 8 });
        for (int i = 0; i < reference.Length; i++) reference[i] = (float)(0.2 * Math.Sin(i * 0.13));
        var audio = model.SoVits.Decode(new[] { 1, 2, 3 }, new[] { 1, 2, 3 }, new[] { reference }, null,
            0, 1, shape => new Tensor<float>(shape));
        Assert.Equal(6 * hop, audio.Length);
        Assert.All(audio.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    public void OddVocoderResidualKernelsRunNativeInference(int kernel)
    {
        var options = GptSoVitsFixture.Options();
        options.VocoderResblockKernels = new[] { kernel };
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        GptSoVitsFixture.Prepare(model);
        model.SetTrainingMode(false);
        var audio = model.Predict(new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 2f, 3f })));
        Assert.NotEmpty(audio.ToArray());
        Assert.All(audio.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(GptSoVitsVersion.V3)]
    [InlineData(GptSoVitsVersion.V5Dev)]
    public void ClonePreservesTranscriptAndOriginalReferenceSampleRate(GptSoVitsVersion version)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = version;
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        var recording = new Tensor<float>(new[] { 6400 });
        for (int i = 0; i < recording.Length; i++) recording[i] = (float)(0.3 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.037));
        model.Voice = model.CreateVoice(recording, 16000, "hello world");
        var input = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 2f, 3f }));
        model.SetTrainingMode(false);
        var expected = model.Predict(input);
        using var clone = (INeuralNetworkModel<float>)model.Clone();
        clone.SetTrainingMode(false);
        var actual = clone.Predict(input);
        Assert.NotEmpty(expected.ToArray());
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5 * (1 + Math.Abs(expected[i])));
        using var restored = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        restored.Deserialize(model.Serialize());
        restored.SetTrainingMode(false);
        var restoredOutput = restored.Predict(input);
        Assert.Equal(expected.Length, restoredOutput.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - restoredOutput[i]) <= 1e-5 * (1 + Math.Abs(expected[i])));
    }

    [Theory]
    [InlineData(GptSoVitsVersion.V3)]
    [InlineData(GptSoVitsVersion.V2)]
    public void EvenResidualKernelsAreRejectedBeforeInference(GptSoVitsVersion version)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = version;
        options.ResblockKernels = options.VocoderResblockKernels = new[] { 4 };
        Assert.Throws<ArgumentException>(() => new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options));
    }
}
