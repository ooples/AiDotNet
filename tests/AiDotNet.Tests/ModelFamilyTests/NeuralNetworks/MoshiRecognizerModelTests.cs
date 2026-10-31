using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tests.TextToSpeech;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// The model-family invariants for Moshi's streaming speech recognizer (<c>SpeechRecognition.Streaming.Moshi</c>, the TTS
/// model's task variant) on the tiny fixture. The generated suite belongs to the TTS model of the same name. Speech reaches
/// the language model only as Mimi's codes, so the distinct-input probes differ in pitch rather than loudness.
/// </summary>
public class MoshiRecognizerModelTests : AudioNNModelTestBase<double>
{
    protected override int[] InputShape => new[] { 800 };

    protected override int[] OutputShape => new[] { 8 };

    protected override int MoreDataShortIterations => 2;

    protected override int MoreDataLongIterations => 4;

    protected override INeuralNetworkModel<double> CreateNetwork() => MoshiFixture.Asr<double>();

    /// <summary>A tone whose pitch, not loudness, follows <paramref name="value"/>; zero stays silence.</summary>
    protected override Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        if (value == 0.0) return new Tensor<double>(shape);
        return MoshiFixture.Tones<double>(shape[0] / (double)MoshiFixture.SampleRate, 60 + 200 * value, 500 - 150 * value);
    }
}
