using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tests.TextToSpeech;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// The model-family invariants for SpeechGPT's speech recognizer (<c>SpeechRecognition.LLMIntegrated.SpeechGPT</c>, the
/// TTS model's task variant) on the tiny fixture. It replaces the generated suite, whose probes do not fit this model:
/// <list type="bullet">
/// <item>mHuBERT normalizes its first convolution with group norm, so a recording scaled 10x gives the same features and
/// units by design; the scaled-input invariant does not apply.</item>
/// <item>Speech reaches the language model only as discrete units, so two short probes that land on the same units give
/// the same transcript. The probes here are a second of 16 kHz audio, and the distinct-input probes differ in pitch
/// rather than loudness.</item>
/// </list>
/// </summary>
public class SpeechGptRecognizerModelTests : AudioNNModelTestBase<double>
{
    protected override int[] InputShape => new[] { 16000 };

    protected override int[] OutputShape => new[] { 8 };

    protected override bool ScaledInputInvariantApplicable => false;

    protected override int MoreDataShortIterations => 2;

    protected override int MoreDataLongIterations => 4;

    protected override INeuralNetworkModel<double> CreateNetwork() => SpeechGptFixture.Asr<double>();

    /// <summary>A tone whose pitch, not loudness, follows <paramref name="value"/>; zero stays silence.</summary>
    protected override Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        if (value == 0.0) return new Tensor<double>(shape);
        return SpeechGptFixture.Tones<double>(shape[0] / 16000.0, 150 + 400 * value, 900 - 300 * value);
    }
}
