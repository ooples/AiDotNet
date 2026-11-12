using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The generated suite's Mega-TTS 2: the parity test's tiny configuration (width 16, two layers or blocks everywhere, 12
/// codes, pooling by 4) with 16 mel bins and a discriminator window of 8 frames; the Transformer schedule at 1e-3 from the
/// first step, which the paper's 4000 warm-up steps would hold near zero for the suite's few steps. The voice is a full
/// prompt: a recording with its phonemes and their durations.
/// </summary>
internal static class MegaTts2Fixture
{
    public static MegaTTS2<T> Create<T>()
    {
        var options = MegaTts2ModelReferenceTests.TinyOptions();
        options.MelChannels = 16;
        options.DiscriminatorWindows = new[] { 8 };
        options.WarmupSteps = 1;
        // Noam's peak, d^-1/2 · warmup^-1/2, is 0.25 at width 16 and one warm-up step; 0.004 of it is 1e-3.
        options.ScheduleFactor = 0.004;
        var model = new MegaTTS2<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options);
        // The prompt: a quarter second of two tones with the phonemes of "Hello world." spread over its frames.
        var ops = MathHelper.GetNumericOperations<T>();
        var tokens = model.TextToTokens("Hello world.");
        int samples = options.SampleRate / 4, frames = 1 + samples / options.HopSize;
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180.0 * i / options.SampleRate) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / options.SampleRate));
        var durations = Enumerable.Range(0, tokens.Length).Select(i => frames / tokens.Length + (i < frames % tokens.Length ? 1 : 0)).ToArray();
        model.Voice = new TtsVoice<T> { ReferenceAudio = new Tensor<T>(new[] { samples }, new Vector<T>(audio)), ReferenceTokens = tokens, ReferenceDurations = durations };
        return model;
    }
}
