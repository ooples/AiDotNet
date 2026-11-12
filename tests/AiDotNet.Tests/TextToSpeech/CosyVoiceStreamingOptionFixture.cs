using System.Reflection;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class CosyVoiceStreamingOptionFixture
{
    internal static long Observe<T>(CosyVoice3<T> model)
    {
        var options = (CosyVoice3Options)model.GetOptions();
        var flow = (CosyVoice3Flow<T>)typeof(CosyVoice3<T>)
            .GetField("_flow", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int[] tokens = { 1, 2, 3, 4 };
        var speaker = new Tensor<T>(new[] { options.SpeakerEmbeddingDim });
        for (int i = 0; i < speaker.Length; i++) speaker[i] = ops.FromDouble(0.1 + i * 0.02);
        var noise = new Tensor<T>(new[] { tokens.Length * options.TokenMelRatio, options.MelChannels });
        for (int i = 0; i < noise.Length; i++) noise[i] = ops.FromDouble(0.2 * Math.Sin(i * 0.17));
        // StaticChunkSize controls streaming flow attention; offline waveform inference never enables it.
        // Exercise the real causal DiT/CFG solver with speech tokens and fixed noise, without LM sampling.
        var mel = flow.Inference(tokens, Array.Empty<int>(), null, speaker, noise, streaming: true);
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(mel.ToArray().Select(value => ops.ToDouble(value)));
    }
}
