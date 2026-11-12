using System.Reflection;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class CosyVoiceTokenizerOptionFixture
{
    internal static long Observe<T>(CosyVoice<T> model)
    {
        var tokenizer = (S3Tokenizer<T>)typeof(CosyVoiceBase<T>)
            .GetProperty("SpeechTokenizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        // Observe the frozen encoder before discrete code selection can hide a head change.
        var mel = new Tensor<T>(new[] { 128, 8 });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < mel.Length; i++) mel[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.17));
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(tokenizer.Encode(mel).Select(ops.ToDouble));
    }
}
