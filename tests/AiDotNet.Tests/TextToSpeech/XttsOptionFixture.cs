using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

internal static class XttsOptionFixture
{
    internal static long ObserveCodec<T>(XTTSv2<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var audio = new Tensor<T>(new[] { 4096 });
        for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(.3 * Math.Sin(.07 * i) + .1 * Math.Cos(.23 * i));
        return TtsPaperNumerics.Fingerprint(model.EncodeToTokens(audio).Select(ops.ToDouble));
    }
}
