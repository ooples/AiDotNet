using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;

namespace AiDotNet.Tests.TextToSpeech;

internal static class DiTToOptionFixture
{
    internal static long ObserveHeads<T>(DiTToTTS<T> model)
    {
        var options = (DiTToTTSOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var noisy = new Tensor<T>(new[] { 4, options.LatentDim });
        var text = new Tensor<T>(new[] { 3, options.TextHiddenDim });
        var mask = new Tensor<T>(new[] { 4, 1 });
        for (int i = 0; i < noisy.Length; i++) noisy[i] = ops.FromDouble(Math.Sin(i * .17));
        for (int i = 0; i < text.Length; i++) text[i] = ops.FromDouble(Math.Cos(i * .13));
        for (int i = 0; i < mask.Length; i++) mask[i] = ops.One;
        var output = model.DiffusionTransformer!.Forward(noisy, mask, text, .37, dropText: false);
        return TtsPaperNumerics.Fingerprint(Enumerable.Range(0, output.Length).Select(i => ops.ToDouble(output[i])));
    }

    internal static long ObserveTextPadding<T>(DiTToTTS<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var previous = model.Voice;
        try
        {
            model.Voice = new TtsVoice<T>
            {
                Reference = new Tensor<T>(new[] { 2, ((DiTToTTSOptions)model.GetOptions()).MelChannels }),
                ReferenceTokens = new Tensor<T>(new[] { 2 }, new Vector<T>(new[] { ops.FromDouble(7), ops.FromDouble(1) }))
            };
            return TtsPaperNumerics.Fingerprint(model.TextWithPrompt("hi").Select(value => (double)value));
        }
        finally { model.Voice = previous; }
    }
}
