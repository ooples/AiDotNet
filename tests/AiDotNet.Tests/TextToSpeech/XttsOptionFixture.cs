using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

internal static class XttsOptionFixture
{
    internal static bool IsSamplerOption(string option) => option is "TopP" or "Temperature" or "RepetitionPenalty";

    internal static void PrepareOptions(XTTSv2Options options, string option)
    {
        if (!IsSamplerOption(option)) return;
        options.TopK = 0;
        options.TopP = 1;
        options.MaxCodecFrames = 16;
    }

    internal static long ObserveSampler<T>(XTTSv2<T> model) => Inference(model, () => Sampler(model));

    private static long Sampler<T>(XTTSv2<T> model)
    {
        var options = (XTTSv2Options)model.GetOptions();
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var language = (XttsLanguageModel<T>)typeof(XTTSv2<T>).GetField("_languageModel", flags)!.GetValue(model)!;
        var condition = new Tensor<T>(new[] { options.PerceiverLatents, options.LLMDim });
        var values = new List<double>();
        for (int seed = 0; seed < 16; seed++)
        {
            var generated = language.Generate(new[] { 4, 5 }, condition, new Random(options.SamplingSeed + seed));
            values.Add(generated.Codes.Length);
            values.AddRange(generated.Codes.Select(code => (double)code));
        }
        return TtsPaperNumerics.Fingerprint(values);
    }

    internal static long ObserveCodec<T>(XTTSv2<T> model) => Inference(model, () => Codec(model));

    private static long Codec<T>(XTTSv2<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var audio = new Tensor<T>(new[] { 4096 });
        for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(.3 * Math.Sin(.07 * i) + .1 * Math.Cos(.23 * i));
        return TtsPaperNumerics.Fingerprint(model.EncodeToTokens(audio).Select(ops.ToDouble));
    }

    private static long Inference<T>(XTTSv2<T> model, Func<long> observe)
    {
        bool training = model.IsTrainingMode;
        model.SetTrainingMode(false);
        try { return observe(); }
        finally { model.SetTrainingMode(training); }
    }
}
