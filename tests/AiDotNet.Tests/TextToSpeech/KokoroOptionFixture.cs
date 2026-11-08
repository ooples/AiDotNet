using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.TextToSpeech;

internal static class KokoroOptionFixture
{
    internal static void PrepareOptions(KokoroOptions options, string option)
    {
        if (option == "MaxTextLength") options.MaxTextLength = 4;
    }

    internal static Tensor<T> Tokens<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(new[] { 4 }, new Vector<T>(new[] { 12, 24, 36, 48 }.Select(value => ops.FromDouble(value))));
    }

    internal static long Observe<T>(Kokoro<T> model, string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var options = (KokoroOptions)model.GetOptions();
        long output;
        try
        {
            var audio = option == "UseNativeEnglishFallback" ? model.Synthesize("hi")
                : option == "DropoutRate" ? model.ForwardForTraining(Tokens<T>()) : model.Predict(Tokens<T>());
            output = TtsPaperNumerics.Fingerprint(Enumerable.Range(0, audio.Length).Select(i => ops.ToDouble(audio[i])));
        }
        catch (ArgumentException) when (option == "MaxTextLength" && options.MaxTextLength < 4) { output = -1; }
        catch (InvalidOperationException) when (option == "UseNativeEnglishFallback" && !options.UseNativeEnglishFallback && model.TextPhonemizer is null) { output = -2; }
        var hashes = new List<long> { output };
        foreach (var chunk in model.GetParameterChunks())
            hashes.Add(TtsPaperNumerics.Fingerprint(Enumerable.Range(0, chunk.Length).Select(i => ops.ToDouble(chunk[i]))));
        return ModelFamilyTests.Base.OptionSensitivityHarness.Combine(hashes.ToArray());
    }
}
