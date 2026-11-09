using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.TextToSpeech;

internal static class KokoroOptionFixture
{
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, List<KeyValuePair<string, long>>> Observations = new();
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
        var parts = Measure(model, option);
        // Keep conditioning/state diagnostics separate from the observable fingerprint: merely
        // storing an option must never make an ignored option pass the sensitivity gate.
        var diagnostics = new List<KeyValuePair<string, long>>(parts);
        var options = (KokoroOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        diagnostics.Add(new("configured synthesis seed", options.SynthesisSeed ?? -1));
        diagnostics.Add(new("configured speed", BitConverter.DoubleToInt64Bits(options.Speed)));
        if (model.Voice?.Reference is { } style)
            diagnostics.Add(new("voice style", TtsPaperNumerics.Fingerprint(style.ToArray().Select(value => ops.ToDouble(value)))));
        int batchNormIndex = 0;
        foreach (var layer in model.Layers.OfType<AiDotNet.NeuralNetworks.Layers.BatchNormalizationLayer<T>>())
        {
            diagnostics.Add(new($"batch norm {batchNormIndex} running mean",
                TtsPaperNumerics.Fingerprint(layer.GetRunningMean().ToArray().Select(value => ops.ToDouble(value)))));
            diagnostics.Add(new($"batch norm {batchNormIndex++} running variance",
                TtsPaperNumerics.Fingerprint(layer.GetRunningVariance().ToArray().Select(value => ops.ToDouble(value)))));
        }
        Observations.Remove(model);
        Observations.Add(model, diagnostics);
        return ModelFamilyTests.Base.OptionSensitivityHarness.Combine(parts.Select(part => part.Value).ToArray());
    }

    internal static IReadOnlyList<KeyValuePair<string, long>> Describe<T>(Kokoro<T> model, string option)
        => Observations.TryGetValue(model, out var parts) ? parts : Measure(model, option);

    private static List<KeyValuePair<string, long>> Measure<T>(Kokoro<T> model, string option)
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
        var hashes = new List<KeyValuePair<string, long>> { new("output", output) };
        int index = 0;
        foreach (var chunk in model.GetParameterChunks())
            hashes.Add(new($"parameter chunk {index++} [{string.Join(",", chunk.Shape)}]",
                TtsPaperNumerics.Fingerprint(Enumerable.Range(0, chunk.Length).Select(i => ops.ToDouble(chunk[i])))));
        return hashes;
    }
}
