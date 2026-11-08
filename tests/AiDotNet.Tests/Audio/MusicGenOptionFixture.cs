using AiDotNet.Audio.MusicGen;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.Audio;

internal static class MusicGenOptionFixture
{
    internal static void PrepareOptions(MusicGenOptions options, string option)
    {
        if (option == "Delays") options.Delays = options.ResolveDelays();
        if (option == "StereoPattern") options.Stereo = true;
        if (option is "TopK" or "TopP" or "Temperature" or "UseSampling")
        {
            options.UseSampling = true;
            options.TopK = 2;
            options.TopP = option == "TopP" ? .5 : 0;
        }
        if (option == "MaxTextLength")
        {
            options.MaxTextLength = 4;
            options.TextVocabularySize = 32128;
        }
        if (option is "ChromaWindow" or "ChromaHop" or "MelodyDropout")
        {
            options.MelodyConditioning = true;
            options.ChromaWindow = 64;
            options.ChromaHop = 16;
        }
        if (option is "DropDescriptionProbability" or "MergeMetadataProbability" or "KeepMetadataFieldProbability")
        {
            options.MergeMetadataProbability = .5;
            options.KeepMetadataFieldProbability = .5;
            options.DropDescriptionProbability = .5;
        }
        if (option is "EmaDecay" or "EmaEveryUpdates") options.EmaEveryUpdates = 1;
        if (option is "WarmupSteps" or "TrainingSteps")
        {
            options.WarmupSteps = 1;
            options.TrainingSteps = 8;
        }
    }

    internal static long? Observe<T>(MusicGenModel<T> model, Tensor<T> input, string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option == "StereoPattern")
            return OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(input, StereoCodes(model))) });
        if (option is "TopK" or "TopP" or "Temperature" or "UseSampling")
            return OptionSensitivityHarness.Fingerprint(Enumerable.Range(0, 64)
                .Select(seed => (double)model.ChooseForTests(new[] { 0.0, -0.1, -0.2, -0.3 }, new Random(seed))));
        if (option == "MaxTextLength")
            return OptionSensitivityHarness.Fingerprint(model.TextToTokens("a long description of music with strings and drums").Select(ops.ToDouble));
        if (option == "WordDropout")
            return Strings(Enumerable.Range(0, 32).Select(seed => model.DropWordsForTests("strings drums piano guitar bass flute", new Random(seed))));
        if (option is "DropDescriptionProbability" or "MergeMetadataProbability" or "KeepMetadataFieldProbability")
        {
            var sample = new MusicGenTrainingSample<T>
            {
                Description = "strings and drums",
                Metadata = new Dictionary<string, string> { ["genre"] = "classical", ["bpm"] = "120", ["key"] = "C" },
            };
            return Strings(Enumerable.Range(0, 32).Select(seed => model.AugmentDescriptionForTests(sample, new Random(seed))));
        }
        if (option is "ChromaWindow" or "ChromaHop")
        {
            var melody = new Tensor<T>(new[] { 256 });
            for (int i = 0; i < melody.Length; i++) melody[i] = ops.FromDouble(.3 * Math.Sin(i * .13) + .1 * Math.Cos(i * .037));
            return OptionSensitivityHarness.Fingerprint(model.MelodyConditionForTests(input, melody).Select(ops.ToDouble));
        }
        if (option == "GuidanceScale")
            return OptionSensitivityHarness.Fingerprint(Enumerable.Range(0, 8)
                .SelectMany(seed => model.GenerateCodes(input, 4, seed).Cast<int>().Select(value => (double)value)));
        if (option is "EmaDecay" or "EmaEveryUpdates")
        {
            // These are the weights WithEmaWeights installs for generation, separate from the live optimizer weights.
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var ema = (System.Collections.IDictionary?)model.GetType().GetField("_ema", flags)!.GetValue(model);
            return ema is null ? 0 : OptionSensitivityHarness.Fingerprint(ema.Values.Cast<double[]>().SelectMany(values => values));
        }
        return null;
    }

    internal static bool TryTrain<T>(MusicGenModel<T> model, Tensor<T> input, string option)
    {
        if (option != "StereoPattern") return false;
        model.Train(input, StereoCodes(model));
        return true;
    }

    private static Tensor<T> StereoCodes<T>(MusicGenModel<T> model)
    {
        var options = (MusicGenOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var codes = new Tensor<T>(new[] { 8, 2 * options.Codec.NumQuantizers });
        for (int f = 0; f < codes.Shape[0]; f++)
            for (int q = 0; q < codes.Shape[1]; q++) codes[f, q] = ops.FromDouble((3 * f + 5 * q) % options.Codec.CodebookSize);
        return codes;
    }

    private static long Strings(IEnumerable<string?> values) => OptionSensitivityHarness.Fingerprint(
        values.SelectMany(value => new[] { (double)(value?.Length ?? -1) }.Concat((value ?? string.Empty).Select(c => (double)c))));
}
