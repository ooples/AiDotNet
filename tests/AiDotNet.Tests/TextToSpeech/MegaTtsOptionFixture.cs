using AiDotNet.Models.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.Tests.TextToSpeech;

internal static class MegaTtsOptionFixture
{
    private const string BilingualText = "Hello 你好。";

    internal static bool HandlesVocoder(string option) => option is "HopSize" or "VocoderRates";
    internal static bool HandlesAcoustic(string option) =>
        option is "DropoutRate" or "LearningRate" or "WeightDecay" or "TrainingSeed" or "WordDurationWeight";
    internal static bool HandlesLanguageModels(string option) => option is "DurationHeads" or "LmHeads";

    internal static void Prepare(ModelOptions options, string option)
    {
        if (option != "Language") return;
        var phones = NatSpeechFrontEnd.EnglishPhones().Concat(NatSpeechFrontEnd.ChinesePhonesOf(BilingualText)).Distinct().ToArray();
        if (options is MegaTTSOptions first) first.PhoneSet = phones;
        if (options is MegaTTS2Options second) second.PhoneSet = phones;
    }

    internal static void Reconcile(ModelOptions options, string option)
    {
        if (!HandlesVocoder(option)) return;
        if (options is MegaTTSOptions first)
            (first.HopSize, first.VocoderRates, first.VocoderKernels) =
                MatchVocoder(option, first.HopSize, first.VocoderRates, first.VocoderKernels);
        if (options is MegaTTS2Options second)
            (second.HopSize, second.VocoderRates, second.VocoderKernels) =
                MatchVocoder(option, second.HopSize, second.VocoderRates, second.VocoderKernels);
    }

    private static (int Hop, int[] Rates, int[] Kernels) MatchVocoder(string option, int hop, int[] rates, int[] kernels)
    {
        if (rates.Length == 0 || rates.Any(rate => rate < 1)) return (hop, rates, kernels);
        if (option == "HopSize")
        {
            int prefix = rates.Take(rates.Length - 1).Aggregate(1, (product, rate) => checked(product * rate));
            if (hop < prefix || hop % prefix != 0) return (hop, rates, kernels);
            rates = (int[])rates.Clone();
            rates[^1] = hop / prefix;
        }
        else
        {
            hop = rates.Aggregate(1, (product, rate) => checked(product * rate));
        }
        return (hop, rates, rates.Select(rate => checked(2 * rate)).ToArray());
    }

    internal static long ObserveTokens<T>(TtsModelBase<T> model) => Fingerprint(model.TextToTokens(BilingualText));

    internal static long ObserveWaveform<T>(TtsModelBase<T> model)
    {
        model.Voice = Prompt(model, "Hello there.", 180);
        return Fingerprint(model.Synthesize("How are you?"));
    }

    internal static long ObserveProsodyInterpolation<T>(MegaTTS2<T> model)
    {
        var target = model.ReadPrompt(Prompt(model, "Hello there.", 180), "target");
        var style = model.ReadPrompt(Prompt(model, "Good morning.", 330), "style");
        var probabilities = new List<double>();
        model.SampleProsody(new[] { target }, new[] { style }, target.Pooled!, new Random(0),
            distribution => probabilities.AddRange(distribution.Cast<double>().ToArray()));
        return OptionSensitivityHarness.Fingerprint(probabilities);
    }

    private static TtsVoice<T> Prompt<T>(TtsModelBase<T> model, string text, double frequency)
    {
        var options = (TtsModelOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var tokens = model.TextToTokens(text);
        int frames = Math.Max(20, 2 * tokens.Length);
        int samples = checked((frames - 1) * options.HopSize);
        var audio = Enumerable.Range(0, samples).Select(i =>
            ops.FromDouble(.35 * Math.Sin(2 * Math.PI * frequency * i / options.SampleRate)
                + .12 * Math.Sin(2 * Math.PI * 2.7 * frequency * i / options.SampleRate))).ToArray();
        return new TtsVoice<T>
        {
            ReferenceAudio = new Tensor<T>(new[] { samples }, new Vector<T>(audio)),
            ReferenceTokens = tokens,
            ReferenceDurations = Durations(frames, tokens.Length),
        };
    }

    private static int[] Durations(int frames, int tokens) =>
        Enumerable.Range(0, tokens).Select(i => frames / tokens + (i < frames % tokens ? 1 : 0)).ToArray();

    internal static TtsTrainingSample<T> Sample<T>(TtsModelBase<T> model, string text = "Hello world.")
    {
        var tokens = model.TextToTokens(text);
        int frames = Math.Max(32, 2 * tokens.Length);
        int channels = ((TtsModelOptions)model.GetOptions()).MelChannels;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var mel = new Tensor<T>(new[] { frames, channels });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < channels; c++)
                mel[f, c] = ops.FromDouble(.45 * Math.Sin(.19 * f + .13 * c));
        return new TtsTrainingSample<T> { Tokens = tokens, Mel = mel, Durations = Durations(frames, tokens.Length) };
    }

    internal static long ObserveObjective<T>(TtsModelBase<T> model, bool languageModels)
    {
        SelectStage(model, languageModels);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(Sample(model))) });
    }

    internal static void Train<T>(TtsModelBase<T> model, bool languageModels, int steps)
    {
        SelectStage(model, languageModels);
        var sample = Sample(model);
        for (int step = 0; step < steps; step++) model.Train(sample);
    }

    private static void SelectStage<T>(TtsModelBase<T> model, bool languageModels)
    {
        if (model.GetOptions() is MegaTTS2Options options)
            options.Stage = languageModels ? MegaTts2Stage.LanguageModels : MegaTts2Stage.Acoustic;
        // Mega-TTS trains its prosody language model (section 3.3) as a separate stage after the acoustic model.
        if (model.GetOptions() is MegaTTSOptions first)
            first.Stage = languageModels ? MegaTtsStage.ProsodyLanguageModel : MegaTtsStage.Acoustic;
    }

    private static long Fingerprint<T>(Tensor<T> values)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return OptionSensitivityHarness.Fingerprint(values.ToArray().Select(ops.ToDouble));
    }
}
