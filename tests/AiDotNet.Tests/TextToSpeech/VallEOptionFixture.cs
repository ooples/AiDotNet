using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Tests.TextToSpeech;

internal static class VallEOptionFixture
{
    private static TtsTrainingSample<T> Sample<T>(VALLE<T> model)
    {
        var options = (VALLEOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int samples = checked(16 * options.HopSize);
        var audio = new Tensor<T>(new[] { samples }, new Vector<T>(Enumerable.Range(0, samples)
            .Select(i => ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180 * i / options.SampleRate)
                + 0.2 * Math.Sin(2 * Math.PI * 470 * i / options.SampleRate))).ToArray()));
        model.Codec.InitializeCodebooks(audio);
        var phones = model.EncodePhonemes(EnglishG2P.Default.Phonemize("hello there"));
        return new TtsTrainingSample<T>
        {
            Audio = audio,
            Tokens = new Tensor<T>(new[] { phones.Length }, new Vector<T>(phones.Select(id => ops.FromDouble(id)).ToArray())),
        };
    }

    internal static long ObserveTrainingPrompt<T>(VALLE<T> model)
    {
        model.CurrentStage = VallETrainingStage.NonAutoRegressive;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(Sample(model))) });
    }

    internal static void TrainWithPrompt<T>(VALLE<T> model, int steps)
    {
        model.CurrentStage = VallETrainingStage.NonAutoRegressive;
        var sample = Sample(model);
        for (int step = 0; step < steps; step++) model.Train(sample);
    }

    internal static long ObserveEndCondition<T>(VALLE<T> model)
    {
        using var _ = new NoGradScope<T>();
        var (sample, prompt, continuation) = EndSample(model);
        var core = model.PaperCore;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var text = new[] { 1 }.Concat(sample.Tokens.ToArray().Select(value => (int)ops.ToDouble(value))).Append(2).ToArray();
        var unusedRandom = new Random(0);
        int eos = core.EndToken;

        double[] Logits(params int[] codes)
        {
            var logits = core.ArLogits(text, codes, training: false, unusedRandom);
            return Enumerable.Range(0, eos + 1).Select(id => ops.ToDouble(logits[codes.Length - 1, id])).ToArray();
        }

        var first = Logits(prompt);
        var second = Logits(prompt, continuation);
        double ordinaryMaximum = second.Take(eos).Max();
        if (first[eos] == first.Max() || second[eos] <= ordinaryMaximum) return 0;
        // Draw from the native probability mass of ordinary codes. The second draw isolates
        // stopping on an argmax EOS from stopping on a sampled EOS.
        int next = Array.IndexOf(second, ordinaryMaximum);
        return core.ArGenerate(text, new[] { prompt }, temperature: 1, topK: 0, maxNewCodes: 2,
            new TwoDraws(DrawFor(first, continuation), DrawFor(second, next))).Count;
    }

    internal static void TrainEndCondition<T>(VALLE<T> model)
    {
        model.CurrentStage = VallETrainingStage.AutoRegressive;
        var (sample, _, _) = EndSample(model);
        for (int step = 0; step < 24; step++) model.Train(sample);
    }

    private static (TtsTrainingSample<T> Sample, int Prompt, int Continuation) EndSample<T>(VALLE<T> model)
    {
        var recording = Sample(model);
        var encoded = model.Codec.Encode(recording.Audio!);
        int first = Enumerable.Range(0, encoded.GetLength(1) - 1)
            .First(frame => encoded[0, frame] != encoded[0, frame + 1]);
        int codebooks = encoded.GetLength(0);
        var codes = new Tensor<T>(new[] { 2, codebooks });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int frame = 0; frame < 2; frame++)
            for (int codebook = 0; codebook < codebooks; codebook++)
                codes[frame, codebook] = ops.FromDouble(encoded[codebook, first + frame]);
        return (new TtsTrainingSample<T> { Tokens = recording.Tokens, CodecTokens = codes },
            encoded[0, first], encoded[0, first + 1]);
    }

    private static double DrawFor(double[] logits, int code)
    {
        double max = logits.Max();
        var masses = logits.Select(value => Math.Exp(value - max)).ToArray();
        return (masses.Take(code).Sum() + 0.5 * masses[code]) / masses.Sum();
    }

    private sealed class TwoDraws(double first, double second) : Random
    {
        private bool _first = true;

        public override double NextDouble()
        {
            if (!_first) return second;
            _first = false;
            return first;
        }
    }
}
