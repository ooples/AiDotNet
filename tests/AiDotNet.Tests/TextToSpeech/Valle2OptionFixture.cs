using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Teacher-forced observations do not require an untrained AR model to avoid EOS.</summary>
internal static class Valle2OptionFixture
{
    internal static long? Observe<T>(VALLE2<T> model, string option)
    {
        var options = (VALLE2Options)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option is "MaxCodesPerTextToken" or "TopP" or "RepetitionWindow" or "RepetitionThreshold" or "SamplingSeed")
        {
            // A uniform tied readout gives a reproducible sampling boundary. For the limit
            // probe, a zero draw always chooses code zero, never EOS.
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var table = (Tensor<T>)model.Core2ForTests.ArCodeEmbedding.GetType().GetField("_table", flags)!.GetValue(model.Core2ForTests.ArCodeEmbedding)!;
            var saved = table.ToArray();
            try
            {
                for (int i = 0; i < table.Length; i++) table[i] = ops.Zero;
                var generate = typeof(VALLE2<T>).GetMethod("Generate", flags)!;
                var prompt = new int[options.GroupSize, options.NumCodebooks];
                for (int f = 0; f < prompt.GetLength(0); f++)
                    for (int q = 0; q < prompt.GetLength(1); q++) prompt[f, q] = (f + 1) % options.CodebookSize;
                var generated = (int[,])generate.Invoke(model, new object[]
                {
                    new[] { 1, 2, 3 }, new[] { 1, 2 }, prompt,
                    new TtsVoice<T>(), option == "MaxCodesPerTextToken" ? new FirstCodeRandom() : new Random(options.SamplingSeed),
                })!;
                return OptionSensitivityHarness.Fingerprint(new[] { (double)generated.GetLength(0) }
                    .Concat(generated.Cast<int>().Select(code => (double)code)));
            }
            finally { for (int i = 0; i < table.Length; i++) table[i] = saved[i]; }
        }
        int frames = Math.Max(16, 4 * options.GroupSize);
        var codes = new Tensor<T>(new[] { frames, options.NumCodebooks });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < options.NumCodebooks; q++)
                codes[f, q] = ops.FromDouble((5 * f + 11 * q) % options.CodebookSize);
        var sample = new TtsTrainingSample<T>
        {
            Tokens = model.TextToTokens("hello there, this sentence exercises the text limit"),
            CodecTokens = codes,
        };
        var stage = model.CurrentStage;
        var observations = new List<double>();
        try
        {
            foreach (var next in new[] { VallETrainingStage.AutoRegressive, VallETrainingStage.NonAutoRegressive })
            {
                model.CurrentStage = next;
                observations.Add(ops.ToDouble(model.EvaluateTrainingObjective(sample)));
            }
            observations.AddRange(model.DecodeFromTokens(codes).Select(ops.ToDouble));
            return OptionSensitivityHarness.Fingerprint(observations);
        }
        finally { model.CurrentStage = stage; }
    }

    private sealed class FirstCodeRandom : Random
    {
        public override double NextDouble() => 0;
    }
}
