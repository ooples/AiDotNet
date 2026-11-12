using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class VallexAudit
{
    private sealed class AuditClone : VALLEXClone<double>
    {
        internal AuditClone(VALLEXCloneOptions options) : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1), options) { }
        internal int[,] GenerateCodes(int[] target, int[] enrolled, int[,] prompt, TtsVoice<double> voice, Random random) =>
            Generate(target, enrolled, prompt, voice, random);
    }

    private sealed class RecordedDraws : Random
    {
        private readonly Queue<double> _draws;
        internal RecordedDraws(JsonElement logits)
        {
            int[] tokens = [7, 11, 1024];
            _draws = new Queue<double>();
            for (int step = 0; step < tokens.Length; step++)
            {
                var row = logits[step].EnumerateArray().Select(x => x.GetDouble()).ToArray();
                double maximum = row.Max(); var probabilities = row.Select(x => Math.Exp(x - maximum)).ToArray();
                _draws.Enqueue((probabilities.Take(tokens[step]).Sum() + probabilities[tokens[step]] / 2) / probabilities.Sum());
            }
        }
        public override double NextDouble() => _draws.Dequeue();
    }

    internal static void Run(string checkpoint, string fixturePath)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixturePath));
        var root = document.RootElement;
        var fixture = root.GetProperty("float64_derived_positions");
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static int[] Ids(JsonElement value) => value.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var options = VALLEXCloneOptions.CommunityCheckpoint();
        options.SynthesisLanguage = VallEXLanguage.Mandarin;
        using var model = new AuditClone(options);
        model.DisableAutoStreaming();
        model.LoadReferenceWeights(checkpoint);
        model.SetTrainingMode(false);
        var core = model.Core!;
        GC.Collect();
        var text = Ids(root.GetProperty("text"));
        var languages = Ids(root.GetProperty("text_languages"));
        var prompt = new int[3, 8]; var rawPrompt = Ids(root.GetProperty("prompt"));
        for (int i = 0; i < rawPrompt.Length; i++) prompt[i / 8, i % 8] = rawPrompt[i];
        using var scope = new NoGradScope<double>();
        void Check(string label, double[] actual, JsonElement expected, double tolerance = 1e-8)
        {
            var target = Values(expected);
            if (actual.Length != target.Length) throw new InvalidDataException(label + " length mismatch.");
            double error = actual.Zip(target, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) throw new InvalidDataException(label + " differs.");
        }
        var sequence = new List<int> { 1025, prompt[0, 0], prompt[1, 0], prompt[2, 0] };
        for (int step = 0; step < 3; step++)
        {
            var logits = core.ArLogits(text, sequence, false, new Random(0), languages);
            Check($"community full AR step {step}", Enumerable.Range(0, 1025).Select(v => logits[sequence.Count - 1, v]).ToArray(),
                fixture.GetProperty("ar_logits")[step]);
            if (step < 2) sequence.Add(step == 0 ? 7 : 11);
        }
        var generated = new int[2, 8]; generated[0, 0] = 7; generated[1, 0] = 11;
        for (int stage = 1; stage < 8; stage++)
        {
            var logits = core.NarLogits(text, prompt, generated, stage, false, new Random(0), languages);
            Check($"community full NAR stage {stage}", logits.ToArray(), fixture.GetProperty("nar_logits")[stage - 1]);
            for (int t = 0; t < 2; t++)
            {
                int best = 0;
                for (int v = 1; v < 1024; v++) if (logits[t, v] > logits[t, best]) best = v;
                generated[t, stage] = best;
            }
        }
        if (!generated.Cast<int>().SequenceEqual(Ids(fixture.GetProperty("codes"))))
            throw new InvalidDataException("Community generated codec tokens differ.");
        var facadeCodes = model.GenerateCodes(text.Skip(2).ToArray(), text.Take(2).ToArray(), prompt,
            new TtsVoice<double> { LanguageId = 0 }, new RecordedDraws(fixture.GetProperty("ar_logits")));
        if (!facadeCodes.Cast<int>().SequenceEqual(Ids(fixture.GetProperty("codes"))))
            throw new InvalidDataException("Facade cross-language context, BOS or sampling produced different codes.");
        Console.WriteLine("Public facade generation: exact original cross-language codec tokens with recorded sampling draws.");
        foreach (string precision in new[] { "original_float32", "original_float64_float32_positions" })
        {
            double drift = 0;
            foreach (string group in new[] { "ar_logits", "nar_logits" })
                for (int i = 0; i < fixture.GetProperty(group).GetArrayLength(); i++)
                    drift = Math.Max(drift, Values(fixture.GetProperty(group)[i])
                        .Zip(Values(root.GetProperty(precision).GetProperty(group)[i]), (a, b) => Math.Abs(a - b)).Max());
            Console.WriteLine($"Author-only {precision} versus double-position drift: {drift:R}; strict native tolerance remains unchanged.");
        }
        Console.WriteLine("PASS: full community VALL-E-X AR and seven NAR stages; not a Microsoft original checkpoint claim.");
    }
}
