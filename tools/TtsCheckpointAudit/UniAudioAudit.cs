using System.Reflection;
using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class UniAudioAudit
{
    internal static void Inspect(string directory)
    {
        var state = new WeightLoader().LoadWeights(Path.Combine(directory, "ep5-iter340000.checkpoint"),
            name => name.StartsWith("model.", StringComparison.Ordinal));
        Console.WriteLine($"Original UniAudio training archive: {state.Count} model tensors; optimizer tensors excluded before materialization.");
        if (state.Count != 493 || !state["model.g_emb.weight"].Shape.ToArray().SequenceEqual(new[] { 4215, 512 }))
            throw new InvalidDataException("Original UniAudio archive axes differ.");
    }

    internal static void Run(string directory, string fixture, string codecFixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        using var codecDocument = JsonDocument.Parse(File.ReadAllText(codecFixture));
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static int[] Ids(JsonElement value) => value.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var failures = new List<string>();
        void Check(string label, Tensor<double> actual, JsonElement target, double tolerance)
        {
            var expected = Values(target);
            if (actual.Length != expected.Length) throw new InvalidDataException(label + ": incompatible output lengths.");
            double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance)
                failures.Add($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        Console.WriteLine("Creating the full original 24-global/8-local UniAudio graph and universal codec.");
        using var model = new UniAudio<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 4215), UniAudioOptions.OfficialV1());
        model.DisableAutoStreaming();
        Console.WriteLine("Loading original ep5-iter340000.checkpoint and universal_model/model.pth through public LoadAuthorWeights.");
        model.LoadAuthorWeights(Path.Combine(directory, "ep5-iter340000.checkpoint"),
            Path.Combine(directory, "universal_model", "model.pth"));
        model.SetTrainingMode(false);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        T Field<T>(string name) => (T)typeof(UniAudio<double>).GetField(name,
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var network = Field<UniAudioMultiScaleTransformer<double>>("_languageModel");
        var codec = Field<UniAudioCodec<double>>("_codec");
        Console.WriteLine("Original double references retain the author's forced-float32 LayerNorm and attention softmax.");
        Console.WriteLine("The additional float64-operator reference isolates those casts using TorchDispatchMode; original forward methods and trained weights are unchanged. Original float32 cross-precision deltas are reported separately.");
        int caseIndex = 0;
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var sequence = Ids(item.GetProperty("sequence"));
            var logits = network.Forward(sequence, item.GetProperty("prefix_tokens").GetInt32());
            var operatorReference = item.GetProperty("float64_operator_precision");
            var operatorLogits = Values(operatorReference.GetProperty("logits"));
            foreach (string precision in new[] { "float64_operator_precision", "float64_mixed_norm_softmax", "float32" })
            {
                var reference = item.GetProperty(precision);
                double authorPrecisionDelta = Values(reference.GetProperty("logits"))
                    .Zip(operatorLogits, (a, b) => Math.Abs(a - b)).Max();
                Console.WriteLine($"Author-only case {caseIndex} / {precision} arithmetic-precision delta: {authorPrecisionDelta:R}.");
                Check($"Full UniAudio case {caseIndex} / {precision}", logits, reference.GetProperty("logits"),
                    authorPrecisionDelta + 2e-8);
                double loss = UniAudioOps<double>.TeacherForcedLoss(AiDotNetEngine.Current, logits, sequence,
                    item.GetProperty("prefix_tokens").GetInt32(), false, 0)[0];
                double lossError = Math.Abs(loss - reference.GetProperty("loss").GetDouble());
                Console.WriteLine($"Full UniAudio case {caseIndex} loss / {precision}: error {lossError:R}.");
                double authorLossDelta = Math.Abs(reference.GetProperty("loss").GetDouble() - operatorReference.GetProperty("loss").GetDouble());
                if (!double.IsFinite(lossError) || lossError > authorLossDelta + 1e-10)
                    failures.Add($"UniAudio case {caseIndex} loss differs by {lossError:R}.");
                var expected = Ids(reference.GetProperty("argmax"));
                for (int row = 0; row < expected.Length; row++)
                {
                    int best = 0;
                    for (int v = 1; v < 4215; v++) if (logits[row, v] > logits[row, best]) best = v;
                    if (best != expected[row]) failures.Add($"UniAudio case {caseIndex} / {precision}: argmax differs at row {row}.");
                }
            }
            caseIndex++;
        }
        foreach (var item in codecDocument.RootElement.GetProperty("cases").EnumerateArray())
        {
            var audio = Values(item.GetProperty("audio"));
            var input = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio));
            var codes = codec.Encode(input);
            var expected = item.GetProperty("float64");
            var expectedCodes = Ids(expected.GetProperty("codes"));
            if (codes.Length != expectedCodes.Length) throw new InvalidDataException("Universal codec token shape differs.");
            for (int q = 0; q < codes.GetLength(0); q++)
                for (int t = 0; t < codes.GetLength(1); t++)
                    if (codes[q, t] != expectedCodes[q * codes.GetLength(1) + t])
                        failures.Add($"Universal codec {audio.Length} samples: code differs at {q}/{t}.");
            Check($"Full universal codec / {audio.Length} samples", codec.Decode(codes), expected.GetProperty("waveform"), 2e-8);
            var publicTokens = model.EncodeToTokens(input);
            var flattened = codec.Flatten(codes);
            for (int i = 0; i < flattened.Length; i++)
                if (publicTokens[i] != flattened[i]) failures.Add("Public UniAudio codec offsets differ.");
            Console.WriteLine($"Universal codec {audio.Length} samples: compared all {codes.Length} IDs and public token offsets.");
        }
        if (failures.Count > 0) throw new InvalidDataException(string.Join(Environment.NewLine, failures));
        Console.WriteLine("Full original UniAudio Transformer and universal codec checkpoint audit passed.");
    }
}
