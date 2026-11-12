using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceGptAudit
{
    internal static void Run(string fixture, string stage)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement.GetProperty("stages").GetProperty(stage);
        var cfg = root.GetProperty("config");
        int[] Integers(string name) => cfg.GetProperty(name).EnumerateArray().Select(x => x.GetInt32()).ToArray();
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var config = new MetaVoiceGptConfiguration
        {
            Dimension = cfg.GetProperty("n_embd").GetInt32(), Layers = cfg.GetProperty("n_layer").GetInt32(),
            Heads = cfg.GetProperty("n_head").GetInt32(), Context = cfg.GetProperty("block_size").GetInt32(),
            InputVocabularies = Integers("vocab_sizes"), Causal = cfg.GetProperty("causal").GetBoolean(),
            OutputVocabularies = cfg.GetProperty("target_vocab_sizes").ValueKind == JsonValueKind.Null ? null : Integers("target_vocab_sizes"),
            RmsNorm = cfg.GetProperty("norm_type").GetString() == "rmsnorm",
            SwiGlu = cfg.GetProperty("nonlinearity_type").GetString() == "swiglu",
            SwiGluMultiple = cfg.GetProperty("swiglu_multiple_of").ValueKind == JsonValueKind.Null ? 256 : cfg.GetProperty("swiglu_multiple_of").GetInt32(),
            SpeakerDropout = cfg.GetProperty("spkemb_dropout").GetDouble(), SpeakerDimension = 256
        };
        Console.WriteLine($"MetaVoice {stage}: importing original {config.Layers}-layer, {config.Dimension}-wide checkpoint.");
        var raw = new WeightLoader().LoadWeights(root.GetProperty("checkpoint_path").GetString()!);
        var state = raw.Where(pair => pair.Key.StartsWith("model.", StringComparison.Ordinal))
            .ToDictionary(pair => pair.Key.Substring(6).Replace("_orig_mod.", ""), pair => pair.Value);
        if (state.Count == 0) state = raw.ToDictionary(pair => pair.Key.Replace("_orig_mod.", ""), pair => pair.Value);
        var imported = new HashSet<string>();
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var value)) throw new InvalidDataException("Missing MetaVoice weight: " + name);
            if (!value.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("MetaVoice weight axes differ: " + name);
            imported.Add(name);
            return value.ToArray().Select(x => (double)x).ToArray();
        }
        var model = new MetaVoiceGpt<double>(AiDotNetEngine.Current, config);
        model.LoadTorch(Read);
        if (imported.Count != state.Count || state.Count != root.GetProperty("state_tensor_count").GetInt32())
            throw new InvalidDataException($"Consumed {imported.Count} of {state.Count} MetaVoice checkpoint entries.");
        foreach (var layer in model.Layers) layer.SetTrainingMode(false);
        int[][] tokens = root.GetProperty("token_streams").EnumerateArray().Select(row => row.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
        var speaker = new Tensor<double>(new[] { 1, 256 }, new Vector<double>(Values(root.GetProperty("speaker"))));
        using var noGrad = new NoGradScope<double>();
        var output = model.Forward(tokens, speaker, lastOnly: config.Causal);
        void Check(string name, Tensor<double> actual, JsonElement matched, JsonElement original)
        {
            var values = actual.ToArray(); var expected = Values(matched); var rawExpected = Values(original);
            if (values.Length != expected.Length) throw new InvalidDataException("MetaVoice output shape differs: " + name);
            double error = values.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            double drift = values.Zip(rawExpected, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"MetaVoice {stage} {name}: matched-float64 error {error:R}; original-float32 drift {drift:R}");
            if (!double.IsFinite(error) || error > 1e-8) throw new InvalidDataException("MetaVoice numerical mismatch: " + name);
        }
        var reference = root.GetProperty("matched_float64"); var original = root.GetProperty("original_float32");
        Check("hidden", output.Hidden, reference.GetProperty("hidden"), original.GetProperty("hidden"));
        for (int i = 0; i < output.Logits.Length; i++)
            Check("head " + i, output.Logits[i], reference.GetProperty("logits")[i], original.GetProperty("logits")[i]);
        Console.WriteLine($"MetaVoice {stage}: all {imported.Count} stored tensors imported, including validated tied-head alias where present.");
    }
}
