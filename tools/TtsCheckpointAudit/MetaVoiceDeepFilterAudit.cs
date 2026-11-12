using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceDeepFilterAudit
{
    internal static void Run(string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture)); var root = document.RootElement;
        string checkpoint = root.GetProperty("checkpoint").GetString()!;
        using (var input = File.OpenRead(checkpoint))
        using (var sha = System.Security.Cryptography.SHA256.Create())
            if (Convert.ToHexString(sha.ComputeHash(input)).ToLowerInvariant() != root.GetProperty("checkpoint_sha256").GetString())
                throw new InvalidDataException("DeepFilterNet checkpoint SHA256 mismatch.");
        var state = new WeightLoader().LoadWeights(checkpoint); int stored = state.Count;
        var model = new MetaVoiceDeepFilterNet<double>();
        model.LoadTorch((name, shape) =>
        {
            if (!state.TryGetValue(name, out var value))
                throw new InvalidDataException("Missing DeepFilterNet tensor: " + name + "; stored=" + stored);
            // WeightLoader represents a Torch rank-zero storage as one native scalar.
            if (!value.Shape.ToArray().SequenceEqual(shape) && !(shape.Length == 0 && value.Length == 1))
                throw new InvalidDataException("Mismatched DeepFilterNet tensor: " + name + "; actual=" + string.Join(",", value.Shape));
            state.Remove(name); return value.ToArray().Select(x => (double)x).ToArray();
        });
        if (state.Count != 0 || stored != 133) throw new InvalidDataException("DeepFilterNet tensor consumption differs.");
        static double[] Values(JsonElement entry) => entry.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static Tensor<double> Tensor(JsonElement item) => new(item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            new Vector<double>(Values(item.GetProperty("values"))));
        static double Error(Tensor<double> value, JsonElement expected)
        {
            var array = Values(expected); if (value.Length != array.Length) throw new InvalidDataException("DeepFilterNet output length differs.");
            return value.ToArray().Zip(array, (a, b) => Math.Abs(a - b)).Max();
        }
        var reference = root.GetProperty("full");
        var spectrum = Tensor(reference.GetProperty("spec")); var erb = Tensor(reference.GetProperty("erb")); var features = Tensor(reference.GetProperty("features"));
        var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { model });
        if (owned.Count != 86) throw new InvalidDataException("DeepFilterNet learned-parameter ownership differs.");
        using var tape = new GradientTape<double>(); var engine = AiDotNetEngine.Current;
        var result = model.Forward(spectrum, erb, features);
        var outputs = new[] { result.Spectrum, result.Mask, result.Snr, result.Coefficients }; Tensor<double>? loss = null;
        double forwardError = 0, gradientError = 0;
        for (int i = 0; i < outputs.Length; i++)
        {
            forwardError = Math.Max(forwardError, Error(outputs[i], reference.GetProperty("outputs")[i].GetProperty("values")));
            var term = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(outputs[i], Tensor(reference.GetProperty("weights")[i])),
                Enumerable.Range(0, outputs[i].Rank).ToArray(), true), new[] { 1 });
            loss = loss is null ? term : engine.TensorAdd(loss, term);
        }
        var gradients = tape.ComputeGradients(loss!, sources: owned.Concat(new[] { spectrum, erb, features }).ToList());
        int unused = 0;
        foreach (var pair in model.NamedParameters)
        {
            var expected = reference.GetProperty("parameters").GetProperty(pair.Key).GetProperty("gradient");
            if (expected.ValueKind == JsonValueKind.Null)
            {
                if (!pair.Key.StartsWith("df_dec.df_fc_a.") || gradients.ContainsKey(pair.Value))
                    throw new InvalidDataException("DeepFilterNet author-unused parameter differs: " + pair.Key);
                unused++; continue;
            }
            if (!gradients.TryGetValue(pair.Value, out var value)) throw new InvalidDataException("DeepFilterNet missing gradient: " + pair.Key);
            gradientError = Math.Max(gradientError, Error(value, expected));
        }
        var inputs = new[] { spectrum, erb, features };
        for (int i = 0; i < inputs.Length; i++) gradientError = Math.Max(gradientError,
            Error(gradients[inputs[i]], reference.GetProperty("input_gradients")[i].GetProperty("values")));
        Console.WriteLine($"DeepFilterNet3: original checkpoint {stored} tensors,86 learned parameters,84 active/2 author-unused; output error {forwardError:R}, all parameter/input gradient error {gradientError:R}.");
        if (unused != 2 || !double.IsFinite(forwardError) || !double.IsFinite(gradientError) || forwardError > 1e-8 || gradientError > 1e-8)
            throw new InvalidDataException("Full trained DeepFilterNet3 neural graph mismatch.");
        var frontend = root.GetProperty("full_frontend");
        var audio = new Tensor<double>(new[] { frontend.GetProperty("audio").GetProperty("values").GetArrayLength() },
            new Vector<double>(Values(frontend.GetProperty("audio").GetProperty("values"))));
        var enhancer = new MetaVoiceDeepFilterEnhancer<double>(checkpoint);
        var waveform = enhancer.Enhance48k(audio);
        double waveformError = Error(waveform, frontend.GetProperty("matched_waveform").GetProperty("values"));
        double float32Error = Error(waveform, frontend.GetProperty("waveform").GetProperty("values"));
        Console.WriteLine($"DeepFilterNet3: actual analysis+features+trained network+synthesis, {waveform.Length} samples; matched-neural-float64/libdf-float32 error {waveformError:R}, original entirely-float32 drift {float32Error:R}.");
        if (!double.IsFinite(waveformError) || waveformError > 2e-6) throw new InvalidDataException("DeepFilterNet3 complete waveform mismatch.");
    }
}
