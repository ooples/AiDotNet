using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.PyTorchParity;

// An explicit standalone diagnostic. Hashing does not run inside benchmark timing/allocation windows.
internal static class HifiGanInitialStateFingerprint
{
    internal static int Run(string output, int seed)
    {
        if (File.Exists(output)) throw new IOException("Refusing to overwrite initial-state evidence.");
        var benchmark = new AiDotNetHifiGanV1Model(seed);
        var model = benchmark.NativeModelForStateDiagnostics;
        var tensors = new List<object>();
        using var all = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        int learnedCount = 0, spectralBuffers = 0;
        long learnedElements = 0;
        void Record(string name, Tensor<float> value, string role)
        {
            float[] values = value.ToArray();
            var bytes = MemoryMarshal.AsBytes(values.AsSpan());
            all.AppendData(Encoding.UTF8.GetBytes(name + "|" + string.Join(",", value.Shape.ToArray()) + "|"));
            all.AppendData(bytes);
            tensors.Add(new { name, role, shape = value.Shape.ToArray(), elements = values.Length,
                sha256 = Convert.ToHexString(SHA256.HashData(bytes)) });
            if (role == "learned") { learnedCount++; learnedElements += values.Length; }
            if (role == "spectral_buffer") spectralBuffers++;
        }
        foreach (var moduleField in new[] { ("generator", "_generator"), ("discriminator", "_discriminators") })
        {
            object module = typeof(HiFiGAN<float>).GetField(moduleField.Item2, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
            var layers = (IReadOnlyList<LayerBase<float>>)module.GetType().GetProperty("Layers")!.GetValue(module)!;
            for (int layer = 0; layer < layers.Count; layer++)
            {
                var conv = (NormedConv1DLayer<float>)layers[layer];
                var type = typeof(NormedConv1DLayer<float>);
                var norm = (ConvolutionNormalization)type.GetField("_normalization", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(conv)!;
                bool bias = (bool)type.GetField("_useBias", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(conv)!;
                foreach (string field in new[] { "_direction", "_gain", "_bias", "_u", "_v" })
                {
                    string role = field == "_direction" || field == "_gain" && norm == ConvolutionNormalization.Weight || field == "_bias" && bias
                        ? "learned" : (field == "_u" || field == "_v") && norm == ConvolutionNormalization.Spectral ? "spectral_buffer" : "inactive_stored_state";
                    var tensor = (Tensor<float>)type.GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(conv)!;
                    Record($"{moduleField.Item1}/{layer:D3}/{field}", tensor, role);
                }
            }
        }
        string stateHash = Convert.ToHexString(all.GetHashAndReset());
        Record("input/mel", benchmark.CurrentBatch.Input, "input");
        Record("input/audio", benchmark.CurrentBatch.Label, "input");
        string inputHash = Convert.ToHexString(all.GetHashAndReset());
        if (learnedElements != benchmark.ParameterCount) throw new InvalidOperationException("State inventory missed learned parameters.");
        string Hash(string file) { using var stream = File.OpenRead(file); return Convert.ToHexString(SHA256.HashData(stream)); }
        string json = JsonSerializer.Serialize(new
        {
            schema = "hifigan-full-initial-state-fingerprint-v1", diagnostic_only = true, pid = Environment.ProcessId,
            seed, engine = AiDotNetEngine.Current.Name, complete_stored_state_sha256 = stateHash, input_sha256 = inputHash,
            parameter_elements = learnedElements, learned_tensor_count = learnedCount, spectral_buffer_count = spectralBuffers,
            ai_sha256 = Hash(typeof(HiFiGAN<float>).Assembly.Location), tensors_sha256 = Hash(typeof(Tensor<float>).Assembly.Location),
            harness_sha256 = Hash(Assembly.GetExecutingAssembly().Location), tensors
        }, new JsonSerializerOptions { WriteIndented = true }) + "\n";
        // Two diagnostics can both pass the existence check above. Publish by moving a private file into place
        // without overwrite, so the second one fails instead of replacing the first receipt.
        string pending = output + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllText(pending, json);
            File.Move(pending, output, overwrite: false);
        }
        finally { File.Delete(pending); }
        Console.WriteLine($"Initial state={stateHash}; input={inputHash}; learned={learnedElements}.");
        return 0;
    }
}
