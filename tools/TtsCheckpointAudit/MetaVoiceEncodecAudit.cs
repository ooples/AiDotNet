using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

internal static class MetaVoiceEncodecAudit
{
    internal static void Run(string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var options = EnCodecOptions.OfficialCheckpoint24kHz();
        options.TargetBandwidthKbps = 6;
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 2294 };
        using var codec = new EnCodec<double>(architecture, options);
        // Existing loader validates every tensor shape and rejects every unconsumed entry.
        codec.LoadPretrainedWeights(root.GetProperty("release").GetProperty("paths").GetProperty("model.safetensors").GetString()!);
        var reference = root.GetProperty("matched_float64"); var original = root.GetProperty("original_float32");
        var audioValues = Values(root.GetProperty("audio"));
        var audio = new Tensor<double>(new[] { 1, 1, audioValues.Length }, new Vector<double>(audioValues));
        var codes = codec.Encode(audio);
        var expectedCodes = reference.GetProperty("codes");
        if (codes.GetLength(0) != 8 || codes.GetLength(1) != expectedCodes[0].GetArrayLength())
            throw new InvalidDataException("MetaVoice EnCodec code axes differ.");
        for (int q = 0; q < 8; q++) for (int t = 0; t < codes.GetLength(1); t++)
            if (codes[q, t] != expectedCodes[q][t].GetInt32()) throw new InvalidDataException($"MetaVoice EnCodec token differs at [{q},{t}].");
        void Check(string name, Tensor<double> actual)
        {
            var expected = Values(reference.GetProperty(name)); var originalValues = Values(original.GetProperty(name));
            if (actual.Length != expected.Length) throw new InvalidDataException("EnCodec output axes differ: " + name);
            double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            double drift = actual.ToArray().Zip(originalValues, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"MetaVoice EnCodec {name}: matched-float64 error {error:R}; original-float32 drift {drift:R}.");
            if (!double.IsFinite(error) || error > 1e-8) throw new InvalidDataException("MetaVoice EnCodec numerical mismatch: " + name);
        }
        Check("latent", codec.DecodeCodesToEmbedding(codes));
        Check("waveform", codec.Decode(codes));
        Console.WriteLine($"MetaVoice EnCodec: all {root.GetProperty("state_tensor_count").GetInt32()} tensors imported; all eight codebooks' IDs match exactly.");
    }
}
