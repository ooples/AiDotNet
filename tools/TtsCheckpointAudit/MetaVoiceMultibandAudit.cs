using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceMultibandAudit
{
    internal static void Run(string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var config = root.GetProperty("configuration");
        int Integer(string name) => config.GetProperty(name).GetInt32();
        var model = new MetaVoiceDiffusionUnet<double>(AiDotNetEngine.Current,
            hidden: Integer("hidden"), depth: Integer("depth"), growth: Integer("growth"),
            codecDimension: Integer("codec_dim"), groups: Integer("norm_groups"),
            residualBlocks: Integer("res_blocks"), kernel: Integer("kernel"), stride: Integer("stride"),
            maximumChannels: Integer("max_channels"));
        Console.WriteLine($"MetaVoice MBD band {root.GetProperty("band").GetInt32()}: importing released tensors.");
        var state = new WeightLoader().LoadWeights(root.GetProperty("weights").GetString()!);
        var imported = new HashSet<string>();
        model.LoadTorch((name, shape) =>
        {
            if (!state.TryGetValue(name, out var value)) throw new InvalidDataException("Missing MBD weight: " + name);
            if (!value.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("MBD tensor axes differ: " + name);
            if (!imported.Add(name)) throw new InvalidDataException("Duplicate MBD tensor: " + name);
            return value.ToArray().Select(x => (double)x).ToArray();
        });
        if (imported.Count != state.Count || state.Count != root.GetProperty("state_tensor_count").GetInt32())
            throw new InvalidDataException($"Consumed {imported.Count} of {state.Count} MBD tensors.");
        foreach (var layer in model.Layers) layer.SetTrainingMode(false);
        var waveformValues = Values(root.GetProperty("waveform"));
        var conditionValues = Values(root.GetProperty("condition"));
        var waveform = new Tensor<double>(new[] { 1, 1, waveformValues.Length }, new Vector<double>(waveformValues));
        int codecDimension = Integer("codec_dim");
        var condition = new Tensor<double>(new[] { 1, codecDimension, conditionValues.Length / codecDimension }, new Vector<double>(conditionValues));
        using var noGrad = new NoGradScope<double>();
        var output = model.Forward(waveform, new[] { root.GetProperty("step").GetInt32() }, condition).ToArray();
        double error = output.Zip(Values(root.GetProperty("matched_float64")), (a, b) => Math.Abs(a - b)).Max();
        double drift = output.Zip(Values(root.GetProperty("original_float32")), (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"MetaVoice MBD: all {imported.Count} tensors imported; matched-float64 error {error:R}; original-float32 drift {drift:R}.");
        if (!double.IsFinite(error) || error > 1e-8) throw new InvalidDataException("Released MBD denoiser waveform mismatch.");
    }
}
