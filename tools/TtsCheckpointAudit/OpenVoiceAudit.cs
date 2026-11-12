using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class OpenVoiceAudit
{
    internal static void Run(string directory, string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        var state = new WeightLoader().LoadWeights(Path.Combine(directory, "converter.safetensors"));
        var imported = new HashSet<string>();
        double[]? Optional(string name)
        {
            if (!state.TryGetValue(name, out var tensor)) return null;
            imported.Add(name);
            return tensor.ToArray().Select(x => (double)x).ToArray();
        }
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException("Missing tensor: " + name);
            if (!tensor.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("Incompatible shape: " + name);
            return Optional(name)!;
        }
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        Tensor<double> Input(string name, params int[] shape) => new(shape, new Vector<double>(Values(root.GetProperty(name))));
        void Check(string name, Tensor<double> actual)
        {
            var target = Values(root.GetProperty(name));
            if (target.Length != actual.Length) throw new InvalidDataException(name + ": wrong output size.");
            double error = actual.ToArray().Zip(target, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"{root.GetProperty("version").GetString()} {name}: maximum error {error:R}");
            if (!double.IsFinite(error) || error > 1e-8) throw new InvalidDataException(name + ": numerical mismatch.");
            if (root.TryGetProperty("original_float32_window", out var original) && original.TryGetProperty(name, out var originalValue))
            {
                double originalError = actual.ToArray().Zip(Values(originalValue), (a, b) => Math.Abs(a - b)).Max();
                Console.WriteLine($"{name}: separately retained original float32-window drift {originalError:R}");
            }
        }
        var engine = AiDotNetEngine.Current;
        var reference = new OpenVoiceReferenceEncoder<double>(engine, 513);
        var converter = new OpenVoiceConverterCore<double>(engine, zeroCondition: root.GetProperty("zero_condition").GetBoolean());
        reference.LoadTorch(Read);
        converter.LoadTorch(Read, Optional);
        if (state.Count != imported.Count) throw new InvalidDataException($"Imported {imported.Count} of {state.Count} tensors.");
        foreach (var layer in reference.Layers.Concat(converter.Layers)) layer.SetTrainingMode(false);
        var source = reference.Forward(Input("reference", 1, 129, 513));
        Check("source", source);
        var mask = new Tensor<double>(new[] { 1, 1, 9 });
        for (int i = 0; i < mask.Length; i++) mask[i] = 1;
        var output = converter.Convert(Input("spectrogram", 1, 513, 9), mask,
            engine.Reshape(source, new[] { 1, 256, 1 }), Input("target", 1, 256, 1), Input("noise", 1, 192, 9));
        Check("posterior", output.Posterior); Check("prior", output.Prior);
        Check("converted", output.Converted); Check("waveform", output.Waveform);
        var native = new OpenVoiceNative<double>(engine, 1024, 256, 192, 192, 256,
            root.GetProperty("zero_condition").GetBoolean());
        native.LoadCheckpoint(root.GetProperty("checkpoint_path").GetString()!);
        foreach (var layer in native.Layers) layer.SetTrainingMode(false);
        var rawAudio = Input("raw_audio", 2304);
        var rawSource = native.Embedding(rawAudio);
        var rawTarget = native.Embedding(Input("raw_reference", 33024));
        Check("raw_source", rawSource); Check("raw_target", rawTarget);
        Check("raw_waveform", native.Convert(rawAudio, rawSource, rawTarget, noise: Input("noise", 1, 192, 9)));
        Console.WriteLine($"All {imported.Count} released tensors imported; reference encoder and complete converter agree.");
    }
}
