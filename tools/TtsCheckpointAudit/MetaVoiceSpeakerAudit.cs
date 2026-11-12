using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceSpeakerAudit
{
    internal static void Run(string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var raw = new WeightLoader().LoadWeights(root.GetProperty("checkpoint_path").GetString()!, name => name.StartsWith("model_state.", StringComparison.Ordinal));
        var state = raw.ToDictionary(pair => pair.Key.Substring("model_state.".Length), pair => pair.Value);
        var imported = new HashSet<string>();
        var model = new MetaVoiceSpeakerEncoder<double>(AiDotNetEngine.Current);
        model.LoadTorch((name, shape) =>
        {
            var value = state[name];
            if (!value.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("Speaker tensor axes differ: " + name);
            imported.Add(name); return value.ToArray().Select(x => (double)x).ToArray();
        });
        var unused = state.Keys.Except(imported).OrderBy(x => x).ToArray();
        if (imported.Count != 14 || !unused.SequenceEqual(new[] { "similarity_bias", "similarity_weight" }) ||
            unused.Any(name => state[name].Length != 1)) throw new InvalidDataException("Unexpected active/unused speaker checkpoint tensors.");
        using var noGrad = new NoGradScope<double>();
        var mels = new Tensor<double>(root.GetProperty("mel_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(), new Vector<double>(Values(root.GetProperty("mels"))));
        void Check(string name, Tensor<double> actual, string matched, string original)
        {
            double error = actual.ToArray().Zip(Values(root.GetProperty(matched)), (a, b) => Math.Abs(a - b)).Max();
            double drift = actual.ToArray().Zip(Values(root.GetProperty(original)), (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"MetaVoice speaker {name}: matched-float64 error {error:R}; original-float32 drift {drift:R}.");
            if (!double.IsFinite(error) || error > 1e-8) throw new InvalidDataException("Speaker comparison mismatch: " + name);
        }
        Check("mel network", model.ForwardMels(mels), "matched_float64", "original_float32");
        var audio = Values(root.GetProperty("audio"));
        Check("raw padded waveform", model.EmbedPreprocessed(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio))),
            "matched_waveform_embedding", "original_waveform_embedding");
        Console.WriteLine("MetaVoice speaker: all14 active tensors imported; two similarity scalars are unused by the unchanged author class.");
    }
}
