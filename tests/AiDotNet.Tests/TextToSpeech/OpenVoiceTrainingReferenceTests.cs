using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceTrainingReferenceTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OriginalPosteriorAndDirectDecoder_WithSeparateFlowKl_MatchEveryParameterAndInputGradient(bool versionTwo)
    {
        AiDotNetEngine.ResetToCpu();
        string directory = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData");
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "openvoice_training.json")));
        var item = metadata.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("zero_condition").GetBoolean() == versionTwo);
        string label = item.GetProperty("label").GetString()!;
        using var stream = File.OpenRead(Path.Combine(directory, "openvoice_training.safetensors")); using var reader = new BinaryReader(stream);
        long headerLength = reader.ReadInt64(); using var header = JsonDocument.Parse(reader.ReadBytes(checked((int)headerLength)));
        double[] Read(string name, int[]? shape = null)
        {
            var entry = header.RootElement.GetProperty(label + "." + name); Assert.Equal("F64", entry.GetProperty("dtype").GetString());
            if (shape is not null) Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            var offsets = entry.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            stream.Position = 8 + headerLength + offsets[0]; var values = new double[checked((int)((offsets[1] - offsets[0]) / 8))];
            for (int i = 0; i < values.Length; i++) values[i] = reader.ReadDouble(); return values;
        }
        var engine = AiDotNetEngine.Current;
        var model = new OpenVoiceConverterCore<double>(engine, 9, 4, 4, 3, versionTwo,
            new[] { 2 }, new[] { 4 }, 8, new[] { 3 }, new[] { new[] { 1, 3, 5 } });
        var loaded = new HashSet<string>();
        model.LoadTorch((name, shape) => { Assert.True(loaded.Add(name)); return Read("weight." + name, shape); }, name =>
        {
            if (!header.RootElement.TryGetProperty(label + ".weight." + name, out _)) return null;
            Assert.True(loaded.Add(name)); return Read("weight." + name);
        });
        var names = item.GetProperty("parameters").EnumerateArray().Select(x => x.GetString()!).ToArray();
        Assert.Equal(253, names.Length); Assert.Equal(names.Length, loaded.Count);
        var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(names.Length, owned.Count);
        var remaining = names.ToDictionary(name => name, name => Read("weight." + name)); var mapping = new Dictionary<Tensor<double>, string>();
        foreach (var parameter in owned)
        {
            var values = parameter.ToArray();
            var candidates = remaining.Where(x => x.Value.Length == values.Length && x.Value.Zip(values, (a, b) => Math.Abs(a - b)).Max() < 1e-12).ToArray();
            Assert.Single(candidates); mapping.Add(parameter, candidates[0].Key); remaining.Remove(candidates[0].Key);
        }
        Assert.Empty(remaining);
        Tensor<double> Input(string name, params int[] shape) => new(shape, new Vector<double>(Read(name)));
        var spectrogram = Input("spectrogram", 2, 9, 7); var speaker = Input("speaker", 2, 3, 1); var noise = Input("noise", 2, 4, 7);
        var priorMean = Input("prior_mean", 2, 4, 7); var priorLogs = Input("prior_log_scale", 2, 4, 7); var mask = Input("mask", 2, 1, 7);
        using var tape = new GradientTape<double>();
        var result = model.TrainingForward(spectrogram, mask, speaker, noise, 1, 3);
        void Close(Tensor<double> actual, string expected)
        {
            double error = actual.ToArray().Zip(Read(expected), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < 2e-9, expected + " error " + error.ToString("R"));
        }
        Close(result.Waveform, "waveform"); Close(result.Posterior, "posterior"); Close(result.Prior, "prior");
        Close(result.Mean, "mean"); Close(result.LogScale, "log_scale");
        var kl = OpenVoiceTrainingLoss<double>.AlignedKl(engine, result.Prior, result.LogScale, priorMean, priorLogs, mask); Close(kl, "kl");
        var waveLoss = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(result.Waveform, Input("wave_probe", 2, 1, 6)), new[] { 0, 1, 2 }, true), new[] { 1 });
        var loss = engine.TensorAdd(engine.TensorMultiplyScalar(waveLoss, .2), engine.TensorMultiplyScalar(kl, .7));
        Assert.True(Math.Abs(loss[0] - item.GetProperty("loss").GetDouble()) < 2e-9);
        var inputs = new[] { spectrogram, speaker, noise, priorMean, priorLogs };
        var gradients = tape.ComputeGradients(loss, sources: owned.Concat(inputs).ToList());
        foreach (var parameter in owned) Close(gradients[parameter], "gradient." + mapping[parameter]);
        string[] inputNames = { "spectrogram", "speaker", "noise", "prior_mean", "prior_log_scale" };
        for (int i = 0; i < inputs.Length; i++) Close(gradients[inputs[i]], inputNames[i] + "_gradient");
    }
}
