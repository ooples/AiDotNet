using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceConverterTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteOfficialConverter_MatchesMaskedLatentsWaveformAndEveryGradient(bool zeroCondition)
    {
        AiDotNetEngine.ResetToCpu();
        string directory = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData");
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "openvoice_converter.json")));
        var item = metadata.RootElement.GetProperty("cases").EnumerateArray()
            .Single(x => x.GetProperty("zero_condition").GetBoolean() == zeroCondition);
        string label = item.GetProperty("label").GetString()!;
        using var stream = File.OpenRead(Path.Combine(directory, "openvoice_converter.safetensors"));
        using var reader = new BinaryReader(stream);
        long length = reader.ReadInt64();
        using var header = JsonDocument.Parse(reader.ReadBytes(checked((int)length)));
        double[] Read(string name, int[]? shape = null)
        {
            var entry = header.RootElement.GetProperty(label + "." + name);
            Assert.Equal("F64", entry.GetProperty("dtype").GetString());
            if (shape is not null) Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            var offsets = entry.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            stream.Position = 8 + length + offsets[0];
            var values = new double[checked((int)((offsets[1] - offsets[0]) / 8))];
            for (int i = 0; i < values.Length; i++) values[i] = reader.ReadDouble();
            return values;
        }
        var engine = AiDotNetEngine.Current;
        var model = new OpenVoiceConverterCore<double>(engine, 9, 4, 4, 3, zeroCondition,
            new[] { 2 }, new[] { 4 }, 8, new[] { 3 }, new[] { new[] { 1, 3, 5 } });
        foreach (var layer in model.Layers) layer.SetTrainingMode(false);
        var imported = new HashSet<string>();
        model.LoadTorch((name, shape) => { imported.Add(name); return Read("weight." + name, shape); }, name =>
        {
            if (!header.RootElement.TryGetProperty(label + ".weight." + name, out _)) return null;
            imported.Add(name); return Read("weight." + name);
        });
        var names = item.GetProperty("parameters").EnumerateArray().Select(x => x.GetString()!).ToArray();
        Assert.Equal(names.Length, imported.Count);
        Assert.All(names, name => Assert.Contains(name, imported));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(names.Length, parameters.Count);
        // Full random tensors identify checkpoint ownership without depending on private field names.
        // Every tensor must match uniquely, and every author parameter must be consumed once.
        var remaining = names.ToDictionary(name => name, name => Read("weight." + name));
        var mapping = new Dictionary<Tensor<double>, string>();
        foreach (var parameter in parameters)
        {
            var values = parameter.ToArray();
            var candidates = remaining.Where(p => p.Value.Length == values.Length &&
                p.Value.Zip(values, (a, b) => Math.Abs(a - b)).Max() < 1e-12).Select(p => p.Key).ToArray();
            Assert.Single(candidates);
            mapping.Add(parameter, candidates[0]); remaining.Remove(candidates[0]);
        }
        Assert.Empty(remaining);
        Tensor<double> Input(string name, params int[] shape) => new(shape, new Vector<double>(Read(name)));
        var spectrogram = Input("spectrogram", 2, 9, 7);
        var source = Input("source", 2, 3, 1); var target = Input("target", 2, 3, 1);
        var noise = Input("noise", 2, 4, 7); var mask = Input("mask", 2, 1, 7);
        using var tape = new GradientTape<double>();
        var result = model.Convert(spectrogram, mask, source, target, noise, .37);
        var outputs = new[] { result.Waveform, result.Posterior, result.Prior, result.Converted };
        string[] outputNames = { "waveform", "posterior", "prior", "converted" };
        Tensor<double>? loss = null;
        for (int index = 0; index < outputs.Length; index++)
        {
            var output = outputs[index]; Close(output, Read(outputNames[index]), outputNames[index]);
            var weights = new Tensor<double>(output.Shape.ToArray());
            for (int i = 0; i < weights.Length; i++) weights[i] = -.8 + 1.5 * i / (weights.Length - 1);
            var term = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(output, weights),
                Enumerable.Range(0, output.Rank).ToArray(), true), (index + 1) / 10.0);
            term = engine.Reshape(term, new[] { 1 });
            loss = loss is null ? term : engine.TensorAdd(loss, term);
        }
        Close(loss!, new[] { item.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(loss!, sources: parameters.Concat(new[] { spectrogram, source, target, noise }).ToList());
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.ContainsKey(parameter), "Missing gradient: " + mapping[parameter]);
            Close(gradients[parameter], Read("gradient." + mapping[parameter]), mapping[parameter]);
        }
        foreach (var pair in new[] { (spectrogram, "spectrogram"), (source, "source"), (target, "target"), (noise, "noise") })
            Close(gradients[pair.Item1], Read(pair.Item2 + "_gradient"), pair.Item2 + " gradient");
        Close(model.Flow(result.Prior, mask, source, true), result.Posterior.ToArray(), "source flow inverse");
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"OpenVoice {name} maximum error {error:R}.");
    }
}
