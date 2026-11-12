using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceObjectiveReferenceTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompletePaperObjective_MatchesIndependentNeuralForwardsAndEveryGradient(bool versionTwo)
    {
        AiDotNetEngine.ResetToCpu();
        string directory = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData");
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "openvoice_objective.json")));
        var item = metadata.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("zero_condition").GetBoolean() == versionTwo);
        string label = item.GetProperty("label").GetString()!;
        using var stream = File.OpenRead(Path.Combine(directory, "openvoice_objective.safetensors"));
        using var reader = new BinaryReader(stream);
        long length = reader.ReadInt64(); using var header = JsonDocument.Parse(reader.ReadBytes(checked((int)length)));
        double[] Read(string name, int[]? shape = null)
        {
            var entry = header.RootElement.GetProperty(label + "." + name);
            Assert.Equal("F64", entry.GetProperty("dtype").GetString());
            if (shape is not null) Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            var offsets = entry.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            stream.Position = 8 + length + offsets[0]; var result = new double[checked((int)((offsets[1] - offsets[0]) / 8))];
            for (int i = 0; i < result.Length; i++) result[i] = reader.ReadDouble();
            return result;
        }
        Tensor<double> Input(string name, params int[] shape) => new(shape, new Vector<double>(Read(name)));
        var engine = AiDotNetEngine.Current;
        var native = new OpenVoiceNative<double>(engine, 16, 2, 4, 4, 3, versionTwo,
            new[] { 2 }, new[] { 4 }, 8, new[] { 3 }, new[] { new[] { 1, 3, 5 } });
        native.LoadTorch((name, shape) => Read("weight.native." + name, shape),
            name => header.RootElement.TryGetProperty(label + ".weight.native." + name, out _) ? Read("weight.native." + name) : null);
        var options = new OpenVoiceTrainingOptions
        {
            IpaVocabularySize = 17, IpaFilterChannels = 8, IpaHeads = 1, IpaLayers = 2, IpaKernel = 3,
            IpaDropout = 0, SegmentFrames = 8, MelChannels = 4, DiscriminatorWidthDivisor = 64
        };
        var graph = new OpenVoiceTrainingGraph<double>(engine, native, options, 4, 22050);
        graph.LoadAuxiliaryTorch((name, shape) => Read("weight.prior." + name, shape), name => Read("weight.discriminator." + name));
        graph.NoiseForAudit = shape => Input("noise", shape); graph.SegmentStartForAudit = 2;
        var sample = new TtsTrainingSample<double> { Tokens = Input("tokens", 4), Audio = Input("audio", 64) };
        var data = graph.Prepare(sample); var draw = graph.Sample(data);
        void Close(IEnumerable<double> actual, double[] expected, string name)
        {
            var values = actual.ToArray(); Assert.Equal(expected.Length, values.Length);
            double error = values.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < 2e-8, name + " error " + error.ToString("R"));
        }
        Close(data.Spectrum.ToArray(), Read("spectrum"), "recording STFT");
        Close(graph.Generate(data, draw).ToArray(), Read("waveform"), "direct reconstruction");

        double[] Transform(string name, double[] values, bool transpose)
        {
            if (!transpose) return values;
            var shape = header.RootElement.GetProperty(label + ".weight." + name).GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            var result = new double[values.Length];
            for (int output = 0; output < shape[0]; output++) for (int input = 0; input < shape[1]; input++)
                result[input * shape[0] + output] = values[output * shape[1] + input];
            return result;
        }
        var weightNames = header.RootElement.EnumerateObject().Where(x => x.Name.StartsWith(label + ".weight.", StringComparison.Ordinal))
            .Select(x => x.Name.Substring((label + ".weight.").Length)).ToArray();
        var remaining = new HashSet<string>(weightNames);
        var owned = TapeTrainingStep<double>.CollectParameters(graph.GeneratorLayers.Concat(graph.DiscriminatorLayers).Cast<ILayer<double>>().ToList());
        var mapping = new Dictionary<Tensor<double>, (string Name, bool Transpose)>();
        foreach (var parameter in owned)
        {
            var candidates = new List<(string Name, bool Transpose)>();
            foreach (string name in remaining)
            {
                var values = Read("weight." + name); if (values.Length != parameter.Length) continue;
                if (values.Zip(parameter.ToArray(), (a, b) => Math.Abs(a - b)).Max() < 1e-12) candidates.Add((name, false));
                else
                {
                    var shape = header.RootElement.GetProperty(label + ".weight." + name).GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                    if ((shape.Length == 2 || shape.Length == 3 && shape[2] == 1) &&
                        Transform(name, values, true).Zip(parameter.ToArray(), (a, b) => Math.Abs(a - b)).Max() < 1e-12) candidates.Add((name, true));
                }
            }
            Assert.Single(candidates); mapping.Add(parameter, candidates[0]); remaining.Remove(candidates[0].Name);
        }
        Assert.Empty(remaining);
        foreach (bool discriminator in new[] { false, true })
        {
            var layers = discriminator ? graph.DiscriminatorLayers : graph.GeneratorLayers;
            var parameters = TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
            Tensor<double> generated;
            using (new NoGradScope<double>()) generated = graph.Generate(data, draw);
            using var tape = new GradientTape<double>();
            var loss = discriminator ? graph.DiscriminatorLoss(graph.RealSegment(data, draw), generated) : graph.GeneratorLoss(data, draw, true, false);
            Close(loss.ToArray(), new[] { item.GetProperty(discriminator ? "discriminator_loss" : "generator_loss").GetDouble() }, discriminator ? "D loss" : "G loss");
            var gradients = tape.ComputeGradients(loss, sources: parameters);
            foreach (var parameter in parameters)
            {
                var (name, transpose) = mapping[parameter];
                string key = discriminator ? "discriminator_gradient." + name.Substring("discriminator.".Length) : "gradient." + name;
                Assert.True(gradients.TryGetValue(parameter, out var gradient), "Missing gradient " + name);
                Close(gradient!.ToArray(), Transform(name, Read(key), transpose), name);
            }
        }
    }
}
