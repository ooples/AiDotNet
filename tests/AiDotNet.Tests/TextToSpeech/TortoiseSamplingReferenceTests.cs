using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class TortoiseSamplingReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void CompleteSpacedSampler_MatchesUnchangedAuthorLoopWithReplayedNoise(int index)
    {
        using var weights = Read("tortoise_diffusion_reference.json");
        using var reference = Read("tortoise_sampling_reference.json");
        var fixture = weights.RootElement;
        var root = reference.RootElement;
        var expected = root.GetProperty("cases")[index];
        var options = new TortoiseTTSOptions
        {
            DiffusionDim = 32, DiffusionHeads = 4, DiffusionLayers = 1,
            MelChannels = 4, LLMDim = 8, CodebookSize = 8, DropoutRate = 0,
            DiffusionTrainingSteps = 4000, DiffusionInferenceSteps = 3,
            SampleRate = root.GetProperty("sample_rate").GetInt32(),
            TokenizerSampleRate = root.GetProperty("tokenizer_sample_rate").GetInt32(),
            ClassifierFreeGuidanceScale = expected.GetProperty("guidance").GetDouble()
        };
        var engine = AiDotNetEngine.Current;
        var model = new TortoiseDiffusion<double>(engine, new List<LayerBase<double>>(), options);
        model.LoadTorchWeights((name, shape) =>
        {
            Assert.Equal(shape, fixture.GetProperty("weight_shapes").GetProperty(name)
                .EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(fixture.GetProperty("weights").GetProperty(name));
        });
        var latents = new Tensor<double>(new[] { 3, 8 }, new Vector<double>(Values(fixture.GetProperty("latents"))));
        var mel = engine.TensorTranspose(new Tensor<double>(new[] { 4, 12 },
            new Vector<double>(Values(fixture.GetProperty("mel")))));
        var random = new RecordedGaussian(expected);
        var actual = model.Sample(latents, mel, random);
        Assert.Equal(new[] { root.GetProperty("frames").GetInt32(), 4 }, actual.Shape.ToArray());
        var target = Values(expected.GetProperty("mel"));
        double error = actual.ToArray().Zip(target, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-4, $"Original float32 multi-step mel maximum error: {error:R}");
        Assert.Equal(random.DrawCount, random.Consumed);
    }

    // Invert the native paired Box-Muller transform to replay only the author's
    // random draws. The native network, schedule, guidance and sampler run normally.
    private sealed class RecordedGaussian : Random
    {
        private readonly Queue<double> _uniforms = new();
        internal int Consumed { get; private set; }
        internal int DrawCount { get; }

        internal RecordedGaussian(JsonElement fixture)
        {
            var blocks = new List<double[]> { Values(fixture.GetProperty("initial")) };
            blocks.AddRange(fixture.GetProperty("noises").EnumerateArray().Select(Values));
            foreach (var values in blocks)
                for (int i = 0; i < values.Length; i += 2)
                {
                    double x = values[i], y = i + 1 < values.Length ? values[i + 1] : 0;
                    _uniforms.Enqueue(1 - Math.Exp(-(x * x + y * y) / 2));
                    double angle = Math.Atan2(y, x);
                    _uniforms.Enqueue((angle < 0 ? angle + 2 * Math.PI : angle) / (2 * Math.PI));
                }
            DrawCount = _uniforms.Count;
        }

        public override double NextDouble()
        {
            Consumed++;
            Assert.NotEmpty(_uniforms);
            return _uniforms.Dequeue();
        }
    }

    private static double[] Values(JsonElement array) => array.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static JsonDocument Read(string file) => JsonDocument.Parse(File.ReadAllText(Path.Combine(
        AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", file)));
}
