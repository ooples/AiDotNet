using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class UniAudioCodecTrainingReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void ResidualQuantizer_TrainingGradientsAndEmaIncludingDeadCodesMatchAuthor(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "uniaudio_rvq_training_reference.json")));
        var fixture = document.RootElement.GetProperty("cases")[index];
        var quantizer = new AiDotNet.Audio.Codecs.ResidualVectorQuantizerLayer<double>(4, 3, 8,
            kmeansIterations: 0, commitmentAsMean: true, expireCodesAfterEma: true);
        for (int q = 0; q < 3; q++)
        {
            var state = fixture.GetProperty("before")[q];
            quantizer.LoadCodebook(q, Values(state.GetProperty("embed")), Values(state.GetProperty("embed_avg")),
                Values(state.GetProperty("cluster_size")), true);
        }
        quantizer.SetTrainingMode(true);
        var shape = Shape(fixture.GetProperty("shape"));
        var input = Tensor(fixture.GetProperty("input"), shape);
        var target = Tensor(fixture.GetProperty("target"), shape);
        using var tape = new GradientTape<double>();
        var output = quantizer.Forward(input);
        var engine = AiDotNetEngine.Current;
        var loss = engine.TensorAdd(MeanSquare(engine.TensorSubtract(output, target)), quantizer.CommitmentLoss!);
        Close(output, fixture.GetProperty("output"));
        Assert.Equal(fixture.GetProperty("commitment").GetDouble(), quantizer.CommitmentLoss![0], 12);
        Assert.Equal(fixture.GetProperty("loss").GetDouble(), loss[0], 12);
        Close(tape.ComputeGradients(loss, sources: new[] { input })[input], fixture.GetProperty("gradient"));
        Assert.Equal(Values(fixture.GetProperty("codes")), quantizer.LastCodes!.Cast<int>().Select(x => (double)x));
        foreach (var pair in new[] { ("embed", "_embed"), ("embed_avg", "_embedAverage"), ("cluster_size", "_clusterSize") })
        {
            var states = (Tensor<double>[])quantizer.GetType().GetField(pair.Item2,
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(quantizer)!;
            for (int q = 0; q < 3; q++) Close(states[q], fixture.GetProperty("after")[q].GetProperty(pair.Item1));
        }
    }

    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "uniaudio_codec_training_reference.json")));
    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static int[] Shape(JsonElement value) => value.EnumerateArray().Select(x => x.GetInt32()).ToArray();
    private static Tensor<double> Tensor(JsonElement value, int[] shape) => new(shape, new Vector<double>(Values(value)));
    private static void Close(Tensor<double> actual, JsonElement expected, double tolerance = 1e-9)
    {
        var values = Values(expected);
        Assert.Equal(values.Length, actual.Length);
        double error = actual.ToArray().Zip(values, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(TtsPaperNumerics.IsFinite(error) && error < tolerance, $"Maximum error {error:R} exceeds {tolerance:R}.");
    }
    private static Tensor<double> MeanSquare(Tensor<double> value)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceMean(engine.TensorMultiply(value, value), Enumerable.Range(0, value.Rank).ToArray(), false);
    }
    private static UniAudioCodecTrainingOptions TinyRecipe() => new()
    {
        DiscriminatorHops = [4], DiscriminatorChannels = [64],
        FullFftSizes = [32, 64], FullWindowSizes = [24, 48], FullHopSizes = [8, 16],
        SubFftSizes = [8, 16], SubWindowSizes = [6, 12], SubHopSizes = [2, 4],
        Subbands = 6,
    };

    [Fact]
    public void SevenLayerCritic_AllOutputsAndParameterInputGradientsMatchAuthor()
    {
        using var document = Read(); var fixture = document.RootElement.GetProperty("critic");
        var owner = new List<LayerBase<double>>();
        var critic = new UniAudioFrequencyDiscriminator<double>(AiDotNetEngine.Current, owner, 64);
        critic.LoadTorchWeights((name, shape) =>
        {
            Assert.Equal(shape, Shape(fixture.GetProperty("shapes").GetProperty(name)));
            return Values(fixture.GetProperty("weights").GetProperty(name));
        }, "");
        var input = Tensor(fixture.GetProperty("input"), Shape(fixture.GetProperty("input_shape")));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(owner.Cast<ILayer<double>>().ToList(), -1);
        using var tape = new GradientTape<double>();
        var output = critic.Forward(input);
        Close(output.Score, fixture.GetProperty("score"));
        var loss = MeanSquare(output.Score);
        for (int i = 0; i < output.Features.Length; i++)
        {
            Close(output.Features[i], fixture.GetProperty("features")[i]);
            loss = AiDotNetEngine.Current.TensorAdd(loss,
                AiDotNetEngine.Current.TensorMultiplyScalar(MeanSquare(output.Features[i]), .03));
        }
        Assert.Equal(fixture.GetProperty("loss").GetDouble(), loss[0], 10);
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { input }).ToList());
        Close(gradients[input], fixture.GetProperty("input_gradient"));
        Assert.Equal(21, parameters.Count);
        for (int i = 0; i < 7; i++)
        {
            string prefix = $"discriminator.{i}.{(i == 0 ? 1 : 2)}.";
            string[] names = ["weight_v", "weight_g", "bias"];
            for (int p = 0; p < 3; p++) Close(gradients[parameters[i * 3 + p]],
                fixture.GetProperty("gradients").GetProperty(prefix + names[p]));
        }
    }

    [Fact]
    public void LinearAndLogMelFrontend_OutputAndWaveformGradientMatchAuthor()
    {
        using var document = Read(); var fixture = document.RootElement.GetProperty("frontend");
        var audio = Tensor(fixture.GetProperty("audio"), [128]);
        using var tape = new GradientTape<double>();
        var output = new UniAudioCriticSpectrogram<double>(AiDotNetEngine.Current, 16000, 4).Forward(audio);
        Close(output, fixture.GetProperty("output"));
        var loss = MeanSquare(output);
        Assert.Equal(fixture.GetProperty("loss").GetDouble(), loss[0], 11);
        Close(tape.ComputeGradients(loss, sources: new[] { audio })[audio], fixture.GetProperty("input_gradient"));
    }

    private static UniAudioCriticOutput<double> Scores(JsonElement fixture, string prefix, List<Tensor<double>>? sources = null)
    {
        var scores = fixture.GetProperty(prefix + "_scores").EnumerateArray().Select((x, i) =>
            Tensor(x, Shape(fixture.GetProperty("score_shapes")[i]))).ToArray();
        var features = fixture.GetProperty(prefix + "_features").EnumerateArray().Select((group, i) =>
            group.EnumerateArray().Select((x, j) => Tensor(x, Shape(fixture.GetProperty("feature_shapes")[i][j]))).ToArray()).ToArray();
        sources?.AddRange(scores);
        if (sources is not null) foreach (var group in features) sources.AddRange(group);
        return new UniAudioCriticOutput<double>(scores, features);
    }

    [Fact]
    public void GeneratorFullAndSubbandSpectralAndGanLoss_AllInputGradientsMatchAuthor()
    {
        using var document = Read(); var fixture = document.RootElement.GetProperty("objective");
        var gan = new UniAudioCodecGan<double>(AiDotNetEngine.Current, 16000, TinyRecipe());
        var real = Tensor(fixture.GetProperty("real"), [480]);
        var fake = Tensor(fixture.GetProperty("fake"), [480]);
        var sources = new List<Tensor<double>> { fake };
        var realOutput = Scores(fixture, "real"); var fakeOutput = Scores(fixture, "fake", sources);
        using var tape = new GradientTape<double>();
        var reconstruction = gan.ReconstructionObjective(real, fake);
        Assert.Equal(fixture.GetProperty("reconstruction_loss").GetDouble(), reconstruction[0], 9);
        var loss = AiDotNetEngine.Current.TensorAdd(reconstruction, gan.AdversarialGeneratorObjective(realOutput, fakeOutput));
        Assert.Equal(fixture.GetProperty("generator_loss").GetDouble(), loss[0], 9);
        var gradients = tape.ComputeGradients(loss, sources: sources);
        Close(gradients[fake], fixture.GetProperty("fake_gradient"), 1e-8);
        for (int i = 0; i < fakeOutput.Scores.Length; i++)
        {
            Close(gradients[fakeOutput.Scores[i]], fixture.GetProperty("score_gradients")[i], 1e-12);
            for (int j = 0; j < fakeOutput.Features[i].Length; j++) Close(gradients[fakeOutput.Features[i][j]],
                fixture.GetProperty("feature_gradients")[i][j], 1e-12);
        }
    }

    [Fact]
    public void DiscriminatorLeastSquaresLoss_AndEveryFakeScoreGradientMatchAuthor()
    {
        using var document = Read(); var fixture = document.RootElement.GetProperty("objective");
        var gan = new UniAudioCodecGan<double>(AiDotNetEngine.Current, 16000, TinyRecipe());
        var real = Scores(fixture, "real"); var fake = Scores(fixture, "fake");
        using var tape = new GradientTape<double>();
        var loss = gan.DiscriminatorObjective(real, fake);
        Assert.Equal(fixture.GetProperty("discriminator_loss").GetDouble(), loss[0], 12);
        var gradients = tape.ComputeGradients(loss, sources: fake.Scores);
        for (int i = 0; i < fake.Scores.Length; i++)
            Close(gradients[fake.Scores[i]], fixture.GetProperty("discriminator_score_gradients")[i], 1e-12);
    }
}
