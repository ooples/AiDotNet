using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Classification;
using AiDotNet.Audio.Features;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The Audio Spectrogram Transformer against the official code (tools/reference-data/ast_reference.py): two small
/// networks (the paper's 10x10 strides and rectangular 8x12 ones) before and after an SGD step on the BCE loss, the Kaldi
/// filterbank with the code's Hanning and the paper's Hamming window, and the paper's ImageNet DeiT adaptation.
/// </summary>
public class AstReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "ast_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    private static Func<string, int[], double[]> Reader(JsonElement state) => (name, shape) =>
    {
        var v = Values(state.GetProperty(name));
        Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
        return v;
    };

    private static Tensor<double> Bce(IEngine engine, Tensor<double> logits, Tensor<double> targets)
    {
        // softplus(x) − x·t, averaged.
        var terms = engine.TensorSubtract(engine.Softplus(logits), engine.TensorMultiply(logits, targets));
        return engine.ReduceMean(terms, new[] { 0, 1 }, false);
    }

    [Theory(Timeout = 300000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task Network_MatchesTheOfficialCode_AndAfterOneSgdStep(int index)
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture().GetProperty("networks")[index];
        var configuration = new AstConfiguration
        {
            MelBins = k.GetProperty("mel").GetInt32(), Frames = k.GetProperty("frames").GetInt32(),
            FrequencyStride = k.GetProperty("fstride").GetInt32(), TimeStride = k.GetProperty("tstride").GetInt32(),
            Dim = 48, Depth = 2, Heads = 4, Classes = 5,
        };
        var layers = new List<LayerBase<double>>();
        var net = new AstNetwork<double>(engine, layers, configuration);
        net.LoadTorchWeights(Reader(k.GetProperty("state")));
        var input = new Tensor<double>(new[] { 2, configuration.Frames, configuration.MelBins }, new Vector<double>(Values(k.GetProperty("input"))));
        var targets = new Tensor<double>(new[] { 2, 5 }, new Vector<double>(Values(k.GetProperty("targets"))));

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> representation, logits, loss;
        using (var tape = new GradientTape<double>())
        {
            representation = net.Encode(input);
            logits = net.Classify(representation);
            loss = Bce(engine, logits, targets);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose(Values(k.GetProperty("representation")), representation, 1e-9, "representation");
        AssertClose(Values(k.GetProperty("logits")), logits, 1e-9, "logits");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-12, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Bce(engine, net.Forward(input), targets)[0];
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-12, $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");
    }

    [Theory]
    [InlineData("hanning", WindowType.Hann)]
    [InlineData("hamming", WindowType.Hamming)]
    public void KaldiFbank_MatchesTorchaudio(string window, WindowType type)
    {
        var k = Fixture().GetProperty("fbank");
        var waveform = Values(k.GetProperty("waveform"));
        var fbank = new KaldiFbank<double>(AiDotNetEngine.Current, 16000, 128, type);
        Assert.Equal(k.GetProperty("frames").GetInt32(), fbank.Frames(waveform.Length));
        var features = fbank.Forward(new Tensor<double>(new[] { waveform.Length }, new Vector<double>(waveform)));
        Assert.Equal(new[] { k.GetProperty("frames").GetInt32(), 128 }, features.Shape.ToArray());
        AssertClose(Values(k.GetProperty(window)), features, 1e-9, window);
    }

    [Fact]
    public void ImageNetAdaptation_FollowsThePaper()
    {
        var k = Fixture().GetProperty("adaptation");
        var configuration = new AstConfiguration { MelBins = 26, Frames = 66, Dim = 48, Depth = 2, Heads = 4, Classes = 5 };
        Assert.Equal(k.GetProperty("f_dim").GetInt32(), configuration.FrequencyPatches);
        Assert.Equal(k.GetProperty("t_dim").GetInt32(), configuration.TimePatches);
        var net = new AstNetwork<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration);
        var state = k.GetProperty("state");
        net.LoadImageNetDeiT(Reader(state), k.GetProperty("grid").GetInt32(), new Random(1));

        var proj = Values(k.GetProperty("proj"));                                            // [48, 1, 16, 16]
        var weights = net.PatchProjection.GetWeights();                                      // [256, 48]
        for (int o = 0; o < 48; o++)
            for (int i = 0; i < 256; i++)
                Assert.True(Math.Abs(proj[o * 256 + i] - weights[i, o]) < 1e-12, $"proj[{o},{i}]");
        var position = net.PositionTable.Forward(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 })));
        AssertClose(Values(k.GetProperty("pos_embed")), position, 1e-12, "pos_embed");
        var token = net.ClassToken.Forward(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 })));
        AssertClose(Values(state.GetProperty("cls_token")), token, 1e-15, "cls_token");
    }
}
