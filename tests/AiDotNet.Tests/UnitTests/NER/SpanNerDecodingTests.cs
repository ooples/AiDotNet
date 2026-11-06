using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NER.SpanBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NER;

/// <summary>
/// Span-based NER keeps <c>PredictLabels</c>'s one-label-per-token contract over the paper heads' span grids, returns
/// nested spans through <c>PredictSpans</c>, and trains custom per-token heads on per-token labels (#2294 review).
/// </summary>
public class SpanNerDecodingTests
{
    private const int Tokens = 6, Width = 16, Labels = 9;

    private static SpanBasedNEROptions Options() => new()
    {
        HiddenDimension = Width, NumAttentionHeads = 2, NumTransformerLayers = 1, IntermediateDimension = 32, NumLabels = Labels,
        MaxSequenceLength = 12, MaxSpanLength = 3, SpanEmbeddingDimension = 8, DropoutRate = 0.0, LearningRate = 1e-3,
    };

    private static NeuralNetworkArchitecture<double> Architecture(List<ILayer<double>>? layers = null) =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.MultiClassClassification, inputSize: Width, outputSize: Labels, layers: layers);

    private static Tensor<double> Sentence(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var x = new Tensor<double>(new[] { Tokens, Width });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, rng.NextDouble() * 2 - 1);
        return x;
    }

    public static IEnumerable<object[]> Models()
    {
        yield return new object[] { "Biaffine" };
        yield return new object[] { "PURE" };
        yield return new object[] { "SpERT" };
        yield return new object[] { "Pyramid" };
        yield return new object[] { "Triaffine" };
        yield return new object[] { "W2NER" };
    }

    private static SpanBasedNERBase<double> Create(string name, List<ILayer<double>>? layers = null) => name switch
    {
        "Biaffine" => new BiaffineNER<double>(Architecture(layers), Options()),
        "PURE" => new PURENER<double>(Architecture(layers), Options()),
        "SpERT" => new SpERTNER<double>(Architecture(layers), Options()),
        "Pyramid" => new PyramidNER<double>(Architecture(layers), Options()),
        "Triaffine" => new TriaffineNER<double>(Architecture(layers), Options()),
        _ => new W2NER<double>(Architecture(layers), Options()),
    };

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Models))]
    public async Task PredictLabels_IsOneLabelPerToken_AndAgreesWithTheSpans(string name)
    {
        await Task.Yield();
        using var model = Create(name);
        var input = Sentence(3);
        model.SetTrainingMode(true);
        var labels = model.PredictLabels(input);
        Assert.Equal(labels.ToArray(), model.PredictLabels(input).ToArray());
        Assert.Equal(new[] { Tokens }, labels.Shape.ToArray());
        var spans = model.PredictSpans(input);
        foreach (var span in spans)
        {
            Assert.InRange(span.Start, 0, Tokens - 1);
            Assert.InRange(span.End, span.Start, Tokens - 1);
            Assert.InRange(span.Label, 1, Labels - 1);
        }
        // Every entity token lies inside a predicted span of its label.
        for (int t = 0; t < Tokens; t++)
        {
            int label = (int)labels[t];
            if (label != 0) Assert.Contains(spans, s => s.Start <= t && t <= s.End && s.Label == label);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CustomTokenHead_TrainsOnPerTokenLabels()
    {
        await Task.Yield();
        var layers = new List<ILayer<double>> { new DenseLayer<double>(Labels, (IActivationFunction<double>)new IdentityActivation<double>()) };
        using var model = Create("Biaffine", layers);
        var input = Sentence(5);
        var target = new Tensor<double>(new[] { Tokens }, new Vector<double>(new double[] { 1, 0, 2, 2, 0, 1 }));
        // A per-token head trains on the per-token labels; a span-grid target of 36 would not match its 6 rows.
        for (int step = 0; step < 3; step++) model.Train(input, target);
        Assert.Equal(new[] { Tokens }, model.PredictLabels(input).Shape.ToArray());
        Assert.Throws<NotSupportedException>(() => model.PredictSpans(input));
    }
}
