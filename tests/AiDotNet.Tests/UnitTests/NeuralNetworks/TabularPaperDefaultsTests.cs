using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Paper defaults and parameter registration of the tabular models: TabPFN (Hollmann et al. 2023), TabR
/// (Gorishniy et al. 2023) and TabNet's ghost batch norm (Arik and Pfister 2021).
/// </summary>
public class TabularPaperDefaultsTests
{
    private static Tensor<double> ContextFeatures() =>
        new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2, 0.6, 0.9 }));

    [Fact]
    public void TabPFN_DefaultsToNoPositionalEncoding_SoTheContextIsASet()
    {
        Assert.False(new TabPFNOptions<double>().UsePositionalEncoding);
        Assert.Equal(4, new TabPFNOptions<double>().NumHeads);

        // Without positions, permuting the context rows (with their labels) cannot change a prediction.
        var model = new TabPFNRegression<double>(2, 1, new TabPFNOptions<double>
        {
            EmbeddingDimension = 8, NumLayers = 1, FeedForwardMultiplier = 2, OutputHeadDimensions = new[] { 4 }
        });
        var query = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.5, 0.5 }));

        model.SetContext(ContextFeatures(), new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 })));
        double original = model.Predict(query)[0];
        model.SetContext(
            new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.6, 0.9, 0.4, 0.7, 1.1, 0.2 })),
            new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.9, 0.1, 0.5 })));
        double permuted = model.Predict(query)[0];

        Assert.Equal(original, permuted, 10);
    }

    [Fact]
    public void TabPFN_HeadsSplitTheEmbedding()
    {
        TabPFNOptions<double> Options(int heads) => new TabPFNOptions<double>
        {
            EmbeddingDimension = 8, NumHeads = heads, NumLayers = 1, FeedForwardMultiplier = 2,
            OutputHeadDimensions = new[] { 4 }
        };
        var single = new TabPFNRegression<double>(2, 1, Options(1));
        var multi = new TabPFNRegression<double>(2, 1, Options(4));

        // Heads re-slice the same projections, so the layouts agree and the same weights can be compared.
        var parameters = single.GetParameters();
        Assert.Equal(parameters.Length, multi.GetParameters().Length);
        multi.SetParameters(parameters);

        var labels = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 }));
        single.SetContext(ContextFeatures(), labels);
        multi.SetContext(ContextFeatures(), labels);
        var query = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.5, 0.5, 1.0, -0.3 }));

        Assert.False(single.Predict(query).ToArray().SequenceEqual(multi.Predict(query).ToArray()),
            "Four heads with their own softmax must differ from one head over the whole embedding.");
        Assert.Throws<ArgumentException>(() => new TabPFNRegression<double>(2, 1, Options(3)));
    }

    [Fact]
    public void TabR_DefaultsToUnnormalizedKeys()
    {
        Assert.False(new TabROptions<double>().NormalizeEmbeddings);
    }

    [Fact]
    public void GhostBatchNorm_ScaleAndShiftAreRegisteredAndGetTapeGradients()
    {
        var bn = new GhostBatchNormalization<double>(2, virtualBatchSize: 128);
        Assert.Equal(4, bn.ParameterCount);
        Assert.Equal(2, bn.GetTrainableParameters().Count);
        var gamma = bn.GetTrainableParameters()[0];
        var beta = bn.GetTrainableParameters()[1];

        // Columns [0, 2] and [1, 5]: batch means 1 and 3, variances 1 and 4, so the normalized values are
        // -/+ 1 / sqrt(1 + eps / var). With loss sum(c * y): dL/dbeta = sum_b c, dL/dgamma = sum_b c * xhat.
        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 1.0, 2.0, 5.0 }));
        var weights = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.5, -1.0, 2.0, 3.0 }));
        var engine = AiDotNetEngine.Current;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var loss = engine.ReduceSum(engine.TensorMultiply(bn.Forward(input), weights), new[] { 0, 1 }, false);
            gradients = tape.ComputeGradients(loss, new[] { gamma, beta });
        }

        double n0 = 1.0 / Math.Sqrt(1.0 + 1e-5 / 1.0);
        double n1 = 1.0 / Math.Sqrt(1.0 + 1e-5 / 4.0);
        Assert.Equal(-0.5 * n0 + 2.0 * n0, gradients[gamma][0], 9);
        Assert.Equal(1.0 * n1 + 3.0 * n1, gradients[gamma][1], 9);
        Assert.Equal(2.5, gradients[beta][0], 9);
        Assert.Equal(2.0, gradients[beta][1], 9);
    }

    [Fact]
    public void AttentiveTransformer_CountsItsGhostBatchNorm()
    {
        var layer = new AttentiveTransformerLayer<double>(3, 4);
        var processed = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));
        var prior = new Tensor<double>(new[] { 2, 4 });
        prior.Fill(1.0);
        _ = layer.Forward(processed, prior);

        // The 3 -> 4 projection (12 weights, 4 biases) plus the norm's gamma and beta (4 each).
        Assert.Equal(16 + 8, layer.ParameterCount);
    }
}
