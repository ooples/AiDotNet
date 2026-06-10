using System;
using System.Threading.Tasks;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro + regression for #43: a NeuralNetwork&lt;double&gt; trained with PairwiseRankingLoss must learn to
/// produce VARYING scores ordered by the relevance target — not collapse to a constant. The GlobalScore
/// ranker observed a constant output (spread &lt; 1e-9) for this exact configuration, forcing a regression
/// fallback. The loss math is verified correct, so this isolates the NeuralNetwork↔ranking-loss training
/// integration (notably whether ComputeTapeLoss's surrogate gradient actually drives the optimizer).
/// </summary>
public class PairwiseRankingNetTests
{
    [Fact(Timeout = 180000)]
    public async Task PairwiseRankingNet_double_learns_to_rank_and_does_not_collapse()
    {
        await Task.CompletedTask; // suite requires Timeout-marked tests to be async

        const int n = 30, f = 3;
        var rng = new Random(7);
        var xData = new double[n * f];
        var yData = new double[n];
        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble(), x1 = rng.NextDouble(), x2 = rng.NextDouble();
            xData[i * f + 0] = x0; xData[i * f + 1] = x1; xData[i * f + 2] = x2;
            yData[i] = 2.0 * x0 + 1.0 * x1 - 0.5 * x2; // relevance the ranker must order by
        }

        // Standardize features (the ranker scales features before training).
        for (int j = 0; j < f; j++)
        {
            double m = 0; for (int i = 0; i < n; i++) m += xData[i * f + j]; m /= n;
            double v = 0; for (int i = 0; i < n; i++) { var d = xData[i * f + j] - m; v += d * d; } v = Math.Sqrt(v / n);
            if (v <= 0) v = 1;
            for (int i = 0; i < n; i++) xData[i * f + j] = (xData[i * f + j] - m) / v;
        }

        var x = new Tensor<double>(new[] { n, f }, new Vector<double>(xData));
        var y = new Tensor<double>(new[] { n, 1 }, new Vector<double>(yData));
        var net = new NeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: f, outputSize: 1),
            lossFunction: new PairwiseRankingLoss<double>(0.0));
        for (int e = 0; e < 400; e++)
        {
            net.Train(x, y);
        }

        var pred = net.Predict(x).ToVector();
        double min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < pred.Length; i++) { min = Math.Min(min, pred[i]); max = Math.Max(max, pred[i]); }
        Assert.True(max - min > 1e-3, $"ranking net collapsed to a constant: spread={max - min}");

        double corr = Pearson(pred, yData);
        Assert.True(corr > 0.5, $"ranking net did not learn the order: corr={corr}");
    }

    private static double Pearson(Vector<double> a, double[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < n; i++) { double da = a[i] - ma, db = b[i] - mb; cov += da * db; va += da * da; vb += db * db; }
        double d = Math.Sqrt(va * vb);
        return d > 0 ? cov / d : 0;
    }
}
