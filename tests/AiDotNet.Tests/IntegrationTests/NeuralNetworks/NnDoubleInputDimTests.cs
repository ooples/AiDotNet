using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression tests for the input-dimension-collapse bug in LayerHelper: the default/Bayesian/DBN layer
/// builders sized the first layer from GetInputShape()[0] (the batch/sample axis = 1 for tabular data)
/// instead of CalculatedInputSize (the per-sample feature count), so a NeuralNetwork built from a
/// [samples × features] dataset consumed ONE input value — collapsing to a near-constant output and, via
/// the facade, throwing "Feature index N exceeds the input dimension 1". These assert the net both
/// (a) does NOT collapse and (b) actually LEARNS the relationship, directly and through the facade.
/// </summary>
public class NnDoubleInputDimTests
{
    private static (Tensor<double> X, Tensor<double> Y, Matrix<double> Xm, Vector<double> Yv, double VarY) MakeData()
    {
        const int n = 200, f = 3;
        var rng = new Random(42);
        var xData = new double[n * f];
        var yData = new double[n];
        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble(), x1 = rng.NextDouble(), x2 = rng.NextDouble();
            xData[i * f + 0] = x0; xData[i * f + 1] = x1; xData[i * f + 2] = x2;
            yData[i] = 2.0 * x0 + 3.0 * x1 - 1.5 * x2; // depends on ALL features
        }

        // Standardize features + target per-column (z-score) — exactly what NeuralNetworkForecaster does
        // via StandardScaler; unscaled inputs/targets + a default LR diverge, which is a data-prep issue,
        // not the NN bug under test. After this, Var(y) == 1, so a learning net must drive MSE well below 1.
        for (int j = 0; j < f; j++)
        {
            double m = 0; for (int i = 0; i < n; i++) m += xData[i * f + j]; m /= n;
            double v = 0; for (int i = 0; i < n; i++) { var d = xData[i * f + j] - m; v += d * d; } v = Math.Sqrt(v / n);
            if (v <= 0) v = 1;
            for (int i = 0; i < n; i++) xData[i * f + j] = (xData[i * f + j] - m) / v;
        }
        double meanY = 0; for (int i = 0; i < n; i++) meanY += yData[i]; meanY /= n;
        double sdY = 0; for (int i = 0; i < n; i++) { var d = yData[i] - meanY; sdY += d * d; } sdY = Math.Sqrt(sdY / n);
        if (sdY <= 0) sdY = 1;
        for (int i = 0; i < n; i++) yData[i] = (yData[i] - meanY) / sdY;

        var xm = new Matrix<double>(n, f);
        for (int i = 0; i < n; i++) for (int j = 0; j < f; j++) xm[i, j] = xData[i * f + j];
        double varY = 1.0; // standardized target
        return (new Tensor<double>(new[] { n, f }, new Vector<double>(xData)),
                new Tensor<double>(new[] { n, 1 }, new Vector<double>(yData)),
                xm, new Vector<double>(yData), varY);
    }

    private static (double Spread, double Mse) Eval(Vector<double> pred, Vector<double> y)
    {
        double min = double.MaxValue, max = double.MinValue, sse = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            min = Math.Min(min, pred[i]); max = Math.Max(max, pred[i]);
            var e = pred[i] - y[i]; sse += e * e;
        }
        return (max - min, sse / pred.Length);
    }

    [Fact(Timeout = 180000)]
    public async Task DirectNn_double_learns_and_does_not_collapse()
    {
        await Task.CompletedTask; // suite requires Timeout-marked tests to be async
        var (x, y, _, yv, varY) = MakeData();
        var net = new NeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 3, outputSize: 1),
            lossFunction: new MeanSquaredErrorLoss<double>());
        // 2000 full-batch steps, not 300: NeuralNetwork<T>'s default optimizer is
        // deliberately conservative (AMSGrad-Adam, LR 5e-4 — halved from the Adam
        // paper default to fix the #1332 MoreData drift), and Adam's per-step
        // parameter displacement is bounded by the LR, so 300 steps move each
        // weight at most ~0.15 — mathematically short of the ~1-magnitude weights
        // this standardized linear map needs. The collapse bug under test is
        // step-count independent; 2000 steps keeps the learning assertion honest
        // against the intentional default-LR choice.
        for (int e = 0; e < 2000; e++) net.Train(x, y);

        var (spread, mse) = Eval(net.Predict(x).ToVector(), yv);
        Assert.True(spread > 1e-3, $"output collapsed: spread={spread}");
        // A constant predictor's MSE == Var(y). A net that actually learned must beat that comfortably.
        Assert.True(mse < 0.5 * varY, $"net did not learn: mse={mse} vs Var(y)={varY}");
    }

    [Fact(Timeout = 180000)]
    public async Task FacadeNn_double_multifeature_no_dim_error_and_learns()
    {
        // NeuralNetwork<T> is IFullModel<T, Tensor<T>, Tensor<T>> (Tensor-based), so the facade is the
        // Tensor-generic builder — this is the path the cross-sectional ranker used when it hit
        // "Feature index N exceeds the input dimension 1".
        var (x, y, _, yv, _) = MakeData();
        var net = new NeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 3, outputSize: 1),
            lossFunction: new MeanSquaredErrorLoss<double>());

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(net)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        var (spread, _) = Eval(result.Predict(x).ToVector(), yv);
        Assert.True(spread > 1e-3, $"facade NN collapsed: spread={spread}");
    }
}
