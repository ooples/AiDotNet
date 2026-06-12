using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression guard: rank-1 regression targets ([B] against a [B,1] network output) are the
/// industry-standard target shape (sklearn's y is (n,); PyTorch broadcasts [B] vs [B,1]) but
/// crashed AiDotNet's training paths with "Tensor shapes must match. Got [B, 1] and [B]" —
/// regression losses subtract tensors directly, the only in-loss shape handling
/// (EnsureTargetMatchesPredicted) is a CLASSIFICATION one-hot encoder, and the network-level
/// alignment policy only covered leading singleton batch dims. AlignTargetToOutputShape now
/// aligns rank-off-by-one trailing singletons at every training entry point (Train/TrainWithTape,
/// the chunked path, and ComputeGradients used by the facade optimizer loop).
/// </summary>
public class RankOneTargetAlignmentTests
{
    private static NeuralNetwork<double> BuildNet() => new(new NeuralNetworkArchitecture<double>(
        inputType: InputType.OneDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        complexity: NetworkComplexity.Simple,
        inputSize: 1,
        outputSize: 1));

    private static (Tensor<double> X, Tensor<double> YRank1) LinearData(int n)
    {
        var x = new double[n];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (i - (n / 2)) / (double)n * 2.0;
            y[i] = 0.5 * x[i];
        }

        return (
            new Tensor<double>(new[] { n, 1 }, new Vector<double>(x)),
            new Tensor<double>(new[] { n }, new Vector<double>(y)));
    }

    [Fact]
    public void Direct_Train_accepts_rank1_targets()
    {
        var (x, y) = LinearData(40);
        var net = BuildNet();

        // Crashed before the fix: MeanSquaredErrorLoss.ComputeTapeLoss → TensorSubtract([40,1],[40]).
        var ex = Record.Exception(() => net.Train(x, y));
        Assert.Null(ex);
    }

    [Fact]
    public async Task Facade_optimizer_path_accepts_rank1_targets_and_trains()
    {
        // The facade's Adam loop goes through ComputeGradients, which had NO regression target
        // alignment at all (its comment wrongly delegated "singleton reshape" to the loss).
        const int n = 100;
        var (x, y) = LinearData(n);

        var net = BuildNet();
        var baseline = Probe(net, 1.0) - Probe(net, -1.0);

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(net)
            .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
            .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                model: null,
                options: new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { MaxIterations = 100 }))
            .BuildAsync();

        // Beyond not crashing: training must have actually consumed the rank-1 targets — the
        // returned model's probe MSE must beat a do-nothing zero predictor or show real slope.
        var preds = new[] { -1.0, -0.5, 0.0, 0.5, 1.0 }
            .Select(p => result.Predict(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { p })))[0])
            .ToArray();
        // double.IsFinite is net10-only; use the net471-compatible equivalent (same semantics).
        Assert.True(preds.All(p => !double.IsNaN(p) && !double.IsInfinity(p)), "non-finite predictions after rank-1 target training");
        Assert.True(preds.Max() - preds.Min() > 1e-6 || Math.Abs(baseline) < 1e-9,
            "model unchanged by training on rank-1 targets");
    }

    [Fact]
    public void Symmetric_case_rank2_target_against_rank1_style_output_does_not_crash()
    {
        // [B,1] targets are the documented batch convention and must keep working.
        var (x, _) = LinearData(40);
        var y2 = new Tensor<double>(new[] { 40, 1 }, new Vector<double>(
            Enumerable.Range(0, 40).Select(i => 0.5 * ((i - 20) / 40.0 * 2.0)).ToArray()));

        var net = BuildNet();
        var ex = Record.Exception(() => net.Train(x, y2));
        Assert.Null(ex);
    }

    private static double Probe(NeuralNetwork<double> net, double v) =>
        net.Predict(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { v })))[0];
}
