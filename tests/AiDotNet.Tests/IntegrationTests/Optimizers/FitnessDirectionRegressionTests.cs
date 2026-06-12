using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression guard for the inverted default-fitness-direction bug: RSquaredFitnessCalculator (the
/// DEFAULT calculator on OptimizationAlgorithmOptions) declared isHigherScoreBetter: false while
/// GetFitnessScore returns the RAW R² with no internal negation (despite docs claiming "the
/// calculator handles this internally"). IsBetterFitness therefore preferred LOWER R², so every
/// optimizer's UpdateBestSolution kept the WORST iterate ever evaluated — typically the untrained
/// pre-training baseline from PrepareAndEvaluateSolution. Observed: a facade NN trained 250 epochs
/// on a perfectly-correlated y = 0.58x returned a model ANTI-correlated with the target (the random
/// init happened to slope negative; per-epoch fitness improved monotonically -4.38 → -4.00 while
/// the returned BestSolution stayed frozen at the baseline).
/// </summary>
public class FitnessDirectionRegressionTests
{
    [Fact]
    public void RSquared_calculators_prefer_higher_scores()
    {
        var r2 = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(r2.IsHigherScoreBetter, "R² is a higher-is-better metric");
        Assert.True(r2.IsBetterFitness(0.9, 0.1), "R² 0.9 must beat 0.1");
        Assert.False(r2.IsBetterFitness(-4.0, 0.5), "negative R² must not beat positive");

        var adj = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(adj.IsHigherScoreBetter, "Adjusted R² is a higher-is-better metric");
        Assert.True(adj.IsBetterFitness(0.9, 0.1));
    }

    [Fact]
    public void Error_calculators_still_prefer_lower_scores()
    {
        var mse = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(mse.IsHigherScoreBetter);
        Assert.True(mse.IsBetterFitness(0.1, 0.9), "MSE 0.1 must beat 0.9");
    }

    [Fact]
    public async Task Facade_optimizer_returns_an_improved_iterate_not_the_baseline()
    {
        // y = 0.5x on pre-scaled inputs: trivially learnable. Under the inverted comparator the
        // build returns the UNTRAINED baseline (random slope, ~50% chance anti-correlated); with
        // the fix the returned model must track the target's increasing relationship.
        const int n = 100;
        var x = new double[n];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (i - 50) / 29.0;
            y[i] = 0.5 * x[i];
        }

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple,
                inputSize: 1,
                outputSize: 1)))
            .ConfigureDataLoader(DataLoaders.FromTensors(
                new Tensor<double>(new[] { n, 1 }, new Vector<double>(x)),
                new Tensor<double>(new[] { n, 1 }, new Vector<double>(y))))
            .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                model: null,
                options: new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { MaxIterations = 250 }))
            .BuildAsync();

        var probes = new[] { -1.5, -0.75, 0.0, 0.75, 1.5 };
        var preds = probes
            .Select(p => result.Predict(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { p })))[0])
            .ToArray();

        int rising = 0;
        for (int i = 1; i < preds.Length; i++)
        {
            if (preds[i] > preds[i - 1])
            {
                rising++;
            }
        }

        // The inverted comparator returns the baseline (random sign — frequently DECREASING).
        // A correctly-selected trained model must be increasing overall and non-trivial. (rising >= 2,
        // not 4: tiny ReLU nets legitimately saturate flat on part of the range while still fitting
        // the other side — the bug signature is a DECREASING/flat-everywhere baseline.)
        Assert.True(rising >= 2 && preds[4] > preds[0] && preds[4] - preds[0] > 0.2,
            "Optimizer returned a non-improved (baseline-like) model — best-solution selection direction regressed. " +
            $"preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");
    }
}
