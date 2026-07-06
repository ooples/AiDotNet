using System;
using System.Collections.Generic;
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

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// ConfigureTrainingGroups end-to-end: the facade's supervised path runs one model.Train call per
/// query group per epoch instead of a single pooled fit — the shape learning-to-rank consumers need
/// (one coherent cross-section, e.g. one trading date, per Train call). These tests prove (a) the
/// grouped loop genuinely trains the network with row/target alignment preserved through the group
/// slicing, and (b) malformed group indices fail loudly with an actionable message.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class TrainingGroupsFacadeTests
{
    private const int N = 100;          // total rows fed to the loader
    private const int TrainRows = 70;   // facade's internal split: 0.7 train

    // Reproducible init seed — see BuildNet for why this is set on the architecture (not a
    // [ThreadStatic] ambient seed).
    private const int InitSeed = 1675;

    private static (Tensor<double> X, Tensor<double> Y) BuildLinearData()
    {
        // y = 2x + 1 — trivially learnable, but ONLY if the grouped slices keep each row paired
        // with its own target. A row/target misalignment in GatherRows would destroy the fit.
        var x = new double[N];
        var y = new double[N];
        for (int i = 0; i < N; i++)
        {
            double xi = i / (double)N;
            x[i] = xi;
            y[i] = (2.0 * xi) + 1.0;
        }

        return (
            new Tensor<double>(new[] { N, 1 }, new Vector<double>(x)),
            new Tensor<double>(new[] { N }, new Vector<double>(y)));
    }

    private static NeuralNetwork<double> BuildNet()
    {
        // Pin per-layer weight init so the "grouped loop genuinely trains" convergence assertions are
        // reproducible. The net is otherwise built from a non-reproducible, order-dependent init
        // (RandomHelper.ThreadSafeRandom, advanced by sibling tests on the same worker thread); a
        // poorly-conditioned draw fails to converge in the optimizer's bounded iteration budget, so the
        // test flaked under parallel scheduling (#1675). The seed lives on the architecture object
        // (not a [ThreadStatic] ambient seed) because the facade builds the net via an async pipeline
        // whose continuation may run on a different thread — the architecture carries the seed across it.
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 1,
            outputSize: 1)
        {
            RandomSeed = InitSeed,
        };
        return new NeuralNetwork<double>(architecture);
    }

    private static IReadOnlyList<IReadOnlyList<int>> PartitionTrainingRows(int groupSize)
    {
        // Partition the TRAINING rows (post-split indices 0..TrainRows-1) into contiguous groups —
        // each standing in for one date's cross-section.
        var groups = new List<IReadOnlyList<int>>();
        for (int start = 0; start < TrainRows; start += groupSize)
        {
            groups.Add(Enumerable.Range(start, Math.Min(groupSize, TrainRows - start)).ToArray());
        }

        return groups;
    }

    [Fact]
    public async Task Grouped_training_learns_with_row_target_alignment_preserved()
    {
        var (x, y) = BuildLinearData();

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildNet())
            .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
            .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                model: null,
                options: new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { MaxIterations = 60 }))
            .ConfigureTrainingGroups(PartitionTrainingRows(groupSize: 10))
            .BuildAsync();

        // Probe across the input range: a trained net must be (1) non-constant and (2) monotonic
        // increasing on y = 2x + 1. An untrained/collapsed net fails (1); scrambled group slices
        // (misaligned rows/targets) fail (2). Statistical invariants, not exact values — NN training
        // is stochastic.
        var probes = new[] { 0.1, 0.3, 0.5, 0.7, 0.9 };
        var preds = new double[probes.Length];
        for (int i = 0; i < probes.Length; i++)
        {
            var p = result.Predict(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { probes[i] })));
            preds[i] = p.Data.Span[0];
        }

        Assert.True(preds.Max() - preds.Min() > 0.3,
            $"Net is near-constant over the input range — grouped loop did not train. preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");

        int rising = 0;
        for (int i = 1; i < preds.Length; i++)
        {
            if (preds[i] > preds[i - 1])
            {
                rising++;
            }
        }

        Assert.True(rising >= 3,
            $"Predictions not increasing in x — row/target alignment broken in group slicing. preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");
    }

    [Fact]
    public async Task Grouped_split_is_order_preserving_so_groups_reference_original_leading_rows()
    {
        // Rows 0..69 follow y = +2x + 1; rows 70..99 follow y = -2x + 1 (CONFLICTING slope).
        // With training groups configured the split must be UNSHUFFLED: the training partition is
        // exactly the leading 70 rows (pure +2x), so the net learns an increasing function. A
        // shuffled split would mix conflicting-tail rows into training and the slope washes out —
        // this is the property that lets LTR callers do leak-free temporal splits (sort by date,
        // most recent dates land in val/test).
        var x = new double[N];
        var y = new double[N];
        for (int i = 0; i < N; i++)
        {
            double xi = i / (double)N;
            x[i] = xi;
            y[i] = i < TrainRows ? (2.0 * xi) + 1.0 : (-2.0 * xi) + 1.0;
        }

        var xT = new Tensor<double>(new[] { N, 1 }, new Vector<double>(x));
        var yT = new Tensor<double>(new[] { N }, new Vector<double>(y));

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildNet())
            .ConfigureDataLoader(DataLoaders.FromTensors(xT, yT))
            .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                model: null,
                options: new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { MaxIterations = 250 }))
            .ConfigureTrainingGroups(PartitionTrainingRows(groupSize: 10))
            .BuildAsync();

        // Probe within the training block's input range [0, 0.7): the function must be increasing
        // (slope +2). If shuffled rows leaked the conflicting tail into training, the learned slope
        // collapses toward 0 and this ordering breaks down.
        var probes = new[] { 0.05, 0.2, 0.35, 0.5, 0.65 };
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

        Assert.True(preds.Max() - preds.Min() > 0.3 && rising >= 3,
            "Grouped training did not learn the leading block's +2x slope — the split is not order-preserving. " +
            $"preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");
    }

    [Fact]
    public async Task Out_of_range_group_index_throws_actionable_error()
    {
        var (x, y) = BuildLinearData();

        // Index N is valid in the ORIGINAL data frame of reference but not in the post-split
        // training set — exactly the mistake the message must explain.
        var badGroups = new IReadOnlyList<int>[] { new[] { 0, 1, N } };

        var ex = await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
                .ConfigureModel(BuildNet())
                .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
                .ConfigureTrainingGroups(badGroups)
                .BuildAsync());

        Assert.Contains("training partition", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Empty_group_list_is_rejected_at_configuration_time()
    {
        Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
                .ConfigureTrainingGroups(Array.Empty<IReadOnlyList<int>>()));
    }
}
