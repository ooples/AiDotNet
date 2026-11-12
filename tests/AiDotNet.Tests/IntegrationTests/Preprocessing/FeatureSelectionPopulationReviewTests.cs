using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the population and model-agnostic selector helpers: Half overflow and counting limits,
/// positions compared as T, the masked top-k fill value and Kernel SHAP weights for wide feature sets.
/// </summary>
public class FeatureSelectionPopulationReviewTests
{
    // |r| = 1 for a feature equal to the target, 0 and 4 alternating over 300 rows: sxx = syy = 1200, and
    // sqrt(sxx * syy) overflowed Half (1.44e6 > 65504), which made r 0.
    [Fact(Timeout = 60000)]
    public async Task AbsTargetCorrelations_Half_DoesNotOverflowTheNormaliser()
    {
        await Task.CompletedTask;
        const int n = 300;
        var data = new Half[n, 1];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)(4 * (i % 2));
            target[i] = (Half)(4 * (i % 2));
        }

        var r = PopulationSearchOps.AbsTargetCorrelations(new Matrix<Half>(data), new Vector<Half>(target), (Half)0);

        Assert.Equal(1.0, (double)r[0], 2);
    }

    // A subset of all 3000 features against a target count of 3000 has no count penalty. Counted in Half the
    // subset size stopped at 2048, so the penalty was |2048 - 3000| = 952.
    [Fact(Timeout = 60000)]
    public async Task SubsetFitness_Half_CountsTheSubsetPast2048()
    {
        await Task.CompletedTask;
        const int p = 3000;
        var ones = new Half[p];
        for (int j = 0; j < p; j++)
            ones[j] = (Half)1;
        var mask = new Tensor<Half>(ones, new[] { 1, p });
        var scores = new Tensor<Half>(new[] { p });

        var fitness = PopulationSearchOps.SubsetFitness(mask, scores, p, (Half)1, (Half)0);

        Assert.Equal(0.0, (double)fitness[0], 1);
    }

    // One-point crossover at 2049 takes positions 0..2048 from the first parent. As Half, position 2049 rounds to
    // 2048 and was taken from the first parent too.
    [Fact(Timeout = 60000)]
    public async Task OnePointCrossover_Half_SplitsExactlyPastPosition2048()
    {
        await Task.CompletedTask;
        const int p = 3000;
        var ones = new Half[p];
        for (int j = 0; j < p; j++)
            ones[j] = (Half)1;

        var child = ParetoOps.OnePointCrossover(new Tensor<Half>(ones, new[] { p }), new Tensor<Half>(new[] { p }), 2049);

        Assert.Equal(1.0, (double)child[2048]);
        Assert.Equal(0.0, (double)child[2049]);
        Assert.Equal(0.0, (double)child[2050]);
    }

    // Top-2 of the masked entries {1, 2}. The old fill (min - 1) for the unmasked entry 0 equals 3e8 in float, so
    // entry 0 tied with entry 1 and won the tie by its lower index.
    [Fact(Timeout = 60000)]
    public async Task TopKWhere_RanksOnlyTheMaskedEntries()
    {
        await Task.CompletedTask;
        var scores = new Tensor<float>(new[] { 3e8f, 3e8f, 5e8f }, new[] { 3 });
        var mask = new Tensor<float>(new[] { 0f, 1f, 1f }, new[] { 3 });

        var selected = FeatureSelectionTensorOps.TopKWhere(scores, mask, 2);

        Assert.Equal(new[] { 1, 2 }, selected);
    }

    // For f(x) = w . x and one reference point, phi_j = w_j (x_j - b_j) exactly. With 200 features C(200, s)
    // overflows float for the coalition sizes drawn (about 100), which zeroed every kernel weight and left all the
    // attribution on the last feature.
    [Fact(Timeout = 120000)]
    public async Task KernelShap_Float_RecoversLinearShapleyValuesFor200Features()
    {
        await Task.CompletedTask;
        const int p = 200;
        const int coalitions = 800;
        var w = new float[p];
        var x = new float[p];
        for (int j = 0; j < p; j++)
        {
            w[j] = 1 + j % 3;
            x[j] = 1;
        }

        var random = RandomHelper.CreateSeededRandom(11);
        var draws = new float[coalitions * p];
        for (int k = 0; k < draws.Length; k++)
            draws[k] = (float)random.NextDouble();

        Func<Tensor<float>, Tensor<float>> predict = rows =>
        {
            var values = rows.ToArray();
            int m = rows.Shape[0];
            var output = new float[m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < p; j++)
                    output[i] += w[j] * values[i * p + j];
            return new Tensor<float>(output, new[] { m });
        };

        var phi = SHAPSelector<float>.KernelShap(
            new Tensor<float>(x, new[] { p }), new Tensor<float>(new[] { p }),
            new Tensor<float>(draws, new[] { coalitions, p }), predict).ToArray();

        for (int j = 0; j < p; j++)
            Assert.Equal(w[j], phi[j], 1);
    }

    // The feature-count objective of a subset selecting all 3000 features is 3000. Summed in Half the count
    // stopped at 2048, so wide subsets looked cheaper than they are on the Pareto front.
    private static Tensor<Half> AllSelected(int p)
    {
        var ones = new Half[p];
        for (int j = 0; j < p; j++)
            ones[j] = (Half)1;
        return new Tensor<Half>(ones, new[] { 1, p });
    }

    [Fact(Timeout = 60000)]
    public async Task NSGAII_Half_CountsSelectedFeaturesPast2048()
    {
        await Task.CompletedTask;
        var (_, count) = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.NSGAIISelector<Half>.EvaluateObjectives(
            AllSelected(3000), new Tensor<Half>(new[] { 3000 }));
        Assert.Equal(3000.0, (double)count[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task SPEAII_Half_CountsSelectedFeaturesPast2048()
    {
        await Task.CompletedTask;
        var (_, count) = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.SPEAIISelector<Half>.EvaluateObjectives(
            AllSelected(3000), new Tensor<Half>(new[] { 3000 }));
        Assert.Equal(3000.0, (double)count[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task MOEAD_Half_CountsSelectedFeaturesPast2048()
    {
        await Task.CompletedTask;
        var (_, count) = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.MOEADSelector<Half>.EvaluateObjectives(
            AllSelected(3000), new Tensor<Half>(new[] { 3000 }));
        Assert.Equal(3000.0, (double)count[0]);
    }
}
