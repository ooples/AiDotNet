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

    private static void AssertAllFinite(Tensor<Half> values)
    {
        for (int j = 0; j < values.Length; j++)
            Assert.True(Half.IsFinite(values[j]), $"element {j} is {values[j]}");
    }

    private static Tensor<Half> HalfTensor(double[] values, params int[] shape)
    {
        var half = new Half[values.Length];
        for (int j = 0; j < values.Length; j++)
            half[j] = (Half)values[j];
        return new Tensor<Half>(half, shape);
    }

    // SCA (Mirjalili 2016, SCA.m): r1 = a - t a / T for t = 1..T, so the last iteration has r1 = 0 and the first
    // a - a / T. The 0-based loop used t = iter and never reached 0.
    [Fact(Timeout = 60000)]
    public async Task SineCosine_ExplorationScale_MatchesScaFromOneToT()
    {
        await Task.CompletedTask;
        Assert.Equal(0.0, AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.SineCosine<double>.ExplorationScale(99, 100, 2.0), 12);
        Assert.Equal(1.98, AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.SineCosine<double>.ExplorationScale(0, 100, 2.0), 12);
    }

    // SSA (Mirjalili et al. 2017, eq. 3.2): c1 = 2 exp(-(4 l / L)^2) with l starting at 1 (SSA.m), so the first
    // iteration uses 2 exp(-(4 / L)^2), not 2.
    [Fact(Timeout = 60000)]
    public async Task SalpSwarm_LeaderCoefficient_StartsAtIterationOne()
    {
        await Task.CompletedTask;
        Assert.Equal(2 * Math.Exp(-Math.Pow(4.0 / 50, 2)),
            AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.SalpSwarm<double>.LeaderCoefficient(0, 50), 12);
        Assert.Equal(2 * Math.Exp(-16.0),
            AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.SalpSwarm<double>.LeaderCoefficient(49, 50), 12);
    }

    // DA.m with a single neighbour keeps A = dX_i and C = 0. With s = a = c = 0.1, f = 1, w = 0.9 at t = 0:
    // S = -(0.6 - 0.5) = -0.1, A = (0.02, -0.01), F = 0.05, so dX = 0.1 S + 0.1 A + F + 0.9 dX = (0.06, 0.03).
    [Fact(Timeout = 60000)]
    public async Task Dragonfly_OneNeighbour_KeepsOwnStepAndNoCohesion()
    {
        await Task.CompletedTask;
        var positions = new Tensor<double>(new[] { 0.5, 0.5, 0.6, 0.6 }, new[] { 2, 2 });
        var steps = new Tensor<double>(new[] { 0.02, -0.01, 0.0, 0.0 }, new[] { 2, 2 });
        var food = new Tensor<double>(new[] { 0.55, 0.55 }, new[] { 2 });
        var enemy = new Tensor<double>(new[] { 5.0, 5.0 }, new[] { 2 });
        var weights = AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.DragonflyAlgorithm<double>.SwarmWeights.ForIteration(
            0, 10, new Tensor<double>(new[] { 0.5, 0.5, 0.5, 0.5 }, new[] { 4 }));
        AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.DragonflyAlgorithm<double>.MoveDragonfly(
            ref positions, ref steps, 0, food, enemy, weights, RandomHelper.CreateSeededRandom(1));
        Assert.Equal(0.56, positions[0, 0], 10);
        Assert.Equal(0.53, positions[0, 1], 10);
        Assert.Equal(0.06, steps[0, 0], 10);
        Assert.Equal(0.03, steps[0, 1], 10);
    }

    // DA.m's random walk draws rand per dimension for A, C and S: dX_j = A_j r_j0 + C_j r_j1 + S_j r_j2 + w dX_j.
    // Neighbours (0.6, 0.6) and (0.4, 0.45) of (0.5, 0.5) give A = (0.02, 0), C = (0, 0.025), S = (0, -0.05).
    [Fact(Timeout = 60000)]
    public async Task Dragonfly_RandomWalk_DrawsPerDimension()
    {
        await Task.CompletedTask;
        var positions = new Tensor<double>(new[] { 0.5, 0.5, 0.6, 0.6, 0.4, 0.45 }, new[] { 3, 2 });
        var steps = new Tensor<double>(new[] { 0.0, 0.0, 0.01, 0.02, 0.03, -0.02 }, new[] { 3, 2 });
        var far = new Tensor<double>(new[] { 5.0, 5.0 }, new[] { 2 });
        var weights = AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.DragonflyAlgorithm<double>.SwarmWeights.ForIteration(
            0, 10, new Tensor<double>(new[] { 0.5, 0.5, 0.5, 0.5 }, new[] { 4 }));
        AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.DragonflyAlgorithm<double>.MoveDragonfly(
            ref positions, ref steps, 0, far, far, weights, RandomHelper.CreateSeededRandom(7));
        var reference = RandomHelper.CreateSeededRandom(7);
        var r = new double[6];
        for (int k = 0; k < 6; k++)
            r[k] = reference.NextDouble();
        Assert.Equal(0.02 * r[0], steps[0, 0], 12);
        Assert.Equal(0.025 * r[4] - 0.05 * r[5], steps[0, 1], 12);
        Assert.Equal(0.5 + 0.02 * r[0], positions[0, 0], 12);
        Assert.Equal(0.5 + 0.025 * r[4] - 0.05 * r[5], positions[0, 1], 12);
    }

    // A feature picked in all 3000 subsamples has selection probability 1; counted in Half it stopped at 2048.
    [Fact(Timeout = 60000)]
    public async Task StabilitySelection_Half_CountsPast2048Bootstraps()
    {
        await Task.CompletedTask;
        var data = new AiDotNet.Tensors.LinearAlgebra.Matrix<Half>(4, 2);
        var target = new AiDotNet.Tensors.LinearAlgebra.Vector<Half>(4);
        for (int i = 0; i < 4; i++)
        {
            data[i, 0] = (Half)i;
            data[i, 1] = (Half)(3 - i);
            target[i] = (Half)i;
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Hybrid.StabilitySelection<Half>(
            nFeaturesToSelect: 1, nBootstraps: 3000, baseSelector: (_, _) => new[] { 0 }, randomState: 3);
        selector.Fit(data, target);
        var probabilities = selector.SelectionProbabilities;
        Assert.NotNull(probabilities);
        if (probabilities is not null)
        {
            Assert.Equal(1.0, (double)probabilities[0]);
            Assert.Equal(0.0, (double)probabilities[1]);
        }
    }

    // LIME's distance is the fraction of replaced features: 3000 of 3000 replaced is 1, not 2048 / 3000.
    [Fact(Timeout = 60000)]
    public async Task LIME_ReplacedFraction_Half_CountsPast2048()
    {
        await Task.CompletedTask;
        var fraction = LIMESelector<Half>.ReplacedFraction(new Tensor<Half>(new[] { 1, 3000 }));
        Assert.Equal(1.0, (double)fraction[0]);
    }

    // Box-Muller with a draw that rounds to 1 in Half: 1 - u1 = 0 is floored, so the normal is finite.
    [Fact(Timeout = 60000)]
    public async Task VAE_StandardNormal_Half_IsFiniteForADrawOfOne()
    {
        await Task.CompletedTask;
        var normal = AiDotNet.Preprocessing.FeatureSelection.Neural.VariationalAutoencoderSelector<Half>.StandardNormal(
            HalfTensor(new[] { 1.0, 0.25 }, 1, 2));
        AssertAllFinite(normal);
    }

    // (x - min) / (range + 1e-10) maps a constant column to 0; 1e-10 is 0 in Half, which gave 0 / 0 = NaN.
    [Fact(Timeout = 60000)]
    public async Task VAE_MinMaxScale_Half_MapsConstantColumnToZero()
    {
        await Task.CompletedTask;
        var scaled = AiDotNet.Preprocessing.FeatureSelection.Neural.VariationalAutoencoderSelector<Half>.MinMaxScale(
            HalfTensor(new[] { 3.0, 1.0, 3.0, 2.0, 3.0, 5.0 }, 3, 2));
        Assert.Equal(0.0, (double)scaled[0, 0]);
        Assert.Equal(0.0, (double)scaled[1, 0]);
        Assert.Equal(0.0, (double)scaled[2, 0]);
        Assert.Equal(0.25, (double)scaled[1, 1], 3);
        Assert.Equal(1.0, (double)scaled[2, 1], 3);
    }

    // Standardisation (also used by EmbeddedWrapperHybrid) maps a constant column to 0, not NaN, in Half.
    [Fact(Timeout = 60000)]
    public async Task StandardizeSample_Half_MapsConstantColumnToZero()
    {
        await Task.CompletedTask;
        var standardized = NeuralSelectorOps.StandardizeSample(HalfTensor(new[] { 2.0, 1.0, 2.0, 3.0 }, 2, 2));
        Assert.Equal(0.0, (double)standardized[0, 0]);
        Assert.Equal(0.0, (double)standardized[1, 0]);
        Assert.Equal(-0.7071, (double)standardized[0, 1], 3);
        Assert.Equal(0.7071, (double)standardized[1, 1], 3);
    }

    // Cuckoo's Box-Muller ln(u1 + 1e-10): with the guard 0 in Half a zero draw gave ln 0 and an infinite step.
    [Fact(Timeout = 60000)]
    public async Task Cuckoo_GaussianFromUniforms_Half_IsFiniteForAZeroDraw()
    {
        await Task.CompletedTask;
        var normal = AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.CuckooSearch<Half>.GaussianFromUniforms(
            HalfTensor(new[] { 0.0, 0.0 }, 1, 2), 0);
        AssertAllFinite(normal);
    }

    // Lee-Seung updates divide by (W^T W H + eps); on all-zero (clipped negative) data W and H collapse to 0 and
    // eps = 1e-10 is 0 in Half, so 0 / 0 made every score NaN. With a positive eps the scores are 0.
    [Fact(Timeout = 60000)]
    public async Task NMF_Half_ZeroDataGivesFiniteScores()
    {
        await Task.CompletedTask;
        var data = new AiDotNet.Tensors.LinearAlgebra.Matrix<Half>(4, 3);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                data[i, j] = (Half)(-1.0 - i - j);
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Projection.NMFBasedSelector<Half>(
            nFeaturesToSelect: 1, nComponents: 2, nIterations: 5);
        selector.Fit(data);
        var scores = selector.FeatureScores;
        Assert.NotNull(scores);
        if (scores is not null)
            for (int j = 0; j < 3; j++)
                Assert.Equal(0.0, (double)scores[j]);
    }

    // The concrete selector's Gumbel noise -log(-log(u + tiny) + tiny): tiny = 1e-20 is 0 in Half, so a draw of 1
    // gave +inf noise and a NaN softmax. The loss must stay finite.
    [Fact(Timeout = 60000)]
    public async Task ConcreteAutoencoder_Half_GumbelOfADrawOfOneIsFinite()
    {
        await Task.CompletedTask;
        var network = new AiDotNet.Preprocessing.FeatureSelection.Neural.ConcreteAutoencoderSelector<Half>.ConcreteNetwork
        {
            Logits = new Tensor<Half>(new[] { 1, 2 }),
            HiddenWeights = HalfTensor(new[] { 1.0 }, 1, 1),
            HiddenBias = new Tensor<Half>(new[] { 1 }),
            OutputWeights = HalfTensor(new[] { 1.0, 1.0 }, 1, 2),
            OutputBias = new Tensor<Half>(new[] { 2 }),
        };
        var loss = network.Loss(HalfTensor(new[] { 0.5, 0.25 }, 2), HalfTensor(new[] { 1.0, 0.0 }, 1, 2), 1.0);
        AssertAllFinite(loss);
    }

    // Tabu expiry iterations compared in Half: at iteration 2051 (2052 in Half) a feature tabu until 2052 was
    // already free and one tabu until 2051 still tabu. Kept in int the mask is exact.
    [Fact(Timeout = 60000)]
    public async Task Tabu_TabuMask_Half_ExactPast2048()
    {
        await Task.CompletedTask;
        var tabu = AiDotNet.Preprocessing.FeatureSelection.Optimization.TabuSearchSelector<Half>.TabuMask(
            new[] { 2052, 2051 }, 2051);
        Assert.Equal(1.0, (double)tabu[0]);
        Assert.Equal(0.0, (double)tabu[1]);
    }

    // Swapping into a subset of 3000 features with total score 0 and a new feature of score 1 scores 1 / 3000;
    // the subset size summed in Half stopped at 2048, giving 1 / 2048.
    [Fact(Timeout = 60000)]
    public async Task Tabu_BestSwap_Half_CountsSubsetPast2048()
    {
        await Task.CompletedTask;
        const int p = 3001;
        var current = new double[p];
        var scores = new double[p];
        for (int j = 0; j < 3000; j++)
            current[j] = 1;
        scores[3000] = 1;
        var move = AiDotNet.Preprocessing.FeatureSelection.Optimization.TabuSearchSelector<Half>.BestSwap(
            HalfTensor(current, p), HalfTensor(scores, p), new Tensor<Half>(new[] { p }), new Tensor<Half>(new[] { 1 }),
            out var moveScore);
        Assert.NotNull(move);
        Assert.Equal(3000, move?.Add);
        Assert.True(Math.Abs((double)moveScore[0] - 1.0 / 3000) < 2e-6, $"move score {(double)moveScore[0]}");
    }

    // Quantum annealing energy: -sum corr + 0.1 |count - target|; 3000 selected with target 3000 is 0 (the Half sum
    // stopped at 2048, adding a 95.2 penalty).
    [Fact(Timeout = 60000)]
    public async Task QuantumAnnealing_ComputeEnergy_Half_CountsPast2048()
    {
        await Task.CompletedTask;
        var ones = new double[3000];
        for (int j = 0; j < 3000; j++)
            ones[j] = 1;
        var energy = AiDotNet.Preprocessing.FeatureSelection.Quantum.QuantumAnnealingSelector<Half>.ComputeEnergy(
            new Tensor<Half>(new[] { 3000 }), HalfTensor(ones, 3000), 3000);
        Assert.Equal(0.0, (double)energy[0]);
    }

    // Selection frequencies of the quantum selectors and stability selection: 3000 of 3000 is 1, and 70000 of
    // 70000 is 1 although both exceed Half's largest value 65504.
    [Fact(Timeout = 60000)]
    public async Task Frequency_Half_ExactPastHalfRange()
    {
        await Task.CompletedTask;
        var frequency = PopulationSearchOps.Frequency<Half>(new Tensor<int>(new[] { 3000, 70000, 0 }, new[] { 3 }), 3000);
        Assert.Equal(1.0, (double)frequency[0]);
        Assert.Equal(0.0, (double)frequency[2]);
        var wide = PopulationSearchOps.Frequency<Half>(new Tensor<int>(new[] { 70000 }, new[] { 1 }), 70000);
        Assert.Equal(1.0, (double)wide[0]);
    }

    // Gumbel noise of the exponential mechanism: Half draws >= 1 - 2^-12 rounded to 1 and gave +inf noise, which
    // forced those features into the private selection. Drawn in double, every value is finite.
    [Fact(Timeout = 60000)]
    public async Task DifferentialPrivacy_GumbelNoise_Half_IsFinite()
    {
        await Task.CompletedTask;
        var noise = AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<Half>.GumbelNoise(
            RandomHelper.CreateSeededRandom(11), 50000);
        AssertAllFinite(noise);
    }
}
