using System;
using System.Linq;
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
    // Half.IsFinite does not exist on net471, which this project also targets.
    private static bool HalfIsFinite(Half value) => !float.IsNaN((float)value) && !float.IsInfinity((float)value);

    // |r| = 1 for a feature equal to the target, 0 and 4 alternating over 300 rows: sxx = syy = 1200, and
    // sqrt(sxx * syy) overflowed Half (1.44e6 > 65504), which made r 0.
    [Fact(Timeout = 60000)]
    public async Task AbsTargetCorrelations_Half_DoesNotOverflowTheNormaliser()
    {
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
        var (_, count) = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.NSGAIISelector<Half>.EvaluateObjectives(
            AllSelected(3000), new Tensor<Half>(new[] { 3000 }));
        Assert.Equal(3000.0, (double)count[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task SPEAII_Half_CountsSelectedFeaturesPast2048()
    {
        await Task.Yield();
        var (_, count) = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.SPEAIISelector<Half>.EvaluateObjectives(
            AllSelected(3000), new Tensor<Half>(new[] { 3000 }));
        Assert.Equal(3000.0, (double)count[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task MOEAD_Half_CountsSelectedFeaturesPast2048()
    {
        await Task.Yield();
        var (_, count) = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.MOEADSelector<Half>.EvaluateObjectives(
            AllSelected(3000), new Tensor<Half>(new[] { 3000 }));
        Assert.Equal(3000.0, (double)count[0]);
    }

    private static void AssertAllFinite(Tensor<Half> values)
    {
        for (int j = 0; j < values.Length; j++)
            Assert.True(HalfIsFinite(values[j]), $"element {j} is {values[j]}");
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
        await Task.Yield();
        Assert.Equal(0.0, AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.SineCosine<double>.ExplorationScale(99, 100, 2.0), 12);
        Assert.Equal(1.98, AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.SineCosine<double>.ExplorationScale(0, 100, 2.0), 12);
    }

    // SSA (Mirjalili et al. 2017, eq. 3.2): c1 = 2 exp(-(4 l / L)^2) with l starting at 1 (SSA.m), so the first
    // iteration uses 2 exp(-(4 / L)^2), not 2.
    [Fact(Timeout = 60000)]
    public async Task SalpSwarm_LeaderCoefficient_StartsAtIterationOne()
    {
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
        var fraction = LIMESelector<Half>.ReplacedFraction(new Tensor<Half>(new[] { 1, 3000 }));
        Assert.Equal(1.0, (double)fraction[0]);
    }

    // Box-Muller with a draw that rounds to 1 in Half: 1 - u1 = 0 is floored, so the normal is finite.
    [Fact(Timeout = 60000)]
    public async Task VAE_StandardNormal_Half_IsFiniteForADrawOfOne()
    {
        await Task.Yield();
        var normal = AiDotNet.Preprocessing.FeatureSelection.Neural.VariationalAutoencoderSelector<Half>.StandardNormal(
            HalfTensor(new[] { 1.0, 0.25 }, 1, 2));
        AssertAllFinite(normal);
    }

    // (x - min) / (range + 1e-10) maps a constant column to 0; 1e-10 is 0 in Half, which gave 0 / 0 = NaN.
    [Fact(Timeout = 60000)]
    public async Task VAE_MinMaxScale_Half_MapsConstantColumnToZero()
    {
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
        var normal = AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.CuckooSearch<Half>.GaussianFromUniforms(
            HalfTensor(new[] { 0.0, 0.0 }, 1, 2), 0);
        AssertAllFinite(normal);
    }

    // Lee-Seung updates divide by (W^T W H + eps); on all-zero (clipped negative) data W and H collapse to 0 and
    // eps = 1e-10 is 0 in Half, so 0 / 0 made every score NaN. With a positive eps the scores are 0.
    [Fact(Timeout = 60000)]
    public async Task NMF_Half_ZeroDataGivesFiniteScores()
    {
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
        var noise = AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<Half>.GumbelNoise(
            RandomHelper.CreateSeededRandom(11), 50000);
        AssertAllFinite(noise);
    }

    // Two identical columns x = 1..4 with y = 2x: (D'D + 1e-6 I) c = D'y gives c = [1, 1] (60 c = 60 - 1e-6 c).
    // In Half the 1e-6 ridge rounded away against the Gram entries 30, so the system was exactly singular.
    [Fact(Timeout = 60000)]
    public async Task RidgeCoefficients_Half_DuplicateColumnsKeepTheRidge()
    {
        await Task.Yield();
        var design = HalfTensor(new double[] { 1, 1, 2, 2, 3, 3, 4, 4 }, 4, 2);
        var y = HalfTensor(new double[] { 2, 4, 6, 8 }, 4);

        var coefficients = SubsetRegressionOps.RidgeCoefficients(design, y, 1e-6);

        Assert.Equal(1.0, (double)coefficients[0], 3);
        Assert.Equal(1.0, (double)coefficients[1], 3);
    }

    // y = (-300, 300) with zero predictions about center 0: SS_res = SS_tot = 180,000, so R^2 = 0. Both Half sums
    // overflowed and inf / inf gave NaN.
    [Fact(Timeout = 60000)]
    public async Task RSquared_Half_SquaredErrorsPast65504()
    {
        await Task.Yield();
        var y = HalfTensor(new double[] { -300, 300 }, 2);

        var r2 = SubsetRegressionOps.RSquared(y, new Tensor<Half>(new[] { 2 }), new Tensor<Half>(new[] { 1 }), out bool valid);

        Assert.True(valid);
        Assert.Equal(0.0, (double)r2[0], 6);
    }

    // 4096 rows of x = +-1 and y = 20 x: one sweep gives beta = (mean(x y) - alpha) / mean(x^2) = 20 - 0.01. The
    // Half correlation sum 81,920 overflowed before the 1 / n scaling.
    [Fact(Timeout = 60000)]
    public async Task LassoCoordinateDescent_Half_CorrelationSumPast65504()
    {
        await Task.Yield();
        const int n = 4096;
        var x = new double[n];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = i % 2 == 0 ? 1 : -1;
            y[i] = 20 * x[i];
        }

        var beta = AiDotNet.Preprocessing.FeatureSelection.Hybrid.EmbeddedWrapperHybridSelector<Half>
            .LassoCoordinateDescent(HalfTensor(x, n, 1), HalfTensor(y, n), 0.01, 1);

        Assert.Equal(19.99, (double)beta[0], 0.02);
    }

    // The scout replaces the exhausted source and, scoring 0.8 against the retained best 0.5, becomes the best.
    // Before, the scout was never compared with the best, so a scout found in the last cycle was lost.
    [Fact(Timeout = 60000)]
    public async Task ArtificialBeeColony_ScoutBetterThanTheBestBecomesTheBest()
    {
        await Task.Yield();
        var foodSources = new Tensor<double>(new[] { 0.9, 0.1, 0.2, 0.3 }, new[] { 2, 2 });
        var fitness = new Tensor<double>(new[] { 0.5, 0.1 }, new[] { 2 });
        var trials = new[] { 0, 3 };
        var best = PopulationSearchOps.Row(foodSources, 0);
        var bestFitness = PopulationSearchOps.At(fitness, 0);
        var scout = new Tensor<double>(new[] { 0.7, 0.6 }, new[] { 1, 2 });

        AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.ArtificialBeeColony<double>.ScoutPhase(
            ref foodSources, ref fitness, trials, 1, scout, _ => new Tensor<double>(new[] { 0.8 }, new[] { 1 }),
            ref best, ref bestFitness);

        Assert.Equal(0.8, bestFitness[0], 12);
        Assert.Equal(new[] { 0.7, 0.6 }, best.ToArray());
        Assert.Equal(0.8, fitness[1], 12);
        Assert.Equal(0, trials[1]);
    }

    // Two fireflies over a target-equal feature and an uncorrelated one, one generation with alpha = 1: the result
    // must score at least as well as the best initial firefly (the initial population is the first draws of the
    // seeded stream). A firefly with no brighter peer moves randomly and can lose fitness, and the old code kept
    // only the final population.
    [Fact(Timeout = 120000)]
    public async Task FireflyAlgorithm_KeepsTheBestEvaluatedPosition()
    {
        await Task.Yield();
        const int n = 40;
        var data = new Matrix<double>(n, 2);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = i % 2;
            data[i, 1] = (i / 2) % 2;
            target[i] = i % 2;
        }

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, 0.0);
        for (int seed = 0; seed < 30; seed++)
        {
            var initial = PopulationSearchOps.Uniform<double>(RandomHelper.CreateSeededRandom(seed), 2, 2);
            double initialBest = PopulationSearchOps.ThresholdFitness(initial, correlations, 1, 0.1).ToArray().Max();

            var selector = new AiDotNet.Preprocessing.FeatureSelection.Metaheuristic.FireflyAlgorithm<double>(
                nFeaturesToSelect: 1, populationSize: 2, maxIterations: 1, alpha: 1.0, randomState: seed);
            selector.Fit(data, target);

            Assert.NotNull(selector.FeatureScores);
            var final = new Tensor<double>(selector.FeatureScores.ToArray(), new[] { 1, 2 });
            Assert.True(PopulationSearchOps.ThresholdFitness(final, correlations, 1, 0.1)[0] >= initialBest,
                $"seed {seed}: the returned position scores below the best initial firefly {initialBest}");
        }
    }

    // One perturbation per instance gives a rank-one [1, mask] system. In Half the 1e-6 ridge was lost, so an
    // all-kept mask left (D'WD + ridge I) singular and the importance non-finite. Solved in double it is finite.
    [Fact(Timeout = 60000)]
    public async Task LimeSelector_Half_SinglePerturbationSystemKeepsTheRidge()
    {
        await Task.Yield();
        const int n = 20;
        var data = new Half[n, 1];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)i;
            target[i] = (Half)i;
        }

        var selector = new LIMESelector<Half>(nFeaturesToSelect: 1, nSamplesToExplain: n, nPerturbations: 1, randomState: 5);
        selector.Fit(new Matrix<Half>(data), new Vector<Half>(target));

        Assert.NotNull(selector.Importances);
        Assert.True(HalfIsFinite(selector.Importances[0]));
    }

    // 100 rows alternating -1024 / 1024 with the default mean-of-features prediction: each local coefficient is about
    // +-1024 (instance minus the mean of the uniform replacements), so the importance is about 1024. The Half
    // right-hand side (up to 500 * 1024) and the running sum of 100 coefficients overflowed.
    [Fact(Timeout = 120000)]
    public async Task LimeSelector_Half_ImportanceSumPast65504()
    {
        await Task.Yield();
        const int n = 100;
        var data = new Half[n, 1];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)(i % 2 == 0 ? -1024 : 1024);
            target[i] = data[i, 0];
        }

        var selector = new LIMESelector<Half>(nFeaturesToSelect: 1, nSamplesToExplain: n, randomState: 5);
        selector.Fit(new Matrix<Half>(data), new Vector<Half>(target));

        Assert.NotNull(selector.Importances);
        Assert.Equal(1024.0, (double)selector.Importances[0], 150.0);
    }

    // 100 rows alternating -1024 / 1024 with the default mean-of-features prediction: the background mean is 0 and
    // every attribution is +-1024, so the mean |SHAP| is 1024. The Half running sum overflowed after 64 rows.
    [Fact(Timeout = 120000)]
    public async Task ShapSelector_Half_AttributionSumPast65504()
    {
        await Task.Yield();
        const int n = 100;
        var data = new Half[n, 1];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)(i % 2 == 0 ? -1024 : 1024);
            target[i] = data[i, 0];
        }

        var selector = new SHAPSelector<Half>(nFeaturesToSelect: 1, nSamples: n, randomState: 5);
        selector.Fit(new Matrix<Half>(data), new Vector<Half>(target));

        Assert.NotNull(selector.ShapValues);
        Assert.Equal(1024.0, (double)selector.ShapValues[0]);
    }

    // Baseline 1024, then permutation scores 1024 and 0: drops 0 and 1024, mean 512 and population std 512. The
    // centred drops +-512 squared past Half's 65504, so the std was inf.
    [Fact(Timeout = 60000)]
    public async Task PermutationImportance_Half_DropDeviationsPast65504()
    {
        await Task.Yield();
        int calls = 0;
        var scores = new[] { 1024.0, 1024.0, 0.0 };
        var selector = new PermutationImportance<Half>(
            nFeaturesToSelect: 1, nPermutations: 2, scorer: (_, _) => (Half)scores[calls++], randomState: 5);
        var data = new Half[4, 1];
        for (int i = 0; i < 4; i++)
            data[i, 0] = (Half)i;
        selector.Fit(new Matrix<Half>(data), new Vector<Half>(new[] { (Half)0, (Half)1, (Half)2, (Half)3 }));

        Assert.NotNull(selector.ImportanceStd);
        Assert.NotNull(selector.ImportanceScores);
        Assert.Equal(512.0, (double)selector.ImportanceScores[0]);
        Assert.Equal(512.0, (double)selector.ImportanceStd[0]);
    }

    // 2049 selected features, zero correlations and a target count of 2048: the energy is the count penalty
    // 0.1 * |2049 - 2048| = 0.1. The count cast to Half rounded to 2048, giving 0.
    [Fact(Timeout = 60000)]
    public async Task QuantumAnnealing_Half_CountPenaltyPast2048()
    {
        await Task.Yield();
        const int p = 2049;
        var energy = AiDotNet.Preprocessing.FeatureSelection.Quantum.QuantumAnnealingSelector<Half>.ComputeEnergy(
            new Tensor<Half>(new[] { p }), HalfTensor(Enumerable.Repeat(1.0, p).ToArray(), p), 2048);

        Assert.Equal(0.1, (double)energy[0], 3);
    }

    // Four bins over [-60000, 60000]: (x - min) / range * 3 = 0, 1, 2, 3. In Half the range (120000) and the upper
    // numerators overflowed and the division gave NaN symbols.
    [Fact(Timeout = 60000)]
    public async Task LempelZiv_Half_DiscretizesPast65504()
    {
        await Task.Yield();
        var x = HalfTensor(new[] { -60000.0, -20000.0, 20000.0, 60000.0 }, 4, 1);

        Assert.Equal(new[] { 0, 1, 2, 3 },
            AiDotNet.Preprocessing.FeatureSelection.Complexity.LempelZivSelector<Half>.Symbols(x, 4));
    }

    // Two bins and n = 2^21 rows give the bound n / log2(n) = 99864.38, which is infinite in Half; dividing in Half
    // turned every complexity into 0. 49932 / 99864.38 = 0.5.
    [Fact(Timeout = 60000)]
    public async Task LempelZiv_Half_NormalizesByABoundPast65504()
    {
        await Task.Yield();
        double bound = 2097152 / (Math.Log(2097152) / Math.Log(2));

        var normalized = AiDotNet.Preprocessing.FeatureSelection.Complexity.LempelZivSelector<Half>.NormalizeComplexity(
            new[] { 49932 }, bound);

        Assert.Equal(49932 / bound, (double)normalized[0], 3);
    }

    // Column [-300, -300, 300, 300] has skewness 0 and excess kurtosis -2, so JB = 4 / 6 * (4 / 4) = 0.6667; column
    // [0, 0, 0, 1] has JB = 0.963. In Half the squared deviations (90000) overflowed, z became 0 and the first JB
    // came out 1.5, ranking the second column as the more Gaussian.
    [Fact(Timeout = 60000)]
    public async Task Gaussianity_Half_MomentsPast65504()
    {
        await Task.Yield();
        var x = HalfTensor(new[] { -300.0, 0, -300, 0, 300, 0, 300, 1 }, 4, 2);

        var jb = AiDotNet.Preprocessing.FeatureSelection.Distribution.GaussianitySelector<Half>.JarqueBera(x);

        Assert.Equal(2.0 / 3.0, (double)jb[0], 2);
        Assert.Equal(0.963, (double)jb[1], 2);
    }

    // Constant columns keep the maximum-value sentinel through the widened path, below which every real JB stays.
    [Fact(Timeout = 60000)]
    public async Task Gaussianity_Half_ConstantColumnKeepsTheMaximum()
    {
        await Task.Yield();
        var x = HalfTensor(new[] { 5.0, -300, 5, -300, 5, 300, 5, 300 }, 4, 2);

        var jb = AiDotNet.Preprocessing.FeatureSelection.Distribution.GaussianitySelector<Half>.JarqueBera(x);

        Assert.Equal(Half.MaxValue, jb[0]);
        Assert.True(jb[1] < Half.MaxValue);
    }

    // One centred column [-300, 300, -300, 300] has covariance 90000, past Half's 65504, so the eigensolver saw inf.
    // Whitening divides by 300 and gives +-1.
    [Fact(Timeout = 60000)]
    public async Task FastIcaWhiten_Half_CovariancePast65504()
    {
        await Task.Yield();
        var z = FastIcaOps.Whiten(HalfTensor(new[] { -300.0, 300, -300, 300 }, 4, 1), 1, out _);

        for (int i = 0; i < 4; i++)
            Assert.Equal(1.0, Math.Abs((double)z[i, 0]), 2);
    }

    // Rows (-1, 0), (-1, 0), (1, 0), (1, 0): the second direction has no variance. Flooring its eigenvalue at 1e-10
    // gave the constant feature a whitening gain of 1e5, so its unmixing score dominated and it was selected.
    [Fact(Timeout = 60000)]
    public async Task IcaBased_ConstantFeatureIsNotSelected()
    {
        await Task.Yield();
        var data = new Matrix<double>(new double[,] { { -1, 0 }, { -1, 0 }, { 1, 0 }, { 1, 0 } });
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Projection.ICABasedSelector<double>(
            nFeaturesToSelect: 1, nComponents: 2, nIterations: 20);

        selector.Fit(data);

        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
        var z = FastIcaOps.Whiten(Tensor<double>.FromMatrix(data), 2, out var whitening);
        Assert.Equal(1, z.Shape[1]);
        Assert.Equal(0.0, whitening[1, 0], 12);
    }

    // [-300, 300] has population std 300 and sample std 424.26; the squares (90000) overflowed Half, giving std inf
    // and standardized values 0.
    [Fact(Timeout = 60000)]
    public async Task NeuralSelectorOps_Half_StandardDeviationsPast65504()
    {
        await Task.Yield();
        var x = HalfTensor(new[] { -300.0, 300 }, 2, 1);

        Assert.Equal(300.0, (double)NeuralSelectorOps.PopulationStdOrOne(x)[0, 0], 1);
        var sample = NeuralSelectorOps.StandardizeSample(x);
        Assert.Equal(-1 / Math.Sqrt(2), (double)sample[0, 0], 2);
        Assert.Equal(1 / Math.Sqrt(2), (double)sample[1, 0], 2);
    }

    // 65536 selected features with zero scores, no penalty and a target of 0: fitness 0. The count was inf in Half
    // and |inf - 0| * 0 gave NaN.
    [Fact(Timeout = 60000)]
    public async Task SubsetFitness_Half_CountPast65504()
    {
        await Task.Yield();
        const int p = 65536;
        var fitness = PopulationSearchOps.SubsetFitness(
            AllSelected(p), new Tensor<Half>(new[] { p }), 0, (Half)0, (Half)0);

        Assert.Equal(0.0, (double)fitness[0]);
    }

    // A repeated index still marks 1, and every other position stays 0.
    [Fact(Timeout = 60000)]
    public async Task IndexMask_RepeatedIndicesMarkOne()
    {
        await Task.Yield();
        var mask = PopulationSearchOps.IndexMask<double>(new[] { 3, 1, 3 }, 5);

        Assert.Equal(new[] { 0.0, 1, 0, 1, 0 }, mask.ToArray());
    }

    // Ten alternating rows x = +-300 with y = +-1 are fitted exactly by y = x / 300 in every fold, so the held-out
    // R^2 is 1. Eight training rows put 720000 on the Gram diagonal, past Half's 65504.
    [Fact(Timeout = 60000)]
    public async Task FilterWrapperCrossValidatedRSquared_Half_GramPast65504()
    {
        await Task.Yield();
        var xs = Enumerable.Range(0, 10).Select(i => i % 2 == 0 ? 300.0 : -300.0).ToArray();
        var r2 = AiDotNet.Preprocessing.FeatureSelection.Hybrid.FilterWrapperHybridSelector<Half>.CrossValidatedRSquared(
            HalfTensor(xs, 10, 1), HalfTensor(xs.Select(v => v / 300).ToArray(), 10),
            new System.Collections.Generic.List<int> { 0 }, Enumerable.Range(0, 10).ToArray());

        Assert.Equal(1.0, (double)r2[0], 2);
    }

    // The bounded neighbour search (one row per block) returns the same neighbours as a single block.
    [Fact(Timeout = 60000)]
    public async Task SmoteNeighbors_BlockedSearchMatchesOneBlock()
    {
        await Task.Yield();
        var values = Enumerable.Range(0, 24).Select(i => Math.Sin(i * 1.7) * 10 + i % 3).ToArray();
        var minority = new Tensor<double>(values, new[] { 8, 3 });

        var whole = AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<double>.MinorityNeighbors(minority, 3);
        var blocked = AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<double>.MinorityNeighbors(
            minority, 3, blockElements: 1);

        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(whole[i], blocked[i]);
            Assert.DoesNotContain(i, blocked[i]);
            Assert.Equal(3, blocked[i].Length);
        }
    }

    // Midpoint of -60000 and 60000 is 0; their difference (120000) overflowed Half and the sample came out inf.
    [Fact(Timeout = 60000)]
    public async Task SmoteInterpolate_Half_EndpointGapPast65504()
    {
        await Task.Yield();
        var samples = AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<Half>.Interpolate(
            HalfTensor(new[] { -60000.0, 60000 }, 2, 1), new[] { 0 }, new[] { 1 }, HalfTensor(new[] { 0.5 }, 1));

        Assert.Equal(0.0, (double)samples[0, 0]);
    }

    // Classes [-300, -100] and [100, 300]: variances 10000 each and squared mean gap 160000, so the Fisher score is
    // 160000 / 20000 = 8. The gap overflowed Half and the score came out inf.
    [Fact(Timeout = 60000)]
    public async Task SmoteFisherScores_Half_MeanGapPast65504()
    {
        await Task.Yield();
        var scores = AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<Half>.FisherScores(
            HalfTensor(new[] { -300.0, -100 }, 2, 1), HalfTensor(new[] { 100.0, 300 }, 2, 1));

        Assert.Equal(8.0, (double)scores[0]);
    }

    // Instance [0, 2], baseline [0, 0] and a model returning the second column have SHAP values [0, 2]. With only
    // an empty coalition drawn the gap was split equally, [1, 1], crediting the ignored first feature.
    [Fact(Timeout = 60000)]
    public async Task KernelShap_OnlyEmptyDrawsFallBackToSingletons()
    {
        await Task.Yield();
        var phi = SHAPSelector<double>.KernelShap(
            new Tensor<double>(new[] { 0.0, 2 }, new[] { 2 }),
            new Tensor<double>(new[] { 0.0, 0 }, new[] { 2 }),
            new Tensor<double>(new[] { 0.0, 0 }, new[] { 1, 2 }),
            x => new Tensor<double>(Enumerable.Range(0, x.Shape[0]).Select(i => x[i, 1]).ToArray(), new[] { x.Shape[0] }));

        Assert.Equal(0.0, phi[0], 9);
        Assert.Equal(2.0, phi[1], 9);
    }

    // One coalition [1, 1, 0] of an additive model f = x1 + x2 + x3 at instance [1, 1, 1]: the design row is [1, 1],
    // its Gram matrix is all ones and only the 1e-10 ridge makes it solvable, giving SHAP values [1, 1, 1]. That
    // ridge vanished on the float Gram diagonal, leaving the system singular.
    [Fact(Timeout = 60000)]
    public async Task KernelShap_Float_SingleCoalitionKeepsTheRidge()
    {
        await Task.Yield();
        var phi = SHAPSelector<float>.KernelShap(
            new Tensor<float>(new[] { 1f, 1, 1 }, new[] { 3 }),
            new Tensor<float>(new[] { 0f, 0, 0 }, new[] { 3 }),
            new Tensor<float>(new[] { 1f, 1, 0 }, new[] { 1, 3 }),
            x => new Tensor<float>(
                Enumerable.Range(0, x.Shape[0]).Select(i => x[i, 0] + x[i, 1] + x[i, 2]).ToArray(), new[] { x.Shape[0] }));

        for (int j = 0; j < 3; j++)
            Assert.Equal(1.0, phi[j], 3);
    }

    // A best solution holding one feature is completed to the requested three by the highest scores.
    [Fact(Timeout = 60000)]
    public async Task Moead_CompleteSelection_FillsToTheRequestedCount()
    {
        await Task.Yield();
        var selection = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.MOEADSelector<double>.CompleteSelection(
            new Tensor<double>(new[] { 0.0, 1, 0, 0 }, new[] { 4 }),
            new Tensor<double>(new[] { 0.1, 0.5, 0.9, 0.3 }, new[] { 4 }), 3);

        Assert.Equal(new[] { 1, 2, 3 }, selection);
    }

    // Asking for every feature: the initial subsets hold 1 to p features and the best solution under the middle
    // weight rarely holds all of them, so the selection came out short of the requested count.
    [Fact(Timeout = 60000)]
    public async Task Moead_SelectsTheRequestedCount()
    {
        await Task.Yield();
        const int n = 30, p = 8;
        var data = new double[n, p];
        var target = new double[n];
        for (int i = 0; i < n; i++)
        {
            target[i] = Math.Sin(i);
            for (int j = 0; j < p; j++)
                data[i, j] = Math.Sin(i * (j + 1) * 0.7) + target[i] * j / p;
        }

        for (int run = 0; run < 10; run++)
        {
            var selector = new AiDotNet.Preprocessing.FeatureSelection.MultiObjective.MOEADSelector<double>(
                nFeaturesToSelect: p, populationSize: 2, nGenerations: 0, neighborhoodSize: 2);
            selector.Fit(new Matrix<double>(data), new Vector<double>(target));

            Assert.Equal(Enumerable.Range(0, p).ToArray(), selector.SelectedIndices);
        }
    }

    // Four places and a first front of six: the loop bound size - selected.Count shrank as survivors were added,
    // so only two were taken.
    [Fact(Timeout = 60000)]
    public async Task Nsga2_SelectSurvivors_FillsEveryPlaceFromALargeFirstFront()
    {
        await Task.Yield();
        var fronts = new System.Collections.Generic.List<int[]> { new[] { 0, 1, 2, 3, 4, 5 } };
        var crowding = new Tensor<double>(new[] { 1.0, 6, 2, 5, 3, 4 }, new[] { 6 });

        var survivors = AiDotNet.Preprocessing.FeatureSelection.MultiObjective.NSGAIISelector<double>.SelectSurvivors(
            fronts, crowding, 4);

        Assert.Equal(new[] { 1, 3, 4, 5 }, survivors.OrderBy(i => i).ToArray());
    }

    // 4096 samples whose attention is 1: the mean is 1. A Half running total stopped at 2048, giving 0.5.
    [Fact(Timeout = 60000)]
    public async Task AttentionMean_Half_TotalPast2048()
    {
        await Task.Yield();
        var mean = AiDotNet.Preprocessing.FeatureSelection.Neural.AttentionBasedSelector<Half>.MeanAttention(
            4096, 1, _ => HalfTensor(new[] { 1.0 }, 1));

        Assert.Equal(1.0, (double)mean[0]);
    }

    // [-40000, 40000] scales to [0, 1]. Its range (80000) overflowed Half and inf / inf gave NaN.
    [Fact(Timeout = 60000)]
    public async Task VaeMinMaxScale_Half_RangePast65504()
    {
        await Task.Yield();
        var scaled = AiDotNet.Preprocessing.FeatureSelection.Neural.VariationalAutoencoderSelector<Half>.MinMaxScale(
            HalfTensor(new[] { -40000.0, 40000 }, 2, 1));

        Assert.Equal(0.0, (double)scaled[0, 0]);
        Assert.Equal(1.0, (double)scaled[1, 0], 3);
    }

    // NaN and infinite budgets or bounds passed the "<= 0" guards and made the noise scale NaN.
    [Theory(Timeout = 60000)]
    [InlineData(double.NaN, 1.0, 1.0)]
    [InlineData(double.PositiveInfinity, 1.0, 1.0)]
    [InlineData(1.0, double.NaN, 1.0)]
    [InlineData(1.0, double.PositiveInfinity, 1.0)]
    [InlineData(1.0, 1.0, double.NaN)]
    [InlineData(1.0, 1.0, double.PositiveInfinity)]
    public async Task DifferentialPrivacy_RejectsNonFiniteSettings(double epsilon, double featureBound, double targetBound)
    {
        await Task.Yield();
        Assert.Throws<ArgumentException>(() =>
            new AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>(
                1, epsilon, null, null, featureBound, targetBound));
    }

    // Bounds 2^-12, n = 128, epsilon = 1 and k = 1 give a noise scale of 5.59e-9, which is 0 in Half: the noise
    // vanished and the slightly larger utility (feature 0) was always selected. The utilities differ by
    // 2^-24 / 64, far below the noise, so the mechanism picks feature 1 a large share of the time.
    [Fact(Timeout = 60000)]
    public async Task DifferentialPrivacy_Half_KeepsNoiseBelowHalfResolution()
    {
        await Task.Yield();
        const int n = 128;
        double b = Math.Pow(2, -12);
        var data = new Half[n, 2];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            double y = i % 2 == 0 ? b : -b;
            target[i] = (Half)y;
            data[i, 0] = (Half)y;
            data[i, 1] = (Half)(i == 0 ? -y : y);
        }

        int second = 0;
        for (int seed = 0; seed < 200; seed++)
        {
            var selector = new AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<Half>(
                1, 1.0, seed, null, b, b);
            selector.Fit(new Matrix<Half>(data), new Vector<Half>(target));
            Assert.NotNull(selector.SelectedIndices);
            if (selector.SelectedIndices[0] == 1)
                second++;
        }

        Assert.InRange(second, 40, 160);
    }

    // A 2 x 100 matrix of 100 with one component: H H' and X H' pass 65504 in Half, the W update divided inf by inf
    // and the scores became NaN.
    [Fact(Timeout = 60000)]
    public async Task NmfBased_Half_FactorProductsPast65504()
    {
        await Task.Yield();
        var data = new Half[2, 100];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 100; j++)
                data[i, j] = (Half)100;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Projection.NMFBasedSelector<Half>(
            nFeaturesToSelect: 5, nComponents: 1, nIterations: 3);

        selector.Fit(new Matrix<Half>(data));

        Assert.NotNull(selector.FeatureScores);
        for (int j = 0; j < 100; j++)
            Assert.True(HalfIsFinite(selector.FeatureScores[j]), $"score {j} is {selector.FeatureScores[j]}");
    }

    // theta = pi / 4 rotated by 2.0 towards bit 1 overshot to 2.785, lowering P(1) to 0.12; it stops at pi / 2.
    [Fact(Timeout = 60000)]
    public async Task QuantumInspired_RotationStaysWithinZeroAndHalfPi()
    {
        await Task.Yield();
        var rotated = AiDotNet.Preprocessing.FeatureSelection.Quantum.QuantumInspiredSelector<double>.Rotate(
            new Tensor<double>(new[] { Math.PI / 4, Math.PI / 4 }, new[] { 2 }),
            new Tensor<double>(new[] { 0.0, 1 }, new[] { 2 }),
            new Tensor<double>(new[] { 1.0, 0 }, new[] { 2 }), 2.0);

        Assert.Equal(Math.PI / 2, rotated[0], 12);
        Assert.Equal(0.0, rotated[1], 12);
        Assert.Throws<ArgumentException>(() =>
            new AiDotNet.Preprocessing.FeatureSelection.Quantum.QuantumInspiredSelector<double>(rotationAngle: double.NaN));
    }
}
