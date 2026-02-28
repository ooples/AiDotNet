using AiDotNet.ActiveLearning;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActiveLearning;

/// <summary>
/// Deep math integration tests for active learning scoring strategies.
/// Tests the full softmax → scoring pipeline with hand-computed expected values.
///
/// Key formulas tested:
/// - Softmax: p_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
/// - Entropy: H = -Σ (p+ε) * log(p+ε), ε=1e-10
/// - Margin: 1 - (P_top1 - P_top2)
/// - Least Confidence: 1 - max(p)
/// - BALD: H(avg_p) - avg(H(p_i))
/// - InformationDensity: Uncertainty * AvgSimilarity^β
/// - Cosine similarity: (a·b)/(|a||b|), normalized to [0,1] as (cos+1)/2
/// - RBF: exp(-γ * d²)
/// - Inverse Euclidean: 1/(1+d)
/// </summary>
public class ActiveLearningDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════════
    // Softmax verification (shared by all strategies)
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void Softmax_UniformLogits_ProducesUniformProbabilities()
    {
        // Logits: [1, 1, 1] → softmax = [1/3, 1/3, 1/3]
        var model = new DeterministicModel(new double[] { 1.0, 1.0, 1.0 });
        var pool = MakePool(1, 2); // 1 sample, 2 features
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Uniform distribution → maximum entropy = ln(3) ≈ 1.0986
        // With epsilon: H = -3 * (1/3 + 1e-10) * ln(1/3 + 1e-10) ≈ ln(3) = 1.098612
        Assert.Equal(Math.Log(3.0), scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Softmax_DominantLogit_ProducesNearOneProbability()
    {
        // Logits: [10, 0, 0] → softmax ≈ [0.9999, 0.00005, 0.00005]
        // exp(10-10)=1, exp(0-10)=exp(-10)≈4.54e-5
        // sum ≈ 1.0 + 2*4.54e-5 = 1.0000908
        // p1 ≈ 1/1.0000908 ≈ 0.999909
        var model = new DeterministicModel(new double[] { 10.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new LeastConfidenceSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // LC = 1 - max(p) ≈ 1 - 0.999909 ≈ 9.08e-5
        Assert.True(scores[0] < 0.001, $"LC score {scores[0]} should be near 0 for dominant logit");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Softmax_NumericalStability_LargeLogitsDoNotOverflow()
    {
        // Logits: [1000, 999, 998] - large logits, but max-shift prevents overflow
        // Shifted: [0, -1, -2] → exp: [1, 0.3679, 0.1353] → sum = 1.5032
        // probs: [0.6652, 0.2447, 0.09003]
        var model = new DeterministicModel(new double[] { 1000.0, 999.0, 998.0 });
        var pool = MakePool(1, 2);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Should not be NaN or Infinity
        Assert.False(double.IsNaN(scores[0]));
        Assert.False(double.IsInfinity(scores[0]));

        // Hand-compute: shifted=[0,-1,-2], exp=[1, e^-1, e^-2]
        double e0 = 1.0, e1 = Math.Exp(-1), e2 = Math.Exp(-2);
        double sum = e0 + e1 + e2;
        double p0 = e0 / sum, p1 = e1 / sum, p2 = e2 / sum;
        double eps = 1e-10;
        double expected = -(  (p0 + eps) * Math.Log(p0 + eps)
                            + (p1 + eps) * Math.Log(p1 + eps)
                            + (p2 + eps) * Math.Log(p2 + eps));
        Assert.Equal(expected, scores[0], Tol);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Entropy scoring: H = -Σ (p+ε) * log(p+ε)
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_HandCalculated_ThreeClassUniform()
    {
        // Logits: [0, 0, 0] → softmax = [1/3, 1/3, 1/3]
        // H = -3 * (1/3 + ε) * ln(1/3 + ε)
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double p = 1.0 / 3.0;
        double eps = 1e-10;
        double expected = -3.0 * (p + eps) * Math.Log(p + eps);
        Assert.Equal(expected, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_HandCalculated_TwoClassLogits()
    {
        // Logits: [2, 0] → shifted: [0, -2]
        // exp: [1, e^-2] = [1, 0.13534]
        // sum = 1.13534
        // p0 = 0.88080, p1 = 0.11920
        // H = -[(0.88080+ε)*ln(0.88080+ε) + (0.11920+ε)*ln(0.11920+ε)]
        var model = new DeterministicModel(new double[] { 2.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double expM2 = Math.Exp(-2.0);
        double sumExp = 1.0 + expM2;
        double p0 = 1.0 / sumExp;
        double p1 = expM2 / sumExp;
        double eps = 1e-10;
        double expected = -((p0 + eps) * Math.Log(p0 + eps) + (p1 + eps) * Math.Log(p1 + eps));
        Assert.Equal(expected, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_CertainPrediction_NearZero()
    {
        // Logits: [100, 0, 0] → nearly all prob on first class
        var model = new DeterministicModel(new double[] { 100.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.True(scores[0] < 0.001, $"Entropy {scores[0]} should be near 0 for near-certain prediction");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_MultipleSamples_CorrectScorePerSample()
    {
        // Two samples with different logits
        // Sample 0: logits [0, 0] → uniform → max entropy = ln(2)
        // Sample 1: logits [5, 0] → skewed → lower entropy
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 5.0, 0.0 }, numClasses: 2);
        var pool = MakePool(2, 3);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Sample 0 (uniform) should have higher entropy than sample 1 (skewed)
        Assert.True(scores[0] > scores[1],
            $"Uniform sample entropy {scores[0]} should be > skewed sample entropy {scores[1]}");

        // Sample 0 should be near ln(2) ≈ 0.6931
        double eps = 1e-10;
        double expectedUniform = -2.0 * (0.5 + eps) * Math.Log(0.5 + eps);
        Assert.Equal(expectedUniform, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_SelectTopScoring_ReturnsHighestEntropyIndices()
    {
        // 3 samples: sample 1 has most uniform logits → highest entropy
        // Sample 0: [5, 0] → skewed
        // Sample 1: [0, 0] → uniform (highest entropy)
        // Sample 2: [3, 0] → moderately skewed
        var model = new DeterministicModel(
            new double[] { 5.0, 0.0, 0.0, 0.0, 3.0, 0.0 }, numClasses: 2);
        var pool = MakePool(3, 2);
        var selected = new EntropySampling<double>().SelectSamples(model, pool, 1);

        Assert.Single(selected);
        Assert.Equal(1, selected[0]); // Sample 1 has highest entropy
    }

    // ═══════════════════════════════════════════════════════════════════
    // Margin scoring: 1 - (P_top1 - P_top2)
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void Margin_HandCalculated_UniformDistribution()
    {
        // Logits: [0, 0, 0] → probs = [1/3, 1/3, 1/3]
        // Margin = P1 - P2 = 1/3 - 1/3 = 0
        // Score = 1 - 0 = 1
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new MarginSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.Equal(1.0, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Margin_HandCalculated_TwoClassLogits()
    {
        // Logits: [1, 0] → shifted: [0, -1]
        // exp: [1, e^-1] → sum = 1 + 0.3679 = 1.3679
        // p0 = 0.73106, p1 = 0.26894
        // margin = 0.73106 - 0.26894 = 0.46212
        // score = 1 - 0.46212 = 0.53788
        var model = new DeterministicModel(new double[] { 1.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new MarginSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double eM1 = Math.Exp(-1);
        double sumExp = 1.0 + eM1;
        double p0 = 1.0 / sumExp;
        double p1 = eM1 / sumExp;
        double expected = 1.0 - (p0 - p1);
        Assert.Equal(expected, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Margin_DominantPrediction_ScoreNearZero()
    {
        // Logits: [20, 0, 0] → nearly all prob on class 0
        // margin ≈ 1.0 - 0.0 = 1.0 → score ≈ 1 - 1 = 0
        var model = new DeterministicModel(new double[] { 20.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new MarginSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.True(scores[0] < 0.01, $"Margin score {scores[0]} should be near 0 for dominant prediction");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Margin_ThreeClassWithTwoTied_ScoreIsOne()
    {
        // Logits: [5, 5, 0] → top two are equal → margin = 0 → score = 1
        var model = new DeterministicModel(new double[] { 5.0, 5.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new MarginSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Shifted: [0, 0, -5], exp: [1, 1, e^-5], sum = 2 + e^-5
        // p0 = p1 = 1/(2+e^-5), margin = 0, score = 1
        Assert.Equal(1.0, scores[0], 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Least Confidence: 1 - max(p)
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void LeastConfidence_HandCalculated_Uniform()
    {
        // Logits: [0, 0, 0] → probs = [1/3, 1/3, 1/3]
        // LC = 1 - 1/3 = 2/3
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new LeastConfidenceSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.Equal(2.0 / 3.0, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void LeastConfidence_HandCalculated_TwoClass()
    {
        // Logits: [3, 0] → shifted: [0, -3]
        // exp: [1, e^-3=0.04979], sum = 1.04979
        // max_p = 1/1.04979 = 0.95257
        // LC = 1 - 0.95257 = 0.04743
        var model = new DeterministicModel(new double[] { 3.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new LeastConfidenceSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double eM3 = Math.Exp(-3);
        double maxP = 1.0 / (1.0 + eM3);
        double expected = 1.0 - maxP;
        Assert.Equal(expected, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void LeastConfidence_BinaryUniform_ScoreIsHalf()
    {
        // Logits: [0, 0] → probs = [0.5, 0.5]
        // LC = 1 - 0.5 = 0.5
        var model = new DeterministicModel(new double[] { 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new LeastConfidenceSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.Equal(0.5, scores[0], Tol);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Strategy comparison: same logits, different scores
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void AllStrategies_UniformLogits_AllMaximal()
    {
        // Uniform logits → all strategies give maximum uncertainty
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);

        var entropyScores = new EntropySampling<double>().ComputeInformativenessScores(model, pool);
        var marginScores = new MarginSampling<double>().ComputeInformativenessScores(model, pool);
        var lcScores = new LeastConfidenceSampling<double>().ComputeInformativenessScores(model, pool);

        // Entropy: ln(3) ≈ 1.0986
        Assert.Equal(Math.Log(3), entropyScores[0], 1e-3);
        // Margin: 1.0 (zero margin)
        Assert.Equal(1.0, marginScores[0], Tol);
        // LC: 2/3
        Assert.Equal(2.0 / 3.0, lcScores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void AllStrategies_SkewedLogits_ConsistentOrdering()
    {
        // Two samples: sample 0 is more uncertain than sample 1
        // Sample 0: [0, 0, 0] → uniform
        // Sample 1: [5, 0, 0] → skewed
        var model = new DeterministicModel(
            new double[] { 0.0, 0.0, 0.0, 5.0, 0.0, 0.0 }, numClasses: 3);
        var pool = MakePool(2, 2);

        var entropyScores = new EntropySampling<double>().ComputeInformativenessScores(model, pool);
        var marginScores = new MarginSampling<double>().ComputeInformativenessScores(model, pool);
        var lcScores = new LeastConfidenceSampling<double>().ComputeInformativenessScores(model, pool);

        // All should rank sample 0 as more uncertain
        Assert.True(entropyScores[0] > entropyScores[1],
            $"Entropy: uniform={entropyScores[0]} should > skewed={entropyScores[1]}");
        Assert.True(marginScores[0] > marginScores[1],
            $"Margin: uniform={marginScores[0]} should > skewed={marginScores[1]}");
        Assert.True(lcScores[0] > lcScores[1],
            $"LC: uniform={lcScores[0]} should > skewed={lcScores[1]}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // BALD: I(y;θ|x) = H(ȳ) - E[H(y_i)]
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void BALD_ScoreIsNonNegative()
    {
        // BALD score = H(avg) - avg(H) ≥ 0 by Jensen's inequality (H is concave)
        var model = new DeterministicModel(new double[] { 1.0, 2.0, 0.5 });
        var pool = MakePool(1, 2);
        var bald = new BALD<double>(numMcSamples: 5, dropoutRate: 0.3);
        var scores = bald.ComputeInformativenessScores(model, pool);

        Assert.True(scores[0] >= -1e-6,
            $"BALD score {scores[0]} should be non-negative (Jensen's inequality)");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BALD_NameIncludesMCSamples()
    {
        var bald = new BALD<double>(numMcSamples: 15);
        Assert.Equal("BALD-MC15", bald.Name);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BALD_MultiSample_ScoresAreFinite()
    {
        var model = new DeterministicModel(
            new double[] { 1.0, 0.0, 3.0, 0.5, 0.0, 2.0, -1.0, 1.5, 0.5 },
            numClasses: 3);
        var pool = MakePool(3, 2);
        var bald = new BALD<double>(numMcSamples: 10);
        var scores = bald.ComputeInformativenessScores(model, pool);

        for (int i = 0; i < 3; i++)
        {
            Assert.False(double.IsNaN(scores[i]), $"BALD score[{i}] is NaN");
            Assert.False(double.IsInfinity(scores[i]), $"BALD score[{i}] is Infinity");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // InformationDensity: ID = Uncertainty * AvgSim^β
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void InformationDensity_CosineSimilarity_HandCalculated()
    {
        // Two identical samples → cosine similarity = 1 → normalized = (1+1)/2 = 1
        // Pool: sample 0 = [1, 0], sample 1 = [1, 0]
        var poolData = new Vector<double>(new double[] { 1.0, 0.0, 1.0, 0.0 });
        var pool = new Tensor<double>(new[] { 2, 2 }, poolData);
        // Logits: both samples get uniform logits [0, 0]
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 }, numClasses: 2);
        var strategy = new InformationDensity<double>(beta: 1.0,
            similarityMeasure: InformationDensity<double>.SimilarityMeasure.Cosine);
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Entropy(uniform 2-class) ≈ ln(2) = 0.6931
        // AvgSim = 1.0 (identical vectors → cosine=1 → normalized=(1+1)/2=1)
        // ID = ln(2) * 1.0^1 = ln(2)
        double eps = 1e-10;
        double expectedEntropy = -2.0 * (0.5 + eps) * Math.Log(0.5 + eps);
        Assert.Equal(expectedEntropy, scores[0], 1e-4);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void InformationDensity_RBFSimilarity_HandCalculated()
    {
        // Pool: sample 0 = [0, 0], sample 1 = [1, 0]
        // Euclidean distance = 1.0
        // RBF(gamma=1) = exp(-1 * 1²) = exp(-1) = 0.3679
        var poolData = new Vector<double>(new double[] { 0.0, 0.0, 1.0, 0.0 });
        var pool = new Tensor<double>(new[] { 2, 2 }, poolData);
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 }, numClasses: 2);
        var strategy = new InformationDensity<double>(beta: 1.0,
            similarityMeasure: InformationDensity<double>.SimilarityMeasure.RBF,
            rbfGamma: 1.0);
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double eps = 1e-10;
        double entropy = -2.0 * (0.5 + eps) * Math.Log(0.5 + eps);
        double rbfSim = Math.Exp(-1.0); // exp(-gamma * d²)
        double expected = entropy * Math.Pow(rbfSim, 1.0);
        Assert.Equal(expected, scores[0], 1e-4);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void InformationDensity_InverseEuclidean_HandCalculated()
    {
        // Pool: sample 0 = [0, 0], sample 1 = [3, 4]
        // Euclidean distance = 5.0
        // InverseEuclidean = 1/(1+5) = 1/6 ≈ 0.1667
        var poolData = new Vector<double>(new double[] { 0.0, 0.0, 3.0, 4.0 });
        var pool = new Tensor<double>(new[] { 2, 2 }, poolData);
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 }, numClasses: 2);
        var strategy = new InformationDensity<double>(beta: 1.0,
            similarityMeasure: InformationDensity<double>.SimilarityMeasure.InverseEuclidean);
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double eps = 1e-10;
        double entropy = -2.0 * (0.5 + eps) * Math.Log(0.5 + eps);
        double invEuc = 1.0 / (1.0 + 5.0);
        double expected = entropy * invEuc;
        Assert.Equal(expected, scores[0], 1e-4);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void InformationDensity_BetaExponent_ScalesCorrectly()
    {
        // With beta=2, the similarity term is squared
        var poolData = new Vector<double>(new double[] { 0.0, 0.0, 1.0, 0.0 });
        var pool = new Tensor<double>(new[] { 2, 2 }, poolData);
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 }, numClasses: 2);

        var strategyB1 = new InformationDensity<double>(beta: 1.0,
            similarityMeasure: InformationDensity<double>.SimilarityMeasure.RBF, rbfGamma: 1.0);
        var strategyB2 = new InformationDensity<double>(beta: 2.0,
            similarityMeasure: InformationDensity<double>.SimilarityMeasure.RBF, rbfGamma: 1.0);

        var scores1 = strategyB1.ComputeInformativenessScores(model, pool);
        var scores2 = strategyB2.ComputeInformativenessScores(model, pool);

        // sim = exp(-1) ≈ 0.3679
        // beta=1: ID = entropy * 0.3679^1
        // beta=2: ID = entropy * 0.3679^2
        // ratio should be sim ≈ 0.3679
        double ratio = scores2[0] / scores1[0];
        double expectedRatio = Math.Exp(-1.0); // sim^(2-1) = sim
        Assert.Equal(expectedRatio, ratio, 1e-4);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void InformationDensity_ThreeSamples_CosineSimilarityHandCalculated()
    {
        // Pool: [1,0], [0,1], [1,1]
        // Cosine similarities (raw):
        //   cos([1,0],[0,1]) = 0/(1*1) = 0 → normalized = 0.5
        //   cos([1,0],[1,1]) = 1/(1*√2) = 1/√2 → normalized = (1/√2 + 1)/2
        //   cos([0,1],[1,1]) = 1/(1*√2) = 1/√2 → normalized = (1/√2 + 1)/2
        var poolData = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var pool = new Tensor<double>(new[] { 3, 2 }, poolData);
        var model = new DeterministicModel(
            new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, numClasses: 2);
        var strategy = new InformationDensity<double>(beta: 1.0,
            similarityMeasure: InformationDensity<double>.SimilarityMeasure.Cosine);
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // For sample 0: avg_sim = (sim(0,1) + sim(0,2)) / 2
        //   sim(0,1) = (0+1)/2 = 0.5
        //   sim(0,2) = (1/√2 + 1)/2 ≈ 0.8536
        //   avg = (0.5 + 0.8536)/2 = 0.6768
        double sim01 = (0.0 + 1.0) / 2.0;
        double sim02 = (1.0 / Math.Sqrt(2.0) + 1.0) / 2.0;
        double avgSim0 = (sim01 + sim02) / 2.0;

        double eps = 1e-10;
        double entropy = -2.0 * (0.5 + eps) * Math.Log(0.5 + eps);
        double expected0 = entropy * avgSim0;
        Assert.Equal(expected0, scores[0], 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Selection and statistics
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void SelectSamples_ReturnsCorrectBatchSize()
    {
        var model = new DeterministicModel(
            new double[] { 0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 0.0 }, numClasses: 2);
        var pool = MakePool(5, 3);

        var selected = new EntropySampling<double>().SelectSamples(model, pool, 3);
        Assert.Equal(3, selected.Length);
        Assert.Equal(selected.Distinct().Count(), selected.Length); // All unique
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SelectSamples_BatchSizeExceedsPool_ReturnsAll()
    {
        var model = new DeterministicModel(new double[] { 1.0, 0.0, 2.0, 0.0 }, numClasses: 2);
        var pool = MakePool(2, 2);

        var selected = new EntropySampling<double>().SelectSamples(model, pool, 10);
        Assert.Equal(2, selected.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void GetSelectionStatistics_AfterCompute_ReturnsCorrectStats()
    {
        // 3 samples with different entropy levels
        var model = new DeterministicModel(
            new double[] { 0.0, 0.0, 3.0, 0.0, 10.0, 0.0 }, numClasses: 2);
        var pool = MakePool(3, 2);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);
        var stats = strategy.GetSelectionStatistics();

        double min = Math.Min(scores[0], Math.Min(scores[1], scores[2]));
        double max = Math.Max(scores[0], Math.Max(scores[1], scores[2]));
        double mean = (scores[0] + scores[1] + scores[2]) / 3.0;

        Assert.Equal(min, stats["MinScore"], Tol);
        Assert.Equal(max, stats["MaxScore"], Tol);
        Assert.Equal(mean, stats["MeanScore"], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_FourClassUniform_MaximumEntropy()
    {
        // Logits: [0, 0, 0, 0] → all equal → H = ln(4)
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new EntropySampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        double eps = 1e-10;
        double p = 0.25;
        double expected = -4.0 * (p + eps) * Math.Log(p + eps);
        Assert.Equal(expected, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void LeastConfidence_FourClassUniform_ScoreIs075()
    {
        // LC = 1 - max(p) = 1 - 0.25 = 0.75
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new LeastConfidenceSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.Equal(0.75, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Margin_FourClassAllEqual_ScoreIsOne()
    {
        // All equal → top two are same → margin=0 → score=1
        var model = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);
        var strategy = new MarginSampling<double>();
        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.Equal(1.0, scores[0], Tol);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void UseBatchDiversity_CanBeSetAndRetrieved()
    {
        var entropy = new EntropySampling<double>();
        Assert.False(entropy.UseBatchDiversity);

        entropy.UseBatchDiversity = true;
        Assert.True(entropy.UseBatchDiversity);

        var margin = new MarginSampling<double>();
        margin.UseBatchDiversity = true;
        Assert.True(margin.UseBatchDiversity);

        var lc = new LeastConfidenceSampling<double>();
        lc.UseBatchDiversity = true;
        Assert.True(lc.UseBatchDiversity);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StrategyNames_AreCorrect()
    {
        Assert.Equal("EntropySampling", new EntropySampling<double>().Name);
        Assert.Equal("MarginSampling", new MarginSampling<double>().Name);
        Assert.Equal("LeastConfidenceSampling", new LeastConfidenceSampling<double>().Name);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Entropy mathematical properties
    // ═══════════════════════════════════════════════════════════════════

    [Fact]
    [Trait("Category", "Integration")]
    public void Entropy_Monotonicity_MoreClassesHigherMaxEntropy()
    {
        // Max entropy for k classes = ln(k)
        // So 4-class uniform > 3-class uniform > 2-class uniform
        var model2 = new DeterministicModel(new double[] { 0.0, 0.0 });
        var model3 = new DeterministicModel(new double[] { 0.0, 0.0, 0.0 });
        var model4 = new DeterministicModel(new double[] { 0.0, 0.0, 0.0, 0.0 });
        var pool = MakePool(1, 2);

        var s2 = new EntropySampling<double>().ComputeInformativenessScores(model2, pool);
        var s3 = new EntropySampling<double>().ComputeInformativenessScores(model3, pool);
        var s4 = new EntropySampling<double>().ComputeInformativenessScores(model4, pool);

        Assert.True(s4[0] > s3[0], $"4-class entropy {s4[0]} should > 3-class {s3[0]}");
        Assert.True(s3[0] > s2[0], $"3-class entropy {s3[0]} should > 2-class {s2[0]}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════

    private static Tensor<double> MakePool(int numSamples, int numFeatures)
    {
        var data = new Vector<double>(numSamples * numFeatures);
        for (int i = 0; i < data.Length; i++)
            data[i] = (i + 1.0) / data.Length;
        return new Tensor<double>(new[] { numSamples, numFeatures }, data);
    }

    /// <summary>
    /// A deterministic model that returns fixed logits for each sample.
    /// The logits array is: [sample0_class0, sample0_class1, ..., sample1_class0, ...]
    /// </summary>
    private class DeterministicModel : IFullModel<double, Tensor<double>, Tensor<double>>
    {
        private readonly double[] _logits;
        private readonly int _numClasses;
        private readonly Vector<double> _parameters;
        private List<int> _activeFeatures;

        public DeterministicModel(double[] logitsPerSample, int numClasses = 0)
        {
            _logits = logitsPerSample;
            // If numClasses not specified, infer from single sample
            _numClasses = numClasses > 0 ? numClasses : logitsPerSample.Length;
            _parameters = new Vector<double>(10);
            _activeFeatures = Enumerable.Range(0, 5).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        public Tensor<double> Predict(Tensor<double> input)
        {
            var numSamples = input.Shape[0];
            var outputSize = numSamples * _numClasses;
            var data = new Vector<double>(outputSize);

            for (int i = 0; i < outputSize && i < _logits.Length; i++)
            {
                data[i] = _logits[i];
            }

            return new Tensor<double>(new[] { numSamples, _numClasses }, data);
        }

        public void Train(Tensor<double> inputs, Tensor<double> targets) { }
        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public Vector<double> GetParameters() => _parameters;
        public void SetParameters(Vector<double> parameters) { }
        public int ParameterCount => _parameters.Length;
        public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> p) => this;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> indices) => _activeFeatures = indices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => new();
        public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy() => this;
        public IFullModel<double, Tensor<double>, Tensor<double>> Clone() => this;
        public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
            => new(ParameterCount);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
            => throw new NotSupportedException();
        public bool SupportsJitCompilation => false;
    }
}
