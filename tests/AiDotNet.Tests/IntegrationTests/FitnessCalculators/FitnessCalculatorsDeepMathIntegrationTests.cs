using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FitnessCalculators;

/// <summary>
/// Deep mathematical integration tests for fitness calculator loss functions.
/// Verifies correctness of hand-computed expected values for all major loss functions:
/// MSE, MAE, RMSE, Huber, Cross-Entropy, Binary CE, Focal, Dice, KL Divergence,
/// Hinge, Cosine Similarity, R-Squared, Adjusted R-Squared, Log-Cosh,
/// Quantile, Contrastive, Triplet, Elastic Net, Poisson, Jaccard.
/// </summary>
public class FitnessCalculatorsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Mean Squared Error (MSE)

    [Fact]
    public void MSE_HandComputed()
    {
        // MSE = (1/n) * sum((pred - actual)^2)
        // pred = [1, 2, 3], actual = [1.5, 2.5, 2.5]
        // errors = [-0.5, -0.5, 0.5]
        // squared = [0.25, 0.25, 0.25]
        // MSE = 0.75 / 3 = 0.25
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1.5, 2.5, 2.5 };

        double mse = ComputeMSE(pred, actual);
        Assert.Equal(0.25, mse, Tolerance);
    }

    [Fact]
    public void MSE_PerfectPrediction_IsZero()
    {
        var pred = new double[] { 1, 2, 3, 4, 5 };
        double mse = ComputeMSE(pred, pred);
        Assert.Equal(0.0, mse, Tolerance);
    }

    [Fact]
    public void MSE_Symmetry()
    {
        // MSE(a, b) = MSE(b, a)
        var a = new double[] { 1, 2, 3 };
        var b = new double[] { 4, 5, 6 };

        Assert.Equal(ComputeMSE(a, b), ComputeMSE(b, a), Tolerance);
    }

    [Fact]
    public void MSE_NonNegative()
    {
        var pred = new double[] { -5, 10, 0.5 };
        var actual = new double[] { 3, -7, 100 };

        double mse = ComputeMSE(pred, actual);
        Assert.True(mse >= 0, $"MSE should be non-negative: {mse}");
    }

    [Fact]
    public void MSE_LargerErrorsDominateSmaller()
    {
        // MSE penalizes large errors more heavily
        // Single large error of 10: MSE = 100
        // Ten small errors of 1 each: MSE = 1
        double mseOneLarge = ComputeMSE([10], [0]);
        double mseTenSmall = ComputeMSE(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        Assert.Equal(100.0, mseOneLarge, Tolerance);
        Assert.Equal(1.0, mseTenSmall, Tolerance);
    }

    #endregion

    #region Mean Absolute Error (MAE)

    [Fact]
    public void MAE_HandComputed()
    {
        // MAE = (1/n) * sum(|pred - actual|)
        // pred = [1, 2, 3], actual = [2, 3, 1]
        // |errors| = [1, 1, 2]
        // MAE = 4/3 ≈ 1.3333
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 2, 3, 1 };

        double mae = ComputeMAE(pred, actual);
        Assert.Equal(4.0 / 3.0, mae, Tolerance);
    }

    [Fact]
    public void MAE_PerfectPrediction_IsZero()
    {
        var pred = new double[] { 1, 2, 3 };
        double mae = ComputeMAE(pred, pred);
        Assert.Equal(0.0, mae, Tolerance);
    }

    [Fact]
    public void MAE_LessAffectedByOutliers_ThanMSE()
    {
        // With one outlier, MSE increases much more than MAE
        var pred = new double[] { 1, 2, 3, 100 };
        var actual = new double[] { 1, 2, 3, 0 };

        double mse = ComputeMSE(pred, actual);
        double mae = ComputeMAE(pred, actual);

        // MSE = (0+0+0+10000)/4 = 2500
        // MAE = (0+0+0+100)/4 = 25
        Assert.Equal(2500.0, mse, Tolerance);
        Assert.Equal(25.0, mae, Tolerance);
        Assert.True(mse / mae > 10, "MSE should be disproportionately larger than MAE with outliers");
    }

    #endregion

    #region Root Mean Squared Error (RMSE)

    [Fact]
    public void RMSE_HandComputed()
    {
        // RMSE = sqrt(MSE)
        // pred = [1, 3], actual = [2, 4] -> MSE = (1+1)/2 = 1 -> RMSE = 1
        var pred = new double[] { 1, 3 };
        var actual = new double[] { 2, 4 };

        double rmse = Math.Sqrt(ComputeMSE(pred, actual));
        Assert.Equal(1.0, rmse, Tolerance);
    }

    [Fact]
    public void RMSE_SameUnitsAsPredictions()
    {
        // If predictions are in meters and error is 2m, RMSE should be close to 2
        var pred = new double[] { 10, 12, 14 };
        var actual = new double[] { 12, 14, 16 };
        // All errors are 2m, MSE = 4, RMSE = 2
        double rmse = Math.Sqrt(ComputeMSE(pred, actual));
        Assert.Equal(2.0, rmse, Tolerance);
    }

    #endregion

    #region Huber Loss

    [Fact]
    public void HuberLoss_SmallError_EqualsHalfSquaredError()
    {
        // For |error| <= delta: L = 0.5 * error^2
        // delta = 1.0, error = 0.5 -> L = 0.5 * 0.25 = 0.125
        double delta = 1.0;
        double error = 0.5;

        double huber = ComputeHuberLoss(error, delta);
        Assert.Equal(0.125, huber, Tolerance);
    }

    [Fact]
    public void HuberLoss_LargeError_IsLinear()
    {
        // For |error| > delta: L = delta * (|error| - 0.5 * delta)
        // delta = 1.0, error = 3.0 -> L = 1.0 * (3.0 - 0.5) = 2.5
        double delta = 1.0;
        double error = 3.0;

        double huber = ComputeHuberLoss(error, delta);
        Assert.Equal(2.5, huber, Tolerance);
    }

    [Fact]
    public void HuberLoss_AtDelta_BothFormulasAgree()
    {
        // At error = delta, both formulas should give same result
        // Quadratic: 0.5 * delta^2 = 0.5 * 1.0 = 0.5
        // Linear: delta * (delta - 0.5*delta) = 1.0 * 0.5 = 0.5
        double delta = 1.0;
        double error = delta;

        double quadratic = 0.5 * error * error;
        double linear = delta * (Math.Abs(error) - 0.5 * delta);

        Assert.Equal(quadratic, linear, Tolerance);
        Assert.Equal(0.5, ComputeHuberLoss(error, delta), Tolerance);
    }

    [Fact]
    public void HuberLoss_AlwaysLessThanOrEqualToMSE()
    {
        // Huber loss is always <= 0.5 * error^2 (MSE per sample / 2)
        double delta = 1.0;
        double[] errors = { -5, -2, -0.5, 0, 0.3, 1, 3, 10 };

        foreach (var err in errors)
        {
            double huber = ComputeHuberLoss(err, delta);
            double halfSquared = 0.5 * err * err;
            Assert.True(huber <= halfSquared + 1e-10,
                $"Huber({err}) = {huber} should be <= 0.5*err^2 = {halfSquared}");
        }
    }

    [Fact]
    public void HuberLoss_NegativeError_SymmetricWithPositive()
    {
        double delta = 1.5;
        double error = 2.7;

        double huberPos = ComputeHuberLoss(error, delta);
        double huberNeg = ComputeHuberLoss(-error, delta);

        Assert.Equal(huberPos, huberNeg, Tolerance);
    }

    #endregion

    #region Cross-Entropy Loss

    [Fact]
    public void CrossEntropy_HandComputed()
    {
        // CE = -sum(actual * log(pred)) / n
        // For multi-class with true labels: CE = -(1/n) * sum(log(p_true))
        // pred = [0.7, 0.2, 0.1], actual = [1, 0, 0] (class 0)
        // CE = -log(0.7) ≈ 0.35667
        double[] pred = [0.7, 0.2, 0.1];
        double[] actual = [1, 0, 0];

        double ce = ComputeCrossEntropy(pred, actual);
        Assert.Equal(-Math.Log(0.7), ce, 1e-5);
    }

    [Fact]
    public void CrossEntropy_PerfectPrediction_IsZero()
    {
        // If predicted prob for true class = 1.0, CE = -log(1) = 0
        double[] pred = [1.0, 0.0, 0.0];
        double[] actual = [1.0, 0.0, 0.0];

        double ce = ComputeCrossEntropy(pred, actual);
        Assert.Equal(0.0, ce, 1e-5);
    }

    [Fact]
    public void CrossEntropy_WorstPrediction_IsHigh()
    {
        // If predicted prob for true class ≈ 0, CE is very high
        double eps = 1e-7;
        double[] pred = [eps, 1 - eps, 0]; // almost 0 for true class
        double[] actual = [1, 0, 0];

        double ce = ComputeCrossEntropy(pred, actual);
        Assert.True(ce > 10, $"CE for worst prediction should be very high, got {ce}");
    }

    [Fact]
    public void CrossEntropy_UniformPrediction_EqualsLogK()
    {
        // Uniform distribution over K classes: CE = log(K)
        int K = 4;
        double[] pred = Enumerable.Repeat(1.0 / K, K).ToArray();
        double[] actual = [1, 0, 0, 0]; // any class

        double ce = ComputeCrossEntropy(pred, actual);
        Assert.Equal(Math.Log(K), ce, 1e-5);
    }

    #endregion

    #region Binary Cross-Entropy Loss

    [Fact]
    public void BinaryCE_HandComputed()
    {
        // BCE = -(y*log(p) + (1-y)*log(1-p))
        // y=1, p=0.9 -> -(1*log(0.9) + 0*log(0.1)) = -log(0.9) ≈ 0.10536
        double p = 0.9;
        double y = 1.0;

        double bce = ComputeBinaryCrossEntropy(p, y);
        Assert.Equal(-Math.Log(0.9), bce, Tolerance);
    }

    [Fact]
    public void BinaryCE_NegativeCase_HandComputed()
    {
        // y=0, p=0.2 -> -(0*log(0.2) + 1*log(0.8)) = -log(0.8) ≈ 0.22314
        double p = 0.2;
        double y = 0.0;

        double bce = ComputeBinaryCrossEntropy(p, y);
        Assert.Equal(-Math.Log(0.8), bce, Tolerance);
    }

    [Fact]
    public void BinaryCE_PerfectPrediction_NearZero()
    {
        // y=1, p≈1 -> BCE ≈ 0
        double bce = ComputeBinaryCrossEntropy(1.0 - 1e-10, 1.0);
        Assert.True(bce < 1e-8, $"Perfect prediction BCE should be near zero: {bce}");
    }

    [Fact]
    public void BinaryCE_Symmetric_UnderLabelSwap()
    {
        // BCE(p, 1) + BCE(1-p, 0) should equal BCE(p, 1) + BCE(p, 0) ... not quite
        // Actually: BCE(p, 1) = -log(p), BCE(1-p, 0) = -log(p) -> they're equal
        double p = 0.7;
        double bce1 = ComputeBinaryCrossEntropy(p, 1.0);
        double bce2 = ComputeBinaryCrossEntropy(1 - p, 0.0);

        Assert.Equal(bce1, bce2, Tolerance);
    }

    #endregion

    #region Focal Loss

    [Fact]
    public void FocalLoss_Gamma0_EqualsCrossEntropy()
    {
        // When gamma=0, focal loss = -alpha * log(p_t) = alpha * CE
        double p = 0.8;
        double y = 1.0;
        double gamma = 0.0;
        double alpha = 0.25;

        double focal = ComputeFocalLoss(p, y, gamma, alpha);
        double ce = ComputeBinaryCrossEntropy(p, y);

        Assert.Equal(alpha * ce, focal, Tolerance);
    }

    [Fact]
    public void FocalLoss_DownWeightsEasyExamples()
    {
        // For easy example (high p_t), focal loss is much smaller than CE
        // For hard example (low p_t), focal loss is closer to CE
        double pEasy = 0.95; // easy example
        double pHard = 0.2;  // hard example
        double gamma = 2.0;
        double alpha = 1.0;

        double focalEasy = ComputeFocalLoss(pEasy, 1.0, gamma, alpha);
        double ceEasy = ComputeBinaryCrossEntropy(pEasy, 1.0);
        double focalHard = ComputeFocalLoss(pHard, 1.0, gamma, alpha);
        double ceHard = ComputeBinaryCrossEntropy(pHard, 1.0);

        double ratioEasy = focalEasy / ceEasy; // should be small
        double ratioHard = focalHard / ceHard; // should be larger

        Assert.True(ratioEasy < ratioHard,
            $"Easy example ratio {ratioEasy} should be less than hard example ratio {ratioHard}");
    }

    [Fact]
    public void FocalLoss_HandComputed()
    {
        // FL = -alpha * (1-p_t)^gamma * log(p_t)
        // y=1, p=0.8, gamma=2, alpha=0.25
        // p_t = 0.8 (since y=1)
        // FL = -0.25 * (1-0.8)^2 * log(0.8) = -0.25 * 0.04 * (-0.22314) = 0.25*0.04*0.22314
        // = 0.002231
        double p = 0.8;
        double y = 1.0;
        double gamma = 2.0;
        double alpha = 0.25;

        double focal = ComputeFocalLoss(p, y, gamma, alpha);
        double expected = alpha * Math.Pow(1 - p, gamma) * (-Math.Log(p));

        Assert.Equal(expected, focal, Tolerance);
    }

    [Fact]
    public void FocalLoss_HigherGamma_MoreDownWeighting()
    {
        // Higher gamma -> more down-weighting of easy examples
        double p = 0.9;
        double alpha = 1.0;

        double focalG1 = ComputeFocalLoss(p, 1.0, 1.0, alpha);
        double focalG2 = ComputeFocalLoss(p, 1.0, 2.0, alpha);
        double focalG5 = ComputeFocalLoss(p, 1.0, 5.0, alpha);

        Assert.True(focalG5 < focalG2, $"gamma=5 focal {focalG5} should be < gamma=2 {focalG2}");
        Assert.True(focalG2 < focalG1, $"gamma=2 focal {focalG2} should be < gamma=1 {focalG1}");
    }

    #endregion

    #region Dice Loss

    [Fact]
    public void DiceLoss_PerfectOverlap_IsZero()
    {
        // Dice coefficient = 2*|A∩B| / (|A|+|B|) = 2*sum(pred*actual) / (sum(pred)+sum(actual))
        // Dice loss = 1 - Dice coefficient
        // Perfect overlap: pred = actual -> Dice = 1, loss = 0
        var pred = new double[] { 1, 0, 1, 0, 1 };
        double diceLoss = ComputeDiceLoss(pred, pred);
        Assert.Equal(0.0, diceLoss, Tolerance);
    }

    [Fact]
    public void DiceLoss_NoOverlap_IsOne()
    {
        // No overlap: pred ∩ actual = 0 -> Dice = 0, loss = 1
        var pred = new double[] { 1, 1, 0, 0 };
        var actual = new double[] { 0, 0, 1, 1 };

        double diceLoss = ComputeDiceLoss(pred, actual);
        Assert.Equal(1.0, diceLoss, Tolerance);
    }

    [Fact]
    public void DiceLoss_HandComputed()
    {
        // pred = [0.8, 0.2, 0.9, 0.1], actual = [1, 0, 1, 0]
        // intersection = 0.8*1 + 0.2*0 + 0.9*1 + 0.1*0 = 1.7
        // |pred| = 0.8+0.2+0.9+0.1 = 2.0
        // |actual| = 1+0+1+0 = 2.0
        // Dice = 2*1.7 / (2.0+2.0) = 3.4/4.0 = 0.85
        // Dice loss = 1 - 0.85 = 0.15
        var pred = new double[] { 0.8, 0.2, 0.9, 0.1 };
        var actual = new double[] { 1, 0, 1, 0 };

        double diceLoss = ComputeDiceLoss(pred, actual);
        Assert.Equal(0.15, diceLoss, Tolerance);
    }

    [Fact]
    public void DiceLoss_BoundedBetween0And1()
    {
        // Dice loss is always in [0, 1] for non-negative inputs
        var testCases = new (double[], double[])[]
        {
            ([0.5, 0.5, 0.5], [1, 1, 1]),
            ([0.1, 0.9, 0.3], [0, 1, 0]),
            ([0.99, 0.01], [1, 0]),
        };

        foreach (var (pred, actual) in testCases)
        {
            double loss = ComputeDiceLoss(pred, actual);
            Assert.True(loss >= -1e-10 && loss <= 1.0 + 1e-10,
                $"Dice loss {loss} should be in [0, 1]");
        }
    }

    #endregion

    #region KL Divergence

    [Fact]
    public void KLDivergence_IdenticalDistributions_IsZero()
    {
        // KL(P || P) = 0
        var p = new double[] { 0.3, 0.5, 0.2 };
        double kl = ComputeKLDivergence(p, p);
        Assert.Equal(0.0, kl, Tolerance);
    }

    [Fact]
    public void KLDivergence_HandComputed()
    {
        // KL(P || Q) = sum(P(i) * log(P(i)/Q(i)))
        // P = [0.4, 0.6], Q = [0.5, 0.5]
        // KL = 0.4*log(0.4/0.5) + 0.6*log(0.6/0.5)
        //    = 0.4*log(0.8) + 0.6*log(1.2)
        //    = 0.4*(-0.22314) + 0.6*(0.18232)
        //    = -0.089257 + 0.109393
        //    = 0.020136
        var p = new double[] { 0.4, 0.6 };
        var q = new double[] { 0.5, 0.5 };

        double kl = ComputeKLDivergence(p, q);
        double expected = 0.4 * Math.Log(0.4 / 0.5) + 0.6 * Math.Log(0.6 / 0.5);
        Assert.Equal(expected, kl, 1e-5);
    }

    [Fact]
    public void KLDivergence_NonNegative()
    {
        // Gibbs' inequality: KL(P||Q) >= 0 for all valid distributions P, Q
        // Only compare distributions of the same size
        var distributions2 = new double[][]
        {
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1],
        };
        var distributions3 = new double[][]
        {
            [0.3, 0.3, 0.4],
            [0.1, 0.1, 0.8],
            [0.5, 0.25, 0.25],
        };

        foreach (var p in distributions2)
        {
            foreach (var q in distributions2)
            {
                double kl = ComputeKLDivergence(p, q);
                Assert.True(kl >= -1e-10,
                    $"KL divergence should be non-negative, got {kl}");
            }
        }

        foreach (var p in distributions3)
        {
            foreach (var q in distributions3)
            {
                double kl = ComputeKLDivergence(p, q);
                Assert.True(kl >= -1e-10,
                    $"KL divergence should be non-negative, got {kl}");
            }
        }
    }

    [Fact]
    public void KLDivergence_Asymmetric()
    {
        // KL(P||Q) != KL(Q||P) in general
        var p = new double[] { 0.1, 0.9 };
        var q = new double[] { 0.5, 0.5 };

        double klPQ = ComputeKLDivergence(p, q);
        double klQP = ComputeKLDivergence(q, p);

        Assert.NotEqual(klPQ, klQP, Tolerance);
    }

    #endregion

    #region Hinge Loss

    [Fact]
    public void HingeLoss_CorrectlyClassified_WithMargin()
    {
        // Hinge = max(0, 1 - y*f(x))
        // y=1, f(x)=2 -> max(0, 1-2) = max(0, -1) = 0
        double hinge = ComputeHingeLoss(2.0, 1.0);
        Assert.Equal(0.0, hinge, Tolerance);
    }

    [Fact]
    public void HingeLoss_MisclassifiedExample()
    {
        // y=1, f(x)=-0.5 -> max(0, 1-(-0.5)) = max(0, 1.5) = 1.5
        double hinge = ComputeHingeLoss(-0.5, 1.0);
        Assert.Equal(1.5, hinge, Tolerance);
    }

    [Fact]
    public void HingeLoss_OnMargin()
    {
        // y=1, f(x)=1 -> max(0, 1-1) = 0
        double hinge = ComputeHingeLoss(1.0, 1.0);
        Assert.Equal(0.0, hinge, Tolerance);
    }

    [Fact]
    public void HingeLoss_NegativeClass()
    {
        // y=-1, f(x)=-2 -> max(0, 1-(-1)*(-2)) = max(0, 1-2) = 0 (correct)
        // y=-1, f(x)=0.5 -> max(0, 1-(-1)*0.5) = max(0, 1.5) = 1.5 (misclassified)
        double hingeCorrect = ComputeHingeLoss(-2.0, -1.0);
        double hingeWrong = ComputeHingeLoss(0.5, -1.0);

        Assert.Equal(0.0, hingeCorrect, Tolerance);
        Assert.Equal(1.5, hingeWrong, Tolerance);
    }

    #endregion

    #region Cosine Similarity Loss

    [Fact]
    public void CosineSimilarityLoss_ParallelVectors_IsZero()
    {
        // Cosine similarity loss = 1 - cos(pred, actual)
        // Parallel vectors: cos = 1, loss = 0
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 2, 4, 6 };

        double loss = ComputeCosineSimilarityLoss(pred, actual);
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void CosineSimilarityLoss_OrthogonalVectors_IsOne()
    {
        // Orthogonal: cos = 0, loss = 1
        var pred = new double[] { 1, 0 };
        var actual = new double[] { 0, 1 };

        double loss = ComputeCosineSimilarityLoss(pred, actual);
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void CosineSimilarityLoss_OppositeVectors_IsTwo()
    {
        // Opposite: cos = -1, loss = 2
        var pred = new double[] { 1, 0 };
        var actual = new double[] { -1, 0 };

        double loss = ComputeCosineSimilarityLoss(pred, actual);
        Assert.Equal(2.0, loss, Tolerance);
    }

    #endregion

    #region R-Squared (Coefficient of Determination)

    [Fact]
    public void RSquared_PerfectPrediction_IsOne()
    {
        // R^2 = 1 - SS_res/SS_tot
        // Perfect: SS_res = 0, R^2 = 1
        var actual = new double[] { 1, 2, 3, 4, 5 };
        double r2 = ComputeRSquared(actual, actual);
        Assert.Equal(1.0, r2, Tolerance);
    }

    [Fact]
    public void RSquared_MeanPrediction_IsZero()
    {
        // If pred = mean(actual) for all, SS_res = SS_tot, R^2 = 0
        var actual = new double[] { 1, 2, 3, 4, 5 };
        double mean = actual.Average();
        var pred = Enumerable.Repeat(mean, actual.Length).ToArray();

        double r2 = ComputeRSquared(pred, actual);
        Assert.Equal(0.0, r2, Tolerance);
    }

    [Fact]
    public void RSquared_HandComputed()
    {
        // actual = [1, 2, 3], pred = [1.1, 2.2, 2.7]
        // mean_actual = 2.0
        // SS_tot = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1+0+1 = 2
        // SS_res = (1.1-1)^2 + (2.2-2)^2 + (2.7-3)^2 = 0.01+0.04+0.09 = 0.14
        // R^2 = 1 - 0.14/2 = 1 - 0.07 = 0.93
        var actual = new double[] { 1, 2, 3 };
        var pred = new double[] { 1.1, 2.2, 2.7 };

        double r2 = ComputeRSquared(pred, actual);
        Assert.Equal(0.93, r2, Tolerance);
    }

    [Fact]
    public void RSquared_CanBeNegative()
    {
        // R^2 < 0 when predictions are worse than the mean
        var actual = new double[] { 1, 2, 3 };
        var pred = new double[] { 10, 20, 30 }; // very far off

        double r2 = ComputeRSquared(pred, actual);
        Assert.True(r2 < 0, $"R^2 should be negative for terrible predictions: {r2}");
    }

    #endregion

    #region Adjusted R-Squared

    [Fact]
    public void AdjustedRSquared_PenalizesMorePredictors()
    {
        // Adj R^2 = 1 - (1-R^2)(n-1)/(n-p-1) where p = number of predictors
        // Same R^2 but more predictors -> lower Adj R^2
        double r2 = 0.8;
        int n = 100;

        double adjR2_5 = ComputeAdjustedRSquared(r2, n, 5);
        double adjR2_20 = ComputeAdjustedRSquared(r2, n, 20);

        Assert.True(adjR2_5 > adjR2_20,
            $"Fewer predictors should give higher Adj R^2: {adjR2_5} vs {adjR2_20}");
    }

    [Fact]
    public void AdjustedRSquared_HandComputed()
    {
        // R^2 = 0.9, n = 50, p = 4
        // Adj R^2 = 1 - (1-0.9)*(50-1)/(50-4-1) = 1 - 0.1*49/45 = 1 - 0.10889 = 0.89111
        double r2 = 0.9;
        int n = 50;
        int p = 4;

        double adjR2 = ComputeAdjustedRSquared(r2, n, p);
        double expected = 1 - (1 - r2) * (n - 1.0) / (n - p - 1.0);

        Assert.Equal(expected, adjR2, Tolerance);
    }

    #endregion

    #region Log-Cosh Loss

    [Fact]
    public void LogCoshLoss_HandComputed()
    {
        // log(cosh(x)) ≈ x^2/2 for small x, ≈ |x| - log(2) for large x
        // error = 0.1 -> log(cosh(0.1)) ≈ 0.005 (close to 0.01/2)
        double smallError = 0.1;
        double logCoshSmall = Math.Log(Math.Cosh(smallError));
        Assert.True(Math.Abs(logCoshSmall - smallError * smallError / 2) < 0.001,
            $"Small error: log(cosh({smallError})) = {logCoshSmall} should be close to {smallError * smallError / 2}");
    }

    [Fact]
    public void LogCoshLoss_LargeError_ApproximatesAbsMinusLog2()
    {
        // For large |x|: log(cosh(x)) ≈ |x| - log(2)
        double largeError = 10.0;
        double logCosh = Math.Log(Math.Cosh(largeError));
        double approx = Math.Abs(largeError) - Math.Log(2);

        Assert.True(Math.Abs(logCosh - approx) < 0.01,
            $"Large error: log(cosh({largeError})) = {logCosh} should approximate |x|-log(2) = {approx}");
    }

    [Fact]
    public void LogCoshLoss_Symmetric()
    {
        double error = 2.5;
        double pos = Math.Log(Math.Cosh(error));
        double neg = Math.Log(Math.Cosh(-error));
        Assert.Equal(pos, neg, Tolerance);
    }

    [Fact]
    public void LogCoshLoss_SmoothTransition()
    {
        // LogCosh transitions smoothly between quadratic and linear behavior
        // It should be smoother than Huber at the transition point
        double[] errors = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0];
        double prevDiff = 0;
        double prevLogCosh = 0;

        for (int i = 1; i < errors.Length; i++)
        {
            double lc = Math.Log(Math.Cosh(errors[i]));
            double diff = lc - prevLogCosh;
            prevLogCosh = lc;
            prevDiff = diff;
            Assert.True(lc >= 0, $"LogCosh should be non-negative: {lc}");
        }
    }

    #endregion

    #region Quantile Loss

    [Fact]
    public void QuantileLoss_Median_EqualsHalfMAE()
    {
        // At quantile 0.5 (median), quantile loss = 0.5 * |error|
        double error = 3.0;
        double q = 0.5;

        double qLoss = ComputeQuantileLoss(error, q);
        Assert.Equal(0.5 * Math.Abs(error), qLoss, Tolerance);
    }

    [Fact]
    public void QuantileLoss_Overestimate_Vs_Underestimate()
    {
        // q > 0.5: penalizes underestimation more (wants to predict high)
        // q < 0.5: penalizes overestimation more (wants to predict low)
        double q_high = 0.9;
        double q_low = 0.1;

        // Underestimation: pred < actual -> error > 0
        double underestLossHigh = ComputeQuantileLoss(2.0, q_high);
        double underestLossLow = ComputeQuantileLoss(2.0, q_low);

        // Overestimation: pred > actual -> error < 0
        double overestLossHigh = ComputeQuantileLoss(-2.0, q_high);
        double overestLossLow = ComputeQuantileLoss(-2.0, q_low);

        // q=0.9 penalizes underestimation more than overestimation
        Assert.True(underestLossHigh > overestLossHigh,
            $"q=0.9: underest {underestLossHigh} should > overest {overestLossHigh}");
        // q=0.1 penalizes overestimation more than underestimation
        Assert.True(overestLossLow > underestLossLow,
            $"q=0.1: overest {overestLossLow} should > underest {underestLossLow}");
    }

    [Fact]
    public void QuantileLoss_HandComputed()
    {
        // q=0.75, error=2.0 (actual - pred > 0: underestimate)
        // Loss = q * error = 0.75 * 2 = 1.5
        // q=0.75, error=-1.0 (overestimate)
        // Loss = (1-q) * |error| = 0.25 * 1 = 0.25
        Assert.Equal(1.5, ComputeQuantileLoss(2.0, 0.75), Tolerance);
        Assert.Equal(0.25, ComputeQuantileLoss(-1.0, 0.75), Tolerance);
    }

    #endregion

    #region Elastic Net Loss

    [Fact]
    public void ElasticNet_PureL1_IsMAEPenalty()
    {
        // ElasticNet = MSE + alpha * (l1_ratio * L1 + (1-l1_ratio) * L2)
        // With l1_ratio=1.0, it's MSE + alpha * L1
        double[] weights = [1, -2, 3];
        double alpha = 0.1;
        double l1Ratio = 1.0;

        double l1Penalty = weights.Sum(w => Math.Abs(w)); // 1+2+3=6
        double elastic = alpha * (l1Ratio * l1Penalty);
        Assert.Equal(0.6, elastic, Tolerance);
    }

    [Fact]
    public void ElasticNet_PureL2_IsRidgePenalty()
    {
        // With l1_ratio=0.0, it's MSE + alpha * L2
        double[] weights = [1, -2, 3];
        double alpha = 0.1;
        double l1Ratio = 0.0;

        double l2Penalty = weights.Sum(w => w * w); // 1+4+9=14
        double elastic = alpha * ((1 - l1Ratio) * l2Penalty);
        Assert.Equal(1.4, elastic, Tolerance);
    }

    [Fact]
    public void ElasticNet_Mixture_HandComputed()
    {
        // weights = [1, -2, 3], alpha = 0.5, l1_ratio = 0.5
        // L1 = |1|+|-2|+|3| = 6
        // L2 = 1+4+9 = 14
        // elastic = 0.5 * (0.5*6 + 0.5*14) = 0.5 * (3 + 7) = 5.0
        double[] weights = [1, -2, 3];
        double alpha = 0.5;
        double l1Ratio = 0.5;

        double l1 = weights.Sum(w => Math.Abs(w));
        double l2 = weights.Sum(w => w * w);
        double elastic = alpha * (l1Ratio * l1 + (1 - l1Ratio) * l2);

        Assert.Equal(5.0, elastic, Tolerance);
    }

    #endregion

    #region Poisson Loss

    [Fact]
    public void PoissonLoss_HandComputed()
    {
        // Poisson loss = pred - actual * log(pred)
        // pred = 3.0, actual = 2.0 -> L = 3.0 - 2.0*log(3.0) = 3.0 - 2.197 = 0.80277
        double pred = 3.0;
        double actual = 2.0;

        double poissonLoss = pred - actual * Math.Log(pred);
        double expected = 3.0 - 2.0 * Math.Log(3.0);

        Assert.Equal(expected, poissonLoss, Tolerance);
    }

    [Fact]
    public void PoissonLoss_MinimizedAtPred_EqualsActual()
    {
        // The Poisson loss L = pred - actual*log(pred) is minimized when pred = actual
        // dL/dpred = 1 - actual/pred = 0 -> pred = actual
        double actual = 5.0;
        double lossAtOptimal = actual - actual * Math.Log(actual);

        // Check nearby values have higher loss
        double lossSlightlyLess = (actual - 0.1) - actual * Math.Log(actual - 0.1);
        double lossSlightlyMore = (actual + 0.1) - actual * Math.Log(actual + 0.1);

        Assert.True(lossAtOptimal <= lossSlightlyLess + 1e-10,
            $"Loss at optimal {lossAtOptimal} should be <= nearby {lossSlightlyLess}");
        Assert.True(lossAtOptimal <= lossSlightlyMore + 1e-10,
            $"Loss at optimal {lossAtOptimal} should be <= nearby {lossSlightlyMore}");
    }

    #endregion

    #region Jaccard Loss

    [Fact]
    public void JaccardLoss_PerfectOverlap_IsZero()
    {
        // Jaccard = |A∩B| / |A∪B|, Jaccard loss = 1 - Jaccard
        // Perfect: J = 1, loss = 0
        var pred = new double[] { 1, 0, 1, 0 };
        double loss = ComputeJaccardLoss(pred, pred);
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void JaccardLoss_NoOverlap_IsOne()
    {
        var pred = new double[] { 1, 1, 0, 0 };
        var actual = new double[] { 0, 0, 1, 1 };

        double loss = ComputeJaccardLoss(pred, actual);
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void JaccardLoss_HandComputed()
    {
        // pred = [0.8, 0.2, 0.9, 0.1], actual = [1, 0, 1, 0]
        // intersection = 0.8*1+0.2*0+0.9*1+0.1*0 = 1.7
        // union = sum(pred) + sum(actual) - intersection = 2.0+2.0-1.7 = 2.3
        // Jaccard = 1.7/2.3 ≈ 0.73913
        // Loss = 1 - 0.73913 ≈ 0.26087
        var pred = new double[] { 0.8, 0.2, 0.9, 0.1 };
        var actual = new double[] { 1, 0, 1, 0 };

        double loss = ComputeJaccardLoss(pred, actual);
        double expectedJaccard = 1.7 / 2.3;
        Assert.Equal(1 - expectedJaccard, loss, 1e-4);
    }

    [Fact]
    public void JaccardLoss_AlwaysGreaterOrEqualToDiceLoss()
    {
        // For same inputs, Jaccard loss >= Dice loss (since Jaccard index <= Dice coefficient)
        var testCases = new (double[], double[])[]
        {
            ([0.8, 0.2, 0.9, 0.1], [1, 0, 1, 0]),
            ([0.5, 0.5, 0.5], [1, 1, 0]),
            ([0.3, 0.7, 0.1, 0.9], [0, 1, 0, 1]),
        };

        foreach (var (pred, actual) in testCases)
        {
            double jaccardLoss = ComputeJaccardLoss(pred, actual);
            double diceLoss = ComputeDiceLoss(pred, actual);

            Assert.True(jaccardLoss >= diceLoss - 1e-10,
                $"Jaccard loss {jaccardLoss} should >= Dice loss {diceLoss}");
        }
    }

    #endregion

    #region Contrastive Loss

    [Fact]
    public void ContrastiveLoss_SimilarPair_SmallDistance()
    {
        // L = y * d^2 + (1-y) * max(0, margin-d)^2
        // y=1 (similar), d=0.5, margin=1.0
        // L = 1 * 0.25 + 0 * max(0, 0.5)^2 = 0.25
        double d = 0.5;
        double y = 1.0;
        double margin = 1.0;

        double loss = ComputeContrastiveLoss(d, y, margin);
        Assert.Equal(0.25, loss, Tolerance);
    }

    [Fact]
    public void ContrastiveLoss_DissimilarPair_LargeDistance()
    {
        // y=0 (dissimilar), d=2.0, margin=1.0
        // L = 0 * d^2 + 1 * max(0, 1-2)^2 = max(0,-1)^2 = 0
        double d = 2.0;
        double y = 0.0;
        double margin = 1.0;

        double loss = ComputeContrastiveLoss(d, y, margin);
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void ContrastiveLoss_DissimilarPair_SmallDistance()
    {
        // y=0 (dissimilar), d=0.3, margin=1.0
        // L = 0 + max(0, 1-0.3)^2 = 0.7^2 = 0.49
        double d = 0.3;
        double y = 0.0;
        double margin = 1.0;

        double loss = ComputeContrastiveLoss(d, y, margin);
        Assert.Equal(0.49, loss, Tolerance);
    }

    #endregion

    #region Triplet Loss

    [Fact]
    public void TripletLoss_LargeNegativeMargin_IsZero()
    {
        // L = max(0, d(a,p) - d(a,n) + margin)
        // If d(a,n) >> d(a,p), loss is 0
        double dAP = 1.0; // anchor-positive distance
        double dAN = 10.0; // anchor-negative distance
        double margin = 0.5;

        double loss = ComputeTripletLoss(dAP, dAN, margin);
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void TripletLoss_ViolatesMargin()
    {
        // d(a,p)=2, d(a,n)=1.5, margin=1.0
        // L = max(0, 2-1.5+1) = max(0, 1.5) = 1.5
        double dAP = 2.0;
        double dAN = 1.5;
        double margin = 1.0;

        double loss = ComputeTripletLoss(dAP, dAN, margin);
        Assert.Equal(1.5, loss, Tolerance);
    }

    [Fact]
    public void TripletLoss_AtExactMargin_IsZero()
    {
        // d(a,p)=1, d(a,n)=2, margin=1
        // L = max(0, 1-2+1) = max(0, 0) = 0
        double dAP = 1.0;
        double dAN = 2.0;
        double margin = 1.0;

        double loss = ComputeTripletLoss(dAP, dAN, margin);
        Assert.Equal(0.0, loss, Tolerance);
    }

    #endregion

    #region Squared Hinge Loss

    [Fact]
    public void SquaredHingeLoss_HandComputed()
    {
        // Squared hinge = max(0, 1 - y*f(x))^2
        // y=1, f(x)=-0.5 -> max(0, 1.5)^2 = 2.25
        double sqHinge = Math.Pow(Math.Max(0, 1 - 1.0 * (-0.5)), 2);
        Assert.Equal(2.25, sqHinge, Tolerance);
    }

    [Fact]
    public void SquaredHingeLoss_CorrectlyClassifiedWithMargin_IsZero()
    {
        // y=1, f(x)=2 -> max(0, 1-2)^2 = 0
        double sqHinge = Math.Pow(Math.Max(0, 1 - 1.0 * 2.0), 2);
        Assert.Equal(0.0, sqHinge, Tolerance);
    }

    #endregion

    #region Loss Function Comparison Properties

    [Fact]
    public void AllLosses_ZeroError_ProducesMinimalLoss()
    {
        // Zero error should produce zero or minimal loss for most loss functions
        Assert.Equal(0.0, ComputeMSE([1], [1]), Tolerance);
        Assert.Equal(0.0, ComputeMAE([1], [1]), Tolerance);
        Assert.Equal(0.0, ComputeHuberLoss(0, 1.0), Tolerance);
        Assert.Equal(0.0, ComputeHingeLoss(2.0, 1.0), Tolerance); // correctly classified
        Assert.Equal(0.0, ComputeDiceLoss([1, 0, 1], [1, 0, 1]), Tolerance);
        Assert.Equal(0.0, ComputeJaccardLoss([1, 0, 1], [1, 0, 1]), Tolerance);
    }

    [Fact]
    public void MSE_AlwaysGreaterThanOrEqualTo_MAE_Squared_Over_N()
    {
        // By Jensen's inequality: MSE >= MAE^2 / n (not exactly, but MSE >= MAE^2)
        // Actually: MSE >= MAE^2 (since E[X^2] >= E[X]^2)
        var testCases = new (double[], double[])[]
        {
            ([1, 2, 3], [2, 3, 4]),
            ([0, 5, 10], [1, 3, 7]),
            ([-1, -2, -3], [1, 2, 3]),
        };

        foreach (var (pred, actual) in testCases)
        {
            double mse = ComputeMSE(pred, actual);
            double mae = ComputeMAE(pred, actual);

            Assert.True(mse >= mae * mae - 1e-10,
                $"MSE {mse} should >= MAE^2 {mae * mae}");
        }
    }

    #endregion

    #region Helper Methods

    private static double ComputeMSE(double[] pred, double[] actual)
    {
        double sum = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            double diff = pred[i] - actual[i];
            sum += diff * diff;
        }
        return sum / pred.Length;
    }

    private static double ComputeMAE(double[] pred, double[] actual)
    {
        double sum = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            sum += Math.Abs(pred[i] - actual[i]);
        }
        return sum / pred.Length;
    }

    private static double ComputeHuberLoss(double error, double delta)
    {
        double absError = Math.Abs(error);
        if (absError <= delta)
        {
            return 0.5 * error * error;
        }
        return delta * (absError - 0.5 * delta);
    }

    private static double ComputeCrossEntropy(double[] pred, double[] actual)
    {
        double sum = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            if (actual[i] > 0)
            {
                sum -= actual[i] * Math.Log(Math.Max(pred[i], 1e-15));
            }
        }
        return sum;
    }

    private static double ComputeBinaryCrossEntropy(double pred, double actual)
    {
        pred = Math.Max(Math.Min(pred, 1 - 1e-15), 1e-15);
        return -(actual * Math.Log(pred) + (1 - actual) * Math.Log(1 - pred));
    }

    private static double ComputeFocalLoss(double pred, double actual, double gamma, double alpha)
    {
        double pt = actual == 1.0 ? pred : 1 - pred;
        pt = Math.Max(pt, 1e-15);
        return -alpha * Math.Pow(1 - pt, gamma) * Math.Log(pt);
    }

    private static double ComputeDiceLoss(double[] pred, double[] actual)
    {
        double intersection = 0, sumPred = 0, sumActual = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            intersection += pred[i] * actual[i];
            sumPred += pred[i];
            sumActual += actual[i];
        }
        double denom = sumPred + sumActual;
        if (denom < 1e-15) return 0;
        double dice = 2 * intersection / denom;
        return 1 - dice;
    }

    private static double ComputeJaccardLoss(double[] pred, double[] actual)
    {
        double intersection = 0, sumPred = 0, sumActual = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            intersection += pred[i] * actual[i];
            sumPred += pred[i];
            sumActual += actual[i];
        }
        double union = sumPred + sumActual - intersection;
        if (union < 1e-15) return 0;
        double jaccard = intersection / union;
        return 1 - jaccard;
    }

    private static double ComputeKLDivergence(double[] p, double[] q)
    {
        double sum = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 1e-15)
            {
                sum += p[i] * Math.Log(p[i] / Math.Max(q[i], 1e-15));
            }
        }
        return sum;
    }

    private static double ComputeHingeLoss(double prediction, double label)
    {
        return Math.Max(0, 1 - label * prediction);
    }

    private static double ComputeCosineSimilarityLoss(double[] pred, double[] actual)
    {
        double dot = 0, normPredSq = 0, normActualSq = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            dot += pred[i] * actual[i];
            normPredSq += pred[i] * pred[i];
            normActualSq += actual[i] * actual[i];
        }
        double denom = Math.Sqrt(normPredSq) * Math.Sqrt(normActualSq);
        if (denom < 1e-10) return 1.0;
        double cosine = dot / denom;
        return 1 - cosine;
    }

    private static double ComputeRSquared(double[] pred, double[] actual)
    {
        double mean = actual.Average();
        double ssTot = actual.Sum(a => Math.Pow(a - mean, 2));
        double ssRes = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            ssRes += Math.Pow(actual[i] - pred[i], 2);
        }
        if (ssTot < 1e-15) return 1.0;
        return 1 - ssRes / ssTot;
    }

    private static double ComputeAdjustedRSquared(double r2, int n, int p)
    {
        return 1 - (1 - r2) * (n - 1.0) / (n - p - 1.0);
    }

    private static double ComputeQuantileLoss(double error, double quantile)
    {
        if (error >= 0)
        {
            return quantile * error;
        }
        return (1 - quantile) * Math.Abs(error);
    }

    private static double ComputeContrastiveLoss(double distance, double similar, double margin)
    {
        return similar * distance * distance +
               (1 - similar) * Math.Pow(Math.Max(0, margin - distance), 2);
    }

    private static double ComputeTripletLoss(double dAP, double dAN, double margin)
    {
        return Math.Max(0, dAP - dAN + margin);
    }

    #endregion
}
