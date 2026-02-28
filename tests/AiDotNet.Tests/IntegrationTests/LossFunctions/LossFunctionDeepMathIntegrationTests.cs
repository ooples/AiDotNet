using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Deep mathematical correctness tests for loss functions. Each test verifies
/// hand-calculated expected values against the implementation.
/// </summary>
public class LossFunctionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;

    #region Helper

    private static Vector<double> V(params double[] values) => new(values);

    #endregion

    #region MSE - Mean Squared Error

    [Fact]
    public void MSE_HandCalculated_TwoElements()
    {
        // predicted=[1, 3], actual=[2, 5]
        // errors = [-1, -2], squared = [1, 4]
        // MSE = (1+4)/2 = 2.5
        var loss = new MeanSquaredErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 3), V(2, 5));
        Assert.Equal(2.5, result, Tolerance);
    }

    [Fact]
    public void MSE_PerfectPrediction_IsZero()
    {
        var loss = new MeanSquaredErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 2, 3), V(1, 2, 3));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MSE_Symmetric()
    {
        // MSE(p, a) should equal MSE(a, p) since (p-a)² = (a-p)²
        var loss = new MeanSquaredErrorLoss<double>();
        var p = V(1.0, 2.0, 3.0);
        var a = V(4.0, 5.0, 6.0);
        Assert.Equal(loss.CalculateLoss(p, a), loss.CalculateLoss(a, p), Tolerance);
    }

    [Fact]
    public void MSE_Derivative_HandCalculated()
    {
        // predicted=[3, 5], actual=[1, 2]
        // dMSE/dp_i = 2*(p_i - a_i)/n
        // = [2*(3-1)/2, 2*(5-2)/2] = [2.0, 3.0]
        var loss = new MeanSquaredErrorLoss<double>();
        var grad = loss.CalculateDerivative(V(3, 5), V(1, 2));
        Assert.Equal(2.0, grad[0], Tolerance);
        Assert.Equal(3.0, grad[1], Tolerance);
    }

    [Fact]
    public void MSE_SingleElement()
    {
        // predicted=[2], actual=[5] → MSE = (2-5)²/1 = 9
        var loss = new MeanSquaredErrorLoss<double>();
        Assert.Equal(9.0, loss.CalculateLoss(V(2), V(5)), Tolerance);
    }

    #endregion

    #region Huber Loss

    [Fact]
    public void Huber_SmallError_IsQuadratic()
    {
        // delta=1.0, error=0.3 (< delta)
        // Huber = 0.5 * 0.3² = 0.045
        // For single element: 0.045 / 1 = 0.045
        var loss = new HuberLoss<double>(delta: 1.0);
        double result = loss.CalculateLoss(V(1.3), V(1.0));
        Assert.Equal(0.045, result, Tolerance);
    }

    [Fact]
    public void Huber_LargeError_IsLinear()
    {
        // delta=1.0, error=3.0 (> delta)
        // Huber = delta * (|error| - 0.5*delta) = 1.0 * (3.0 - 0.5) = 2.5
        var loss = new HuberLoss<double>(delta: 1.0);
        double result = loss.CalculateLoss(V(4.0), V(1.0));
        Assert.Equal(2.5, result, Tolerance);
    }

    [Fact]
    public void Huber_AtDeltaBoundary_QuadraticAndLinearMatch()
    {
        // At exactly error = delta, both branches should give same value
        // Quadratic: 0.5 * delta² = 0.5 * 1.0² = 0.5
        // Linear: delta * (|delta| - 0.5*delta) = 1.0 * (1.0 - 0.5) = 0.5
        var loss = new HuberLoss<double>(delta: 1.0);
        double result = loss.CalculateLoss(V(2.0), V(1.0));
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void Huber_MultipleElements_HandCalculated()
    {
        // delta=1.0, predicted=[1.3, 4.0], actual=[1.0, 1.0]
        // errors = [0.3, 3.0]
        // Element 0: |0.3| <= 1.0 → 0.5*0.09 = 0.045
        // Element 1: |3.0| > 1.0 → 1.0*(3.0-0.5) = 2.5
        // Average: (0.045 + 2.5)/2 = 1.2725
        var loss = new HuberLoss<double>(delta: 1.0);
        double result = loss.CalculateLoss(V(1.3, 4.0), V(1.0, 1.0));
        Assert.Equal(1.2725, result, Tolerance);
    }

    [Fact]
    public void Huber_CustomDelta_HandCalculated()
    {
        // delta=0.5, error=1.0 (> delta)
        // Huber = 0.5 * (1.0 - 0.25) = 0.375
        var loss = new HuberLoss<double>(delta: 0.5);
        double result = loss.CalculateLoss(V(2.0), V(1.0));
        Assert.Equal(0.375, result, Tolerance);
    }

    [Fact]
    public void Huber_Derivative_QuadraticRegion()
    {
        // delta=1.0, error=0.5 (< delta)
        // Derivative = diff / n = 0.5 / 1 = 0.5
        var loss = new HuberLoss<double>(delta: 1.0);
        var grad = loss.CalculateDerivative(V(1.5), V(1.0));
        Assert.Equal(0.5, grad[0], Tolerance);
    }

    [Fact]
    public void Huber_Derivative_LinearRegion()
    {
        // delta=1.0, error=3.0 (> delta), sign=+1
        // Derivative = delta * sign(diff) / n = 1.0 * 1 / 1 = 1.0
        var loss = new HuberLoss<double>(delta: 1.0);
        var grad = loss.CalculateDerivative(V(4.0), V(1.0));
        Assert.Equal(1.0, grad[0], Tolerance);
    }

    [Fact]
    public void Huber_PerfectPrediction_IsZero()
    {
        var loss = new HuberLoss<double>(delta: 1.0);
        Assert.Equal(0.0, loss.CalculateLoss(V(5, 10), V(5, 10)), Tolerance);
    }

    #endregion

    #region Binary Cross Entropy

    [Fact]
    public void BCE_HandCalculated()
    {
        // predicted=[0.9], actual=[1.0]
        // BCE = -[1*log(0.9) + 0*log(0.1)] / 1
        //     = -log(0.9) ≈ 0.10536051565
        var loss = new BinaryCrossEntropyLoss<double>();
        double result = loss.CalculateLoss(V(0.9), V(1.0));
        Assert.Equal(-Math.Log(0.9), result, 1e-6);
    }

    [Fact]
    public void BCE_PerfectPrediction_NearZero()
    {
        // predicted ≈ 1.0 when actual = 1.0 → loss ≈ 0
        // Using 0.999999 to avoid exact 1.0
        var loss = new BinaryCrossEntropyLoss<double>();
        double result = loss.CalculateLoss(V(0.999999), V(1.0));
        Assert.True(result < 0.001, $"Loss should be near zero, got {result}");
    }

    [Fact]
    public void BCE_ConfidentWrongPrediction_HighLoss()
    {
        // predicted=0.01 when actual=1.0 → loss = -log(0.01) ≈ 4.605
        var loss = new BinaryCrossEntropyLoss<double>();
        double result = loss.CalculateLoss(V(0.01), V(1.0));
        Assert.True(result > 4.0, $"Loss should be high for confident mistake, got {result}");
    }

    [Fact]
    public void BCE_MultiElement_HandCalculated()
    {
        // predicted=[0.9, 0.2], actual=[1, 0]
        // BCE = -[ (1*log(0.9) + 0*log(0.1)) + (0*log(0.2) + 1*log(0.8)) ] / 2
        //     = -[log(0.9) + log(0.8)] / 2
        //     = -(−0.1054 + (−0.2231)) / 2 = 0.16425...
        var loss = new BinaryCrossEntropyLoss<double>();
        double result = loss.CalculateLoss(V(0.9, 0.2), V(1.0, 0.0));
        double expected = -(Math.Log(0.9) + Math.Log(0.8)) / 2.0;
        Assert.Equal(expected, result, 1e-6);
    }

    [Fact]
    public void BCE_Derivative_HandCalculated()
    {
        // predicted=[0.8], actual=[1.0]
        // Derivative = (p - y) / (p*(1-p)) / n
        // = (0.8 - 1.0) / (0.8 * 0.2) / 1
        // = -0.2 / 0.16 = -1.25
        var loss = new BinaryCrossEntropyLoss<double>();
        var grad = loss.CalculateDerivative(V(0.8), V(1.0));
        Assert.Equal(-1.25, grad[0], 1e-4);
    }

    #endregion

    #region KL Divergence

    [Fact]
    public void KL_IdenticalDistributions_IsZero()
    {
        // KL(P||Q) = sum(P * log(P/Q)) = 0 when P = Q
        var loss = new KullbackLeiblerDivergence<double>();
        double result = loss.CalculateLoss(V(0.5, 0.5), V(0.5, 0.5));
        Assert.Equal(0.0, result, 1e-10);
    }

    [Fact]
    public void KL_HandCalculated()
    {
        // P = actual = [0.4, 0.6], Q = predicted = [0.5, 0.5]
        // KL(P||Q) = 0.4*log(0.4/0.5) + 0.6*log(0.6/0.5)
        //          = 0.4*log(0.8) + 0.6*log(1.2)
        //          = 0.4*(−0.22314) + 0.6*(0.18232)
        //          = −0.08926 + 0.10939 = 0.02014
        var loss = new KullbackLeiblerDivergence<double>();
        double result = loss.CalculateLoss(V(0.5, 0.5), V(0.4, 0.6));
        double expected = 0.4 * Math.Log(0.4 / 0.5) + 0.6 * Math.Log(0.6 / 0.5);
        Assert.Equal(expected, result, 1e-8);
    }

    [Fact]
    public void KL_NonNegative()
    {
        // KL divergence is always >= 0 (Gibbs' inequality)
        var loss = new KullbackLeiblerDivergence<double>();
        double result = loss.CalculateLoss(V(0.3, 0.7), V(0.6, 0.4));
        Assert.True(result >= -1e-10, $"KL divergence should be non-negative, got {result}");
    }

    [Fact]
    public void KL_NotSymmetric()
    {
        // KL(P||Q) != KL(Q||P) in general
        var loss = new KullbackLeiblerDivergence<double>();
        double kl_pq = loss.CalculateLoss(V(0.3, 0.7), V(0.6, 0.4));
        double kl_qp = loss.CalculateLoss(V(0.6, 0.4), V(0.3, 0.7));
        Assert.NotEqual(kl_pq, kl_qp);
    }

    [Fact]
    public void KL_Derivative_HandCalculated()
    {
        // d KL(P||Q) / dQ_i = -P_i / Q_i
        // P=[0.4, 0.6], Q=[0.5, 0.5]
        // derivative = [-0.4/0.5, -0.6/0.5] = [-0.8, -1.2]
        var loss = new KullbackLeiblerDivergence<double>();
        var grad = loss.CalculateDerivative(V(0.5, 0.5), V(0.4, 0.6));
        Assert.Equal(-0.8, grad[0], 1e-6);
        Assert.Equal(-1.2, grad[1], 1e-6);
    }

    #endregion

    #region Dice Loss

    [Fact]
    public void Dice_PerfectOverlap_IsZero()
    {
        // pred = actual = [1, 1, 0, 0]
        // intersection = 1*1 + 1*1 + 0*0 + 0*0 = 2
        // sumP = 2, sumA = 2
        // DiceCoeff = 2*2 / (2+2) = 1.0
        // DiceLoss = 1 - 1.0 = 0.0
        var loss = new DiceLoss<double>();
        double result = loss.CalculateLoss(V(1, 1, 0, 0), V(1, 1, 0, 0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Dice_NoOverlap_IsOne()
    {
        // pred = [1, 1, 0, 0], actual = [0, 0, 1, 1]
        // intersection = 0
        // sumP = 2, sumA = 2
        // DiceCoeff = 0 / 4 = 0.0
        // DiceLoss = 1 - 0.0 = 1.0
        var loss = new DiceLoss<double>();
        double result = loss.CalculateLoss(V(1, 1, 0, 0), V(0, 0, 1, 1));
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Dice_PartialOverlap_HandCalculated()
    {
        // pred = [0.8, 0.6, 0.3], actual = [1, 1, 0]
        // intersection = 0.8*1 + 0.6*1 + 0.3*0 = 1.4
        // sumP = 0.8 + 0.6 + 0.3 = 1.7, sumA = 2.0
        // DiceCoeff = 2*1.4 / (1.7 + 2.0) = 2.8 / 3.7 ≈ 0.75676
        // DiceLoss = 1 - 0.75676 ≈ 0.24324
        var loss = new DiceLoss<double>();
        double result = loss.CalculateLoss(V(0.8, 0.6, 0.3), V(1.0, 1.0, 0.0));
        double expected = 1.0 - (2.0 * 1.4) / (1.7 + 2.0);
        Assert.Equal(expected, result, 1e-6);
    }

    [Fact]
    public void Dice_Bounded_ZeroToOne()
    {
        // Dice loss should always be in [0, 1] for valid inputs
        var loss = new DiceLoss<double>();
        double result = loss.CalculateLoss(V(0.5, 0.3, 0.9), V(1.0, 0.0, 1.0));
        Assert.True(result >= 0.0, $"Dice loss should be >= 0, got {result}");
        Assert.True(result <= 1.0, $"Dice loss should be <= 1, got {result}");
    }

    #endregion

    #region Focal Loss

    [Fact]
    public void Focal_GammaZero_EquivalentToBCE()
    {
        // When gamma=0, focal loss = -alpha * log(pt)
        // This is equivalent to weighted BCE
        // predicted=[0.9], actual=[1.0], alpha=1.0, gamma=0
        // pt = 0.9, alphaT = 1.0
        // focal = -(1.0) * (1)^0 * log(0.9) = -log(0.9) / n = 0.10536
        var focal = new FocalLoss<double>(gamma: 0.0, alpha: 1.0);
        double result = focal.CalculateLoss(V(0.9), V(1.0));
        double expected = -Math.Log(0.9);
        Assert.Equal(expected, result, 1e-6);
    }

    [Fact]
    public void Focal_HighGamma_DownWeightsEasyExamples()
    {
        // Easy example: predicted=0.95 for actual=1.0
        // gamma=0: loss ≈ -log(0.95) = 0.0513
        // gamma=2: loss ≈ -(1-0.95)^2 * log(0.95) = 0.0025 * 0.0513 ≈ 0.000128
        // With gamma=2, easy example has MUCH lower loss
        var focal_g0 = new FocalLoss<double>(gamma: 0.0, alpha: 1.0);
        var focal_g2 = new FocalLoss<double>(gamma: 2.0, alpha: 1.0);
        double loss_g0 = focal_g0.CalculateLoss(V(0.95), V(1.0));
        double loss_g2 = focal_g2.CalculateLoss(V(0.95), V(1.0));
        Assert.True(loss_g2 < loss_g0 * 0.1,
            $"Gamma=2 loss ({loss_g2}) should be << gamma=0 loss ({loss_g0}) for easy examples");
    }

    [Fact]
    public void Focal_HardExample_NotDownWeightedMuch()
    {
        // Hard example: predicted=0.2 for actual=1.0
        // gamma=0: loss ≈ -log(0.2) = 1.6094
        // gamma=2: loss ≈ -(1-0.2)^2 * log(0.2) = 0.64 * 1.6094 = 1.030
        // Hard example still has significant loss even with gamma=2
        var focal_g0 = new FocalLoss<double>(gamma: 0.0, alpha: 1.0);
        var focal_g2 = new FocalLoss<double>(gamma: 2.0, alpha: 1.0);
        double loss_g0 = focal_g0.CalculateLoss(V(0.2), V(1.0));
        double loss_g2 = focal_g2.CalculateLoss(V(0.2), V(1.0));
        // Hard example should retain > 50% of original loss
        Assert.True(loss_g2 > loss_g0 * 0.5,
            $"Gamma=2 loss ({loss_g2}) should retain most of gamma=0 loss ({loss_g0}) for hard examples");
    }

    [Fact]
    public void Focal_AlphaWeighting_HandCalculated()
    {
        // alpha=0.25 for positive class (actual=1)
        // predicted=0.9, actual=1.0, gamma=0
        // loss = -0.25 * (1-0.9)^0 * log(0.9) / 1 = -0.25 * log(0.9) ≈ 0.02634
        var focal = new FocalLoss<double>(gamma: 0.0, alpha: 0.25);
        double result = focal.CalculateLoss(V(0.9), V(1.0));
        double expected = -0.25 * Math.Log(0.9);
        Assert.Equal(expected, result, 1e-6);
    }

    [Fact]
    public void Focal_NegativeClass_HandCalculated()
    {
        // predicted=0.1 for actual=0.0, alpha=0.25, gamma=2
        // pt = 1 - 0.1 = 0.9 (probability of correct class)
        // alphaT = 1 - 0.25 = 0.75 (weight for negative class)
        // focal = -0.75 * (1-0.9)^2 * log(0.9) / 1
        //       = -0.75 * 0.01 * log(0.9)
        //       = 0.75 * 0.01 * 0.10536 ≈ 0.000790
        var focal = new FocalLoss<double>(gamma: 2.0, alpha: 0.25);
        double result = focal.CalculateLoss(V(0.1), V(0.0));
        double expected = -0.75 * Math.Pow(0.1, 2) * Math.Log(0.9);
        Assert.Equal(expected, result, 1e-6);
    }

    #endregion

    #region Cosine Similarity Loss

    [Fact]
    public void CosineSimilarity_IdenticalVectors_IsZero()
    {
        // cos(v, v) = 1.0, loss = 1 - 1 = 0
        var loss = new CosineSimilarityLoss<double>();
        double result = loss.CalculateLoss(V(1, 2, 3), V(1, 2, 3));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_IsOne()
    {
        // cos([1,0], [0,1]) = 0, loss = 1 - 0 = 1
        var loss = new CosineSimilarityLoss<double>();
        double result = loss.CalculateLoss(V(1, 0), V(0, 1));
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OppositeVectors_IsTwo()
    {
        // cos([1,0], [-1,0]) = -1, loss = 1 - (-1) = 2
        var loss = new CosineSimilarityLoss<double>();
        double result = loss.CalculateLoss(V(1, 0), V(-1, 0));
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_HandCalculated()
    {
        // predicted=[3, 4], actual=[4, 3]
        // dot = 3*4 + 4*3 = 24
        // ||p|| = sqrt(9+16) = 5, ||a|| = sqrt(16+9) = 5
        // cos = 24 / 25 = 0.96
        // loss = 1 - 0.96 = 0.04
        var loss = new CosineSimilarityLoss<double>();
        double result = loss.CalculateLoss(V(3, 4), V(4, 3));
        Assert.Equal(0.04, result, 1e-6);
    }

    [Fact]
    public void CosineSimilarity_ScaleInvariant()
    {
        // Cosine similarity doesn't depend on magnitude
        // [1, 2] and [2, 4] should have same similarity to [3, 6]
        var loss = new CosineSimilarityLoss<double>();
        double loss1 = loss.CalculateLoss(V(1, 2), V(3, 4));
        double loss2 = loss.CalculateLoss(V(10, 20), V(3, 4));
        Assert.Equal(loss1, loss2, 1e-6);
    }

    #endregion

    #region Quantile Loss

    [Fact]
    public void Quantile_Median_HandCalculated()
    {
        // quantile=0.5, predicted=[2], actual=[5]
        // diff = actual - predicted = 3 > 0 → underestimation
        // loss = 0.5 * 3 = 1.5 per element, averaged = 1.5
        var loss = new QuantileLoss<double>(quantile: 0.5);
        double result = loss.CalculateLoss(V(2), V(5));
        Assert.Equal(1.5, result, Tolerance);
    }

    [Fact]
    public void Quantile_Median_Symmetric()
    {
        // For quantile=0.5, underprediction and overprediction by same amount
        // should give same loss (because 0.5 = 1 - 0.5)
        var loss = new QuantileLoss<double>(quantile: 0.5);
        double underLoss = loss.CalculateLoss(V(3), V(5));  // diff=2 → 0.5*2 = 1.0
        double overLoss = loss.CalculateLoss(V(7), V(5));   // diff=-2 → 0.5*2 = 1.0
        Assert.Equal(underLoss, overLoss, Tolerance);
    }

    [Fact]
    public void Quantile_90th_PenalizesUnderestimationMore()
    {
        // quantile=0.9
        // Underprediction: predicted=3, actual=5, diff=2 → 0.9 * 2 = 1.8
        // Overprediction: predicted=7, actual=5, diff=-2 → (1-0.9) * 2 = 0.2
        var loss = new QuantileLoss<double>(quantile: 0.9);
        double underLoss = loss.CalculateLoss(V(3), V(5));
        double overLoss = loss.CalculateLoss(V(7), V(5));
        Assert.Equal(1.8, underLoss, Tolerance);
        Assert.Equal(0.2, overLoss, Tolerance);
        Assert.True(underLoss > overLoss);
    }

    [Fact]
    public void Quantile_10th_PenalizesOverestimationMore()
    {
        // quantile=0.1
        // Underprediction: predicted=3, actual=5, diff=2 → 0.1 * 2 = 0.2
        // Overprediction: predicted=7, actual=5, diff=-2 → (1-0.1) * 2 = 1.8
        var loss = new QuantileLoss<double>(quantile: 0.1);
        double underLoss = loss.CalculateLoss(V(3), V(5));
        double overLoss = loss.CalculateLoss(V(7), V(5));
        Assert.Equal(0.2, underLoss, Tolerance);
        Assert.Equal(1.8, overLoss, Tolerance);
        Assert.True(overLoss > underLoss);
    }

    [Fact]
    public void Quantile_PerfectPrediction_IsZero()
    {
        var loss = new QuantileLoss<double>(quantile: 0.75);
        double result = loss.CalculateLoss(V(5, 10), V(5, 10));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Quantile_InvalidQuantile_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new QuantileLoss<double>(quantile: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new QuantileLoss<double>(quantile: 1.5));
    }

    [Fact]
    public void Quantile_Derivative_HandCalculated()
    {
        // quantile=0.75, predicted=[3], actual=[5]
        // diff = 5-3 = 2 > 0 → underestimation
        // derivative = -quantile / n = -0.75 / 1 = -0.75
        var loss = new QuantileLoss<double>(quantile: 0.75);
        var grad = loss.CalculateDerivative(V(3), V(5));
        Assert.Equal(-0.75, grad[0], Tolerance);
    }

    [Fact]
    public void Quantile_Derivative_Overestimation()
    {
        // quantile=0.75, predicted=[7], actual=[5]
        // diff = 5-7 = -2 <= 0 → overestimation
        // derivative = (1-quantile) / n = 0.25 / 1 = 0.25
        var loss = new QuantileLoss<double>(quantile: 0.75);
        var grad = loss.CalculateDerivative(V(7), V(5));
        Assert.Equal(0.25, grad[0], Tolerance);
    }

    #endregion

    #region Log-Cosh Loss

    [Fact]
    public void LogCosh_SmallError_ApproximatesMSEHalf()
    {
        // For small x: log(cosh(x)) ≈ x²/2
        // error=0.01 → log(cosh(0.01)) ≈ 0.01²/2 = 0.00005
        var loss = new LogCoshLoss<double>();
        double result = loss.CalculateLoss(V(1.01), V(1.0));
        double expected = Math.Log(Math.Cosh(0.01));
        Assert.Equal(expected, result, 1e-10);
    }

    [Fact]
    public void LogCosh_LargeError_ApproximatesMAE()
    {
        // For large |x|: log(cosh(x)) ≈ |x| - log(2)
        // error=10.0 → log(cosh(10)) ≈ 10 - log(2) = 9.3069
        var loss = new LogCoshLoss<double>();
        double result = loss.CalculateLoss(V(11.0), V(1.0));
        double expected = Math.Log(Math.Cosh(10.0));
        Assert.Equal(expected, result, 1e-6);
    }

    [Fact]
    public void LogCosh_Symmetric()
    {
        // log(cosh(x)) = log(cosh(-x)) since cosh is even function
        var loss = new LogCoshLoss<double>();
        double pos = loss.CalculateLoss(V(3.0), V(1.0));  // error = 2
        double neg = loss.CalculateLoss(V(-1.0), V(1.0)); // error = -2
        Assert.Equal(pos, neg, Tolerance);
    }

    [Fact]
    public void LogCosh_PerfectPrediction_IsZero()
    {
        var loss = new LogCoshLoss<double>();
        Assert.Equal(0.0, loss.CalculateLoss(V(5, 10), V(5, 10)), Tolerance);
    }

    [Fact]
    public void LogCosh_Derivative_IsTanh()
    {
        // d/dx log(cosh(x)) = tanh(x)
        // error=0.5 → tanh(0.5) ≈ 0.46212
        var loss = new LogCoshLoss<double>();
        var grad = loss.CalculateDerivative(V(1.5), V(1.0));
        Assert.Equal(Math.Tanh(0.5), grad[0], 1e-6);
    }

    [Fact]
    public void LogCosh_AlwaysNonNegative()
    {
        var loss = new LogCoshLoss<double>();
        double result = loss.CalculateLoss(V(-5, 3, 0), V(2, -1, 7));
        Assert.True(result >= 0.0, $"Log-cosh loss should be non-negative, got {result}");
    }

    #endregion

    #region Poisson Loss

    [Fact]
    public void Poisson_HandCalculated()
    {
        // predicted=[3.0], actual=[2.0]
        // Poisson = (pred - actual * log(pred)) / n
        //         = (3.0 - 2.0 * log(3.0)) / 1
        //         = 3.0 - 2.0 * 1.0986 = 3.0 - 2.1972 = 0.8028
        var loss = new PoissonLoss<double>();
        double result = loss.CalculateLoss(V(3.0), V(2.0));
        double expected = (3.0 - 2.0 * Math.Log(3.0));
        Assert.Equal(expected, result, 1e-6);
    }

    [Fact]
    public void Poisson_PerfectPrediction_IsMinimum()
    {
        // When predicted = actual: loss = y - y*log(y)
        // For y=5: 5 - 5*log(5) = 5 - 8.047 = -3.047
        // This should be the minimum for actual=5
        var loss = new PoissonLoss<double>();
        double atPerfect = loss.CalculateLoss(V(5.0), V(5.0));
        double slightly_off = loss.CalculateLoss(V(5.1), V(5.0));
        Assert.True(atPerfect <= slightly_off + 1e-6,
            $"Loss at perfect ({atPerfect}) should be <= loss at 5.1 ({slightly_off})");
    }

    [Fact]
    public void Poisson_Derivative_HandCalculated()
    {
        // d/dp PoissonLoss = (1 - actual/predicted) / n
        // predicted=3.0, actual=2.0
        // = (1 - 2/3) / 1 = 1/3 ≈ 0.3333
        var loss = new PoissonLoss<double>();
        var grad = loss.CalculateDerivative(V(3.0), V(2.0));
        Assert.Equal(1.0 / 3.0, grad[0], 1e-6);
    }

    [Fact]
    public void Poisson_Derivative_AtPerfect_IsZero()
    {
        // At optimal: d/dp = (1 - actual/predicted) = 0 when predicted = actual
        var loss = new PoissonLoss<double>();
        var grad = loss.CalculateDerivative(V(5.0), V(5.0));
        Assert.Equal(0.0, grad[0], 1e-6);
    }

    #endregion

    #region Charbonnier Loss

    [Fact]
    public void Charbonnier_HandCalculated()
    {
        // epsilon=1e-6, predicted=[3], actual=[1]
        // diff = 2, diffSquared = 4
        // Charbonnier = sqrt(4 + 1e-12) / 1 ≈ 2.0
        var loss = new CharbonnierLoss<double>(epsilon: 1e-6);
        double result = loss.CalculateLoss(V(3.0), V(1.0));
        Assert.Equal(Math.Sqrt(4.0 + 1e-12), result, 1e-6);
    }

    [Fact]
    public void Charbonnier_ZeroError_ReturnsEpsilon()
    {
        // When diff=0: sqrt(0 + ε²) = ε
        var loss = new CharbonnierLoss<double>(epsilon: 0.001);
        double result = loss.CalculateLoss(V(5.0), V(5.0));
        Assert.Equal(0.001, result, 1e-8);
    }

    [Fact]
    public void Charbonnier_ApproximatesL1_ForLargeErrors()
    {
        // For large |diff|: sqrt(diff² + ε²) ≈ |diff|
        var loss = new CharbonnierLoss<double>(epsilon: 1e-6);
        double result = loss.CalculateLoss(V(100.0), V(0.0));
        Assert.Equal(100.0, result, 1e-3); // Within 0.001 of L1
    }

    [Fact]
    public void Charbonnier_SmoothAtZero()
    {
        // Unlike L1, Charbonnier has a well-defined derivative at 0
        // derivative = diff / sqrt(diff² + ε²) / n
        // at diff=0: 0 / sqrt(0 + ε²) = 0
        var loss = new CharbonnierLoss<double>(epsilon: 1e-6);
        var grad = loss.CalculateDerivative(V(5.0), V(5.0));
        Assert.Equal(0.0, grad[0], 1e-6);
    }

    [Fact]
    public void Charbonnier_Derivative_HandCalculated()
    {
        // epsilon=1e-6, predicted=[3], actual=[1]
        // diff = 2, diffSquared = 4
        // derivative = 2 / sqrt(4 + 1e-12) / 1 ≈ 2/2 = 1.0
        var loss = new CharbonnierLoss<double>(epsilon: 1e-6);
        var grad = loss.CalculateDerivative(V(3.0), V(1.0));
        Assert.Equal(2.0 / Math.Sqrt(4.0 + 1e-12), grad[0], 1e-6);
    }

    #endregion

    #region Contrastive Loss

    [Fact]
    public void Contrastive_SimilarPair_DistanceSquared()
    {
        // Similar pair (label=1): loss = distance²
        // output1=[1, 0], output2=[4, 0]
        // distance = sqrt(9) = 3
        // loss = 3² = 9
        var loss = new ContrastiveLoss<double>(margin: 2.0);
        double result = loss.CalculateLoss(V(1, 0), V(4, 0), 1.0);
        Assert.Equal(9.0, result, Tolerance);
    }

    [Fact]
    public void Contrastive_DissimilarPair_InsideMargin()
    {
        // Dissimilar pair (label=0), distance < margin
        // output1=[0, 0], output2=[1, 0]
        // distance = 1.0, margin = 2.0
        // loss = (1-0) * max(0, 2.0 - 1.0)² = 1.0² = 1.0
        var loss = new ContrastiveLoss<double>(margin: 2.0);
        double result = loss.CalculateLoss(V(0, 0), V(1, 0), 0.0);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Contrastive_DissimilarPair_OutsideMargin()
    {
        // Dissimilar pair (label=0), distance > margin
        // output1=[0, 0], output2=[3, 0]
        // distance = 3.0, margin = 2.0
        // loss = max(0, 2.0 - 3.0)² = max(0, -1)² = 0
        var loss = new ContrastiveLoss<double>(margin: 2.0);
        double result = loss.CalculateLoss(V(0, 0), V(3, 0), 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Contrastive_IdenticalPairs_ZeroLossForSimilar()
    {
        // Similar pair with zero distance → loss = 0
        var loss = new ContrastiveLoss<double>(margin: 1.0);
        double result = loss.CalculateLoss(V(5, 5), V(5, 5), 1.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Triplet Loss

    [Fact]
    public void Triplet_HandCalculated_ActiveTriplet()
    {
        // anchor=[0, 0], positive=[1, 0], negative=[3, 0]
        // d_pos = sqrt(1) = 1, d_neg = sqrt(9) = 3
        // loss = max(0, 1 - 3 + 1.0) = max(0, -1) = 0   Wait, let me check margin=1
        // loss = max(0, d_pos - d_neg + margin) = max(0, 1 - 3 + 1) = max(0, -1) = 0
        // Need a case where triplet is active (loss > 0)
        // anchor=[0, 0], positive=[2, 0], negative=[1, 0], margin=1
        // d_pos = 2, d_neg = 1
        // loss = max(0, 2 - 1 + 1) = max(0, 2) = 2
        var loss = new TripletLoss<double>(margin: 1.0);
        var anchor = new Matrix<double>(new double[,] { { 0.0, 0.0 } });
        var positive = new Matrix<double>(new double[,] { { 2.0, 0.0 } });
        var negative = new Matrix<double>(new double[,] { { 1.0, 0.0 } });
        double result = loss.CalculateLoss(anchor, positive, negative);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void Triplet_InactiveTriplet_ZeroLoss()
    {
        // anchor=[0, 0], positive=[1, 0], negative=[5, 0], margin=1
        // d_pos = 1, d_neg = 5
        // loss = max(0, 1 - 5 + 1) = max(0, -3) = 0
        var loss = new TripletLoss<double>(margin: 1.0);
        var anchor = new Matrix<double>(new double[,] { { 0.0, 0.0 } });
        var positive = new Matrix<double>(new double[,] { { 1.0, 0.0 } });
        var negative = new Matrix<double>(new double[,] { { 5.0, 0.0 } });
        double result = loss.CalculateLoss(anchor, positive, negative);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Triplet_LargerMargin_MoreActiveTriplets()
    {
        // Same triplet, different margins
        var anchor = new Matrix<double>(new double[,] { { 0.0, 0.0 } });
        var positive = new Matrix<double>(new double[,] { { 1.0, 0.0 } });
        var negative = new Matrix<double>(new double[,] { { 3.0, 0.0 } });

        // margin=1: loss = max(0, 1-3+1) = 0 (inactive)
        var loss1 = new TripletLoss<double>(margin: 1.0);
        Assert.Equal(0.0, loss1.CalculateLoss(anchor, positive, negative), Tolerance);

        // margin=5: loss = max(0, 1-3+5) = 3 (active)
        var loss5 = new TripletLoss<double>(margin: 5.0);
        Assert.Equal(3.0, loss5.CalculateLoss(anchor, positive, negative), Tolerance);
    }

    [Fact]
    public void Triplet_BatchAveraged()
    {
        // Batch of 2 triplets
        // Triplet 1: anchor=[0,0], pos=[2,0], neg=[1,0], margin=1 → loss = max(0, 2-1+1) = 2
        // Triplet 2: anchor=[0,0], pos=[1,0], neg=[5,0], margin=1 → loss = max(0, 1-5+1) = 0
        // Average: (2 + 0) / 2 = 1.0
        var loss = new TripletLoss<double>(margin: 1.0);
        var anchor = new Matrix<double>(new double[,] { { 0.0, 0.0 }, { 0.0, 0.0 } });
        var positive = new Matrix<double>(new double[,] { { 2.0, 0.0 }, { 1.0, 0.0 } });
        var negative = new Matrix<double>(new double[,] { { 1.0, 0.0 }, { 5.0, 0.0 } });
        double result = loss.CalculateLoss(anchor, positive, negative);
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region ElasticNet Loss

    [Fact]
    public void ElasticNet_PureL1_HandCalculated()
    {
        // l1Ratio=1.0, alpha=0.1
        // predicted=[3], actual=[1]
        // MSE = (3-1)²/1 = 4
        // L1 = |3| = 3
        // L2 = 0 (l1Ratio=1, so no L2)
        // Total = 4 + 0.1 * 1.0 * 3 + 0 = 4.3
        var loss = new ElasticNetLoss<double>(l1Ratio: 1.0, alpha: 0.1);
        double result = loss.CalculateLoss(V(3), V(1));
        Assert.Equal(4.3, result, Tolerance);
    }

    [Fact]
    public void ElasticNet_PureL2_HandCalculated()
    {
        // l1Ratio=0.0, alpha=0.1
        // predicted=[3], actual=[1]
        // MSE = 4
        // L1 = 0 (l1Ratio=0)
        // L2 = 0.5 * 3² = 4.5
        // Total = 4 + 0 + 0.1 * 0.5 * 9 = 4 + 0.45 = 4.45
        var loss = new ElasticNetLoss<double>(l1Ratio: 0.0, alpha: 0.1);
        double result = loss.CalculateLoss(V(3), V(1));
        Assert.Equal(4.45, result, Tolerance);
    }

    [Fact]
    public void ElasticNet_Mixed_HandCalculated()
    {
        // l1Ratio=0.5, alpha=0.1
        // predicted=[3], actual=[1]
        // MSE = 4
        // L1 term = alpha * l1Ratio * |pred| = 0.1 * 0.5 * 3 = 0.15
        // L2 term = alpha * (1-l1Ratio) * 0.5 * pred² = 0.1 * 0.5 * 0.5 * 9 = 0.225
        // Total = 4 + 0.15 + 0.225 = 4.375
        var loss = new ElasticNetLoss<double>(l1Ratio: 0.5, alpha: 0.1);
        double result = loss.CalculateLoss(V(3), V(1));
        Assert.Equal(4.375, result, Tolerance);
    }

    [Fact]
    public void ElasticNet_ZeroAlpha_ReducesToMSE()
    {
        // When alpha=0, regularization terms vanish
        var elastic = new ElasticNetLoss<double>(l1Ratio: 0.5, alpha: 0.0);
        var mse = new MeanSquaredErrorLoss<double>();
        var pred = V(1, 2, 3);
        var actual = V(4, 5, 6);
        Assert.Equal(mse.CalculateLoss(pred, actual), elastic.CalculateLoss(pred, actual), Tolerance);
    }

    [Fact]
    public void ElasticNet_InvalidL1Ratio_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ElasticNetLoss<double>(l1Ratio: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ElasticNetLoss<double>(l1Ratio: 1.1));
    }

    #endregion

    #region Cross-Loss Consistency

    [Fact]
    public void Huber_LargeDelta_ConvergesToMSE()
    {
        // With very large delta, all errors are in the quadratic region
        // So Huber loss ≈ 0.5 * MSE
        var huber = new HuberLoss<double>(delta: 1000.0);
        var mse = new MeanSquaredErrorLoss<double>();
        var pred = V(1, 2, 3, 4, 5);
        var actual = V(2, 4, 6, 8, 10);
        double huberResult = huber.CalculateLoss(pred, actual);
        double mseResult = mse.CalculateLoss(pred, actual);
        // Huber with large delta = 0.5 * sum(error²) / n = 0.5 * MSE
        Assert.Equal(0.5 * mseResult, huberResult, 1e-6);
    }

    [Fact]
    public void LogCosh_VsHuber_BothSmootherThanMSE()
    {
        // For large outliers, both LogCosh and Huber give less loss than MSE
        var mse = new MeanSquaredErrorLoss<double>();
        var huber = new HuberLoss<double>(delta: 1.0);
        var logcosh = new LogCoshLoss<double>();

        // Outlier: predicted=100, actual=0
        var pred = V(100.0);
        var actual = V(0.0);

        double mseLoss = mse.CalculateLoss(pred, actual);    // 10000
        double huberLoss = huber.CalculateLoss(pred, actual); // 1*(100-0.5) = 99.5
        double logcoshLoss = logcosh.CalculateLoss(pred, actual); // ≈ 100 - log(2) ≈ 99.3

        Assert.True(huberLoss < mseLoss, "Huber should be less than MSE for outliers");
        Assert.True(logcoshLoss < mseLoss, "LogCosh should be less than MSE for outliers");
    }

    [Fact]
    public void Charbonnier_VsMAE_ApproachesAsEpsilonShrinks()
    {
        // As epsilon → 0, Charbonnier → |error|
        var charb_small = new CharbonnierLoss<double>(epsilon: 1e-12);
        double charb = charb_small.CalculateLoss(V(3.0), V(0.0));
        double mae = 3.0; // |3 - 0|
        Assert.Equal(mae, charb, 1e-6);
    }

    #endregion

    #region Gradient Numerical Verification

    [Fact]
    public void MSE_GradientCheck_NumericalDifference()
    {
        // Verify MSE gradient via numerical differentiation
        var loss = new MeanSquaredErrorLoss<double>();
        var pred = V(2.0, 4.0);
        var actual = V(1.0, 3.0);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void Huber_GradientCheck_NumericalDifference()
    {
        var loss = new HuberLoss<double>(delta: 1.0);
        var pred = V(2.0, 4.0);
        var actual = V(1.5, 1.0); // 0.5 (quadratic) and 3.0 (linear)
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void LogCosh_GradientCheck_NumericalDifference()
    {
        var loss = new LogCoshLoss<double>();
        var pred = V(2.5, 0.5);
        var actual = V(1.0, 3.0);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void Poisson_GradientCheck_NumericalDifference()
    {
        var loss = new PoissonLoss<double>();
        var pred = V(2.0, 5.0);
        var actual = V(3.0, 4.0);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void Charbonnier_GradientCheck_NumericalDifference()
    {
        var loss = new CharbonnierLoss<double>(epsilon: 1e-3);
        var pred = V(2.0, 0.5);
        var actual = V(1.0, 3.0);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void Dice_GradientCheck_NumericalDifference()
    {
        var loss = new DiceLoss<double>();
        var pred = V(0.8, 0.3, 0.6);
        var actual = V(1.0, 0.0, 1.0);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void CosineSimilarity_GradientCheck_NumericalDifference()
    {
        var loss = new CosineSimilarityLoss<double>();
        var pred = V(3.0, 4.0);
        var actual = V(1.0, 2.0);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    [Fact]
    public void KL_GradientCheck_NumericalDifference()
    {
        var loss = new KullbackLeiblerDivergence<double>();
        var pred = V(0.4, 0.6);
        var actual = V(0.3, 0.7);
        var analyticalGrad = loss.CalculateDerivative(pred, actual);

        double h = 1e-7;
        for (int i = 0; i < pred.Length; i++)
        {
            var pred_plus = pred.ToArray();
            var pred_minus = pred.ToArray();
            pred_plus[i] += h;
            pred_minus[i] -= h;
            double numericalGrad = (loss.CalculateLoss(new Vector<double>(pred_plus), actual) -
                                     loss.CalculateLoss(new Vector<double>(pred_minus), actual)) / (2 * h);
            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-4);
        }
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void AllLossFunctions_LengthMismatch_Throws()
    {
        var pred2 = V(1, 2);
        var actual3 = V(1, 2, 3);

        Assert.Throws<ArgumentException>(() => new MeanSquaredErrorLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new HuberLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new BinaryCrossEntropyLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new LogCoshLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new PoissonLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new CharbonnierLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new DiceLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new CosineSimilarityLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new QuantileLoss<double>().CalculateLoss(pred2, actual3));
        Assert.Throws<ArgumentException>(() => new KullbackLeiblerDivergence<double>().CalculateLoss(pred2, actual3));
    }

    [Fact]
    public void Contrastive_StandardOverload_ThrowsNotSupported()
    {
        var loss = new ContrastiveLoss<double>();
        Assert.Throws<NotSupportedException>(() => loss.CalculateLoss(V(1), V(2)));
        Assert.Throws<NotSupportedException>(() => loss.CalculateDerivative(V(1), V(2)));
    }

    [Fact]
    public void Triplet_StandardOverload_ThrowsNotSupported()
    {
        var loss = new TripletLoss<double>();
        Assert.Throws<NotSupportedException>(() => loss.CalculateLoss(V(1), V(2)));
        Assert.Throws<NotSupportedException>(() => loss.CalculateDerivative(V(1), V(2)));
    }

    [Fact]
    public void BCE_NumericalStability_NearZeroAndOne()
    {
        // Should not produce NaN or infinity for extreme values
        var loss = new BinaryCrossEntropyLoss<double>();
        double result1 = loss.CalculateLoss(V(1e-15), V(1.0));
        double result2 = loss.CalculateLoss(V(1.0 - 1e-15), V(0.0));
        Assert.False(double.IsNaN(result1), "BCE near zero should not be NaN");
        Assert.False(double.IsNaN(result2), "BCE near one should not be NaN");
        Assert.False(double.IsInfinity(result1), "BCE near zero should not be infinity");
        Assert.False(double.IsInfinity(result2), "BCE near one should not be infinity");
    }

    [Fact]
    public void Poisson_NumericalStability_NearZeroPrediction()
    {
        // Should handle near-zero predictions without NaN/Inf
        var loss = new PoissonLoss<double>();
        double result = loss.CalculateLoss(V(1e-15), V(5.0));
        Assert.False(double.IsNaN(result), "Poisson loss near zero should not be NaN");
    }

    #endregion
}
