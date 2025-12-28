using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Comprehensive mathematical tests for loss functions.
/// All expected values are hand-calculated to verify correctness.
/// </summary>
public class LossFunctionsMathematicalTests
{
    private const double Tolerance = 1e-6;
    private const double GradientTolerance = 1e-4;
    private const double NumericalEpsilon = 1e-5;

    #region Mean Squared Error - Mathematical Verification

    /// <summary>
    /// MSE = (1/n) * Σ(predicted - actual)²
    /// For [1, 2, 3] vs [2, 4, 6]: errors = [-1, -2, -3], squared = [1, 4, 9], mean = 14/3 ≈ 4.6667
    /// </summary>
    [Fact]
    public void MSE_HandCalculated_MatchesExpected()
    {
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var loss = mse.CalculateLoss(predicted, actual);

        // errors = [1-2, 2-4, 3-6] = [-1, -2, -3]
        // squared = [1, 4, 9]
        // sum = 14
        // mean = 14/3 = 4.666...
        double expected = 14.0 / 3.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    /// <summary>
    /// MSE derivative = 2*(predicted - actual)/n
    /// For [1, 2, 3] vs [2, 4, 6]: derivative = 2*[-1, -2, -3]/3 = [-2/3, -4/3, -6/3]
    /// </summary>
    [Fact]
    public void MSE_Derivative_HandCalculated_MatchesExpected()
    {
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var derivative = mse.CalculateDerivative(predicted, actual);

        // derivative = 2*(predicted - actual)/n
        // = 2*[1-2, 2-4, 3-6]/3
        // = 2*[-1, -2, -3]/3
        // = [-2/3, -4/3, -2]
        Assert.Equal(-2.0 / 3.0, derivative[0], Tolerance);
        Assert.Equal(-4.0 / 3.0, derivative[1], Tolerance);
        Assert.Equal(-2.0, derivative[2], Tolerance);
    }

    [Fact]
    public void MSE_GradientCheck_NumericalVsAnalytical()
    {
        var mse = new MeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.5, 2.5, 3.5 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var analyticalGradient = mse.CalculateDerivative(predicted, actual);

        // Numerical gradient check
        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += NumericalEpsilon;
            predictedMinus[i] -= NumericalEpsilon;

            double lossPlus = mse.CalculateLoss(predictedPlus, actual);
            double lossMinus = mse.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Mean Absolute Error - Mathematical Verification

    /// <summary>
    /// MAE = (1/n) * Σ|predicted - actual|
    /// For [1, 2, 3] vs [2, 4, 6]: errors = [1, 2, 3], mean = 6/3 = 2
    /// </summary>
    [Fact]
    public void MAE_HandCalculated_MatchesExpected()
    {
        var mae = new MeanAbsoluteErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var loss = mae.CalculateLoss(predicted, actual);

        // errors = |1-2| + |2-4| + |3-6| = 1 + 2 + 3 = 6
        // mean = 6/3 = 2
        double expected = 2.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    /// <summary>
    /// MAE derivative = sign(predicted - actual)/n
    /// </summary>
    [Fact]
    public void MAE_Derivative_HandCalculated_MatchesExpected()
    {
        var mae = new MeanAbsoluteErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 5.0, 1.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 4.0, 3.0 });

        var derivative = mae.CalculateDerivative(predicted, actual);

        // sign(predicted - actual)/n
        // signs = [sign(3), sign(-3), sign(0)] = [1, -1, 0]
        // derivative = [1/3, -1/3, 0]
        Assert.Equal(1.0 / 3.0, derivative[0], Tolerance);
        Assert.Equal(-1.0 / 3.0, derivative[1], Tolerance);
        Assert.Equal(0.0, derivative[2], Tolerance);
    }

    #endregion

    #region Binary Cross Entropy - Mathematical Verification

    /// <summary>
    /// BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
    /// For p=[0.9, 0.1] vs y=[1, 0]:
    /// = -(1/2) * [1*log(0.9) + 0*log(0.1) + 0*log(0.1) + 1*log(0.9)]
    /// = -(1/2) * [log(0.9) + log(0.9)]
    /// = -log(0.9) ≈ 0.1054
    /// </summary>
    [Fact]
    public void BCE_HandCalculated_MatchesExpected()
    {
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.9, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        var loss = bce.CalculateLoss(predicted, actual);

        // BCE = -(1/2) * [1*log(0.9) + 0*log(0.1) + 0*log(0.1) + 1*log(0.9)]
        // = -(1/2) * [log(0.9) + log(0.9)]
        // = -log(0.9)
        double expected = -Math.Log(0.9);
        Assert.Equal(expected, loss, Tolerance);
    }

    /// <summary>
    /// BCE with 50% probability should give loss = log(2) ≈ 0.693
    /// </summary>
    [Fact]
    public void BCE_FiftyPercent_ReturnsLog2()
    {
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = bce.CalculateLoss(predicted, actual);

        // BCE = -[1*log(0.5) + 0*log(0.5)] = -log(0.5) = log(2)
        double expected = Math.Log(2);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void BCE_GradientCheck_NumericalVsAnalytical()
    {
        var bce = new BinaryCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.7, 0.3 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        var analyticalGradient = bce.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] = Math.Min(0.999, predictedPlus[i] + NumericalEpsilon);
            predictedMinus[i] = Math.Max(0.001, predictedMinus[i] - NumericalEpsilon);

            double lossPlus = bce.CalculateLoss(predictedPlus, actual);
            double lossMinus = bce.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Cross Entropy - Mathematical Verification

    /// <summary>
    /// CE = -(1/n) * Σ[y_i * log(p_i)]
    /// For one-hot [1, 0, 0] vs softmax [0.7, 0.2, 0.1]:
    /// = -(1/3) * [1*log(0.7) + 0*log(0.2) + 0*log(0.1)]
    /// = -log(0.7)/3
    /// </summary>
    [Fact]
    public void CrossEntropy_OneHotEncoded_MatchesExpected()
    {
        var ce = new CrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.7, 0.2, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        var loss = ce.CalculateLoss(predicted, actual);

        // CE = -(1/3) * [1*log(0.7)]
        double expected = -Math.Log(0.7) / 3.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void CrossEntropy_GradientCheck_NumericalVsAnalytical()
    {
        var ce = new CrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.6, 0.3, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        var analyticalGradient = ce.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] = Math.Min(0.999, predictedPlus[i] + NumericalEpsilon);
            predictedMinus[i] = Math.Max(0.001, predictedMinus[i] - NumericalEpsilon);

            double lossPlus = ce.CalculateLoss(predictedPlus, actual);
            double lossMinus = ce.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Huber Loss - Mathematical Verification

    /// <summary>
    /// Huber loss with delta=1:
    /// L = 0.5 * x² for |x| <= delta
    /// L = delta * (|x| - 0.5 * delta) for |x| > delta
    ///
    /// For error 0.5 (small): L = 0.5 * 0.5² = 0.125
    /// For error 2.0 (large): L = 1 * (2 - 0.5) = 1.5
    /// </summary>
    [Fact]
    public void Huber_SmallError_MatchesMSEHalf()
    {
        var huber = new HuberLoss<double>(delta: 1.0);
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 0.0 });

        var loss = huber.CalculateLoss(predicted, actual);

        // For error 0.5 <= delta=1: L = 0.5 * 0.5² = 0.125
        double expected = 0.5 * 0.5 * 0.5;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Huber_LargeError_MatchesLinear()
    {
        var huber = new HuberLoss<double>(delta: 1.0);
        var predicted = new Vector<double>(new[] { 2.0 });
        var actual = new Vector<double>(new[] { 0.0 });

        var loss = huber.CalculateLoss(predicted, actual);

        // For error 2.0 > delta=1: L = delta * (|error| - 0.5 * delta) = 1 * (2 - 0.5) = 1.5
        double expected = 1.0 * (2.0 - 0.5 * 1.0);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Huber_GradientCheck_NumericalVsAnalytical()
    {
        var huber = new HuberLoss<double>(delta: 1.0);
        var predicted = new Vector<double>(new[] { 0.5, 2.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0 });

        var analyticalGradient = huber.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += NumericalEpsilon;
            predictedMinus[i] -= NumericalEpsilon;

            double lossPlus = huber.CalculateLoss(predictedPlus, actual);
            double lossMinus = huber.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Hinge Loss - Mathematical Verification

    /// <summary>
    /// Hinge loss: L = max(0, 1 - y * f(x))
    /// For y=1, f(x)=2: L = max(0, 1 - 2) = 0 (correct with large margin)
    /// For y=1, f(x)=0.5: L = max(0, 1 - 0.5) = 0.5 (correct but small margin)
    /// For y=1, f(x)=-1: L = max(0, 1 - (-1)) = 2 (wrong prediction)
    /// </summary>
    [Fact]
    public void Hinge_LargeMargin_ReturnsZero()
    {
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = hinge.CalculateLoss(predicted, actual);

        // max(0, 1 - 1*2) = max(0, -1) = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void Hinge_SmallMargin_ReturnsPositive()
    {
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = hinge.CalculateLoss(predicted, actual);

        // max(0, 1 - 1*0.5) = max(0, 0.5) = 0.5
        Assert.Equal(0.5, loss, Tolerance);
    }

    [Fact]
    public void Hinge_WrongPrediction_ReturnsLarge()
    {
        var hinge = new HingeLoss<double>();
        var predicted = new Vector<double>(new[] { -1.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = hinge.CalculateLoss(predicted, actual);

        // max(0, 1 - 1*(-1)) = max(0, 2) = 2
        Assert.Equal(2.0, loss, Tolerance);
    }

    #endregion

    #region Root Mean Squared Error - Mathematical Verification

    /// <summary>
    /// RMSE = sqrt(MSE) = sqrt((1/n) * Σ(predicted - actual)²)
    /// For [1, 2, 3] vs [2, 4, 6]: MSE = 14/3, RMSE = sqrt(14/3) ≈ 2.16
    /// </summary>
    [Fact]
    public void RMSE_HandCalculated_MatchesExpected()
    {
        var rmse = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var loss = rmse.CalculateLoss(predicted, actual);

        // RMSE = sqrt(14/3) ≈ 2.16
        double expected = Math.Sqrt(14.0 / 3.0);
        Assert.Equal(expected, loss, Tolerance);
    }

    #endregion

    #region Log Cosh Loss - Mathematical Verification

    /// <summary>
    /// LogCosh = (1/n) * Σ log(cosh(predicted - actual))
    /// For small x: log(cosh(x)) ≈ x²/2
    /// </summary>
    [Fact]
    public void LogCosh_SmallError_ApproximatesSquaredHalf()
    {
        var logCosh = new LogCoshLoss<double>();
        var predicted = new Vector<double>(new[] { 0.1 });
        var actual = new Vector<double>(new[] { 0.0 });

        var loss = logCosh.CalculateLoss(predicted, actual);

        // For small x, log(cosh(x)) ≈ x²/2
        // log(cosh(0.1)) ≈ 0.1²/2 = 0.005
        double expected = Math.Log(Math.Cosh(0.1));
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void LogCosh_HandCalculated_MatchesExpected()
    {
        var logCosh = new LogCoshLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0 });

        var loss = logCosh.CalculateLoss(predicted, actual);

        // LogCosh = (log(cosh(1)) + log(cosh(2))) / 2
        double expected = (Math.Log(Math.Cosh(1.0)) + Math.Log(Math.Cosh(2.0))) / 2.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void LogCosh_GradientCheck_NumericalVsAnalytical()
    {
        var logCosh = new LogCoshLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, 1.5 });
        var actual = new Vector<double>(new[] { 0.0, 1.0 });

        var analyticalGradient = logCosh.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += NumericalEpsilon;
            predictedMinus[i] -= NumericalEpsilon;

            double lossPlus = logCosh.CalculateLoss(predictedPlus, actual);
            double lossMinus = logCosh.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Quantile Loss - Mathematical Verification

    /// <summary>
    /// Quantile loss for quantile q:
    /// L = q * (y - p) when y > p (underprediction)
    /// L = (1-q) * (p - y) when p > y (overprediction)
    /// </summary>
    [Fact]
    public void Quantile_Median_BehavesLikeMAEHalf()
    {
        var quantile = new QuantileLoss<double>(0.5);
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var loss = quantile.CalculateLoss(predicted, actual);

        // At q=0.5, quantile loss = 0.5 * MAE
        // errors = [1, 2, 3], sum = 6, mean = 2
        // quantile loss = 0.5 * 2 = 1
        double expected = 1.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Quantile_HighQuantile_PenalizesUnderprediction()
    {
        var quantile90 = new QuantileLoss<double>(0.9);
        var quantile10 = new QuantileLoss<double>(0.1);

        // Underprediction: predicted < actual
        var predicted = new Vector<double>(new[] { 1.0 });
        var actual = new Vector<double>(new[] { 2.0 });

        var loss90 = quantile90.CalculateLoss(predicted, actual);
        var loss10 = quantile10.CalculateLoss(predicted, actual);

        // High quantile (0.9) penalizes underprediction more
        Assert.True(loss90 > loss10);
    }

    #endregion

    #region Focal Loss - Mathematical Verification

    /// <summary>
    /// Focal loss: FL = -α * (1-p)^γ * log(p) for y=1
    /// With α=1, γ=2: FL = -(1-p)² * log(p)
    /// </summary>
    [Fact]
    public void Focal_EasyExample_DownweightsLoss()
    {
        var focal = new FocalLoss<double>(gamma: 2.0, alpha: 1.0);
        var bce = new BinaryCrossEntropyLoss<double>();

        // Easy example: high confidence correct prediction
        var predicted = new Vector<double>(new[] { 0.9 });
        var actual = new Vector<double>(new[] { 1.0 });

        var focalLoss = focal.CalculateLoss(predicted, actual);
        var bceLoss = bce.CalculateLoss(predicted, actual);

        // Focal should be much smaller than BCE for easy examples
        Assert.True(focalLoss < bceLoss);

        // For p=0.9, γ=2: focal = (1-0.9)² * BCE = 0.01 * BCE
        double expectedRatio = Math.Pow(1 - 0.9, 2);
        Assert.True(focalLoss / bceLoss < 0.02); // Should be close to 0.01
    }

    #endregion

    #region Cosine Similarity Loss - Mathematical Verification

    /// <summary>
    /// Cosine similarity = (a · b) / (||a|| * ||b||)
    /// Loss = 1 - similarity (to make it a minimization objective)
    /// </summary>
    [Fact]
    public void CosineSimilarity_IdenticalVectors_MinimalLoss()
    {
        var cosine = new CosineSimilarityLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var loss = cosine.CalculateLoss(predicted, actual);

        // Identical vectors: cosine similarity = 1, loss should be minimal (0 or negative)
        Assert.True(loss <= 0.0 + Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_HigherLoss()
    {
        var cosine = new CosineSimilarityLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 0.0 });
        var actual = new Vector<double>(new[] { 0.0, 1.0 });

        var loss = cosine.CalculateLoss(predicted, actual);

        // Orthogonal vectors: cosine similarity = 0, loss should be higher
        // Loss might be 1 - 0 = 1 or -0 depending on implementation
        Assert.True(loss >= 0.0);
    }

    /// <summary>
    /// Hand-calculated cosine similarity:
    /// a = [3, 4], b = [4, 3]
    /// a · b = 12 + 12 = 24
    /// ||a|| = 5, ||b|| = 5
    /// similarity = 24/25 = 0.96
    /// </summary>
    [Fact]
    public void CosineSimilarity_HandCalculated_MatchesExpected()
    {
        var cosine = new CosineSimilarityLoss<double>();
        var predicted = new Vector<double>(new[] { 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 4.0, 3.0 });

        var loss = cosine.CalculateLoss(predicted, actual);

        // a · b = 3*4 + 4*3 = 24
        // ||a|| = sqrt(9+16) = 5
        // ||b|| = sqrt(16+9) = 5
        // similarity = 24/25 = 0.96
        // loss = 1 - 0.96 = 0.04 (if loss = 1 - similarity)
        // or loss = -0.96 (if loss = -similarity)
        double similarity = 24.0 / 25.0;
        Assert.True(Math.Abs(loss - (1 - similarity)) < Tolerance ||
                    Math.Abs(loss - (-similarity)) < Tolerance);
    }

    #endregion

    #region KL Divergence - Mathematical Verification

    /// <summary>
    /// KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    /// </summary>
    [Fact]
    public void KLDivergence_IdenticalDistributions_ReturnsZero()
    {
        var kl = new KullbackLeiblerDivergence<double>();
        var p = new Vector<double>(new[] { 0.25, 0.25, 0.25, 0.25 });
        var q = new Vector<double>(new[] { 0.25, 0.25, 0.25, 0.25 });

        var loss = kl.CalculateLoss(p, q);

        // Same distribution: KL = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void KLDivergence_DifferentDistributions_ReturnsPositive()
    {
        var kl = new KullbackLeiblerDivergence<double>();
        var p = new Vector<double>(new[] { 0.9, 0.1 });
        var q = new Vector<double>(new[] { 0.5, 0.5 });

        var loss = kl.CalculateLoss(p, q);

        // KL(P||Q) = 0.9*log(0.9/0.5) + 0.1*log(0.1/0.5)
        // = 0.9*log(1.8) + 0.1*log(0.2)
        double expected = 0.9 * Math.Log(0.9 / 0.5) + 0.1 * Math.Log(0.1 / 0.5);
        Assert.True(loss >= 0.0); // KL is always non-negative
    }

    #endregion

    #region Dice Loss - Mathematical Verification

    /// <summary>
    /// Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    /// Dice loss = 1 - Dice coefficient
    /// For soft predictions: Dice = 2 * Σ(p*y) / (Σp + Σy)
    /// </summary>
    [Fact]
    public void Dice_PerfectOverlap_MinimalLoss()
    {
        var dice = new DiceLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var loss = dice.CalculateLoss(predicted, actual);

        // Perfect overlap: Dice = 2*2/(2+2) = 1, loss = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void Dice_NoOverlap_MaximalLoss()
    {
        var dice = new DiceLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var loss = dice.CalculateLoss(predicted, actual);

        // No overlap: Dice = 2*0/(2+2) = 0, loss = 1
        Assert.Equal(1.0, loss, Tolerance);
    }

    /// <summary>
    /// Hand-calculated Dice:
    /// p = [0.8, 0.6, 0.2, 0.1]
    /// y = [1, 1, 0, 0]
    /// intersection = 0.8*1 + 0.6*1 + 0.2*0 + 0.1*0 = 1.4
    /// sum_p = 1.7, sum_y = 2
    /// Dice = 2*1.4/(1.7+2) = 2.8/3.7 ≈ 0.757
    /// Loss = 1 - 0.757 ≈ 0.243
    /// </summary>
    [Fact]
    public void Dice_HandCalculated_MatchesExpected()
    {
        var dice = new DiceLoss<double>();
        var predicted = new Vector<double>(new[] { 0.8, 0.6, 0.2, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var loss = dice.CalculateLoss(predicted, actual);

        double intersection = 0.8 + 0.6;
        double sumP = 0.8 + 0.6 + 0.2 + 0.1;
        double sumY = 2.0;
        double diceCoeff = (2.0 * intersection) / (sumP + sumY);
        double expectedLoss = 1.0 - diceCoeff;

        Assert.Equal(expectedLoss, loss, Tolerance);
    }

    #endregion

    #region Jaccard Loss - Mathematical Verification

    /// <summary>
    /// Jaccard/IoU = |X ∩ Y| / |X ∪ Y|
    /// Jaccard loss = 1 - IoU
    /// </summary>
    [Fact]
    public void Jaccard_PerfectOverlap_MinimalLoss()
    {
        var jaccard = new JaccardLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var loss = jaccard.CalculateLoss(predicted, actual);

        // Perfect overlap: IoU = 2/2 = 1, loss = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void Jaccard_NoOverlap_MaximalLoss()
    {
        var jaccard = new JaccardLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var loss = jaccard.CalculateLoss(predicted, actual);

        // No overlap: IoU = 0/4 = 0, loss = 1
        Assert.Equal(1.0, loss, Tolerance);
    }

    #endregion

    #region Poisson Loss - Mathematical Verification

    /// <summary>
    /// Poisson loss = predicted - actual * log(predicted)
    /// (summed over all elements, then averaged)
    /// </summary>
    [Fact]
    public void Poisson_HandCalculated_MatchesExpected()
    {
        var poisson = new PoissonLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 });

        var loss = poisson.CalculateLoss(predicted, actual);

        // Poisson = (pred - actual*log(pred)) for each element
        // = (2 - 1*log(2)) + (3 - 2*log(3)) / 2
        // = (2 - 0.693) + (3 - 2.197) / 2
        // = 1.307 + 0.803 / 2
        // = 2.11 / 2 = 1.055
        double elem1 = 2.0 - 1.0 * Math.Log(2.0);
        double elem2 = 3.0 - 2.0 * Math.Log(3.0);
        double expected = (elem1 + elem2) / 2.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    #endregion

    #region Elastic Net Loss - Mathematical Verification

    /// <summary>
    /// Elastic Net = MSE + λ₁ * L1(weights) + λ₂ * L2(weights)
    /// This is typically used with model weights, but for loss function testing
    /// we verify the combined L1 and L2 penalty behavior.
    /// </summary>
    [Fact]
    public void ElasticNet_CombinesMSEWithRegularization()
    {
        var elasticNet = new ElasticNetLoss<double>(l1Ratio: 0.5, alpha: 0.1);
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var loss = elasticNet.CalculateLoss(predicted, actual);

        // With identical predicted/actual, MSE component = 0
        // The total loss depends on how regularization is applied
        Assert.True(loss >= 0.0);
    }

    #endregion

    #region Categorical Cross Entropy - Mathematical Verification

    /// <summary>
    /// CCE = -Σ(actual * log(predicted))
    /// For one-hot [1, 0, 0] vs softmax [0.7, 0.2, 0.1]:
    /// = -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)]
    /// = -log(0.7)
    /// </summary>
    [Fact]
    public void CategoricalCrossEntropy_OneHotEncoded_MatchesExpected()
    {
        var cce = new CategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.7, 0.2, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        var loss = cce.CalculateLoss(predicted, actual);

        // CCE = -[1*log(0.7)] = -log(0.7)
        double expected = -Math.Log(0.7);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void CategoricalCrossEntropy_PerfectPrediction_MinimalLoss()
    {
        var cce = new CategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.999, 0.0005, 0.0005 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        var loss = cce.CalculateLoss(predicted, actual);

        // Near-perfect prediction should have minimal loss
        Assert.True(loss < 0.01);
    }

    [Fact]
    public void CategoricalCrossEntropy_Derivative_MatchesSoftmaxCombinedFormula()
    {
        // CCE derivative when used with softmax simplifies to (predicted - actual)
        var cce = new CategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, 0.3, 0.2 });
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        var gradient = cce.CalculateDerivative(predicted, actual);

        // Expected: (predicted - actual) - not averaged for CCE
        Assert.Equal(0.5 - 1.0, gradient[0], Tolerance);
        Assert.Equal(0.3 - 0.0, gradient[1], Tolerance);
        Assert.Equal(0.2 - 0.0, gradient[2], Tolerance);
    }

    [Fact]
    public void CategoricalCrossEntropy_WrongClass_HighLoss()
    {
        var cce = new CategoricalCrossEntropyLoss<double>();
        // Confidently predicting wrong class
        var predicted = new Vector<double>(new[] { 0.1, 0.8, 0.1 }); // Predicts class 1
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });    // Actual class 0

        var loss = cce.CalculateLoss(predicted, actual);

        // CCE = -log(0.1) which is high
        double expected = -Math.Log(0.1);
        Assert.Equal(expected, loss, Tolerance);
    }

    #endregion

    #region Squared Hinge Loss - Mathematical Verification

    /// <summary>
    /// Squared Hinge: L = max(0, 1 - y*f(x))²
    /// For y=1, f(x)=2: L = max(0, 1-2)² = max(0,-1)² = 0
    /// For y=1, f(x)=0.5: L = max(0, 1-0.5)² = 0.5² = 0.25
    /// For y=1, f(x)=-1: L = max(0, 1-(-1))² = 2² = 4
    /// </summary>
    [Fact]
    public void SquaredHinge_LargeMargin_ReturnsZero()
    {
        var sqHinge = new SquaredHingeLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = sqHinge.CalculateLoss(predicted, actual);

        // max(0, 1 - 1*2)² = max(0, -1)² = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void SquaredHinge_SmallMargin_ReturnsSquared()
    {
        var sqHinge = new SquaredHingeLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = sqHinge.CalculateLoss(predicted, actual);

        // max(0, 1 - 1*0.5)² = max(0, 0.5)² = 0.25
        Assert.Equal(0.25, loss, Tolerance);
    }

    [Fact]
    public void SquaredHinge_WrongPrediction_ReturnsLargeSquared()
    {
        var sqHinge = new SquaredHingeLoss<double>();
        var predicted = new Vector<double>(new[] { -1.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = sqHinge.CalculateLoss(predicted, actual);

        // max(0, 1 - 1*(-1))² = max(0, 2)² = 4
        Assert.Equal(4.0, loss, Tolerance);
    }

    [Fact]
    public void SquaredHinge_GradientCheck_NumericalVsAnalytical()
    {
        var sqHinge = new SquaredHingeLoss<double>();
        var predicted = new Vector<double>(new[] { 0.3, 1.5, -0.5 });
        var actual = new Vector<double>(new[] { 1.0, 1.0, -1.0 });

        var analyticalGradient = sqHinge.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += NumericalEpsilon;
            predictedMinus[i] -= NumericalEpsilon;

            double lossPlus = sqHinge.CalculateLoss(predictedPlus, actual);
            double lossMinus = sqHinge.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Exponential Loss - Mathematical Verification

    /// <summary>
    /// Exponential: L = (1/n) * Σ exp(-y * f(x))
    /// For y=1, f(x)=1: exp(-1) ≈ 0.368
    /// For y=1, f(x)=-1: exp(1) ≈ 2.718
    /// </summary>
    [Fact]
    public void Exponential_CorrectPrediction_SmallLoss()
    {
        var exp = new ExponentialLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = exp.CalculateLoss(predicted, actual);

        // exp(-1*1) = exp(-1) ≈ 0.368
        double expected = Math.Exp(-1);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Exponential_WrongPrediction_LargeLoss()
    {
        var exp = new ExponentialLoss<double>();
        var predicted = new Vector<double>(new[] { -1.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = exp.CalculateLoss(predicted, actual);

        // exp(-1*(-1)) = exp(1) ≈ 2.718
        double expected = Math.Exp(1);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Exponential_HandCalculated_MatchesExpected()
    {
        var exp = new ExponentialLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0, -1.0 });
        var actual = new Vector<double>(new[] { 1.0, -1.0 });

        var loss = exp.CalculateLoss(predicted, actual);

        // (exp(-1*2) + exp(-(-1)*(-1))) / 2
        // = (exp(-2) + exp(-1)) / 2
        double expected = (Math.Exp(-2) + Math.Exp(-1)) / 2.0;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Exponential_GradientCheck_NumericalVsAnalytical()
    {
        var exp = new ExponentialLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, -0.5 });
        var actual = new Vector<double>(new[] { 1.0, -1.0 });

        var analyticalGradient = exp.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += NumericalEpsilon;
            predictedMinus[i] -= NumericalEpsilon;

            double lossPlus = exp.CalculateLoss(predictedPlus, actual);
            double lossMinus = exp.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Margin Loss - Mathematical Verification

    /// <summary>
    /// Margin Loss (Capsule Networks):
    /// L = T_c * max(0, m+ - v)² + λ * (1-T_c) * max(0, v - m-)²
    /// Default: m+ = 0.9, m- = 0.1, λ = 0.5
    /// </summary>
    [Fact]
    public void Margin_ClassPresent_HighOutput_NoLoss()
    {
        var margin = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        var predicted = new Vector<double>(new[] { 0.95 });
        var actual = new Vector<double>(new[] { 1.0 }); // Class present

        var loss = margin.CalculateLoss(predicted, actual);

        // T_c=1, v=0.95 >= m+=0.9, so term1 = 0
        // (1-T_c)=0, so term2 = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void Margin_ClassPresent_LowOutput_HasLoss()
    {
        var margin = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 }); // Class present

        var loss = margin.CalculateLoss(predicted, actual);

        // T_c=1, v=0.5 < m+=0.9
        // term1 = 1 * max(0, 0.9-0.5)² = 0.4² = 0.16
        double expected = 0.16;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Margin_ClassAbsent_HighOutput_HasLoss()
    {
        var margin = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 0.0 }); // Class absent

        var loss = margin.CalculateLoss(predicted, actual);

        // T_c=0, so term1 = 0
        // (1-T_c)=1, v=0.5 > m-=0.1
        // term2 = 0.5 * max(0, 0.5-0.1)² = 0.5 * 0.4² = 0.5 * 0.16 = 0.08
        double expected = 0.08;
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void Margin_ClassAbsent_LowOutput_NoLoss()
    {
        var margin = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        var predicted = new Vector<double>(new[] { 0.05 });
        var actual = new Vector<double>(new[] { 0.0 }); // Class absent

        var loss = margin.CalculateLoss(predicted, actual);

        // T_c=0, so term1 = 0
        // (1-T_c)=1, v=0.05 < m-=0.1
        // term2 = 0.5 * max(0, 0.05-0.1)² = 0.5 * 0 = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    #endregion

    #region Mean Bias Error - Mathematical Verification

    /// <summary>
    /// MBE = (1/n) * Σ(actual - predicted)
    /// Positive MBE = under-prediction, Negative MBE = over-prediction
    /// </summary>
    [Fact]
    public void MeanBiasError_UnderPrediction_ReturnsPositive()
    {
        var mbe = new MeanBiasErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 }); // All actual > predicted

        var loss = mbe.CalculateLoss(predicted, actual);

        // MBE = mean(actual - predicted) = mean([1, 1, 1]) = 1
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void MeanBiasError_OverPrediction_ReturnsNegative()
    {
        var mbe = new MeanBiasErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 3.0, 4.0, 5.0 });
        var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 }); // All actual < predicted

        var loss = mbe.CalculateLoss(predicted, actual);

        // MBE = mean(actual - predicted) = mean([-1, -1, -1]) = -1
        Assert.Equal(-1.0, loss, Tolerance);
    }

    [Fact]
    public void MeanBiasError_Balanced_ReturnsZero()
    {
        var mbe = new MeanBiasErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 3.0 });
        var actual = new Vector<double>(new[] { 2.0, 2.0 }); // Errors cancel out

        var loss = mbe.CalculateLoss(predicted, actual);

        // MBE = mean([1, -1]) = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void MeanBiasError_Derivative_IsConstant()
    {
        var mbe = new MeanBiasErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 5.0, 10.0, 15.0 });

        var derivative = mbe.CalculateDerivative(predicted, actual);

        // Derivative of MBE w.r.t. predicted is -1/n for all elements
        double expected = -1.0 / 3.0;
        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.Equal(expected, derivative[i], Tolerance);
        }
    }

    #endregion

    #region Modified Huber Loss - Mathematical Verification

    /// <summary>
    /// Modified Huber:
    /// z = y * f(x)
    /// For z >= -1: max(0, 1-z)²
    /// For z < -1: -4*z
    /// </summary>
    [Fact]
    public void ModifiedHuber_CorrectWithMargin_ReturnsZero()
    {
        var mh = new ModifiedHuberLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = mh.CalculateLoss(predicted, actual);

        // z = 1*2 = 2 >= 1, so loss = max(0, 1-2)² = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_SmallMargin_ReturnsQuadratic()
    {
        var mh = new ModifiedHuberLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = mh.CalculateLoss(predicted, actual);

        // z = 1*0.5 = 0.5, -1 <= z < 1
        // loss = max(0, 1-0.5)² = 0.5² = 0.25
        Assert.Equal(0.25, loss, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_VeryWrong_ReturnsLinear()
    {
        var mh = new ModifiedHuberLoss<double>();
        var predicted = new Vector<double>(new[] { -2.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = mh.CalculateLoss(predicted, actual);

        // z = 1*(-2) = -2 < -1
        // loss = -4*z = -4*(-2) = 8
        Assert.Equal(8.0, loss, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_BoundaryCase_ReturnsQuadratic()
    {
        var mh = new ModifiedHuberLoss<double>();
        var predicted = new Vector<double>(new[] { -1.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = mh.CalculateLoss(predicted, actual);

        // z = 1*(-1) = -1 (boundary case, z >= -1)
        // loss = max(0, 1-(-1))² = 2² = 4
        Assert.Equal(4.0, loss, Tolerance);
    }

    #endregion

    #region Wasserstein Loss - Mathematical Verification

    /// <summary>
    /// Wasserstein: L = -mean(predicted * actual)
    /// Labels are +1 for real, -1 for fake
    /// </summary>
    [Fact]
    public void Wasserstein_RealSamplesHighScore_LowLoss()
    {
        var wass = new WassersteinLoss<double>();
        var predicted = new Vector<double>(new[] { 5.0, 3.0 }); // High scores
        var actual = new Vector<double>(new[] { 1.0, 1.0 });    // Real samples

        var loss = wass.CalculateLoss(predicted, actual);

        // L = -mean([5*1, 3*1]) = -mean([5, 3]) = -4
        Assert.Equal(-4.0, loss, Tolerance);
    }

    [Fact]
    public void Wasserstein_FakeSamplesLowScore_LowLoss()
    {
        var wass = new WassersteinLoss<double>();
        var predicted = new Vector<double>(new[] { -5.0, -3.0 }); // Low (negative) scores
        var actual = new Vector<double>(new[] { -1.0, -1.0 });    // Fake samples

        var loss = wass.CalculateLoss(predicted, actual);

        // L = -mean([(-5)*(-1), (-3)*(-1)]) = -mean([5, 3]) = -4
        Assert.Equal(-4.0, loss, Tolerance);
    }

    [Fact]
    public void Wasserstein_MixedSamples_HandCalculated()
    {
        var wass = new WassersteinLoss<double>();
        var predicted = new Vector<double>(new[] { 3.0, -2.0 }); // Real high, Fake low
        var actual = new Vector<double>(new[] { 1.0, -1.0 });    // Real, Fake

        var loss = wass.CalculateLoss(predicted, actual);

        // L = -mean([3*1, (-2)*(-1)]) = -mean([3, 2]) = -2.5
        Assert.Equal(-2.5, loss, Tolerance);
    }

    [Fact]
    public void Wasserstein_GradientCheck_NumericalVsAnalytical()
    {
        var wass = new WassersteinLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, -1.0 });
        var actual = new Vector<double>(new[] { 1.0, -1.0 });

        var analyticalGradient = wass.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += NumericalEpsilon;
            predictedMinus[i] -= NumericalEpsilon;

            double lossPlus = wass.CalculateLoss(predictedPlus, actual);
            double lossMinus = wass.CalculateLoss(predictedMinus, actual);

            double numericalGradient = (lossPlus - lossMinus) / (2 * NumericalEpsilon);
            Assert.Equal(numericalGradient, analyticalGradient[i], GradientTolerance);
        }
    }

    #endregion

    #region Weighted Cross Entropy - Mathematical Verification

    /// <summary>
    /// Weighted BCE = -(1/n) Σ weight * [y*log(p) + (1-y)*log(1-p)]
    /// With uniform weights of 1, this should match standard BCE
    /// </summary>
    [Fact]
    public void WeightedCrossEntropy_HandCalculated_MatchesExpected()
    {
        var wce = new WeightedCrossEntropyLoss<double>(); // Default uniform weights

        var predicted = new Vector<double>(new[] { 0.9 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = wce.CalculateLoss(predicted, actual);

        // WCE = -(1/1) * [1*log(0.9) + 0*log(0.1)] = -log(0.9)
        double expected = -Math.Log(0.9);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void WeightedCrossEntropy_UniformWeights_MatchesBCE()
    {
        var bce = new BinaryCrossEntropyLoss<double>();
        var wce = new WeightedCrossEntropyLoss<double>(); // Default uniform weights

        var predicted = new Vector<double>(new[] { 0.9, 0.3 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        var bceLoss = bce.CalculateLoss(predicted, actual);
        var wceLoss = wce.CalculateLoss(predicted, actual);

        // With uniform weights of 1, WCE should equal BCE
        Assert.Equal(bceLoss, wceLoss, Tolerance);
    }

    [Fact]
    public void WeightedCrossEntropy_WithCustomWeights_ScalesCorrectly()
    {
        var weights1 = new Vector<double>(new[] { 1.0 });
        var weights2 = new Vector<double>(new[] { 2.0 });
        var wce1 = new WeightedCrossEntropyLoss<double>(weights1);
        var wce2 = new WeightedCrossEntropyLoss<double>(weights2);

        var predicted = new Vector<double>(new[] { 0.9 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss1 = wce1.CalculateLoss(predicted, actual);
        var loss2 = wce2.CalculateLoss(predicted, actual);

        // With double weight, loss should be doubled
        Assert.Equal(2 * loss1, loss2, Tolerance);
    }

    [Fact]
    public void WeightedCrossEntropy_ZeroWeight_ZeroLoss()
    {
        var weights = new Vector<double>(new[] { 0.0 });
        var wce = new WeightedCrossEntropyLoss<double>(weights);

        var predicted = new Vector<double>(new[] { 0.5 });
        var actual = new Vector<double>(new[] { 1.0 });

        var loss = wce.CalculateLoss(predicted, actual);

        // With zero weight, loss should be zero
        Assert.Equal(0.0, loss, Tolerance);
    }

    #endregion

    #region Contrastive Loss - Mathematical Verification

    /// <summary>
    /// Contrastive Loss for similar pairs: distance²
    /// Contrastive Loss for dissimilar pairs: max(0, margin - distance)²
    /// </summary>
    [Fact]
    public void Contrastive_SimilarPairs_PenalizesDistance()
    {
        var contrastive = new ContrastiveLoss<double>(margin: 1.0);
        var output1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var output2 = new Vector<double>(new[] { 2.0, 0.0, 0.0 });

        var loss = contrastive.CalculateLoss(output1, output2, 1.0); // Similar pair

        // Distance = sqrt((2-1)² + 0 + 0) = 1
        // Loss = 1 * distance² = 1² = 1
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void Contrastive_DissimilarPairs_BelowMargin_PenalizesProximity()
    {
        var contrastive = new ContrastiveLoss<double>(margin: 2.0);
        var output1 = new Vector<double>(new[] { 0.0, 0.0 });
        var output2 = new Vector<double>(new[] { 1.0, 0.0 }); // Distance = 1 < margin = 2

        var loss = contrastive.CalculateLoss(output1, output2, 0.0); // Dissimilar pair

        // Distance = 1 < margin = 2
        // Loss = (1-0) * max(0, 2 - 1)² = 1² = 1
        Assert.Equal(1.0, loss, Tolerance);
    }

    [Fact]
    public void Contrastive_DissimilarPairs_AboveMargin_NoLoss()
    {
        var contrastive = new ContrastiveLoss<double>(margin: 1.0);
        var output1 = new Vector<double>(new[] { 0.0, 0.0 });
        var output2 = new Vector<double>(new[] { 2.0, 0.0 }); // Distance = 2 > margin = 1

        var loss = contrastive.CalculateLoss(output1, output2, 0.0); // Dissimilar pair

        // Distance = 2 > margin = 1
        // Loss = (1-0) * max(0, 1 - 2)² = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void Contrastive_StandardAPI_ThrowsNotSupported()
    {
        var contrastive = new ContrastiveLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 });

        Assert.Throws<NotSupportedException>(() => contrastive.CalculateLoss(predicted, actual));
    }

    #endregion

    #region Triplet Loss - Mathematical Verification

    /// <summary>
    /// Triplet Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    /// </summary>
    [Fact]
    public void Triplet_GoodEmbedding_NoLoss()
    {
        var triplet = new TripletLoss<double>(margin: 1.0);
        var anchor = new Matrix<double>(new double[,] { { 0.0, 0.0 } });
        var positive = new Matrix<double>(new double[,] { { 1.0, 0.0 } }); // Distance = 1
        var negative = new Matrix<double>(new double[,] { { 5.0, 0.0 } }); // Distance = 5

        var loss = triplet.CalculateLoss(anchor, positive, negative);

        // d_pos = 1, d_neg = 5
        // loss = max(0, 1 - 5 + 1) = max(0, -3) = 0
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void Triplet_BadEmbedding_HasLoss()
    {
        var triplet = new TripletLoss<double>(margin: 1.0);
        var anchor = new Matrix<double>(new double[,] { { 0.0, 0.0 } });
        var positive = new Matrix<double>(new double[,] { { 3.0, 0.0 } }); // Distance = 3
        var negative = new Matrix<double>(new double[,] { { 1.0, 0.0 } }); // Distance = 1

        var loss = triplet.CalculateLoss(anchor, positive, negative);

        // d_pos = 3, d_neg = 1
        // loss = max(0, 3 - 1 + 1) = max(0, 3) = 3
        Assert.Equal(3.0, loss, Tolerance);
    }

    [Fact]
    public void Triplet_BatchedSamples_AveragesLoss()
    {
        var triplet = new TripletLoss<double>(margin: 1.0);
        var anchor = new Matrix<double>(new double[,] {
            { 0.0, 0.0 },
            { 0.0, 0.0 }
        });
        var positive = new Matrix<double>(new double[,] {
            { 1.0, 0.0 },  // d_pos = 1
            { 2.0, 0.0 }   // d_pos = 2
        });
        var negative = new Matrix<double>(new double[,] {
            { 5.0, 0.0 },  // d_neg = 5 -> loss = max(0, 1-5+1) = 0
            { 2.5, 0.0 }   // d_neg = 2.5 -> loss = max(0, 2-2.5+1) = 0.5
        });

        var loss = triplet.CalculateLoss(anchor, positive, negative);

        // Average of [0, 0.5] = 0.25
        Assert.Equal(0.25, loss, Tolerance);
    }

    [Fact]
    public void Triplet_StandardAPI_ThrowsNotSupported()
    {
        var triplet = new TripletLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 });

        Assert.Throws<NotSupportedException>(() => triplet.CalculateLoss(predicted, actual));
    }

    #endregion

    #region Ordinal Regression Loss - Mathematical Verification

    /// <summary>
    /// Ordinal Regression Loss uses binary logistic loss at each threshold.
    /// For K classes, there are K-1 thresholds.
    /// L = Σ log(1 + exp(-indicator * predicted)) for each threshold
    /// </summary>
    [Fact]
    public void OrdinalRegression_ConstructionWithValidClasses()
    {
        var ordinal = new OrdinalRegressionLoss<double>(numClasses: 5);
        Assert.NotNull(ordinal);
    }

    [Fact]
    public void OrdinalRegression_InvalidClassesThrows()
    {
        Assert.Throws<ArgumentException>(() => new OrdinalRegressionLoss<double>(numClasses: 1));
    }

    [Fact]
    public void OrdinalRegression_HandCalculated_ThreeClasses()
    {
        // 3 classes: 0, 1, 2 with 2 thresholds
        var ordinal = new OrdinalRegressionLoss<double>(numClasses: 3);
        var predicted = new Vector<double>(new[] { 1.0 });
        var actual = new Vector<double>(new[] { 1.0 }); // Class 1 (middle)

        // For actual=1, threshold j=0: indicator=1 (1>0), j=1: indicator=0 (1>1 is false)
        // Loss = log(1 + exp(-1*1)) + log(1 + exp(-0*1))
        //      = log(1 + exp(-1)) + log(2)
        double threshold0Loss = Math.Log(1 + Math.Exp(-1));
        double threshold1Loss = Math.Log(2); // exp(-0) = 1
        double expected = threshold0Loss + threshold1Loss;

        var loss = ordinal.CalculateLoss(predicted, actual);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void OrdinalRegression_LowestClass_OnlyNegativeIndicators()
    {
        // 3 classes: for actual=0, all indicators are 0 (0>j is false for all j>=0)
        var ordinal = new OrdinalRegressionLoss<double>(numClasses: 3);
        var predicted = new Vector<double>(new[] { 1.0 });
        var actual = new Vector<double>(new[] { 0.0 }); // Lowest class

        // Both thresholds: indicator=0
        // Loss = 2 * log(1 + exp(0)) = 2 * log(2)
        double expected = 2 * Math.Log(2);

        var loss = ordinal.CalculateLoss(predicted, actual);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void OrdinalRegression_HighestClass_AllPositiveIndicators()
    {
        // 3 classes: for actual=2, all indicators are 1 (2>0 and 2>1 are true)
        var ordinal = new OrdinalRegressionLoss<double>(numClasses: 3);
        var predicted = new Vector<double>(new[] { 2.0 });
        var actual = new Vector<double>(new[] { 2.0 }); // Highest class

        // Both thresholds: indicator=1
        // Loss = 2 * log(1 + exp(-2))
        double expected = 2 * Math.Log(1 + Math.Exp(-2));

        var loss = ordinal.CalculateLoss(predicted, actual);
        Assert.Equal(expected, loss, Tolerance);
    }

    #endregion

    #region Sparse Categorical Cross Entropy - Mathematical Verification

    /// <summary>
    /// SCCE = -log(predicted[class_index])
    /// predicted = class probabilities, actual = class indices
    /// </summary>
    [Fact]
    public void SparseCategoricalCrossEntropy_SingleSample_MatchesExpected()
    {
        var scce = new SparseCategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.7 }); // 3 class probabilities
        var actual = new Vector<double>(new[] { 2.0 }); // Class index 2

        var loss = scce.CalculateLoss(predicted, actual);

        // Loss = -log(predicted[2]) = -log(0.7)
        double expected = -Math.Log(0.7);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void SparseCategoricalCrossEntropy_FirstClass_MatchesExpected()
    {
        var scce = new SparseCategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.8, 0.1, 0.1 }); // 3 class probabilities
        var actual = new Vector<double>(new[] { 0.0 }); // Class index 0

        var loss = scce.CalculateLoss(predicted, actual);

        // Loss = -log(predicted[0]) = -log(0.8)
        double expected = -Math.Log(0.8);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void SparseCategoricalCrossEntropy_PerfectPrediction_MinimalLoss()
    {
        var scce = new SparseCategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.001, 0.001, 0.998 });
        var actual = new Vector<double>(new[] { 2.0 }); // Class index 2

        var loss = scce.CalculateLoss(predicted, actual);

        // Loss should be very small for near-perfect prediction
        Assert.True(loss < 0.01);
    }

    [Fact]
    public void SparseCategoricalCrossEntropy_InvalidClassIndex_Throws()
    {
        var scce = new SparseCategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, 0.5 }); // 2 classes (0, 1)
        var actual = new Vector<double>(new[] { 5.0 }); // Invalid class index

        Assert.Throws<ArgumentException>(() => scce.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void SparseCategoricalCrossEntropy_NegativeClassIndex_Throws()
    {
        var scce = new SparseCategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.5, 0.5 });
        var actual = new Vector<double>(new[] { -1.0 }); // Negative class index

        Assert.Throws<ArgumentException>(() => scce.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void SparseCategoricalCrossEntropy_GradientCheck_CorrectClass()
    {
        var scce = new SparseCategoricalCrossEntropyLoss<double>();
        var predicted = new Vector<double>(new[] { 0.3, 0.5, 0.2 });
        var actual = new Vector<double>(new[] { 1.0 }); // Class index 1

        var gradient = scce.CalculateDerivative(predicted, actual);

        // Gradient should be -1/p for the correct class, 0 for others
        // For class 1 (index 1): gradient = -1 / 0.5 = -2
        Assert.Equal(-2.0, gradient[1], Tolerance);
        Assert.Equal(0.0, gradient[0], Tolerance);
        Assert.Equal(0.0, gradient[2], Tolerance);
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void BCE_ExtremeValues_NoNaNOrInfinity()
    {
        var bce = new BinaryCrossEntropyLoss<double>();

        // Very close to 0 and 1
        var predicted = new Vector<double>(new[] { 0.0001, 0.9999 });
        var actual = new Vector<double>(new[] { 0.0, 1.0 });

        var loss = bce.CalculateLoss(predicted, actual);

        Assert.False(double.IsNaN(loss));
        Assert.False(double.IsInfinity(loss));
    }

    [Fact]
    public void CrossEntropy_ZeroProbability_NoNaNOrInfinity()
    {
        var ce = new CrossEntropyLoss<double>();

        // Very small probability
        var predicted = new Vector<double>(new[] { 0.0001, 0.9999 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        var loss = ce.CalculateLoss(predicted, actual);

        Assert.False(double.IsNaN(loss));
        Assert.False(double.IsInfinity(loss));
    }

    [Fact]
    public void AllLossFunctions_LargeValues_NoOverflow()
    {
        var losses = new ILossFunction<double>[]
        {
            new MeanSquaredErrorLoss<double>(),
            new MeanAbsoluteErrorLoss<double>(),
            new HuberLoss<double>(),
            new LogCoshLoss<double>(),
            new HingeLoss<double>()
        };

        var predicted = new Vector<double>(new[] { 1000.0, 2000.0, 3000.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        foreach (var loss in losses)
        {
            var lossValue = loss.CalculateLoss(predicted, actual);
            Assert.False(double.IsNaN(lossValue), $"{loss.GetType().Name} returned NaN");
            Assert.False(double.IsInfinity(lossValue), $"{loss.GetType().Name} returned Infinity");
        }
    }

    [Fact]
    public void AllLossFunctions_ZeroVector_HandledGracefully()
    {
        var losses = new ILossFunction<double>[]
        {
            new MeanSquaredErrorLoss<double>(),
            new MeanAbsoluteErrorLoss<double>(),
            new HuberLoss<double>()
        };

        var predicted = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        foreach (var loss in losses)
        {
            var lossValue = loss.CalculateLoss(predicted, actual);
            Assert.False(double.IsNaN(lossValue), $"{loss.GetType().Name} returned NaN for zero vectors");
            Assert.Equal(0.0, lossValue, Tolerance);
        }
    }

    #endregion
}
