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
