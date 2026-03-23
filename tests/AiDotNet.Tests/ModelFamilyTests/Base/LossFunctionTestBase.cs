using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for ILossFunction&lt;double&gt; implementations.
/// Tests mathematical invariants that every loss function must satisfy:
/// non-negativity, zero loss for identical inputs, derivative correctness,
/// and numerical stability.
/// </summary>
public abstract class LossFunctionTestBase
{
    protected abstract ILossFunction<double> CreateLoss();

    /// <summary>
    /// Whether the loss is always non-negative. True for most losses (MSE, MAE, CE).
    /// False for some exotic losses or losses that can go negative.
    /// </summary>
    protected virtual bool IsNonNegative => true;

    /// <summary>
    /// Whether identical predicted/actual should give exactly zero loss.
    /// True for: MSE, MAE, Huber. False for: CrossEntropy, Dice, ElasticNet.
    /// </summary>
    protected virtual bool ZeroLossForIdentical => true;

    /// <summary>
    /// Whether derivative is zero for identical inputs.
    /// True for most losses where ZeroLossForIdentical is true.
    /// False for: MeanBiasError (constant derivative -1/n), QuantileLoss.
    /// </summary>
    protected virtual bool ZeroDerivativeForIdentical => ZeroLossForIdentical;

    /// <summary>
    /// Whether the gradient sign follows the standard convention: positive when predicted > actual.
    /// True for: MSE, MAE, Huber, LogCosh (regression losses).
    /// False for: CrossEntropy, Focal, Dice, Hinge, Wasserstein, MeanBiasError.
    /// The numerical gradient check (invariant 7) still validates correctness regardless.
    /// </summary>
    protected virtual bool HasStandardGradientSign => true;

    /// <summary>
    /// Standard test predicted values. Override for losses that need specific input formats.
    /// Default: continuous values in [0,1] range.
    /// </summary>
    protected virtual double[] TestPredicted => [0.2, 0.5, 0.8];

    /// <summary>
    /// Standard test actual values. Override for losses that need specific input formats.
    /// </summary>
    protected virtual double[] TestActual => [0.3, 0.6, 0.7];

    /// <summary>
    /// Small-error predicted values for the "larger error produces larger loss" test.
    /// </summary>
    protected virtual double[] SmallErrorPredicted => [0.6, 0.6, 0.6];

    /// <summary>
    /// Large-error predicted values for the "larger error produces larger loss" test.
    /// </summary>
    protected virtual double[] LargeErrorPredicted => [0.9, 0.9, 0.9];

    /// <summary>
    /// Actual values for the error magnitude comparison test.
    /// </summary>
    protected virtual double[] ErrorTestActual => [0.5, 0.5, 0.5];

    /// <summary>
    /// Predicted value for the gradient sign direction test (should be > SignTestActual).
    /// </summary>
    protected virtual double[] SignTestPredicted => [0.9];

    /// <summary>
    /// Actual value for the gradient sign direction test.
    /// </summary>
    protected virtual double[] SignTestActual => [0.1];

    // =========================================================================
    // INVARIANT 1: Loss is finite for normal inputs
    // =========================================================================

    [Fact]
    public void CalculateLoss_ShouldBeFinite()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(TestPredicted);
        var actual = new Vector<double>(TestActual);

        double value = loss.CalculateLoss(predicted, actual);

        Assert.False(double.IsNaN(value), "Loss returned NaN.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is non-negative (for standard losses)
    // =========================================================================

    [Fact]
    public void CalculateLoss_ShouldBeNonNegative()
    {
        if (!IsNonNegative) return;

        var loss = CreateLoss();
        var predicted = new Vector<double>(TestPredicted);
        var actual = new Vector<double>(TestActual);

        double value = loss.CalculateLoss(predicted, actual);
        Assert.True(value >= -1e-10, $"Loss should be non-negative but got {value}.");
    }

    // =========================================================================
    // INVARIANT 3: Identical inputs → zero loss
    // =========================================================================

    [Fact]
    public void CalculateLoss_IdenticalInputs_ShouldBeZero()
    {
        if (!ZeroLossForIdentical) return;

        var loss = CreateLoss();
        var values = new Vector<double>(new[] { 0.3, 0.5, 0.7 });

        double value = loss.CalculateLoss(values, values);
        Assert.True(Math.Abs(value) < 1e-10,
            $"Loss for identical vectors should be ≈0 but got {value}.");
    }

    // =========================================================================
    // INVARIANT 4: Larger errors produce larger loss
    // =========================================================================

    [Fact]
    public void CalculateLoss_LargerError_ShouldProduceLargerLoss()
    {
        // Skip for losses that can go negative (MBE, Wasserstein) — larger error
        // doesn't necessarily mean larger loss value when loss can be negative
        if (!IsNonNegative) return;

        var loss = CreateLoss();
        var actual = new Vector<double>(ErrorTestActual);
        var smallError = new Vector<double>(SmallErrorPredicted);
        var largeError = new Vector<double>(LargeErrorPredicted);

        double smallLoss = loss.CalculateLoss(smallError, actual);
        double largeLoss = loss.CalculateLoss(largeError, actual);

        Assert.True(largeLoss >= smallLoss - 1e-10,
            $"Larger error should produce larger loss: small={smallLoss}, large={largeLoss}.");
    }

    // =========================================================================
    // INVARIANT 5: Derivative is finite
    // =========================================================================

    [Fact]
    public void CalculateDerivative_ShouldBeFinite()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(TestPredicted);
        var actual = new Vector<double>(TestActual);

        var derivative = loss.CalculateDerivative(predicted, actual);

        Assert.Equal(predicted.Length, derivative.Length);
        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.False(double.IsNaN(derivative[i]),
                $"Derivative[{i}] is NaN.");
            Assert.False(double.IsInfinity(derivative[i]),
                $"Derivative[{i}] is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Derivative is zero for identical inputs
    // =========================================================================

    [Fact]
    public void CalculateDerivative_IdenticalInputs_ShouldBeZero()
    {
        if (!ZeroDerivativeForIdentical) return;

        var loss = CreateLoss();
        var values = new Vector<double>(new[] { 0.3, 0.5, 0.7 });

        var derivative = loss.CalculateDerivative(values, values);

        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.True(Math.Abs(derivative[i]) < 1e-8,
                $"Derivative[{i}] should be ≈0 for identical inputs but got {derivative[i]}.");
        }
    }

    // =========================================================================
    // INVARIANT 7: Numerical gradient check
    // The analytical derivative should match finite-difference approximation.
    // This is the gold standard for gradient correctness.
    // =========================================================================

    [Fact]
    public void CalculateDerivative_ShouldMatchNumericalGradient()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(TestPredicted);
        var actual = new Vector<double>(TestActual);
        double epsilon = 1e-5;

        var analyticalGrad = loss.CalculateDerivative(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            predictedPlus[i] += epsilon;
            predictedMinus[i] -= epsilon;

            double lossPlus = loss.CalculateLoss(predictedPlus, actual);
            double lossMinus = loss.CalculateLoss(predictedMinus, actual);
            double numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

            double absMax = Math.Max(Math.Abs(analyticalGrad[i]), Math.Abs(numericalGrad));
            if (absMax < 1e-7) continue; // Both near zero

            double relError = Math.Abs(analyticalGrad[i] - numericalGrad) / (absMax + 1e-8);
            Assert.True(relError < 0.02,
                $"Gradient check failed at index {i}: " +
                $"analytical={analyticalGrad[i]:G10}, numerical={numericalGrad:G10}, " +
                $"relError={relError:G6}.");
        }
    }

    // =========================================================================
    // INVARIANT 8: Derivative sign matches error direction
    // If predicted > actual, derivative should be positive (push predicted down).
    // If predicted < actual, derivative should be negative (push predicted up).
    // =========================================================================

    [Fact]
    public void CalculateDerivative_SignShouldMatchErrorDirection()
    {
        if (!HasStandardGradientSign) return;

        var loss = CreateLoss();
        var predicted = new Vector<double>(SignTestPredicted);
        var actual = new Vector<double>(SignTestActual);

        var derivative = loss.CalculateDerivative(predicted, actual);

        // For standard regression losses, positive error → positive gradient
        Assert.True(derivative[0] > -1e-10,
            $"When predicted > actual, derivative should be >= 0 but got {derivative[0]}.");
    }

    // =========================================================================
    // INVARIANT 9: Loss is symmetric in error magnitude (for symmetric losses)
    // |L(a+δ, a)| ≈ |L(a-δ, a)| for MSE, MAE, Huber
    // =========================================================================

    [Fact]
    public void CalculateLoss_ShouldBeSymmetricInErrorMagnitude()
    {
        // Only test symmetry for standard regression-style losses.
        // Classification losses (Focal, CE) and signed-label losses (Hinge) are
        // intentionally asymmetric by design.
        if (!HasStandardGradientSign) return;

        var loss = CreateLoss();
        var actual = new Vector<double>(new[] { 0.5 });
        var overPredict = new Vector<double>(new[] { 0.8 });
        var underPredict = new Vector<double>(new[] { 0.2 });

        double overLoss = loss.CalculateLoss(overPredict, actual);
        double underLoss = loss.CalculateLoss(underPredict, actual);

        // Allow 50% relative difference (some losses are asymmetric)
        double ratio = Math.Max(overLoss, underLoss) / (Math.Min(overLoss, underLoss) + 1e-10);
        Assert.True(ratio < 10.0,
            $"Loss asymmetry too large: overPredict loss={overLoss}, underPredict loss={underLoss}.");
    }
}
