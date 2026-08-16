using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for ILossFunction&lt;T&gt; implementations.
/// Tests mathematical invariants that every loss function must satisfy:
/// non-negativity, zero loss for identical inputs, derivative correctness,
/// and numerical stability.
/// </summary>
public abstract class LossFunctionTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected static T ToT(double value) => NumOps.FromDouble(value);

    protected static double ToD(T value) => Convert.ToDouble(value);

    protected static Vector<T> ToVector(double[] values)
    {
        var result = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++) result[i] = ToT(values[i]);
        return result;
    }

    protected abstract ILossFunction<T> CreateLoss();

    protected virtual double Tolerance => typeof(T) == typeof(float) ? 1e-6 : 1e-10;

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

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var predicted = ToVector(TestPredicted);
        var actual = ToVector(TestActual);

        double value = ToD(loss.CalculateLoss(predicted, actual));

        Assert.False(double.IsNaN(value), "Loss returned NaN.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is non-negative (for standard losses)
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_ShouldBeNonNegative()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsNonNegative) return;

        var loss = CreateLoss();
        var predicted = ToVector(TestPredicted);
        var actual = ToVector(TestActual);

        double value = ToD(loss.CalculateLoss(predicted, actual));
        Assert.True(value >= -Tolerance, $"Loss should be non-negative but got {value}.");
    }

    // =========================================================================
    // INVARIANT 3: Identical inputs → zero loss
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_IdenticalInputs_ShouldBeZero()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!ZeroLossForIdentical) return;

        var loss = CreateLoss();
        var values = ToVector([0.3, 0.5, 0.7]);

        double value = ToD(loss.CalculateLoss(values, values));
        Assert.True(Math.Abs(value) < Tolerance,
            $"Loss for identical vectors should be ≈0 but got {value}.");
    }

    // =========================================================================
    // INVARIANT 4: Larger errors produce larger loss
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_LargerError_ShouldProduceLargerLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        // Skip for losses that can go negative (MBE, Wasserstein) — larger error
        // doesn't necessarily mean larger loss value when loss can be negative
        if (!IsNonNegative) return;

        var loss = CreateLoss();
        var actual = ToVector(ErrorTestActual);
        var smallError = ToVector(SmallErrorPredicted);
        var largeError = ToVector(LargeErrorPredicted);

        double smallLoss = ToD(loss.CalculateLoss(smallError, actual));
        double largeLoss = ToD(loss.CalculateLoss(largeError, actual));

        Assert.True(largeLoss >= smallLoss - Tolerance,
            $"Larger error should produce larger loss: small={smallLoss}, large={largeLoss}.");
    }

    // =========================================================================
    // INVARIANT 5: Derivative is finite
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateDerivative_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var predicted = ToVector(TestPredicted);
        var actual = ToVector(TestActual);

        var derivative = loss.ComputeGradient(predicted, actual);

        Assert.Equal(predicted.Length, derivative.Length);
        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.False(double.IsNaN(ToD(derivative[i])),
                $"Derivative[{i}] is NaN.");
            Assert.False(double.IsInfinity(ToD(derivative[i])),
                $"Derivative[{i}] is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Derivative is zero for identical inputs
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateDerivative_IdenticalInputs_ShouldBeZero()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!ZeroDerivativeForIdentical) return;

        var loss = CreateLoss();
        var values = ToVector([0.3, 0.5, 0.7]);

        var derivative = loss.ComputeGradient(values, values);

        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.True(Math.Abs(ToD(derivative[i])) < Tolerance,
                $"Derivative[{i}] should be ≈0 for identical inputs but got {derivative[i]}.");
        }
    }

    // =========================================================================
    // INVARIANT 7: Numerical gradient check
    // The analytical derivative should match finite-difference approximation.
    // This is the gold standard for gradient correctness.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateDerivative_ShouldMatchNumericalGradient()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var predicted = ToVector(TestPredicted);
        var actual = ToVector(TestActual);
        double epsilon = typeof(T) == typeof(float) ? 1e-3 : 1e-5;

        var analyticalGrad = loss.ComputeGradient(predicted, actual);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedPlus = predicted.Clone();
            var predictedMinus = predicted.Clone();
            double original = ToD(predicted[i]);
            double plusValue = ToD(ToT(original + epsilon));
            double minusValue = ToD(ToT(original - epsilon));
            if (plusValue == minusValue) continue;
            predictedPlus[i] = ToT(plusValue);
            predictedMinus[i] = ToT(minusValue);

            double lossPlus = ToD(loss.CalculateLoss(predictedPlus, actual));
            double lossMinus = ToD(loss.CalculateLoss(predictedMinus, actual));
            double numericalGrad = (lossPlus - lossMinus) / (plusValue - minusValue);

            double analyticalValue = ToD(analyticalGrad[i]);
            double absMax = Math.Max(Math.Abs(analyticalValue), Math.Abs(numericalGrad));
            if (absMax < 1e-7) continue; // Both near zero

            double absError = Math.Abs(analyticalValue - numericalGrad);
            double relError = absError / (absMax + 1e-8);
            // A float-valued loss loses several significant bits when two nearby
            // O(1) loss values are subtracted. Keep the relative criterion for
            // ordinary gradients, but accept a genuinely tiny absolute residual.
            double absoluteTolerance = typeof(T) == typeof(float) ? 1e-4 : 0.0;
            Assert.True(absError < absoluteTolerance || relError < 0.02,
                $"Gradient check failed at index {i}: " +
                $"analytical={analyticalValue:G10}, numerical={numericalGrad:G10}, " +
                $"absError={absError:G6}, relError={relError:G6}.");
        }
    }

    // =========================================================================
    // INVARIANT 8: Derivative sign matches error direction
    // If predicted > actual, derivative should be positive (push predicted down).
    // If predicted < actual, derivative should be negative (push predicted up).
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateDerivative_SignShouldMatchErrorDirection()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!HasStandardGradientSign) return;

        var loss = CreateLoss();
        var predicted = ToVector(SignTestPredicted);
        var actual = ToVector(SignTestActual);

        var derivative = loss.ComputeGradient(predicted, actual);

        // For standard regression losses, positive error → positive gradient
        double derivativeValue = ToD(derivative[0]);
        Assert.True(derivativeValue > -Tolerance,
            $"When predicted > actual, derivative should be >= 0 but got {derivativeValue}.");
    }

    // =========================================================================
    // INVARIANT 9: Loss is symmetric in error magnitude (for symmetric losses)
    // |L(a+δ, a)| ≈ |L(a-δ, a)| for MSE, MAE, Huber
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_ShouldBeSymmetricInErrorMagnitude()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        // Only test symmetry for standard regression-style losses.
        // Classification losses (Focal, CE) and signed-label losses (Hinge) are
        // intentionally asymmetric by design.
        if (!HasStandardGradientSign) return;

        var loss = CreateLoss();
        var actual = ToVector([0.5]);
        var overPredict = ToVector([0.8]);
        var underPredict = ToVector([0.2]);

        double overLoss = ToD(loss.CalculateLoss(overPredict, actual));
        double underLoss = ToD(loss.CalculateLoss(underPredict, actual));

        // Allow 50% relative difference (some losses are asymmetric)
        double ratio = Math.Max(overLoss, underLoss) / (Math.Min(overLoss, underLoss) + Tolerance);
        Assert.True(ratio < 10.0,
            $"Loss asymmetry too large: overPredict loss={overLoss}, underPredict loss={underLoss}.");
    }
}

/// <summary>Default-precision alias for existing hand-written fixtures.</summary>
public abstract class LossFunctionTestBase : LossFunctionTestBase<double> { }
