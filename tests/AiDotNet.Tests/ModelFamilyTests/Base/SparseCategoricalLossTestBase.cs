using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for sparse categorical loss functions where predicted and actual vectors
/// have different lengths. Predicted = class probabilities (num_classes), actual = class indices (batch_size).
/// Tests mathematical invariants: non-negativity, finiteness, gradient correctness.
/// </summary>
public abstract class SparseCategoricalLossTestBase
{
    protected abstract ILossFunction<double> CreateLoss();

    // =========================================================================
    // INVARIANT 1: Loss is finite for valid inputs
    // =========================================================================

    [Fact]
    public void CalculateLoss_ShouldBeFinite()
    {
        var loss = CreateLoss();
        // 4-class problem, single sample with class index 2
        var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.6, 0.1 });
        var actual = new Vector<double>(new[] { 2.0 });

        double value = loss.CalculateLoss(predicted, actual);

        Assert.False(double.IsNaN(value), "Loss returned NaN.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is non-negative
    // =========================================================================

    [Fact]
    public void CalculateLoss_ShouldBeNonNegative()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.6, 0.1 });
        var actual = new Vector<double>(new[] { 2.0 });

        double value = loss.CalculateLoss(predicted, actual);
        Assert.True(value >= -1e-10,
            $"Sparse categorical cross-entropy should be non-negative but got {value}.");
    }

    // =========================================================================
    // INVARIANT 3: Higher probability at correct class → lower loss
    // =========================================================================

    [Fact]
    public void CalculateLoss_HigherConfidence_ShouldReduceLoss()
    {
        var loss = CreateLoss();
        var actual = new Vector<double>(new[] { 1.0 }); // class 1

        var lowConfidence = new Vector<double>(new[] { 0.4, 0.3, 0.3 });
        var highConfidence = new Vector<double>(new[] { 0.1, 0.8, 0.1 });

        double lowLoss = loss.CalculateLoss(lowConfidence, actual);
        double highLoss = loss.CalculateLoss(highConfidence, actual);

        Assert.True(highLoss < lowLoss + 1e-10,
            $"Higher confidence should produce lower loss: low={lowLoss}, high={highLoss}.");
    }

    // =========================================================================
    // INVARIANT 4: Perfect prediction → near-zero loss
    // =========================================================================

    [Fact]
    public void CalculateLoss_PerfectPrediction_ShouldBeNearZero()
    {
        var loss = CreateLoss();
        // Near-perfect prediction (can't use exactly 1.0 due to log)
        var predicted = new Vector<double>(new[] { 0.001, 0.998, 0.001 });
        var actual = new Vector<double>(new[] { 1.0 }); // class 1

        double value = loss.CalculateLoss(predicted, actual);
        Assert.True(value < 0.01,
            $"Loss for near-perfect prediction should be near zero but got {value}.");
    }

    // =========================================================================
    // INVARIANT 5: Derivative is finite
    // =========================================================================

    [Fact]
    public void CalculateDerivative_ShouldBeFinite()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.6, 0.1 });
        var actual = new Vector<double>(new[] { 2.0 });

        var derivative = loss.CalculateDerivative(predicted, actual);

        Assert.Equal(predicted.Length, derivative.Length);
        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.False(double.IsNaN(derivative[i]), $"Derivative[{i}] is NaN.");
            Assert.False(double.IsInfinity(derivative[i]), $"Derivative[{i}] is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Gradient at correct class should be negative (push probability up)
    // =========================================================================

    [Fact]
    public void CalculateDerivative_CorrectClass_ShouldBeNegative()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(new[] { 0.3, 0.4, 0.3 });
        var actual = new Vector<double>(new[] { 1.0 }); // class 1

        var derivative = loss.CalculateDerivative(predicted, actual);

        // Gradient at the correct class should be negative (to increase probability)
        Assert.True(derivative[1] < 0,
            $"Gradient at correct class should be negative but got {derivative[1]}.");
    }

    // =========================================================================
    // INVARIANT 7: Invalid class index should throw
    // =========================================================================

    [Fact]
    public void CalculateLoss_InvalidClassIndex_ShouldThrow()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(new[] { 0.5, 0.5 });
        var actual = new Vector<double>(new[] { 5.0 }); // out of bounds

        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }

    // =========================================================================
    // INVARIANT 8: Multiple samples in batch
    // =========================================================================

    [Fact]
    public void CalculateLoss_BatchInput_ShouldBeFinite()
    {
        var loss = CreateLoss();
        var predicted = new Vector<double>(new[] { 0.2, 0.3, 0.5 });
        var actual = new Vector<double>(new[] { 0.0, 2.0, 1.0 }); // batch of 3 samples

        double value = loss.CalculateLoss(predicted, actual);

        Assert.False(double.IsNaN(value), "Batch loss returned NaN.");
        Assert.False(double.IsInfinity(value), "Batch loss returned Infinity.");
    }
}
