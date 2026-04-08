using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for triplet-style loss functions that operate on (anchor, positive, negative) matrix triplets.
/// Tests mathematical invariants: non-negativity, margin enforcement, gradient correctness, and finiteness.
/// </summary>
public abstract class TripletLossTestBase
{
    protected abstract TripletLoss<double> CreateLoss();

    private static Matrix<double> CreateMatrix(int rows, int cols, double fillValue)
    {
        var m = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = fillValue + i * 0.1 + j * 0.05;
        return m;
    }

    private static Matrix<double> CreateConstantMatrix(int rows, int cols, double value)
    {
        var m = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = value;
        return m;
    }

    // =========================================================================
    // INVARIANT 1: Loss is finite for normal inputs
    // =========================================================================

    [Fact]
    public void CalculateLoss_ShouldBeFinite()
    {
        var loss = CreateLoss();
        var anchor = CreateMatrix(2, 3, 0.1);
        var positive = CreateMatrix(2, 3, 0.15);
        var negative = CreateMatrix(2, 3, 0.9);

        double value = loss.CalculateLoss(anchor, positive, negative);

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
        var anchor = CreateMatrix(2, 3, 0.1);
        var positive = CreateMatrix(2, 3, 0.15);
        var negative = CreateMatrix(2, 3, 0.9);

        double value = loss.CalculateLoss(anchor, positive, negative);
        Assert.True(value >= -1e-10, $"Triplet loss should be non-negative but got {value}.");
    }

    // =========================================================================
    // INVARIANT 3: Loss is zero when positive is much closer than negative
    // =========================================================================

    [Fact]
    public void CalculateLoss_WellSeparated_ShouldBeZero()
    {
        var loss = CreateLoss();
        var anchor = CreateConstantMatrix(1, 3, 0.5);
        var positive = CreateConstantMatrix(1, 3, 0.5);
        var negative = CreateConstantMatrix(1, 3, 100.0);

        double value = loss.CalculateLoss(anchor, positive, negative);
        Assert.True(value < 1e-10,
            $"Loss should be 0 when positive distance << negative distance, but got {value}.");
    }

    // =========================================================================
    // INVARIANT 4: Loss increases when positive moves further from anchor
    // =========================================================================

    [Fact]
    public void CalculateLoss_CloserPositive_ShouldProduceSmallerLoss()
    {
        var loss = CreateLoss();
        var anchor = CreateConstantMatrix(1, 3, 0.0);
        var negative = CreateConstantMatrix(1, 3, 2.0);
        var closePositive = CreateConstantMatrix(1, 3, 0.1);
        var farPositive = CreateConstantMatrix(1, 3, 1.5);

        double closeLoss = loss.CalculateLoss(anchor, closePositive, negative);
        double farLoss = loss.CalculateLoss(anchor, farPositive, negative);

        Assert.True(farLoss >= closeLoss - 1e-10,
            $"Farther positive should produce larger loss: close={closeLoss}, far={farLoss}.");
    }

    // =========================================================================
    // INVARIANT 5: Gradients are finite
    // =========================================================================

    [Fact]
    public void CalculateDerivative_ShouldBeFinite()
    {
        var loss = CreateLoss();
        var anchor = CreateMatrix(2, 3, 0.1);
        var positive = CreateMatrix(2, 3, 0.15);
        var negative = CreateMatrix(2, 3, 0.9);

        var (anchorGrad, positiveGrad, negativeGrad) = loss.CalculateDerivative(anchor, positive, negative);

        for (int i = 0; i < anchorGrad.Rows; i++)
        {
            for (int j = 0; j < anchorGrad.Columns; j++)
            {
                Assert.False(double.IsNaN(anchorGrad[i, j]), $"Anchor gradient[{i},{j}] is NaN.");
                Assert.False(double.IsInfinity(anchorGrad[i, j]), $"Anchor gradient[{i},{j}] is Infinity.");
                Assert.False(double.IsNaN(positiveGrad[i, j]), $"Positive gradient[{i},{j}] is NaN.");
                Assert.False(double.IsInfinity(positiveGrad[i, j]), $"Positive gradient[{i},{j}] is Infinity.");
                Assert.False(double.IsNaN(negativeGrad[i, j]), $"Negative gradient[{i},{j}] is NaN.");
                Assert.False(double.IsInfinity(negativeGrad[i, j]), $"Negative gradient[{i},{j}] is Infinity.");
            }
        }
    }

    // =========================================================================
    // INVARIANT 6: Dimension validation
    // =========================================================================

    [Fact]
    public void CalculateLoss_MismatchedDimensions_ShouldThrow()
    {
        var loss = CreateLoss();
        var anchor = CreateMatrix(2, 3, 0.1);
        var positive = CreateMatrix(1, 3, 0.1); // wrong rows

        Assert.Throws<ArgumentException>(() =>
            loss.CalculateLoss(anchor, positive, anchor));
    }
}
