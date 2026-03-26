using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for knowledge distillation strategies implementing IDistillationStrategy.
/// Tests mathematical invariants: loss non-negativity, identical-inputs-zero-loss,
/// finiteness, gradient shape consistency, and temperature scaling properties.
/// </summary>
public abstract class DistillationStrategyTestBase
{
    /// <summary>Factory method — subclasses return their concrete strategy instance.</summary>
    protected abstract IDistillationStrategy<double> CreateStrategy();

    /// <summary>Batch size for test data.</summary>
    protected virtual int BatchSize => 4;

    /// <summary>Number of output classes for test data.</summary>
    protected virtual int NumClasses => 8;

    /// <summary>Creates teacher output logits matrix [batch, classes].</summary>
    protected virtual Matrix<double> CreateTeacherOutput()
    {
        var rng = new Random(42);
        var data = new Matrix<double>(BatchSize, NumClasses);
        for (int i = 0; i < BatchSize; i++)
        {
            for (int j = 0; j < NumClasses; j++)
            {
                data[i, j] = rng.NextDouble() * 4.0 - 2.0;
            }
        }

        return data;
    }

    /// <summary>Creates student output logits matrix [batch, classes].</summary>
    protected virtual Matrix<double> CreateStudentOutput()
    {
        var rng = new Random(123);
        var data = new Matrix<double>(BatchSize, NumClasses);
        for (int i = 0; i < BatchSize; i++)
        {
            for (int j = 0; j < NumClasses; j++)
            {
                data[i, j] = rng.NextDouble() * 4.0 - 2.0;
            }
        }

        return data;
    }

    // =========================================================================
    // INVARIANT 1: Distillation loss is non-negative
    // =========================================================================

    [Fact]
    public void ComputeLoss_IsNonNegative()
    {
        var strategy = CreateStrategy();
        var teacher = CreateTeacherOutput();
        var student = CreateStudentOutput();

        double loss = strategy.ComputeLoss(student, teacher);

        Assert.True(loss >= -1e-10,
            $"Distillation loss should be non-negative but got {loss}. " +
            "KL divergence and MSE are both non-negative measures.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is finite
    // =========================================================================

    [Fact]
    public void ComputeLoss_IsFinite()
    {
        var strategy = CreateStrategy();
        var teacher = CreateTeacherOutput();
        var student = CreateStudentOutput();

        double loss = strategy.ComputeLoss(student, teacher);

        Assert.False(double.IsNaN(loss), "Distillation loss is NaN.");
        Assert.False(double.IsInfinity(loss), "Distillation loss is Infinity.");
    }

    // =========================================================================
    // INVARIANT 3: Identical teacher and student produce zero or near-zero loss
    // =========================================================================

    [Fact]
    public void ComputeLoss_IdenticalInputs_ProducesMinimalLoss()
    {
        var strategy = CreateStrategy();
        var output = CreateTeacherOutput();

        double loss = strategy.ComputeLoss(output, output);

        Assert.True(loss < 0.01,
            $"Identical teacher and student should produce near-zero loss but got {loss}. " +
            "When distributions match perfectly, divergence should be zero.");
    }

    // =========================================================================
    // INVARIANT 4: Gradient has correct shape
    // =========================================================================

    [Fact]
    public void ComputeGradient_HasCorrectShape()
    {
        var strategy = CreateStrategy();
        var teacher = CreateTeacherOutput();
        var student = CreateStudentOutput();

        var gradient = strategy.ComputeGradient(student, teacher);

        Assert.Equal(BatchSize, gradient.Rows);
        Assert.Equal(NumClasses, gradient.Columns);
    }

    // =========================================================================
    // INVARIANT 5: Gradient is finite
    // =========================================================================

    [Fact]
    public void ComputeGradient_IsFinite()
    {
        var strategy = CreateStrategy();
        var teacher = CreateTeacherOutput();
        var student = CreateStudentOutput();

        var gradient = strategy.ComputeGradient(student, teacher);

        for (int i = 0; i < gradient.Rows; i++)
        {
            for (int j = 0; j < gradient.Columns; j++)
            {
                Assert.False(double.IsNaN(gradient[i, j]),
                    $"Gradient has NaN at [{i},{j}].");
                Assert.False(double.IsInfinity(gradient[i, j]),
                    $"Gradient has Infinity at [{i},{j}].");
            }
        }
    }

    // =========================================================================
    // INVARIANT 6: Gradient for identical inputs is near-zero
    // =========================================================================

    [Fact]
    public void ComputeGradient_IdenticalInputs_IsNearZero()
    {
        var strategy = CreateStrategy();
        var output = CreateTeacherOutput();

        var gradient = strategy.ComputeGradient(output, output);

        double maxAbsGrad = 0;
        for (int i = 0; i < gradient.Rows; i++)
        {
            for (int j = 0; j < gradient.Columns; j++)
            {
                maxAbsGrad = Math.Max(maxAbsGrad, Math.Abs(gradient[i, j]));
            }
        }

        Assert.True(maxAbsGrad < 0.1,
            $"Gradient for identical inputs should be near-zero but max abs gradient is {maxAbsGrad:E4}. " +
            "When student matches teacher perfectly, there's nothing to learn.");
    }

    // =========================================================================
    // INVARIANT 7: Temperature is positive
    // =========================================================================

    [Fact]
    public void Temperature_IsPositive()
    {
        var strategy = CreateStrategy();
        Assert.True(strategy.Temperature > 0,
            $"Temperature must be positive but got {strategy.Temperature}.");
    }

    // =========================================================================
    // INVARIANT 8: Alpha is in valid range [0, 1]
    // =========================================================================

    [Fact]
    public void Alpha_IsInValidRange()
    {
        var strategy = CreateStrategy();
        Assert.True(strategy.Alpha >= 0 && strategy.Alpha <= 1,
            $"Alpha must be in [0, 1] but got {strategy.Alpha}.");
    }
}
