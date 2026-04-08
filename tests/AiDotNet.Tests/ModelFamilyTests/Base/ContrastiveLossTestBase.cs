using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for noise contrastive estimation loss functions that operate on
/// target logits (Vector) and noise logits (Matrix).
/// Tests mathematical invariants: non-negativity, finiteness, and gradient correctness.
/// </summary>
public abstract class ContrastiveLossTestBase
{
    protected abstract NoiseContrastiveEstimationLoss<double> CreateLoss();

    /// <summary>The number of noise samples configured for this loss instance.</summary>
    protected virtual int NumNoiseSamples => 10;

    // =========================================================================
    // INVARIANT 1: Loss is finite for normal inputs
    // =========================================================================

    [Fact]
    public void Calculate_ShouldBeFinite()
    {
        var loss = CreateLoss();
        int batchSize = 3;
        var targetLogits = new Vector<double>(batchSize);
        var noiseLogits = new Matrix<double>(batchSize, NumNoiseSamples);

        // Fill with reasonable values
        for (int i = 0; i < batchSize; i++)
        {
            targetLogits[i] = 1.0 + i * 0.5;
            for (int j = 0; j < NumNoiseSamples; j++)
            {
                noiseLogits[i, j] = -0.5 + j * 0.1;
            }
        }

        double value = loss.Calculate(targetLogits, noiseLogits);

        Assert.False(double.IsNaN(value), "Loss returned NaN.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is non-negative
    // =========================================================================

    [Fact]
    public void Calculate_ShouldBeNonNegative()
    {
        var loss = CreateLoss();
        int batchSize = 3;
        var targetLogits = new Vector<double>(batchSize);
        var noiseLogits = new Matrix<double>(batchSize, NumNoiseSamples);

        for (int i = 0; i < batchSize; i++)
        {
            targetLogits[i] = 2.0;
            for (int j = 0; j < NumNoiseSamples; j++)
            {
                noiseLogits[i, j] = -1.0;
            }
        }

        double value = loss.Calculate(targetLogits, noiseLogits);
        Assert.True(value >= -1e-10, $"NCE loss should be non-negative but got {value}.");
    }

    // =========================================================================
    // INVARIANT 3: Higher target logits should reduce loss
    // =========================================================================

    [Fact]
    public void Calculate_HigherTargetLogits_ShouldReduceLoss()
    {
        var loss = CreateLoss();
        int batchSize = 2;
        var noiseLogits = new Matrix<double>(batchSize, NumNoiseSamples);
        for (int i = 0; i < batchSize; i++)
            for (int j = 0; j < NumNoiseSamples; j++)
                noiseLogits[i, j] = -1.0;

        var lowTarget = new Vector<double>(new[] { 0.5, 0.5 });
        var highTarget = new Vector<double>(new[] { 5.0, 5.0 });

        double lowLoss = loss.Calculate(lowTarget, noiseLogits);
        double highLoss = loss.Calculate(highTarget, noiseLogits);

        Assert.True(highLoss <= lowLoss + 1e-10,
            $"Higher target logits should reduce loss: low={lowLoss}, high={highLoss}.");
    }

    // =========================================================================
    // INVARIANT 4: Gradients are finite
    // =========================================================================

    [Fact]
    public void CalculateDerivative_ShouldBeFinite()
    {
        var loss = CreateLoss();
        int batchSize = 3;
        var targetLogits = new Vector<double>(batchSize);
        var noiseLogits = new Matrix<double>(batchSize, NumNoiseSamples);

        for (int i = 0; i < batchSize; i++)
        {
            targetLogits[i] = 1.0;
            for (int j = 0; j < NumNoiseSamples; j++)
                noiseLogits[i, j] = -0.5;
        }

        var (targetGrad, noiseGrad) = loss.CalculateDerivative(targetLogits, noiseLogits);

        for (int i = 0; i < batchSize; i++)
        {
            Assert.False(double.IsNaN(targetGrad[i]), $"Target gradient[{i}] is NaN.");
            Assert.False(double.IsInfinity(targetGrad[i]), $"Target gradient[{i}] is Infinity.");
            for (int j = 0; j < NumNoiseSamples; j++)
            {
                Assert.False(double.IsNaN(noiseGrad[i, j]), $"Noise gradient[{i},{j}] is NaN.");
                Assert.False(double.IsInfinity(noiseGrad[i, j]), $"Noise gradient[{i},{j}] is Infinity.");
            }
        }
    }

    // =========================================================================
    // INVARIANT 5: Dimension validation
    // =========================================================================

    [Fact]
    public void Calculate_MismatchedDimensions_ShouldThrow()
    {
        var loss = CreateLoss();
        var targetLogits = new Vector<double>(3);
        var noiseLogits = new Matrix<double>(2, NumNoiseSamples); // wrong rows

        Assert.Throws<ArgumentException>(() => loss.Calculate(targetLogits, noiseLogits));
    }
}
