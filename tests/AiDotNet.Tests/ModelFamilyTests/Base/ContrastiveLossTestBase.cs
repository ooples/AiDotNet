using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for noise contrastive estimation loss functions that operate on
/// target logits (Vector) and noise logits (Matrix).
/// Tests mathematical invariants: non-negativity, finiteness, and gradient correctness.
/// </summary>
public abstract class ContrastiveLossTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected static T ToT(double value) => NumOps.FromDouble(value);
    protected static double ToD(T value) => Convert.ToDouble(value);
    protected virtual double Tolerance => typeof(T) == typeof(float) ? 1e-6 : 1e-10;

    protected abstract NoiseContrastiveEstimationLoss<T> CreateLoss();

    /// <summary>The number of noise samples configured for this loss instance.</summary>
    protected virtual int NumNoiseSamples => 10;

    // =========================================================================
    // INVARIANT 1: Loss is finite for normal inputs
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Calculate_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        int batchSize = 3;
        var targetLogits = new Vector<T>(batchSize);
        var noiseLogits = new Matrix<T>(batchSize, NumNoiseSamples);

        // Fill with reasonable values
        for (int i = 0; i < batchSize; i++)
        {
            targetLogits[i] = ToT(1.0 + i * 0.5);
            for (int j = 0; j < NumNoiseSamples; j++)
            {
                noiseLogits[i, j] = ToT(-0.5 + j * 0.1);
            }
        }

        double value = ToD(loss.Calculate(targetLogits, noiseLogits));

        Assert.False(double.IsNaN(value), "Loss returned NaN.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is non-negative
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Calculate_ShouldBeNonNegative()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        int batchSize = 3;
        var targetLogits = new Vector<T>(batchSize);
        var noiseLogits = new Matrix<T>(batchSize, NumNoiseSamples);

        for (int i = 0; i < batchSize; i++)
        {
            targetLogits[i] = ToT(2.0);
            for (int j = 0; j < NumNoiseSamples; j++)
            {
                noiseLogits[i, j] = ToT(-1.0);
            }
        }

        double value = ToD(loss.Calculate(targetLogits, noiseLogits));
        Assert.True(value >= -Tolerance, $"NCE loss should be non-negative but got {value}.");
    }

    // =========================================================================
    // INVARIANT 3: Higher target logits should reduce loss
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Calculate_HigherTargetLogits_ShouldReduceLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        int batchSize = 2;
        var noiseLogits = new Matrix<T>(batchSize, NumNoiseSamples);
        for (int i = 0; i < batchSize; i++)
            for (int j = 0; j < NumNoiseSamples; j++)
                noiseLogits[i, j] = ToT(-1.0);

        var lowTarget = new Vector<T>(new[] { ToT(0.5), ToT(0.5) });
        var highTarget = new Vector<T>(new[] { ToT(5.0), ToT(5.0) });

        double lowLoss = ToD(loss.Calculate(lowTarget, noiseLogits));
        double highLoss = ToD(loss.Calculate(highTarget, noiseLogits));

        Assert.True(highLoss <= lowLoss + Tolerance,
            $"Higher target logits should reduce loss: low={lowLoss}, high={highLoss}.");
    }

    // =========================================================================
    // INVARIANT 4: Gradients are finite
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateDerivative_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        int batchSize = 3;
        var targetLogits = new Vector<T>(batchSize);
        var noiseLogits = new Matrix<T>(batchSize, NumNoiseSamples);

        for (int i = 0; i < batchSize; i++)
        {
            targetLogits[i] = ToT(1.0);
            for (int j = 0; j < NumNoiseSamples; j++)
                noiseLogits[i, j] = ToT(-0.5);
        }

        // NCE's tape forward already takes the target logits and the noise-logit matrix, so the
        // two gradients come straight off one backward pass.
        using var tape = new GradientTape<T>();

        var targetT = Tensor<T>.FromVector(targetLogits);
        var noiseT = Tensor<T>.FromMatrix(noiseLogits);

        var scalar = loss.ComputeTapeLoss(targetT, noiseT);
        var gradients = tape.ComputeGradients(scalar, new[] { targetT, noiseT });

        var targetGrad = gradients[targetT].ToVector();
        var noiseGrad = gradients[noiseT].ToMatrix();

        for (int i = 0; i < batchSize; i++)
        {
            Assert.False(double.IsNaN(ToD(targetGrad[i])), $"Target gradient[{i}] is NaN.");
            Assert.False(double.IsInfinity(ToD(targetGrad[i])), $"Target gradient[{i}] is Infinity.");
            for (int j = 0; j < NumNoiseSamples; j++)
            {
                Assert.False(double.IsNaN(ToD(noiseGrad[i, j])), $"Noise gradient[{i},{j}] is NaN.");
                Assert.False(double.IsInfinity(ToD(noiseGrad[i, j])), $"Noise gradient[{i},{j}] is Infinity.");
            }
        }
    }

    // =========================================================================
    // INVARIANT 5: Dimension validation
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Calculate_MismatchedDimensions_ShouldThrow()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var targetLogits = new Vector<T>(3);
        var noiseLogits = new Matrix<T>(2, NumNoiseSamples); // wrong rows

        Assert.Throws<ArgumentException>(() => loss.Calculate(targetLogits, noiseLogits));
    }
}

/// <summary>Default-precision alias for existing hand-written fixtures.</summary>
public abstract class ContrastiveLossTestBase : ContrastiveLossTestBase<double> { }
