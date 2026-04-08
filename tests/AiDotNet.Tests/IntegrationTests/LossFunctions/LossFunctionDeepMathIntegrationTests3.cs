using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Deep mathematical tests for loss functions not yet covered.
/// Tests hand-calculated values, derivative correctness, and mathematical invariants.
/// </summary>
public class LossFunctionDeepMathIntegrationTests3
{
    private const double Tol = 1e-6;
    private const double GradTol = 1e-4;

    // ====================================================================
    // ScaleInvariantDepthLoss
    // L = (1/n) * Σ(d²) - (λ/n²) * (Σd)²  where d = log(pred) - log(actual)
    // ====================================================================

    [Fact]
    public void ScaleInvariantDepth_PerfectPrediction_ReturnsZero()
    {
        var loss = new ScaleInvariantDepthLoss<double>(0.5);
        var pred = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = loss.CalculateLoss(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void ScaleInvariantDepth_HandCalculated()
    {
        var loss = new ScaleInvariantDepthLoss<double>(0.5);
        var pred = new Vector<double>(new[] { 2.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 });

        // d = [log(2)-log(1), log(4)-log(2)] = [ln2, ln2]
        // d² = [ln²(2), ln²(2)]
        // Σd² = 2*ln²(2)
        // (Σd)² = (2*ln2)² = 4*ln²(2)
        // L = (1/2)*2*ln²(2) - (0.5/4)*4*ln²(2) = ln²(2) - 0.5*ln²(2) = 0.5*ln²(2)
        double ln2 = Math.Log(2);
        double expected = 0.5 * ln2 * ln2;
        var result = loss.CalculateLoss(pred, actual);
        Assert.Equal(expected, result, 1e-4);
    }

    [Fact]
    public void ScaleInvariantDepth_ScaleInvariant_WithLambda1()
    {
        // With lambda=1.0, uniform scaling should not affect the loss
        var loss = new ScaleInvariantDepthLoss<double>(1.0);
        var pred = new Vector<double>(new[] { 2.0, 4.0, 6.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        // Scaling prediction by constant: d = log(k*pred) - log(actual) = log(k) + d_orig
        var scaledPred = new Vector<double>(new[] { 20.0, 40.0, 60.0 }); // 10x scaling
        var loss1 = loss.CalculateLoss(pred, actual);
        var loss2 = loss.CalculateLoss(scaledPred, actual);
        // With lambda=1.0, these should be equal (fully scale-invariant)
        Assert.Equal(loss1, loss2, 1e-4);
    }

    [Fact]
    public void ScaleInvariantDepth_NonNegative()
    {
        var loss = new ScaleInvariantDepthLoss<double>(0.5);
        var pred = new Vector<double>(new[] { 1.0, 3.0, 0.5 });
        var actual = new Vector<double>(new[] { 2.0, 1.0, 4.0 });
        var result = loss.CalculateLoss(pred, actual);
        Assert.True(result >= -Tol, $"ScaleInvariantDepth loss should be >= 0, got {result}");
    }

    [Fact]
    public void ScaleInvariantDepth_NumericalGradient()
    {
        var loss = new ScaleInvariantDepthLoss<double>(0.5);
        var pred = new Vector<double>(new[] { 1.5, 2.5, 3.5 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var analytical = loss.CalculateDerivative(pred, actual);

        double h = 1e-5;
        for (int i = 0; i < pred.Length; i++)
        {
            var predPlus = new Vector<double>(new[] { pred[0], pred[1], pred[2] });
            var predMinus = new Vector<double>(new[] { pred[0], pred[1], pred[2] });
            predPlus[i] += h;
            predMinus[i] -= h;
            double numerical = (loss.CalculateLoss(predPlus, actual) -
                               loss.CalculateLoss(predMinus, actual)) / (2 * h);
            Assert.True(Math.Abs(analytical[i] - numerical) < GradTol,
                $"Gradient mismatch at [{i}]: analytical={analytical[i]}, numerical={numerical}");
        }
    }

    // NCE Loss requires Calculate(Vector<T>, Matrix<T>) - tested in specialized tests

    // PerceptualLoss requires a feature extractor - tested in specialized tests

    // ====================================================================
    // QuantumLoss
    // ====================================================================

    [Fact]
    public void Quantum_PerfectPrediction_ReturnsZero()
    {
        var loss = new QuantumLoss<double>();
        // Quantum state: pairs of (real, imag) for each complex amplitude
        var pred = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 }); // |1+0i, 0+0i>
        var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 });
        var result = loss.CalculateLoss(pred, actual);
        Assert.Equal(0.0, result, 1e-4);
    }

    [Fact]
    public void Quantum_NonNegative()
    {
        var loss = new QuantumLoss<double>();
        // Even-length vectors for complex number pairs
        var pred = new Vector<double>(new[] { 0.5, 0.3, 0.2, 0.1 });
        var actual = new Vector<double>(new[] { 0.1, 0.6, 0.3, 0.4 });
        var result = loss.CalculateLoss(pred, actual);
        Assert.True(result >= -Tol, $"Quantum loss should be non-negative, got {result}");
    }

    // ====================================================================
    // RealESRGANLoss: combination of L1 + perceptual + adversarial
    // ====================================================================

    [Fact]
    public void RealESRGAN_PerfectPrediction_ReturnsZero()
    {
        var loss = new RealESRGANLoss<double>();
        var pred = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = loss.CalculateLoss(pred, actual);
        Assert.Equal(0.0, result, 1e-4);
    }

    [Fact]
    public void RealESRGAN_NonNegative()
    {
        var loss = new RealESRGANLoss<double>();
        var pred = new Vector<double>(new[] { 0.5, 1.5, 2.5 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = loss.CalculateLoss(pred, actual);
        Assert.True(result >= -Tol, $"RealESRGAN loss should be non-negative, got {result}");
    }

    // CTCLoss requires specialized sequence API (Tensor + int[][] targets) - tested separately
}
