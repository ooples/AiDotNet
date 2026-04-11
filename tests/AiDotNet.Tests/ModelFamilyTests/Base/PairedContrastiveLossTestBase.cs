using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for contrastive loss functions that operate on paired embeddings
/// with a similarity label: CalculateLoss(Vector, Vector, T).
/// Tests mathematical invariants: non-negativity, margin behavior, gradient correctness.
/// </summary>
public abstract class PairedContrastiveLossTestBase
{
    protected abstract ContrastiveLoss<double> CreateLoss();

    // =========================================================================
    // INVARIANT 1: Loss is finite for similar pairs
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_SimilarPair_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var v1 = new Vector<double>(new[] { 0.5, 1.0, -0.3 });
        var v2 = new Vector<double>(new[] { 0.6, 0.9, -0.2 });

        double value = loss.CalculateLoss(v1, v2, 1.0); // similar

        Assert.False(double.IsNaN(value), "Loss returned NaN for similar pair.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity for similar pair.");
    }

    // =========================================================================
    // INVARIANT 2: Loss is finite for dissimilar pairs
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_DissimilarPair_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var v1 = new Vector<double>(new[] { 0.5, 1.0, -0.3 });
        var v2 = new Vector<double>(new[] { -1.0, 0.0, 2.0 });

        double value = loss.CalculateLoss(v1, v2, 0.0); // dissimilar

        Assert.False(double.IsNaN(value), "Loss returned NaN for dissimilar pair.");
        Assert.False(double.IsInfinity(value), "Loss returned Infinity for dissimilar pair.");
    }

    // =========================================================================
    // INVARIANT 3: Loss is non-negative
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_ShouldBeNonNegative()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var v1 = new Vector<double>(new[] { 0.5, 1.0, -0.3 });
        var v2 = new Vector<double>(new[] { 0.6, 0.9, -0.2 });

        Assert.True(loss.CalculateLoss(v1, v2, 1.0) >= -1e-10, "Similar loss should be non-negative.");
        Assert.True(loss.CalculateLoss(v1, v2, 0.0) >= -1e-10, "Dissimilar loss should be non-negative.");
    }

    // =========================================================================
    // INVARIANT 4: Identical vectors → zero loss for similar pairs
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_IdenticalVectors_SimilarLabel_ShouldBeZero()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var v = new Vector<double>(new[] { 0.5, 1.0, -0.3 });

        double value = loss.CalculateLoss(v, v, 1.0);
        Assert.True(Math.Abs(value) < 1e-10,
            $"Similar pair with identical vectors should have zero loss, got {value}.");
    }

    // =========================================================================
    // INVARIANT 5: Closer similar pairs → lower loss
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_CloserSimilarPair_ShouldProduceLowerLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var anchor = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var close = new Vector<double>(new[] { 0.1, 0.1, 0.1 });
        var far = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

        double closeLoss = loss.CalculateLoss(anchor, close, 1.0);
        double farLoss = loss.CalculateLoss(anchor, far, 1.0);

        Assert.True(closeLoss < farLoss + 1e-10,
            $"Closer similar pair should have lower loss: close={closeLoss}, far={farLoss}.");
    }

    // =========================================================================
    // INVARIANT 6: Well-separated dissimilar pairs → zero loss
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateLoss_WellSeparatedDissimilarPair_ShouldBeZero()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var v1 = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 10.0, 10.0, 10.0 }); // far apart

        double value = loss.CalculateLoss(v1, v2, 0.0);
        Assert.True(value < 1e-10,
            $"Well-separated dissimilar pair should have zero loss, got {value}.");
    }

    // =========================================================================
    // INVARIANT 7: Gradients are finite
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task CalculateDerivative_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var loss = CreateLoss();
        var v1 = new Vector<double>(new[] { 0.5, 1.0, -0.3 });
        var v2 = new Vector<double>(new[] { 0.6, 0.9, -0.2 });

        var (grad1Similar, grad2Similar) = loss.CalculateDerivative(v1, v2, 1.0);
        var (grad1Dissimilar, grad2Dissimilar) = loss.CalculateDerivative(v1, v2, 0.0);

        for (int i = 0; i < v1.Length; i++)
        {
            Assert.False(double.IsNaN(grad1Similar[i]), $"Similar grad1[{i}] is NaN.");
            Assert.False(double.IsNaN(grad2Similar[i]), $"Similar grad2[{i}] is NaN.");
            Assert.False(double.IsNaN(grad1Dissimilar[i]), $"Dissimilar grad1[{i}] is NaN.");
            Assert.False(double.IsNaN(grad2Dissimilar[i]), $"Dissimilar grad2[{i}] is NaN.");
        }
    }
}
