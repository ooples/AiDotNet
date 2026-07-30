using System;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.KnowledgeDistillation;

/// <summary>
/// Contract tests for the FSP ("flow of solution procedure") loss in
/// <see cref="FlowBasedDistillationStrategy{T}"/>, from Yim et al., CVPR 2017.
/// </summary>
/// <remarks>
/// <para>
/// The strategy previously computed a single scalar inner product per layer pair, self-described as a
/// "simplified flow computation". The paper's FSP is a MATRIX: entry (i, j) records how strongly
/// channel i of the first layer co-activates with channel j of the second, so the matrix describes the
/// DIRECTION of the transformation between layers. Collapsing it to one number keeps only overall
/// magnitude, which two entirely different transformations can share — so the distillation signal was
/// largely absent while the loss still looked plausible.
/// </para>
/// <para>
/// These tests pin the matrix form numerically against hand-computed values, so the loss cannot
/// silently regress to a scalar again, and they cover the vectorized implementation: the FSP sum over
/// spatial positions is expressed as one matrix product <c>F1 * F2^T / positions</c> rather than nested
/// loops over channels and pixels, which for a 256-channel pair at 32x32 is the difference between one
/// blocked GEMM and ~67 million virtually-dispatched scalar operations.
/// </para>
/// </remarks>
public class FspDistillationTests
{
    private readonly ITestOutputHelper _out;

    public FspDistillationTests(ITestOutputHelper output) => _out = output;

    private static Tensor<double> Map(int channels, int height, int width, Func<int, int, int, double> f)
    {
        var t = new Tensor<double>([channels, height, width]);
        for (int c = 0; c < channels; c++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                    t[c * height * width + h * width + w] = f(c, h, w);
        return t;
    }

    private static bool Fin(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    /// <summary>
    /// Identical student and teacher have identical flow, so the FSP loss must be exactly zero. A
    /// non-zero value would mean the two networks' matrices are being formed inconsistently.
    /// </summary>
    [Fact]
    public void IdenticalFeatures_GiveZeroLoss()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();

        var l1 = Map(2, 2, 2, (c, h, w) => c + h + w + 1);
        var l2 = Map(3, 2, 2, (c, h, w) => (c + 1) * (h + w + 1));

        double loss = strategy.ComputeFlowLoss(
            new[] { l1, l2 },
            new[] { l1, l2 });

        _out.WriteLine($"identical-feature FSP loss = {loss}");
        Assert.True(Fin(loss));
        Assert.Equal(0.0, loss, 10);
    }

    /// <summary>
    /// Two networks whose flow differs must produce a strictly positive loss, otherwise the term is
    /// inert as a training signal.
    /// </summary>
    [Fact]
    public void DifferingFlow_GivesPositiveLoss()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();

        var s1 = Map(2, 2, 2, (c, h, w) => c + h + w + 1);
        var s2 = Map(2, 2, 2, (c, h, w) => (c + 1) * (h + w + 1));
        var t1 = Map(2, 2, 2, (c, h, w) => c + h + w + 1);
        var t2 = Map(2, 2, 2, (c, h, w) => -(c + 1) * (h + w + 2));   // different transformation

        double loss = strategy.ComputeFlowLoss(new[] { s1, s2 }, new[] { t1, t2 });

        _out.WriteLine($"differing-flow FSP loss = {loss}");
        Assert.True(Fin(loss));
        Assert.True(loss > 0.0);
    }

    /// <summary>
    /// THE test that a scalar cannot pass. Two layer pairs are constructed to have the SAME summed
    /// magnitude but different per-channel structure. A scalar inner product cannot tell them apart;
    /// the FSP matrix can, so the loss must be non-zero.
    /// </summary>
    [Fact]
    public void SameTotalMagnitudeButDifferentStructure_IsStillPenalized()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();

        // One spatial position, two channels each, so the FSP matrix is a 2x2 outer product.
        // Student: a = [1, 0], b = [0, 1]  ->  G = [[0,1],[0,0]]
        // Teacher: a = [0, 1], b = [1, 0]  ->  G = [[0,0],[1,0]]
        // Both have the same total (1) and the same scalar inner product a.b = 0, but different matrices.
        var sA = Map(2, 1, 1, (c, _, _) => c == 0 ? 1.0 : 0.0);
        var sB = Map(2, 1, 1, (c, _, _) => c == 1 ? 1.0 : 0.0);
        var tA = Map(2, 1, 1, (c, _, _) => c == 1 ? 1.0 : 0.0);
        var tB = Map(2, 1, 1, (c, _, _) => c == 0 ? 1.0 : 0.0);

        double loss = strategy.ComputeFlowLoss(new[] { sA, sB }, new[] { tA, tB });

        _out.WriteLine($"structure-differing FSP loss = {loss} (a scalar inner product gives 0 for both)");
        Assert.True(loss > 0.0,
            "The loss is zero for two pairs whose scalar inner products match but whose FSP matrices " +
            "differ — the matrix structure is being collapsed away.");
    }

    /// <summary>
    /// Pins the FSP matrix numerically: for single-position maps it is the outer product, so with
    /// a = [1, 2] and b = [3, 4, 5] the entries are exactly a[i]*b[j]. Verified through the loss by
    /// comparing against a teacher whose second layer is zero, which makes the squared Frobenius norm
    /// the sum of squares of those known entries.
    /// </summary>
    [Fact]
    public void FspMatrix_MatchesHandComputedOuterProduct()
    {
        // flowWeight = 1 so the reported loss is the raw squared Frobenius norm.
        var strategy = new FlowBasedDistillationStrategy<double>(flowWeight: 1.0);

        var a = Map(2, 1, 1, (c, _, _) => c == 0 ? 1.0 : 2.0);
        var b = Map(3, 1, 1, (c, _, _) => 3.0 + c);            // [3, 4, 5]
        var zero = Map(3, 1, 1, (_, _, _) => 0.0);

        double loss = strategy.ComputeFlowLoss(new[] { a, b }, new[] { a, zero });

        // G_student = [[3,4,5],[6,8,10]]; G_teacher = 0. Sum of squares:
        // 9+16+25+36+64+100 = 250. One layer pair, so no averaging effect.
        _out.WriteLine($"loss = {loss}, expected 250");
        Assert.Equal(250.0, loss, 8);
    }

    /// <summary>
    /// The spatial sum is normalized by the number of positions, so replicating the same values across
    /// a larger grid must NOT change the FSP matrix. Without that normalization the loss would scale
    /// with resolution and the per-pair weighting would mean something different at every layer.
    /// </summary>
    [Fact]
    public void SpatialNormalization_MakesLossResolutionIndependent()
    {
        var strategy = new FlowBasedDistillationStrategy<double>(flowWeight: 1.0);

        double LossAt(int size)
        {
            var a = Map(2, size, size, (c, _, _) => c == 0 ? 1.0 : 2.0);
            var b = Map(2, size, size, (c, _, _) => 3.0 + c);
            var zero = Map(2, size, size, (_, _, _) => 0.0);
            return strategy.ComputeFlowLoss(new[] { a, b }, new[] { a, zero });
        }

        double small = LossAt(1);
        double large = LossAt(4);

        _out.WriteLine($"loss 1x1 = {small}; loss 4x4 = {large}");
        Assert.Equal(small, large, 8);
    }

    /// <summary>
    /// A pair whose two layers differ spatially is aligned before the matrix is formed, so it must not
    /// throw and must stay finite — the paper aligns such pairs rather than rejecting them.
    /// </summary>
    [Fact]
    public void MismatchedSpatialSizesWithinAPair_AreAligned()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();

        var coarse = Map(2, 2, 2, (c, h, w) => c + h + w + 1.0);
        var fine = Map(3, 4, 4, (c, h, w) => (c + 1) * (h + w + 1.0));

        double loss = strategy.ComputeFlowLoss(
            new[] { coarse, fine },
            new[] { coarse, fine });

        _out.WriteLine($"aligned-pair loss (identical inputs) = {loss}");
        Assert.True(Fin(loss));
        Assert.Equal(0.0, loss, 10);
    }

    /// <summary>
    /// Teacher and student FSP matrices must be the same size to be compared, which requires matching
    /// channel counts at the distilled layers. A mismatch is a configuration error and must fail
    /// loudly rather than produce a meaningless number.
    /// </summary>
    [Fact]
    public void MismatchedChannelCounts_Throw()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();

        var s1 = Map(2, 2, 2, (c, h, w) => 1.0);
        var s2 = Map(2, 2, 2, (c, h, w) => 1.0);
        var t1 = Map(2, 2, 2, (c, h, w) => 1.0);
        var t2 = Map(5, 2, 2, (c, h, w) => 1.0);   // 5 channels vs the student's 2

        Assert.Throws<ArgumentException>(
            () => strategy.ComputeFlowLoss(new[] { s1, s2 }, new[] { t1, t2 }));
    }

    /// <summary>An FSP matrix describes flow BETWEEN layers, so a single layer is undefined.</summary>
    [Fact]
    public void FewerThanTwoLayers_Throws()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();
        var only = Map(2, 2, 2, (c, h, w) => 1.0);

        Assert.Throws<ArgumentException>(
            () => strategy.ComputeFlowLoss(new[] { only }, new[] { only }));
    }

    /// <summary>
    /// The legacy flat-vector overload is the degenerate single-position case and must agree with the
    /// tensor path on equivalent input, so existing callers keep working and get the matrix form.
    /// </summary>
    [Fact]
    public void VectorOverload_AgreesWithTensorPath()
    {
        var strategy = new FlowBasedDistillationStrategy<double>(flowWeight: 1.0);

        var sVec = new[]
        {
            new Vector<double>(new[] { 1.0, 2.0 }),
            new Vector<double>(new[] { 3.0, 4.0 })
        };
        var tVec = new[]
        {
            new Vector<double>(new[] { 1.0, 2.0 }),
            new Vector<double>(new[] { 0.0, 0.0 })
        };

        double vectorLoss = strategy.ComputeFlowLoss(sVec, tVec);

        var sTen = new[]
        {
            Map(2, 1, 1, (c, _, _) => c == 0 ? 1.0 : 2.0),
            Map(2, 1, 1, (c, _, _) => c == 0 ? 3.0 : 4.0)
        };
        var tTen = new[]
        {
            Map(2, 1, 1, (c, _, _) => c == 0 ? 1.0 : 2.0),
            Map(2, 1, 1, (_, _, _) => 0.0)
        };

        double tensorLoss = strategy.ComputeFlowLoss(sTen, tTen);

        _out.WriteLine($"vector={vectorLoss} tensor={tensorLoss}");
        Assert.Equal(tensorLoss, vectorLoss, 8);
    }
}
