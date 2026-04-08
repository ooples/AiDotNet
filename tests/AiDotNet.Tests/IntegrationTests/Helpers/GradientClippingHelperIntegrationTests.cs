using AiDotNet.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for GradientClippingHelper:
/// ClipByValue, ClipByNorm, ClipByGlobalNorm, ClipAdaptive,
/// ComputeNorm, ComputeGlobalNorm, AreGradientsExploding, AreGradientsVanishing.
/// </summary>
public class GradientClippingHelperIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Constants

    [Fact(Timeout = 120000)]
    public async Task DefaultMaxNorm_IsOne()
    {
        Assert.Equal(1.0, GradientClippingHelper.DefaultMaxNorm);
    }

    [Fact(Timeout = 120000)]
    public async Task DefaultMaxValue_IsOne()
    {
        Assert.Equal(1.0, GradientClippingHelper.DefaultMaxValue);
    }

    #endregion

    #region ClipByValue

    [Fact(Timeout = 120000)]
    public async Task ClipByValue_WithinRange_Unchanged()
    {
        var grads = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var clipped = GradientClippingHelper.ClipByValue(grads, 1.0);
        Assert.NotNull(clipped);
        Assert.Equal(0.5, clipped[0], Tolerance);
        Assert.Equal(-0.3, clipped[1], Tolerance);
        Assert.Equal(0.8, clipped[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByValue_ExceedsRange_ClipsToMax()
    {
        var grads = new Vector<double>(new double[] { 5.0, -3.0, 0.5 });
        var clipped = GradientClippingHelper.ClipByValue(grads, 1.0);
        Assert.NotNull(clipped);
        Assert.Equal(1.0, clipped[0], Tolerance);
        Assert.Equal(-1.0, clipped[1], Tolerance);
        Assert.Equal(0.5, clipped[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByValue_Null_ReturnsNull()
    {
        var result = GradientClippingHelper.ClipByValue<double>(null);
        Assert.Null(result);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByValue_CustomMaxValue()
    {
        var grads = new Vector<double>(new double[] { 10.0, -10.0 });
        var clipped = GradientClippingHelper.ClipByValue(grads, 0.5);
        Assert.NotNull(clipped);
        Assert.Equal(0.5, clipped[0], Tolerance);
        Assert.Equal(-0.5, clipped[1], Tolerance);
    }

    #endregion

    #region ClipByValueInPlace

    [Fact(Timeout = 120000)]
    public async Task ClipByValueInPlace_ModifiesOriginal()
    {
        var grads = new Vector<double>(new double[] { 5.0, -3.0 });
        GradientClippingHelper.ClipByValueInPlace(grads, 1.0);
        Assert.Equal(1.0, grads[0], Tolerance);
        Assert.Equal(-1.0, grads[1], Tolerance);
    }

    #endregion

    #region ClipByNorm

    [Fact(Timeout = 120000)]
    public async Task ClipByNorm_BelowThreshold_Unchanged()
    {
        var grads = new Vector<double>(new double[] { 0.3, 0.4 }); // norm = 0.5
        var clipped = GradientClippingHelper.ClipByNorm(grads, 1.0);
        Assert.NotNull(clipped);
        Assert.Equal(0.3, clipped[0], Tolerance);
        Assert.Equal(0.4, clipped[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByNorm_AboveThreshold_ScalesDown()
    {
        var grads = new Vector<double>(new double[] { 3.0, 4.0 }); // norm = 5.0
        var clipped = GradientClippingHelper.ClipByNorm(grads, 1.0);
        Assert.NotNull(clipped);

        // After clipping, norm should be 1.0
        double norm = Math.Sqrt(clipped[0] * clipped[0] + clipped[1] * clipped[1]);
        Assert.Equal(1.0, norm, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByNorm_PreservesDirection()
    {
        var grads = new Vector<double>(new double[] { 6.0, 8.0 }); // norm = 10
        var clipped = GradientClippingHelper.ClipByNorm(grads, 2.0);
        Assert.NotNull(clipped);

        // Direction should be preserved (ratio of components)
        double originalRatio = 6.0 / 8.0;
        double clippedRatio = clipped[0] / clipped[1];
        Assert.Equal(originalRatio, clippedRatio, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByNorm_Null_ReturnsNull()
    {
        Vector<double>? nullVector = null;
        var result = GradientClippingHelper.ClipByNorm(nullVector);
        Assert.Null(result);
    }

    #endregion

    #region ClipByNormInPlace

    [Fact(Timeout = 120000)]
    public async Task ClipByNormInPlace_AboveThreshold_ReturnsTrueAndClips()
    {
        var grads = new Vector<double>(new double[] { 3.0, 4.0 }); // norm = 5
        bool wasClipped = GradientClippingHelper.ClipByNormInPlace(grads, 1.0);
        Assert.True(wasClipped);

        double norm = Math.Sqrt(grads[0] * grads[0] + grads[1] * grads[1]);
        Assert.Equal(1.0, norm, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByNormInPlace_BelowThreshold_ReturnsFalse()
    {
        var grads = new Vector<double>(new double[] { 0.1, 0.2 }); // norm < 1
        bool wasClipped = GradientClippingHelper.ClipByNormInPlace(grads, 1.0);
        Assert.False(wasClipped);
    }

    #endregion

    #region ClipByGlobalNorm

    [Fact(Timeout = 120000)]
    public async Task ClipByGlobalNorm_BelowThreshold_Unchanged()
    {
        var grads1 = new Vector<double>(new double[] { 0.1, 0.2 });
        var grads2 = new Vector<double>(new double[] { 0.1, 0.1 });
        var list = new List<Vector<double>> { grads1, grads2 };

        var clipped = GradientClippingHelper.ClipByGlobalNorm(list, 1.0);
        Assert.NotNull(clipped);
        Assert.Equal(2, clipped.Count);
        Assert.Equal(0.1, clipped[0][0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByGlobalNorm_AboveThreshold_ScalesAll()
    {
        var grads1 = new Vector<double>(new double[] { 3.0, 4.0 }); // norm = 5
        var grads2 = new Vector<double>(new double[] { 6.0, 8.0 }); // norm = 10
        // global norm = sqrt(25 + 100) = sqrt(125) = ~11.18
        var list = new List<Vector<double>> { grads1, grads2 };

        var clipped = GradientClippingHelper.ClipByGlobalNorm(list, 1.0);
        Assert.NotNull(clipped);

        // Global norm of clipped should be 1.0
        double globalNormSq = 0;
        foreach (var g in clipped)
            for (int i = 0; i < g.Length; i++)
                globalNormSq += g[i] * g[i];
        Assert.Equal(1.0, Math.Sqrt(globalNormSq), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByGlobalNorm_Null_ReturnsNull()
    {
        var result = GradientClippingHelper.ClipByGlobalNorm<double>(null);
        Assert.Null(result);
    }

    #endregion

    #region ClipByNorm (Tensor)

    [Fact(Timeout = 120000)]
    public async Task ClipByNorm_Tensor_AboveThreshold_ScalesDown()
    {
        var grads = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 3.0, 4.0, 6.0, 8.0 }));
        // norm = sqrt(9+16+36+64) = sqrt(125) = ~11.18
        var clipped = GradientClippingHelper.ClipByNorm(grads, 1.0);
        Assert.NotNull(clipped);
        Assert.Equal(new[] { 2, 2 }, clipped.Shape.ToArray());

        // Check norm of clipped tensor is 1.0
        double normSq = 0;
        for (int i = 0; i < clipped.Length; i++)
            normSq += clipped[i] * clipped[i];
        Assert.Equal(1.0, Math.Sqrt(normSq), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipByNorm_Tensor_BelowThreshold_Unchanged()
    {
        var grads = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 0.3, 0.4 }));
        var clipped = GradientClippingHelper.ClipByNorm(grads, 5.0);
        Assert.NotNull(clipped);
        Assert.Equal(0.3, clipped[0], Tolerance);
        Assert.Equal(0.4, clipped[1], Tolerance);
    }

    #endregion

    #region ComputeNorm

    [Fact(Timeout = 120000)]
    public async Task ComputeNorm_KnownVector()
    {
        var v = new Vector<double>(new double[] { 3.0, 4.0 });
        double norm = GradientClippingHelper.ComputeNorm(v);
        Assert.Equal(5.0, norm, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ComputeNorm_ZeroVector()
    {
        var v = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
        double norm = GradientClippingHelper.ComputeNorm(v);
        Assert.Equal(0.0, norm, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ComputeNorm_UnitVector()
    {
        var v = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
        double norm = GradientClippingHelper.ComputeNorm(v);
        Assert.Equal(1.0, norm, Tolerance);
    }

    #endregion

    #region ComputeGlobalNorm

    [Fact(Timeout = 120000)]
    public async Task ComputeGlobalNorm_MultipleVectors()
    {
        var v1 = new Vector<double>(new double[] { 3.0, 0.0 });
        var v2 = new Vector<double>(new double[] { 0.0, 4.0 });
        var list = new List<Vector<double>> { v1, v2 };

        double globalNorm = GradientClippingHelper.ComputeGlobalNorm(list);
        Assert.Equal(5.0, globalNorm, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ComputeGlobalNorm_Empty_ReturnsZero()
    {
        var list = new List<Vector<double>>();
        double globalNorm = GradientClippingHelper.ComputeGlobalNorm(list);
        Assert.Equal(0.0, globalNorm, Tolerance);
    }

    #endregion

    #region ClipAdaptive

    [Fact(Timeout = 120000)]
    public async Task ClipAdaptive_SmallGradients_Unchanged()
    {
        var grads = new Vector<double>(new double[] { 0.001, 0.001 });
        var parameters = new Vector<double>(new double[] { 10.0, 10.0 });
        // param norm = ~14.14, clip at 0.01 * 14.14 = 0.1414, grad norm = ~0.0014
        var clipped = GradientClippingHelper.ClipAdaptive(grads, parameters, 0.01);
        Assert.NotNull(clipped);
        Assert.Equal(0.001, clipped[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClipAdaptive_LargeGradients_GetsClipped()
    {
        var grads = new Vector<double>(new double[] { 100.0, 100.0 }); // norm = ~141.4
        var parameters = new Vector<double>(new double[] { 1.0, 1.0 }); // param norm = ~1.414
        // adaptive max = 0.01 * 1.414 = 0.01414, but min threshold = 0.001
        var clipped = GradientClippingHelper.ClipAdaptive(grads, parameters, 0.01);
        Assert.NotNull(clipped);

        double clippedNorm = GradientClippingHelper.ComputeNorm(clipped);
        double originalNorm = GradientClippingHelper.ComputeNorm(grads);
        Assert.True(clippedNorm < originalNorm, "Clipped norm should be less than original");
    }

    [Fact(Timeout = 120000)]
    public async Task ClipAdaptive_MismatchedLengths_Throws()
    {
        var grads = new Vector<double>(new double[] { 1.0, 2.0 });
        var parameters = new Vector<double>(new double[] { 1.0 });
        Assert.Throws<ArgumentException>(() => GradientClippingHelper.ClipAdaptive(grads, parameters));
    }

    [Fact(Timeout = 120000)]
    public async Task ClipAdaptive_NullGradients_ReturnsNull()
    {
        var parameters = new Vector<double>(new double[] { 1.0 });
        var result = GradientClippingHelper.ClipAdaptive<double>(null, parameters);
        Assert.Null(result);
    }

    #endregion

    #region AreGradientsExploding

    [Fact(Timeout = 120000)]
    public async Task AreGradientsExploding_Normal_ReturnsFalse()
    {
        var grads = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        Assert.False(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact(Timeout = 120000)]
    public async Task AreGradientsExploding_VeryLarge_ReturnsTrue()
    {
        var grads = new Vector<double>(new double[] { 1e7, 1e7 });
        Assert.True(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact(Timeout = 120000)]
    public async Task AreGradientsExploding_ContainsNaN_ReturnsTrue()
    {
        var grads = new Vector<double>(new double[] { 1.0, double.NaN });
        Assert.True(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact(Timeout = 120000)]
    public async Task AreGradientsExploding_ContainsInfinity_ReturnsTrue()
    {
        var grads = new Vector<double>(new double[] { 1.0, double.PositiveInfinity });
        Assert.True(GradientClippingHelper.AreGradientsExploding(grads));
    }

    #endregion

    #region AreGradientsVanishing

    [Fact(Timeout = 120000)]
    public async Task AreGradientsVanishing_Normal_ReturnsFalse()
    {
        var grads = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        Assert.False(GradientClippingHelper.AreGradientsVanishing(grads));
    }

    [Fact(Timeout = 120000)]
    public async Task AreGradientsVanishing_VerySmall_ReturnsTrue()
    {
        var grads = new Vector<double>(new double[] { 1e-10, 1e-10 });
        Assert.True(GradientClippingHelper.AreGradientsVanishing(grads));
    }

    [Fact(Timeout = 120000)]
    public async Task AreGradientsVanishing_Zero_ReturnsTrue()
    {
        var grads = new Vector<double>(new double[] { 0.0, 0.0 });
        Assert.True(GradientClippingHelper.AreGradientsVanishing(grads));
    }

    [Fact(Timeout = 120000)]
    public async Task AreGradientsVanishing_Null_ReturnsTrue()
    {
        Assert.True(GradientClippingHelper.AreGradientsVanishing<double>(null));
    }

    #endregion
}
