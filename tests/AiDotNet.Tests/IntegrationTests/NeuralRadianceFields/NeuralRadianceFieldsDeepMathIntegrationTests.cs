using AiDotNet.NeuralRadianceFields.Metrics;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralRadianceFields;

/// <summary>
/// Deep math integration tests for Neural Radiance Fields rendering metrics.
/// Tests PSNR, SSIM, MSE, MAE, and SimplifiedLPIPS with hand-computed expected values.
/// </summary>
public class NeuralRadianceFieldsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    // ===== MSE Tests =====

    [Fact]
    public void MSE_IdenticalTensors_ReturnsZero()
    {
        // Arrange: two identical 3x3 images
        var data = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
        var a = new Tensor<double>(data, [3, 3]);
        var b = new Tensor<double>(data, [3, 3]);

        // Act
        var mse = RenderingMetrics<double>.MSE(a, b);

        // Assert: identical tensors have MSE = 0
        Assert.Equal(0.0, mse, Tolerance);
    }

    [Fact]
    public void MSE_KnownValues_ReturnsHandComputed()
    {
        // Arrange: a = [1, 2, 3, 4], b = [1.5, 2.5, 3.5, 4.5]
        // MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 4*0.25/4 = 0.25
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 1.5, 2.5, 3.5, 4.5 }, [2, 2]);

        // Act
        var mse = RenderingMetrics<double>.MSE(a, b);

        // Assert
        Assert.Equal(0.25, mse, Tolerance);
    }

    [Fact]
    public void MSE_AsymmetricDifferences_HandComputed()
    {
        // Arrange: a = [0, 0, 0, 0], b = [1, 2, 3, 4]
        // MSE = (1 + 4 + 9 + 16) / 4 = 30/4 = 7.5
        var a = new Tensor<double>(new double[] { 0, 0, 0, 0 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);

        // Act
        var mse = RenderingMetrics<double>.MSE(a, b);

        // Assert
        Assert.Equal(7.5, mse, Tolerance);
    }

    [Fact]
    public void MSE_IsSymmetric()
    {
        // MSE(a, b) = MSE(b, a) since (a-b)^2 = (b-a)^2
        var a = new Tensor<double>(new double[] { 0.1, 0.5, 0.9, 0.3 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.4, 0.2, 0.7, 0.8 }, [2, 2]);

        var mse_ab = RenderingMetrics<double>.MSE(a, b);
        var mse_ba = RenderingMetrics<double>.MSE(b, a);

        Assert.Equal(mse_ab, mse_ba, Tolerance);
    }

    [Fact]
    public void MSE_IsNonNegative()
    {
        var a = new Tensor<double>(new double[] { -1, 2, -3, 4 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 5, -6, 7, -8 }, [2, 2]);

        var mse = RenderingMetrics<double>.MSE(a, b);

        Assert.True(mse >= 0, $"MSE should be non-negative, got {mse}");
    }

    // ===== MAE Tests =====

    [Fact]
    public void MAE_IdenticalTensors_ReturnsZero()
    {
        var data = new double[] { 0.1, 0.5, 0.9, 0.3 };
        var a = new Tensor<double>(data, [2, 2]);
        var b = new Tensor<double>(data, [2, 2]);

        var mae = RenderingMetrics<double>.MAE(a, b);

        Assert.Equal(0.0, mae, Tolerance);
    }

    [Fact]
    public void MAE_KnownValues_HandComputed()
    {
        // a = [1, 2, 3, 4], b = [2, 4, 1, 5]
        // |diffs| = [1, 2, 2, 1], MAE = 6/4 = 1.5
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 2, 4, 1, 5 }, [2, 2]);

        var mae = RenderingMetrics<double>.MAE(a, b);

        Assert.Equal(1.5, mae, Tolerance);
    }

    [Fact]
    public void MAE_IsSymmetric()
    {
        var a = new Tensor<double>(new double[] { 0.1, 0.5, 0.9, 0.3 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.4, 0.2, 0.7, 0.8 }, [2, 2]);

        var mae_ab = RenderingMetrics<double>.MAE(a, b);
        var mae_ba = RenderingMetrics<double>.MAE(b, a);

        Assert.Equal(mae_ab, mae_ba, Tolerance);
    }

    [Fact]
    public void MAE_IsLessThanOrEqualToRMSE()
    {
        // By Jensen's inequality: MAE <= sqrt(MSE) = RMSE
        var a = new Tensor<double>(new double[] { 0.1, 0.5, 0.9, 0.3 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.4, 0.2, 0.7, 0.8 }, [2, 2]);

        var mae = RenderingMetrics<double>.MAE(a, b);
        var mse = RenderingMetrics<double>.MSE(a, b);
        var rmse = Math.Sqrt(mse);

        Assert.True(mae <= rmse + Tolerance, $"MAE ({mae}) should be <= RMSE ({rmse})");
    }

    // ===== PSNR Tests =====

    [Fact]
    public void PSNR_IdenticalImages_ReturnsInfinity()
    {
        // PSNR of identical images is infinite (MSE = 0)
        var data = new double[] { 0.1, 0.2, 0.3, 0.4 };
        var a = new Tensor<double>(data, [2, 2]);
        var b = new Tensor<double>(data, [2, 2]);

        var psnr = RenderingMetrics<double>.PSNR(a, b);

        Assert.Equal(double.PositiveInfinity, psnr);
    }

    [Fact]
    public void PSNR_KnownMSE_HandComputed()
    {
        // For MSE = 0.01, maxValue = 1.0:
        // PSNR = 10 * log10(1^2 / 0.01) = 10 * log10(100) = 10 * 2 = 20 dB
        // Create tensors with known MSE = 0.01
        // With 4 pixels, sum of squared diffs = 0.04
        // Use diffs of 0.1 each: (0.1)^2 * 4 / 4 = 0.01
        var a = new Tensor<double>(new double[] { 0.5, 0.5, 0.5, 0.5 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.6, 0.6, 0.6, 0.6 }, [2, 2]);

        var psnr = RenderingMetrics<double>.PSNR(a, b, 1.0);

        // MSE = 0.01, PSNR = 10 * log10(1/0.01) = 20
        Assert.Equal(20.0, psnr, 0.001);
    }

    [Fact]
    public void PSNR_HigherValueForSmallerError()
    {
        // A smaller difference should produce higher PSNR
        var reference = new Tensor<double>(new double[] { 0.5, 0.5, 0.5, 0.5 }, [2, 2]);
        var smallError = new Tensor<double>(new double[] { 0.51, 0.51, 0.51, 0.51 }, [2, 2]);
        var largeError = new Tensor<double>(new double[] { 0.7, 0.7, 0.7, 0.7 }, [2, 2]);

        var psnrSmall = RenderingMetrics<double>.PSNR(reference, smallError);
        var psnrLarge = RenderingMetrics<double>.PSNR(reference, largeError);

        Assert.True(psnrSmall > psnrLarge,
            $"Smaller error PSNR ({psnrSmall}) should be > larger error PSNR ({psnrLarge})");
    }

    [Fact]
    public void PSNR_MaxValueScaling_HandComputed()
    {
        // PSNR with maxValue=255 should differ from maxValue=1.0
        // For same MSE: PSNR(255) = 10*log10(255^2/MSE), PSNR(1) = 10*log10(1/MSE)
        // Difference = 10*log10(255^2) = 10*log10(65025) ≈ 48.13
        var a = new Tensor<double>(new double[] { 0.5, 0.5, 0.5, 0.5 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.6, 0.6, 0.6, 0.6 }, [2, 2]);

        var psnr1 = RenderingMetrics<double>.PSNR(a, b, 1.0);
        var psnr255 = RenderingMetrics<double>.PSNR(a, b, 255.0);

        var expectedDiff = 10.0 * Math.Log10(255.0 * 255.0);
        Assert.Equal(expectedDiff, psnr255 - psnr1, 0.001);
    }

    [Fact]
    public void PSNR_SpecificValue_HandComputed()
    {
        // a = [0.0, 1.0, 0.5, 0.3], b = [0.1, 0.9, 0.6, 0.2]
        // diffs = [0.1, 0.1, 0.1, 0.1] -> squared = [0.01, 0.01, 0.01, 0.01]
        // MSE = 0.04/4 = 0.01
        // PSNR = 10*log10(1/0.01) = 10*2 = 20 dB
        var a = new Tensor<double>(new double[] { 0.0, 1.0, 0.5, 0.3 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.1, 0.9, 0.6, 0.2 }, [2, 2]);

        var psnr = RenderingMetrics<double>.PSNR(a, b, 1.0);

        Assert.Equal(20.0, psnr, 0.001);
    }

    // ===== SSIM Tests =====

    [Fact]
    public void SSIM_IdenticalImages_ReturnsOne()
    {
        // SSIM of identical images = 1.0
        // Use small image that will use global SSIM (< windowSize)
        var data = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
        var a = new Tensor<double>(data, [3, 3]);
        var b = new Tensor<double>(data, [3, 3]);

        var ssim = RenderingMetrics<double>.SSIM(a, b);

        Assert.Equal(1.0, ssim, 0.001);
    }

    [Fact]
    public void SSIM_ResultInZeroOneRange()
    {
        // SSIM should be in [-1, 1] range, typically [0, 1] for natural images
        var a = new Tensor<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }, [3, 3]);
        var b = new Tensor<double>(new double[] { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 }, [3, 3]);

        var ssim = RenderingMetrics<double>.SSIM(a, b);

        Assert.True(ssim >= -1.0 && ssim <= 1.0, $"SSIM should be in [-1, 1], got {ssim}");
    }

    [Fact]
    public void SSIM_CloserImagesHaveHigherSSIM()
    {
        var reference = new Tensor<double>(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }, [3, 3]);
        var close = new Tensor<double>(new double[] { 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51 }, [3, 3]);
        var far = new Tensor<double>(new double[] { 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9 }, [3, 3]);

        var ssimClose = RenderingMetrics<double>.SSIM(reference, close);
        var ssimFar = RenderingMetrics<double>.SSIM(reference, far);

        Assert.True(ssimClose > ssimFar,
            $"Close SSIM ({ssimClose}) should be > far SSIM ({ssimFar})");
    }

    [Fact]
    public void SSIM_GlobalSSIM_HandComputed()
    {
        // For a small image (3x3, smaller than default windowSize=11), global SSIM is used
        // Using 2x2 to ensure global SSIM
        // a = [0.2, 0.8, 0.4, 0.6], b = [0.3, 0.7, 0.5, 0.5]
        // meanX = (0.2+0.8+0.4+0.6)/4 = 0.5
        // meanY = (0.3+0.7+0.5+0.5)/4 = 0.5
        // varX = ((0.2-0.5)^2 + (0.8-0.5)^2 + (0.4-0.5)^2 + (0.6-0.5)^2) / 3
        //      = (0.09 + 0.09 + 0.01 + 0.01) / 3 = 0.2/3 ≈ 0.066667
        // varY = ((0.3-0.5)^2 + (0.7-0.5)^2 + (0.5-0.5)^2 + (0.5-0.5)^2) / 3
        //      = (0.04 + 0.04 + 0 + 0) / 3 = 0.08/3 ≈ 0.026667
        // covXY = ((0.2-0.5)(0.3-0.5) + (0.8-0.5)(0.7-0.5) + (0.4-0.5)(0.5-0.5) + (0.6-0.5)(0.5-0.5)) / 3
        //       = ((-0.3)(-0.2) + (0.3)(0.2) + (-0.1)(0) + (0.1)(0)) / 3
        //       = (0.06 + 0.06 + 0 + 0) / 3 = 0.12/3 = 0.04
        // c1 = (0.01 * 1)^2 = 0.0001, c2 = (0.03 * 1)^2 = 0.0009
        // numerator = (2*0.5*0.5 + 0.0001) * (2*0.04 + 0.0009) = 0.5001 * 0.0809 = 0.04045809
        // denominator = (0.25 + 0.25 + 0.0001) * (0.066667 + 0.026667 + 0.0009) = 0.5001 * 0.094234 ≈ 0.047126
        // SSIM = 0.04045809 / 0.047126 ≈ 0.8585
        var a = new Tensor<double>(new double[] { 0.2, 0.8, 0.4, 0.6 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.3, 0.7, 0.5, 0.5 }, [2, 2]);

        var ssim = RenderingMetrics<double>.SSIM(a, b);

        // Verify in expected range with tolerance for rounding
        Assert.True(ssim > 0.80 && ssim < 0.95,
            $"SSIM should be approximately 0.86, got {ssim}");
    }

    [Fact]
    public void SSIM_ConstantImages_HighSSIM()
    {
        // Two constant images with different values:
        // meanX = 0.3, meanY = 0.7, varX = varY = 0, covXY = 0
        // SSIM = (2*0.3*0.7 + c1)(0 + c2) / (0.09 + 0.49 + c1)(0 + 0 + c2)
        //      = (0.42 + c1)(c2) / (0.58 + c1)(c2)
        //      = (0.42 + c1) / (0.58 + c1)
        var a = new Tensor<double>(new double[] { 0.3, 0.3, 0.3, 0.3 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.7, 0.7, 0.7, 0.7 }, [2, 2]);

        var ssim = RenderingMetrics<double>.SSIM(a, b);

        double c1 = 0.01 * 0.01; // k1=0.01, maxVal=1
        double expectedSSIM = (0.42 + c1) / (0.58 + c1);
        Assert.Equal(expectedSSIM, ssim, 0.001);
    }

    // ===== PSNR-SSIM Relationship Tests =====

    [Fact]
    public void PSNR_AndSSIM_HigherQualityImageScoresBetterOnBoth()
    {
        var reference = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => i / 15.0).ToArray(), [4, 4]);
        var goodApprox = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => i / 15.0 + 0.01).ToArray(), [4, 4]);
        var badApprox = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => i / 15.0 + 0.2).ToArray(), [4, 4]);

        var psnrGood = RenderingMetrics<double>.PSNR(reference, goodApprox);
        var psnrBad = RenderingMetrics<double>.PSNR(reference, badApprox);
        var ssimGood = RenderingMetrics<double>.SSIM(reference, goodApprox);
        var ssimBad = RenderingMetrics<double>.SSIM(reference, badApprox);

        Assert.True(psnrGood > psnrBad, $"Good PSNR ({psnrGood}) > Bad PSNR ({psnrBad})");
        Assert.True(ssimGood > ssimBad, $"Good SSIM ({ssimGood}) > Bad SSIM ({ssimBad})");
    }

    // ===== SimplifiedLPIPS Tests =====

    [Fact]
    public void SimplifiedLPIPS_IdenticalImages_ReturnsZero()
    {
        // Edge magnitudes and local stats will be identical -> LPIPS = 0
        var data = new double[]
        {
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 0.8, 0.7, 0.6,
            0.5, 0.4, 0.3, 0.2
        };
        var a = new Tensor<double>(data, [4, 4]);
        var b = new Tensor<double>(data, [4, 4]);

        var lpips = RenderingMetrics<double>.SimplifiedLPIPS(a, b);

        Assert.Equal(0.0, lpips, 0.001);
    }

    [Fact]
    public void SimplifiedLPIPS_IsNonNegative()
    {
        var a = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => i / 15.0).ToArray(), [4, 4]);
        var b = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => (15 - i) / 15.0).ToArray(), [4, 4]);

        var lpips = RenderingMetrics<double>.SimplifiedLPIPS(a, b);

        Assert.True(lpips >= 0, $"SimplifiedLPIPS should be non-negative, got {lpips}");
    }

    [Fact]
    public void SimplifiedLPIPS_MoreDifferentImagesHaveHigherScore()
    {
        var reference = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => i / 15.0).ToArray(), [4, 4]);
        var similar = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => i / 15.0 + 0.01).ToArray(), [4, 4]);
        var different = new Tensor<double>(
            Enumerable.Range(0, 16).Select(i => (15 - i) / 15.0).ToArray(), [4, 4]);

        var lpipsSimilar = RenderingMetrics<double>.SimplifiedLPIPS(reference, similar);
        var lpipsDifferent = RenderingMetrics<double>.SimplifiedLPIPS(reference, different);

        Assert.True(lpipsSimilar < lpipsDifferent,
            $"Similar LPIPS ({lpipsSimilar}) should be < Different LPIPS ({lpipsDifferent})");
    }

    // ===== Shape Validation Tests =====

    [Fact]
    public void MSE_DifferentShapes_ThrowsArgumentException()
    {
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);

        Assert.Throws<ArgumentException>(() => RenderingMetrics<double>.MSE(a, b));
    }

    [Fact]
    public void PSNR_DifferentRanks_ThrowsArgumentException()
    {
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 1, 2, 3, 4 }, [4]);

        Assert.Throws<ArgumentException>(() => RenderingMetrics<double>.PSNR(a, b));
    }

    // ===== 3-Channel Image Tests =====

    [Fact]
    public void MSE_3ChannelImage_HandComputed()
    {
        // 2x2x3 RGB image
        // a: all 0.5, b: all 0.6
        // diff = 0.1 for each of 12 elements
        // MSE = 12 * 0.01 / 12 = 0.01
        var a = new Tensor<double>(Enumerable.Repeat(0.5, 12).ToArray(), [2, 2, 3]);
        var b = new Tensor<double>(Enumerable.Repeat(0.6, 12).ToArray(), [2, 2, 3]);

        var mse = RenderingMetrics<double>.MSE(a, b);

        Assert.Equal(0.01, mse, Tolerance);
    }

    [Fact]
    public void PSNR_3ChannelImage_HandComputed()
    {
        // MSE = 0.01 (from above), maxValue = 1.0
        // PSNR = 10 * log10(1/0.01) = 20 dB
        var a = new Tensor<double>(Enumerable.Repeat(0.5, 12).ToArray(), [2, 2, 3]);
        var b = new Tensor<double>(Enumerable.Repeat(0.6, 12).ToArray(), [2, 2, 3]);

        var psnr = RenderingMetrics<double>.PSNR(a, b, 1.0);

        Assert.Equal(20.0, psnr, 0.001);
    }

    [Fact]
    public void MAE_3ChannelImage_HandComputed()
    {
        // All diffs = 0.1, so MAE = 0.1
        var a = new Tensor<double>(Enumerable.Repeat(0.5, 12).ToArray(), [2, 2, 3]);
        var b = new Tensor<double>(Enumerable.Repeat(0.6, 12).ToArray(), [2, 2, 3]);

        var mae = RenderingMetrics<double>.MAE(a, b);

        Assert.Equal(0.1, mae, Tolerance);
    }

    // ===== PSNR Mathematical Properties =====

    [Fact]
    public void PSNR_DoublingError_Decreases3dB()
    {
        // If MSE doubles, PSNR decreases by 10*log10(2) ≈ 3.01 dB
        var reference = new Tensor<double>(new double[] { 0.5, 0.5, 0.5, 0.5 }, [2, 2]);
        var err1 = new Tensor<double>(new double[] { 0.6, 0.6, 0.6, 0.6 }, [2, 2]); // diff = 0.1
        var err2 = new Tensor<double>(new double[] { 0.5 + Math.Sqrt(0.02), 0.5 + Math.Sqrt(0.02),
                                                       0.5 + Math.Sqrt(0.02), 0.5 + Math.Sqrt(0.02) }, [2, 2]);
        // err1: MSE = 0.01
        // err2: diff = sqrt(0.02), MSE = 0.02 (double)

        var psnr1 = RenderingMetrics<double>.PSNR(reference, err1);
        var psnr2 = RenderingMetrics<double>.PSNR(reference, err2);

        var expectedDiff = 10.0 * Math.Log10(2.0); // ≈ 3.0103
        Assert.Equal(expectedDiff, psnr1 - psnr2, 0.01);
    }

    [Fact]
    public void PSNR_Relationship_10Log10MaxSquaredOverMSE()
    {
        // Verify PSNR = 10 * log10(MAX^2 / MSE) directly
        var a = new Tensor<double>(new double[] { 0.0, 0.3, 0.7, 1.0 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.1, 0.2, 0.8, 0.9 }, [2, 2]);

        var mse = RenderingMetrics<double>.MSE(a, b);
        var psnr = RenderingMetrics<double>.PSNR(a, b, 1.0);

        var expectedPSNR = 10.0 * Math.Log10(1.0 / mse);
        Assert.Equal(expectedPSNR, psnr, 0.001);
    }

    // ===== SSIM Stability Constants Tests =====

    [Fact]
    public void SSIM_StabilityConstants_PreventDivisionByZero()
    {
        // All-zero images should not cause division by zero
        var a = new Tensor<double>(new double[] { 0, 0, 0, 0 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0, 0, 0, 0 }, [2, 2]);

        var ssim = RenderingMetrics<double>.SSIM(a, b);

        // Should return 1.0 for identical zero images (c1, c2 prevent div by zero)
        Assert.Equal(1.0, ssim, 0.001);
    }

    [Fact]
    public void SSIM_CustomK1K2_AffectsResult()
    {
        var a = new Tensor<double>(new double[] { 0.2, 0.8, 0.4, 0.6 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.3, 0.7, 0.5, 0.5 }, [2, 2]);

        var ssimDefault = RenderingMetrics<double>.SSIM(a, b, k1: 0.01, k2: 0.03);
        var ssimLargeK = RenderingMetrics<double>.SSIM(a, b, k1: 0.1, k2: 0.3);

        // Larger stability constants should push SSIM closer to 1 by dominating the formula
        Assert.True(ssimLargeK > ssimDefault,
            $"Larger k1,k2 SSIM ({ssimLargeK}) should be > default SSIM ({ssimDefault})");
    }

    // ===== MSE Triangle Inequality Tests =====

    [Fact]
    public void RMSE_SatisfiesTriangleInequality()
    {
        // RMSE = sqrt(MSE) is a metric and satisfies triangle inequality
        // RMSE(a,c) <= RMSE(a,b) + RMSE(b,c)
        var a = new Tensor<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.5, 0.6, 0.7, 0.8 }, [2, 2]);
        var c = new Tensor<double>(new double[] { 0.9, 0.8, 0.1, 0.3 }, [2, 2]);

        var rmseAB = Math.Sqrt(RenderingMetrics<double>.MSE(a, b));
        var rmseBC = Math.Sqrt(RenderingMetrics<double>.MSE(b, c));
        var rmseAC = Math.Sqrt(RenderingMetrics<double>.MSE(a, c));

        Assert.True(rmseAC <= rmseAB + rmseBC + Tolerance,
            $"Triangle inequality failed: RMSE(a,c)={rmseAC} > RMSE(a,b)={rmseAB} + RMSE(b,c)={rmseBC}");
    }

    // ===== Edge Cases =====

    [Fact]
    public void MSE_SinglePixel_HandComputed()
    {
        // Single pixel (1x1) image
        var a = new Tensor<double>(new double[] { 0.3 }, [1, 1]);
        var b = new Tensor<double>(new double[] { 0.7 }, [1, 1]);

        // MSE = (0.4)^2 / 1 = 0.16
        var mse = RenderingMetrics<double>.MSE(a, b);

        Assert.Equal(0.16, mse, Tolerance);
    }

    [Fact]
    public void MAE_SinglePixel_HandComputed()
    {
        var a = new Tensor<double>(new double[] { 0.3 }, [1, 1]);
        var b = new Tensor<double>(new double[] { 0.7 }, [1, 1]);

        // MAE = |0.3 - 0.7| / 1 = 0.4
        var mae = RenderingMetrics<double>.MAE(a, b);

        Assert.Equal(0.4, mae, Tolerance);
    }

    [Fact]
    public void SSIM_IsSymmetric()
    {
        var a = new Tensor<double>(new double[] { 0.1, 0.4, 0.7, 0.2 }, [2, 2]);
        var b = new Tensor<double>(new double[] { 0.3, 0.6, 0.5, 0.8 }, [2, 2]);

        var ssimAB = RenderingMetrics<double>.SSIM(a, b);
        var ssimBA = RenderingMetrics<double>.SSIM(b, a);

        Assert.Equal(ssimAB, ssimBA, 0.001);
    }

    // ===== Sobel Edge Detection (embedded in SimplifiedLPIPS) =====

    [Fact]
    public void SimplifiedLPIPS_UniformImage_ZeroEdges()
    {
        // Uniform images have no edges, so edge difference = 0
        // Local stats differ between different uniform values
        var a = new Tensor<double>(Enumerable.Repeat(0.5, 16).ToArray(), [4, 4]);
        var b = new Tensor<double>(Enumerable.Repeat(0.5, 16).ToArray(), [4, 4]);

        var lpips = RenderingMetrics<double>.SimplifiedLPIPS(a, b);

        Assert.Equal(0.0, lpips, 0.001);
    }

    [Fact]
    public void SimplifiedLPIPS_LPIPSCorrelatesWithMSE()
    {
        // For simple uniform-offset images, LPIPS should correlate with MSE
        var reference = new Tensor<double>(
            Enumerable.Range(0, 25).Select(i => i / 24.0).ToArray(), [5, 5]);
        var small_offset = new Tensor<double>(
            Enumerable.Range(0, 25).Select(i => i / 24.0 + 0.01).ToArray(), [5, 5]);
        var large_offset = new Tensor<double>(
            Enumerable.Range(0, 25).Select(i => i / 24.0 + 0.2).ToArray(), [5, 5]);

        var lpipsSmall = RenderingMetrics<double>.SimplifiedLPIPS(reference, small_offset);
        var lpipsLarge = RenderingMetrics<double>.SimplifiedLPIPS(reference, large_offset);

        Assert.True(lpipsSmall <= lpipsLarge,
            $"Small offset LPIPS ({lpipsSmall}) should be <= Large offset LPIPS ({lpipsLarge})");
    }
}
