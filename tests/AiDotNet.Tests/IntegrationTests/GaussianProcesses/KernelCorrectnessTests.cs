using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// Correctness tests for kernel implementations.
/// Tests mathematical properties that all valid kernels must satisfy.
/// </summary>
public class KernelCorrectnessTests
{
    private const double Tolerance = 1e-6;
    private const double RelativeTolerance = 1e-4;

    #region Cosine Kernel Tests

    [Fact]
    public void CosineKernel_SameVector_ReturnsOne()
    {
        var kernel = new CosineKernel<double>();
        var x = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var result = kernel.Calculate(x, x);

        Assert.True(Math.Abs(result - 1.0) < Tolerance,
            $"Cosine kernel of same vector should be 1.0, got {result}");
    }

    [Fact]
    public void CosineKernel_OrthogonalVectors_ReturnsZero()
    {
        var kernel = new CosineKernel<double>();
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { 0.0, 1.0 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(Math.Abs(result) < Tolerance,
            $"Cosine kernel of orthogonal vectors should be 0, got {result}");
    }

    [Fact]
    public void CosineKernel_OppositeVectors_ReturnsNegativeOne()
    {
        var kernel = new CosineKernel<double>();
        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(Math.Abs(result + 1.0) < Tolerance,
            $"Cosine kernel of opposite vectors should be -1.0, got {result}");
    }

    [Fact]
    public void CosineKernel_IsSymmetric()
    {
        var kernel = new CosineKernel<double>();
        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

        var k12 = kernel.Calculate(x1, x2);
        var k21 = kernel.Calculate(x2, x1);

        Assert.True(Math.Abs(k12 - k21) < Tolerance,
            $"Kernel should be symmetric: k(x1,x2)={k12}, k(x2,x1)={k21}");
    }

    [Fact]
    public void CosineKernel_WithOutputScale_ScalesCorrectly()
    {
        var scale = 2.5;
        var kernel = new CosineKernel<double>(outputScale: scale);
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { 1.0, 0.0 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(Math.Abs(result - scale) < Tolerance,
            $"Scaled cosine kernel of same vector should be {scale}, got {result}");
    }

    #endregion

    #region Arc Kernel Tests

    [Fact]
    public void ArcKernel_Order0_SameVector_ReturnsOne()
    {
        var kernel = new ArcKernel<double>(order: 0);
        var x = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var result = kernel.Calculate(x, x);

        Assert.True(Math.Abs(result - 1.0) < Tolerance,
            $"Arc kernel order 0 of same vector should be 1.0, got {result}");
    }

    [Fact]
    public void ArcKernel_Order0_OrthogonalVectors_ReturnsHalf()
    {
        var kernel = new ArcKernel<double>(order: 0);
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { 0.0, 1.0 });

        var result = kernel.Calculate(x1, x2);

        // For orthogonal vectors, theta = pi/2, so J_0 = 1 - (pi/2)/pi = 0.5
        Assert.True(Math.Abs(result - 0.5) < Tolerance,
            $"Arc kernel order 0 of orthogonal vectors should be 0.5, got {result}");
    }

    [Fact]
    public void ArcKernel_Order0_OppositeVectors_ReturnsZero()
    {
        var kernel = new ArcKernel<double>(order: 0);
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { -1.0, 0.0 });

        var result = kernel.Calculate(x1, x2);

        // For opposite vectors, theta = pi, so J_0 = 1 - pi/pi = 0
        Assert.True(Math.Abs(result) < Tolerance,
            $"Arc kernel order 0 of opposite vectors should be 0, got {result}");
    }

    [Fact]
    public void ArcKernel_IsSymmetric()
    {
        var kernel = new ArcKernel<double>(order: 1);
        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

        var k12 = kernel.Calculate(x1, x2);
        var k21 = kernel.Calculate(x2, x1);

        Assert.True(Math.Abs(k12 - k21) < Tolerance,
            $"Kernel should be symmetric: k(x1,x2)={k12}, k(x2,x1)={k21}");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void ArcKernel_AllOrders_ReturnValidValues(int order)
    {
        var kernel = new ArcKernel<double>(order: order);
        var x1 = new Vector<double>(new double[] { 1.0, 2.0 });
        var x2 = new Vector<double>(new double[] { 3.0, 4.0 });

        var result = kernel.Calculate(x1, x2);

        Assert.False(double.IsNaN(result), $"Arc kernel order {order} returned NaN");
        Assert.False(double.IsInfinity(result), $"Arc kernel order {order} returned infinity");
    }

    #endregion

    #region Cylindrical Kernel Tests

    [Fact]
    public void CylindricalKernel_SamePoint_ReturnsNonZero()
    {
        var rbf = new GaussianKernel<double>(1.0);
        var kernel = new CylindricalKernel<double>(
            rbf,
            regularDims: new[] { 0 },
            angularDims: new[] { 1 },
            periods: new[] { 2 * Math.PI });

        var x = new Vector<double>(new double[] { 1.0, 0.5 });
        var result = kernel.Calculate(x, x);

        Assert.True(result > 0.99,
            $"Cylindrical kernel of same point should be close to 1, got {result}");
    }

    [Fact]
    public void CylindricalKernel_AngularDim_WrapsAround()
    {
        var rbf = new GaussianKernel<double>(1.0);
        var period = 24.0; // 24 hours
        var kernel = new CylindricalKernel<double>(
            rbf,
            regularDims: Array.Empty<int>(),
            angularDims: new[] { 0 },
            periods: new[] { period },
            angularLengthScales: new[] { 1.0 });

        // Points at 23:00 and 01:00 are only 2 hours apart, not 22
        var x1 = new Vector<double>(new double[] { 23.0 });
        var x2 = new Vector<double>(new double[] { 1.0 });

        // Compare with point 2 hours from x1
        var x3 = new Vector<double>(new double[] { 21.0 });

        var k12 = kernel.Calculate(x1, x2);
        var k13 = kernel.Calculate(x1, x3);

        // k12 should be similar to k13 (both 2 hours apart)
        var ratio = k12 / k13;
        Assert.True(Math.Abs(ratio - 1.0) < 0.1,
            $"Angular wrapping not working: k(23,1)={k12}, k(23,21)={k13}, ratio={ratio}");
    }

    [Fact]
    public void CylindricalKernel_WithRBF_FactoryWorks()
    {
        var kernel = CylindricalKernel<double>.WithRBF(
            totalDims: 3,
            angularDimIndices: new[] { 2 },
            regularLengthScale: 1.0,
            periods: new[] { 2 * Math.PI });

        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { 1.1, 2.1, 3.1 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(result > 0 && result <= 1,
            $"Cylindrical kernel should return value in (0, 1], got {result}");
    }

    #endregion

    #region Spectral Delta Kernel Tests

    [Fact]
    public void SpectralDeltaKernel_SamePoint_ReturnsVariance()
    {
        var variance = 2.5;
        var kernel = new SpectralDeltaKernel<double>(
            frequency: 1.0,
            bandwidth: 0.1,
            variance: variance);

        var x = new Vector<double>(new double[] { 0.0 });
        var result = kernel.Calculate(x, x);

        // At tau=0, cos(0)=1 and envelope=1, so result = variance
        Assert.True(Math.Abs(result - variance) < Tolerance,
            $"Spectral delta kernel at same point should return variance {variance}, got {result}");
    }

    [Fact]
    public void SpectralDeltaKernel_HasCorrectPeriod()
    {
        var period = 7.0; // weekly
        var kernel = SpectralDeltaKernel<double>.FromPeriod(period: period);

        Assert.True(Math.Abs(kernel.Period - period) < Tolerance,
            $"Kernel period should be {period}, got {kernel.Period}");
    }

    [Fact]
    public void SpectralDeltaKernel_OscillatesWithFrequency()
    {
        var frequency = 1.0;
        var kernel = new SpectralDeltaKernel<double>(
            frequency: frequency,
            bandwidth: 0.01, // Small bandwidth so pattern persists
            variance: 1.0);

        var x1 = new Vector<double>(new double[] { 0.0 });
        var x2 = new Vector<double>(new double[] { 0.5 }); // Half period
        var x3 = new Vector<double>(new double[] { 1.0 }); // Full period

        var k01 = kernel.Calculate(x1, x1); // tau=0, should be positive
        var k02 = kernel.Calculate(x1, x2); // tau=0.5, cos(pi)=-1, should be negative
        var k03 = kernel.Calculate(x1, x3); // tau=1, cos(2pi)=1, should be positive

        Assert.True(k01 > 0, $"k(0,0) should be positive, got {k01}");
        Assert.True(k02 < 0, $"k(0,0.5) should be negative, got {k02}");
        Assert.True(k03 > 0, $"k(0,1) should be positive, got {k03}");
    }

    [Fact]
    public void SpectralDeltaKernel_PSD_IsCenteredAtFrequency()
    {
        var frequency = 5.0;
        var bandwidth = 0.5;
        var kernel = new SpectralDeltaKernel<double>(
            frequency: frequency,
            bandwidth: bandwidth,
            variance: 1.0);

        // PSD should peak at the frequency
        var psdAtFreq = kernel.GetPowerSpectralDensity(frequency);
        var psdOffFreq = kernel.GetPowerSpectralDensity(frequency + 2 * bandwidth);

        Assert.True(psdAtFreq > psdOffFreq,
            $"PSD should peak at frequency: PSD({frequency})={psdAtFreq}, PSD({frequency + 2 * bandwidth})={psdOffFreq}");
    }

    #endregion

    #region Grid Kernel Tests

    [Fact]
    public void GridKernel_SamePoint_ReturnsPositive()
    {
        var gridX = Enumerable.Range(0, 5).Select(i => (double)i).ToArray();
        var gridY = Enumerable.Range(0, 5).Select(i => (double)i).ToArray();

        var kernel = GridKernel<double>.WithRBF(
            new[] { gridX, gridY },
            lengthscales: new[] { 1.0, 1.0 });

        var x = new Vector<double>(new double[] { 2.0, 3.0 });
        var result = kernel.Calculate(x, x);

        Assert.True(result > 0,
            $"Grid kernel of same point should be positive, got {result}");
    }

    [Fact]
    public void GridKernel_Precompute_ComputesEigenvalues()
    {
        var gridX = Enumerable.Range(0, 4).Select(i => (double)i).ToArray();
        var gridY = Enumerable.Range(0, 3).Select(i => (double)i).ToArray();

        var kernel = GridKernel<double>.WithRBF(
            new[] { gridX, gridY },
            lengthscales: new[] { 1.0, 1.0 });

        kernel.Precompute();

        var eigenvalues = kernel.GetEigenvalues();

        Assert.Equal(4 * 3, eigenvalues.Length);
        Assert.True(eigenvalues.All(e => e >= 0),
            "All eigenvalues should be non-negative for PSD kernel");
    }

    [Fact]
    public void GridKernel_LogDeterminant_IsFinite()
    {
        var gridX = Enumerable.Range(0, 4).Select(i => (double)i).ToArray();
        var gridY = Enumerable.Range(0, 3).Select(i => (double)i).ToArray();

        var kernel = GridKernel<double>.WithRBF(
            new[] { gridX, gridY },
            lengthscales: new[] { 1.0, 1.0 });

        kernel.Precompute();

        var logDet = kernel.LogDeterminant();

        Assert.False(double.IsNaN(logDet), "Log-determinant should not be NaN");
        Assert.False(double.IsInfinity(logDet), "Log-determinant should be finite");
    }

    [Fact]
    public void GridKernel_KroneckerMultiply_GivesCorrectResult()
    {
        // Small grid for exact verification
        var gridX = new[] { 0.0, 1.0 };
        var gridY = new[] { 0.0, 1.0 };

        var kernel = GridKernel<double>.WithRBF(
            new[] { gridX, gridY },
            lengthscales: new[] { 1.0, 1.0 });

        kernel.Precompute();

        // Test with identity-like vector
        var v = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 0.0 });
        var result = kernel.KroneckerMultiply(v);

        // First element should be positive (kernel of origin with itself)
        Assert.True(result[0] > 0, "K[0,0] should be positive");
    }

    #endregion

    #region Grid Interpolation Kernel Tests

    [Fact]
    public void GridInterpolationKernel_SamePoint_ReturnsPositive()
    {
        var bounds = new[] { (0.0, 10.0), (0.0, 10.0) };
        var kernel = GridInterpolationKernel<double>.WithUniformGrid(
            new GaussianKernel<double>(1.0),
            bounds,
            gridPointsPerDim: 10);

        var x = new Vector<double>(new double[] { 5.0, 5.0 });
        var result = kernel.Calculate(x, x);

        Assert.True(result > 0,
            $"Grid interpolation kernel of same point should be positive, got {result}");
    }

    [Fact]
    public void GridInterpolationKernel_ClosePoints_HighSimilarity()
    {
        var bounds = new[] { (0.0, 10.0), (0.0, 10.0) };
        var kernel = GridInterpolationKernel<double>.WithUniformGrid(
            new GaussianKernel<double>(1.0),
            bounds,
            gridPointsPerDim: 20);

        var x1 = new Vector<double>(new double[] { 5.0, 5.0 });
        var x2 = new Vector<double>(new double[] { 5.1, 5.1 });
        var x3 = new Vector<double>(new double[] { 8.0, 8.0 });

        var kClose = kernel.Calculate(x1, x2);
        var kFar = kernel.Calculate(x1, x3);

        Assert.True(kClose > kFar,
            $"Close points should have higher similarity: k(close)={kClose}, k(far)={kFar}");
    }

    [Fact]
    public void GridInterpolationKernel_InterpolationWeights_SumToOne()
    {
        var bounds = new[] { (0.0, 10.0) };
        var kernel = GridInterpolationKernel<double>.WithUniformGrid(
            new GaussianKernel<double>(1.0),
            bounds,
            gridPointsPerDim: 10,
            interpolationOrder: 4);

        var X = new Matrix<double>(1, 1);
        X[0, 0] = 5.5; // Not on grid

        var (indices, weights) = kernel.ComputeInterpolationMatrix(X);

        var weightSum = weights[0].Sum();
        Assert.True(Math.Abs(weightSum - 1.0) < Tolerance,
            $"Interpolation weights should sum to 1, got {weightSum}");
    }

    #endregion

    #region Product Structure Kernel Tests

    [Fact]
    public void ProductStructureKernel_SamePoint_ReturnsPositive()
    {
        var kernel = ProductStructureKernel<double>.WithRBF(
            new[] { new[] { 0 }, new[] { 1 } },
            lengthscales: new[] { 1.0, 1.0 });

        var x = new Vector<double>(new double[] { 1.0, 2.0 });
        var result = kernel.Calculate(x, x);

        Assert.True(result > 0,
            $"Product kernel of same point should be positive, got {result}");
    }

    [Fact]
    public void ProductStructureKernel_IsProductOfGroupKernels()
    {
        // Create individual RBF kernels with same lengthscale
        var kernel1 = new GaussianKernel<double>(1.0);
        var kernel2 = new GaussianKernel<double>(2.0);

        // Product kernel
        var productKernel = new ProductStructureKernel<double>(
            new IKernelFunction<double>[] { kernel1, kernel2 },
            new[] { new[] { 0 }, new[] { 1 } });

        var x1 = new Vector<double>(new double[] { 1.0, 2.0 });
        var x2 = new Vector<double>(new double[] { 1.5, 3.0 });

        // Individual kernel values
        var v1a = new Vector<double>(new double[] { 1.0 });
        var v1b = new Vector<double>(new double[] { 1.5 });
        var v2a = new Vector<double>(new double[] { 2.0 });
        var v2b = new Vector<double>(new double[] { 3.0 });

        var k1 = kernel1.Calculate(v1a, v1b);
        var k2 = kernel2.Calculate(v2a, v2b);
        var expected = k1 * k2;

        var actual = productKernel.Calculate(x1, x2);

        Assert.True(Math.Abs(actual - expected) < Tolerance,
            $"Product kernel should be product of group kernels: expected {expected}, got {actual}");
    }

    [Fact]
    public void ProductStructureKernel_FullyFactorized_Works()
    {
        var kernel = ProductStructureKernel<double>.FullyFactorized(3, baseLengthscale: 1.0);

        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { 1.1, 2.1, 3.1 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(result > 0 && result < 1,
            $"Fully factorized kernel should return value in (0, 1), got {result}");
    }

    #endregion

    #region General Kernel Properties

    [Theory]
    [InlineData(typeof(CosineKernel<>))]
    [InlineData(typeof(SpectralDeltaKernel<>))]
    public void Kernel_IsSymmetric(Type kernelType)
    {
        IKernelFunction<double> kernel;
        if (kernelType == typeof(CosineKernel<>))
            kernel = new CosineKernel<double>();
        else if (kernelType == typeof(SpectralDeltaKernel<>))
            kernel = new SpectralDeltaKernel<double>(frequency: 1.0, bandwidth: 0.1);
        else
            throw new ArgumentException($"Unknown kernel type: {kernelType}");

        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

        var k12 = kernel.Calculate(x1, x2);
        var k21 = kernel.Calculate(x2, x1);

        Assert.True(Math.Abs(k12 - k21) < Tolerance,
            $"{kernelType.Name} should be symmetric: k(x1,x2)={k12}, k(x2,x1)={k21}");
    }

    #endregion
}
