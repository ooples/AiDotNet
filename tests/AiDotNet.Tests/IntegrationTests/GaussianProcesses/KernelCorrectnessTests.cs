using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task CosineKernel_SameVector_ReturnsOne()
    {
        var kernel = new CosineKernel<double>();
        var x = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var result = kernel.Calculate(x, x);

        Assert.True(Math.Abs(result - 1.0) < Tolerance,
            $"Cosine kernel of same vector should be 1.0, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task CosineKernel_OrthogonalVectors_ReturnsZero()
    {
        var kernel = new CosineKernel<double>();
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { 0.0, 1.0 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(Math.Abs(result) < Tolerance,
            $"Cosine kernel of orthogonal vectors should be 0, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task CosineKernel_OppositeVectors_ReturnsNegativeOne()
    {
        var kernel = new CosineKernel<double>();
        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });

        var result = kernel.Calculate(x1, x2);

        Assert.True(Math.Abs(result + 1.0) < Tolerance,
            $"Cosine kernel of opposite vectors should be -1.0, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task CosineKernel_IsSymmetric()
    {
        var kernel = new CosineKernel<double>();
        var x1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var x2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

        var k12 = kernel.Calculate(x1, x2);
        var k21 = kernel.Calculate(x2, x1);

        Assert.True(Math.Abs(k12 - k21) < Tolerance,
            $"Kernel should be symmetric: k(x1,x2)={k12}, k(x2,x1)={k21}");
    }

    [Fact(Timeout = 120000)]
    public async Task CosineKernel_WithOutputScale_ScalesCorrectly()
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

    [Fact(Timeout = 120000)]
    public async Task ArcKernel_Order0_SameVector_ReturnsOne()
    {
        var kernel = new ArcKernel<double>(order: 0);
        var x = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var result = kernel.Calculate(x, x);

        Assert.True(Math.Abs(result - 1.0) < Tolerance,
            $"Arc kernel order 0 of same vector should be 1.0, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task ArcKernel_Order0_OrthogonalVectors_ReturnsHalf()
    {
        var kernel = new ArcKernel<double>(order: 0);
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { 0.0, 1.0 });

        var result = kernel.Calculate(x1, x2);

        // For orthogonal vectors, theta = pi/2, so J_0 = 1 - (pi/2)/pi = 0.5
        Assert.True(Math.Abs(result - 0.5) < Tolerance,
            $"Arc kernel order 0 of orthogonal vectors should be 0.5, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task ArcKernel_Order0_OppositeVectors_ReturnsZero()
    {
        var kernel = new ArcKernel<double>(order: 0);
        var x1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var x2 = new Vector<double>(new double[] { -1.0, 0.0 });

        var result = kernel.Calculate(x1, x2);

        // For opposite vectors, theta = pi, so J_0 = 1 - pi/pi = 0
        Assert.True(Math.Abs(result) < Tolerance,
            $"Arc kernel order 0 of opposite vectors should be 0, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task ArcKernel_IsSymmetric()
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

    [Fact(Timeout = 120000)]
    public async Task CylindricalKernel_SamePoint_ReturnsNonZero()
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

    [Fact(Timeout = 120000)]
    public async Task CylindricalKernel_AngularDim_WrapsAround()
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

    [Fact(Timeout = 120000)]
    public async Task CylindricalKernel_WithRBF_FactoryWorks()
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

    [Fact(Timeout = 120000)]
    public async Task SpectralDeltaKernel_SamePoint_ReturnsVariance()
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

    [Fact(Timeout = 120000)]
    public async Task SpectralDeltaKernel_HasCorrectPeriod()
    {
        var period = 7.0; // weekly
        var kernel = SpectralDeltaKernel<double>.FromPeriod(period: period);

        Assert.True(Math.Abs(kernel.Period - period) < Tolerance,
            $"Kernel period should be {period}, got {kernel.Period}");
    }

    [Fact(Timeout = 120000)]
    public async Task SpectralDeltaKernel_OscillatesWithFrequency()
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

    [Fact(Timeout = 120000)]
    public async Task SpectralDeltaKernel_PSD_IsCenteredAtFrequency()
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

    [Fact(Timeout = 120000)]
    public async Task GridKernel_SamePoint_ReturnsPositive()
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

    [Fact(Timeout = 120000)]
    public async Task GridKernel_Precompute_ComputesEigenvalues()
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

    [Fact(Timeout = 120000)]
    public async Task GridKernel_LogDeterminant_IsFinite()
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

    [Fact(Timeout = 120000)]
    public async Task GridKernel_KroneckerMultiply_GivesCorrectResult()
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

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_SamePoint_ReturnsPositive()
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

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_ClosePoints_HighSimilarity()
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

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_InterpolationWeights_SumToOne()
    {
        var bounds = new[] { (0.0, 10.0) };
        var kernel = GridInterpolationKernel<double>.WithUniformGrid(
            new GaussianKernel<double>(1.0),
            bounds,
            gridPointsPerDim: 10);

        var X = new Matrix<double>(1, 1);
        X[0, 0] = 5.5; // Not on grid

        var (indices, weights) = kernel.ComputeInterpolationMatrix(X);

        var weightSum = weights.GetRow(0).Sum();
        Assert.True(Math.Abs(weightSum - 1.0) < Tolerance,
            $"Interpolation weights should sum to 1, got {weightSum}");
    }

    /// <summary>
    /// Reference values from GPyTorch 1.15.2 (<c>gpytorch.utils.interpolation.Interpolation().interpolate</c>,
    /// float64) on the grids linspace(0, 1, 6) x linspace(-1, 2, 7). Rows 0, 2, 4 and 5 sit on the grid
    /// boundary (one-hot weights), rows 1, 3 and 6 use the cubic convolution stencil of Keys (1981).
    /// </summary>
    private static readonly (double X0, double X1, (int Index, double Weight)[] Entries)[] GPyTorchInterpolation =
    {
        (0.05, -0.9, new[] { (0, 1.0) }),
        (0.5, 0.5, new[] { (10, -0.0625), (17, 0.5625), (24, 0.5625), (31, -0.0625) }),
        (0.93, 1.95, new[] { (41, 1.0) }),
        (0.37, 0.0, new[] { (2, -0.009562499999999474), (9, 0.11493750000000014), (16, 0.9488125), (23, -0.05418750000000028) }),
        (1.0, 2.0, new[] { (41, 1.0) }),
        (0.0, -1.0, new[] { (0, 1.0) }),
        (0.61, 1.13, new[]
        {
            (17, 0.0016061792500000143), (18, -0.01934427525000023), (19, -0.005388737250000059), (20, 0.0005643332500000023),
            (24, -0.07075642274999981), (25, 0.8521662307500003), (26, 0.23738805674999983), (27, -0.024860364749999815),
            (31, -0.00212229224999998), (32, 0.025560164249999833), (33, 0.007120298249999947), (34, -0.0007456702499999894),
            (38, 8.453575000002406e-05), (39, -0.0010181197500002926), (40, -0.00028361775000008125), (41, 2.970175000000831e-05),
        }),
    };

    private static GridInterpolationKernel<double> GPyTorchReferenceKernel()
        => new GridInterpolationKernel<double>(
            new GaussianKernel<double>(0.7),
            new[]
            {
                new Vector<double>(new[] { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 }),
                new Vector<double>(new[] { -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0 }),
            });

    private static Matrix<double> GPyTorchReferencePoints()
    {
        var X = new Matrix<double>(GPyTorchInterpolation.Length, 2);
        for (int i = 0; i < GPyTorchInterpolation.Length; i++)
        {
            X[i, 0] = GPyTorchInterpolation[i].X0;
            X[i, 1] = GPyTorchInterpolation[i].X1;
        }

        return X;
    }

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_InterpolationMatrix_MatchesGPyTorchCubicConvolution()
    {
        var kernel = GPyTorchReferenceKernel();
        var (indices, weights) = kernel.ComputeInterpolationMatrix(GPyTorchReferencePoints());

        for (int row = 0; row < GPyTorchInterpolation.Length; row++)
        {
            // Dense W row from the stencil entries (duplicate indices accumulate, as in a sparse matrix).
            var dense = new double[kernel.TotalGridPoints];
            for (int k = 0; k < weights.Columns; k++)
                dense[indices[row, k]] += weights[row, k];

            var expected = new double[kernel.TotalGridPoints];
            foreach (var (index, weight) in GPyTorchInterpolation[row].Entries)
                expected[index] = weight;

            for (int j = 0; j < dense.Length; j++)
                Assert.True(Math.Abs(dense[j] - expected[j]) < 1e-12,
                    $"W[{row}, {j}] = {dense[j]}, GPyTorch gives {expected[j]}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_KissGpProducts_MatchGPyTorchReference()
    {
        // K = W K_UU W^T with K_UU = kron(K_0, K_1) of RBF(sigma = 0.7) on the grids (numpy on the GPyTorch W).
        double[,] expectedK =
        {
            { 1.0, 7.7999274666594232e-02, 3.7018912300479529e-05, 3.1327902471306457e-01, 3.7018912300479529e-05, 1.0, 5.2752740854810519e-03 },
            { 7.7999274666594232e-02, 9.9909630802313021e-01, 7.7999274666594232e-02, 7.6133181427211216e-01, 7.7999274666594232e-02, 7.7999274666594232e-02, 6.6814876427500658e-01 },
            { 3.7018912300479529e-05, 7.7999274666594232e-02, 1.0, 1.1266746652444877e-02, 1.0, 3.7018912300479529e-05, 3.9541043587652430e-01 },
            { 3.1327902471306457e-01, 7.6133181427211216e-01, 1.1266746652444877e-02, 9.9976847841711636e-01, 1.1266746652444877e-02, 3.1327902471306457e-01, 2.6176300076880099e-01 },
            { 3.7018912300479529e-05, 7.7999274666594232e-02, 1.0, 1.1266746652444877e-02, 1.0, 3.7018912300479529e-05, 3.9541043587652430e-01 },
            { 1.0, 7.7999274666594232e-02, 3.7018912300479529e-05, 3.1327902471306457e-01, 3.7018912300479529e-05, 1.0, 5.2752740854810519e-03 },
            { 5.2752740854810519e-03, 6.6814876427500669e-01, 3.9541043587652430e-01, 2.6176300076880105e-01, 3.9541043587652430e-01, 5.2752740854810519e-03, 9.8390584670879833e-01 },
        };
        double[] expectedKv = { 1.9342888440212764, 0.6233840169184299, 0.32783879915132835, 1.5933501884939223, 0.32783879915132835, 1.9342888440212764, 0.09373246056742443 };

        var kernel = GPyTorchReferenceKernel();
        var X = GPyTorchReferencePoints();
        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < X.Rows; j++)
            {
                double k = kernel.Calculate(X.GetRow(i), X.GetRow(j));
                Assert.True(Math.Abs(k - expectedK[i, j]) < 1e-10, $"K[{i}, {j}] = {k}, expected {expectedK[i, j]}");
            }
        }

        kernel.Precompute();
        var (indices, weights) = kernel.ComputeInterpolationMatrix(X);
        var v = new Vector<double>(new[] { 0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.25 });
        var kv = kernel.KissGpMultiply(v, indices, weights);
        for (int i = 0; i < expectedKv.Length; i++)
            Assert.True(Math.Abs(kv[i] - expectedKv[i]) < 1e-10, $"(K v)[{i}] = {kv[i]}, expected {expectedKv[i]}");
    }

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_FastGridMultiply_MatchesKroneckerProduct()
    {
        // kron(K_0, K_1) u for u = linspace(-1, 1, 42), entries 0, 5, 17 and 41 (numpy).
        var kernel = GPyTorchReferenceKernel();
        kernel.Precompute();
        var u = new Vector<double>(Enumerable.Range(0, 42).Select(i => -1.0 + 2.0 * i / 41.0));
        var result = kernel.FastGridMultiply(u);

        Assert.True(Math.Abs(result[0] - (-2.8442687610594057)) < 1e-10, $"got {result[0]}");
        Assert.True(Math.Abs(result[5] - (-1.356640897601578)) < 1e-10, $"got {result[5]}");
        Assert.True(Math.Abs(result[17] - (-0.6851744225573786)) < 1e-10, $"got {result[17]}");
        Assert.True(Math.Abs(result[41] - 2.8442687610594057) < 1e-10, $"got {result[41]}");
    }

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_PointOutsideGrid_Throws()
    {
        var kernel = GPyTorchReferenceKernel();
        var outside = new Vector<double>(new[] { 1.2, 0.0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => kernel.Calculate(outside, outside));
    }

    [Fact(Timeout = 120000)]
    public async Task GridInterpolationKernel_NonUniformGrid_Throws()
    {
        Assert.Throws<ArgumentException>(() => new GridInterpolationKernel<double>(
            new GaussianKernel<double>(1.0),
            new[] { new Vector<double>(new[] { 0.0, 0.1, 0.3, 0.4, 0.5 }) }));
    }

    #endregion

    #region Product Structure Kernel Tests

    [Fact(Timeout = 120000)]
    public async Task ProductStructureKernel_SamePoint_ReturnsPositive()
    {
        var kernel = ProductStructureKernel<double>.WithRBF(
            new[] { new[] { 0 }, new[] { 1 } },
            lengthscales: new[] { 1.0, 1.0 });

        var x = new Vector<double>(new double[] { 1.0, 2.0 });
        var result = kernel.Calculate(x, x);

        Assert.True(result > 0,
            $"Product kernel of same point should be positive, got {result}");
    }

    [Fact(Timeout = 120000)]
    public async Task ProductStructureKernel_IsProductOfGroupKernels()
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

    [Fact(Timeout = 120000)]
    public async Task ProductStructureKernel_FullyFactorized_Works()
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
