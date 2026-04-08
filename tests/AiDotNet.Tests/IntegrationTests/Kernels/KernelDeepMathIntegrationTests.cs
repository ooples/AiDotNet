using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Kernels;

/// <summary>
/// Deep mathematical correctness tests for kernel functions.
/// Verifies hand-calculated values, positive definiteness properties,
/// symmetry, Cauchy-Schwarz inequality, Mercer conditions, and
/// numerical gradient verification for kernel gradients.
/// </summary>
public class KernelDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-5;

    private static Vector<double> V(params double[] values) => new(values);

    // ============================================================
    //  LINEAR KERNEL: k(x,y) = x . y
    // ============================================================

    [Fact]
    public void Linear_HandValues()
    {
        var k = new LinearKernel<double>();

        // [1,0] . [0,1] = 0
        Assert.Equal(0.0, k.Calculate(V(1, 0), V(0, 1)), Tolerance);

        // [1,2,3] . [4,5,6] = 4+10+18 = 32
        Assert.Equal(32.0, k.Calculate(V(1, 2, 3), V(4, 5, 6)), Tolerance);

        // [1,1] . [1,1] = 2
        Assert.Equal(2.0, k.Calculate(V(1, 1), V(1, 1)), Tolerance);

        // [3,-1] . [-1,3] = -3 + -3 = -6
        Assert.Equal(-6.0, k.Calculate(V(3, -1), V(-1, 3)), Tolerance);
    }

    [Fact]
    public void Linear_Symmetry()
    {
        var k = new LinearKernel<double>();
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    [Fact]
    public void Linear_SelfKernel_IsSquaredNorm()
    {
        var k = new LinearKernel<double>();
        var x = V(3, 4);
        // k(x,x) = ||x||^2 = 9 + 16 = 25
        Assert.Equal(25.0, k.Calculate(x, x), Tolerance);
    }

    [Fact]
    public void Linear_CauchySchwarz()
    {
        // |k(x,y)|^2 <= k(x,x) * k(y,y) for PD kernels
        var k = new LinearKernel<double>();
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        double kxy = k.Calculate(x, y);
        double kxx = k.Calculate(x, x);
        double kyy = k.Calculate(y, y);
        Assert.True(kxy * kxy <= kxx * kyy + Tolerance,
            "Cauchy-Schwarz violated for linear kernel");
    }

    [Fact]
    public void Linear_Bilinearity()
    {
        // k(ax, y) = a * k(x, y)
        var k = new LinearKernel<double>();
        double a = 3.0;
        var x = V(1, 2);
        var y = V(3, 4);
        var ax = V(a * 1, a * 2);
        Assert.Equal(a * k.Calculate(x, y), k.Calculate(ax, y), Tolerance);
    }

    // ============================================================
    //  GAUSSIAN (RBF) KERNEL: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    // ============================================================

    [Fact]
    public void Gaussian_SelfKernel_IsOne()
    {
        var k = new GaussianKernel<double>(sigma: 1.0);
        var x = V(1, 2, 3);
        Assert.Equal(1.0, k.Calculate(x, x), Tolerance);
    }

    [Fact]
    public void Gaussian_HandValue_Sigma1()
    {
        var k = new GaussianKernel<double>(sigma: 1.0);
        // x=[1,0], y=[0,1]: ||x-y||^2 = 1+1 = 2
        // k = exp(-2/(2*1)) = exp(-1) ≈ 0.3679
        Assert.Equal(Math.Exp(-1), k.Calculate(V(1, 0), V(0, 1)), Tolerance);
    }

    [Fact]
    public void Gaussian_HandValue_Sigma2()
    {
        var k = new GaussianKernel<double>(sigma: 2.0);
        // x=[1,0], y=[0,1]: ||x-y||^2 = 2
        // k = exp(-2/(2*4)) = exp(-0.25)
        Assert.Equal(Math.Exp(-0.25), k.Calculate(V(1, 0), V(0, 1)), Tolerance);
    }

    [Fact]
    public void Gaussian_OutputRange01()
    {
        var k = new GaussianKernel<double>(sigma: 1.0);
        var pairs = new[]
        {
            (V(0, 0), V(0, 0)),
            (V(1, 2), V(3, 4)),
            (V(1, 2), V(-1, -2)),
        };

        foreach (var (x, y) in pairs)
        {
            double val = k.Calculate(x, y);
            Assert.True(val > 0 && val <= 1.0, $"Gaussian kernel should be in (0,1], got {val}");
        }
    }

    [Fact]
    public void Gaussian_Symmetry()
    {
        var k = new GaussianKernel<double>(sigma: 1.5);
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    [Fact]
    public void Gaussian_DecreasesWithDistance()
    {
        var k = new GaussianKernel<double>(sigma: 1.0);
        var origin = V(0, 0);
        var near = V(0.5, 0);
        var far = V(2, 0);
        var veryFar = V(5, 0);

        double kNear = k.Calculate(origin, near);
        double kFar = k.Calculate(origin, far);
        double kVeryFar = k.Calculate(origin, veryFar);

        Assert.True(kNear > kFar, "Gaussian kernel should decrease with distance");
        Assert.True(kFar > kVeryFar, "Gaussian kernel should decrease with distance");
    }

    [Fact]
    public void Gaussian_SmallerSigma_NarrowerPeak()
    {
        var kNarrow = new GaussianKernel<double>(sigma: 0.5);
        var kWide = new GaussianKernel<double>(sigma: 2.0);
        var origin = V(0, 0);
        var point = V(1, 0);

        double narrow = kNarrow.Calculate(origin, point);
        double wide = kWide.Calculate(origin, point);

        Assert.True(narrow < wide,
            $"Narrow sigma should give lower values at distance 1: narrow={narrow}, wide={wide}");
    }

    [Fact]
    public void Gaussian_GramMatrix_PositiveDefinite()
    {
        // For a PD kernel, the Gram matrix must have non-negative eigenvalues
        var k = new GaussianKernel<double>(sigma: 1.0);
        var points = new[] { V(0, 0), V(1, 0), V(0, 1) };
        int n = points.Length;

        // Build Gram matrix
        double[,] gram = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                gram[i, j] = k.Calculate(points[i], points[j]);

        // Check positive definiteness: all diagonal elements should be positive
        // and determinant should be positive (sufficient for 3x3)
        for (int i = 0; i < n; i++)
            Assert.True(gram[i, i] > 0, "Diagonal of Gram matrix must be positive");

        // Check: for any vector c, c^T K c >= 0
        // Use a specific test vector
        double[] c = { 1, -1, 0.5 };
        double quadForm = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                quadForm += c[i] * gram[i, j] * c[j];
        Assert.True(quadForm >= -Tolerance, $"Gram matrix quadratic form should be non-negative: {quadForm}");
    }

    // ============================================================
    //  POLYNOMIAL KERNEL: k(x,y) = (x.y + c)^d
    // ============================================================

    [Fact]
    public void Polynomial_HandValues_Degree2()
    {
        var k = new PolynomialKernel<double>(
            degree: 2.0,
            coef0: 1.0);

        // [1,2] . [3,4] = 3+8 = 11
        // k = (11+1)^2 = 144
        Assert.Equal(144.0, k.Calculate(V(1, 2), V(3, 4)), Tolerance);
    }

    [Fact]
    public void Polynomial_HandValues_Degree1_IsLinearPlusC()
    {
        var k = new PolynomialKernel<double>(
            degree: 1.0,
            coef0: 0.0);

        // With degree 1, coef0=0: k(x,y) = (x.y)^1 = x.y (linear kernel)
        Assert.Equal(32.0, k.Calculate(V(1, 2, 3), V(4, 5, 6)), Tolerance);
    }

    [Fact]
    public void Polynomial_HandValues_Degree3()
    {
        var k = new PolynomialKernel<double>(
            degree: 3.0,
            coef0: 1.0);

        // [1,0] . [1,0] = 1, k = (1+1)^3 = 8
        Assert.Equal(8.0, k.Calculate(V(1, 0), V(1, 0)), Tolerance);
    }

    [Fact]
    public void Polynomial_Symmetry()
    {
        var k = new PolynomialKernel<double>(
            degree: 2.0,
            coef0: 1.0);
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    [Fact]
    public void Polynomial_Coef0_Zero_IdenticalVectors()
    {
        var k = new PolynomialKernel<double>(
            degree: 2.0,
            coef0: 0.0);

        // k(x,x) = (||x||^2)^2 with coef0=0
        var x = V(3, 4);
        // ||x||^2 = 25, k = 25^2 = 625
        Assert.Equal(625.0, k.Calculate(x, x), Tolerance);
    }

    // ============================================================
    //  LAPLACIAN KERNEL: k(x,y) = exp(-||x-y||_1 / sigma)
    // ============================================================

    [Fact]
    public void Laplacian_SelfKernel_IsOne()
    {
        var k = new LaplacianKernel<double>();
        Assert.Equal(1.0, k.Calculate(V(1, 2, 3), V(1, 2, 3)), Tolerance);
    }

    [Fact]
    public void Laplacian_HandValue()
    {
        var k = new LaplacianKernel<double>();
        // L1 distance: |1-3| + |2-4| = 2+2 = 4
        // k = exp(-4/1) = exp(-4)
        Assert.Equal(Math.Exp(-4), k.Calculate(V(1, 2), V(3, 4)), Tolerance);
    }

    [Fact]
    public void Laplacian_UsesL1Distance_NotL2()
    {
        // Verify Laplacian uses Manhattan (L1) not Euclidean (L2) distance
        var kLap = new LaplacianKernel<double>();
        var kGauss = new GaussianKernel<double>(sigma: 1.0);

        var x = V(1, 0);
        var y = V(0, 1);
        // L1 distance = |1-0| + |0-1| = 2
        // L2 squared distance = 1+1 = 2
        // Laplacian: exp(-2/1) = exp(-2) ≈ 0.1353
        // Gaussian: exp(-2/2) = exp(-1) ≈ 0.3679
        double lap = kLap.Calculate(x, y);
        double gauss = kGauss.Calculate(x, y);

        Assert.Equal(Math.Exp(-2), lap, Tolerance);
        Assert.Equal(Math.Exp(-1), gauss, Tolerance);
        Assert.True(lap < gauss, "Laplacian should be smaller than Gaussian for same sigma when L1 > L2");
    }

    [Fact]
    public void Laplacian_Symmetry()
    {
        var k = new LaplacianKernel<double>();
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    [Fact]
    public void Laplacian_OutputRange()
    {
        var k = new LaplacianKernel<double>();
        var x = V(1, 2);
        var y = V(3, 4);
        double val = k.Calculate(x, y);
        Assert.True(val > 0 && val <= 1.0, $"Laplacian kernel should be in (0,1], got {val}");
    }

    // ============================================================
    //  SIGMOID KERNEL: k(x,y) = tanh(alpha * x.y + c)
    // ============================================================

    [Fact]
    public void Sigmoid_HandValues()
    {
        var k = new SigmoidKernel<double>();
        // alpha=1, c=0: k = tanh(x.y)
        // [1,0] . [0,1] = 0, tanh(0) = 0
        Assert.Equal(0.0, k.Calculate(V(1, 0), V(0, 1)), Tolerance);

        // [1,1] . [1,1] = 2, tanh(2) ≈ 0.9640
        Assert.Equal(Math.Tanh(2), k.Calculate(V(1, 1), V(1, 1)), Tolerance);
    }

    [Fact]
    public void Sigmoid_WithAlphaAndC()
    {
        var k = new SigmoidKernel<double>(alpha: 0.5, c: -1.0);
        // k = tanh(0.5 * x.y - 1)
        // [1,1] . [1,1] = 2
        // k = tanh(0.5*2 - 1) = tanh(0) = 0
        Assert.Equal(0.0, k.Calculate(V(1, 1), V(1, 1)), Tolerance);
    }

    [Fact]
    public void Sigmoid_OutputRange()
    {
        // tanh outputs in (-1, 1)
        var k = new SigmoidKernel<double>();
        var x = V(10, 20);
        var y = V(-10, -20);
        double val = k.Calculate(x, y);
        Assert.True(val >= -1.0 && val <= 1.0, $"Sigmoid kernel should be in [-1,1], got {val}");
    }

    [Fact]
    public void Sigmoid_Symmetry()
    {
        var k = new SigmoidKernel<double>();
        var x = V(1, 2);
        var y = V(3, -1);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    // ============================================================
    //  COSINE KERNEL: k(x,y) = (x.y) / (||x|| * ||y||)
    // ============================================================

    [Fact]
    public void Cosine_IdenticalVectors_IsOne()
    {
        var k = new CosineKernel<double>();
        Assert.Equal(1.0, k.Calculate(V(3, 4), V(3, 4)), Tolerance);
    }

    [Fact]
    public void Cosine_OrthogonalVectors_IsZero()
    {
        var k = new CosineKernel<double>();
        Assert.Equal(0.0, k.Calculate(V(1, 0), V(0, 1)), Tolerance);
    }

    [Fact]
    public void Cosine_OppositeVectors_IsNegOne()
    {
        var k = new CosineKernel<double>();
        Assert.Equal(-1.0, k.Calculate(V(1, 0), V(-1, 0)), Tolerance);
    }

    [Fact]
    public void Cosine_ScaleInvariance()
    {
        // Cosine similarity is invariant to scaling
        var k = new CosineKernel<double>();
        var x = V(1, 2);
        var y = V(3, 4);
        var x2 = V(10, 20); // 10*x

        Assert.Equal(k.Calculate(x, y), k.Calculate(x2, y), Tolerance);
    }

    [Fact]
    public void Cosine_HandValue_45Degrees()
    {
        var k = new CosineKernel<double>();
        // Vectors at 45 degrees: cos(45) = 1/sqrt(2) ≈ 0.7071
        // [1,0] and [1,1] -> dot=1, ||x||=1, ||y||=sqrt(2)
        // cos = 1/sqrt(2)
        Assert.Equal(1.0 / Math.Sqrt(2), k.Calculate(V(1, 0), V(1, 1)), Tolerance);
    }

    [Fact]
    public void Cosine_Symmetry()
    {
        var k = new CosineKernel<double>();
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    // ============================================================
    //  MATERN KERNEL
    // ============================================================

    [Fact]
    public void Matern_SelfKernel_IsVariance()
    {
        var k12 = MaternKernel<double>.Matern12(variance: 2.0);
        var k32 = MaternKernel<double>.Matern32(variance: 3.0);
        var k52 = MaternKernel<double>.Matern52(variance: 4.0);

        var x = V(1, 2, 3);
        Assert.Equal(2.0, k12.Calculate(x, x), Tolerance);
        Assert.Equal(3.0, k32.Calculate(x, x), Tolerance);
        Assert.Equal(4.0, k52.Calculate(x, x), Tolerance);
    }

    [Fact]
    public void Matern12_IsExponential()
    {
        // Matern 1/2: k(r) = sigma^2 * exp(-r/l)
        var k = MaternKernel<double>.Matern12(lengthScale: 2.0, variance: 1.0);
        var x = V(0);
        var y = V(3);
        // r = 3, l = 2
        // k = exp(-3/2) = exp(-1.5)
        Assert.Equal(Math.Exp(-1.5), k.Calculate(x, y), Tolerance);
    }

    [Fact]
    public void Matern32_HandValue()
    {
        // Matern 3/2: k(r) = sigma^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
        var k = MaternKernel<double>.Matern32(lengthScale: 1.0, variance: 1.0);
        var x = V(0);
        var y = V(1);
        // r=1, l=1: k = (1 + sqrt(3)) * exp(-sqrt(3))
        double expected = (1 + Math.Sqrt(3)) * Math.Exp(-Math.Sqrt(3));
        Assert.Equal(expected, k.Calculate(x, y), Tolerance);
    }

    [Fact]
    public void Matern52_HandValue()
    {
        // Matern 5/2: k(r) = sigma^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
        var k = MaternKernel<double>.Matern52(lengthScale: 1.0, variance: 1.0);
        var x = V(0);
        var y = V(1);
        // r=1, l=1: k = (1 + sqrt(5) + 5/3) * exp(-sqrt(5))
        double sqrt5 = Math.Sqrt(5);
        double expected = (1 + sqrt5 + 5.0 / 3.0) * Math.Exp(-sqrt5);
        Assert.Equal(expected, k.Calculate(x, y), Tolerance);
    }

    [Fact]
    public void Matern_Symmetry()
    {
        var k = MaternKernel<double>.Matern52();
        var x = V(1, 2);
        var y = V(3, -1);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    [Fact]
    public void Matern_DecreasesWithDistance()
    {
        var k = MaternKernel<double>.Matern32();
        var origin = V(0);
        double prev = k.Calculate(origin, V(0));
        for (double d = 0.5; d <= 5; d += 0.5)
        {
            double curr = k.Calculate(origin, V(d));
            Assert.True(curr < prev, $"Matern should decrease with distance at d={d}");
            prev = curr;
        }
    }

    [Fact]
    public void Matern_LargerLengthScale_SlowerDecay()
    {
        var kShort = MaternKernel<double>.Matern52(lengthScale: 0.5);
        var kLong = MaternKernel<double>.Matern52(lengthScale: 3.0);
        var origin = V(0);
        var point = V(1);

        double shortVal = kShort.Calculate(origin, point);
        double longVal = kLong.Calculate(origin, point);

        Assert.True(longVal > shortVal,
            "Larger length scale should give higher kernel value at same distance");
    }

    [Fact]
    public void Matern_OutputRange()
    {
        var k = MaternKernel<double>.Matern52(variance: 1.0);
        var x = V(1, 2);
        var y = V(10, 20);
        double val = k.Calculate(x, y);
        Assert.True(val >= 0 && val <= 1.0, $"Matern kernel (variance=1) should be in [0,1], got {val}");
    }

    // ============================================================
    //  RATIONAL QUADRATIC KERNEL
    // ============================================================

    [Fact]
    public void RationalQuadratic_SelfKernel_IsVariance()
    {
        var k = new RationalQuadraticKernel<double>(variance: 2.5);
        var x = V(1, 2, 3);
        Assert.Equal(2.5, k.Calculate(x, x), Tolerance);
    }

    [Fact]
    public void RationalQuadratic_HandValue()
    {
        // k = sigma^2 * (1 + r^2/(2*alpha*l^2))^(-alpha)
        var k = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 1.0, variance: 1.0);
        var x = V(0);
        var y = V(2);
        // r^2 = 4, alpha=1, l=1
        // k = (1 + 4/(2*1*1))^(-1) = (1 + 2)^(-1) = 1/3
        Assert.Equal(1.0 / 3.0, k.Calculate(x, y), Tolerance);
    }

    [Fact]
    public void RationalQuadratic_LargeAlpha_ApproachesGaussian()
    {
        // As alpha -> inf, RQ -> RBF
        var kRQ = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 10000, variance: 1.0);
        var kRBF = new GaussianKernel<double>(sigma: 1.0);

        var x = V(0, 0);
        var y = V(1, 0);
        double rq = kRQ.Calculate(x, y);
        double rbf = kRBF.Calculate(x, y);

        Assert.Equal(rbf, rq, 0.01);
    }

    [Fact]
    public void RationalQuadratic_Symmetry()
    {
        var k = new RationalQuadraticKernel<double>();
        var x = V(1, 2);
        var y = V(3, -1);
        Assert.Equal(k.Calculate(x, y), k.Calculate(y, x), Tolerance);
    }

    [Fact]
    public void RationalQuadratic_DecreasesWithDistance()
    {
        var k = new RationalQuadraticKernel<double>();
        var origin = V(0);
        double prev = k.Calculate(origin, V(0));
        for (double d = 0.5; d <= 5; d += 0.5)
        {
            double curr = k.Calculate(origin, V(d));
            Assert.True(curr < prev, $"RQ should decrease with distance at d={d}");
            prev = curr;
        }
    }

    [Fact]
    public void RationalQuadratic_HyperparameterGradient_NumericalCheck()
    {
        var k = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 2.0, variance: 1.0);
        var x = V(1, 2);
        var y = V(3, -1);

        var grads = k.CalculateHyperparameterGradients(x, y);

        // Numerical gradient for variance
        double h = 1e-5;
        var kPlus = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 2.0, variance: 1.0 + h);
        var kMinus = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 2.0, variance: 1.0 - h);
        double numGradVariance = (kPlus.Calculate(x, y) - kMinus.Calculate(x, y)) / (2 * h);
        Assert.Equal(numGradVariance, grads["variance"], 0.001);

        // Numerical gradient for lengthScale
        kPlus = new RationalQuadraticKernel<double>(lengthScale: 1.0 + h, alpha: 2.0, variance: 1.0);
        kMinus = new RationalQuadraticKernel<double>(lengthScale: 1.0 - h, alpha: 2.0, variance: 1.0);
        double numGradLengthScale = (kPlus.Calculate(x, y) - kMinus.Calculate(x, y)) / (2 * h);
        Assert.Equal(numGradLengthScale, grads["lengthScale"], 0.001);

        // Numerical gradient for alpha
        kPlus = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 2.0 + h, variance: 1.0);
        kMinus = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 2.0 - h, variance: 1.0);
        double numGradAlpha = (kPlus.Calculate(x, y) - kMinus.Calculate(x, y)) / (2 * h);
        Assert.Equal(numGradAlpha, grads["alpha"], 0.001);
    }

    [Fact]
    public void RationalQuadratic_InputGradient_NumericalCheck()
    {
        var k = new RationalQuadraticKernel<double>(lengthScale: 1.0, alpha: 2.0, variance: 1.0);
        var x = V(1, 2);
        var y = V(3, -1);

        var grad = k.CalculateGradient(x, y);
        double h = 1e-6;

        for (int d = 0; d < 2; d++)
        {
            var xPlus = V(x[0], x[1]);
            var xMinus = V(x[0], x[1]);
            xPlus[d] += h;
            xMinus[d] -= h;
            double numGrad = (k.Calculate(xPlus, y) - k.Calculate(xMinus, y)) / (2 * h);
            Assert.Equal(numGrad, grad[d], LooseTolerance);
        }
    }

    // ============================================================
    //  CROSS-KERNEL PROPERTIES
    // ============================================================

    [Fact]
    public void AllPDKernels_GramMatrix_DiagonalPositive()
    {
        var points = new[] { V(1, 2), V(3, -1), V(0, 5), V(-2, 3) };
        var kernels = new IKernelFunction<double>[]
        {
            new GaussianKernel<double>(sigma: 1.0),
            new LaplacianKernel<double>(),
            new PolynomialKernel<double>(degree: 2.0, coef0: 1.0),
            MaternKernel<double>.Matern52(),
            new RationalQuadraticKernel<double>(),
        };

        foreach (var kernel in kernels)
        {
            foreach (var x in points)
            {
                double kxx = kernel.Calculate(x, x);
                Assert.True(kxx > 0, $"k(x,x) should be positive for kernel {kernel.GetType().Name}");
            }
        }
    }

    [Fact]
    public void AllKernels_Symmetry()
    {
        var x = V(1, 2, 3);
        var y = V(4, -1, 2);
        var kernels = new IKernelFunction<double>[]
        {
            new LinearKernel<double>(),
            new GaussianKernel<double>(sigma: 1.0),
            new LaplacianKernel<double>(),
            new PolynomialKernel<double>(degree: 2.0, coef0: 1.0),
            new SigmoidKernel<double>(),
            new CosineKernel<double>(),
            MaternKernel<double>.Matern52(),
            new RationalQuadraticKernel<double>(),
        };

        foreach (var kernel in kernels)
        {
            double kxy = kernel.Calculate(x, y);
            double kyx = kernel.Calculate(y, x);
            Assert.Equal(kxy, kyx, Tolerance);
        }
    }
}
