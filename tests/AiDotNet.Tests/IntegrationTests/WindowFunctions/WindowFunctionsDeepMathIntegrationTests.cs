using AiDotNet.Interfaces;
using AiDotNet.WindowFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WindowFunctions;

/// <summary>
/// Deep mathematical correctness tests for window functions.
/// Verifies exact values, symmetry, boundary conditions, and cross-window identities
/// against hand-calculated reference values from signal processing theory.
/// </summary>
public class WindowFunctionsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-6;
    private const int DefaultSize = 11; // Odd size for exact center

    // ============================================================
    //  RECTANGULAR WINDOW
    // ============================================================

    [Fact]
    public void Rectangular_AllOnes()
    {
        var window = new RectangularWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize; i++)
        {
            Assert.Equal(1.0, window[i], Tolerance);
        }
    }

    [Fact]
    public void Rectangular_SumEqualsLength()
    {
        int size = 64;
        var window = new RectangularWindow<double>().Create(size);
        double sum = 0;
        for (int i = 0; i < size; i++) sum += window[i];
        Assert.Equal(size, sum, Tolerance);
    }

    // ============================================================
    //  HANNING (HANN) WINDOW
    // ============================================================

    [Fact]
    public void Hanning_ZeroAtEdges()
    {
        var window = new HanningWindow<double>().Create(DefaultSize);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.0, window[DefaultSize - 1], Tolerance);
    }

    [Fact]
    public void Hanning_OneAtCenter()
    {
        // For odd-sized window, center value should be 1.0
        var window = new HanningWindow<double>().Create(DefaultSize);
        int center = DefaultSize / 2; // index 5 for size 11
        Assert.Equal(1.0, window[center], Tolerance);
    }

    [Fact]
    public void Hanning_Symmetry()
    {
        var window = new HanningWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize / 2; i++)
        {
            Assert.Equal(window[i], window[DefaultSize - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Hanning_HandValue_N4()
    {
        // Size 4: w(n) = 0.5*(1 - cos(2*pi*n/3))
        // n=0: 0.5*(1-cos(0)) = 0
        // n=1: 0.5*(1-cos(2pi/3)) = 0.5*(1-(-0.5)) = 0.75
        // n=2: 0.5*(1-cos(4pi/3)) = 0.5*(1-(-0.5)) = 0.75
        // n=3: 0.5*(1-cos(2pi)) = 0
        var window = new HanningWindow<double>().Create(4);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.75, window[1], Tolerance);
        Assert.Equal(0.75, window[2], Tolerance);
        Assert.Equal(0.0, window[3], Tolerance);
    }

    [Fact]
    public void Hanning_HandValue_Specific()
    {
        // Size 5: w(n) = 0.5*(1 - cos(2*pi*n/4))
        // n=0: 0, n=1: 0.5*(1-cos(pi/2))=0.5, n=2: 0.5*(1-cos(pi))=1.0, n=3: 0.5, n=4: 0
        var window = new HanningWindow<double>().Create(5);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.5, window[1], Tolerance);
        Assert.Equal(1.0, window[2], Tolerance);
        Assert.Equal(0.5, window[3], Tolerance);
        Assert.Equal(0.0, window[4], Tolerance);
    }

    // ============================================================
    //  HAMMING WINDOW
    // ============================================================

    [Fact]
    public void Hamming_NotZeroAtEdges()
    {
        // Hamming window has nonzero edges: 0.54 - 0.46 = 0.08
        var window = new HammingWindow<double>().Create(DefaultSize);
        Assert.Equal(0.08, window[0], Tolerance);
        Assert.Equal(0.08, window[DefaultSize - 1], Tolerance);
    }

    [Fact]
    public void Hamming_OneAtCenter()
    {
        // At center: 0.54 - 0.46*cos(pi) = 0.54 + 0.46 = 1.0
        var window = new HammingWindow<double>().Create(DefaultSize);
        int center = DefaultSize / 2;
        Assert.Equal(1.0, window[center], Tolerance);
    }

    [Fact]
    public void Hamming_Symmetry()
    {
        var window = new HammingWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize / 2; i++)
        {
            Assert.Equal(window[i], window[DefaultSize - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Hamming_HandValue_N5()
    {
        // Size 5: w(n) = 0.54 - 0.46*cos(2*pi*n/4)
        // n=0: 0.54 - 0.46*cos(0) = 0.54 - 0.46 = 0.08
        // n=1: 0.54 - 0.46*cos(pi/2) = 0.54 - 0.46*0 = 0.54
        // n=2: 0.54 - 0.46*cos(pi) = 0.54 + 0.46 = 1.0
        // n=3: same as n=1 = 0.54
        // n=4: same as n=0 = 0.08
        var window = new HammingWindow<double>().Create(5);
        Assert.Equal(0.08, window[0], Tolerance);
        Assert.Equal(0.54, window[1], Tolerance);
        Assert.Equal(1.0, window[2], Tolerance);
        Assert.Equal(0.54, window[3], Tolerance);
        Assert.Equal(0.08, window[4], Tolerance);
    }

    [Fact]
    public void Hamming_GreaterThanOrEqualHanning()
    {
        // Hamming >= Hanning at all points (because Hamming lifts the minimum from 0 to 0.08)
        var hamming = new HammingWindow<double>().Create(DefaultSize);
        var hanning = new HanningWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize; i++)
        {
            Assert.True(hamming[i] >= hanning[i] - 1e-10,
                $"Hamming[{i}]={hamming[i]} should be >= Hanning[{i}]={hanning[i]}");
        }
    }

    // ============================================================
    //  BLACKMAN WINDOW
    // ============================================================

    [Fact]
    public void Blackman_ZeroAtEdges()
    {
        // At n=0: 0.42 - 0.5*cos(0) + 0.08*cos(0) = 0.42 - 0.5 + 0.08 = 0.0
        var window = new BlackmanWindow<double>().Create(DefaultSize);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.0, window[DefaultSize - 1], Tolerance);
    }

    [Fact]
    public void Blackman_OneAtCenter()
    {
        // At center: 0.42 - 0.5*cos(pi) + 0.08*cos(2pi) = 0.42 + 0.5 + 0.08 = 1.0
        var window = new BlackmanWindow<double>().Create(DefaultSize);
        int center = DefaultSize / 2;
        Assert.Equal(1.0, window[center], Tolerance);
    }

    [Fact]
    public void Blackman_Symmetry()
    {
        var window = new BlackmanWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize / 2; i++)
        {
            Assert.Equal(window[i], window[DefaultSize - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Blackman_HandValue_N5()
    {
        // Size 5: w(n) = 0.42 - 0.5*cos(2*pi*n/4) + 0.08*cos(4*pi*n/4)
        // n=0: 0.42 - 0.5*1 + 0.08*1 = 0.0
        // n=1: 0.42 - 0.5*cos(pi/2) + 0.08*cos(pi) = 0.42 - 0 - 0.08 = 0.34
        // n=2: 0.42 - 0.5*cos(pi) + 0.08*cos(2pi) = 0.42 + 0.5 + 0.08 = 1.0
        // n=3: same as n=1 = 0.34
        // n=4: same as n=0 = 0.0
        var window = new BlackmanWindow<double>().Create(5);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.34, window[1], Tolerance);
        Assert.Equal(1.0, window[2], Tolerance);
        Assert.Equal(0.34, window[3], Tolerance);
        Assert.Equal(0.0, window[4], Tolerance);
    }

    [Fact]
    public void Blackman_LessThanOrEqualHanning()
    {
        // Blackman <= Hanning at all points (Blackman has more aggressive tapering)
        var blackman = new BlackmanWindow<double>().Create(DefaultSize);
        var hanning = new HanningWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize; i++)
        {
            Assert.True(blackman[i] <= hanning[i] + 1e-10,
                $"Blackman[{i}]={blackman[i]} should be <= Hanning[{i}]={hanning[i]}");
        }
    }

    // ============================================================
    //  TRIANGULAR WINDOW
    // ============================================================

    [Fact]
    public void Triangular_ZeroAtEdges()
    {
        // w(0) = 1 - |2*0 - L|/L = 1 - L/L = 0
        var window = new TriangularWindow<double>().Create(DefaultSize);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.0, window[DefaultSize - 1], Tolerance);
    }

    [Fact]
    public void Triangular_OneAtCenter()
    {
        var window = new TriangularWindow<double>().Create(DefaultSize);
        int center = DefaultSize / 2;
        Assert.Equal(1.0, window[center], Tolerance);
    }

    [Fact]
    public void Triangular_Symmetry()
    {
        var window = new TriangularWindow<double>().Create(DefaultSize);
        for (int i = 0; i < DefaultSize / 2; i++)
        {
            Assert.Equal(window[i], window[DefaultSize - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Triangular_Linear_Increase()
    {
        // First half should linearly increase
        var window = new TriangularWindow<double>().Create(DefaultSize);
        for (int i = 1; i <= DefaultSize / 2; i++)
        {
            Assert.True(window[i] > window[i - 1],
                $"Triangular[{i}]={window[i]} should be > Triangular[{i - 1}]={window[i - 1]}");
        }
    }

    [Fact]
    public void Triangular_HandValue_N5()
    {
        // Size 5, L=4: w(n) = 1 - |2n - 4|/4
        // n=0: 1 - 4/4 = 0.0
        // n=1: 1 - 2/4 = 0.5
        // n=2: 1 - 0/4 = 1.0
        // n=3: 1 - 2/4 = 0.5
        // n=4: 1 - 4/4 = 0.0
        var window = new TriangularWindow<double>().Create(5);
        Assert.Equal(0.0, window[0], Tolerance);
        Assert.Equal(0.5, window[1], Tolerance);
        Assert.Equal(1.0, window[2], Tolerance);
        Assert.Equal(0.5, window[3], Tolerance);
        Assert.Equal(0.0, window[4], Tolerance);
    }

    // ============================================================
    //  GAUSSIAN WINDOW
    // ============================================================

    [Fact]
    public void Gaussian_OneAtCenter()
    {
        var window = new GaussianWindow<double>(0.5).Create(DefaultSize);
        int center = DefaultSize / 2;
        Assert.Equal(1.0, window[center], Tolerance);
    }

    [Fact]
    public void Gaussian_Symmetry()
    {
        var window = new GaussianWindow<double>(0.4).Create(DefaultSize);
        for (int i = 0; i < DefaultSize / 2; i++)
        {
            Assert.Equal(window[i], window[DefaultSize - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Gaussian_HandValue_AtEdge()
    {
        // Size 11, sigma=0.5
        // halfN = 5.0
        // At n=0: x = (0 - 5) / (0.5 * 5) = -2.0
        // w(0) = exp(-0.5 * 4) = exp(-2) ≈ 0.1353
        var window = new GaussianWindow<double>(0.5).Create(11);
        Assert.Equal(Math.Exp(-2.0), window[0], LooseTolerance);
    }

    [Fact]
    public void Gaussian_HandValue_AtQuarter()
    {
        // Size 11, sigma=0.5, halfN=5
        // At n=2: x = (2-5)/(0.5*5) = -1.2
        // w(2) = exp(-0.5*1.44) = exp(-0.72)
        var window = new GaussianWindow<double>(0.5).Create(11);
        Assert.Equal(Math.Exp(-0.72), window[2], LooseTolerance);
    }

    [Fact]
    public void Gaussian_AllPositive()
    {
        var window = new GaussianWindow<double>(0.5).Create(DefaultSize);
        for (int i = 0; i < DefaultSize; i++)
        {
            Assert.True(window[i] > 0, $"Gaussian[{i}]={window[i]} should be positive");
        }
    }

    [Fact]
    public void Gaussian_NarrowerSigma_HasSmallerEdges()
    {
        // Narrower sigma => more tapered edges
        var narrow = new GaussianWindow<double>(0.3).Create(DefaultSize);
        var wide = new GaussianWindow<double>(0.7).Create(DefaultSize);
        Assert.True(narrow[0] < wide[0],
            $"Narrow sigma edge {narrow[0]} should be < wide sigma edge {wide[0]}");
    }

    // ============================================================
    //  KAISER WINDOW
    // ============================================================

    [Fact]
    public void Kaiser_OneAtCenter()
    {
        var window = new KaiserWindow<double>(5.0).Create(DefaultSize);
        int center = DefaultSize / 2;
        Assert.Equal(1.0, window[center], Tolerance);
    }

    [Fact]
    public void Kaiser_Symmetry()
    {
        var window = new KaiserWindow<double>(5.0).Create(DefaultSize);
        for (int i = 0; i < DefaultSize / 2; i++)
        {
            Assert.Equal(window[i], window[DefaultSize - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Kaiser_AllPositive()
    {
        var window = new KaiserWindow<double>(5.0).Create(DefaultSize);
        for (int i = 0; i < DefaultSize; i++)
        {
            Assert.True(window[i] > 0, $"Kaiser[{i}]={window[i]} should be positive");
        }
    }

    [Fact]
    public void Kaiser_HigherBeta_SmallerEdges()
    {
        var lowBeta = new KaiserWindow<double>(2.0).Create(DefaultSize);
        var highBeta = new KaiserWindow<double>(10.0).Create(DefaultSize);
        Assert.True(highBeta[0] < lowBeta[0],
            $"High beta edge {highBeta[0]} should be < low beta edge {lowBeta[0]}");
    }

    [Fact]
    public void Kaiser_Beta0_ApproachesRectangular()
    {
        // With beta=0, Kaiser approaches Rectangular (but normalized)
        var window = new KaiserWindow<double>(0.0).Create(DefaultSize);
        for (int i = 0; i < DefaultSize; i++)
        {
            Assert.Equal(1.0, window[i], LooseTolerance);
        }
    }

    // ============================================================
    //  CROSS-WINDOW PROPERTIES
    // ============================================================

    [Fact]
    public void AllWindows_MaxAtCenter()
    {
        int size = 21;
        int center = size / 2;

        // Rectangular excluded: all values are 1.0, so max is at every index
        var windows = new (string name, IWindowFunction<double> wf)[]
        {
            ("Hanning", new HanningWindow<double>()),
            ("Hamming", new HammingWindow<double>()),
            ("Blackman", new BlackmanWindow<double>()),
            ("Triangular", new TriangularWindow<double>()),
            ("Gaussian", new GaussianWindow<double>(0.5)),
            ("Kaiser", new KaiserWindow<double>(5.0)),
        };

        foreach (var (name, wf) in windows)
        {
            var w = wf.Create(size);
            double maxVal = double.MinValue;
            int maxIdx = -1;
            for (int i = 0; i < size; i++)
            {
                if (w[i] > maxVal) { maxVal = w[i]; maxIdx = i; }
            }
            Assert.Equal(center, maxIdx);
        }
    }

    [Fact]
    public void AllWindows_ValuesInRange01()
    {
        int size = 64;
        var windows = new (string name, IWindowFunction<double> wf)[]
        {
            ("Rectangular", new RectangularWindow<double>()),
            ("Hanning", new HanningWindow<double>()),
            ("Hamming", new HammingWindow<double>()),
            ("Blackman", new BlackmanWindow<double>()),
            ("Triangular", new TriangularWindow<double>()),
            ("Gaussian", new GaussianWindow<double>(0.5)),
            ("Kaiser", new KaiserWindow<double>(5.0)),
        };

        foreach (var (name, wf) in windows)
        {
            var w = wf.Create(size);
            for (int i = 0; i < size; i++)
            {
                Assert.True(w[i] >= -1e-10 && w[i] <= 1.0 + 1e-10,
                    $"{name}[{i}]={w[i]} should be in [0,1]");
            }
        }
    }

    [Fact]
    public void WindowSum_Ordering()
    {
        // Rectangular has highest sum, then Hamming, Hanning, Blackman
        int size = 64;
        double sumRect = SumWindow(new RectangularWindow<double>().Create(size));
        double sumHamming = SumWindow(new HammingWindow<double>().Create(size));
        double sumHanning = SumWindow(new HanningWindow<double>().Create(size));
        double sumBlackman = SumWindow(new BlackmanWindow<double>().Create(size));

        Assert.True(sumRect > sumHamming, $"Rectangular sum {sumRect} should be > Hamming sum {sumHamming}");
        Assert.True(sumHamming > sumHanning, $"Hamming sum {sumHamming} should be > Hanning sum {sumHanning}");
        Assert.True(sumHanning > sumBlackman, $"Hanning sum {sumHanning} should be > Blackman sum {sumBlackman}");
    }

    [Fact]
    public void Hanning_Sum_HandValue()
    {
        // For Hanning window of size N:
        // Sum = 0.5 * N (for large N, since mean of raised cosine = 0.5)
        int size = 1001;
        var window = new HanningWindow<double>().Create(size);
        double sum = SumWindow(window);
        // Sum ≈ N/2 for large N
        Assert.Equal(size / 2.0, sum, 2.0); // within 2 of the expected
    }

    // ============================================================
    //  SIZE-1 EDGE CASE
    // ============================================================

    [Fact]
    public void AllWindows_Size1_ReturnsSingleOne()
    {
        var windows = new (string name, IWindowFunction<double> wf)[]
        {
            ("Hanning", new HanningWindow<double>()),
            ("Hamming", new HammingWindow<double>()),
            ("Gaussian", new GaussianWindow<double>(0.5)),
            ("Kaiser", new KaiserWindow<double>(5.0)),
        };

        foreach (var (name, wf) in windows)
        {
            var w = wf.Create(1);
            Assert.Equal(1, w.Length);
            Assert.Equal(1.0, w[0], Tolerance);
        }
    }

    // ============================================================
    //  EVEN SIZE TESTS
    // ============================================================

    [Fact]
    public void Hanning_EvenSize_Symmetric()
    {
        int size = 10;
        var window = new HanningWindow<double>().Create(size);
        for (int i = 0; i < size / 2; i++)
        {
            Assert.Equal(window[i], window[size - 1 - i], Tolerance);
        }
    }

    [Fact]
    public void Hamming_EvenSize_EdgeValues()
    {
        int size = 10;
        var window = new HammingWindow<double>().Create(size);
        Assert.Equal(0.08, window[0], Tolerance);
        Assert.Equal(0.08, window[size - 1], Tolerance);
    }

    // ============================================================
    //  FLOAT TYPE CROSS-CHECK
    // ============================================================

    [Fact]
    public void Hanning_FloatAndDouble_Consistent()
    {
        int size = 11;
        var doubleWindow = new HanningWindow<double>().Create(size);
        var floatWindow = new HanningWindow<float>().Create(size);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(doubleWindow[i], floatWindow[i], 1e-5);
        }
    }

    [Fact]
    public void Hamming_FloatAndDouble_Consistent()
    {
        int size = 11;
        var doubleWindow = new HammingWindow<double>().Create(size);
        var floatWindow = new HammingWindow<float>().Create(size);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(doubleWindow[i], floatWindow[i], 1e-5);
        }
    }

    // ============================================================
    //  ENERGY NORMALIZATION TESTS
    // ============================================================

    [Fact]
    public void Hanning_CoherentGain()
    {
        // Coherent gain = sum(w) / N
        // For Hanning: ≈ 0.5
        int size = 1024;
        var window = new HanningWindow<double>().Create(size);
        double coherentGain = SumWindow(window) / size;
        Assert.Equal(0.5, coherentGain, 0.01);
    }

    [Fact]
    public void Hamming_CoherentGain()
    {
        // For Hamming: ≈ 0.54
        int size = 1024;
        var window = new HammingWindow<double>().Create(size);
        double coherentGain = SumWindow(window) / size;
        Assert.Equal(0.54, coherentGain, 0.01);
    }

    [Fact]
    public void Blackman_CoherentGain()
    {
        // For Blackman: ≈ 0.42
        int size = 1024;
        var window = new BlackmanWindow<double>().Create(size);
        double coherentGain = SumWindow(window) / size;
        Assert.Equal(0.42, coherentGain, 0.01);
    }

    // ============================================================
    //  HELPERS
    // ============================================================

    private static double SumWindow(Vector<double> window)
    {
        double sum = 0;
        for (int i = 0; i < window.Length; i++)
            sum += window[i];
        return sum;
    }
}
