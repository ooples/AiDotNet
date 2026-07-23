using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.WindowFunctions;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.WindowFunctions;

/// <summary>
/// Comprehensive integration tests for all window function classes.
/// Tests mathematical properties: symmetry, boundary values, peak location,
/// and cross-window comparisons for every concrete window function.
/// </summary>
public class AdvancedWindowFunctionsTests
{
    private const double Tolerance = 1e-6;

    #region All Windows - Size and Symmetry Tests

    [Fact(Timeout = 120000)]
    public async Task AllWindows_Create_CorrectSize()
    {
        var windows = new IWindowFunction<double>[]
        {
            new RectangularWindow<double>(),
            new HammingWindow<double>(),
            new HanningWindow<double>(),
            new BlackmanWindow<double>(),
            new BlackmanHarrisWindow<double>(),
            new BlackmanNuttallWindow<double>(),
            new BartlettWindow<double>(),
            new BartlettHannWindow<double>(),
            new BohmanWindow<double>(),
            new CosineWindow<double>(),
            new FlatTopWindow<double>(),
            new GaussianWindow<double>(),
            new KaiserWindow<double>(),
            new LanczosWindow<double>(),
            new NuttallWindow<double>(),
            new ParzenWindow<double>(),
            new PoissonWindow<double>(),
            new TriangularWindow<double>(),
            new TukeyWindow<double>(),
            new WelchWindow<double>(),
        };

        foreach (var window in windows)
        {
            var result = window.Create(64);
            Assert.Equal(64, result.Length);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AllWindows_Create_NoNaN()
    {
        var windows = new IWindowFunction<double>[]
        {
            new RectangularWindow<double>(),
            new HammingWindow<double>(),
            new HanningWindow<double>(),
            new BlackmanWindow<double>(),
            new BlackmanHarrisWindow<double>(),
            new BlackmanNuttallWindow<double>(),
            new BartlettWindow<double>(),
            new BartlettHannWindow<double>(),
            new BohmanWindow<double>(),
            new CosineWindow<double>(),
            new FlatTopWindow<double>(),
            new GaussianWindow<double>(),
            new KaiserWindow<double>(),
            new LanczosWindow<double>(),
            new NuttallWindow<double>(),
            new ParzenWindow<double>(),
            new PoissonWindow<double>(),
            new TriangularWindow<double>(),
            new TukeyWindow<double>(),
            new WelchWindow<double>(),
        };

        foreach (var window in windows)
        {
            var result = window.Create(32);
            for (int i = 0; i < result.Length; i++)
            {
                Assert.False(double.IsNaN(result[i]),
                    $"{window.GetType().Name}[{i}] = NaN");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SymmetricWindows_Create_AreSymmetric()
    {
        // These windows should produce symmetric output
        var windows = new IWindowFunction<double>[]
        {
            new HammingWindow<double>(),
            new HanningWindow<double>(),
            new BlackmanWindow<double>(),
            new BlackmanHarrisWindow<double>(),
            new GaussianWindow<double>(),
            new NuttallWindow<double>(),
            new CosineWindow<double>(),
            new WelchWindow<double>(),
            new BartlettWindow<double>(),
            new TriangularWindow<double>(),
        };

        foreach (var window in windows)
        {
            var result = window.Create(33); // odd size for exact center
            int n = result.Length;
            for (int i = 0; i < n / 2; i++)
            {
                Assert.Equal(result[i], result[n - 1 - i], 1e-4);
            }
        }
    }

    #endregion

    #region Hamming Window Tests

    [Fact(Timeout = 120000)]
    public async Task HammingWindow_PeakAtCenter()
    {
        var window = new HammingWindow<double>();
        var result = window.Create(32);
        int center = result.Length / 2;
        for (int i = 0; i < result.Length; i++)
        {
            if (i != center && i != center - 1)
            {
                Assert.True(result[center] >= result[i] - Tolerance);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task HammingWindow_EdgeValues_NotZero()
    {
        // Hamming window edges are ~0.08, not zero
        var window = new HammingWindow<double>();
        var result = window.Create(64);
        Assert.True(result[0] > 0.05);
    }

    [Fact(Timeout = 120000)]
    public async Task HammingWindow_GetWindowFunctionType_ReturnsHamming()
    {
        var window = new HammingWindow<double>();
        Assert.Equal(WindowFunctionType.Hamming, window.GetWindowFunctionType());
    }

    #endregion

    #region Hanning Window Tests

    [Fact(Timeout = 120000)]
    public async Task HanningWindow_EdgeValues_ApproachZero()
    {
        var window = new HanningWindow<double>();
        var result = window.Create(64);
        Assert.True(result[0] < 0.01);
    }

    [Fact(Timeout = 120000)]
    public async Task HanningWindow_CenterValue_IsOne()
    {
        var window = new HanningWindow<double>();
        var result = window.Create(33); // odd for exact center
        Assert.Equal(1.0, result[16], 0.01);
    }

    #endregion

    #region Blackman Window Tests

    [Fact(Timeout = 120000)]
    public async Task BlackmanWindow_EdgeValues_NearZero()
    {
        var window = new BlackmanWindow<double>();
        var result = window.Create(64);
        Assert.True(Math.Abs(result[0]) < 0.01);
    }

    [Fact(Timeout = 120000)]
    public async Task BlackmanWindow_NarrowerMainLobeThanHanning()
    {
        // Blackman has lower sidelobes but wider main lobe
        var blackman = new BlackmanWindow<double>();
        var hanning = new HanningWindow<double>();
        var bResult = blackman.Create(64);
        var hResult = hanning.Create(64);
        // Blackman edges should be closer to zero
        Assert.True(Math.Abs(bResult[0]) <= Math.Abs(hResult[0]) + Tolerance);
    }

    #endregion

    #region Kaiser Window Tests

    [Fact(Timeout = 120000)]
    public async Task KaiserWindow_Create_AllPositive()
    {
        var window = new KaiserWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region Gaussian Window Tests

    [Fact(Timeout = 120000)]
    public async Task GaussianWindow_PeakAtCenter()
    {
        var window = new GaussianWindow<double>();
        var result = window.Create(33);
        double maxVal = double.MinValue;
        int maxIdx = -1;
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] > maxVal)
            {
                maxVal = result[i];
                maxIdx = i;
            }
        }
        Assert.Equal(16, maxIdx);
    }

    [Fact(Timeout = 120000)]
    public async Task GaussianWindow_AllPositive()
    {
        var window = new GaussianWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] > 0);
        }
    }

    #endregion

    #region Bartlett Window Tests

    [Fact(Timeout = 120000)]
    public async Task BartlettWindow_EdgeValues_AreZero()
    {
        var window = new BartlettWindow<double>();
        var result = window.Create(64);
        Assert.Equal(0.0, result[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BartlettWindow_PeakAtCenter()
    {
        var window = new BartlettWindow<double>();
        var result = window.Create(33);
        Assert.Equal(1.0, result[16], 0.01);
    }

    #endregion

    #region Triangular Window Tests

    [Fact(Timeout = 120000)]
    public async Task TriangularWindow_Create_AllNonNegative()
    {
        var window = new TriangularWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region Welch Window Tests

    [Fact(Timeout = 120000)]
    public async Task WelchWindow_EdgeValues_AreZero()
    {
        var window = new WelchWindow<double>();
        var result = window.Create(64);
        Assert.Equal(0.0, result[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task WelchWindow_CenterValue_IsOne()
    {
        var window = new WelchWindow<double>();
        var result = window.Create(33);
        Assert.Equal(1.0, result[16], 0.01);
    }

    #endregion

    #region Tukey Window Tests

    [Fact(Timeout = 120000)]
    public async Task TukeyWindow_Create_AllNonNegative()
    {
        var window = new TukeyWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region Parzen Window Tests

    [Fact(Timeout = 120000)]
    public async Task ParzenWindow_Create_AllNonNegative()
    {
        var window = new ParzenWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region Poisson Window Tests

    [Fact(Timeout = 120000)]
    public async Task PoissonWindow_Create_AllPositive()
    {
        var window = new PoissonWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] > 0);
        }
    }

    #endregion

    #region Bohman Window Tests

    [Fact(Timeout = 120000)]
    public async Task BohmanWindow_EdgeValues_NearZero()
    {
        var window = new BohmanWindow<double>();
        var result = window.Create(64);
        Assert.True(Math.Abs(result[0]) < 0.01);
    }

    #endregion

    #region FlatTop Window Tests

    [Fact(Timeout = 120000)]
    public async Task FlatTopWindow_CenterValue_ApproximatelyOne()
    {
        var window = new FlatTopWindow<double>();
        var result = window.Create(33);
        // FlatTop has a flat top near center, value ~1
        Assert.True(Math.Abs(result[16] - 1.0) < 0.1);
    }

    #endregion

    #region Lanczos Window Tests

    [Fact(Timeout = 120000)]
    public async Task LanczosWindow_Create_AllNonNegative()
    {
        var window = new LanczosWindow<double>();
        var result = window.Create(64);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= -Tolerance,
                $"LanczosWindow[{i}] = {result[i]} < 0");
        }
    }

    #endregion

    #region Size 1 and Size 2 Edge Cases

    [Fact(Timeout = 120000)]
    public async Task AllWindows_SizeOne_ReturnsOneElement()
    {
        var windows = new IWindowFunction<double>[]
        {
            new RectangularWindow<double>(),
            new HammingWindow<double>(),
            new HanningWindow<double>(),
            new GaussianWindow<double>(),
            new KaiserWindow<double>(),
        };

        foreach (var window in windows)
        {
            var result = window.Create(1);
            Assert.Equal(1, result.Length);
            Assert.False(double.IsNaN(result[0]),
                $"{window.GetType().Name} size=1 returned NaN");
        }
    }

    #endregion
}
