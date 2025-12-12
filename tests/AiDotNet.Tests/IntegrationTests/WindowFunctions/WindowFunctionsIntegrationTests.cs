using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.WindowFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WindowFunctions;

/// <summary>
/// Integration tests for window function classes.
/// Tests window creation and properties for various window functions.
/// </summary>
public class WindowFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Rectangular Window Tests

    [Fact]
    public void RectangularWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new RectangularWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void RectangularWindow_Create_AllValuesAreOne()
    {
        // Arrange
        var window = new RectangularWindow<double>();

        // Act
        var result = window.Create(32);

        // Assert
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(1.0, result[i], Tolerance);
        }
    }

    #endregion

    #region Hamming Window Tests

    [Fact]
    public void HammingWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new HammingWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void HammingWindow_Create_SymmetricAroundCenter()
    {
        // Arrange
        var window = new HammingWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert - Check symmetry
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(result[i], result[63 - i], Tolerance);
        }
    }

    [Fact]
    public void HammingWindow_Create_ValuesInValidRange()
    {
        // Arrange
        var window = new HammingWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert - Hamming window values should be between 0.08 and 1.0
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0.0);
            Assert.True(result[i] <= 1.1);
        }
    }

    [Fact]
    public void HammingWindow_Create_CenterIsMaximum()
    {
        // Arrange
        var window = new HammingWindow<double>();

        // Act
        var result = window.Create(65);

        // Assert - Center should be maximum
        double max = double.MinValue;
        int maxIndex = 0;
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] > max)
            {
                max = result[i];
                maxIndex = i;
            }
        }
        Assert.Equal(32, maxIndex);
    }

    #endregion

    #region Hanning Window Tests

    [Fact]
    public void HanningWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new HanningWindow<double>();

        // Act
        var result = window.Create(128);

        // Assert
        Assert.Equal(128, result.Length);
    }

    [Fact]
    public void HanningWindow_Create_EdgesAreZero()
    {
        // Arrange
        var window = new HanningWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert - Hanning window should be zero at edges
        Assert.True(result[0] < 0.01);
    }

    [Fact]
    public void HanningWindow_Create_SymmetricAroundCenter()
    {
        // Arrange
        var window = new HanningWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert - Check symmetry
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(result[i], result[63 - i], Tolerance);
        }
    }

    #endregion

    #region Blackman Window Tests

    [Fact]
    public void BlackmanWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new BlackmanWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void BlackmanWindow_Create_ValuesInValidRange()
    {
        // Arrange
        var window = new BlackmanWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= -0.01);
            Assert.True(result[i] <= 1.01);
        }
    }

    #endregion

    #region Bartlett Window Tests

    [Fact]
    public void BartlettWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new BartlettWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void BartlettWindow_Create_TriangularShape()
    {
        // Arrange
        var window = new BartlettWindow<double>();

        // Act
        var result = window.Create(65);

        // Assert - Center should be 1.0, edges should be 0
        Assert.Equal(1.0, result[32], Tolerance);
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.True(result[64] < 0.05);
    }

    #endregion

    #region Triangular Window Tests

    [Fact]
    public void TriangularWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new TriangularWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void TriangularWindow_Create_Symmetric()
    {
        // Arrange
        var window = new TriangularWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(result[i], result[63 - i], Tolerance);
        }
    }

    #endregion

    #region Gaussian Window Tests

    [Fact]
    public void GaussianWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new GaussianWindow<double>(sigma: 0.4);

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void GaussianWindow_Create_CenterIsMaximum()
    {
        // Arrange
        var window = new GaussianWindow<double>(sigma: 0.4);

        // Act
        var result = window.Create(65);

        // Assert - Center should have maximum value
        double max = double.MinValue;
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] > max) max = result[i];
        }
        Assert.Equal(result[32], max, Tolerance);
    }

    [Fact]
    public void GaussianWindow_Create_ValuesDecreaseFromCenter()
    {
        // Arrange
        var window = new GaussianWindow<double>(sigma: 0.4);

        // Act
        var result = window.Create(65);

        // Assert - Values should decrease from center
        for (int i = 1; i <= 32; i++)
        {
            Assert.True(result[32 - i] <= result[32 - i + 1]);
            Assert.True(result[32 + i] <= result[32 + i - 1]);
        }
    }

    #endregion

    #region Kaiser Window Tests

    [Fact]
    public void KaiserWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new KaiserWindow<double>(beta: 5.0);

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void KaiserWindow_Create_Symmetric()
    {
        // Arrange
        var window = new KaiserWindow<double>(beta: 5.0);

        // Act
        var result = window.Create(64);

        // Assert
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(result[i], result[63 - i], Tolerance);
        }
    }

    [Fact]
    public void KaiserWindow_DifferentBeta_ProducesDifferentWindows()
    {
        // Arrange
        var window1 = new KaiserWindow<double>(beta: 2.0);
        var window2 = new KaiserWindow<double>(beta: 8.0);

        // Act
        var result1 = window1.Create(64);
        var result2 = window2.Create(64);

        // Assert - Different beta should give different values
        bool different = false;
        for (int i = 0; i < 64; i++)
        {
            if (Math.Abs(result1[i] - result2[i]) > Tolerance)
            {
                different = true;
                break;
            }
        }
        Assert.True(different);
    }

    #endregion

    #region BlackmanHarris Window Tests

    [Fact]
    public void BlackmanHarrisWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new BlackmanHarrisWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void BlackmanHarrisWindow_Create_ValuesInValidRange()
    {
        // Arrange
        var window = new BlackmanHarrisWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= -0.01);
            Assert.True(result[i] <= 1.01);
        }
    }

    #endregion

    #region FlatTop Window Tests

    [Fact]
    public void FlatTopWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new FlatTopWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void FlatTopWindow_Create_HasFlatTop()
    {
        // Arrange
        var window = new FlatTopWindow<double>();

        // Act
        var result = window.Create(65);

        // Assert - Center region should be relatively flat
        double center = result[32];
        Assert.True(Math.Abs(result[31] - center) < 0.1);
        Assert.True(Math.Abs(result[33] - center) < 0.1);
    }

    #endregion

    #region Nuttall Window Tests

    [Fact]
    public void NuttallWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new NuttallWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void NuttallWindow_Create_Symmetric()
    {
        // Arrange
        var window = new NuttallWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(result[i], result[63 - i], Tolerance);
        }
    }

    #endregion

    #region Tukey Window Tests

    [Fact]
    public void TukeyWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new TukeyWindow<double>(alpha: 0.5);

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void TukeyWindow_AlphaZero_IsRectangular()
    {
        // Arrange
        var window = new TukeyWindow<double>(alpha: 0.0);

        // Act
        var result = window.Create(64);

        // Assert - Should be all ones like rectangular
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(1.0, result[i], Tolerance);
        }
    }

    [Fact]
    public void TukeyWindow_AlphaOne_IsHanning()
    {
        // Arrange
        var tukeyWindow = new TukeyWindow<double>(alpha: 1.0);
        var hanningWindow = new HanningWindow<double>();

        // Act
        var tukeyResult = tukeyWindow.Create(64);
        var hanningResult = hanningWindow.Create(64);

        // Assert - Should be similar to Hanning
        for (int i = 0; i < 64; i++)
        {
            Assert.Equal(hanningResult[i], tukeyResult[i], 0.01);
        }
    }

    #endregion

    #region Welch Window Tests

    [Fact]
    public void WelchWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new WelchWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void WelchWindow_Create_ParabolicShape()
    {
        // Arrange
        var window = new WelchWindow<double>();

        // Act
        var result = window.Create(65);

        // Assert - Center should be 1.0, edges should be 0
        Assert.Equal(1.0, result[32], Tolerance);
        Assert.Equal(0.0, result[0], Tolerance);
    }

    #endregion

    #region Parzen Window Tests

    [Fact]
    public void ParzenWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new ParzenWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    [Fact]
    public void ParzenWindow_Create_ValuesInValidRange()
    {
        // Arrange
        var window = new ParzenWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0.0);
            Assert.True(result[i] <= 1.01);
        }
    }

    #endregion

    #region Lanczos Window Tests

    [Fact]
    public void LanczosWindow_Create_ReturnsCorrectSize()
    {
        // Arrange
        var window = new LanczosWindow<double>();

        // Act
        var result = window.Create(64);

        // Assert
        Assert.Equal(64, result.Length);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllWindowFunctions_Create_ReturnCorrectSize()
    {
        // Arrange
        var windows = new IWindowFunction<double>[]
        {
            new RectangularWindow<double>(),
            new HammingWindow<double>(),
            new HanningWindow<double>(),
            new BlackmanWindow<double>(),
            new BartlettWindow<double>(),
            new TriangularWindow<double>(),
            new GaussianWindow<double>(sigma: 0.4),
            new KaiserWindow<double>(beta: 5.0),
            new BlackmanHarrisWindow<double>(),
            new FlatTopWindow<double>(),
            new NuttallWindow<double>(),
            new TukeyWindow<double>(alpha: 0.5),
            new WelchWindow<double>(),
            new ParzenWindow<double>(),
            new LanczosWindow<double>()
        };

        // Act & Assert
        foreach (var window in windows)
        {
            var result = window.Create(128);
            Assert.Equal(128, result.Length);
        }
    }

    [Fact]
    public void AllWindowFunctions_Create_NoNaNValues()
    {
        // Arrange
        var windows = new IWindowFunction<double>[]
        {
            new RectangularWindow<double>(),
            new HammingWindow<double>(),
            new HanningWindow<double>(),
            new BlackmanWindow<double>(),
            new BartlettWindow<double>(),
            new TriangularWindow<double>(),
            new GaussianWindow<double>(sigma: 0.4),
            new KaiserWindow<double>(beta: 5.0),
            new BlackmanHarrisWindow<double>()
        };

        // Act & Assert
        foreach (var window in windows)
        {
            var result = window.Create(64);
            for (int i = 0; i < result.Length; i++)
            {
                Assert.False(double.IsNaN(result[i]));
                Assert.False(double.IsInfinity(result[i]));
            }
        }
    }

    [Fact]
    public void WindowFunctions_SmallSize_DoesNotThrow()
    {
        // Arrange
        var window = new HammingWindow<double>();

        // Act & Assert - Should handle small sizes
        var result1 = window.Create(1);
        Assert.Equal(1, result1.Length);

        var result2 = window.Create(2);
        Assert.Equal(2, result2.Length);
    }

    [Fact]
    public void WindowFunctions_LargeSize_HandlesCorrectly()
    {
        // Arrange
        var window = new HammingWindow<double>();

        // Act
        var result = window.Create(8192);

        // Assert
        Assert.Equal(8192, result.Length);
        Assert.False(double.IsNaN(result[0]));
        Assert.False(double.IsNaN(result[8191]));
    }

    #endregion
}
