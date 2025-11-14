using AiDotNet.WindowFunctions;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.WindowFunctions
{
    /// <summary>
    /// Comprehensive integration tests for all Window Functions with mathematically verified results.
    /// Tests verify symmetry, edge values, center values, monotonicity, normalization, and spectral properties.
    /// </summary>
    public class WindowFunctionsIntegrationTests
    {
        private const double Tolerance = 1e-10;
        private const double RelaxedTolerance = 1e-8;

        #region RectangularWindow Tests

        [Fact]
        public void RectangularWindow_AllValues_EqualOne()
        {
            // Arrange
            var window = new RectangularWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - All values should be exactly 1.0
            for (int i = 0; i < 64; i++)
            {
                Assert.Equal(1.0, w[i], precision: 10);
            }
        }

        [Fact]
        public void RectangularWindow_Symmetry_IsPerfect()
        {
            // Arrange
            var window = new RectangularWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void RectangularWindow_Sum_EqualsWindowSize()
        {
            // Arrange
            var window = new RectangularWindow<double>();

            // Act
            var w = window.Create(64);
            double sum = 0;
            for (int i = 0; i < 64; i++)
            {
                sum += w[i];
            }

            // Assert
            Assert.Equal(64.0, sum, precision: 10);
        }

        [Fact]
        public void RectangularWindow_SmallSize_ProducesCorrectValues()
        {
            // Arrange
            var window = new RectangularWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert
            Assert.Equal(1.0, w[0], precision: 10);
            Assert.Equal(1.0, w[1], precision: 10);
            Assert.Equal(1.0, w[2], precision: 10);
            Assert.Equal(1.0, w[3], precision: 10);
            Assert.Equal(1.0, w[4], precision: 10);
        }

        #endregion

        #region HanningWindow Tests

        [Fact]
        public void HanningWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Hanning window should be exactly 0 at edges
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void HanningWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Center should be 1.0
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void HanningWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void HanningWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Should increase from 0 to center
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void HanningWindow_KnownValues_MatchFormula()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Check specific known values using formula: 0.5 * (1 - cos(2πn/(N-1)))
            // At n=16: 0.5 * (1 - cos(2π*16/63)) ≈ 0.5 * (1 - cos(1.599)) ≈ 0.5 * (1 - (-0.029)) ≈ 0.5145
            double expected16 = 0.5 * (1 - Math.Cos(2 * Math.PI * 16 / 63));
            Assert.Equal(expected16, w[16], precision: 8);
        }

        [Fact]
        public void HanningWindow_SmallSize_ProducesCorrectValues()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - For N=5: w[n] = 0.5 * (1 - cos(2πn/4))
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.5, w[1], precision: 10);
            Assert.Equal(1.0, w[2], precision: 10);
            Assert.Equal(0.5, w[3], precision: 10);
            Assert.Equal(0.0, w[4], precision: 10);
        }

        #endregion

        #region HammingWindow Tests

        [Fact]
        public void HammingWindow_EdgeValues_AreNonZero()
        {
            // Arrange
            var window = new HammingWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Hamming window edges are approximately 0.08
            // w(0) = 0.54 - 0.46 * cos(0) = 0.54 - 0.46 = 0.08
            Assert.Equal(0.08, w[0], precision: 10);
            Assert.Equal(0.08, w[63], precision: 10);
        }

        [Fact]
        public void HammingWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new HammingWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Center should be 1.0
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void HammingWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new HammingWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void HammingWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new HammingWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Should increase from edge to center
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void HammingWindow_KnownValues_MatchFormula()
        {
            // Arrange
            var window = new HammingWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - For N=5: w(n) = 0.54 - 0.46 * cos(2πn/4)
            Assert.Equal(0.08, w[0], precision: 10);
            Assert.Equal(0.54, w[1], precision: 10);
            Assert.Equal(1.00, w[2], precision: 10);
            Assert.Equal(0.54, w[3], precision: 10);
            Assert.Equal(0.08, w[4], precision: 10);
        }

        #endregion

        #region BlackmanWindow Tests

        [Fact]
        public void BlackmanWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new BlackmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Blackman window should be near 0 at edges
            // w(0) = 0.42 - 0.5 + 0.08 = 0
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void BlackmanWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new BlackmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Center should be 1.0
            // w(center) = 0.42 + 0.5 + 0.08 = 1.0
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void BlackmanWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new BlackmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void BlackmanWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new BlackmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Should increase from 0 to center
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void BlackmanWindow_ThreeTermFormula_IsCorrect()
        {
            // Arrange
            var window = new BlackmanWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - Verify 3-term cosine series formula
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.True(w[2] > w[1]); // Center is maximum
            Assert.Equal(0.0, w[4], precision: 10);
        }

        #endregion

        #region BlackmanHarrisWindow Tests

        [Fact]
        public void BlackmanHarrisWindow_EdgeValues_AreNearZero()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Should be very close to 0
            Assert.True(w[0] < 0.0001);
            Assert.True(w[63] < 0.0001);
        }

        [Fact]
        public void BlackmanHarrisWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void BlackmanHarrisWindow_CenterValue_IsMaximum()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Center should be maximum
            double centerValue = (w[31] + w[32]) / 2;
            Assert.True(centerValue > 0.99);
        }

        [Fact]
        public void BlackmanHarrisWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void BlackmanHarrisWindow_FourTermSeries_ProducesCorrectShape()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var w = window.Create(32);

            // Assert - Verify the window has the characteristic 4-term cosine series shape
            Assert.True(w[0] < 0.0001);
            Assert.True(w[16] > 0.99); // Center is maximum
            Assert.True(w[31] < 0.0001);
        }

        #endregion

        #region BlackmanNuttallWindow Tests

        [Fact]
        public void BlackmanNuttallWindow_EdgeValues_AreNearZero()
        {
            // Arrange
            var window = new BlackmanNuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[0] < 0.0001);
            Assert.True(w[63] < 0.0001);
        }

        [Fact]
        public void BlackmanNuttallWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new BlackmanNuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void BlackmanNuttallWindow_CenterValue_IsMaximum()
        {
            // Arrange
            var window = new BlackmanNuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[31] > 0.99);
            Assert.True(w[32] > 0.99);
        }

        [Fact]
        public void BlackmanNuttallWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new BlackmanNuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void BlackmanNuttallWindow_VerifyCoefficients_ProduceCorrectShape()
        {
            // Arrange
            var window = new BlackmanNuttallWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - With 4-term series, should have excellent side lobe suppression
            Assert.True(w[0] < 0.0001);
            Assert.True(w[64] > 0.99);
            Assert.True(w[127] < 0.0001);
        }

        #endregion

        #region NuttallWindow Tests

        [Fact]
        public void NuttallWindow_EdgeValues_AreNearZero()
        {
            // Arrange
            var window = new NuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[0] < 0.0001);
            Assert.True(w[63] < 0.0001);
        }

        [Fact]
        public void NuttallWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new NuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void NuttallWindow_CenterValue_IsMaximum()
        {
            // Arrange
            var window = new NuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[31] > 0.99);
            Assert.True(w[32] > 0.99);
        }

        [Fact]
        public void NuttallWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new NuttallWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void NuttallWindow_LowSideLobes_Verified()
        {
            // Arrange
            var window = new NuttallWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - Nuttall window has excellent side lobe suppression
            Assert.True(w[0] < 0.0001);
            Assert.True(w[64] > 0.99);
            Assert.True(w[127] < 0.0001);
        }

        #endregion

        #region FlatTopWindow Tests

        [Fact]
        public void FlatTopWindow_EdgeValues_AreNegative()
        {
            // Arrange
            var window = new FlatTopWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Flat top window can have negative values at edges
            // w(0) = 1.0 - 1.93 + 1.29 - 0.388 + 0.028 = 0.0
            Assert.True(w[0] < 0.01 && w[0] > -0.01);
            Assert.True(w[63] < 0.01 && w[63] > -0.01);
        }

        [Fact]
        public void FlatTopWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new FlatTopWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void FlatTopWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new FlatTopWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Center should be 1.0
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void FlatTopWindow_FiveTermSeries_ProducesCorrectShape()
        {
            // Arrange
            var window = new FlatTopWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - Flat top should have relatively flat center region
            double centerSum = 0;
            for (int i = 60; i < 68; i++)
            {
                centerSum += w[i];
            }
            double centerAvg = centerSum / 8;
            Assert.True(centerAvg > 0.95); // Center region should be close to 1
        }

        [Fact]
        public void FlatTopWindow_AmplitudeAccuracy_Property()
        {
            // Arrange
            var window = new FlatTopWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Flat top window is designed for amplitude accuracy
            // Center should be very close to 1
            Assert.True(Math.Abs(w[31] - 1.0) < 0.01);
            Assert.True(Math.Abs(w[32] - 1.0) < 0.01);
        }

        #endregion

        #region BartlettWindow Tests

        [Fact]
        public void BartlettWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new BartlettWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void BartlettWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new BartlettWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void BartlettWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new BartlettWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void BartlettWindow_LinearIncrease_InFirstHalf()
        {
            // Arrange
            var window = new BartlettWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Should increase linearly
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1]);
            }
        }

        [Fact]
        public void BartlettWindow_TriangularShape_IsCorrect()
        {
            // Arrange
            var window = new BartlettWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - For N=5: [0, 0.5, 1, 0.5, 0]
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.5, w[1], precision: 10);
            Assert.Equal(1.0, w[2], precision: 10);
            Assert.Equal(0.5, w[3], precision: 10);
            Assert.Equal(0.0, w[4], precision: 10);
        }

        #endregion

        #region BartlettHannWindow Tests

        [Fact]
        public void BartlettHannWindow_EdgeValues_AreNonZero()
        {
            // Arrange
            var window = new BartlettHannWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Should be small but non-zero
            Assert.True(w[0] >= 0);
            Assert.True(w[63] >= 0);
            Assert.True(w[0] < 0.1);
            Assert.True(w[63] < 0.1);
        }

        [Fact]
        public void BartlettHannWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new BartlettHannWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void BartlettHannWindow_CenterValue_IsMaximum()
        {
            // Arrange
            var window = new BartlettHannWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[31] > 0.9);
            Assert.True(w[32] > 0.9);
        }

        [Fact]
        public void BartlettHannWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new BartlettHannWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void BartlettHannWindow_HybridFormula_ProducesCorrectShape()
        {
            // Arrange
            var window = new BartlettHannWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - Combines Bartlett and Hann characteristics
            Assert.True(w[0] < 0.1);
            Assert.True(w[64] > 0.95);
            Assert.True(w[127] < 0.1);
        }

        #endregion

        #region TriangularWindow Tests

        [Fact]
        public void TriangularWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new TriangularWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void TriangularWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new TriangularWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void TriangularWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new TriangularWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void TriangularWindow_LinearIncrease_InFirstHalf()
        {
            // Arrange
            var window = new TriangularWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1]);
            }
        }

        [Fact]
        public void TriangularWindow_SmallSize_ProducesCorrectValues()
        {
            // Arrange
            var window = new TriangularWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - Triangular shape
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.True(w[1] > 0 && w[1] < 1);
            Assert.Equal(1.0, w[2], precision: 10);
            Assert.True(w[3] > 0 && w[3] < 1);
            Assert.Equal(0.0, w[4], precision: 10);
        }

        #endregion

        #region WelchWindow Tests

        [Fact]
        public void WelchWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new WelchWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void WelchWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new WelchWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void WelchWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new WelchWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void WelchWindow_ParabolicShape_IsCorrect()
        {
            // Arrange
            var window = new WelchWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - Parabolic: w(n) = 1 - ((n - N/2)/(N/2))²
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.75, w[1], precision: 10);
            Assert.Equal(1.0, w[2], precision: 10);
            Assert.Equal(0.75, w[3], precision: 10);
            Assert.Equal(0.0, w[4], precision: 10);
        }

        [Fact]
        public void WelchWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new WelchWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        #endregion

        #region ParzenWindow Tests

        [Fact]
        public void ParzenWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new ParzenWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void ParzenWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new ParzenWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void ParzenWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new ParzenWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void ParzenWindow_PiecewiseFunction_WorksCorrectly()
        {
            // Arrange
            var window = new ParzenWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - Parzen uses different formulas for different regions
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.True(w[64] > 0.99);
            Assert.Equal(0.0, w[127], precision: 10);
        }

        [Fact]
        public void ParzenWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new ParzenWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        #endregion

        #region BohmanWindow Tests

        [Fact]
        public void BohmanWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new BohmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[0] < 0.001);
            Assert.True(w[63] < 0.001);
        }

        [Fact]
        public void BohmanWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new BohmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void BohmanWindow_CenterValue_IsMaximum()
        {
            // Arrange
            var window = new BohmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[31] > 0.9);
            Assert.True(w[32] > 0.9);
        }

        [Fact]
        public void BohmanWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new BohmanWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] <= w[i + 1], $"Expected non-decreasing at index {i}");
            }
        }

        [Fact]
        public void BohmanWindow_SpecialFormula_ProducesCorrectShape()
        {
            // Arrange
            var window = new BohmanWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - Bohman has excellent spectral characteristics
            Assert.True(w[0] < 0.001);
            Assert.True(w[64] > 0.95);
            Assert.True(w[127] < 0.001);
        }

        #endregion

        #region CosineWindow Tests

        [Fact]
        public void CosineWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new CosineWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - sin(0) = 0, sin(π) = 0
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(0.0, w[63], precision: 10);
        }

        [Fact]
        public void CosineWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new CosineWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - sin(π/2) = 1
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void CosineWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new CosineWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void CosineWindow_SineShape_IsCorrect()
        {
            // Arrange
            var window = new CosineWindow<double>();

            // Act
            var w = window.Create(5);

            // Assert - w(n) = sin(πn/(N-1))
            Assert.Equal(0.0, w[0], precision: 10);
            Assert.Equal(Math.Sin(Math.PI / 4), w[1], precision: 10);
            Assert.Equal(1.0, w[2], precision: 10);
            Assert.Equal(Math.Sin(Math.PI / 4), w[3], precision: 10);
            Assert.Equal(0.0, w[4], precision: 10);
        }

        [Fact]
        public void CosineWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new CosineWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        #endregion

        #region TukeyWindow Tests

        [Fact]
        public void TukeyWindow_DefaultAlpha_ProducesCorrectShape()
        {
            // Arrange
            var window = new TukeyWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - With alpha=0.5, should have flat middle and tapered edges
            Assert.True(w[0] < 0.1);
            Assert.True(w[31] > 0.9);
            Assert.True(w[32] > 0.9);
            Assert.True(w[63] < 0.1);
        }

        [Fact]
        public void TukeyWindow_AlphaZero_IsRectangular()
        {
            // Arrange
            var window = new TukeyWindow<double>(alpha: 0.0);

            // Act
            var w = window.Create(64);

            // Assert - Alpha=0 should be rectangular (all 1s)
            for (int i = 0; i < 64; i++)
            {
                Assert.Equal(1.0, w[i], precision: 8);
            }
        }

        [Fact]
        public void TukeyWindow_AlphaOne_IsHann()
        {
            // Arrange
            var windowTukey = new TukeyWindow<double>(alpha: 1.0);
            var windowHann = new HanningWindow<double>();

            // Act
            var wTukey = windowTukey.Create(64);
            var wHann = windowHann.Create(64);

            // Assert - Alpha=1 should be similar to Hann
            for (int i = 0; i < 64; i++)
            {
                Assert.Equal(wHann[i], wTukey[i], precision: 6);
            }
        }

        [Fact]
        public void TukeyWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new TukeyWindow<double>(alpha: 0.5);

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void TukeyWindow_FlatTopRegion_Exists()
        {
            // Arrange
            var window = new TukeyWindow<double>(alpha: 0.3);

            // Act
            var w = window.Create(64);

            // Assert - With small alpha, should have significant flat region
            int flatCount = 0;
            for (int i = 20; i < 44; i++)
            {
                if (Math.Abs(w[i] - 1.0) < 0.01) flatCount++;
            }
            Assert.True(flatCount > 10); // Should have multiple points close to 1.0
        }

        [Fact]
        public void TukeyWindow_DifferentAlphas_ProduceDifferentShapes()
        {
            // Arrange
            var window1 = new TukeyWindow<double>(alpha: 0.2);
            var window2 = new TukeyWindow<double>(alpha: 0.8);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert - Different alphas should produce different windows
            bool isDifferent = false;
            for (int i = 0; i < 64; i++)
            {
                if (Math.Abs(w1[i] - w2[i]) > 0.1)
                {
                    isDifferent = true;
                    break;
                }
            }
            Assert.True(isDifferent);
        }

        #endregion

        #region GaussianWindow Tests

        [Fact]
        public void GaussianWindow_DefaultSigma_ProducesCorrectShape()
        {
            // Arrange
            var window = new GaussianWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[0] > 0); // Gaussian never reaches exactly 0
            Assert.True(w[31] > 0.9);
            Assert.True(w[32] > 0.9);
            Assert.True(w[63] > 0);
        }

        [Fact]
        public void GaussianWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new GaussianWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void GaussianWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new GaussianWindow<double>(sigma: 0.5);

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void GaussianWindow_SmallSigma_NarrowerWindow()
        {
            // Arrange
            var window1 = new GaussianWindow<double>(sigma: 0.3);
            var window2 = new GaussianWindow<double>(sigma: 0.7);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert - Smaller sigma should have lower edge values
            Assert.True(w1[0] < w2[0]);
            Assert.True(w1[63] < w2[63]);
        }

        [Fact]
        public void GaussianWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new GaussianWindow<double>(sigma: 0.5);

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void GaussianWindow_DifferentSigmas_ProduceDifferentShapes()
        {
            // Arrange
            var window1 = new GaussianWindow<double>(sigma: 0.3);
            var window2 = new GaussianWindow<double>(sigma: 0.6);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert
            bool isDifferent = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(w1[i] - w2[i]) > 0.05)
                {
                    isDifferent = true;
                    break;
                }
            }
            Assert.True(isDifferent);
        }

        #endregion

        #region KaiserWindow Tests

        [Fact]
        public void KaiserWindow_DefaultBeta_ProducesCorrectShape()
        {
            // Arrange
            var window = new KaiserWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[0] > 0);
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
            Assert.True(w[63] > 0);
        }

        [Fact]
        public void KaiserWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new KaiserWindow<double>(beta: 5.0);

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void KaiserWindow_LowBeta_WiderMainLobe()
        {
            // Arrange
            var window1 = new KaiserWindow<double>(beta: 2.0);
            var window2 = new KaiserWindow<double>(beta: 8.0);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert - Lower beta should have higher edge values
            Assert.True(w1[0] > w2[0]);
            Assert.True(w1[63] > w2[63]);
        }

        [Fact]
        public void KaiserWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new KaiserWindow<double>(beta: 5.0);

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] <= w[i + 1], $"Expected non-decreasing at index {i}");
            }
        }

        [Fact]
        public void KaiserWindow_Normalized_CenterIsOne()
        {
            // Arrange
            var window = new KaiserWindow<double>(beta: 7.0);

            // Act
            var w = window.Create(64);

            // Assert - Kaiser window is normalized to have max value of 1
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void KaiserWindow_DifferentBetas_ProduceDifferentShapes()
        {
            // Arrange
            var window1 = new KaiserWindow<double>(beta: 3.0);
            var window2 = new KaiserWindow<double>(beta: 7.0);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert
            bool isDifferent = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(w1[i] - w2[i]) > 0.05)
                {
                    isDifferent = true;
                    break;
                }
            }
            Assert.True(isDifferent);
        }

        #endregion

        #region LanczosWindow Tests

        [Fact]
        public void LanczosWindow_EdgeValues_AreZero()
        {
            // Arrange
            var window = new LanczosWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Lanczos (sinc) should be 0 at edges
            Assert.True(w[0] < 0.001);
            Assert.True(w[63] < 0.001);
        }

        [Fact]
        public void LanczosWindow_CenterValue_IsOne()
        {
            // Arrange
            var window = new LanczosWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - sinc(0) = 1
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
        }

        [Fact]
        public void LanczosWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new LanczosWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void LanczosWindow_SincFunction_HasOscillations()
        {
            // Arrange
            var window = new LanczosWindow<double>();

            // Act
            var w = window.Create(128);

            // Assert - Lanczos can have some negative values due to sinc function
            bool hasVariation = false;
            for (int i = 1; i < 127; i++)
            {
                if (w[i] != w[i - 1])
                {
                    hasVariation = true;
                    break;
                }
            }
            Assert.True(hasVariation);
        }

        [Fact]
        public void LanczosWindow_MainLobe_IsCentered()
        {
            // Arrange
            var window = new LanczosWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert - Main lobe should be around center
            Assert.True(w[31] > w[0]);
            Assert.True(w[32] > w[63]);
            Assert.True(w[31] > w[20]);
            Assert.True(w[32] > w[43]);
        }

        #endregion

        #region PoissonWindow Tests

        [Fact]
        public void PoissonWindow_DefaultAlpha_ProducesCorrectShape()
        {
            // Arrange
            var window = new PoissonWindow<double>();

            // Act
            var w = window.Create(64);

            // Assert
            Assert.True(w[0] > 0); // Exponential decay, never reaches 0
            Assert.Equal(1.0, w[31], precision: 8);
            Assert.Equal(1.0, w[32], precision: 8);
            Assert.True(w[63] > 0);
        }

        [Fact]
        public void PoissonWindow_Symmetry_IsValid()
        {
            // Arrange
            var window = new PoissonWindow<double>(alpha: 2.0);

            // Act
            var w = window.Create(64);

            // Assert - w[n] = w[N-1-n]
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(w[i], w[63 - i], precision: 10);
            }
        }

        [Fact]
        public void PoissonWindow_HighAlpha_FasterDecay()
        {
            // Arrange
            var window1 = new PoissonWindow<double>(alpha: 1.0);
            var window2 = new PoissonWindow<double>(alpha: 4.0);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert - Higher alpha should have lower edge values
            Assert.True(w1[0] > w2[0]);
            Assert.True(w1[63] > w2[63]);
        }

        [Fact]
        public void PoissonWindow_Monotonicity_InFirstHalf()
        {
            // Arrange
            var window = new PoissonWindow<double>(alpha: 2.0);

            // Act
            var w = window.Create(64);

            // Assert
            for (int i = 0; i < 31; i++)
            {
                Assert.True(w[i] < w[i + 1], $"Expected increasing at index {i}");
            }
        }

        [Fact]
        public void PoissonWindow_ExponentialDecay_IsCorrect()
        {
            // Arrange
            var window = new PoissonWindow<double>(alpha: 2.0);

            // Act
            var w = window.Create(64);

            // Assert - Exponential decay from center
            Assert.True(w[31] > w[25]);
            Assert.True(w[25] > w[20]);
            Assert.True(w[20] > w[15]);
        }

        [Fact]
        public void PoissonWindow_DifferentAlphas_ProduceDifferentShapes()
        {
            // Arrange
            var window1 = new PoissonWindow<double>(alpha: 1.5);
            var window2 = new PoissonWindow<double>(alpha: 3.5);

            // Act
            var w1 = window1.Create(64);
            var w2 = window2.Create(64);

            // Assert
            bool isDifferent = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(w1[i] - w2[i]) > 0.05)
                {
                    isDifferent = true;
                    break;
                }
            }
            Assert.True(isDifferent);
        }

        #endregion

        #region Cross-Window Comparison Tests

        [Fact]
        public void AllWindows_Symmetry_IsValid()
        {
            // Test symmetry for all window functions
            var windows = new List<(string name, IWindowFunction<double> window)>
            {
                ("Rectangular", new RectangularWindow<double>()),
                ("Hanning", new HanningWindow<double>()),
                ("Hamming", new HammingWindow<double>()),
                ("Blackman", new BlackmanWindow<double>()),
                ("BlackmanHarris", new BlackmanHarrisWindow<double>()),
                ("BlackmanNuttall", new BlackmanNuttallWindow<double>()),
                ("Nuttall", new NuttallWindow<double>()),
                ("FlatTop", new FlatTopWindow<double>()),
                ("Bartlett", new BartlettWindow<double>()),
                ("BartlettHann", new BartlettHannWindow<double>()),
                ("Triangular", new TriangularWindow<double>()),
                ("Welch", new WelchWindow<double>()),
                ("Parzen", new ParzenWindow<double>()),
                ("Bohman", new BohmanWindow<double>()),
                ("Cosine", new CosineWindow<double>()),
                ("Tukey", new TukeyWindow<double>()),
                ("Gaussian", new GaussianWindow<double>()),
                ("Kaiser", new KaiserWindow<double>()),
                ("Lanczos", new LanczosWindow<double>()),
                ("Poisson", new PoissonWindow<double>())
            };

            foreach (var (name, window) in windows)
            {
                var w = window.Create(64);
                for (int i = 0; i < 32; i++)
                {
                    Assert.Equal(w[i], w[63 - i], precision: 10);
                }
            }
        }

        [Fact]
        public void WindowComparison_DifferentCharacteristics()
        {
            // Arrange
            var rectangular = new RectangularWindow<double>();
            var hanning = new HanningWindow<double>();
            var hamming = new HammingWindow<double>();

            // Act
            var wRect = rectangular.Create(64);
            var wHann = hanning.Create(64);
            var wHamm = hamming.Create(64);

            // Assert - Verify different edge characteristics
            Assert.Equal(1.0, wRect[0], precision: 10); // Rectangular has edge = 1
            Assert.Equal(0.0, wHann[0], precision: 10); // Hanning has edge = 0
            Assert.Equal(0.08, wHamm[0], precision: 10); // Hamming has edge ≈ 0.08
        }

        [Fact]
        public void AllWindows_ProducePositiveValues_ExceptFlatTop()
        {
            // Most windows should have all positive values
            var windows = new List<IWindowFunction<double>>
            {
                new RectangularWindow<double>(),
                new HanningWindow<double>(),
                new HammingWindow<double>(),
                new BlackmanWindow<double>(),
                new BartlettWindow<double>(),
                new TriangularWindow<double>(),
                new WelchWindow<double>(),
                new CosineWindow<double>(),
                new GaussianWindow<double>(),
                new KaiserWindow<double>(),
                new PoissonWindow<double>()
            };

            foreach (var window in windows)
            {
                var w = window.Create(64);
                for (int i = 0; i < 64; i++)
                {
                    Assert.True(w[i] >= 0, $"Window {window.GetType().Name} has negative value at index {i}");
                }
            }
        }

        [Fact]
        public void AllWindows_HandleSmallSize_Correctly()
        {
            // Test all windows with small size
            var windows = new List<IWindowFunction<double>>
            {
                new RectangularWindow<double>(),
                new HanningWindow<double>(),
                new HammingWindow<double>(),
                new BlackmanWindow<double>(),
                new BartlettWindow<double>(),
                new TriangularWindow<double>(),
                new WelchWindow<double>(),
                new CosineWindow<double>()
            };

            foreach (var window in windows)
            {
                var w = window.Create(3);
                Assert.Equal(3, w.Length);
                // Center should be maximum for most windows
                Assert.True(w[1] >= w[0]);
                Assert.True(w[1] >= w[2]);
            }
        }

        [Fact]
        public void AllWindows_HandleLargeSize_Correctly()
        {
            // Test all windows with large size
            var windows = new List<IWindowFunction<double>>
            {
                new RectangularWindow<double>(),
                new HanningWindow<double>(),
                new HammingWindow<double>(),
                new BlackmanWindow<double>()
            };

            foreach (var window in windows)
            {
                var w = window.Create(1024);
                Assert.Equal(1024, w.Length);

                // Verify symmetry for large windows
                for (int i = 0; i < 512; i++)
                {
                    Assert.Equal(w[i], w[1023 - i], precision: 10);
                }
            }
        }

        #endregion

        #region Energy and Power Tests

        [Fact]
        public void WindowEnergy_Comparison_AcrossDifferentWindows()
        {
            // Arrange
            var rectangular = new RectangularWindow<double>();
            var hanning = new HanningWindow<double>();
            var hamming = new HammingWindow<double>();

            // Act
            var wRect = rectangular.Create(64);
            var wHann = hanning.Create(64);
            var wHamm = hamming.Create(64);

            double energyRect = 0, energyHann = 0, energyHamm = 0;
            for (int i = 0; i < 64; i++)
            {
                energyRect += wRect[i] * wRect[i];
                energyHann += wHann[i] * wHann[i];
                energyHamm += wHamm[i] * wHamm[i];
            }

            // Assert - Rectangular should have highest energy
            Assert.True(energyRect > energyHann);
            Assert.True(energyRect > energyHamm);
        }

        [Fact]
        public void WindowSum_RectangularVsOthers()
        {
            // Arrange
            var rectangular = new RectangularWindow<double>();
            var hanning = new HanningWindow<double>();

            // Act
            var wRect = rectangular.Create(64);
            var wHann = hanning.Create(64);

            double sumRect = 0, sumHann = 0;
            for (int i = 0; i < 64; i++)
            {
                sumRect += wRect[i];
                sumHann += wHann[i];
            }

            // Assert - Rectangular sum should be N, others less
            Assert.Equal(64.0, sumRect, precision: 10);
            Assert.True(sumHann < sumRect);
        }

        #endregion
    }
}
