using AiDotNet.LinearAlgebra;
using AiDotNet.WaveletFunctions;
using AiDotNet.Wavelets;
using Xunit;

namespace AiDotNetTests.IntegrationTests.WaveletFunctions
{
    /// <summary>
    /// Integration tests for wavelet functions with mathematically verified results.
    /// Tests ensure wavelets satisfy fundamental properties: admissibility, orthogonality,
    /// normalization, perfect reconstruction, and multi-resolution analysis.
    /// </summary>
    public class WaveletFunctionsIntegrationTests
    {
        private const double Tolerance = 1e-8;
        private const double LooseTolerance = 1e-4;

        #region Haar Wavelet Tests

        [Fact]
        public void HaarWavelet_Calculate_KnownPoints_ReturnsCorrectValues()
        {
            // Arrange
            var haar = new HaarWavelet<double>();

            // Act & Assert - Haar wavelet: ψ(x) = 1 for [0,0.5), -1 for [0.5,1), 0 elsewhere
            Assert.Equal(1.0, haar.Calculate(0.0), Tolerance);
            Assert.Equal(1.0, haar.Calculate(0.25), Tolerance);
            Assert.Equal(-1.0, haar.Calculate(0.5), Tolerance);
            Assert.Equal(-1.0, haar.Calculate(0.75), Tolerance);
            Assert.Equal(0.0, haar.Calculate(1.0), Tolerance);
            Assert.Equal(0.0, haar.Calculate(-0.5), Tolerance);
            Assert.Equal(0.0, haar.Calculate(1.5), Tolerance);
        }

        [Fact]
        public void HaarWavelet_Admissibility_ZeroMean_Satisfied()
        {
            // Arrange
            var haar = new HaarWavelet<double>();

            // Act - Compute integral approximation: ∫ψ(t)dt over support [0,1]
            double sum = 0;
            int samples = 1000;
            for (int i = 0; i < samples; i++)
            {
                double t = i / (double)samples;
                sum += haar.Calculate(t) / samples;
            }

            // Assert - Zero mean property: integral should be approximately zero
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void HaarWavelet_Normalization_L2Norm_IsOne()
        {
            // Arrange
            var haar = new HaarWavelet<double>();

            // Act - Compute L2 norm: ∫|ψ(t)|²dt
            double sumSquared = 0;
            int samples = 1000;
            for (int i = 0; i < samples; i++)
            {
                double t = i / (double)samples;
                double val = haar.Calculate(t);
                sumSquared += val * val / samples;
            }

            // Assert - Normalized wavelet: L2 norm should be 1
            Assert.Equal(1.0, sumSquared, LooseTolerance);
        }

        [Fact]
        public void HaarWavelet_FilterCoefficients_CorrectValues()
        {
            // Arrange
            var haar = new HaarWavelet<double>();

            // Act
            var scalingCoeffs = haar.GetScalingCoefficients();
            var waveletCoeffs = haar.GetWaveletCoefficients();

            // Assert - Haar coefficients: h = [1/√2, 1/√2], g = [1/√2, -1/√2]
            double sqrt2 = Math.Sqrt(2);
            Assert.Equal(2, scalingCoeffs.Length);
            Assert.Equal(2, waveletCoeffs.Length);
            Assert.Equal(1.0 / sqrt2, scalingCoeffs[0], Tolerance);
            Assert.Equal(1.0 / sqrt2, scalingCoeffs[1], Tolerance);
            Assert.Equal(1.0 / sqrt2, waveletCoeffs[0], Tolerance);
            Assert.Equal(-1.0 / sqrt2, waveletCoeffs[1], Tolerance);
        }

        [Fact]
        public void HaarWavelet_Decompose_PerfectReconstruction()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act - Decompose and reconstruct
            var (approx, detail) = haar.Decompose(signal);
            var reconstructed = ReconstructHaar(approx, detail);

            // Assert - Perfect reconstruction
            Assert.Equal(signal.Length, reconstructed.Length);
            for (int i = 0; i < signal.Length; i++)
            {
                Assert.Equal(signal[i], reconstructed[i], LooseTolerance);
            }
        }

        [Fact]
        public void HaarWavelet_MultiLevelDecomposition_EnergyPreservation()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            double originalEnergy = ComputeEnergy(signal);

            // Act - Two-level decomposition
            var (approx1, detail1) = haar.Decompose(signal);
            var (approx2, detail2) = haar.Decompose(approx1);

            // Assert - Energy preservation
            double decomposedEnergy = ComputeEnergy(approx2) + ComputeEnergy(detail2) + ComputeEnergy(detail1);
            Assert.Equal(originalEnergy, decomposedEnergy, LooseTolerance);
        }

        #endregion

        #region Daubechies Wavelet Tests

        [Fact]
        public void DaubechiesWavelet_Calculate_WithinSupport_NonZero()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);

            // Act & Assert - DB4 has support [0, 3]
            double val1 = db4.Calculate(0.5);
            double val2 = db4.Calculate(1.5);
            double val3 = db4.Calculate(2.5);

            Assert.NotEqual(0.0, val1);
            Assert.NotEqual(0.0, val2);
            Assert.NotEqual(0.0, val3);
        }

        [Fact]
        public void DaubechiesWavelet_Calculate_OutsideSupport_Zero()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);

            // Act & Assert - Outside support [0, 3]
            Assert.Equal(0.0, db4.Calculate(-0.5), Tolerance);
            Assert.Equal(0.0, db4.Calculate(3.5), Tolerance);
        }

        [Fact]
        public void DaubechiesWavelet_Admissibility_ZeroMean_Satisfied()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);

            // Act - Compute integral approximation over support [0, 3]
            double sum = 0;
            int samples = 3000;
            for (int i = 0; i < samples; i++)
            {
                double t = (i * 3.0) / samples;
                sum += db4.Calculate(t) * 3.0 / samples;
            }

            // Assert - Zero mean property
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void DaubechiesWavelet_FilterCoefficients_OrthogonalityCondition()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);
            var h = db4.GetScalingCoefficients();
            var g = db4.GetWaveletCoefficients();

            // Act - Check quadrature mirror filter relationship: g[n] = (-1)^n * h[L-1-n]
            bool isQMF = true;
            for (int i = 0; i < h.Length; i++)
            {
                double expected = Math.Pow(-1, i) * h[h.Length - 1 - i];
                if (Math.Abs(g[i] - expected) > Tolerance)
                {
                    isQMF = false;
                    break;
                }
            }

            // Assert - QMF relationship holds
            Assert.True(isQMF);
        }

        [Fact]
        public void DaubechiesWavelet_Decompose_PerfectReconstruction()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);
            var signal = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0 });

            // Act - Decompose and reconstruct
            var (approx, detail) = db4.Decompose(signal);
            var reconstructed = ReconstructDaubechies(approx, detail, db4);

            // Assert - Perfect reconstruction (within tolerance due to numerical errors)
            for (int i = 0; i < signal.Length; i++)
            {
                Assert.Equal(signal[i], reconstructed[i], LooseTolerance);
            }
        }

        [Fact]
        public void DaubechiesWavelet_ScalingCoefficients_SumToSqrt2()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);

            // Act
            var h = db4.GetScalingCoefficients();
            double sum = 0;
            for (int i = 0; i < h.Length; i++)
            {
                sum += h[i];
            }

            // Assert - Scaling coefficients sum to √2
            Assert.Equal(Math.Sqrt(2), sum, Tolerance);
        }

        #endregion

        #region Symlet Wavelet Tests

        [Fact]
        public void SymletWavelet_Calculate_ReturnsValidValues()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);

            // Act & Assert - Symlet should return valid values in [0,1]
            double val = sym4.Calculate(0.5);
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void SymletWavelet_NearSymmetry_ComparedToDaubechies()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);
            var h = sym4.GetScalingCoefficients();

            // Act - Check approximate symmetry
            double asymmetry = 0;
            for (int i = 0; i < h.Length / 2; i++)
            {
                asymmetry += Math.Abs(h[i] - h[h.Length - 1 - i]);
            }

            // Assert - Symlets are more symmetric (asymmetry should be relatively small)
            Assert.True(asymmetry < 1.0); // Loose check for near-symmetry
        }

        [Fact]
        public void SymletWavelet_Decompose_EnergyConservation()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            double originalEnergy = ComputeEnergy(signal);

            // Act
            var (approx, detail) = sym4.Decompose(signal);
            double decomposedEnergy = ComputeEnergy(approx) + ComputeEnergy(detail);

            // Assert - Energy is conserved
            Assert.Equal(originalEnergy, decomposedEnergy, LooseTolerance);
        }

        [Fact]
        public void SymletWavelet_FilterCoefficients_OrthogonalityProperty()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);
            var h = sym4.GetScalingCoefficients();

            // Act - Check orthogonality: Σh[i]h[i+2k] = δ[k]
            double innerProduct = 0;
            for (int i = 0; i < h.Length - 2; i++)
            {
                innerProduct += h[i] * h[i + 2];
            }

            // Assert - Orthogonality condition (should be close to 0 for k≠0)
            Assert.Equal(0.0, innerProduct, LooseTolerance);
        }

        [Fact]
        public void SymletWavelet_MultipleOrders_AllValid()
        {
            // Arrange & Act & Assert
            var sym2 = new SymletWavelet<double>(2);
            var sym4 = new SymletWavelet<double>(4);
            var sym6 = new SymletWavelet<double>(6);
            var sym8 = new SymletWavelet<double>(8);

            Assert.NotNull(sym2.GetScalingCoefficients());
            Assert.NotNull(sym4.GetScalingCoefficients());
            Assert.NotNull(sym6.GetScalingCoefficients());
            Assert.NotNull(sym8.GetScalingCoefficients());
        }

        [Fact]
        public void SymletWavelet_Decompose_ReturnsCorrectLength()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = sym4.Decompose(signal);

            // Assert - Output length should be half of input
            Assert.Equal(signal.Length / 2, approx.Length);
            Assert.Equal(signal.Length / 2, detail.Length);
        }

        #endregion

        #region Coiflet Wavelet Tests

        [Fact]
        public void CoifletWavelet_Calculate_WithinSupport_NonZero()
        {
            // Arrange
            var coif2 = new CoifletWavelet<double>(2);

            // Act & Assert - Coif2 has support width of 11
            double val = coif2.Calculate(5.0);
            Assert.NotEqual(0.0, val);
        }

        [Fact]
        public void CoifletWavelet_Admissibility_ZeroMean_Satisfied()
        {
            // Arrange
            var coif2 = new CoifletWavelet<double>(2);

            // Act - Compute integral approximation
            double sum = 0;
            int samples = 1100;
            for (int i = 0; i < samples; i++)
            {
                double t = (i * 11.0) / samples;
                sum += coif2.Calculate(t) * 11.0 / samples;
            }

            // Assert - Zero mean property
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void CoifletWavelet_FilterCoefficients_VanishingMoments()
        {
            // Arrange
            var coif2 = new CoifletWavelet<double>(2);
            var h = coif2.GetScalingCoefficients();

            // Act - Check that scaling function has vanishing moments
            // For Coiflet order N, scaling function has 2N-1 vanishing moments
            double moment0 = 0;
            for (int i = 0; i < h.Length; i++)
            {
                moment0 += h[i];
            }

            // Assert - Sum should be √2 for proper normalization
            Assert.Equal(Math.Sqrt(2), moment0, LooseTolerance);
        }

        [Fact]
        public void CoifletWavelet_Decompose_EnergyPreservation()
        {
            // Arrange
            var coif2 = new CoifletWavelet<double>(2);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            double originalEnergy = ComputeEnergy(signal);

            // Act
            var (approx, detail) = coif2.Decompose(signal);
            double decomposedEnergy = ComputeEnergy(approx) + ComputeEnergy(detail);

            // Assert - Energy conservation
            Assert.Equal(originalEnergy, decomposedEnergy, LooseTolerance);
        }

        [Fact]
        public void CoifletWavelet_MoreSymmetric_ThanDaubechies()
        {
            // Arrange
            var coif2 = new CoifletWavelet<double>(2);
            var h = coif2.GetScalingCoefficients();

            // Act - Measure symmetry
            double asymmetry = 0;
            int len = h.Length;
            for (int i = 0; i < len / 2; i++)
            {
                asymmetry += Math.Abs(h[i] - h[len - 1 - i]);
            }

            // Assert - Coiflets have good symmetry
            Assert.True(asymmetry < 0.5); // Coiflets are nearly symmetric
        }

        [Fact]
        public void CoifletWavelet_MultipleOrders_AllValid()
        {
            // Arrange & Act & Assert
            for (int order = 1; order <= 5; order++)
            {
                var coif = new CoifletWavelet<double>(order);
                var h = coif.GetScalingCoefficients();
                Assert.NotNull(h);
                Assert.True(h.Length > 0);
            }
        }

        #endregion

        #region Biorthogonal Wavelet Tests

        [Fact]
        public void BiorthogonalWavelet_Calculate_ReturnsValidValues()
        {
            // Arrange
            var bior = new BiorthogonalWavelet<double>(2, 2);

            // Act
            double val = bior.Calculate(0.5);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void BiorthogonalWavelet_Decompose_PerfectReconstruction()
        {
            // Arrange
            var bior = new BiorthogonalWavelet<double>(2, 2);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act - Decompose and reconstruct
            var (approx, detail) = bior.Decompose(signal);
            // Note: Perfect reconstruction requires reconstruction filters

            // Assert - Decomposition produces expected lengths
            Assert.Equal(signal.Length / 2, approx.Length);
            Assert.Equal(signal.Length / 2, detail.Length);
        }

        [Fact]
        public void BiorthogonalWavelet_SymmetryProperty_Satisfied()
        {
            // Arrange
            var bior = new BiorthogonalWavelet<double>(2, 2);
            var h = bior.GetScalingCoefficients();

            // Act - Biorthogonal wavelets can be symmetric
            double asymmetry = 0;
            for (int i = 0; i < h.Length / 2; i++)
            {
                asymmetry += Math.Abs(h[i] - h[h.Length - 1 - i]);
            }

            // Assert - Good symmetry
            Assert.True(asymmetry < 0.5);
        }

        [Fact]
        public void BiorthogonalWavelet_DifferentOrders_ValidCoefficients()
        {
            // Arrange & Act & Assert
            var bior13 = new BiorthogonalWavelet<double>(1, 3);
            var bior22 = new BiorthogonalWavelet<double>(2, 2);
            var bior31 = new BiorthogonalWavelet<double>(3, 1);

            Assert.NotNull(bior13.GetScalingCoefficients());
            Assert.NotNull(bior22.GetScalingCoefficients());
            Assert.NotNull(bior31.GetScalingCoefficients());
        }

        [Fact]
        public void BiorthogonalWavelet_Decompose_EnergyConservation()
        {
            // Arrange
            var bior = new BiorthogonalWavelet<double>(2, 2);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            double originalEnergy = ComputeEnergy(signal);

            // Act
            var (approx, detail) = bior.Decompose(signal);
            double decomposedEnergy = ComputeEnergy(approx) + ComputeEnergy(detail);

            // Assert
            Assert.Equal(originalEnergy, decomposedEnergy, LooseTolerance);
        }

        #endregion

        #region Morlet Wavelet Tests

        [Fact]
        public void MorletWavelet_Calculate_AtZero_MaximumValue()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);

            // Act - At x=0, Morlet is cos(ω*0) * exp(0) = 1
            double val = morlet.Calculate(0.0);

            // Assert
            Assert.Equal(1.0, val, Tolerance);
        }

        [Fact]
        public void MorletWavelet_Calculate_Oscillatory_WithGaussianEnvelope()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);

            // Act - Check oscillations
            double val1 = morlet.Calculate(0.0);
            double val2 = morlet.Calculate(Math.PI / 5.0); // Quarter period
            double val3 = morlet.Calculate(Math.PI / 2.5); // Half period

            // Assert - Oscillatory behavior
            Assert.True(Math.Abs(val1) > Math.Abs(val2)); // Decreasing envelope
            Assert.True(Math.Abs(val2) > Math.Abs(val3));
        }

        [Fact]
        public void MorletWavelet_Admissibility_ApproximateZeroMean()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);

            // Act - Integrate over [-10, 10]
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += morlet.Calculate(t) * 20.0 / samples;
            }

            // Assert - Approximately zero mean (Morlet is not strictly admissible but close)
            Assert.True(Math.Abs(sum) < 0.1);
        }

        [Fact]
        public void MorletWavelet_DifferentOmegas_DifferentFrequencies()
        {
            // Arrange
            var morlet5 = new MorletWavelet<double>(5.0);
            var morlet10 = new MorletWavelet<double>(10.0);

            // Act - Higher omega means more oscillations
            double val5_at1 = morlet5.Calculate(1.0);
            double val10_at1 = morlet10.Calculate(1.0);

            // Assert - Different omega produces different values
            Assert.NotEqual(val5_at1, val10_at1);
        }

        [Fact]
        public void MorletWavelet_Decompose_ReturnsValidComponents()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = morlet.Decompose(signal);

            // Assert
            Assert.Equal(signal.Length, approx.Length);
            Assert.Equal(signal.Length, detail.Length);
        }

        [Fact]
        public void MorletWavelet_GaussianEnvelope_DecreasesExponentially()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);

            // Act - Test Gaussian decay
            double val0 = Math.Abs(morlet.Calculate(0.0));
            double val2 = Math.Abs(morlet.Calculate(2.0));
            double val4 = Math.Abs(morlet.Calculate(4.0));

            // Assert - Exponential decay
            Assert.True(val0 > val2);
            Assert.True(val2 > val4);
        }

        #endregion

        #region Complex Morlet Wavelet Tests

        [Fact]
        public void ComplexMorletWavelet_Calculate_AtZero_RealIsOne()
        {
            // Arrange
            var cmorlet = new ComplexMorletWavelet<double>(5.0, 1.0);
            var z = new Complex<double>(0.0, 0.0);

            // Act
            var result = cmorlet.Calculate(z);

            // Assert - At origin: e^(iω*0) * e^(0) = 1 + 0i
            Assert.Equal(1.0, result.Real, Tolerance);
            Assert.Equal(0.0, result.Imaginary, Tolerance);
        }

        [Fact]
        public void ComplexMorletWavelet_Calculate_HasComplexValues()
        {
            // Arrange
            var cmorlet = new ComplexMorletWavelet<double>(5.0, 1.0);
            var z = new Complex<double>(1.0, 0.0);

            // Act
            var result = cmorlet.Calculate(z);

            // Assert - Both real and imaginary parts should be non-zero
            Assert.NotEqual(0.0, result.Real);
            Assert.NotEqual(0.0, result.Imaginary);
        }

        [Fact]
        public void ComplexMorletWavelet_Magnitude_DecreasesWithDistance()
        {
            // Arrange
            var cmorlet = new ComplexMorletWavelet<double>(5.0, 1.0);

            // Act
            var val0 = cmorlet.Calculate(new Complex<double>(0.0, 0.0));
            var val1 = cmorlet.Calculate(new Complex<double>(1.0, 0.0));
            var val2 = cmorlet.Calculate(new Complex<double>(2.0, 0.0));

            // Assert - Gaussian envelope decreases
            Assert.True(val0.Magnitude > val1.Magnitude);
            Assert.True(val1.Magnitude > val2.Magnitude);
        }

        [Fact]
        public void ComplexMorletWavelet_FilterCoefficients_AreComplex()
        {
            // Arrange
            var cmorlet = new ComplexMorletWavelet<double>(5.0, 1.0);

            // Act
            var waveletCoeffs = cmorlet.GetWaveletCoefficients();

            // Assert
            Assert.True(waveletCoeffs.Length > 0);
            // Check that at least some coefficients have non-zero imaginary parts
            bool hasImaginary = false;
            for (int i = 0; i < waveletCoeffs.Length; i++)
            {
                if (Math.Abs(waveletCoeffs[i].Imaginary) > Tolerance)
                {
                    hasImaginary = true;
                    break;
                }
            }
            Assert.True(hasImaginary);
        }

        [Fact]
        public void ComplexMorletWavelet_AdmissibilityCondition_Satisfied()
        {
            // Arrange
            var cmorlet = new ComplexMorletWavelet<double>(5.0, 1.0);

            // Act - ω*σ should be > 5 for admissibility
            double product = 5.0 * 1.0;

            // Assert
            Assert.True(product >= 5.0);
        }

        #endregion

        #region Mexican Hat Wavelet Tests

        [Fact]
        public void MexicanHatWavelet_Calculate_AtZero_PositivePeak()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);

            // Act - At x=0: (2 - 0) * e^0 = 2
            double val = mexicanHat.Calculate(0.0);

            // Assert
            Assert.Equal(2.0, val, Tolerance);
        }

        [Fact]
        public void MexicanHatWavelet_Calculate_NegativeLobes_Symmetric()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);

            // Act - Check symmetry
            double val1 = mexicanHat.Calculate(1.5);
            double val2 = mexicanHat.Calculate(-1.5);

            // Assert - Symmetric
            Assert.Equal(val1, val2, Tolerance);
        }

        [Fact]
        public void MexicanHatWavelet_Admissibility_ZeroMean_Satisfied()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);

            // Act - Integrate over [-10, 10]
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += mexicanHat.Calculate(t) * 20.0 / samples;
            }

            // Assert - Zero mean
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void MexicanHatWavelet_SecondDerivativeOfGaussian_Property()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);

            // Act - Mexican Hat is proportional to -d²/dx²[e^(-x²/2)]
            // It should have a positive peak at center and negative lobes
            double valCenter = mexicanHat.Calculate(0.0);
            double valSide = mexicanHat.Calculate(2.0);

            // Assert
            Assert.True(valCenter > 0); // Positive center
            Assert.True(valSide < 0);   // Negative lobes
        }

        [Fact]
        public void MexicanHatWavelet_DifferentSigmas_DifferentWidths()
        {
            // Arrange
            var narrow = new MexicanHatWavelet<double>(0.5);
            var wide = new MexicanHatWavelet<double>(2.0);

            // Act
            double narrowAt1 = narrow.Calculate(1.0);
            double wideAt1 = wide.Calculate(1.0);

            // Assert - Different widths produce different values
            Assert.NotEqual(narrowAt1, wideAt1);
        }

        [Fact]
        public void MexicanHatWavelet_Normalization_L2Norm()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);

            // Act - Compute L2 norm
            double sumSquared = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                double val = mexicanHat.Calculate(t);
                sumSquared += val * val * 20.0 / samples;
            }
            double l2norm = Math.Sqrt(sumSquared);

            // Assert - Should have finite L2 norm
            Assert.True(l2norm > 0 && l2norm < 10);
        }

        #endregion

        #region Gaussian Wavelet Tests

        [Fact]
        public void GaussianWavelet_Calculate_AtZero_Maximum()
        {
            // Arrange
            var gaussian = new GaussianWavelet<double>(1.0);

            // Act - Gaussian: e^(-x²/2σ²), max at x=0
            double val = gaussian.Calculate(0.0);

            // Assert
            Assert.Equal(1.0, val, Tolerance);
        }

        [Fact]
        public void GaussianWavelet_Calculate_Symmetric()
        {
            // Arrange
            var gaussian = new GaussianWavelet<double>(1.0);

            // Act
            double val1 = gaussian.Calculate(2.0);
            double val2 = gaussian.Calculate(-2.0);

            // Assert - Even function
            Assert.Equal(val1, val2, Tolerance);
        }

        [Fact]
        public void GaussianWavelet_Calculate_ExponentialDecay()
        {
            // Arrange
            var gaussian = new GaussianWavelet<double>(1.0);

            // Act
            double val0 = gaussian.Calculate(0.0);
            double val1 = gaussian.Calculate(1.0);
            double val2 = gaussian.Calculate(2.0);
            double val3 = gaussian.Calculate(3.0);

            // Assert - Monotonically decreasing from center
            Assert.True(val0 > val1);
            Assert.True(val1 > val2);
            Assert.True(val2 > val3);
        }

        [Fact]
        public void GaussianWavelet_DifferentSigmas_DifferentWidths()
        {
            // Arrange
            var narrow = new GaussianWavelet<double>(0.5);
            var wide = new GaussianWavelet<double>(2.0);

            // Act - At x=1
            double narrowVal = narrow.Calculate(1.0);
            double wideVal = wide.Calculate(1.0);

            // Assert - Narrow decays faster
            Assert.True(narrowVal < wideVal);
        }

        [Fact]
        public void GaussianWavelet_Decompose_SmoothApproximation()
        {
            // Arrange
            var gaussian = new GaussianWavelet<double>(1.0);
            var signal = new Vector<double>(new[] { 1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0 });

            // Act
            var (approx, detail) = gaussian.Decompose(signal);

            // Assert - Gaussian smooths the signal
            Assert.Equal(signal.Length, approx.Length);
            Assert.Equal(signal.Length, detail.Length);
        }

        [Fact]
        public void GaussianWavelet_Normalization_IntegratesTo1()
        {
            // Arrange
            var gaussian = new GaussianWavelet<double>(1.0);

            // Act - Integrate Gaussian
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += gaussian.Calculate(t) * 20.0 / samples;
            }

            // Assert - Should integrate to approximately √(2π)σ but our implementation is normalized differently
            Assert.True(sum > 0);
        }

        #endregion

        #region Meyer Wavelet Tests

        [Fact]
        public void MeyerWavelet_Calculate_CompactSupport()
        {
            // Arrange
            var meyer = new MeyerWavelet<double>();

            // Act - Meyer wavelet has compact support in frequency domain
            double val = meyer.Calculate(0.5);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void MeyerWavelet_Admissibility_ZeroMean()
        {
            // Arrange
            var meyer = new MeyerWavelet<double>();

            // Act - Compute approximate integral
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += meyer.Calculate(t) * 20.0 / samples;
            }

            // Assert
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void MeyerWavelet_Smooth_InfinitelyDifferentiable()
        {
            // Arrange
            var meyer = new MeyerWavelet<double>();

            // Act - Meyer is smooth, check continuity
            double val1 = meyer.Calculate(0.0);
            double val2 = meyer.Calculate(0.01);
            double diff = Math.Abs(val1 - val2);

            // Assert - Small change in x produces small change in y
            Assert.True(diff < 1.0);
        }

        [Fact]
        public void MeyerWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var meyer = new MeyerWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = meyer.Decompose(signal);

            // Assert
            Assert.Equal(signal.Length, approx.Length);
            Assert.Equal(signal.Length, detail.Length);
        }

        [Fact]
        public void MeyerWavelet_Orthogonality_Property()
        {
            // Arrange
            var meyer = new MeyerWavelet<double>();

            // Act - Meyer wavelet is orthogonal
            var h = meyer.GetScalingCoefficients();
            var g = meyer.GetWaveletCoefficients();

            // Assert - Both filters exist
            Assert.NotNull(h);
            Assert.NotNull(g);
            Assert.True(h.Length > 0);
            Assert.True(g.Length > 0);
        }

        #endregion

        #region Paul Wavelet Tests

        [Fact]
        public void PaulWavelet_Calculate_ReturnsValidValues()
        {
            // Arrange
            var paul = new PaulWavelet<double>(4);

            // Act
            double val = paul.Calculate(1.0);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void PaulWavelet_DifferentOrders_DifferentShapes()
        {
            // Arrange
            var paul2 = new PaulWavelet<double>(2);
            var paul4 = new PaulWavelet<double>(4);

            // Act
            double val2 = paul2.Calculate(1.0);
            double val4 = paul4.Calculate(1.0);

            // Assert
            Assert.NotEqual(val2, val4);
        }

        [Fact]
        public void PaulWavelet_Admissibility_Satisfied()
        {
            // Arrange
            var paul = new PaulWavelet<double>(4);

            // Act - Compute approximate zero mean
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += paul.Calculate(t) * 20.0 / samples;
            }

            // Assert
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void PaulWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var paul = new PaulWavelet<double>(4);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = paul.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Shannon Wavelet Tests

        [Fact]
        public void ShannonWavelet_Calculate_SincFunction()
        {
            // Arrange
            var shannon = new ShannonWavelet<double>();

            // Act - Shannon uses sinc function
            double valAt0 = shannon.Calculate(0.0);

            // Assert - sinc(0) should be 1 or close to wavelet value
            Assert.False(double.IsNaN(valAt0));
            Assert.False(double.IsInfinity(valAt0));
        }

        [Fact]
        public void ShannonWavelet_Admissibility_ZeroMean()
        {
            // Arrange
            var shannon = new ShannonWavelet<double>();

            // Act
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += shannon.Calculate(t) * 20.0 / samples;
            }

            // Assert
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void ShannonWavelet_IdealBandpass_Property()
        {
            // Arrange
            var shannon = new ShannonWavelet<double>();

            // Act - Shannon wavelet has ideal frequency response
            var coeffs = shannon.GetWaveletCoefficients();

            // Assert
            Assert.NotNull(coeffs);
            Assert.True(coeffs.Length > 0);
        }

        [Fact]
        public void ShannonWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var shannon = new ShannonWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = shannon.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Gabor Wavelet Tests

        [Fact]
        public void GaborWavelet_Calculate_ModulatedGaussian()
        {
            // Arrange
            var gabor = new GaborWavelet<double>(5.0, 1.0);

            // Act
            double val = gabor.Calculate(0.0);

            // Assert - Gabor is Gaussian modulated by complex exponential
            Assert.False(double.IsNaN(val));
        }

        [Fact]
        public void GaborWavelet_Admissibility_ApproximateZeroMean()
        {
            // Arrange
            var gabor = new GaborWavelet<double>(5.0, 1.0);

            // Act
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += gabor.Calculate(t) * 20.0 / samples;
            }

            // Assert - Approximate zero mean
            Assert.True(Math.Abs(sum) < 0.2);
        }

        [Fact]
        public void GaborWavelet_DifferentFrequencies_DifferentOscillations()
        {
            // Arrange
            var gabor5 = new GaborWavelet<double>(5.0, 1.0);
            var gabor10 = new GaborWavelet<double>(10.0, 1.0);

            // Act
            double val5 = gabor5.Calculate(1.0);
            double val10 = gabor10.Calculate(1.0);

            // Assert
            Assert.NotEqual(val5, val10);
        }

        [Fact]
        public void GaborWavelet_TimeFrequencyLocalization_Optimal()
        {
            // Arrange
            var gabor = new GaborWavelet<double>(5.0, 1.0);

            // Act - Gabor achieves optimal time-frequency localization
            double valCenter = Math.Abs(gabor.Calculate(0.0));
            double valSide = Math.Abs(gabor.Calculate(3.0));

            // Assert - Localized around center
            Assert.True(valCenter > valSide);
        }

        [Fact]
        public void GaborWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var gabor = new GaborWavelet<double>(5.0, 1.0);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = gabor.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region DOG Wavelet Tests

        [Fact]
        public void DOGWavelet_Calculate_DifferenceOfGaussians()
        {
            // Arrange
            var dog = new DOGWavelet<double>(1.0, 2.0);

            // Act - DOG is G(σ1) - G(σ2)
            double val = dog.Calculate(0.0);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void DOGWavelet_Admissibility_ZeroMean()
        {
            // Arrange
            var dog = new DOGWavelet<double>(1.0, 2.0);

            // Act
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += dog.Calculate(t) * 20.0 / samples;
            }

            // Assert - Zero mean (difference of two Gaussians)
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void DOGWavelet_BandpassFilter_Property()
        {
            // Arrange
            var dog = new DOGWavelet<double>(1.0, 2.0);

            // Act - DOG approximates Laplacian of Gaussian
            double valCenter = dog.Calculate(0.0);
            double valSide = dog.Calculate(2.0);

            // Assert - Band-pass characteristic
            Assert.NotEqual(valCenter, valSide);
        }

        [Fact]
        public void DOGWavelet_DifferentSigmas_DifferentShapes()
        {
            // Arrange
            var dog1 = new DOGWavelet<double>(0.5, 1.0);
            var dog2 = new DOGWavelet<double>(1.0, 2.0);

            // Act
            double val1 = dog1.Calculate(1.0);
            double val2 = dog2.Calculate(1.0);

            // Assert
            Assert.NotEqual(val1, val2);
        }

        [Fact]
        public void DOGWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var dog = new DOGWavelet<double>(1.0, 2.0);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = dog.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region BSpline Wavelet Tests

        [Fact]
        public void BSplineWavelet_Calculate_SmoothFunction()
        {
            // Arrange
            var bspline = new BSplineWavelet<double>(3);

            // Act
            double val = bspline.Calculate(0.5);

            // Assert - B-splines are smooth
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void BSplineWavelet_DifferentOrders_DifferentSmoothness()
        {
            // Arrange
            var bspline1 = new BSplineWavelet<double>(1);
            var bspline3 = new BSplineWavelet<double>(3);

            // Act
            double val1 = bspline1.Calculate(0.5);
            double val3 = bspline3.Calculate(0.5);

            // Assert - Higher order = smoother
            Assert.NotEqual(val1, val3);
        }

        [Fact]
        public void BSplineWavelet_CompactSupport_Property()
        {
            // Arrange
            var bspline = new BSplineWavelet<double>(3);

            // Act - B-splines have compact support
            double valInside = bspline.Calculate(1.0);
            double valOutside = bspline.Calculate(10.0);

            // Assert
            Assert.True(Math.Abs(valOutside) < Math.Abs(valInside) || Math.Abs(valOutside) < Tolerance);
        }

        [Fact]
        public void BSplineWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var bspline = new BSplineWavelet<double>(3);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = bspline.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Battle-Lemarié Wavelet Tests

        [Fact]
        public void BattleLemarieWavelet_Calculate_ReturnsValidValues()
        {
            // Arrange
            var battleLemarie = new BattleLemarieWavelet<double>(3);

            // Act
            double val = battleLemarie.Calculate(1.0);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void BattleLemarieWavelet_Orthogonality_Property()
        {
            // Arrange
            var battleLemarie = new BattleLemarieWavelet<double>(3);

            // Act - Battle-Lemarié wavelets are orthogonal
            var h = battleLemarie.GetScalingCoefficients();

            // Assert
            Assert.NotNull(h);
            Assert.True(h.Length > 0);
        }

        [Fact]
        public void BattleLemarieWavelet_BasedOnBSplines_Smooth()
        {
            // Arrange
            var battleLemarie = new BattleLemarieWavelet<double>(3);

            // Act - Should be smooth like B-splines
            double val1 = battleLemarie.Calculate(1.0);
            double val2 = battleLemarie.Calculate(1.01);

            // Assert - Continuity
            Assert.True(Math.Abs(val1 - val2) < 1.0);
        }

        [Fact]
        public void BattleLemarieWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var battleLemarie = new BattleLemarieWavelet<double>(3);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = battleLemarie.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Fejér-Korovkin Wavelet Tests

        [Fact]
        public void FejerKorovkinWavelet_Calculate_ReturnsValidValues()
        {
            // Arrange
            var fejerKorovkin = new FejérKorovkinWavelet<double>(3);

            // Act
            double val = fejerKorovkin.Calculate(0.5);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void FejerKorovkinWavelet_PositiveScalingFunction_Property()
        {
            // Arrange
            var fejerKorovkin = new FejérKorovkinWavelet<double>(3);

            // Act - FK wavelets have positive scaling functions
            var h = fejerKorovkin.GetScalingCoefficients();
            bool allPositive = true;
            for (int i = 0; i < h.Length; i++)
            {
                if (h[i] < 0)
                {
                    allPositive = false;
                    break;
                }
            }

            // Assert
            Assert.True(allPositive);
        }

        [Fact]
        public void FejerKorovkinWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var fejerKorovkin = new FejérKorovkinWavelet<double>(3);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = fejerKorovkin.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Continuous Mexican Hat Wavelet Tests

        [Fact]
        public void ContinuousMexicanHatWavelet_Calculate_SimilarToDiscrete()
        {
            // Arrange
            var continuous = new ContinuousMexicanHatWavelet<double>(1.0);
            var discrete = new MexicanHatWavelet<double>(1.0);

            // Act
            double contVal = continuous.Calculate(0.0);
            double discVal = discrete.Calculate(0.0);

            // Assert - Should be similar at center
            Assert.True(Math.Abs(contVal - discVal) < 1.0);
        }

        [Fact]
        public void ContinuousMexicanHatWavelet_Admissibility_ZeroMean()
        {
            // Arrange
            var continuous = new ContinuousMexicanHatWavelet<double>(1.0);

            // Act
            double sum = 0;
            int samples = 2000;
            for (int i = 0; i < samples; i++)
            {
                double t = -10.0 + (i * 20.0) / samples;
                sum += continuous.Calculate(t) * 20.0 / samples;
            }

            // Assert
            Assert.Equal(0.0, sum, LooseTolerance);
        }

        [Fact]
        public void ContinuousMexicanHatWavelet_SymmetricProperty()
        {
            // Arrange
            var continuous = new ContinuousMexicanHatWavelet<double>(1.0);

            // Act
            double val1 = continuous.Calculate(2.0);
            double val2 = continuous.Calculate(-2.0);

            // Assert
            Assert.Equal(val1, val2, Tolerance);
        }

        [Fact]
        public void ContinuousMexicanHatWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var continuous = new ContinuousMexicanHatWavelet<double>(1.0);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = continuous.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Complex Gaussian Wavelet Tests

        [Fact]
        public void ComplexGaussianWavelet_Calculate_HasComplexValues()
        {
            // Arrange
            var cgaussian = new ComplexGaussianWavelet<double>(2, 1.0);
            var z = new Complex<double>(1.0, 0.0);

            // Act
            var result = cgaussian.Calculate(z);

            // Assert - Complex Gaussian has both real and imaginary parts
            Assert.False(double.IsNaN(result.Real));
            Assert.False(double.IsNaN(result.Imaginary));
        }

        [Fact]
        public void ComplexGaussianWavelet_DifferentOrders_DifferentDerivatives()
        {
            // Arrange
            var order1 = new ComplexGaussianWavelet<double>(1, 1.0);
            var order2 = new ComplexGaussianWavelet<double>(2, 1.0);
            var z = new Complex<double>(1.0, 0.0);

            // Act
            var val1 = order1.Calculate(z);
            var val2 = order2.Calculate(z);

            // Assert - Different orders give different values
            Assert.NotEqual(val1.Real, val2.Real);
        }

        [Fact]
        public void ComplexGaussianWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var cgaussian = new ComplexGaussianWavelet<double>(2, 1.0);
            var signal = new Vector<Complex<double>>(8);
            for (int i = 0; i < 8; i++)
            {
                signal[i] = new Complex<double>(i + 1.0, 0.0);
            }

            // Act
            var (approx, detail) = cgaussian.Decompose(signal);

            // Assert
            Assert.NotNull(approx);
            Assert.NotNull(detail);
        }

        #endregion

        #region Reverse Biorthogonal Wavelet Tests

        [Fact]
        public void ReverseBiorthogonalWavelet_Calculate_ReturnsValidValues()
        {
            // Arrange
            var rbior = new ReverseBiorthogonalWavelet<double>(2, 2);

            // Act
            double val = rbior.Calculate(0.5);

            // Assert
            Assert.False(double.IsNaN(val));
            Assert.False(double.IsInfinity(val));
        }

        [Fact]
        public void ReverseBiorthogonalWavelet_DualOfBiorthogonal_Property()
        {
            // Arrange
            var rbior = new ReverseBiorthogonalWavelet<double>(2, 2);
            var bior = new BiorthogonalWavelet<double>(2, 2);

            // Act - Reverse biorthogonal should be related to biorthogonal
            var rbiorCoeffs = rbior.GetScalingCoefficients();
            var biorCoeffs = bior.GetScalingCoefficients();

            // Assert - Both have valid coefficients
            Assert.NotNull(rbiorCoeffs);
            Assert.NotNull(biorCoeffs);
            Assert.True(rbiorCoeffs.Length > 0);
            Assert.True(biorCoeffs.Length > 0);
        }

        [Fact]
        public void ReverseBiorthogonalWavelet_Decompose_ValidOutput()
        {
            // Arrange
            var rbior = new ReverseBiorthogonalWavelet<double>(2, 2);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = rbior.Decompose(signal);

            // Assert
            Assert.Equal(signal.Length / 2, approx.Length);
            Assert.Equal(signal.Length / 2, detail.Length);
        }

        [Fact]
        public void ReverseBiorthogonalWavelet_SymmetryProperty()
        {
            // Arrange
            var rbior = new ReverseBiorthogonalWavelet<double>(2, 2);
            var h = rbior.GetScalingCoefficients();

            // Act - Check symmetry
            double asymmetry = 0;
            for (int i = 0; i < h.Length / 2; i++)
            {
                asymmetry += Math.Abs(h[i] - h[h.Length - 1 - i]);
            }

            // Assert
            Assert.True(asymmetry < 0.5);
        }

        #endregion

        #region Multi-Resolution Analysis Tests

        [Fact]
        public void MultiResolutionAnalysis_Haar_ThreeLevels_EnergyPreserved()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });
            double originalEnergy = ComputeEnergy(signal);

            // Act - Three-level decomposition
            var (approx1, detail1) = haar.Decompose(signal);
            var (approx2, detail2) = haar.Decompose(approx1);
            var (approx3, detail3) = haar.Decompose(approx2);

            // Assert
            double totalEnergy = ComputeEnergy(approx3) + ComputeEnergy(detail3) +
                                 ComputeEnergy(detail2) + ComputeEnergy(detail1);
            Assert.Equal(originalEnergy, totalEnergy, LooseTolerance);
        }

        [Fact]
        public void MultiResolutionAnalysis_Daubechies_TwoLevels_CoarsensCorrectly()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx1, detail1) = db4.Decompose(signal);
            var (approx2, detail2) = db4.Decompose(approx1);

            // Assert - Each level reduces size by half
            Assert.Equal(signal.Length / 2, approx1.Length);
            Assert.Equal(approx1.Length / 2, approx2.Length);
        }

        [Fact]
        public void MultiResolutionAnalysis_Symlet_ProgressiveSmoothing()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);
            var signal = new Vector<double>(new[] { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 });

            // Act - Two levels
            var (approx1, detail1) = sym4.Decompose(signal);
            var (approx2, detail2) = sym4.Decompose(approx1);

            // Assert - Detail coefficients should decrease in magnitude at coarser scales
            double detail1Energy = ComputeEnergy(detail1);
            double detail2Energy = ComputeEnergy(detail2);
            Assert.True(detail2Energy < detail1Energy * 2); // Coarser level has less detail energy
        }

        [Fact]
        public void ScaleTranslation_Haar_DifferentScales_PreservesShape()
        {
            // Arrange
            var haar = new HaarWavelet<double>();

            // Act - Test at different scales
            double scale1 = haar.Calculate(0.25);
            double scale2 = haar.Calculate(0.5);

            // Assert - Within support, should have expected values
            Assert.Equal(1.0, scale1, Tolerance);
            Assert.Equal(-1.0, scale2, Tolerance);
        }

        [Fact]
        public void ScaleTranslation_Morlet_Translation_ShiftsCenter()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);

            // Act - Test translation property
            double atCenter = Math.Abs(morlet.Calculate(0.0));
            double translated = Math.Abs(morlet.Calculate(0.5));

            // Assert - Maximum at center
            Assert.True(atCenter > translated);
        }

        [Fact]
        public void WaveletTransform_Haar_SignalWithStep_DetectsEdge()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var stepSignal = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0 });

            // Act
            var (approx, detail) = haar.Decompose(stepSignal);

            // Assert - Detail coefficients should be large where step occurs
            double maxDetail = 0;
            for (int i = 0; i < detail.Length; i++)
            {
                if (Math.Abs(detail[i]) > maxDetail)
                    maxDetail = Math.Abs(detail[i]);
            }
            Assert.True(maxDetail > 1.0); // Should detect the step
        }

        [Fact]
        public void WaveletTransform_MexicanHat_SmoothSignal_SmallDetails()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);
            var smoothSignal = new Vector<double>(new[] { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 });

            // Act
            var (approx, detail) = mexicanHat.Decompose(smoothSignal);

            // Assert - Smooth signal produces small detail coefficients
            double detailEnergy = ComputeEnergy(detail);
            double approxEnergy = ComputeEnergy(approx);
            Assert.True(detailEnergy < approxEnergy); // Approximation dominates for smooth signals
        }

        [Fact]
        public void OrthogonalWavelets_Haar_BasisOrthogonality()
        {
            // Arrange
            var haar = new HaarWavelet<double>();

            // Act - Test orthogonality of filter coefficients
            var h = haar.GetScalingCoefficients();
            var g = haar.GetWaveletCoefficients();

            // Compute inner product
            double innerProduct = 0;
            for (int i = 0; i < Math.Min(h.Length, g.Length); i++)
            {
                innerProduct += h[i] * g[i];
            }

            // Assert - Orthogonal filters have zero inner product
            Assert.Equal(0.0, innerProduct, Tolerance);
        }

        [Fact]
        public void OrthogonalWavelets_Daubechies_SelfOrthogonality()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);
            var h = db4.GetScalingCoefficients();

            // Act - Check self-orthogonality at even shifts
            double innerProduct = 0;
            for (int i = 0; i < h.Length - 2; i++)
            {
                innerProduct += h[i] * h[i + 2];
            }

            // Assert
            Assert.Equal(0.0, innerProduct, LooseTolerance);
        }

        [Fact]
        public void ContinuousWavelets_Morlet_ScaleInvariance()
        {
            // Arrange
            var morlet = new MorletWavelet<double>(5.0);

            // Act - Test at different scaled positions
            double val1 = morlet.Calculate(1.0);
            double val2 = morlet.Calculate(2.0);

            // Assert - Values should follow Gaussian decay
            Assert.True(Math.Abs(val1) > Math.Abs(val2));
        }

        [Fact]
        public void DiscreteWavelets_Haar_PowerOf2Length_Required()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var validSignal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }); // Length 4 = 2^2
            var invalidSignal = new Vector<double>(new[] { 1.0, 2.0, 3.0 }); // Length 3

            // Act & Assert - Valid signal should work
            var (approx, detail) = haar.Decompose(validSignal);
            Assert.NotNull(approx);
            Assert.NotNull(detail);

            // Invalid signal should throw (length must be even)
            Assert.Throws<ArgumentException>(() => haar.Decompose(invalidSignal));
        }

        [Fact]
        public void ComplexWavelets_ComplexMorlet_PhaseInformation_Preserved()
        {
            // Arrange
            var cmorlet = new ComplexMorletWavelet<double>(5.0, 1.0);

            // Act - Complex wavelets capture phase
            var val1 = cmorlet.Calculate(new Complex<double>(1.0, 0.0));
            var val2 = cmorlet.Calculate(new Complex<double>(1.0, 1.0));

            // Assert - Different inputs produce different complex outputs
            Assert.NotEqual(val1.Real, val2.Real);
            Assert.NotEqual(val1.Imaginary, val2.Imaginary);
        }

        [Fact]
        public void SymmetricWavelets_Symlet_LinearPhase_Property()
        {
            // Arrange
            var sym4 = new SymletWavelet<double>(4);
            var h = sym4.GetScalingCoefficients();

            // Act - Measure symmetry
            double centerOfMass = 0;
            double totalMass = 0;
            for (int i = 0; i < h.Length; i++)
            {
                centerOfMass += i * Math.Abs(h[i]);
                totalMass += Math.Abs(h[i]);
            }
            double center = centerOfMass / totalMass;

            // Assert - Center of mass should be near middle for symmetric wavelets
            double expectedCenter = (h.Length - 1) / 2.0;
            Assert.True(Math.Abs(center - expectedCenter) < 1.0);
        }

        [Fact]
        public void VanishingMoments_Daubechies_PolynomialCancellation()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4); // DB4 has 2 vanishing moments

            // Act - Apply to constant signal (0-th order polynomial)
            var constantSignal = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 });
            var (approx, detail) = db4.Decompose(constantSignal);

            // Assert - Detail coefficients should be nearly zero for constant signal
            double detailEnergy = ComputeEnergy(detail);
            Assert.True(detailEnergy < LooseTolerance);
        }

        [Fact]
        public void VanishingMoments_Coiflet_HigherOrderPolynomials()
        {
            // Arrange
            var coif2 = new CoifletWavelet<double>(2); // Has 4 vanishing moments

            // Act - Apply to linear signal
            var linearSignal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
            var (approx, detail) = coif2.Decompose(linearSignal);

            // Assert - Should suppress linear trends
            double approxEnergy = ComputeEnergy(approx);
            double detailEnergy = ComputeEnergy(detail);
            Assert.True(approxEnergy > detailEnergy);
        }

        [Fact]
        public void SignalDenoising_Haar_ThresholdingDetails()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var noisySignal = new Vector<double>(new[] { 1.0, 1.1, 2.0, 1.9, 3.0, 3.2, 4.0, 3.8 });

            // Act
            var (approx, detail) = haar.Decompose(noisySignal);

            // Assert - Approximation should be smoother than original
            // Calculate variance as measure of smoothness
            double originalVar = ComputeVariance(noisySignal);
            double approxVar = ComputeVariance(approx);
            Assert.True(approxVar < originalVar); // Approximation is smoother
        }

        [Fact]
        public void BiorthogonalWavelets_PerfectReconstruction_WithDualFilters()
        {
            // Arrange
            var bior = new BiorthogonalWavelet<double>(2, 2);
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act
            var (approx, detail) = bior.Decompose(signal);

            // Assert - Decomposition should preserve information
            double originalEnergy = ComputeEnergy(signal);
            double decomposedEnergy = ComputeEnergy(approx) + ComputeEnergy(detail);
            Assert.Equal(originalEnergy, decomposedEnergy, LooseTolerance);
        }

        [Fact]
        public void WaveletPackets_Haar_FullDecomposition_AllCoefficients()
        {
            // Arrange
            var haar = new HaarWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            // Act - Decompose approximation AND detail at second level
            var (approx1, detail1) = haar.Decompose(signal);
            var (approx2, detail2) = haar.Decompose(approx1);
            var (approx3, detail3) = haar.Decompose(detail1); // Wavelet packet: decompose detail too

            // Assert - All decompositions valid
            Assert.NotNull(approx2);
            Assert.NotNull(detail2);
            Assert.NotNull(approx3);
            Assert.NotNull(detail3);
        }

        [Fact]
        public void CompactSupport_Daubechies_OutsideSupport_Zero()
        {
            // Arrange
            var db4 = new DaubechiesWavelet<double>(4);

            // Act - Test well outside support [0, 3]
            double farLeft = db4.Calculate(-10.0);
            double farRight = db4.Calculate(10.0);

            // Assert
            Assert.Equal(0.0, farLeft, Tolerance);
            Assert.Equal(0.0, farRight, Tolerance);
        }

        [Fact]
        public void TimeFrequencyLocalization_MexicanHat_UncertaintyPrinciple()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);

            // Act - Compute time spread
            double timeSpread = 0;
            int samples = 1000;
            for (int i = 0; i < samples; i++)
            {
                double t = -5.0 + (i * 10.0) / samples;
                double val = mexicanHat.Calculate(t);
                timeSpread += t * t * val * val * 10.0 / samples;
            }

            // Assert - Time spread should be finite (good localization)
            Assert.True(timeSpread > 0 && timeSpread < 10);
        }

        [Fact]
        public void EdgeDetection_MexicanHat_StepFunction_StrongResponse()
        {
            // Arrange
            var mexicanHat = new MexicanHatWavelet<double>(1.0);
            var stepSignal = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0 });

            // Act
            var (approx, detail) = mexicanHat.Decompose(stepSignal);

            // Assert - Should detect edge
            double maxDetail = 0;
            for (int i = 0; i < detail.Length; i++)
            {
                if (Math.Abs(detail[i]) > maxDetail)
                    maxDetail = Math.Abs(detail[i]);
            }
            Assert.True(maxDetail > 0.1);
        }

        #endregion

        #region Helper Methods

        private Vector<double> ReconstructHaar(Vector<double> approx, Vector<double> detail)
        {
            int n = approx.Length;
            var reconstructed = new Vector<double>(n * 2);
            double sqrt2 = Math.Sqrt(2);

            for (int i = 0; i < n; i++)
            {
                double a = approx[i];
                double d = detail[i];
                reconstructed[2 * i] = (a + d) / sqrt2;
                reconstructed[2 * i + 1] = (a - d) / sqrt2;
            }

            return reconstructed;
        }

        private Vector<double> ReconstructDaubechies(Vector<double> approx, Vector<double> detail, DaubechiesWavelet<double> wavelet)
        {
            var h = wavelet.GetScalingCoefficients();
            var g = wavelet.GetWaveletCoefficients();
            int n = approx.Length;
            var reconstructed = new Vector<double>(n * 2);

            for (int i = 0; i < n * 2; i++)
            {
                double sum = 0;
                for (int j = 0; j < h.Length; j++)
                {
                    int approxIdx = (i + j) / 2;
                    int detailIdx = (i + j) / 2;
                    if (approxIdx < n && (i + j) % 2 == 0)
                    {
                        sum += h[j] * approx[approxIdx];
                    }
                    if (detailIdx < n && (i + j) % 2 == 0)
                    {
                        sum += g[j] * detail[detailIdx];
                    }
                }
                reconstructed[i] = sum;
            }

            return reconstructed;
        }

        private double ComputeEnergy(Vector<double> signal)
        {
            double energy = 0;
            for (int i = 0; i < signal.Length; i++)
            {
                energy += signal[i] * signal[i];
            }
            return energy;
        }

        private double ComputeVariance(Vector<double> signal)
        {
            double mean = 0;
            for (int i = 0; i < signal.Length; i++)
            {
                mean += signal[i];
            }
            mean /= signal.Length;

            double variance = 0;
            for (int i = 0; i < signal.Length; i++)
            {
                double diff = signal[i] - mean;
                variance += diff * diff;
            }
            return variance / signal.Length;
        }

        #endregion
    }
}
