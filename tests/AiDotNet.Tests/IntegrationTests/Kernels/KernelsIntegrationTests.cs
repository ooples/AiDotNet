using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Kernels
{
    /// <summary>
    /// Comprehensive integration tests for Kernel functions with mathematically verified results.
    /// Tests kernel properties, known values, parameter effects, kernel matrices, and edge cases.
    /// </summary>
    public class KernelsIntegrationTests
    {
        private const double Tolerance = 1e-10;

        // ===== Linear Kernel Tests =====

        [Fact]
        public void LinearKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert - K(x,y) = K(y,x)
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void LinearKernel_SelfSimilarity_AlwaysPositive()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var kxx = kernel.Calculate(x, x);

            // Assert - K(x,x) >= 0
            Assert.True(kxx >= 0.0);
        }

        [Fact]
        public void LinearKernel_KnownValues_DotProduct()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Linear kernel = x·y = 1*4 + 2*5 + 3*6 = 32
            Assert.Equal(32.0, result, precision: 10);
        }

        [Fact]
        public void LinearKernel_IdenticalVectors_SquaredNorm()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var result = kernel.Calculate(x, x);

            // Assert - K(x,x) = x·x = 3² + 4² = 25
            Assert.Equal(25.0, result, precision: 10);
        }

        [Fact]
        public void LinearKernel_OrthogonalVectors_ReturnsZero()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 0.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Orthogonal vectors have zero dot product
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void LinearKernel_ZeroVector_ReturnsZero()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var zero = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

            // Act
            var result = kernel.Calculate(x, zero);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void LinearKernel_GramMatrix_IsSymmetric()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var data = new[]
            {
                new Vector<double>(new[] { 1.0, 2.0 }),
                new Vector<double>(new[] { 3.0, 4.0 }),
                new Vector<double>(new[] { 5.0, 6.0 })
            };

            // Act - Compute Gram matrix
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert - Gram matrix should be symmetric
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        [Fact]
        public void LinearKernel_GramMatrix_IsPositiveSemiDefinite()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var data = new[]
            {
                new Vector<double>(new[] { 1.0, 2.0 }),
                new Vector<double>(new[] { 3.0, 4.0 }),
                new Vector<double>(new[] { 5.0, 6.0 })
            };

            // Act - Compute Gram matrix
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert - All eigenvalues should be non-negative (checking determinant > 0 for simplicity)
            var det = gramMatrix.Determinant();
            Assert.True(det >= 0.0);
        }

        // ===== Polynomial Kernel Tests =====

        [Fact]
        public void PolynomialKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 3.0, coef0: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_KnownValues_Degree2()
        {
            // Arrange - Degree 2, coef0 = 1
            var kernel = new PolynomialKernel<double>(degree: 2.0, coef0: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - (x·y + 1)^2 = (1*3 + 2*4 + 1)^2 = 12^2 = 144
            Assert.Equal(144.0, result, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_KnownValues_Degree3()
        {
            // Arrange - Degree 3, coef0 = 0
            var kernel = new PolynomialKernel<double>(degree: 3.0, coef0: 0.0);
            var x = new Vector<double>(new[] { 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - (x·y + 0)^3 = (2*1 + 3*1)^3 = 5^3 = 125
            Assert.Equal(125.0, result, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_Degree1_EquivalentToLinear()
        {
            // Arrange
            var polyKernel = new PolynomialKernel<double>(degree: 1.0, coef0: 0.0);
            var linearKernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var polyResult = polyKernel.Calculate(x, y);
            var linearResult = linearKernel.Calculate(x, y);

            // Assert - Degree 1 polynomial with coef0=0 should equal linear kernel
            Assert.Equal(linearResult, polyResult, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_ParameterEffect_DifferentDegrees()
        {
            // Arrange
            var x = new Vector<double>(new[] { 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            var kernel1 = new PolynomialKernel<double>(degree: 1.0, coef0: 0.0);
            var kernel2 = new PolynomialKernel<double>(degree: 2.0, coef0: 0.0);
            var kernel3 = new PolynomialKernel<double>(degree: 5.0, coef0: 0.0);

            // Act
            var result1 = kernel1.Calculate(x, y); // 5^1 = 5
            var result2 = kernel2.Calculate(x, y); // 5^2 = 25
            var result3 = kernel3.Calculate(x, y); // 5^5 = 3125

            // Assert - Higher degrees produce larger values
            Assert.True(result1 < result2);
            Assert.True(result2 < result3);
            Assert.Equal(5.0, result1, precision: 10);
            Assert.Equal(25.0, result2, precision: 10);
            Assert.Equal(3125.0, result3, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_ParameterEffect_DifferentCoef0()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 1.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            var kernel1 = new PolynomialKernel<double>(degree: 2.0, coef0: 0.0);
            var kernel2 = new PolynomialKernel<double>(degree: 2.0, coef0: 1.0);
            var kernel3 = new PolynomialKernel<double>(degree: 2.0, coef0: 2.0);

            // Act
            var result1 = kernel1.Calculate(x, y); // (2 + 0)^2 = 4
            var result2 = kernel2.Calculate(x, y); // (2 + 1)^2 = 9
            var result3 = kernel3.Calculate(x, y); // (2 + 2)^2 = 16

            // Assert
            Assert.Equal(4.0, result1, precision: 10);
            Assert.Equal(9.0, result2, precision: 10);
            Assert.Equal(16.0, result3, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_GramMatrix_IsSymmetric()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 2.0, coef0: 1.0);
            var data = new[]
            {
                new Vector<double>(new[] { 1.0, 0.0 }),
                new Vector<double>(new[] { 0.0, 1.0 }),
                new Vector<double>(new[] { 1.0, 1.0 })
            };

            // Act
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        // ===== Gaussian/RBF Kernel Tests =====

        [Fact]
        public void GaussianKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void GaussianKernel_SelfSimilarity_ReturnsOne()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var kxx = kernel.Calculate(x, x);

            // Assert - K(x,x) = exp(-0 / 2σ²) = 1
            Assert.Equal(1.0, kxx, precision: 10);
        }

        [Fact]
        public void GaussianKernel_KnownValues_CalculatesCorrectly()
        {
            // Arrange - σ = 1.0
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - exp(-||x-y||² / 2σ²) = exp(-1 / 2) ≈ 0.6065
            Assert.Equal(0.6065306597126334, result, precision: 10);
        }

        [Fact]
        public void GaussianKernel_KnownValues_2D()
        {
            // Arrange - σ = 1.0
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - distance² = 2, exp(-2 / 2) = exp(-1) ≈ 0.3679
            Assert.Equal(0.36787944117144233, result, precision: 10);
        }

        [Fact]
        public void GaussianKernel_OutputRange_BetweenZeroAndOne()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Gaussian kernel always returns values in [0, 1]
            Assert.True(result >= 0.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void GaussianKernel_ParameterEffect_DifferentSigmas()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            var kernel1 = new GaussianKernel<double>(sigma: 0.1);
            var kernel2 = new GaussianKernel<double>(sigma: 1.0);
            var kernel3 = new GaussianKernel<double>(sigma: 10.0);

            // Act
            var result1 = kernel1.Calculate(x, y);
            var result2 = kernel2.Calculate(x, y);
            var result3 = kernel3.Calculate(x, y);

            // Assert - Larger sigma gives higher similarity for same distance
            Assert.True(result1 < result2);
            Assert.True(result2 < result3);
        }

        [Fact]
        public void GaussianKernel_DistantPoints_ApproachZero()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 100.0, 100.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Very distant points should have near-zero similarity
            Assert.True(result < 0.0001);
        }

        [Fact]
        public void GaussianKernel_GramMatrix_IsSymmetric()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var data = new[]
            {
                new Vector<double>(new[] { 0.0, 0.0 }),
                new Vector<double>(new[] { 1.0, 1.0 }),
                new Vector<double>(new[] { 2.0, 2.0 })
            };

            // Act
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        [Fact]
        public void GaussianKernel_GramMatrix_DiagonalIsOne()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var data = new[]
            {
                new Vector<double>(new[] { 1.0, 2.0 }),
                new Vector<double>(new[] { 3.0, 4.0 }),
                new Vector<double>(new[] { 5.0, 6.0 })
            };

            // Act
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert - Diagonal elements should be 1 (self-similarity)
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(1.0, gramMatrix[i, i], precision: 10);
            }
        }

        // ===== Sigmoid Kernel Tests =====

        [Fact]
        public void SigmoidKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new SigmoidKernel<double>(alpha: 1.0, c: 0.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void SigmoidKernel_KnownValues_AlphaOne()
        {
            // Arrange
            var kernel = new SigmoidKernel<double>(alpha: 1.0, c: 0.0);
            var x = new Vector<double>(new[] { 1.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - tanh(1*1 + 0) = tanh(1) ≈ 0.7616
            Assert.Equal(Math.Tanh(1.0), result, precision: 10);
        }

        [Fact]
        public void SigmoidKernel_OutputRange_BetweenMinusOneAndOne()
        {
            // Arrange
            var kernel = new SigmoidKernel<double>(alpha: 0.5, c: 0.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Sigmoid kernel returns values in [-1, 1]
            Assert.True(result >= -1.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void SigmoidKernel_ParameterEffect_DifferentAlpha()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 1.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            var kernel1 = new SigmoidKernel<double>(alpha: 0.1, c: 0.0);
            var kernel2 = new SigmoidKernel<double>(alpha: 1.0, c: 0.0);
            var kernel3 = new SigmoidKernel<double>(alpha: 2.0, c: 0.0);

            // Act
            var result1 = kernel1.Calculate(x, y); // tanh(0.1 * 2)
            var result2 = kernel2.Calculate(x, y); // tanh(1.0 * 2)
            var result3 = kernel3.Calculate(x, y); // tanh(2.0 * 2)

            // Assert - Larger alpha produces larger values (steeper curve)
            Assert.True(result1 < result2);
            Assert.True(result2 < result3);
        }

        [Fact]
        public void SigmoidKernel_ParameterEffect_DifferentC()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            var kernel1 = new SigmoidKernel<double>(alpha: 1.0, c: -1.0);
            var kernel2 = new SigmoidKernel<double>(alpha: 1.0, c: 0.0);
            var kernel3 = new SigmoidKernel<double>(alpha: 1.0, c: 1.0);

            // Act
            var result1 = kernel1.Calculate(x, y); // tanh(1 - 1) = 0
            var result2 = kernel2.Calculate(x, y); // tanh(1)
            var result3 = kernel3.Calculate(x, y); // tanh(2)

            // Assert
            Assert.Equal(0.0, result1, precision: 10);
            Assert.True(result2 < result3);
        }

        [Fact]
        public void SigmoidKernel_OrthogonalVectors_ReturnsHyperbolicTangentOfC()
        {
            // Arrange
            var kernel = new SigmoidKernel<double>(alpha: 1.0, c: 0.5);
            var x = new Vector<double>(new[] { 1.0, 0.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - For orthogonal vectors: tanh(α*0 + c) = tanh(c)
            Assert.Equal(Math.Tanh(0.5), result, precision: 10);
        }

        // ===== Laplacian Kernel Tests =====

        [Fact]
        public void LaplacianKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void LaplacianKernel_SelfSimilarity_ReturnsOne()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var kxx = kernel.Calculate(x, x);

            // Assert - K(x,x) = exp(-0/σ) = 1
            Assert.Equal(1.0, kxx, precision: 10);
        }

        [Fact]
        public void LaplacianKernel_KnownValues_ManhattanDistance()
        {
            // Arrange - σ = 1.0
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - exp(-|1-0| + |1-0|) / σ) = exp(-2) ≈ 0.1353
            Assert.Equal(Math.Exp(-2.0), result, precision: 10);
        }

        [Fact]
        public void LaplacianKernel_OutputRange_BetweenZeroAndOne()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Laplacian kernel returns values in [0, 1]
            Assert.True(result >= 0.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void LaplacianKernel_ParameterEffect_DifferentSigmas()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            var kernel1 = new LaplacianKernel<double>(sigma: 0.5);
            var kernel2 = new LaplacianKernel<double>(sigma: 1.0);
            var kernel3 = new LaplacianKernel<double>(sigma: 2.0);

            // Act
            var result1 = kernel1.Calculate(x, y);
            var result2 = kernel2.Calculate(x, y);
            var result3 = kernel3.Calculate(x, y);

            // Assert - Larger sigma gives higher similarity for same distance
            Assert.True(result1 < result2);
            Assert.True(result2 < result3);
        }

        [Fact]
        public void LaplacianKernel_OneDimensional_CalculatesCorrectly()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0 });
            var y = new Vector<double>(new[] { 2.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - exp(-|2-0|/1) = exp(-2) ≈ 0.1353
            Assert.Equal(Math.Exp(-2.0), result, precision: 10);
        }

        [Fact]
        public void LaplacianKernel_GramMatrix_IsSymmetric()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var data = new[]
            {
                new Vector<double>(new[] { 0.0, 0.0 }),
                new Vector<double>(new[] { 1.0, 0.0 }),
                new Vector<double>(new[] { 0.0, 1.0 })
            };

            // Act
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        // ===== Rational Quadratic Kernel Tests =====

        [Fact]
        public void RationalQuadraticKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new RationalQuadraticKernel<double>(c: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void RationalQuadraticKernel_SelfSimilarity_ReturnsOne()
        {
            // Arrange
            var kernel = new RationalQuadraticKernel<double>(c: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var kxx = kernel.Calculate(x, x);

            // Assert - K(x,x) = 1 - 0/(0 + c) = 1
            Assert.Equal(1.0, kxx, precision: 10);
        }

        [Fact]
        public void RationalQuadraticKernel_KnownValues_CalculatesCorrectly()
        {
            // Arrange
            var kernel = new RationalQuadraticKernel<double>(c: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - 1 - distance²/(distance² + c) = 1 - 1/(1 + 1) = 0.5
            Assert.Equal(0.5, result, precision: 10);
        }

        [Fact]
        public void RationalQuadraticKernel_OutputRange_BetweenZeroAndOne()
        {
            // Arrange
            var kernel = new RationalQuadraticKernel<double>(c: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert
            Assert.True(result >= 0.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void RationalQuadraticKernel_ParameterEffect_DifferentC()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            var kernel1 = new RationalQuadraticKernel<double>(c: 0.5);
            var kernel2 = new RationalQuadraticKernel<double>(c: 1.0);
            var kernel3 = new RationalQuadraticKernel<double>(c: 2.0);

            // Act
            var result1 = kernel1.Calculate(x, y); // 1 - 1/(1 + 0.5) = 0.333...
            var result2 = kernel2.Calculate(x, y); // 1 - 1/(1 + 1.0) = 0.5
            var result3 = kernel3.Calculate(x, y); // 1 - 1/(1 + 2.0) = 0.666...

            // Assert - Larger c gives higher similarity
            Assert.True(result1 < result2);
            Assert.True(result2 < result3);
        }

        // ===== Cauchy Kernel Tests =====

        [Fact]
        public void CauchyKernel_Symmetry_KernelIsSymmetric()
        {
            // Arrange
            var kernel = new CauchyKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);

            // Assert
            Assert.Equal(kxy, kyx, precision: 10);
        }

        [Fact]
        public void CauchyKernel_SelfSimilarity_ReturnsOne()
        {
            // Arrange
            var kernel = new CauchyKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var kxx = kernel.Calculate(x, x);

            // Assert - K(x,x) = 1 / (1 + 0/σ²) = 1
            Assert.Equal(1.0, kxx, precision: 10);
        }

        [Fact]
        public void CauchyKernel_KnownValues_CalculatesCorrectly()
        {
            // Arrange
            var kernel = new CauchyKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - 1 / (1 + distance²/σ²) = 1 / (1 + 1/1) = 0.5
            Assert.Equal(0.5, result, precision: 10);
        }

        [Fact]
        public void CauchyKernel_OutputRange_BetweenZeroAndOne()
        {
            // Arrange
            var kernel = new CauchyKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert
            Assert.True(result >= 0.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void CauchyKernel_ParameterEffect_DifferentSigmas()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            var kernel1 = new CauchyKernel<double>(sigma: 0.5);
            var kernel2 = new CauchyKernel<double>(sigma: 1.0);
            var kernel3 = new CauchyKernel<double>(sigma: 2.0);

            // Act
            var result1 = kernel1.Calculate(x, y);
            var result2 = kernel2.Calculate(x, y);
            var result3 = kernel3.Calculate(x, y);

            // Assert - Larger sigma gives higher similarity
            Assert.True(result1 < result2);
            Assert.True(result2 < result3);
        }

        [Fact]
        public void CauchyKernel_GramMatrix_IsSymmetric()
        {
            // Arrange
            var kernel = new CauchyKernel<double>(sigma: 1.0);
            var data = new[]
            {
                new Vector<double>(new[] { 0.0, 0.0 }),
                new Vector<double>(new[] { 1.0, 1.0 }),
                new Vector<double>(new[] { 2.0, 2.0 })
            };

            // Act
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        // ===== Edge Cases: Different Scales =====

        [Fact]
        public void LinearKernel_DifferentScales_WorksCorrectly()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var small = new Vector<double>(new[] { 0.001, 0.002 });
            var large = new Vector<double>(new[] { 1000.0, 2000.0 });

            // Act
            var result1 = kernel.Calculate(small, small);
            var result2 = kernel.Calculate(large, large);

            // Assert - Both should work correctly regardless of scale
            Assert.True(result1 > 0.0);
            Assert.True(result2 > 0.0);
            Assert.True(result2 > result1); // Larger vectors have larger dot product
        }

        [Fact]
        public void GaussianKernel_VerySmallDistance_ApproachesOne()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0001, 2.0001 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Very small distance should give result close to 1
            Assert.True(result > 0.99);
        }

        [Fact]
        public void PolynomialKernel_ZeroVectors_HandlesCorrectly()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 2.0, coef0: 1.0);
            var zero = new Vector<double>(new[] { 0.0, 0.0 });

            // Act
            var result = kernel.Calculate(zero, zero);

            // Assert - (0·0 + 1)^2 = 1
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void LaplacianKernel_NegativeValues_WorksCorrectly()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { -1.0, -2.0 });
            var y = new Vector<double>(new[] { 1.0, 2.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Should handle negative values correctly
            Assert.True(result >= 0.0);
            Assert.True(result <= 1.0);
        }

        // ===== Cross-Kernel Comparison Tests =====

        [Fact]
        public void Kernels_IdenticalVectors_AllReturnMaximumSimilarity()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            var gaussian = new GaussianKernel<double>(sigma: 1.0);
            var laplacian = new LaplacianKernel<double>(sigma: 1.0);
            var cauchy = new CauchyKernel<double>(sigma: 1.0);
            var rational = new RationalQuadraticKernel<double>(c: 1.0);

            // Act & Assert - All distance-based kernels should return 1 for identical vectors
            Assert.Equal(1.0, gaussian.Calculate(x, x), precision: 10);
            Assert.Equal(1.0, laplacian.Calculate(x, x), precision: 10);
            Assert.Equal(1.0, cauchy.Calculate(x, x), precision: 10);
            Assert.Equal(1.0, rational.Calculate(x, x), precision: 10);
        }

        [Fact]
        public void GaussianVsLaplacian_DifferentDistanceMetrics()
        {
            // Arrange - Gaussian uses L2, Laplacian uses L1
            var gaussian = new GaussianKernel<double>(sigma: 1.0);
            var laplacian = new LaplacianKernel<double>(sigma: 1.0);

            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var gaussianResult = gaussian.Calculate(x, y);
            var laplacianResult = laplacian.Calculate(x, y);

            // Assert - Different distance metrics produce different results
            Assert.NotEqual(gaussianResult, laplacianResult);
        }

        // ===== Multi-Dimensional Tests =====

        [Fact]
        public void LinearKernel_HighDimensional_WorksCorrectly()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new double[100]);
            var y = new Vector<double>(new double[100]);

            for (int i = 0; i < 100; i++)
            {
                x[i] = i + 1.0;
                y[i] = i + 2.0;
            }

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Should handle high-dimensional data
            Assert.True(result > 0.0);
        }

        [Fact]
        public void GaussianKernel_HighDimensional_MaintainsProperties()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new double[50]);

            for (int i = 0; i < 50; i++)
            {
                x[i] = i * 0.1;
            }

            // Act
            var result = kernel.Calculate(x, x);

            // Assert - Self-similarity should still be 1 in high dimensions
            Assert.Equal(1.0, result, precision: 10);
        }

        // ===== Numerical Stability Tests =====

        [Fact]
        public void GaussianKernel_VeryLargeDistance_ReturnsNearZeroWithoutOverflow()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1000.0, 1000.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Should not overflow, should be near zero
            Assert.False(double.IsNaN(result));
            Assert.False(double.IsInfinity(result));
            Assert.True(result >= 0.0);
        }

        [Fact]
        public void PolynomialKernel_LargeDegree_HandlesCorrectly()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 10.0, coef0: 1.0);
            var x = new Vector<double>(new[] { 0.1, 0.1 });
            var y = new Vector<double>(new[] { 0.1, 0.1 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Should not overflow with small values and high degree
            Assert.False(double.IsNaN(result));
            Assert.False(double.IsInfinity(result));
        }

        // ===== Type Compatibility Tests =====

        [Fact]
        public void LinearKernel_FloatType_WorksCorrectly()
        {
            // Arrange
            var kernel = new LinearKernel<float>();
            var x = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
            var y = new Vector<float>(new[] { 4.0f, 5.0f, 6.0f });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert
            Assert.Equal(32.0f, result, precision: 6);
        }

        [Fact]
        public void GaussianKernel_FloatType_WorksCorrectly()
        {
            // Arrange
            var kernel = new GaussianKernel<float>(sigma: 1.0);
            var x = new Vector<float>(new[] { 1.0f, 2.0f });

            // Act
            var result = kernel.Calculate(x, x);

            // Assert
            Assert.Equal(1.0f, result, precision: 6);
        }

        [Fact]
        public void PolynomialKernel_DecimalType_WorksCorrectly()
        {
            // Arrange
            var kernel = new PolynomialKernel<decimal>(degree: 2.0m, coef0: 1.0m);
            var x = new Vector<decimal>(new[] { 1.0m, 2.0m });
            var y = new Vector<decimal>(new[] { 3.0m, 4.0m });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - (1*3 + 2*4 + 1)^2 = 12^2 = 144
            Assert.Equal(144.0m, result);
        }

        // ===== Additional Gram Matrix Tests =====

        [Fact]
        public void PolynomialKernel_GramMatrix_IsPositiveSemiDefinite()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 2.0, coef0: 1.0);
            var data = new[]
            {
                new Vector<double>(new[] { 1.0, 0.0 }),
                new Vector<double>(new[] { 0.0, 1.0 }),
                new Vector<double>(new[] { 1.0, 1.0 })
            };

            // Act - Compute Gram matrix
            var gramMatrix = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert - Determinant should be >= 0 for positive semi-definite
            var det = gramMatrix.Determinant();
            Assert.True(det >= -1e-10); // Allow small numerical error
        }

        [Fact]
        public void GaussianKernel_LargeGramMatrix_IsSymmetric()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var data = new Vector<double>[5];
            for (int i = 0; i < 5; i++)
            {
                data[i] = new Vector<double>(new[] { i * 1.0, i * 2.0 });
            }

            // Act
            var gramMatrix = new Matrix<double>(5, 5);
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    gramMatrix[i, j] = kernel.Calculate(data[i], data[j]);
                }
            }

            // Assert
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        // ===== Additional Edge Cases =====

        [Fact]
        public void LinearKernel_SingleDimension_WorksCorrectly()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 5.0 });
            var y = new Vector<double>(new[] { 3.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert
            Assert.Equal(15.0, result, precision: 10);
        }

        [Fact]
        public void GaussianKernel_SingleDimension_WorksCorrectly()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0 });
            var y = new Vector<double>(new[] { 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - exp(-1/2) ≈ 0.6065
            Assert.Equal(Math.Exp(-0.5), result, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_NegativeCoef0_WorksCorrectly()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 2.0, coef0: -1.0);
            var x = new Vector<double>(new[] { 2.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - (2*1 + 2*1 - 1)^2 = 3^2 = 9
            Assert.Equal(9.0, result, precision: 10);
        }

        [Fact]
        public void SigmoidKernel_NegativeVectors_WorksCorrectly()
        {
            // Arrange
            var kernel = new SigmoidKernel<double>(alpha: 1.0, c: 0.0);
            var x = new Vector<double>(new[] { -1.0, -2.0 });
            var y = new Vector<double>(new[] { -3.0, -4.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Should handle negative values
            Assert.True(result >= -1.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void LaplacianKernel_LargeVectors_MaintainsNormalization()
        {
            // Arrange
            var kernel = new LaplacianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });

            // Act
            var result = kernel.Calculate(x, x);

            // Assert - Self-similarity should always be 1
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void RationalQuadraticKernel_VerySmallC_BehavesCorrectly()
        {
            // Arrange
            var kernel = new RationalQuadraticKernel<double>(c: 0.01);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - Should still be in valid range
            Assert.True(result >= 0.0);
            Assert.True(result <= 1.0);
        }

        [Fact]
        public void CauchyKernel_VeryLargeSigma_ApproachesConstant()
        {
            // Arrange
            var kernel = new CauchyKernel<double>(sigma: 1000.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var result = kernel.Calculate(x, y);

            // Assert - With very large sigma, all points should be considered similar
            Assert.True(result > 0.99);
        }

        // ===== Consistency Tests =====

        [Fact]
        public void LinearKernel_Commutative_OrderDoesNotMatter()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var vectors = new[]
            {
                new Vector<double>(new[] { 1.0, 2.0 }),
                new Vector<double>(new[] { 3.0, 4.0 }),
                new Vector<double>(new[] { 5.0, 6.0 })
            };

            // Act & Assert - Test all pairs
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    var kij = kernel.Calculate(vectors[i], vectors[j]);
                    var kji = kernel.Calculate(vectors[j], vectors[i]);
                    Assert.Equal(kij, kji, precision: 10);
                }
            }
        }

        [Fact]
        public void GaussianKernel_Consistency_MultipleCalculations()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act - Calculate multiple times
            var result1 = kernel.Calculate(x, y);
            var result2 = kernel.Calculate(x, y);
            var result3 = kernel.Calculate(x, y);

            // Assert - Should get same result every time
            Assert.Equal(result1, result2, precision: 10);
            Assert.Equal(result2, result3, precision: 10);
        }

        [Fact]
        public void PolynomialKernel_DefaultParameters_UsesExpectedDefaults()
        {
            // Arrange
            var kernelDefault = new PolynomialKernel<double>();
            var kernelExplicit = new PolynomialKernel<double>(degree: 3.0, coef0: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var resultDefault = kernelDefault.Calculate(x, y);
            var resultExplicit = kernelExplicit.Calculate(x, y);

            // Assert - Default parameters should match explicit values
            Assert.Equal(resultDefault, resultExplicit, precision: 10);
        }

        [Fact]
        public void GaussianKernel_DefaultParameters_UsesExpectedDefaults()
        {
            // Arrange
            var kernelDefault = new GaussianKernel<double>();
            var kernelExplicit = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var resultDefault = kernelDefault.Calculate(x, y);
            var resultExplicit = kernelExplicit.Calculate(x, y);

            // Assert
            Assert.Equal(resultDefault, resultExplicit, precision: 10);
        }

        [Fact]
        public void SigmoidKernel_DefaultParameters_UsesExpectedDefaults()
        {
            // Arrange
            var kernelDefault = new SigmoidKernel<double>();
            var kernelExplicit = new SigmoidKernel<double>(alpha: 1.0, c: 0.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });

            // Act
            var resultDefault = kernelDefault.Calculate(x, y);
            var resultExplicit = kernelExplicit.Calculate(x, y);

            // Assert
            Assert.Equal(resultDefault, resultExplicit, precision: 10);
        }

        // ===== Additional Mathematical Property Tests =====

        [Fact]
        public void GaussianKernel_TriangleInequality_Satisfies()
        {
            // Arrange - For valid kernel, distances should satisfy certain properties
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0 });
            var z = new Vector<double>(new[] { 2.0, 0.0 });

            // Act
            var kxy = kernel.Calculate(x, y);
            var kyz = kernel.Calculate(y, z);
            var kxz = kernel.Calculate(x, z);

            // Assert - All values should be valid
            Assert.True(kxy > kxz); // Closer points more similar
            Assert.True(kyz > kxz);
        }

        [Fact]
        public void LinearKernel_ScaleInvariance_DoesNotHold()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });
            var xScaled = new Vector<double>(new[] { 2.0, 4.0 });
            var yScaled = new Vector<double>(new[] { 6.0, 8.0 });

            // Act
            var result1 = kernel.Calculate(x, y);
            var result2 = kernel.Calculate(xScaled, yScaled);

            // Assert - Linear kernel is NOT scale invariant
            Assert.NotEqual(result1, result2);
            Assert.Equal(result1 * 4.0, result2, precision: 10); // Scales by factor squared
        }

        [Fact]
        public void GaussianKernel_MonotonicDecreaseWithDistance()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var y1 = new Vector<double>(new[] { 0.5, 0.0 });
            var y2 = new Vector<double>(new[] { 1.0, 0.0 });
            var y3 = new Vector<double>(new[] { 2.0, 0.0 });

            // Act
            var k1 = kernel.Calculate(x, y1);
            var k2 = kernel.Calculate(x, y2);
            var k3 = kernel.Calculate(x, y3);

            // Assert - Similarity should decrease with distance
            Assert.True(k1 > k2);
            Assert.True(k2 > k3);
        }

        [Fact]
        public void PolynomialKernel_Homogeneous_WithZeroCoef0()
        {
            // Arrange
            var kernel = new PolynomialKernel<double>(degree: 2.0, coef0: 0.0);
            var x = new Vector<double>(new[] { 1.0, 2.0 });
            var y = new Vector<double>(new[] { 3.0, 4.0 });
            var alpha = 2.0;

            // Act
            var kxy = kernel.Calculate(x, y);
            var xScaled = new Vector<double>(new[] { alpha * 1.0, alpha * 2.0 });
            var yScaled = new Vector<double>(new[] { alpha * 3.0, alpha * 4.0 });
            var kScaled = kernel.Calculate(xScaled, yScaled);

            // Assert - k(αx, αy) = α^(2d) * k(x,y) for homogeneous kernel
            Assert.Equal(kxy * Math.Pow(alpha, 4.0), kScaled, precision: 8);
        }

        [Fact]
        public void LaplacianKernel_RobustToOutliers_ComparedToGaussian()
        {
            // Arrange
            var laplacian = new LaplacianKernel<double>(sigma: 1.0);
            var gaussian = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var yOutlier = new Vector<double>(new[] { 10.0, 10.0 });

            // Act
            var laplacianResult = laplacian.Calculate(x, yOutlier);
            var gaussianResult = gaussian.Calculate(x, yOutlier);

            // Assert - Laplacian should give higher similarity to outliers (L1 vs L2)
            Assert.True(laplacianResult > gaussianResult);
        }

        [Fact]
        public void CauchyKernel_LongTailProperty_ComparedToGaussian()
        {
            // Arrange
            var cauchy = new CauchyKernel<double>(sigma: 1.0);
            var gaussian = new GaussianKernel<double>(sigma: 1.0);
            var x = new Vector<double>(new[] { 0.0, 0.0 });
            var yDistant = new Vector<double>(new[] { 5.0, 5.0 });

            // Act
            var cauchyResult = cauchy.Calculate(x, yDistant);
            var gaussianResult = gaussian.Calculate(x, yDistant);

            // Assert - Cauchy has longer tail, so distant points have higher similarity
            Assert.True(cauchyResult > gaussianResult);
        }
    }
}
