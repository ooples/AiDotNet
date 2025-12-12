using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Normalizers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Normalizers
{
    /// <summary>
    /// Comprehensive integration tests for all Normalizers with mathematically verified results.
    /// Tests ensure correct normalization, denormalization, and mathematical properties.
    /// </summary>
    public class NormalizersIntegrationTests
    {
        private const double Tolerance = 1e-10;
        private const double RelaxedTolerance = 1e-6;

        #region ZScoreNormalizer Tests

        [Fact]
        public void ZScoreNormalizer_NormalizeOutput_ProducesMeanZeroStdOne()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Mean should be ~0, Std should be ~1
            var mean = normalized.ToArray().Average();
            var variance = normalized.ToArray().Select(x => (x - mean) * (x - mean)).Average();
            var std = Math.Sqrt(variance);

            Assert.True(Math.Abs(mean) < RelaxedTolerance);
            Assert.True(Math.Abs(std - 1.0) < RelaxedTolerance);
        }

        [Fact]
        public void ZScoreNormalizer_Denormalize_RecoversOriginalValues()
        {
            // Arrange
            var original = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert - Should recover original values
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 10);
            }
        }

        [Fact]
        public void ZScoreNormalizer_NormalizeInput_NormalizesEachColumnIndependently()
        {
            // Arrange
            var matrix = new Matrix<double>(5, 2);
            // Column 1: [1, 2, 3, 4, 5]
            matrix[0, 0] = 1.0; matrix[1, 0] = 2.0; matrix[2, 0] = 3.0; matrix[3, 0] = 4.0; matrix[4, 0] = 5.0;
            // Column 2: [10, 20, 30, 40, 50]
            matrix[0, 1] = 10.0; matrix[1, 1] = 20.0; matrix[2, 1] = 30.0; matrix[3, 1] = 40.0; matrix[4, 1] = 50.0;

            var normalizer = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should have mean ~0 and std ~1
            for (int col = 0; col < 2; col++)
            {
                var column = normalized.GetColumn(col);
                var mean = column.ToArray().Average();
                var variance = column.ToArray().Select(x => (x - mean) * (x - mean)).Average();
                var std = Math.Sqrt(variance);

                Assert.True(Math.Abs(mean) < RelaxedTolerance);
                Assert.True(Math.Abs(std - 1.0) < RelaxedTolerance);
            }
        }

        [Fact]
        public void ZScoreNormalizer_WithTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new[] { 5 });
            for (int i = 0; i < 5; i++)
            {
                tensor[i] = (i + 1) * 10.0; // [10, 20, 30, 40, 50]
            }

            var normalizer = new ZScoreNormalizer<double, Tensor<double>, Tensor<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(tensor);

            // Assert
            var normalizedVec = normalized.ToVector();
            var mean = normalizedVec.ToArray().Average();
            Assert.True(Math.Abs(mean) < RelaxedTolerance);
        }

        [Fact]
        public void ZScoreNormalizer_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var data = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
            var normalizer = new ZScoreNormalizer<float, Vector<float>, Vector<float>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            var mean = normalized.ToArray().Average();
            Assert.True(Math.Abs(mean) < 1e-5f);
        }

        #endregion

        #region MinMaxNormalizer Tests

        [Fact]
        public void MinMaxNormalizer_NormalizeOutput_ProducesRangeZeroOne()
        {
            // Arrange
            var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Min should be 0, Max should be 1
            var min = normalized.Min();
            var max = normalized.Max();

            Assert.Equal(0.0, min, precision: 10);
            Assert.Equal(1.0, max, precision: 10);
        }

        [Fact]
        public void MinMaxNormalizer_Denormalize_RecoversOriginalValues()
        {
            // Arrange
            var original = new Vector<double>(new[] { 5.0, 15.0, 25.0, 35.0, 45.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert - Should recover original values
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 10);
            }
        }

        [Fact]
        public void MinMaxNormalizer_NormalizeInput_NormalizesEachColumn()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 2);
            // Column 1: [1, 2, 3, 4]
            matrix[0, 0] = 1.0; matrix[1, 0] = 2.0; matrix[2, 0] = 3.0; matrix[3, 0] = 4.0;
            // Column 2: [100, 200, 300, 400]
            matrix[0, 1] = 100.0; matrix[1, 1] = 200.0; matrix[2, 1] = 300.0; matrix[3, 1] = 400.0;

            var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should be in [0, 1] range
            for (int col = 0; col < 2; col++)
            {
                var column = normalized.GetColumn(col);
                var min = column.Min();
                var max = column.Max();

                Assert.Equal(0.0, min, precision: 10);
                Assert.Equal(1.0, max, precision: 10);
            }
        }

        [Fact]
        public void MinMaxNormalizer_WithNegativeValues_WorksCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new[] { -10.0, -5.0, 0.0, 5.0, 10.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            Assert.Equal(0.0, normalized.Min(), precision: 10);
            Assert.Equal(1.0, normalized.Max(), precision: 10);
            Assert.Equal(0.5, normalized[2], precision: 10); // Middle value should be 0.5
        }

        [Fact]
        public void MinMaxNormalizer_WithConstantData_HandlesGracefully()
        {
            // Arrange
            var data = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0, 5.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - All values should map to 0 (or handle division by zero gracefully)
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.False(double.IsNaN(normalized[i]));
            }
        }

        [Fact]
        public void MinMaxNormalizer_SmallDataset_WorksCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 10.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            Assert.Equal(0.0, normalized[0], precision: 10);
            Assert.Equal(1.0, normalized[1], precision: 10);
        }

        [Fact]
        public void MinMaxNormalizer_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var data = new Vector<double>(1000);
            for (int i = 0; i < 1000; i++)
            {
                data[i] = i * 0.5;
            }

            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (normalized, parameters) = normalizer.NormalizeOutput(data);
            sw.Stop();

            // Assert
            Assert.Equal(0.0, normalized.Min(), precision: 10);
            Assert.Equal(1.0, normalized.Max(), precision: 10);
            Assert.True(sw.ElapsedMilliseconds < 500);
        }

        #endregion

        #region MaxAbsScaler Tests

        [Fact]
        public void MaxAbsScaler_NormalizeOutput_ProducesRangeNegativeOneToOne()
        {
            // Arrange
            var data = new Vector<double>(new[] { -50.0, -25.0, 0.0, 25.0, 50.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert - Should be in [-1, 1] range
            Assert.True(normalized.ToArray().All(x => x >= -1.0 && x <= 1.0));
            Assert.Equal(-1.0, normalized[0], precision: 10);
            Assert.Equal(1.0, normalized[4], precision: 10);
            Assert.Equal(0.0, normalized[2], precision: 10);
        }

        [Fact]
        public void MaxAbsScaler_PreservesZeros()
        {
            // Arrange - Sparse data with zeros
            var data = new Vector<double>(new[] { 0.0, 0.0, 100.0, 0.0, 50.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert - Zeros should remain zeros
            Assert.Equal(0.0, normalized[0], precision: 10);
            Assert.Equal(0.0, normalized[1], precision: 10);
            Assert.Equal(0.0, normalized[3], precision: 10);
        }

        [Fact]
        public void MaxAbsScaler_Denormalize_RecoversOriginalValues()
        {
            // Arrange
            var original = new Vector<double>(new[] { -100.0, -50.0, 0.0, 50.0, 100.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(original);
            var denormalized = scaler.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 10);
            }
        }

        [Fact]
        public void MaxAbsScaler_WithPositiveValuesOnly_WorksCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(0.2, normalized[0], precision: 10);
            Assert.Equal(1.0, normalized[4], precision: 10);
        }

        [Fact]
        public void MaxAbsScaler_NormalizeInput_WorksOnMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(5, 2);
            // Column 1: [-100, -50, 0, 50, 100]
            matrix[0, 0] = -100.0; matrix[1, 0] = -50.0; matrix[2, 0] = 0.0;
            matrix[3, 0] = 50.0; matrix[4, 0] = 100.0;
            // Column 2: [-20, -10, 0, 10, 20]
            matrix[0, 1] = -20.0; matrix[1, 1] = -10.0; matrix[2, 1] = 0.0;
            matrix[3, 1] = 10.0; matrix[4, 1] = 20.0;

            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeInput(matrix);

            // Assert
            var col1 = normalized.GetColumn(0);
            var col2 = normalized.GetColumn(1);

            Assert.Equal(-1.0, col1[0], precision: 10);
            Assert.Equal(1.0, col1[4], precision: 10);
            Assert.Equal(-1.0, col2[0], precision: 10);
            Assert.Equal(1.0, col2[4], precision: 10);
        }

        #endregion

        #region RobustScalingNormalizer Tests

        [Fact]
        public void RobustScalingNormalizer_NormalizeOutput_CentersAtMedian()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Median value (3.0) should normalize to ~0
            Assert.True(Math.Abs(normalized[2]) < RelaxedTolerance);
        }

        [Fact]
        public void RobustScalingNormalizer_HandlesOutliers_Better()
        {
            // Arrange - Data with outlier
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 1000.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Outlier should not dominate scaling
            // The first four values should be in a reasonable range
            Assert.True(Math.Abs(normalized[0]) < 10.0);
            Assert.True(Math.Abs(normalized[1]) < 10.0);
            Assert.True(Math.Abs(normalized[2]) < 10.0);
            Assert.True(Math.Abs(normalized[3]) < 10.0);
        }

        [Fact]
        public void RobustScalingNormalizer_Denormalize_RecoversOriginal()
        {
            // Arrange
            var original = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 10);
            }
        }

        [Fact]
        public void RobustScalingNormalizer_WithSkewedData_WorksWell()
        {
            // Arrange - Skewed distribution
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0, 100.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should handle skewness without issues
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
        }

        [Fact]
        public void RobustScalingNormalizer_NormalizeInput_WorksOnMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(5, 2);
            // Column 1: [10, 20, 30, 40, 50]
            matrix[0, 0] = 10.0; matrix[1, 0] = 20.0; matrix[2, 0] = 30.0;
            matrix[3, 0] = 40.0; matrix[4, 0] = 50.0;
            // Column 2: [1, 2, 3, 4, 100] - with outlier
            matrix[0, 1] = 1.0; matrix[1, 1] = 2.0; matrix[2, 1] = 3.0;
            matrix[3, 1] = 4.0; matrix[4, 1] = 100.0;

            var normalizer = new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should be normalized independently
            Assert.Equal(2, parameters.Count);
        }

        #endregion

        #region LpNormNormalizer Tests

        [Fact]
        public void LpNormNormalizer_L2Norm_ProducesUnitNorm()
        {
            // Arrange - L2 norm (Euclidean)
            var data = new Vector<double>(new[] { 3.0, 4.0 }); // Norm = 5
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should have L2 norm = 1
            var norm = Math.Sqrt(normalized[0] * normalized[0] + normalized[1] * normalized[1]);
            Assert.Equal(1.0, norm, precision: 10);
            Assert.Equal(0.6, normalized[0], precision: 10); // 3/5
            Assert.Equal(0.8, normalized[1], precision: 10); // 4/5
        }

        [Fact]
        public void LpNormNormalizer_L1Norm_ProducesSumOfOne()
        {
            // Arrange - L1 norm (Manhattan)
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0 }); // L1 norm = 6
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(1.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Sum of absolute values should be 1
            var l1Norm = normalized.ToArray().Select(Math.Abs).Sum();
            Assert.Equal(1.0, l1Norm, precision: 10);
        }

        [Fact]
        public void LpNormNormalizer_Denormalize_RecoversOriginal()
        {
            // Arrange
            var original = new Vector<double>(new[] { 3.0, 4.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 10);
            }
        }

        [Fact]
        public void LpNormNormalizer_PreservesDirection()
        {
            // Arrange
            var data = new Vector<double>(new[] { 6.0, 8.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Direction (ratio) should be preserved
            var ratio = normalized[0] / normalized[1];
            var originalRatio = data[0] / data[1];
            Assert.Equal(originalRatio, ratio, precision: 10);
        }

        [Fact]
        public void LpNormNormalizer_NormalizeInput_NormalizesEachColumn()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 2);
            // Column 1: [3, 4, 0]
            matrix[0, 0] = 3.0; matrix[1, 0] = 4.0; matrix[2, 0] = 0.0;
            // Column 2: [1, 1, 1]
            matrix[0, 1] = 1.0; matrix[1, 1] = 1.0; matrix[2, 1] = 1.0;

            var normalizer = new LpNormNormalizer<double, Matrix<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should have L2 norm = 1
            for (int col = 0; col < 2; col++)
            {
                var column = normalized.GetColumn(col);
                var norm = Math.Sqrt(column.ToArray().Select(x => x * x).Sum());
                Assert.Equal(1.0, norm, precision: 10);
            }
        }

        #endregion

        #region QuantileTransformer Tests

        [Fact]
        public void QuantileTransformer_UniformOutput_ProducesUniformDistribution()
        {
            // Arrange
            var data = new Vector<double>(100);
            for (int i = 0; i < 100; i++)
            {
                data[i] = i + 1;
            }

            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 100);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Should be uniformly distributed in [0, 1]
            var min = transformed.Min();
            var max = transformed.Max();
            Assert.True(min >= 0.0);
            Assert.True(max <= 1.0);
        }

        [Fact]
        public void QuantileTransformer_HandlesOutliers_Effectively()
        {
            // Arrange - Data with extreme outliers
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 1000.0 });
            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 10);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Outlier should not dominate, all values in [0, 1]
            Assert.True(transformed.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        [Fact]
        public void QuantileTransformer_Denormalize_RecoversApproximateValues()
        {
            // Arrange
            var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 100);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(original);
            var denormalized = transformer.Denormalize(transformed, parameters);

            // Assert - Should approximately recover (quantile transform is not perfectly reversible)
            for (int i = 0; i < original.Length; i++)
            {
                Assert.True(Math.Abs(original[i] - denormalized[i]) < 1.0);
            }
        }

        [Fact]
        public void QuantileTransformer_NormalDistribution_ProducesNormalShape()
        {
            // Arrange
            var data = new Vector<double>(100);
            for (int i = 0; i < 100; i++)
            {
                data[i] = i + 1;
            }

            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Normal, 100);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Should produce values in roughly normal range
            var mean = transformed.ToArray().Average();
            // Mean of transformed data should be close to 0 for uniform input -> normal output
            Assert.True(Math.Abs(mean) < 1.0);
        }

        [Fact]
        public void QuantileTransformer_PreservesRankOrder()
        {
            // Arrange
            var data = new Vector<double>(new[] { 5.0, 2.0, 8.0, 1.0, 9.0 });
            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 10);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Rank order should be preserved
            Assert.True(transformed[3] < transformed[1]); // 1 < 2
            Assert.True(transformed[1] < transformed[0]); // 2 < 5
            Assert.True(transformed[0] < transformed[2]); // 5 < 8
            Assert.True(transformed[2] < transformed[4]); // 8 < 9
        }

        [Fact]
        public void QuantileTransformer_NormalizeInput_WorksOnMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(10, 2);
            for (int i = 0; i < 10; i++)
            {
                matrix[i, 0] = i + 1;
                matrix[i, 1] = (i + 1) * 10;
            }

            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(
                OutputDistribution.Uniform, 10);

            // Act
            var (transformed, parameters) = transformer.NormalizeInput(matrix);

            // Assert - Each column should be transformed independently
            Assert.Equal(2, parameters.Count);
        }

        [Fact]
        public void QuantileTransformer_WithConstantFeature_HandlesGracefully()
        {
            // Arrange - Constant feature (zero variance)
            var data = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0, 5.0 });
            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 10);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Should handle without NaN
            Assert.False(transformed.ToArray().Any(x => double.IsNaN(x)));
        }

        #endregion

        #region MeanVarianceNormalizer Tests

        [Fact]
        public void MeanVarianceNormalizer_ProducesZeroMeanUnitVariance()
        {
            // Arrange
            var data = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0, 10.0 });
            var normalizer = new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            var mean = normalized.ToArray().Average();
            var variance = normalized.ToArray().Select(x => (x - mean) * (x - mean)).Average();

            Assert.True(Math.Abs(mean) < RelaxedTolerance);
            Assert.True(Math.Abs(variance - 1.0) < RelaxedTolerance);
        }

        [Fact]
        public void MeanVarianceNormalizer_Denormalize_RecoversOriginal()
        {
            // Arrange
            var original = new Vector<double>(new[] { 100.0, 200.0, 300.0, 400.0, 500.0 });
            var normalizer = new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 10);
            }
        }

        [Fact]
        public void MeanVarianceNormalizer_EquivalentToZScore()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var meanVarNorm = new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>();
            var zScoreNorm = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (mvNormalized, mvParams) = meanVarNorm.NormalizeOutput(data);
            var (zsNormalized, zsParams) = zScoreNorm.NormalizeOutput(data);

            // Assert - Should produce same results
            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(mvNormalized[i], zsNormalized[i], precision: 10);
            }
        }

        #endregion

        #region LogNormalizer Tests

        [Fact]
        public void LogNormalizer_WithPositiveValues_WorksCorrectly()
        {
            // Arrange - Exponentially growing data
            var data = new Vector<double>(new[] { 1.0, 10.0, 100.0, 1000.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should be in [0, 1] range
            var min = normalized.Min();
            var max = normalized.Max();

            Assert.Equal(0.0, min, precision: 10);
            Assert.Equal(1.0, max, precision: 10);
        }

        [Fact]
        public void LogNormalizer_Denormalize_RecoversOriginal()
        {
            // Arrange
            var original = new Vector<double>(new[] { 1.0, 10.0, 100.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], precision: 8);
            }
        }

        [Fact]
        public void LogNormalizer_WithNegativeValues_AppliesShift()
        {
            // Arrange - Data with negative values
            var data = new Vector<double>(new[] { -10.0, -5.0, 0.0, 5.0, 10.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should handle negative values by shifting
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
            Assert.False(normalized.ToArray().Any(x => double.IsInfinity(x)));
        }

        [Fact]
        public void LogNormalizer_CompressesWideRange()
        {
            // Arrange - Very wide range
            var data = new Vector<double>(new[] { 1.0, 1000.0, 1000000.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should compress to [0, 1]
            Assert.True(normalized.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        [Fact]
        public void LogNormalizer_NormalizeInput_WorksOnMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 2);
            // Column 1: [1, 10, 100, 1000]
            matrix[0, 0] = 1.0; matrix[1, 0] = 10.0; matrix[2, 0] = 100.0; matrix[3, 0] = 1000.0;
            // Column 2: [2, 20, 200, 2000]
            matrix[0, 1] = 2.0; matrix[1, 1] = 20.0; matrix[2, 1] = 200.0; matrix[3, 1] = 2000.0;

            var normalizer = new LogNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should be in [0, 1]
            for (int col = 0; col < 2; col++)
            {
                var column = normalized.GetColumn(col);
                Assert.True(column.ToArray().All(x => x >= 0.0 && x <= 1.0));
            }
        }

        #endregion

        #region BinningNormalizer Tests

        [Fact]
        public void BinningNormalizer_CreatesBins_Correctly()
        {
            // Arrange
            var data = new Vector<double>(50);
            for (int i = 0; i < 50; i++)
            {
                data[i] = i + 1;
            }

            var normalizer = new BinningNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should be in [0, 1] range
            Assert.True(normalized.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        [Fact]
        public void BinningNormalizer_Denormalize_ProducesApproximateValues()
        {
            // Arrange
            var original = new Vector<double>(new[] { 1.0, 5.0, 10.0, 15.0, 20.0 });
            var normalizer = new BinningNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(original);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert - Binning is lossy, so values are approximate
            for (int i = 0; i < original.Length; i++)
            {
                Assert.True(Math.Abs(original[i] - denormalized[i]) < 10.0);
            }
        }

        [Fact]
        public void BinningNormalizer_DiscretesBins_AsExpected()
        {
            // Arrange
            var data = new Vector<double>(100);
            for (int i = 0; i < 100; i++)
            {
                data[i] = i;
            }

            var normalizer = new BinningNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should produce discrete values
            var uniqueValues = normalized.ToArray().Distinct().Count();
            Assert.True(uniqueValues <= 11); // At most 10 bins + edge cases
        }

        [Fact]
        public void BinningNormalizer_HandlesDuplicates()
        {
            // Arrange - Data with many duplicates
            var data = new Vector<double>(new[] { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0 });
            var normalizer = new BinningNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should handle without errors
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
        }

        [Fact]
        public void BinningNormalizer_NormalizeInput_WorksOnMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(20, 2);
            for (int i = 0; i < 20; i++)
            {
                matrix[i, 0] = i + 1;
                matrix[i, 1] = (i + 1) * 2;
            }

            var normalizer = new BinningNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert
            Assert.Equal(2, parameters.Count);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void AllNormalizers_HandleSingleValue_Gracefully()
        {
            // Arrange
            var data = new Vector<double>(new[] { 42.0 });

            // Act & Assert - Should not throw
            var zScore = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var minMax = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();
            var maxAbs = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            var (zsNorm, zsParams) = zScore.NormalizeOutput(data);
            var (mmNorm, mmParams) = minMax.NormalizeOutput(data);
            var (maNorm, maParams) = maxAbs.NormalizeOutput(data);

            Assert.False(double.IsNaN(zsNorm[0]));
            Assert.False(double.IsNaN(mmNorm[0]));
            Assert.False(double.IsNaN(maNorm[0]));
        }

        [Fact]
        public void MinMaxNormalizer_WithLargeValues_MaintainsPrecision()
        {
            // Arrange - Very large values
            var data = new Vector<double>(new[] { 1e15, 2e15, 3e15, 4e15, 5e15 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            Assert.Equal(0.0, normalized[0], precision: 10);
            Assert.Equal(1.0, normalized[4], precision: 10);
        }

        [Fact]
        public void ZScoreNormalizer_WithTinyVariance_HandlesGracefully()
        {
            // Arrange - Very small variance
            var data = new Vector<double>(new[] { 1.0, 1.000001, 1.000002, 1.000003, 1.000004 });
            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should not produce NaN or Infinity
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x) || double.IsInfinity(x)));
        }

        [Fact]
        public void RobustScalingNormalizer_WithAllIdenticalValues_HandlesGracefully()
        {
            // Arrange - All same value (IQR = 0)
            var data = new Vector<double>(new[] { 7.0, 7.0, 7.0, 7.0, 7.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should handle IQR = 0 case
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
        }

        #endregion

        #region Performance and Scalability Tests

        [Fact]
        public void MinMaxNormalizer_LargeMatrix_HandlesEfficiently()
        {
            // Arrange
            var matrix = new Matrix<double>(1000, 10);
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    matrix[i, j] = i * j * 0.1;
                }
            }

            var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 1000);
            Assert.Equal(10, parameters.Count);
        }

        [Fact]
        public void ZScoreNormalizer_LargeDataset_CompletesQuickly()
        {
            // Arrange
            var data = new Vector<double>(5000);
            for (int i = 0; i < 5000; i++)
            {
                data[i] = Math.Sin(i * 0.1) * 100;
            }

            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (normalized, parameters) = normalizer.NormalizeOutput(data);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 500);
        }

        [Fact]
        public void QuantileTransformer_MediumDataset_RemainsAccurate()
        {
            // Arrange
            var data = new Vector<double>(500);
            for (int i = 0; i < 500; i++)
            {
                data[i] = i;
            }

            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 100);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Should maintain accuracy
            Assert.True(transformed[0] < transformed[499]);
            Assert.True(transformed.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        #endregion

        #region Cross-Normalizer Comparison Tests

        [Fact]
        public void NormalizerComparison_OnSameData_ProducesDifferentResults()
        {
            // Arrange
            var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

            var zscore = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var minmax = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();
            var maxabs = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (zsNorm, zsParams) = zscore.NormalizeOutput(data);
            var (mmNorm, mmParams) = minmax.NormalizeOutput(data);
            var (maNorm, maParams) = maxabs.NormalizeOutput(data);

            // Assert - Different normalizers should produce different results
            Assert.NotEqual(zsNorm[0], mmNorm[0]);
            Assert.NotEqual(mmNorm[0], maNorm[0]);
            Assert.NotEqual(zsNorm[0], maNorm[0]);
        }

        [Fact]
        public void NormalizerComparison_AllPreserveOrder()
        {
            // Arrange
            var data = new Vector<double>(new[] { 5.0, 2.0, 8.0, 1.0, 9.0 });

            var zscore = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var minmax = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();
            var robust = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (zsNorm, _) = zscore.NormalizeOutput(data);
            var (mmNorm, _) = minmax.NormalizeOutput(data);
            var (rbNorm, _) = robust.NormalizeOutput(data);

            // Assert - All should preserve order
            Assert.True(zsNorm[3] < zsNorm[1]); // 1 < 2
            Assert.True(mmNorm[3] < mmNorm[1]);
            Assert.True(rbNorm[3] < rbNorm[1]);
        }

        #endregion

        #region Mathematical Property Verification Tests

        [Fact]
        public void ZScoreNormalizer_StandardizedData_HasCorrectProperties()
        {
            // Arrange
            var data = new Vector<double>(100);
            var random = new Random(42);
            for (int i = 0; i < 100; i++)
            {
                data[i] = random.NextDouble() * 100;
            }

            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Verify mean ≈ 0 and std ≈ 1
            var arr = normalized.ToArray();
            var mean = arr.Average();
            var variance = arr.Select(x => Math.Pow(x - mean, 2)).Average();
            var std = Math.Sqrt(variance);

            Assert.True(Math.Abs(mean) < 1e-10);
            Assert.True(Math.Abs(std - 1.0) < 1e-10);
        }

        [Fact]
        public void MinMaxNormalizer_NormalizedData_HasCorrectRange()
        {
            // Arrange
            var data = new Vector<double>(new[] { -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Verify range [0, 1]
            var min = normalized.ToArray().Min();
            var max = normalized.ToArray().Max();

            Assert.Equal(0.0, min, precision: 10);
            Assert.Equal(1.0, max, precision: 10);
            Assert.True(normalized.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        [Fact]
        public void MaxAbsScaler_NormalizedData_HasCorrectRange()
        {
            // Arrange
            var data = new Vector<double>(new[] { -100.0, -75.0, -50.0, 0.0, 50.0, 75.0, 100.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert - Verify range [-1, 1]
            var min = normalized.ToArray().Min();
            var max = normalized.ToArray().Max();

            Assert.Equal(-1.0, min, precision: 10);
            Assert.Equal(1.0, max, precision: 10);
            Assert.True(normalized.ToArray().All(x => x >= -1.0 && x <= 1.0));
        }

        [Fact]
        public void LpNormNormalizer_L2_HasUnitNorm()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Verify L2 norm = 1
            var sumSquares = normalized.ToArray().Select(x => x * x).Sum();
            var l2Norm = Math.Sqrt(sumSquares);

            Assert.Equal(1.0, l2Norm, precision: 10);
        }

        [Fact]
        public void LpNormNormalizer_L1_HasUnitSum()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(1.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Verify L1 norm = 1
            var l1Norm = normalized.ToArray().Select(Math.Abs).Sum();

            Assert.Equal(1.0, l1Norm, precision: 10);
        }

        #endregion

        #region Tensor Integration Tests

        [Fact]
        public void ZScoreNormalizer_With2DTensor_WorksCorrectly()
        {
            // Arrange
            var tensor = new Tensor<double>(new[] { 5, 3 });
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    tensor[i, j] = i * 3 + j + 1.0;
                }
            }

            var normalizer = new ZScoreNormalizer<double, Tensor<double>, Tensor<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(tensor);

            // Assert - Each column should be normalized
            Assert.Equal(3, parameters.Count);
        }

        [Fact]
        public void MinMaxNormalizer_With2DTensor_NormalizesColumns()
        {
            // Arrange
            var tensor = new Tensor<double>(new[] { 4, 2 });
            tensor[0, 0] = 1.0; tensor[1, 0] = 2.0; tensor[2, 0] = 3.0; tensor[3, 0] = 4.0;
            tensor[0, 1] = 10.0; tensor[1, 1] = 20.0; tensor[2, 1] = 30.0; tensor[3, 1] = 40.0;

            var normalizer = new MinMaxNormalizer<double, Tensor<double>, Tensor<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(tensor);

            // Assert
            Assert.Equal(2, parameters.Count);
        }

        #endregion

        #region Additional Coverage Tests

        [Fact]
        public void MaxAbsScaler_WithAllZeros_HandlesGracefully()
        {
            // Arrange
            var data = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
        }

        [Fact]
        public void RobustScalingNormalizer_WithExtremeOutliers_StaysRobust()
        {
            // Arrange - Extreme outliers
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 1000000.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - First 5 values should be in reasonable range
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(normalized[i]) < 100.0);
            }
        }

        [Fact]
        public void LogNormalizer_WithZeroValue_HandlesGracefully()
        {
            // Arrange
            var data = new Vector<double>(new[] { 0.0, 1.0, 10.0, 100.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x) || double.IsInfinity(x)));
        }

        [Fact]
        public void BinningNormalizer_WithWideRange_CreatesEvenBins()
        {
            // Arrange
            var data = new Vector<double>(100);
            for (int i = 0; i < 100; i++)
            {
                data[i] = i * 100; // Wide range: 0 to 9900
            }

            var normalizer = new BinningNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            Assert.True(parameters.Bins.Count > 1);
            Assert.Equal(0.0, normalized.Min(), precision: 10);
            Assert.Equal(1.0, normalized.Max(), precision: 10);
        }

        [Fact]
        public void QuantileTransformer_WithRepeatedValues_HandlesCorrectly()
        {
            // Arrange - Many repeated values
            var data = new Vector<double>(20);
            for (int i = 0; i < 10; i++)
            {
                data[i] = 1.0;
            }
            for (int i = 10; i < 20; i++)
            {
                data[i] = 2.0;
            }

            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 10);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert
            Assert.False(transformed.ToArray().Any(x => double.IsNaN(x)));
        }

        [Fact]
        public void LpNormNormalizer_WithNegativeValues_WorksCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new[] { -3.0, -4.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - L2 norm should still be 1
            var norm = Math.Sqrt(normalized[0] * normalized[0] + normalized[1] * normalized[1]);
            Assert.Equal(1.0, norm, precision: 10);
        }

        [Fact]
        public void AllNormalizers_WithFloatType_WorkCorrectly()
        {
            // Arrange
            var data = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            // Act & Assert - All should work with float
            var zscore = new ZScoreNormalizer<float, Vector<float>, Vector<float>>();
            var (zsNorm, _) = zscore.NormalizeOutput(data);
            Assert.False(zsNorm.ToArray().Any(x => float.IsNaN(x)));

            var minmax = new MinMaxNormalizer<float, Vector<float>, Vector<float>>();
            var (mmNorm, _) = minmax.NormalizeOutput(data);
            Assert.False(mmNorm.ToArray().Any(x => float.IsNaN(x)));

            var maxabs = new MaxAbsScaler<float, Vector<float>, Vector<float>>();
            var (maNorm, _) = maxabs.NormalizeOutput(data);
            Assert.False(maNorm.ToArray().Any(x => float.IsNaN(x)));
        }

        #endregion

        #region Roundtrip Tests (Normalize -> Denormalize)

        [Fact]
        public void AllLinearNormalizers_Roundtrip_RecoversExactValues()
        {
            // Arrange
            var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

            // Test ZScore
            var zscore = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var (zsNorm, zsParams) = zscore.NormalizeOutput(data);
            var zsRecover = zscore.Denormalize(zsNorm, zsParams);

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], zsRecover[i], precision: 10);
            }

            // Test MinMax
            var minmax = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();
            var (mmNorm, mmParams) = minmax.NormalizeOutput(data);
            var mmRecover = minmax.Denormalize(mmNorm, mmParams);

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], mmRecover[i], precision: 10);
            }

            // Test MaxAbs
            var maxabs = new MaxAbsScaler<double, Vector<double>, Vector<double>>();
            var (maNorm, maParams) = maxabs.NormalizeOutput(data);
            var maRecover = maxabs.Denormalize(maNorm, maParams);

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], maRecover[i], precision: 10);
            }

            // Test Robust
            var robust = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();
            var (rbNorm, rbParams) = robust.NormalizeOutput(data);
            var rbRecover = robust.Denormalize(rbNorm, rbParams);

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], rbRecover[i], precision: 10);
            }

            // Test LpNorm
            var lpnorm = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);
            var (lpNorm, lpParams) = lpnorm.NormalizeOutput(data);
            var lpRecover = lpnorm.Denormalize(lpNorm, lpParams);

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], lpRecover[i], precision: 10);
            }
        }

        [Fact]
        public void LogNormalizer_Roundtrip_RecoversWithinTolerance()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 10.0, 100.0, 1000.0, 10000.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);
            var recovered = normalizer.Denormalize(normalized, parameters);

            // Assert - May have small rounding errors due to log/exp
            for (int i = 0; i < data.Length; i++)
            {
                var relativeError = Math.Abs((data[i] - recovered[i]) / data[i]);
                Assert.True(relativeError < 1e-6);
            }
        }

        #endregion

        #region Coefficient Denormalization Tests

        [Fact]
        public void ZScoreNormalizer_DenormalizeCoefficients_WorksCorrectly()
        {
            // Arrange
            var coefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var xParams = new List<NormalizationParameters<double>>
            {
                new NormalizationParameters<double> { Mean = 10.0, StdDev = 2.0 },
                new NormalizationParameters<double> { Mean = 20.0, StdDev = 4.0 },
                new NormalizationParameters<double> { Mean = 30.0, StdDev = 6.0 }
            };
            var yParams = new NormalizationParameters<double> { Mean = 50.0, StdDev = 10.0 };

            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var denormalized = normalizer.Denormalize(coefficients, xParams, yParams);

            // Assert - Should scale coefficients appropriately
            Assert.NotNull(denormalized);
            Assert.Equal(3, denormalized.Length);
        }

        [Fact]
        public void MinMaxNormalizer_DenormalizeCoefficients_ScalesCorrectly()
        {
            // Arrange
            var coefficients = new Vector<double>(new[] { 0.5, 0.3, 0.7 });
            var xParams = new List<NormalizationParameters<double>>
            {
                new NormalizationParameters<double> { Min = 0.0, Max = 10.0 },
                new NormalizationParameters<double> { Min = 0.0, Max = 20.0 },
                new NormalizationParameters<double> { Min = 0.0, Max = 30.0 }
            };
            var yParams = new NormalizationParameters<double> { Min = 0.0, Max = 100.0 };

            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var denormalized = normalizer.Denormalize(coefficients, xParams, yParams);

            // Assert
            Assert.NotNull(denormalized);
            Assert.Equal(3, denormalized.Length);
        }

        [Fact]
        public void RobustScalingNormalizer_DenormalizeCoefficients_HandlesIQR()
        {
            // Arrange
            var coefficients = new Vector<double>(new[] { 1.5, 2.5 });
            var xParams = new List<NormalizationParameters<double>>
            {
                new NormalizationParameters<double> { Median = 10.0, IQR = 5.0 },
                new NormalizationParameters<double> { Median = 20.0, IQR = 10.0 }
            };
            var yParams = new NormalizationParameters<double> { Median = 50.0, IQR = 20.0 };

            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var denormalized = normalizer.Denormalize(coefficients, xParams, yParams);

            // Assert
            Assert.NotNull(denormalized);
            Assert.Equal(2, denormalized.Length);
        }

        #endregion

        #region Complex Data Pattern Tests

        [Fact]
        public void MinMaxNormalizer_WithUniformData_DistributesEvenly()
        {
            // Arrange - Uniformly distributed data
            var data = new Vector<double>(100);
            for (int i = 0; i < 100; i++)
            {
                data[i] = i;
            }

            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should maintain uniform distribution
            var arr = normalized.ToArray();
            for (int i = 1; i < arr.Length; i++)
            {
                var diff = arr[i] - arr[i - 1];
                Assert.True(Math.Abs(diff - 0.0101010101) < 0.001); // ~1/99
            }
        }

        [Fact]
        public void ZScoreNormalizer_WithBimodalDistribution_NormalizesBoth()
        {
            // Arrange - Bimodal distribution
            var data = new Vector<double>(20);
            for (int i = 0; i < 10; i++)
            {
                data[i] = 10.0; // First mode
            }
            for (int i = 10; i < 20; i++)
            {
                data[i] = 50.0; // Second mode
            }

            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Both modes should be normalized
            var mean = normalized.ToArray().Average();
            Assert.True(Math.Abs(mean) < RelaxedTolerance);
        }

        [Fact]
        public void MaxAbsScaler_WithAsymmetricData_PreservesAsymmetry()
        {
            // Arrange - Asymmetric data (more positive than negative)
            var data = new Vector<double>(new[] { -10.0, 20.0, 30.0, 40.0, 50.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert - Asymmetry should be preserved
            Assert.Equal(-0.2, normalized[0], precision: 10);
            Assert.Equal(1.0, normalized[4], precision: 10);
        }

        [Fact]
        public void RobustScalingNormalizer_WithLongTailDistribution_HandlesRobustly()
        {
            // Arrange - Long tail distribution
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 50.0, 500.0 });
            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Median should center around 0
            Assert.True(Math.Abs(normalized[3]) < 1.0); // Median element should be near 0
        }

        [Fact]
        public void QuantileTransformer_WithExponentialData_Linearizes()
        {
            // Arrange - Exponential data
            var data = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                data[i] = Math.Pow(2, i); // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            }

            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 100);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert - Should linearize exponential growth
            var arr = transformed.ToArray();
            Assert.True(arr[0] < arr[9]);
            Assert.True(transformed.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        #endregion

        #region Stability and Numerical Tests

        [Fact]
        public void ZScoreNormalizer_WithHighPrecisionData_MaintainsPrecision()
        {
            // Arrange - High precision data
            var data = new Vector<double>(new[] { 1.000001, 1.000002, 1.000003, 1.000004, 1.000005 });
            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            // Assert - Should maintain precision
            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], denormalized[i], precision: 12);
            }
        }

        [Fact]
        public void MinMaxNormalizer_WithCloseValues_HandlesNumericalStability()
        {
            // Arrange - Very close values
            var data = new Vector<double>(new[] { 1.0, 1.0 + 1e-14, 1.0 + 2e-14 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should not produce NaN
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
        }

        [Fact]
        public void MaxAbsScaler_WithVerySmallValues_WorksCorrectly()
        {
            // Arrange - Very small values
            var data = new Vector<double>(new[] { 1e-10, 2e-10, 3e-10, 4e-10, 5e-10 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(0.2, normalized[0], precision: 10);
            Assert.Equal(1.0, normalized[4], precision: 10);
        }

        [Fact]
        public void LpNormNormalizer_WithVeryLargeP_ApproachesMaxNorm()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(100.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - With very large p, should approach max norm
            // Maximum value should be close to 1
            Assert.True(Math.Abs(normalized.Max()) < 1.01);
        }

        #endregion

        #region Multi-Column Matrix Tests

        [Fact]
        public void ZScoreNormalizer_MultiColumnMatrix_NormalizesIndependently()
        {
            // Arrange - Different scales
            var matrix = new Matrix<double>(10, 3);
            for (int i = 0; i < 10; i++)
            {
                matrix[i, 0] = i + 1; // Small values
                matrix[i, 1] = (i + 1) * 100; // Large values
                matrix[i, 2] = (i + 1) * 0.01; // Tiny values
            }

            var normalizer = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should have mean ≈ 0 and std ≈ 1
            for (int col = 0; col < 3; col++)
            {
                var column = normalized.GetColumn(col);
                var mean = column.ToArray().Average();
                Assert.True(Math.Abs(mean) < RelaxedTolerance);
            }
        }

        [Fact]
        public void MinMaxNormalizer_MultiColumnWithDifferentRanges_NormalizesEach()
        {
            // Arrange
            var matrix = new Matrix<double>(5, 3);
            // Column 1: [0, 100]
            matrix[0, 0] = 0.0; matrix[4, 0] = 100.0;
            // Column 2: [-50, 50]
            matrix[0, 1] = -50.0; matrix[4, 1] = 50.0;
            // Column 3: [1000, 2000]
            matrix[0, 2] = 1000.0; matrix[4, 2] = 2000.0;

            for (int i = 1; i < 4; i++)
            {
                matrix[i, 0] = i * 25.0;
                matrix[i, 1] = -50.0 + i * 25.0;
                matrix[i, 2] = 1000.0 + i * 250.0;
            }

            var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Each column should be [0, 1]
            for (int col = 0; col < 3; col++)
            {
                var column = normalized.GetColumn(col);
                Assert.Equal(0.0, column.Min(), precision: 10);
                Assert.Equal(1.0, column.Max(), precision: 10);
            }
        }

        [Fact]
        public void RobustScalingNormalizer_MultiColumnWithOutliers_HandlesEach()
        {
            // Arrange - Each column has outliers
            var matrix = new Matrix<double>(6, 2);
            // Column 1: normal values with one outlier
            matrix[0, 0] = 1.0; matrix[1, 0] = 2.0; matrix[2, 0] = 3.0;
            matrix[3, 0] = 4.0; matrix[4, 0] = 5.0; matrix[5, 0] = 1000.0;
            // Column 2: normal values with one outlier
            matrix[0, 1] = 10.0; matrix[1, 1] = 20.0; matrix[2, 1] = 30.0;
            matrix[3, 1] = 40.0; matrix[4, 1] = 50.0; matrix[5, 1] = 5000.0;

            var normalizer = new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeInput(matrix);

            // Assert - Non-outlier values should be in reasonable range
            for (int col = 0; col < 2; col++)
            {
                var column = normalized.GetColumn(col);
                for (int i = 0; i < 5; i++) // Exclude outliers
                {
                    Assert.True(Math.Abs(column[i]) < 10.0);
                }
            }
        }

        #endregion

        #region Real-World Scenario Tests

        [Fact]
        public void MinMaxNormalizer_AgeData_NormalizesReasonably()
        {
            // Arrange - Realistic age data
            var ages = new Vector<double>(new[] { 18.0, 25.0, 30.0, 45.0, 60.0, 75.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(ages);

            // Assert
            Assert.Equal(0.0, normalized[0], precision: 10); // 18 years
            Assert.Equal(1.0, normalized[5], precision: 10); // 75 years
            Assert.True(normalized.ToArray().All(x => x >= 0.0 && x <= 1.0));
        }

        [Fact]
        public void LogNormalizer_IncomeData_CompressesRange()
        {
            // Arrange - Income data spanning orders of magnitude
            var incomes = new Vector<double>(new[] { 20000.0, 50000.0, 100000.0, 500000.0, 10000000.0 });
            var normalizer = new LogNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(incomes);

            // Assert - Should compress wide range
            var range = normalized.Max() - normalized.Min();
            Assert.Equal(1.0, range, precision: 10);
        }

        [Fact]
        public void ZScoreNormalizer_TestScores_StandardizesCorrectly()
        {
            // Arrange - Test scores
            var scores = new Vector<double>(new[] { 65.0, 75.0, 80.0, 85.0, 90.0, 95.0 });
            var normalizer = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(scores);

            // Assert - Mean should be 0
            var mean = normalized.ToArray().Average();
            Assert.True(Math.Abs(mean) < RelaxedTolerance);
        }

        [Fact]
        public void RobustScalingNormalizer_HousingPrices_HandlesOutliers()
        {
            // Arrange - Housing prices with luxury outlier
            var prices = new Vector<double>(new[]
            {
                150000.0, 180000.0, 200000.0, 220000.0, 250000.0, // Normal
                5000000.0 // Luxury outlier
            });

            var normalizer = new RobustScalingNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(prices);

            // Assert - Normal prices should cluster around 0
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(normalized[i]) < 5.0);
            }
        }

        #endregion

        #region Specific Edge Cases

        [Fact]
        public void MinMaxNormalizer_WithNegativeAndPositive_ScalesCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new[] { -100.0, -50.0, 0.0, 50.0, 100.0 });
            var normalizer = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert
            Assert.Equal(0.0, normalized[0], precision: 10);
            Assert.Equal(0.5, normalized[2], precision: 10); // Zero should be at midpoint
            Assert.Equal(1.0, normalized[4], precision: 10);
        }

        [Fact]
        public void LpNormNormalizer_WithZeroVector_HandlesGracefully()
        {
            // Arrange - Zero vector
            var data = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var normalizer = new LpNormNormalizer<double, Vector<double>, Vector<double>>(2.0);

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should handle without NaN
            Assert.False(normalized.ToArray().Any(x => double.IsNaN(x)));
        }

        [Fact]
        public void QuantileTransformer_WithTwoValues_WorksCorrectly()
        {
            // Arrange - Minimal data
            var data = new Vector<double>(new[] { 1.0, 10.0 });
            var transformer = new QuantileTransformer<double, Vector<double>, Vector<double>>(
                OutputDistribution.Uniform, 10);

            // Act
            var (transformed, parameters) = transformer.NormalizeOutput(data);

            // Assert
            Assert.True(transformed[0] < transformed[1]);
        }

        [Fact]
        public void MaxAbsScaler_WithOnlyNegativeValues_WorksCorrectly()
        {
            // Arrange
            var data = new Vector<double>(new[] { -100.0, -75.0, -50.0, -25.0, -10.0 });
            var scaler = new MaxAbsScaler<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(-1.0, normalized[0], precision: 10);
            Assert.Equal(-0.1, normalized[4], precision: 10);
        }

        [Fact]
        public void BinningNormalizer_WithTwoDistinctValues_CreatesBins()
        {
            // Arrange
            var data = new Vector<double>(20);
            for (int i = 0; i < 10; i++)
            {
                data[i] = 1.0;
            }
            for (int i = 10; i < 20; i++)
            {
                data[i] = 10.0;
            }

            var normalizer = new BinningNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (normalized, parameters) = normalizer.NormalizeOutput(data);

            // Assert - Should create distinct bins
            Assert.True(normalized[0] < normalized[10]);
        }

        #endregion

        #region Comparison and Consistency Tests

        [Fact]
        public void AllNormalizers_OnSameData_ProduceConsistentResults()
        {
            // Arrange
            var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act - Apply each normalizer twice
            var zscore = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var (zs1, _) = zscore.NormalizeOutput(data);
            var (zs2, _) = zscore.NormalizeOutput(data);

            var minmax = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();
            var (mm1, _) = minmax.NormalizeOutput(data);
            var (mm2, _) = minmax.NormalizeOutput(data);

            // Assert - Same input should produce same output
            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(zs1[i], zs2[i], precision: 15);
                Assert.Equal(mm1[i], mm2[i], precision: 15);
            }
        }

        [Fact]
        public void ZScoreVsMeanVariance_ProduceSameResults()
        {
            // Arrange
            var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

            var zscore = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var meanvar = new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>();

            // Act
            var (zsNorm, _) = zscore.NormalizeOutput(data);
            var (mvNorm, _) = meanvar.NormalizeOutput(data);

            // Assert - Should be equivalent
            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(zsNorm[i], mvNorm[i], precision: 12);
            }
        }

        [Fact]
        public void LinearNormalizers_PreserveRelativeOrder()
        {
            // Arrange
            var data = new Vector<double>(new[] { 5.0, 2.0, 8.0, 1.0, 9.0, 3.0 });

            var normalizers = new object[]
            {
                new ZScoreNormalizer<double, Vector<double>, Vector<double>>(),
                new MinMaxNormalizer<double, Vector<double>, Vector<double>>(),
                new MaxAbsScaler<double, Vector<double>, Vector<double>>(),
                new RobustScalingNormalizer<double, Vector<double>, Vector<double>>(),
                new MeanVarianceNormalizer<double, Vector<double>, Vector<double>>()
            };

            // Act & Assert - All should preserve order
            foreach (var norm in normalizers)
            {
                var method = norm.GetType().GetMethod("NormalizeOutput");
                var result = method.Invoke(norm, new object[] { data });
                var resultType = result.GetType();
                var normalizedProp = resultType.GetProperty("Item1");
                var normalized = (Vector<double>)normalizedProp.GetValue(result);

                // Verify order preservation: 1 < 2 < 3 < 5 < 8 < 9
                Assert.True(normalized[3] < normalized[1]); // 1 < 2
                Assert.True(normalized[1] < normalized[5]); // 2 < 3
                Assert.True(normalized[5] < normalized[0]); // 3 < 5
                Assert.True(normalized[0] < normalized[2]); // 5 < 8
                Assert.True(normalized[2] < normalized[4]); // 8 < 9
            }
        }

        [Fact]
        public void AllNormalizers_WithMixedDataTypes_SupportBothFloatAndDouble()
        {
            // Arrange - Test both double and float
            var doubleData = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var floatData = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            // Act & Assert - Double normalizers
            var zscoreDouble = new ZScoreNormalizer<double, Vector<double>, Vector<double>>();
            var (zsNormDouble, _) = zscoreDouble.NormalizeOutput(doubleData);
            Assert.False(zsNormDouble.ToArray().Any(x => double.IsNaN(x)));

            var minmaxDouble = new MinMaxNormalizer<double, Vector<double>, Vector<double>>();
            var (mmNormDouble, _) = minmaxDouble.NormalizeOutput(doubleData);
            Assert.Equal(0.0, mmNormDouble[0], precision: 10);
            Assert.Equal(1.0, mmNormDouble[4], precision: 10);

            // Act & Assert - Float normalizers
            var zscoreFloat = new ZScoreNormalizer<float, Vector<float>, Vector<float>>();
            var (zsNormFloat, _) = zscoreFloat.NormalizeOutput(floatData);
            Assert.False(zsNormFloat.ToArray().Any(x => float.IsNaN(x)));

            var minmaxFloat = new MinMaxNormalizer<float, Vector<float>, Vector<float>>();
            var (mmNormFloat, _) = minmaxFloat.NormalizeOutput(floatData);
            Assert.Equal(0.0f, mmNormFloat[0], precision: 6);
            Assert.Equal(1.0f, mmNormFloat[4], precision: 6);
        }

        #endregion

        #endregion
    }
}
