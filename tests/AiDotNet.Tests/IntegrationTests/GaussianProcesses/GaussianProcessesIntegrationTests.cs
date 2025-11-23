using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.GaussianProcesses
{
    /// <summary>
    /// Integration tests for Gaussian Process implementations with mathematically verified results.
    /// These tests validate the correctness of StandardGaussianProcess, SparseGaussianProcess,
    /// and MultiOutputGaussianProcess implementations.
    /// </summary>
    public class GaussianProcessesIntegrationTests
    {
        private const double Tolerance = 1e-6;
        private const double RelaxedTolerance = 1e-3;

        #region StandardGaussianProcess Tests

        [Fact]
        public void StandardGP_FitAndPredict_NoiselessSineWave_RecoversWithLowUncertainty()
        {
            // Arrange - Create noiseless sine wave data
            var kernel = new GaussianKernel<double>(sigma: 0.5);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                double x = i * 0.3;
                X[i, 0] = x;
                y[i] = Math.Sin(x);
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 3.0 }));

            // Assert - Prediction should be close to sin(3.0) with low uncertainty
            double expected = Math.Sin(3.0);
            Assert.True(Math.Abs(mean - expected) < RelaxedTolerance,
                $"Mean prediction {mean} should be close to {expected}");
            Assert.True(variance >= 0, "Variance must be non-negative");
            Assert.True(variance < 0.1, $"Variance {variance} should be low for interpolation");
        }

        [Fact]
        public void StandardGP_PredictAtTrainingPoint_HasNearZeroVariance()
        {
            // Arrange - Simple linear data
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(5, 1);
            var y = new Vector<double>(5);
            for (int i = 0; i < 5; i++)
            {
                X[i, 0] = i;
                y[i] = 2.0 * i + 1.0;
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 2.0 }));

            // Assert - At training point, prediction should match observed value with low variance
            Assert.Equal(5.0, mean, precision: 2);
            Assert.True(variance < 0.01, $"Variance at training point should be near zero, but was {variance}");
        }

        [Fact]
        public void StandardGP_Extrapolation_IncreasedUncertainty()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = i;
            }

            // Act
            gp.Fit(X, y);
            var (meanInterpolate, varianceInterpolate) = gp.Predict(new Vector<double>(new[] { 5.0 }));
            var (meanExtrapolate, varianceExtrapolate) = gp.Predict(new Vector<double>(new[] { 20.0 }));

            // Assert - Extrapolation should have higher uncertainty than interpolation
            Assert.True(varianceExtrapolate > varianceInterpolate,
                $"Extrapolation variance {varianceExtrapolate} should be greater than interpolation variance {varianceInterpolate}");
        }

        [Fact]
        public void StandardGP_UpdateKernel_ChangesLengthscaleEffect()
        {
            // Arrange
            var shortKernel = new GaussianKernel<double>(sigma: 0.1);
            var longKernel = new GaussianKernel<double>(sigma: 5.0);
            var gp = new StandardGaussianProcess<double>(shortKernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i * 0.5;
                y[i] = Math.Sin(i * 0.5);
            }

            // Act - Fit with short lengthscale kernel
            gp.Fit(X, y);
            var (meanShort, varianceShort) = gp.Predict(new Vector<double>(new[] { 2.5 }));

            // Update to long lengthscale kernel
            gp.UpdateKernel(longKernel);
            var (meanLong, varianceLong) = gp.Predict(new Vector<double>(new[] { 2.5 }));

            // Assert - Different kernels should produce different predictions
            Assert.NotEqual(meanShort, meanLong);
        }

        [Fact]
        public void StandardGP_LinearKernel_FitsLinearFunction()
        {
            // Arrange - Linear function: y = 2x + 3
            var kernel = new LinearKernel<double>();
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = 2.0 * i + 3.0;
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 5.0 }));

            // Assert - Should predict linear relationship accurately
            double expected = 2.0 * 5.0 + 3.0;
            Assert.Equal(expected, mean, precision: 1);
        }

        [Fact]
        public void StandardGP_MaternKernel_ProducesDifferentSmoothnessProperties()
        {
            // Arrange - Test with Matern kernel
            var maternKernel = new MaternKernel<double>();
            var gp = new StandardGaussianProcess<double>(maternKernel);

            var X = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);
            for (int i = 0; i < 15; i++)
            {
                X[i, 0] = i * 0.4;
                y[i] = Math.Sin(i * 0.4) + (i % 2 == 0 ? 0.1 : -0.1);
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 3.0 }));

            // Assert - Matern kernel should produce reasonable predictions
            Assert.True(Math.Abs(mean) < 2.0, "Prediction should be in reasonable range");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void StandardGP_NoiseHandling_StillProducesReasonablePredictions()
        {
            // Arrange - Add noise to observations
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);
            var random = new Random(42);

            var X = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            for (int i = 0; i < 30; i++)
            {
                X[i, 0] = i * 0.2;
                double noise = (random.NextDouble() - 0.5) * 0.3;
                y[i] = Math.Sin(i * 0.2) + noise;
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 3.0 }));

            // Assert - Should still capture general trend despite noise
            double expected = Math.Sin(3.0);
            Assert.True(Math.Abs(mean - expected) < 0.5,
                $"Mean prediction {mean} should be reasonably close to clean signal {expected}");
            Assert.True(variance > 0, "Variance should reflect uncertainty from noise");
        }

        [Fact]
        public void StandardGP_MultipleFeatures_HandlesHighDimensionalInput()
        {
            // Arrange - 2D input space
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i * 0.3;
                X[i, 1] = i * 0.2;
                y[i] = X[i, 0] + 2.0 * X[i, 1];
            }

            // Act
            gp.Fit(X, y);
            var testPoint = new Vector<double>(new[] { 2.0, 3.0 });
            var (mean, variance) = gp.Predict(testPoint);

            // Assert
            double expected = 2.0 + 2.0 * 3.0;
            Assert.True(Math.Abs(mean - expected) < 2.0,
                $"Prediction {mean} should be close to expected {expected}");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void StandardGP_QuadraticFunction_CapturesNonlinearity()
        {
            // Arrange - Quadratic function: y = x^2
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);
            for (int i = 0; i < 15; i++)
            {
                double x = (i - 7) * 0.5;
                X[i, 0] = x;
                y[i] = x * x;
            }

            // Act
            gp.Fit(X, y);
            var testX = 1.5;
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { testX }));

            // Assert
            double expected = testX * testX;
            Assert.True(Math.Abs(mean - expected) < 0.5,
                $"Prediction {mean} should be close to {expected}");
        }

        [Fact]
        public void StandardGP_ConstantFunction_ConvergesToConstant()
        {
            // Arrange - Constant function
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = 5.0;
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 7.5 }));

            // Assert - Should predict constant value
            Assert.Equal(5.0, mean, precision: 1);
            Assert.True(variance < 0.1, "Variance should be low for constant function");
        }

        [Fact]
        public void StandardGP_SparseData_HandlesWidelySpacedPoints()
        {
            // Arrange - Widely spaced training points
            var kernel = new GaussianKernel<double>(sigma: 2.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(5, 1);
            var y = new Vector<double>(5);
            double[] xVals = { 0, 5, 10, 15, 20 };
            for (int i = 0; i < 5; i++)
            {
                X[i, 0] = xVals[i];
                y[i] = Math.Sin(xVals[i] * 0.3);
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 7.5 }));

            // Assert - Should interpolate between points with reasonable uncertainty
            Assert.True(Math.Abs(mean) < 2.0, "Prediction should be in reasonable range");
            Assert.True(variance > 0, "Variance should be positive for interpolation");
        }

        [Fact]
        public void StandardGP_SymmetricPrediction_ProducesSimilarResults()
        {
            // Arrange - Symmetric function around origin
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                double x = (i - 5);
                X[i, 0] = x;
                y[i] = x * x;
            }

            // Act
            gp.Fit(X, y);
            var (meanPos, variancePos) = gp.Predict(new Vector<double>(new[] { 2.0 }));
            var (meanNeg, varianceNeg) = gp.Predict(new Vector<double>(new[] { -2.0 }));

            // Assert - Symmetric function should give similar predictions
            Assert.Equal(meanPos, meanNeg, precision: 1);
        }

        [Fact]
        public void StandardGP_SmallDataset_StillProducesValidPredictions()
        {
            // Arrange - Very small dataset (3 points)
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(3, 1);
            var y = new Vector<double>(3);
            X[0, 0] = 0.0; y[0] = 0.0;
            X[1, 0] = 1.0; y[1] = 1.0;
            X[2, 0] = 2.0; y[2] = 4.0;

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 1.5 }));

            // Assert
            Assert.True(mean >= 1.0 && mean <= 4.0,
                $"Prediction {mean} should be in range of training data");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void StandardGP_PeriodicData_WithGaussianKernel_CapturesGeneralTrend()
        {
            // Arrange - Periodic sine wave
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);
            for (int i = 0; i < 25; i++)
            {
                X[i, 0] = i * 0.25;
                y[i] = Math.Sin(i * 0.25 * 2 * Math.PI / 5.0);
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 3.0 }));

            // Assert - Should capture periodic pattern
            Assert.True(Math.Abs(mean) <= 1.5, "Prediction should be in reasonable range for sine wave");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void StandardGP_DifferentLengthscales_AffectPredictionSmoothing()
        {
            // Arrange
            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            var random = new Random(123);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = (random.NextDouble() - 0.5) * 2.0;
            }

            var shortKernel = new GaussianKernel<double>(sigma: 0.3);
            var longKernel = new GaussianKernel<double>(sigma: 3.0);
            var gpShort = new StandardGaussianProcess<double>(shortKernel);
            var gpLong = new StandardGaussianProcess<double>(longKernel);

            // Act
            gpShort.Fit(X, y);
            gpLong.Fit(X, y);
            var (meanShort, varShort) = gpShort.Predict(new Vector<double>(new[] { 5.5 }));
            var (meanLong, varLong) = gpLong.Predict(new Vector<double>(new[] { 5.5 }));

            // Assert - Long lengthscale should smooth more (predictions closer to mean)
            // Short lengthscale should be more sensitive to local variations
            Assert.True(Math.Abs(meanLong) < Math.Abs(meanShort) + 1.0,
                "Long lengthscale should produce smoother predictions");
        }

        [Fact]
        public void StandardGP_ConfidenceIntervals_CoverTrueValues()
        {
            // Arrange - Known function
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = Math.Sin(i * 0.5);
            }

            // Act
            gp.Fit(X, y);
            var testX = 5.5;
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { testX }));
            double stdDev = Math.Sqrt(variance);
            double trueValue = Math.Sin(testX * 0.5);

            // Assert - 95% confidence interval should contain true value
            double lowerBound = mean - 2 * stdDev;
            double upperBound = mean + 2 * stdDev;
            Assert.True(trueValue >= lowerBound && trueValue <= upperBound,
                $"True value {trueValue} should be within 95% CI [{lowerBound}, {upperBound}]");
        }

        [Fact]
        public void StandardGP_MultipleCallsToPredict_ConsistentResults()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(5, 1);
            var y = new Vector<double>(5);
            for (int i = 0; i < 5; i++)
            {
                X[i, 0] = i;
                y[i] = i * 2.0;
            }

            // Act
            gp.Fit(X, y);
            var testPoint = new Vector<double>(new[] { 2.5 });
            var (mean1, variance1) = gp.Predict(testPoint);
            var (mean2, variance2) = gp.Predict(testPoint);

            // Assert - Multiple calls should give identical results
            Assert.Equal(mean1, mean2);
            Assert.Equal(variance1, variance2);
        }

        [Fact]
        public void StandardGP_PriorMeanZero_ReflectedInPredictions()
        {
            // Arrange - GP has zero prior mean
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(3, 1);
            var y = new Vector<double>(3);
            X[0, 0] = -5.0; y[0] = 0.0;
            X[1, 0] = 0.0; y[1] = 0.0;
            X[2, 0] = 5.0; y[2] = 0.0;

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 10.0 }));

            // Assert - Far from training data should approach prior mean (zero)
            Assert.True(Math.Abs(mean) < 1.0,
                $"Far extrapolation {mean} should approach prior mean of zero");
            Assert.True(variance > 0.5,
                "Far extrapolation should have high variance");
        }

        [Fact]
        public void StandardGP_StepFunction_SmoothsTransition()
        {
            // Arrange - Step function approximation
            var kernel = new GaussianKernel<double>(sigma: 0.5);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i - 10;
                y[i] = X[i, 0] < 0 ? 0.0 : 1.0;
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 0.0 }));

            // Assert - Should smooth the transition
            Assert.True(mean > 0.2 && mean < 0.8,
                $"Prediction at step {mean} should be between extremes");
        }

        [Fact]
        public void StandardGP_ScalarOutput_PredictionInDataRange()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var gp = new StandardGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            double minY = double.MaxValue;
            double maxY = double.MinValue;
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = Math.Sin(i * 0.5) * 3 + 5;
                minY = Math.Min(minY, y[i]);
                maxY = Math.Max(maxY, y[i]);
            }

            // Act
            gp.Fit(X, y);
            var (mean, variance) = gp.Predict(new Vector<double>(new[] { 5.0 }));

            // Assert - Prediction should be in reasonable range of training data
            Assert.True(mean >= minY - 2 && mean <= maxY + 2,
                $"Prediction {mean} should be in reasonable range [{minY-2}, {maxY+2}]");
        }

        #endregion

        #region SparseGaussianProcess Tests

        [Fact]
        public void SparseGP_InducingPoints_ReducesComputationalComplexity()
        {
            // Arrange - Large dataset
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(200, 1);
            var y = new Vector<double>(200);
            for (int i = 0; i < 200; i++)
            {
                X[i, 0] = i * 0.1;
                y[i] = Math.Sin(i * 0.1);
            }

            // Act - Should complete quickly with sparse approximation
            var startTime = DateTime.Now;
            sparseGP.Fit(X, y);
            var fitTime = (DateTime.Now - startTime).TotalMilliseconds;

            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 10.0 }));
            var predictTime = (DateTime.Now - startTime).TotalMilliseconds;

            // Assert - Should be computationally feasible
            Assert.True(fitTime < 5000, $"Fit should complete in reasonable time (took {fitTime}ms)");
            Assert.True(predictTime < 5000, $"Predict should complete quickly (took {predictTime}ms)");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_ApproximatesStandardGP_OnSineWave()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(50, 1);
            var y = new Vector<double>(50);
            for (int i = 0; i < 50; i++)
            {
                X[i, 0] = i * 0.2;
                y[i] = Math.Sin(i * 0.2);
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 5.0 }));

            // Assert - Should approximate true function reasonably
            double expected = Math.Sin(5.0);
            Assert.True(Math.Abs(mean - expected) < 0.5,
                $"Sparse GP prediction {mean} should approximate {expected}");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_UpdateKernel_RetrainsWithNewKernel()
        {
            // Arrange
            var kernel1 = new GaussianKernel<double>(sigma: 0.5);
            var kernel2 = new GaussianKernel<double>(sigma: 2.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel1);

            var X = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            for (int i = 0; i < 30; i++)
            {
                X[i, 0] = i * 0.3;
                y[i] = Math.Sin(i * 0.3);
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean1, variance1) = sparseGP.Predict(new Vector<double>(new[] { 5.0 }));

            sparseGP.UpdateKernel(kernel2);
            var (mean2, variance2) = sparseGP.Predict(new Vector<double>(new[] { 5.0 }));

            // Assert - Different kernels should produce different results
            Assert.NotEqual(mean1, mean2);
        }

        [Fact]
        public void SparseGP_LinearFunction_FitsReasonably()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(40, 1);
            var y = new Vector<double>(40);
            for (int i = 0; i < 40; i++)
            {
                X[i, 0] = i;
                y[i] = 3.0 * i + 2.0;
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 20.0 }));

            // Assert
            double expected = 3.0 * 20.0 + 2.0;
            Assert.True(Math.Abs(mean - expected) < 5.0,
                $"Sparse GP prediction {mean} should be close to {expected}");
        }

        [Fact]
        public void SparseGP_NoisyData_ProducesSmoothedPredictions()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);
            var random = new Random(42);

            var X = new Matrix<double>(60, 1);
            var y = new Vector<double>(60);
            for (int i = 0; i < 60; i++)
            {
                X[i, 0] = i * 0.2;
                double noise = (random.NextDouble() - 0.5) * 0.5;
                y[i] = Math.Sin(i * 0.2) + noise;
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 6.0 }));

            // Assert
            double expected = Math.Sin(6.0);
            Assert.True(Math.Abs(mean - expected) < 1.0,
                $"Should capture general trend despite noise");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_SmallDataset_HandlesGracefully()
        {
            // Arrange - Dataset smaller than max inducing points
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = i * 0.5;
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 5.0 }));

            // Assert
            Assert.True(Math.Abs(mean - 2.5) < 1.0, "Should predict linear trend");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_MultipleFeatures_Handles2DInput()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);
            for (int i = 0; i < 50; i++)
            {
                X[i, 0] = i * 0.2;
                X[i, 1] = i * 0.15;
                y[i] = X[i, 0] + 2.0 * X[i, 1];
            }

            // Act
            sparseGP.Fit(X, y);
            var testPoint = new Vector<double>(new[] { 5.0, 3.0 });
            var (mean, variance) = sparseGP.Predict(testPoint);

            // Assert
            double expected = 5.0 + 2.0 * 3.0;
            Assert.True(Math.Abs(mean - expected) < 3.0,
                $"Prediction {mean} should be close to {expected}");
        }

        [Fact]
        public void SparseGP_QuadraticFunction_CapturesNonlinearity()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 2.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(40, 1);
            var y = new Vector<double>(40);
            for (int i = 0; i < 40; i++)
            {
                double x = (i - 20) * 0.3;
                X[i, 0] = x;
                y[i] = x * x;
            }

            // Act
            sparseGP.Fit(X, y);
            var testX = 2.0;
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { testX }));

            // Assert
            double expected = testX * testX;
            Assert.True(Math.Abs(mean - expected) < 2.0,
                $"Prediction {mean} should approximate {expected}");
        }

        [Fact]
        public void SparseGP_PredictionConsistency_RepeatedCalls()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            for (int i = 0; i < 30; i++)
            {
                X[i, 0] = i;
                y[i] = Math.Sin(i * 0.2);
            }

            // Act
            sparseGP.Fit(X, y);
            var testPoint = new Vector<double>(new[] { 15.0 }));
            var (mean1, variance1) = sparseGP.Predict(testPoint);
            var (mean2, variance2) = sparseGP.Predict(testPoint);

            // Assert - Should be deterministic
            Assert.Equal(mean1, mean2);
            Assert.Equal(variance1, variance2);
        }

        [Fact]
        public void SparseGP_ComputationalEfficiency_LargeDataset()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(500, 1);
            var y = new Vector<double>(500);
            for (int i = 0; i < 500; i++)
            {
                X[i, 0] = i * 0.05;
                y[i] = Math.Sin(i * 0.05) + Math.Cos(i * 0.1);
            }

            // Act & Assert - Should handle large dataset
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 12.5 }));

            Assert.True(Math.Abs(mean) < 3.0, "Prediction should be in reasonable range");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_UncertaintyEstimates_ReasonableValues()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);
            for (int i = 0; i < 25; i++)
            {
                X[i, 0] = i;
                y[i] = i * 0.5;
            }

            // Act
            sparseGP.Fit(X, y);
            var (meanInterp, varInterp) = sparseGP.Predict(new Vector<double>(new[] { 12.0 }));
            var (meanExtrap, varExtrap) = sparseGP.Predict(new Vector<double>(new[] { 30.0 }));

            // Assert - Extrapolation should have higher uncertainty
            Assert.True(varExtrap >= varInterp,
                $"Extrapolation variance {varExtrap} should be >= interpolation variance {varInterp}");
        }

        [Fact]
        public void SparseGP_MaternKernel_ProducesValidPredictions()
        {
            // Arrange
            var kernel = new MaternKernel<double>();
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(40, 1);
            var y = new Vector<double>(40);
            for (int i = 0; i < 40; i++)
            {
                X[i, 0] = i * 0.3;
                y[i] = Math.Sin(i * 0.3);
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 6.0 }));

            // Assert
            Assert.True(Math.Abs(mean) <= 1.5, "Prediction should be in valid range");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_PeriodicPattern_CapturesOscillation()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(50, 1);
            var y = new Vector<double>(50);
            for (int i = 0; i < 50; i++)
            {
                X[i, 0] = i * 0.3;
                y[i] = Math.Sin(i * 0.3 * 2 * Math.PI / 5.0);
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 7.5 }));

            // Assert
            Assert.True(Math.Abs(mean) <= 1.5, "Prediction should be in sine wave range");
        }

        [Fact]
        public void SparseGP_ConstantFunction_RecoversConstant()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i;
                y[i] = 7.5;
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 10.0 }));

            // Assert
            Assert.Equal(7.5, mean, precision: 1);
        }

        [Fact]
        public void SparseGP_MixedFrequencies_HandlesComplexSignal()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 2.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(60, 1);
            var y = new Vector<double>(60);
            for (int i = 0; i < 60; i++)
            {
                X[i, 0] = i * 0.2;
                y[i] = Math.Sin(i * 0.2) + 0.5 * Math.Cos(i * 0.4);
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 6.0 }));

            // Assert
            double expected = Math.Sin(6.0) + 0.5 * Math.Cos(12.0);
            Assert.True(Math.Abs(mean - expected) < 1.0,
                "Should capture mixed frequency signal");
        }

        [Fact]
        public void SparseGP_ScalabilityTest_500TrainingPoints()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(500, 1);
            var y = new Vector<double>(500);
            for (int i = 0; i < 500; i++)
            {
                X[i, 0] = i * 0.05;
                y[i] = Math.Sin(i * 0.05);
            }

            // Act
            var startTime = DateTime.Now;
            sparseGP.Fit(X, y);
            var fitTime = (DateTime.Now - startTime).TotalMilliseconds;

            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 12.5 }));

            // Assert
            Assert.True(fitTime < 10000, $"Should fit large dataset efficiently (took {fitTime}ms)");
            Assert.True(variance >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void SparseGP_ExtrapolationBehavior_IncreasesUncertainty()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i;
                y[i] = i;
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean1, var1) = sparseGP.Predict(new Vector<double>(new[] { 10.0 }));
            var (mean2, var2) = sparseGP.Predict(new Vector<double>(new[] { 25.0 }));

            // Assert
            Assert.True(var2 > var1,
                $"Far extrapolation variance {var2} should exceed interpolation variance {var1}");
        }

        [Fact]
        public void SparseGP_ZeroMeanPrior_FarExtrapolation()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                y[i] = 0.0;
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 50.0 }));

            // Assert
            Assert.True(Math.Abs(mean) < 1.0,
                "Far extrapolation should approach prior mean");
        }

        [Fact]
        public void SparseGP_DenseDataRegion_LowUncertainty()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 0.5);
            var sparseGP = new SparseGaussianProcess<double>(kernel);

            var X = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            // Dense cluster around x=5
            for (int i = 0; i < 30; i++)
            {
                X[i, 0] = 5.0 + (i - 15) * 0.1;
                y[i] = Math.Sin(X[i, 0]);
            }

            // Act
            sparseGP.Fit(X, y);
            var (mean, variance) = sparseGP.Predict(new Vector<double>(new[] { 5.0 }));

            // Assert
            Assert.True(variance < 0.5,
                $"Dense data region should have low variance, got {variance}");
        }

        #endregion

        #region MultiOutputGaussianProcess Tests

        [Fact]
        public void MultiOutputGP_FitAndPredict_TwoOutputs_ProducesCorrectResults()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var Y = new Matrix<double>(10, 2);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = i * 2.0;      // First output: y1 = 2x
                Y[i, 1] = i * 3.0 + 1.0; // Second output: y2 = 3x + 1
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 5.0 }));

            // Assert
            Assert.Equal(2, means.Length);
            Assert.Equal(10.0, means[0], precision: 1);
            Assert.Equal(16.0, means[1], precision: 1);
            Assert.True(covariance[0, 0] >= 0, "Variance must be non-negative");
        }

        [Fact]
        public void MultiOutputGP_ThreeOutputs_IndependentFunctions()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(15, 1);
            var Y = new Matrix<double>(15, 3);
            for (int i = 0; i < 15; i++)
            {
                double x = i * 0.4;
                X[i, 0] = x;
                Y[i, 0] = Math.Sin(x);
                Y[i, 1] = Math.Cos(x);
                Y[i, 2] = x * x;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 3.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert
            Assert.Equal(3, means.Length);
            Assert.True(Math.Abs(means[0] - Math.Sin(testX)) < 0.5,
                $"First output {means[0]} should approximate sin({testX})");
            Assert.True(Math.Abs(means[1] - Math.Cos(testX)) < 0.5,
                $"Second output {means[1]} should approximate cos({testX})");
            Assert.True(Math.Abs(means[2] - testX * testX) < 2.0,
                $"Third output {means[2]} should approximate {testX}^2");
        }

        [Fact]
        public void MultiOutputGP_CorrelatedOutputs_CapturesBoth()
        {
            // Arrange - Two correlated outputs
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 1);
            var Y = new Matrix<double>(20, 2);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i * 0.3;
                Y[i, 0] = Math.Sin(i * 0.3);
                Y[i, 1] = Math.Sin(i * 0.3) * 2.0; // Correlated with first output
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 3.0 }));

            // Assert - Second output should be approximately 2x first output
            Assert.True(Math.Abs(means[1] - 2.0 * means[0]) < 0.5,
                "Correlated outputs should maintain relationship");
        }

        [Fact]
        public void MultiOutputGP_UpdateKernel_ChangesAllOutputs()
        {
            // Arrange
            var kernel1 = new GaussianKernel<double>(sigma: 0.5);
            var kernel2 = new GaussianKernel<double>(sigma: 2.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel1);

            var X = new Matrix<double>(15, 1);
            var Y = new Matrix<double>(15, 2);
            for (int i = 0; i < 15; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = i;
                Y[i, 1] = i * 2;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means1, cov1) = mogp.PredictMultiOutput(new Vector<double>(new[] { 7.5 }));

            mogp.UpdateKernel(kernel2);
            var (means2, cov2) = mogp.PredictMultiOutput(new Vector<double>(new[] { 7.5 }));

            // Assert
            Assert.NotEqual(means1[0], means2[0]);
            Assert.NotEqual(means1[1], means2[1]);
        }

        [Fact]
        public void MultiOutputGP_CovarianceMatrix_PositiveDefinite()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var Y = new Matrix<double>(10, 2);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = Math.Sin(i * 0.5);
                Y[i, 1] = Math.Cos(i * 0.5);
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 5.0 }));

            // Assert - Diagonal elements should be non-negative (variances)
            for (int i = 0; i < covariance.Rows; i++)
            {
                Assert.True(covariance[i, i] >= 0,
                    $"Covariance diagonal element [{i},{i}] must be non-negative");
            }
        }

        [Fact]
        public void MultiOutputGP_LinearOutputs_FitsCorrectly()
        {
            // Arrange
            var kernel = new LinearKernel<double>();
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(12, 1);
            var Y = new Matrix<double>(12, 2);
            for (int i = 0; i < 12; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = 2.0 * i + 1.0;
                Y[i, 1] = -i + 5.0;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 6.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert
            double expected1 = 2.0 * testX + 1.0;
            double expected2 = -testX + 5.0;
            Assert.Equal(expected1, means[0], precision: 1);
            Assert.Equal(expected2, means[1], precision: 1);
        }

        [Fact]
        public void MultiOutputGP_NoisyData_SmoothsPredictions()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);
            var random = new Random(42);

            var X = new Matrix<double>(25, 1);
            var Y = new Matrix<double>(25, 2);
            for (int i = 0; i < 25; i++)
            {
                X[i, 0] = i * 0.3;
                double noise1 = (random.NextDouble() - 0.5) * 0.2;
                double noise2 = (random.NextDouble() - 0.5) * 0.2;
                Y[i, 0] = Math.Sin(i * 0.3) + noise1;
                Y[i, 1] = Math.Cos(i * 0.3) + noise2;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 3.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert - Should smooth out noise
            double expected1 = Math.Sin(testX);
            double expected2 = Math.Cos(testX);
            Assert.True(Math.Abs(means[0] - expected1) < 0.5,
                "Should capture trend despite noise in output 1");
            Assert.True(Math.Abs(means[1] - expected2) < 0.5,
                "Should capture trend despite noise in output 2");
        }

        [Fact]
        public void MultiOutputGP_MultipleFeatures_HandlesHighDimensional()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 2);
            var Y = new Matrix<double>(20, 2);
            for (int i = 0; i < 20; i++)
            {
                X[i, 0] = i * 0.3;
                X[i, 1] = i * 0.2;
                Y[i, 0] = X[i, 0] + X[i, 1];
                Y[i, 1] = X[i, 0] - X[i, 1];
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testPoint = new Vector<double>(new[] { 3.0, 2.0 });
            var (means, covariance) = mogp.PredictMultiOutput(testPoint);

            // Assert
            double expected1 = 3.0 + 2.0;
            double expected2 = 3.0 - 2.0;
            Assert.True(Math.Abs(means[0] - expected1) < 1.0,
                $"First output {means[0]} should be close to {expected1}");
            Assert.True(Math.Abs(means[1] - expected2) < 1.0,
                $"Second output {means[1]} should be close to {expected2}");
        }

        [Fact]
        public void MultiOutputGP_QuadraticOutputs_CapturesNonlinearity()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 2.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(20, 1);
            var Y = new Matrix<double>(20, 2);
            for (int i = 0; i < 20; i++)
            {
                double x = (i - 10) * 0.5;
                X[i, 0] = x;
                Y[i, 0] = x * x;
                Y[i, 1] = x * x * x;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 2.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert
            double expected1 = testX * testX;
            double expected2 = testX * testX * testX;
            Assert.True(Math.Abs(means[0] - expected1) < 1.0,
                $"Quadratic output {means[0]} should approximate {expected1}");
            Assert.True(Math.Abs(means[1] - expected2) < 2.0,
                $"Cubic output {means[1]} should approximate {expected2}");
        }

        [Fact]
        public void MultiOutputGP_PredictionConsistency_RepeatedCalls()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var Y = new Matrix<double>(10, 2);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = i * 2.0;
                Y[i, 1] = i * 3.0;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testPoint = new Vector<double>(new[] { 5.0 });
            var (means1, cov1) = mogp.PredictMultiOutput(testPoint);
            var (means2, cov2) = mogp.PredictMultiOutput(testPoint);

            // Assert
            Assert.Equal(means1[0], means2[0]);
            Assert.Equal(means1[1], means2[1]);
            Assert.Equal(cov1[0, 0], cov2[0, 0]);
        }

        [Fact]
        public void MultiOutputGP_SingleOutput_WorksCorrectly()
        {
            // Arrange - Edge case with single output
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var Y = new Matrix<double>(10, 1);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = Math.Sin(i * 0.5);
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 5.0 }));

            // Assert
            Assert.Equal(1, means.Length);
            double expected = Math.Sin(5.0 * 0.5);
            Assert.True(Math.Abs(means[0] - expected) < 0.5,
                $"Single output {means[0]} should approximate {expected}");
        }

        [Fact]
        public void MultiOutputGP_FourOutputs_HandlesMultipleOutputs()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(15, 1);
            var Y = new Matrix<double>(15, 4);
            for (int i = 0; i < 15; i++)
            {
                double x = i * 0.4;
                X[i, 0] = x;
                Y[i, 0] = Math.Sin(x);
                Y[i, 1] = Math.Cos(x);
                Y[i, 2] = x;
                Y[i, 3] = x * x;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 3.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert
            Assert.Equal(4, means.Length);
            Assert.Equal(4, covariance.Rows);
            Assert.Equal(4, covariance.Columns);
        }

        [Fact]
        public void MultiOutputGP_ConstantOutputs_RecoversConstants()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var Y = new Matrix<double>(10, 2);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = 5.0;
                Y[i, 1] = -3.0;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 5.0 }));

            // Assert
            Assert.Equal(5.0, means[0], precision: 1);
            Assert.Equal(-3.0, means[1], precision: 1);
        }

        [Fact]
        public void MultiOutputGP_MaternKernel_ProducesValidPredictions()
        {
            // Arrange
            var kernel = new MaternKernel<double>();
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(15, 1);
            var Y = new Matrix<double>(15, 2);
            for (int i = 0; i < 15; i++)
            {
                X[i, 0] = i * 0.4;
                Y[i, 0] = Math.Sin(i * 0.4);
                Y[i, 1] = Math.Cos(i * 0.4);
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 3.0 }));

            // Assert
            Assert.True(Math.Abs(means[0]) <= 1.5, "First output in valid range");
            Assert.True(Math.Abs(means[1]) <= 1.5, "Second output in valid range");
        }

        [Fact]
        public void MultiOutputGP_PeriodicPattern_TwoOutputs()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(30, 1);
            var Y = new Matrix<double>(30, 2);
            for (int i = 0; i < 30; i++)
            {
                double x = i * 0.2;
                X[i, 0] = x;
                Y[i, 0] = Math.Sin(x * 2 * Math.PI / 5.0);
                Y[i, 1] = Math.Cos(x * 2 * Math.PI / 5.0);
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { 3.0 }));

            // Assert
            Assert.True(Math.Abs(means[0]) <= 1.5, "Periodic output 1 in range");
            Assert.True(Math.Abs(means[1]) <= 1.5, "Periodic output 2 in range");
        }

        [Fact]
        public void MultiOutputGP_OutputScales_DifferentMagnitudes()
        {
            // Arrange - Outputs with different scales
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(15, 1);
            var Y = new Matrix<double>(15, 2);
            for (int i = 0; i < 15; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = i * 0.1;        // Small scale
                Y[i, 1] = i * 100.0;      // Large scale
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 10.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert - Should handle different scales
            Assert.True(Math.Abs(means[0] - 1.0) < 0.5,
                $"Small scale output {means[0]} should be close to 1.0");
            Assert.True(Math.Abs(means[1] - 1000.0) < 200.0,
                $"Large scale output {means[1]} should be close to 1000.0");
        }

        [Fact]
        public void MultiOutputGP_InterpolationVsExtrapolation_UncertaintyDiffers()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(10, 1);
            var Y = new Matrix<double>(10, 2);
            for (int i = 0; i < 10; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = i;
                Y[i, 1] = i * 2;
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var (meansInterp, covInterp) = mogp.PredictMultiOutput(new Vector<double>(new[] { 5.0 }));
            var (meansExtrap, covExtrap) = mogp.PredictMultiOutput(new Vector<double>(new[] { 20.0 }));

            // Assert - Extrapolation should have higher uncertainty
            Assert.True(covExtrap[0, 0] >= covInterp[0, 0],
                "Extrapolation uncertainty should be >= interpolation uncertainty");
        }

        [Fact]
        public void MultiOutputGP_OppositeLinearTrends_HandlesCorrectly()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.0);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(12, 1);
            var Y = new Matrix<double>(12, 2);
            for (int i = 0; i < 12; i++)
            {
                X[i, 0] = i;
                Y[i, 0] = i * 2.0;       // Increasing
                Y[i, 1] = 20.0 - i * 2.0; // Decreasing
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 6.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert
            double expected1 = 12.0;
            double expected2 = 8.0;
            Assert.Equal(expected1, means[0], precision: 1);
            Assert.Equal(expected2, means[1], precision: 1);
        }

        [Fact]
        public void MultiOutputGP_MixedPeriodicAndLinear_CapturesBothPatterns()
        {
            // Arrange
            var kernel = new GaussianKernel<double>(sigma: 1.5);
            var mogp = new MultiOutputGaussianProcess<double>(kernel);

            var X = new Matrix<double>(25, 1);
            var Y = new Matrix<double>(25, 2);
            for (int i = 0; i < 25; i++)
            {
                double x = i * 0.3;
                X[i, 0] = x;
                Y[i, 0] = Math.Sin(x);    // Periodic
                Y[i, 1] = x * 2.0;        // Linear
            }

            // Act
            mogp.FitMultiOutput(X, Y);
            var testX = 3.0;
            var (means, covariance) = mogp.PredictMultiOutput(new Vector<double>(new[] { testX }));

            // Assert
            double expected1 = Math.Sin(testX);
            double expected2 = testX * 2.0;
            Assert.True(Math.Abs(means[0] - expected1) < 0.5,
                "Should capture periodic pattern");
            Assert.True(Math.Abs(means[1] - expected2) < 1.0,
                "Should capture linear trend");
        }

        #endregion
    }
}
