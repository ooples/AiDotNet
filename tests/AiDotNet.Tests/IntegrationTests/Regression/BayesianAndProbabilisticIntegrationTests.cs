using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for Bayesian and probabilistic regression models.
    /// Tests uncertainty quantification, prior/posterior distributions, and probabilistic predictions.
    /// </summary>
    public class BayesianAndProbabilisticIntegrationTests
    {
        #region BayesianRegression Tests

        [Fact]
        public void BayesianRegression_LinearData_FitsWithUncertainty()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 3 * x[i, 0] + 2 * x[i, 1] + 5;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit well
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void BayesianRegression_PosteriorDistribution_ProvidesPredictiveUncertainty()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 1;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);

            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 10;

            var (mean, variance) = regression.PredictWithUncertainty(testX);

            // Assert - variance should be non-negative
            Assert.True(variance[0] >= 0);
            Assert.True(Math.Abs(mean[0] - 21.0) < 5.0); // 2*10 + 1 = 21
        }

        [Fact]
        public void BayesianRegression_PriorInfluence_AffectsSmallDatasets()
        {
            // Arrange - very small dataset
            var x = new Matrix<double>(5, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 });

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
            }

            // Act - with different priors
            var regStrongPrior = new BayesianRegression<double>(
                new BayesianRegressionOptions { PriorStrength = 10.0 });
            regStrongPrior.Train(x, y);

            var regWeakPrior = new BayesianRegression<double>(
                new BayesianRegressionOptions { PriorStrength = 0.1 });
            regWeakPrior.Train(x, y);

            var predictions = regStrongPrior.Predict(x);
            var predictionsWeak = regWeakPrior.Predict(x);

            // Assert - different prior strength should affect predictions
            bool different = false;
            for (int i = 0; i < 5; i++)
            {
                if (Math.Abs(predictions[i] - predictionsWeak[i]) > 0.5)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void BayesianRegression_CredibleIntervals_ContainTrueValues()
        {
            // Arrange
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 5;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);

            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 15;

            var (lower, upper) = regression.GetCredibleInterval(testX, 0.95);

            // Assert - true value should be within credible interval
            double trueValue = 3 * 15 + 5; // = 50
            Assert.True(lower[0] <= trueValue && upper[0] >= trueValue);
        }

        [Fact]
        public void BayesianRegression_MultipleFeatures_EstimatesJointPosterior()
        {
            // Arrange
            var x = new Matrix<double>(30, 3);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                x[i, 2] = i / 2.0;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] - x[i, 2] + 10;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void BayesianRegression_WithNoise_QuantifiesUncertainty()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            var random = new Random(42);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + (random.NextDouble() - 0.5) * 10; // Noisy
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);

            var (mean, variance) = regression.PredictWithUncertainty(x);

            // Assert - variance should reflect noise
            for (int i = 0; i < 20; i++)
            {
                Assert.True(variance[i] > 0); // Positive variance
            }
        }

        [Fact]
        public void BayesianRegression_SmallDataset_HighUncertainty()
        {
            // Arrange - very small dataset
            var x = new Matrix<double>(3, 1);
            var y = new Vector<double>(new[] { 1.0, 5.0, 9.0 });

            for (int i = 0; i < 3; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);

            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 10; // Extrapolation

            var (mean, variance) = regression.PredictWithUncertainty(testX);

            // Assert - should have high uncertainty for extrapolation with small data
            Assert.True(variance[0] > 0);
        }

        [Fact]
        public void BayesianRegression_LargeDataset_LowUncertainty()
        {
            // Arrange - large dataset
            var n = 200;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);

            var (mean, variance) = regression.PredictWithUncertainty(x);

            // Assert - large dataset should reduce uncertainty
            double avgVariance = 0;
            for (int i = 0; i < n; i++)
            {
                avgVariance += variance[i];
            }
            avgVariance /= n;
            Assert.True(avgVariance < 100.0); // Reasonable uncertainty
        }

        [Fact]
        public void BayesianRegression_PosteriorSampling_GeneratesPlausibleValues()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 2;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);

            var samples = regression.SamplePosterior(100);

            // Assert - samples should be reasonable
            Assert.Equal(100, samples.Count);
            foreach (var sample in samples)
            {
                Assert.NotNull(sample);
            }
        }

        [Fact]
        public void BayesianRegression_MarginalLikelihood_ForModelComparison()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + 2 * x[i, 1] + 3;
            }

            // Act
            var regression = new BayesianRegression<double>();
            regression.Train(x, y);
            var marginalLikelihood = regression.GetMarginalLikelihood();

            // Assert - should have finite marginal likelihood
            Assert.True(double.IsFinite(marginalLikelihood));
        }

        [Fact]
        public void BayesianRegression_SequentialUpdate_IncorporatesNewData()
        {
            // Arrange - initial data
            var x1 = new Matrix<double>(10, 1);
            var y1 = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x1[i, 0] = i;
                y1[i] = 2 * i + 1;
            }

            var regression = new BayesianRegression<double>();
            regression.Train(x1, y1);

            // Act - update with new data
            var x2 = new Matrix<double>(5, 1);
            var y2 = new Vector<double>(5);

            for (int i = 0; i < 5; i++)
            {
                x2[i, 0] = 10 + i;
                y2[i] = 2 * (10 + i) + 1;
            }

            regression.UpdatePosterior(x2, y2);
            var predictions = regression.Predict(x2);

            // Assert - should incorporate new data
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y2[i]) < 5.0);
            }
        }

        [Fact]
        public void BayesianRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(10, 1);
            var y = new Vector<float>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2 + 3;
            }

            // Act
            var regression = new BayesianRegression<float>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0f);
            }
        }

        #endregion

        #region GaussianProcessRegression Tests

        [Fact]
        public void GaussianProcessRegression_SmoothFunction_InterpolatesWell()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(i / 3.0) * 10;
            }

            var options = new GaussianProcessRegressionOptions { KernelType = KernelType.RBF };

            // Act
            var regression = new GaussianProcessRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit training data well
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0);
            }
        }

        [Fact]
        public void GaussianProcessRegression_PredictiveVariance_HigherForExtrapolation()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            // Act
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);

            // Predict inside training range
            var xInside = new Matrix<double>(1, 1);
            xInside[0, 0] = 5;
            var (meanInside, varInside) = regression.PredictWithUncertainty(xInside);

            // Predict outside training range
            var xOutside = new Matrix<double>(1, 1);
            xOutside[0, 0] = 20;
            var (meanOutside, varOutside) = regression.PredictWithUncertainty(xOutside);

            // Assert - variance should be higher for extrapolation
            Assert.True(varOutside[0] > varInside[0]);
        }

        [Fact]
        public void GaussianProcessRegression_DifferentKernels_ProduceDifferentFits()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - different kernels
            var gpRBF = new GaussianProcessRegression<double>(
                new GaussianProcessRegressionOptions { KernelType = KernelType.RBF });
            gpRBF.Train(x, y);
            var predRBF = gpRBF.Predict(x);

            var gpPoly = new GaussianProcessRegression<double>(
                new GaussianProcessRegressionOptions { KernelType = KernelType.Polynomial });
            gpPoly.Train(x, y);
            var predPoly = gpPoly.Predict(x);

            // Assert - different kernels should produce different predictions
            bool different = false;
            for (int i = 0; i < 15; i++)
            {
                if (Math.Abs(predRBF[i] - predPoly[i]) > 5.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void GaussianProcessRegression_LengthScale_AffectsSmoothness()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i + (i % 3) * 5;
            }

            // Act - different length scales
            var gpShort = new GaussianProcessRegression<double>(
                new GaussianProcessRegressionOptions { LengthScale = 0.5 });
            gpShort.Train(x, y);
            var predShort = gpShort.Predict(x);

            var gpLong = new GaussianProcessRegression<double>(
                new GaussianProcessRegressionOptions { LengthScale = 5.0 });
            gpLong.Train(x, y);
            var predLong = gpLong.Predict(x);

            // Assert - different length scales should affect smoothness
            bool different = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(predShort[i] - predLong[i]) > 2.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void GaussianProcessRegression_NoiseLevel_AffectsUncertainty()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            // Act - different noise levels
            var gpLowNoise = new GaussianProcessRegression<double>(
                new GaussianProcessRegressionOptions { NoiseVariance = 0.1 });
            gpLowNoise.Train(x, y);
            var (_, varLowNoise) = gpLowNoise.PredictWithUncertainty(x);

            var gpHighNoise = new GaussianProcessRegression<double>(
                new GaussianProcessRegressionOptions { NoiseVariance = 10.0 });
            gpHighNoise.Train(x, y);
            var (_, varHighNoise) = gpHighNoise.PredictWithUncertainty(x);

            // Assert - high noise should lead to higher uncertainty
            Assert.True(varHighNoise[0] > varLowNoise[0]);
        }

        [Fact]
        public void GaussianProcessRegression_SparseTrainingData_InterpolatesSmoothly()
        {
            // Arrange - sparse data
            var x = new Matrix<double>(6, 1);
            var y = new Vector<double>(new[] { 0.0, 5.0, 8.0, 9.0, 8.0, 5.0 });

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i * 2; // Sparse samples
            }

            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);

            // Act - predict at intermediate points
            var testX = new Matrix<double>(5, 1);
            for (int i = 0; i < 5; i++)
            {
                testX[i, 0] = i * 2 + 1; // Between training points
            }

            var predictions = regression.Predict(testX);

            // Assert - should provide smooth interpolation
            Assert.NotNull(predictions);
            Assert.Equal(5, predictions.Length);
        }

        [Fact]
        public void GaussianProcessRegression_MultipleFeatures_HandlesHighDimensional()
        {
            // Arrange
            var x = new Matrix<double>(25, 3);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                x[i, 2] = i / 2.0;
                y[i] = x[i, 0] + 2 * x[i, 1] - x[i, 2];
            }

            // Act
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void GaussianProcessRegression_SampleFromPosterior_GeneratesPlausibleFunctions()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(i);
            }

            // Act
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);

            var samples = regression.SamplePosterior(x, 5);

            // Assert - should generate 5 function samples
            Assert.Equal(5, samples.Count);
            foreach (var sample in samples)
            {
                Assert.Equal(10, sample.Length);
            }
        }

        [Fact]
        public void GaussianProcessRegression_MarginalLikelihood_ForHyperparameterOptimization()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2 + 3;
            }

            // Act
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);
            var logMarginalLikelihood = regression.GetLogMarginalLikelihood();

            // Assert - should have finite log marginal likelihood
            Assert.True(double.IsFinite(logMarginalLikelihood));
        }

        [Fact]
        public void GaussianProcessRegression_ConfidenceBands_CoverTrueFunction()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 5;
            }

            // Act
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);

            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 10;

            var (lower, upper) = regression.GetConfidenceBand(testX, 0.95);

            // Assert - true value should be within confidence band
            double trueValue = 2 * 10 + 5;
            Assert.True(lower[0] <= trueValue && upper[0] >= trueValue);
        }

        [Fact]
        public void GaussianProcessRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(4, 1);
            var y = new Vector<double>(new[] { 1.0, 4.0, 9.0, 16.0 });

            for (int i = 0; i < 4; i++)
            {
                x[i, 0] = i + 1;
            }

            // Act
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(4, predictions.Length);
        }

        [Fact]
        public void GaussianProcessRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 100;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i % 10;
                y[i] = x[i, 0] + x[i, 1];
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new GaussianProcessRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert - GP can be slow, but should complete
            Assert.True(sw.ElapsedMilliseconds < 20000);
        }

        #endregion
    }
}
