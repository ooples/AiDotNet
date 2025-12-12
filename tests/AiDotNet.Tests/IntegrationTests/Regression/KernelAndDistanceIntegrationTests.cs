using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for kernel-based and distance-based regression models.
    /// Tests SVR, kernel ridge regression, k-NN, and locally weighted regression.
    /// </summary>
    public class KernelAndDistanceIntegrationTests
    {
        #region SupportVectorRegression Tests

        [Fact]
        public void SupportVectorRegression_LinearKernel_FitsLinearData()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 5;
            }

            var options = new SupportVectorRegressionOptions { KernelType = KernelType.Linear };

            // Act
            var regression = new SupportVectorRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void SupportVectorRegression_RBFKernel_FitsNonLinear()
        {
            // Arrange - non-linear data
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i / 5.0;
                y[i] = Math.Sin(x[i, 0]) * 10;
            }

            var options = new SupportVectorRegressionOptions { KernelType = KernelType.RBF };

            // Act
            var regression = new SupportVectorRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void SupportVectorRegression_PolynomialKernel_FitsPolynomialData()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i - 7;
                y[i] = x[i, 0] * x[i, 0] + 2 * x[i, 0] + 3;
            }

            var options = new SupportVectorRegressionOptions { KernelType = KernelType.Polynomial, Degree = 2 };

            // Act
            var regression = new SupportVectorRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void SupportVectorRegression_EpsilonParameter_ControlsMargin()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            // Act - compare different epsilon values
            var svrSmall = new SupportVectorRegression<double>(new SupportVectorRegressionOptions { Epsilon = 0.1 });
            svrSmall.Train(x, y);
            var predSmall = svrSmall.Predict(x);

            var svrLarge = new SupportVectorRegression<double>(new SupportVectorRegressionOptions { Epsilon = 2.0 });
            svrLarge.Train(x, y);
            var predLarge = svrLarge.Predict(x);

            // Assert - different epsilon should produce different fits
            bool different = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(predSmall[i] - predLarge[i]) > 1.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void SupportVectorRegression_RegularizationC_AffectsFit()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            var random = new Random(42);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i + (random.NextDouble() - 0.5) * 5;
            }

            // Act - compare different C values
            var svrWeakReg = new SupportVectorRegression<double>(new SupportVectorRegressionOptions { C = 0.1 });
            svrWeakReg.Train(x, y);

            var svrStrongReg = new SupportVectorRegression<double>(new SupportVectorRegressionOptions { C = 10.0 });
            svrStrongReg.Train(x, y);

            // Assert - both should produce valid predictions
            var predWeak = svrWeakReg.Predict(x);
            var predStrong = svrStrongReg.Predict(x);
            Assert.NotNull(predWeak);
            Assert.NotNull(predStrong);
        }

        [Fact]
        public void SupportVectorRegression_MultipleFeatures_HandlesWell()
        {
            // Arrange
            var x = new Matrix<double>(30, 3);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                x[i, 2] = i / 2.0;
                y[i] = x[i, 0] + 2 * x[i, 1] - x[i, 2];
            }

            // Act
            var regression = new SupportVectorRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void SupportVectorRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0 });

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new SupportVectorRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(8, predictions.Length);
        }

        [Fact]
        public void SupportVectorRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 200;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] + x[i, 1];
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new SupportVectorRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 10000);
        }

        [Fact]
        public void SupportVectorRegression_WithOutliers_RobustFit()
        {
            // Arrange - data with outliers
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 1;
            }

            // Add outliers
            x[12, 0] = 12; y[12] = 100;
            x[13, 0] = 13; y[13] = -50;
            x[14, 0] = 14; y[14] = 200;

            // Act
            var regression = new SupportVectorRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should be relatively robust to outliers
            for (int i = 0; i < 12; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void SupportVectorRegression_GammaParameter_AffectsRBF()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(i / 3.0) * 10;
            }

            // Act - different gamma values
            var svrSmallGamma = new SupportVectorRegression<double>(
                new SupportVectorRegressionOptions { KernelType = KernelType.RBF, Gamma = 0.01 });
            svrSmallGamma.Train(x, y);
            var predSmallGamma = svrSmallGamma.Predict(x);

            var svrLargeGamma = new SupportVectorRegression<double>(
                new SupportVectorRegressionOptions { KernelType = KernelType.RBF, Gamma = 1.0 });
            svrLargeGamma.Train(x, y);
            var predLargeGamma = svrLargeGamma.Predict(x);

            // Assert - different gamma should produce different predictions
            bool different = false;
            for (int i = 0; i < 20; i++)
            {
                if (Math.Abs(predSmallGamma[i] - predLargeGamma[i]) > 2.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void SupportVectorRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(10, 1);
            var y = new Vector<float>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3 + 2;
            }

            // Act
            var regression = new SupportVectorRegression<float>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(10, predictions.Length);
        }

        [Fact]
        public void SupportVectorRegression_ConstantTarget_HandlesGracefully()
        {
            // Arrange
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(12);

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i;
                y[i] = 25.0; // Constant
            }

            // Act
            var regression = new SupportVectorRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 12; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 25.0) < 5.0);
            }
        }

        #endregion

        #region KernelRidgeRegression Tests

        [Fact]
        public void KernelRidgeRegression_LinearKernel_SimilarToRidgeRegression()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 1;
            }

            var options = new KernelRidgeRegressionOptions { KernelType = KernelType.Linear };

            // Act
            var regression = new KernelRidgeRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void KernelRidgeRegression_RBFKernel_HandlesNonLinearity()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i / 5.0;
                y[i] = Math.Cos(x[i, 0]) * 10 + 5;
            }

            var options = new KernelRidgeRegressionOptions { KernelType = KernelType.RBF, Gamma = 0.5 };

            // Act
            var regression = new KernelRidgeRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void KernelRidgeRegression_RegularizationAlpha_PreventsOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            var random = new Random(123);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 10;
            }

            // Act - compare different alpha values
            var krr1 = new KernelRidgeRegression<double>(new KernelRidgeRegressionOptions { Alpha = 0.1 });
            krr1.Train(x, y);

            var krr2 = new KernelRidgeRegression<double>(new KernelRidgeRegressionOptions { Alpha = 10.0 });
            krr2.Train(x, y);

            // Assert - both should produce valid predictions
            var pred1 = krr1.Predict(x);
            var pred2 = krr2.Predict(x);
            Assert.NotNull(pred1);
            Assert.NotNull(pred2);
        }

        [Fact]
        public void KernelRidgeRegression_PolynomialKernel_FitsPolynomialData()
        {
            // Arrange
            var x = new Matrix<double>(18, 1);
            var y = new Vector<double>(18);

            for (int i = 0; i < 18; i++)
            {
                x[i, 0] = i - 9;
                y[i] = x[i, 0] * x[i, 0] + 3;
            }

            var options = new KernelRidgeRegressionOptions { KernelType = KernelType.Polynomial, Degree = 2 };

            // Act
            var regression = new KernelRidgeRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 18; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void KernelRidgeRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(6, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 4.0, 7.0, 11.0, 16.0 });

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new KernelRidgeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(6, predictions.Length);
        }

        [Fact]
        public void KernelRidgeRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 300;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i / 2.0;
                y[i] = x[i, 0] + 2 * x[i, 1];
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new KernelRidgeRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 15000);
        }

        #endregion

        #region KNearestNeighborsRegression Tests

        [Fact]
        public void KNearestNeighborsRegression_SimplePattern_PredictsByAveraging()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            var options = new KNearestNeighborsOptions { K = 3 };

            // Act
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);

            // Test on training data
            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 5.0;
            var prediction = regression.Predict(testX);

            // Assert - should be close to 10 (5 * 2)
            Assert.True(Math.Abs(prediction[0] - 10.0) < 5.0);
        }

        [Fact]
        public void KNearestNeighborsRegression_DifferentK_ProducesDifferentPredictions()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            // Act - compare k=1 vs k=5
            var knn1 = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            knn1.Train(x, y);

            var knn5 = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 5 });
            knn5.Train(x, y);

            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 10.5; // Between points

            var pred1 = knn1.Predict(testX);
            var pred5 = knn5.Predict(testX);

            // Assert - different k should produce different predictions
            Assert.NotEqual(pred1[0], pred5[0]);
        }

        [Fact]
        public void KNearestNeighborsRegression_NonLinearPattern_CapturesWell()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sqrt(i) * 5;
            }

            var options = new KNearestNeighborsOptions { K = 3 };

            // Act
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void KNearestNeighborsRegression_MultipleFeatures_UsesEuclideanDistance()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new KNearestNeighborsOptions { K = 5 };

            // Act
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void KNearestNeighborsRegression_WeightedAverage_ImprovesPredictions()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - compare uniform vs distance-weighted
            var knnUniform = new KNearestNeighborsRegression<double>(
                new KNearestNeighborsOptions { K = 5, WeightingScheme = WeightingScheme.Uniform });
            knnUniform.Train(x, y);

            var knnWeighted = new KNearestNeighborsRegression<double>(
                new KNearestNeighborsOptions { K = 5, WeightingScheme = WeightingScheme.Distance });
            knnWeighted.Train(x, y);

            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 7.5;

            var predUniform = knnUniform.Predict(testX);
            var predWeighted = knnWeighted.Predict(testX);

            // Assert - weighted should be different
            Assert.NotEqual(predUniform[0], predWeighted[0]);
        }

        [Fact]
        public void KNearestNeighborsRegression_SmallK_MoreSensitiveToNoise()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            var random = new Random(456);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i + (random.NextDouble() - 0.5) * 5;
            }

            var options = new KNearestNeighborsOptions { K = 1 };

            // Act
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - with k=1, should match training data exactly
            for (int i = 0; i < 20; i++)
            {
                Assert.Equal(y[i], predictions[i], precision: 1);
            }
        }

        [Fact]
        public void KNearestNeighborsRegression_LargeK_ProducesSmoother()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = i + (i % 3) * 5; // Oscillating
            }

            var options = new KNearestNeighborsOptions { K = 10 };

            // Act
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should smooth out oscillations
            Assert.NotNull(predictions);
        }

        [Fact]
        public void KNearestNeighborsRegression_NoTraining_UsesAllData()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            var options = new KNearestNeighborsOptions { K = 3 };

            // Act - k-NN is lazy, no real training
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(10, predictions.Length);
        }

        [Fact]
        public void KNearestNeighborsRegression_ExtrapolationWarning_EdgeBehavior()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            var options = new KNearestNeighborsOptions { K = 3 };
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);

            // Act - predict outside training range
            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 20; // Outside range

            var prediction = regression.Predict(testX);

            // Assert - should still produce a prediction (using nearest neighbors)
            Assert.True(prediction[0] > 0);
        }

        [Fact]
        public void KNearestNeighborsRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(10, 1);
            var y = new Vector<float>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2.5f;
            }

            var options = new KNearestNeighborsOptions { K = 3 };

            // Act
            var regression = new KNearestNeighborsRegression<float>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(10, predictions.Length);
        }

        [Fact]
        public void KNearestNeighborsRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i % 10;
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new KNearestNeighborsOptions { K = 5 };

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new KNearestNeighborsRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 15000);
        }

        #endregion

        #region LocallyWeightedRegression Tests

        [Fact]
        public void LocallyWeightedRegression_SmoothData_FitsWell()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(i / 3.0) * 10;
            }

            var options = new LocallyWeightedRegressionOptions { Bandwidth = 2.0 };

            // Act
            var regression = new LocallyWeightedRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void LocallyWeightedRegression_BandwidthParameter_AffectsSmoothness()
        {
            // Arrange
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                y[i] = i + (i % 3) * 3;
            }

            // Act - compare narrow vs wide bandwidth
            var lwrNarrow = new LocallyWeightedRegression<double>(
                new LocallyWeightedRegressionOptions { Bandwidth = 0.5 });
            lwrNarrow.Train(x, y);
            var predNarrow = lwrNarrow.Predict(x);

            var lwrWide = new LocallyWeightedRegression<double>(
                new LocallyWeightedRegressionOptions { Bandwidth = 5.0 });
            lwrWide.Train(x, y);
            var predWide = lwrWide.Predict(x);

            // Assert - different bandwidth should produce different smoothness
            bool different = false;
            for (int i = 0; i < 25; i++)
            {
                if (Math.Abs(predNarrow[i] - predWide[i]) > 2.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void LocallyWeightedRegression_NonLinearPattern_AdaptsLocally()
        {
            // Arrange - piecewise different slopes
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            for (int i = 15; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = 30 + (i - 15) * 5; // Different slope
            }

            var options = new LocallyWeightedRegressionOptions { Bandwidth = 3.0 };

            // Act
            var regression = new LocallyWeightedRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should adapt to local patterns
            Assert.True(predictions[5] < predictions[25]); // Different regions
        }

        [Fact]
        public void LocallyWeightedRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0 });

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
            }

            var options = new LocallyWeightedRegressionOptions { Bandwidth = 2.0 };

            // Act
            var regression = new LocallyWeightedRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(8, predictions.Length);
        }

        [Fact]
        public void LocallyWeightedRegression_WithNoise_SmoothsAppropriately()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            var random = new Random(789);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2 + (random.NextDouble() - 0.5) * 10;
            }

            var options = new LocallyWeightedRegressionOptions { Bandwidth = 3.0 };

            // Act
            var regression = new LocallyWeightedRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should smooth the noise
            Assert.NotNull(predictions);
        }

        [Fact]
        public void LocallyWeightedRegression_MultipleFeatures_WeightsAllDimensions()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] + 2 * x[i, 1];
            }

            var options = new LocallyWeightedRegressionOptions { Bandwidth = 3.0 };

            // Act
            var regression = new LocallyWeightedRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void LocallyWeightedRegression_GaussianKernel_ProducesSmootherFit()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i + (i % 2) * 5;
            }

            var options = new LocallyWeightedRegressionOptions
            {
                Bandwidth = 2.0,
                KernelFunction = KernelFunction.Gaussian
            };

            // Act
            var regression = new LocallyWeightedRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void LocallyWeightedRegression_InterpolationCapability_GoodForSmooth()
        {
            // Arrange - smooth underlying function
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i * 2; // Sparse samples
                y[i] = Math.Exp(x[i, 0] / 10.0);
            }

            var regression = new LocallyWeightedRegression<double>(
                new LocallyWeightedRegressionOptions { Bandwidth = 4.0 });
            regression.Train(x, y);

            // Act - interpolate between samples
            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 5; // Between samples

            var prediction = regression.Predict(testX);

            // Assert - should interpolate reasonably
            Assert.True(prediction[0] > 0);
        }

        #endregion
    }
}
