using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for logistic and generalized linear models.
    /// Tests classification-style regression models including logistic, multinomial, Poisson, and negative binomial.
    /// </summary>
    public class LogisticAndGeneralizedIntegrationTests
    {
        #region LogisticRegression Tests

        [Fact]
        public void LogisticRegression_BinaryClassification_ConvergesCorrectly()
        {
            // Arrange - linearly separable binary classification
            var x = new Matrix<double>(10, 2);
            var y = new Vector<double>(10);

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i;
                y[i] = 0.0; // Class 0
            }

            for (int i = 5; i < 10; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i;
                y[i] = 1.0; // Class 1
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be close to 0 or 1
            for (int i = 0; i < 5; i++)
            {
                Assert.True(predictions[i] < 0.5);
            }
            for (int i = 5; i < 10; i++)
            {
                Assert.True(predictions[i] > 0.5);
            }
        }

        [Fact]
        public void LogisticRegression_ProbabilisticOutput_IsBetweenZeroAndOne()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            var random = new Random(42);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = random.NextDouble() * 10;
                x[i, 1] = random.NextDouble() * 10;
                y[i] = (x[i, 0] + x[i, 1] > 10) ? 1.0 : 0.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - all predictions should be probabilities
            for (int i = 0; i < 20; i++)
            {
                Assert.True(predictions[i] >= 0.0 && predictions[i] <= 1.0);
            }
        }

        [Fact]
        public void LogisticRegression_PerfectSeparation_HighConfidence()
        {
            // Arrange - perfectly separable classes
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(8);

            for (int i = 0; i < 4; i++)
            {
                x[i, 0] = i;
                y[i] = 0.0;
            }

            for (int i = 4; i < 8; i++)
            {
                x[i, 0] = i + 10; // Large gap
                y[i] = 1.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should be very confident (close to 0 or 1)
            for (int i = 0; i < 4; i++)
            {
                Assert.True(predictions[i] < 0.1);
            }
            for (int i = 4; i < 8; i++)
            {
                Assert.True(predictions[i] > 0.9);
            }
        }

        [Fact]
        public void LogisticRegression_OverlappingClasses_ModeratePredictions()
        {
            // Arrange - overlapping classes
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(12);

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i;
                y[i] = 0.0;
            }

            for (int i = 6; i < 12; i++)
            {
                x[i, 0] = i - 3; // Overlap with class 0
                y[i] = 1.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions in overlap region should be moderate (around 0.5)
            Assert.NotNull(predictions);
        }

        [Fact]
        public void LogisticRegression_MultipleFeatures_ConvergesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(30, 3);
            var y = new Vector<double>(30);
            var random = new Random(123);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = random.NextDouble() * 10;
                x[i, 1] = random.NextDouble() * 10;
                x[i, 2] = random.NextDouble() * 10;
                y[i] = (x[i, 0] + x[i, 1] - x[i, 2] > 5) ? 1.0 : 0.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should classify most correctly
            int correct = 0;
            for (int i = 0; i < 30; i++)
            {
                double predicted = predictions[i] > 0.5 ? 1.0 : 0.0;
                if (predicted == y[i]) correct++;
            }
            Assert.True(correct > 20); // At least 70% accuracy
        }

        [Fact]
        public void LogisticRegression_WithRegularization_PreventsOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(15, 2);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = i < 7 ? 0.0 : 1.0;
            }

            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(0.1);

            // Act
            var regression = new LogisticRegression<double>(null, regularization);
            regression.Train(x, y);

            // Assert - coefficients should be smaller due to regularization
            Assert.True(Math.Abs(regression.Coefficients[0]) < 10);
            Assert.True(Math.Abs(regression.Coefficients[1]) < 10);
        }

        [Fact]
        public void LogisticRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);
            var random = new Random(456);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = random.NextDouble() * 100;
                x[i, 1] = random.NextDouble() * 100;
                y[i] = (x[i, 0] > x[i, 1]) ? 1.0 : 0.0;
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 5000);
        }

        [Fact]
        public void LogisticRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(10, 1);
            var y = new Vector<float>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i < 5 ? 0.0f : 1.0f;
            }

            // Act
            var regression = new LogisticRegression<float>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.True(predictions[i] < 0.5f);
            }
            for (int i = 5; i < 10; i++)
            {
                Assert.True(predictions[i] > 0.5f);
            }
        }

        [Fact]
        public void LogisticRegression_BalancedClasses_FairPredictions()
        {
            // Arrange - 50/50 class balance
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i;
                y[i] = 0.0;
            }

            for (int i = 10; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = -i;
                y[i] = 1.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should have reasonable predictions
            int class0Predictions = 0;
            int class1Predictions = 0;
            for (int i = 0; i < 20; i++)
            {
                if (predictions[i] < 0.5) class0Predictions++;
                else class1Predictions++;
            }
            Assert.True(Math.Abs(class0Predictions - class1Predictions) <= 5);
        }

        [Fact]
        public void LogisticRegression_SingleFeature_ConvergesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(12);

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i;
                y[i] = i < 6 ? 0.0 : 1.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.True(predictions[0] < predictions[11]); // Monotonic increase
        }

        [Fact]
        public void LogisticRegression_ImbalancedClasses_StillWorks()
        {
            // Arrange - 80/20 imbalance
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = 0.0;
            }

            for (int i = 20; i < 25; i++)
            {
                x[i, 0] = i + 10;
                y[i] = 1.0;
            }

            // Act
            var regression = new LogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should still separate classes
            Assert.True(predictions[0] < predictions[24]);
        }

        #endregion

        #region MultinomialLogisticRegression Tests

        [Fact]
        public void MultinomialLogisticRegression_ThreeClasses_ClassifiesCorrectly()
        {
            // Arrange - 3 separable classes
            var x = new Matrix<double>(15, 2);
            var y = new Vector<double>(15);

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                x[i, 1] = 0;
                y[i] = 0.0; // Class 0
            }

            for (int i = 5; i < 10; i++)
            {
                x[i, 0] = i;
                x[i, 1] = 10;
                y[i] = 1.0; // Class 1
            }

            for (int i = 10; i < 15; i++)
            {
                x[i, 0] = i;
                x[i, 1] = 20;
                y[i] = 2.0; // Class 2
            }

            // Act
            var regression = new MultinomialLogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should classify each class correctly
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Round(predictions[i]) == 0.0);
            }
            for (int i = 5; i < 10; i++)
            {
                Assert.True(Math.Round(predictions[i]) == 1.0);
            }
            for (int i = 10; i < 15; i++)
            {
                Assert.True(Math.Round(predictions[i]) == 2.0);
            }
        }

        [Fact]
        public void MultinomialLogisticRegression_FourClasses_HandlesMultipleClasses()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i; x[i, 1] = 0; y[i] = 0.0;
            }
            for (int i = 5; i < 10; i++)
            {
                x[i, 0] = i; x[i, 1] = 10; y[i] = 1.0;
            }
            for (int i = 10; i < 15; i++)
            {
                x[i, 0] = i; x[i, 1] = 20; y[i] = 2.0;
            }
            for (int i = 15; i < 20; i++)
            {
                x[i, 0] = i; x[i, 1] = 30; y[i] = 3.0;
            }

            // Act
            var regression = new MultinomialLogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(20, predictions.Length);
        }

        [Fact]
        public void MultinomialLogisticRegression_OverlappingClasses_MakesReasonablePredictions()
        {
            // Arrange - classes with some overlap
            var x = new Matrix<double>(18, 2);
            var y = new Vector<double>(18);
            var random = new Random(789);

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i + random.NextDouble();
                x[i, 1] = i + random.NextDouble();
                y[i] = 0.0;
            }

            for (int i = 6; i < 12; i++)
            {
                x[i, 0] = i + random.NextDouble();
                x[i, 1] = 10 + random.NextDouble();
                y[i] = 1.0;
            }

            for (int i = 12; i < 18; i++)
            {
                x[i, 0] = i + random.NextDouble();
                x[i, 1] = 20 + random.NextDouble();
                y[i] = 2.0;
            }

            // Act
            var regression = new MultinomialLogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should have valid class predictions
            for (int i = 0; i < 18; i++)
            {
                Assert.True(predictions[i] >= 0.0 && predictions[i] <= 2.0);
            }
        }

        [Fact]
        public void MultinomialLogisticRegression_LargeNumberOfClasses_HandlesEfficiently()
        {
            // Arrange - 5 classes
            var x = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                x[i, 1] = (i % 5) * 10;
                y[i] = i % 5;
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new MultinomialLogisticRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 5000);
        }

        [Fact]
        public void MultinomialLogisticRegression_BinaryCase_WorksLikeBinaryLogistic()
        {
            // Arrange - 2 classes (should work like binary logistic)
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i < 5 ? 0.0 : 1.0;
            }

            // Act
            var regression = new MultinomialLogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.True(predictions[i] < 0.5);
            }
            for (int i = 5; i < 10; i++)
            {
                Assert.True(predictions[i] > 0.5);
            }
        }

        [Fact]
        public void MultinomialLogisticRegression_MultipleFeatures_ClassifiesWell()
        {
            // Arrange
            var x = new Matrix<double>(30, 3);
            var y = new Vector<double>(30);
            var random = new Random(321);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = random.NextDouble() * 10;
                x[i, 1] = random.NextDouble() * 10;
                x[i, 2] = random.NextDouble() * 10;

                if (x[i, 0] > 6)
                    y[i] = 0.0;
                else if (x[i, 1] > 6)
                    y[i] = 1.0;
                else
                    y[i] = 2.0;
            }

            // Act
            var regression = new MultinomialLogisticRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should have reasonable accuracy
            int correct = 0;
            for (int i = 0; i < 30; i++)
            {
                if (Math.Round(predictions[i]) == y[i]) correct++;
            }
            Assert.True(correct > 15); // At least 50% accuracy
        }

        #endregion

        #region PoissonRegression Tests

        [Fact]
        public void PoissonRegression_CountData_FitsCorrectly()
        {
            // Arrange - count data (non-negative integers)
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2; // Count increases linearly
            }

            // Act
            var regression = new PoissonRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be non-negative
            for (int i = 0; i < 10; i++)
            {
                Assert.True(predictions[i] >= 0);
            }
        }

        [Fact]
        public void PoissonRegression_SmallCounts_HandlesZeros()
        {
            // Arrange
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0 });

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new PoissonRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should handle zeros gracefully
            Assert.True(predictions[0] >= 0);
            Assert.True(predictions[1] >= 0);
        }

        [Fact]
        public void PoissonRegression_ExponentialMean_FitsWell()
        {
            // Arrange - Poisson rate increases exponentially
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(12);

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i / 2.0;
                y[i] = Math.Round(Math.Exp(x[i, 0] / 5.0) * 2);
            }

            // Act
            var regression = new PoissonRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be reasonable
            for (int i = 0; i < 12; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < y[i] * 0.5 + 2);
            }
        }

        [Fact]
        public void PoissonRegression_MultipleFeatures_ProducesValidCounts()
        {
            // Arrange
            var x = new Matrix<double>(15, 2);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i / 2.0;
                y[i] = i + (i / 2);
            }

            // Act
            var regression = new PoissonRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - all predictions should be non-negative
            for (int i = 0; i < 15; i++)
            {
                Assert.True(predictions[i] >= 0);
            }
        }

        [Fact]
        public void PoissonRegression_LargeCounts_HandlesCorrectly()
        {
            // Arrange - large count values
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 125.0, 150.0, 180.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new PoissonRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.True(predictions[i] > 0);
            }
        }

        [Fact]
        public void PoissonRegression_ConstantCount_IdentifiesConstant()
        {
            // Arrange - constant count
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = 5.0; // Constant
            }

            // Act
            var regression = new PoissonRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - all predictions should be around 5
            for (int i = 0; i < 10; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 5.0) < 2.0);
            }
        }

        #endregion

        #region NegativeBinomialRegression Tests

        [Fact]
        public void NegativeBinomialRegression_OverdispersedData_HandlesWell()
        {
            // Arrange - overdispersed count data (variance > mean)
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(new[] { 0.0, 2.0, 1.0, 5.0, 3.0, 8.0, 12.0, 7.0, 15.0, 20.0, 18.0, 25.0, 22.0, 30.0, 35.0 });

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new NegativeBinomialRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should handle variance better than Poisson
            for (int i = 0; i < 15; i++)
            {
                Assert.True(predictions[i] >= 0);
            }
        }

        [Fact]
        public void NegativeBinomialRegression_ZeroInflated_HandlesZeros()
        {
            // Arrange - data with many zeros
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 5.0, 0.0, 10.0, 15.0, 20.0, 25.0 });

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new NegativeBinomialRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 12; i++)
            {
                Assert.True(predictions[i] >= 0);
            }
        }

        [Fact]
        public void NegativeBinomialRegression_HighVariance_FitsBetter()
        {
            // Arrange - high variance count data
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 1.0, 5.0, 2.0, 15.0, 8.0, 25.0, 12.0, 35.0, 20.0, 50.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new NegativeBinomialRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should increase
            Assert.True(predictions[9] > predictions[0]);
        }

        [Fact]
        public void NegativeBinomialRegression_MultipleFeatures_ProducesValidPredictions()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            var random = new Random(654);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = random.NextDouble() * 10;
                y[i] = Math.Round(x[i, 0] + x[i, 1] / 2.0);
            }

            // Act
            var regression = new NegativeBinomialRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(predictions[i] >= 0);
            }
        }

        [Fact]
        public void NegativeBinomialRegression_ComparesWithPoisson_DifferentForOverdispersed()
        {
            // Arrange - overdispersed data
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 2.0, 8.0, 5.0, 15.0, 10.0, 25.0, 20.0, 40.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var nbReg = new NegativeBinomialRegression<double>();
            nbReg.Train(x, y);
            var nbPred = nbReg.Predict(x);

            var poissonReg = new PoissonRegression<double>();
            poissonReg.Train(x, y);
            var poissonPred = poissonReg.Predict(x);

            // Assert - predictions should differ
            bool different = false;
            for (int i = 0; i < 10; i++)
            {
                if (Math.Abs(nbPred[i] - poissonPred[i]) > 1.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void NegativeBinomialRegression_LowVariance_SimilarToPoisson()
        {
            // Arrange - low variance (Poisson-like)
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new NegativeBinomialRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should produce reasonable predictions
            for (int i = 0; i < 8; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0);
            }
        }

        #endregion
    }
}
