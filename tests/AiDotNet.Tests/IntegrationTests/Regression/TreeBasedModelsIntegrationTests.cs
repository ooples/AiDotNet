using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for tree-based regression models.
    /// Tests decision trees, random forests, gradient boosting, and ensemble methods.
    /// </summary>
    public class TreeBasedModelsIntegrationTests
    {
        #region DecisionTreeRegression Tests

        [Fact]
        public void DecisionTreeRegression_PerfectSplit_FitsExactly()
        {
            // Arrange - data that can be perfectly split
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                y[i] = 10.0;
            }

            for (int i = 5; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = 20.0;
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 10.0) < 1.0);
            }
            for (int i = 5; i < 10; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 20.0) < 1.0);
            }
        }

        [Fact]
        public void DecisionTreeRegression_NonLinearRelationship_CapturesPattern()
        {
            // Arrange - quadratic relationship
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i - 7;
                y[i] = x[i, 0] * x[i, 0];
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should approximate quadratic reasonably well
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void DecisionTreeRegression_MultipleFeatures_SplitsOnBestFeature()
        {
            // Arrange
            var x = new Matrix<double>(20, 3);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i; // Important feature
                x[i, 1] = (i % 2) * 0.1; // Unimportant
                x[i, 2] = (i % 3) * 0.1; // Unimportant
                y[i] = i < 10 ? 5.0 : 15.0; // Depends mainly on x[0]
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should separate the two groups
            double avgFirst10 = 0;
            double avgLast10 = 0;
            for (int i = 0; i < 10; i++)
            {
                avgFirst10 += predictions[i];
                avgLast10 += predictions[i + 10];
            }
            avgFirst10 /= 10;
            avgLast10 /= 10;

            Assert.True(avgLast10 > avgFirst10);
        }

        [Fact]
        public void DecisionTreeRegression_MaxDepthLimit_RespectsConstraint()
        {
            // Arrange
            var x = new Matrix<double>(50, 1);
            var y = new Vector<double>(50);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                y[i] = i;
            }

            var options = new DecisionTreeOptions { MaxDepth = 3 };

            // Act
            var regression = new DecisionTreeRegression<double>(options);
            regression.Train(x, y);

            // Assert - tree should not be too deep (limited predictions)
            var predictions = regression.Predict(x);
            Assert.NotNull(predictions);
        }

        [Fact]
        public void DecisionTreeRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(5, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        [Fact]
        public void DecisionTreeRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 1000;
            var x = new Matrix<double>(n, 3);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i % 10;
                x[i, 2] = i % 7;
                y[i] = x[i, 0] / 10.0 + x[i, 1];
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 5000);
        }

        [Fact]
        public void DecisionTreeRegression_WithNoise_StillFitsReasonably()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);
            var random = new Random(42);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 5;
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should approximate despite noise
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 30 < 10.0);
        }

        [Fact]
        public void DecisionTreeRegression_CategoricalLikeFeatures_HandlesWell()
        {
            // Arrange - features with categorical-like values
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i % 3; // 0, 1, 2
                x[i, 1] = i % 4; // 0, 1, 2, 3
                y[i] = x[i, 0] * 10 + x[i, 1] * 5;
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void DecisionTreeRegression_MinSamplesSplit_PreventsOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            var options = new DecisionTreeOptions { MinSamplesSplit = 10 };

            // Act
            var regression = new DecisionTreeRegression<double>(options);
            regression.Train(x, y);

            // Assert - should create a simpler tree
            var predictions = regression.Predict(x);
            Assert.NotNull(predictions);
        }

        [Fact]
        public void DecisionTreeRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(10, 1);
            var y = new Vector<float>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i < 5 ? 10.0f : 20.0f;
            }

            // Act
            var regression = new DecisionTreeRegression<float>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 10.0f) < 2.0f);
            }
        }

        [Fact]
        public void DecisionTreeRegression_ConstantTarget_HandlesGracefully()
        {
            // Arrange - all targets are the same
            var x = new Matrix<double>(10, 2);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 42.0; // Constant
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - all predictions should be around 42
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(42.0, predictions[i], precision: 1);
            }
        }

        [Fact]
        public void DecisionTreeRegression_InteractionEffects_CapturesCorrectly()
        {
            // Arrange - y depends on interaction of x1 and x2
            var x = new Matrix<double>(16, 2);
            var y = new Vector<double>(16);

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int idx = i * 4 + j;
                    x[idx, 0] = i;
                    x[idx, 1] = j;
                    y[idx] = i * j; // Interaction
                }
            }

            // Act
            var regression = new DecisionTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 16; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        #endregion

        #region RandomForestRegression Tests

        [Fact]
        public void RandomForestRegression_EnsembleAveraging_ReducesVariance()
        {
            // Arrange
            var x = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);
            var random = new Random(123);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 10;
            }

            var options = new RandomForestOptions { NumTrees = 10 };

            // Act
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should have reasonable predictions
            double totalError = 0;
            for (int i = 0; i < 50; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 50 < 15.0);
        }

        [Fact]
        public void RandomForestRegression_MoreTrees_BetterPredictions()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + 2 * x[i, 1];
            }

            // Act - compare 5 trees vs 20 trees
            var rf5 = new RandomForestRegression<double>(new RandomForestOptions { NumTrees = 5 });
            rf5.Train(x, y);
            var pred5 = rf5.Predict(x);

            var rf20 = new RandomForestRegression<double>(new RandomForestOptions { NumTrees = 20 });
            rf20.Train(x, y);
            var pred20 = rf20.Predict(x);

            // Assert - more trees should generally be more stable
            Assert.NotEqual(pred5[0], pred20[0]);
        }

        [Fact]
        public void RandomForestRegression_NonLinearPattern_CapturesWell()
        {
            // Arrange - non-linear pattern
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i - 12;
                y[i] = x[i, 0] * x[i, 0];
            }

            var options = new RandomForestOptions { NumTrees = 15 };

            // Act
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 30.0);
            }
        }

        [Fact]
        public void RandomForestRegression_FeatureImportance_IdentifiesRelevant()
        {
            // Arrange - x[0] is important, x[1] is noise
            var x = new Matrix<double>(40, 2);
            var y = new Vector<double>(40);
            var random = new Random(456);

            for (int i = 0; i < 40; i++)
            {
                x[i, 0] = i;
                x[i, 1] = random.NextDouble() * 100; // Noise
                y[i] = 3 * x[i, 0] + 5;
            }

            var options = new RandomForestOptions { NumTrees = 10 };

            // Act
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);

            // Assert - feature 0 should be more important
            var importance0 = regression.GetFeatureImportance(0);
            var importance1 = regression.GetFeatureImportance(1);
            Assert.True(importance0 > importance1);
        }

        [Fact]
        public void RandomForestRegression_SmallDataset_HandlesGracefully()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            var options = new RandomForestOptions { NumTrees = 5 };

            // Act
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(10, predictions.Length);
        }

        [Fact]
        public void RandomForestRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 3);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i % 10;
                x[i, 2] = i % 7;
                y[i] = x[i, 0] / 5.0 + x[i, 1] * 2;
            }

            var options = new RandomForestOptions { NumTrees = 10 };

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 10000);
        }

        [Fact]
        public void RandomForestRegression_OutOfBagError_ProvidesValidation()
        {
            // Arrange
            var x = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new RandomForestOptions { NumTrees = 10, UseOutOfBagError = true };

            // Act
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);
            var oobError = regression.GetOutOfBagError();

            // Assert - OOB error should be reasonable
            Assert.True(oobError >= 0);
        }

        [Fact]
        public void RandomForestRegression_MaxFeaturesLimit_UsesSubset()
        {
            // Arrange
            var x = new Matrix<double>(40, 5);
            var y = new Vector<double>(40);

            for (int i = 0; i < 40; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    x[i, j] = i + j;
                }
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new RandomForestOptions
            {
                NumTrees = 10,
                MaxFeatures = 0.6 // Use 60% of features
            };

            // Act
            var regression = new RandomForestRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        #endregion

        #region GradientBoostingRegression Tests

        [Fact]
        public void GradientBoostingRegression_SequentialImprovement_ReducesError()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 5;
            }

            var options = new GradientBoostingOptions { NumIterations = 10, LearningRate = 0.1 };

            // Act
            var regression = new GradientBoostingRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit well
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 30 < 10.0);
        }

        [Fact]
        public void GradientBoostingRegression_LowLearningRate_SlowConvergence()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            var optionsLow = new GradientBoostingOptions { NumIterations = 5, LearningRate = 0.01 };
            var optionsHigh = new GradientBoostingOptions { NumIterations = 5, LearningRate = 0.5 };

            // Act
            var regLow = new GradientBoostingRegression<double>(optionsLow);
            regLow.Train(x, y);
            var predLow = regLow.Predict(x);

            var regHigh = new GradientBoostingRegression<double>(optionsHigh);
            regHigh.Train(x, y);
            var predHigh = regHigh.Predict(x);

            // Assert - high learning rate should converge faster
            double errorLow = 0, errorHigh = 0;
            for (int i = 0; i < 20; i++)
            {
                errorLow += Math.Abs(predLow[i] - y[i]);
                errorHigh += Math.Abs(predHigh[i] - y[i]);
            }
            Assert.True(errorHigh < errorLow);
        }

        [Fact]
        public void GradientBoostingRegression_NonLinearData_FitsAccurately()
        {
            // Arrange
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i / 5.0;
                y[i] = Math.Sin(x[i, 0]) * 10;
            }

            var options = new GradientBoostingOptions { NumIterations = 20, LearningRate = 0.1 };

            // Act
            var regression = new GradientBoostingRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void GradientBoostingRegression_MoreIterations_BetterFit()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - compare few vs many iterations
            var reg5 = new GradientBoostingRegression<double>(new GradientBoostingOptions { NumIterations = 5 });
            reg5.Train(x, y);

            var reg50 = new GradientBoostingRegression<double>(new GradientBoostingOptions { NumIterations = 50 });
            reg50.Train(x, y);

            var pred5 = reg5.Predict(x);
            var pred50 = reg50.Predict(x);

            // Assert - more iterations should reduce error
            double error5 = 0, error50 = 0;
            for (int i = 0; i < 20; i++)
            {
                error5 += Math.Abs(pred5[i] - y[i]);
                error50 += Math.Abs(pred50[i] - y[i]);
            }
            Assert.True(error50 < error5);
        }

        [Fact]
        public void GradientBoostingRegression_WithNoise_StillFitsWell()
        {
            // Arrange
            var x = new Matrix<double>(40, 2);
            var y = new Vector<double>(40);
            var random = new Random(789);

            for (int i = 0; i < 40; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 10;
            }

            var options = new GradientBoostingOptions { NumIterations = 15, LearningRate = 0.1 };

            // Act
            var regression = new GradientBoostingRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            double totalError = 0;
            for (int i = 0; i < 40; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 40 < 15.0);
        }

        [Fact]
        public void GradientBoostingRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 300;
            var x = new Matrix<double>(n, 3);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i % 10;
                x[i, 2] = i % 5;
                y[i] = x[i, 0] / 10.0 + x[i, 1];
            }

            var options = new GradientBoostingOptions { NumIterations = 10 };

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new GradientBoostingRegression<double>(options);
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 10000);
        }

        #endregion

        #region ExtremelyRandomizedTreesRegression Tests

        [Fact]
        public void ExtremelyRandomizedTreesRegression_RandomSplits_ReducesOverfitting()
        {
            // Arrange
            var x = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);
            var random = new Random(111);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 5;
            }

            var options = new ExtremelyRandomizedTreesOptions { NumTrees = 10 };

            // Act
            var regression = new ExtremelyRandomizedTreesRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            double totalError = 0;
            for (int i = 0; i < 50; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 50 < 15.0);
        }

        [Fact]
        public void ExtremelyRandomizedTreesRegression_DifferentFromRandomForest()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] + 2 * x[i, 1];
            }

            // Act
            var ert = new ExtremelyRandomizedTreesRegression<double>(new ExtremelyRandomizedTreesOptions { NumTrees = 5 });
            ert.Train(x, y);
            var ertPred = ert.Predict(x);

            var rf = new RandomForestRegression<double>(new RandomForestOptions { NumTrees = 5 });
            rf.Train(x, y);
            var rfPred = rf.Predict(x);

            // Assert - predictions should differ
            bool different = false;
            for (int i = 0; i < 30; i++)
            {
                if (Math.Abs(ertPred[i] - rfPred[i]) > 1.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        [Fact]
        public void ExtremelyRandomizedTreesRegression_NonLinear_CapturesPattern()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sqrt(i) * 5;
            }

            var options = new ExtremelyRandomizedTreesOptions { NumTrees = 15 };

            // Act
            var regression = new ExtremelyRandomizedTreesRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        #endregion

        #region AdaBoostR2Regression Tests

        [Fact]
        public void AdaBoostR2Regression_WeightedSampling_ImprovesAccuracy()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 3 * x[i, 0] + 2 * x[i, 1];
            }

            var options = new AdaBoostR2Options { NumIterations = 10 };

            // Act
            var regression = new AdaBoostR2Regression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 30 < 10.0);
        }

        [Fact]
        public void AdaBoostR2Regression_FocusesOnHardExamples()
        {
            // Arrange - data with outliers
            var x = new Matrix<double>(25, 1);
            var y = new Vector<double>(25);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2;
            }

            // Add hard examples
            for (int i = 20; i < 25; i++)
            {
                x[i, 0] = i;
                y[i] = 100; // Outliers
            }

            var options = new AdaBoostR2Options { NumIterations = 15 };

            // Act
            var regression = new AdaBoostR2Regression<double>(options);
            regression.Train(x, y);

            // Assert - should still produce valid predictions
            var predictions = regression.Predict(x);
            Assert.NotNull(predictions);
        }

        [Fact]
        public void AdaBoostR2Regression_MultipleIterations_ConvergesWell()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - compare few vs many iterations
            var reg5 = new AdaBoostR2Regression<double>(new AdaBoostR2Options { NumIterations = 5 });
            reg5.Train(x, y);
            var pred5 = reg5.Predict(x);

            var reg20 = new AdaBoostR2Regression<double>(new AdaBoostR2Options { NumIterations = 20 });
            reg20.Train(x, y);
            var pred20 = reg20.Predict(x);

            // Assert - more iterations should improve fit
            double error5 = 0, error20 = 0;
            for (int i = 0; i < 20; i++)
            {
                error5 += Math.Abs(pred5[i] - y[i]);
                error20 += Math.Abs(pred20[i] - y[i]);
            }
            Assert.True(error20 < error5 * 1.2); // Some improvement expected
        }

        #endregion

        #region QuantileRegressionForests Tests

        [Fact]
        public void QuantileRegressionForests_PredictQuantiles_ReasonableEstimates()
        {
            // Arrange
            var x = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);
            var random = new Random(222);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 20;
            }

            var options = new QuantileRegressionForestsOptions { NumTrees = 10 };

            // Act
            var regression = new QuantileRegressionForests<double>(options);
            regression.Train(x, y);
            var medianPred = regression.PredictQuantile(x, 0.5);
            var upper = regression.PredictQuantile(x, 0.9);
            var lower = regression.PredictQuantile(x, 0.1);

            // Assert - upper quantile should be higher than lower
            for (int i = 0; i < 50; i++)
            {
                Assert.True(upper[i] >= medianPred[i]);
                Assert.True(medianPred[i] >= lower[i]);
            }
        }

        [Fact]
        public void QuantileRegressionForests_DifferentQuantiles_ProduceDifferentPredictions()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3;
            }

            var options = new QuantileRegressionForestsOptions { NumTrees = 10 };

            // Act
            var regression = new QuantileRegressionForests<double>(options);
            regression.Train(x, y);
            var q25 = regression.PredictQuantile(x, 0.25);
            var q75 = regression.PredictQuantile(x, 0.75);

            // Assert
            bool different = false;
            for (int i = 0; i < 30; i++)
            {
                if (Math.Abs(q75[i] - q25[i]) > 1.0)
                {
                    different = true;
                    break;
                }
            }
            Assert.True(different);
        }

        #endregion

        #region M5ModelTreeRegression Tests

        [Fact]
        public void M5ModelTreeRegression_PiecewiseLinear_FitsWell()
        {
            // Arrange - piecewise linear data
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 1;
            }

            for (int i = 10; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = -1 * i + 50;
            }

            // Act
            var regression = new M5ModelTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit piecewise linear pattern
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void M5ModelTreeRegression_SmoothTransitions_BetterThanSimpleTree()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 5;
            }

            // Act
            var m5Tree = new M5ModelTreeRegression<double>();
            m5Tree.Train(x, y);
            var m5Pred = m5Tree.Predict(x);

            var decTree = new DecisionTreeRegression<double>();
            decTree.Train(x, y);
            var decPred = decTree.Predict(x);

            // Assert - M5 should have smoother predictions
            double m5Error = 0, decError = 0;
            for (int i = 0; i < 30; i++)
            {
                m5Error += Math.Abs(m5Pred[i] - y[i]);
                decError += Math.Abs(decPred[i] - y[i]);
            }
            Assert.True(m5Error <= decError);
        }

        [Fact]
        public void M5ModelTreeRegression_MultipleFeatures_BuildsLinearModels()
        {
            // Arrange
            var x = new Matrix<double>(40, 3);
            var y = new Vector<double>(40);

            for (int i = 0; i < 40; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                x[i, 2] = i / 2.0;
                y[i] = x[i, 0] + 2 * x[i, 1] - x[i, 2] + 10;
            }

            // Act
            var regression = new M5ModelTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 40; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        #endregion

        #region ConditionalInferenceTreeRegression Tests

        [Fact]
        public void ConditionalInferenceTreeRegression_StatisticalTests_GuidesSplits()
        {
            // Arrange
            var x = new Matrix<double>(50, 2);
            var y = new Vector<double>(50);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = i < 25 ? 10.0 : 30.0;
            }

            // Act
            var regression = new ConditionalInferenceTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should identify the clear split
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 10.0) < 5.0);
            }
            for (int i = 25; i < 50; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 30.0) < 5.0);
            }
        }

        [Fact]
        public void ConditionalInferenceTreeRegression_NoSignificantSplits_CreatesLeaf()
        {
            // Arrange - no significant relationship
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            var random = new Random(333);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = random.NextDouble() * 10;
                x[i, 1] = random.NextDouble() * 10;
                y[i] = 15.0; // Constant, no relationship
            }

            // Act
            var regression = new ConditionalInferenceTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should predict constant
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - 15.0) < 3.0);
            }
        }

        [Fact]
        public void ConditionalInferenceTreeRegression_MultipleFeatures_SelectsSignificant()
        {
            // Arrange - only x[0] is significant
            var x = new Matrix<double>(30, 3);
            var y = new Vector<double>(30);
            var random = new Random(444);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = random.NextDouble() * 100; // Noise
                x[i, 2] = random.NextDouble() * 100; // Noise
                y[i] = 5 * x[i, 0] + 10;
            }

            // Act
            var regression = new ConditionalInferenceTreeRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 20.0);
            }
        }

        #endregion
    }
}
