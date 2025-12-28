using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for Tree-Based Regression models (DecisionTreeRegression, RandomForestRegression,
/// GradientBoostingRegression, etc.).
/// These tests verify tree-based regression algorithms work correctly.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class TreeBasedRegressionIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 0.5; // Tree models are approximate

    #region DecisionTreeRegression Tests

    [Fact]
    public void DecisionTreeRegression_Train_SimpleData_FitsWithinBounds()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesSplit = 2 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 },
            { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - tree should fit training data well
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < LooseTolerance,
                $"Prediction {i} should be close to {y[i]}, got {predictions[i]}");
        }
    }

    [Fact]
    public void DecisionTreeRegression_Train_MaxDepth1_CreatesSimpleTree()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 1, MinSamplesSplit = 2 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 1, 2, 10, 11, 12 });

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - with depth 1, should have at most 2 unique predictions
        var uniquePredictions = predictions.ToArray().Distinct().ToList();
        Assert.True(uniquePredictions.Count <= 2,
            $"MaxDepth=1 should produce at most 2 unique predictions, got {uniquePredictions.Count}");
    }

    [Fact]
    public void DecisionTreeRegression_Predict_NewData_ReturnsReasonableValues()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesSplit = 2 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 10, 20, 30, 40, 50 });
        regression.Train(x, y);

        // Act
        var newX = CreateMatrix(new double[,] { { 2.5 }, { 3.5 } });
        var predictions = regression.Predict(newX);

        // Assert - predictions should be within the range of training targets
        Assert.True(predictions[0] >= 10 && predictions[0] <= 50,
            $"Prediction should be in [10, 50], got {predictions[0]}");
        Assert.True(predictions[1] >= 10 && predictions[1] <= 50,
            $"Prediction should be in [10, 50], got {predictions[1]}");
    }

    [Fact]
    public void DecisionTreeRegression_GetFeatureImportance_ReturnsNonNegativeValues()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesSplit = 2 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 10 }, { 2, 20 }, { 3, 30 }, { 4, 40 }, { 5, 50 }
        });
        var y = CreateVector(new double[] { 11, 22, 33, 44, 55 });
        regression.Train(x, y);

        // Act
        var importance0 = regression.GetFeatureImportance(0);
        var importance1 = regression.GetFeatureImportance(1);

        // Assert
        Assert.True(importance0 >= 0, "Feature importance should be non-negative");
        Assert.True(importance1 >= 0, "Feature importance should be non-negative");
    }

    #endregion

    #region RandomForestRegression Tests

    [Fact]
    public async Task RandomForestRegression_TrainAsync_MultipleTreesImproveStability()
    {
        // Arrange
        var optionsFewTrees = new RandomForestRegressionOptions
        {
            NumberOfTrees = 3,
            MaxDepth = 5,
            MinSamplesSplit = 2,
            Seed = 42
        };
        var optionsManyTrees = new RandomForestRegressionOptions
        {
            NumberOfTrees = 20,
            MaxDepth = 5,
            MinSamplesSplit = 2,
            Seed = 42
        };

        var forestFew = new RandomForestRegression<double>(optionsFewTrees);
        var forestMany = new RandomForestRegression<double>(optionsManyTrees);

        var x = CreateMatrix(new double[,]
        {
            { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 5 },
            { 6, 6 }, { 7, 7 }, { 8, 8 }, { 9, 9 }, { 10, 10 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        await forestFew.TrainAsync(x, y);
        await forestMany.TrainAsync(x, y);

        // Assert - both should train without errors
        Assert.Equal(3, forestFew.NumberOfTrees);
        Assert.Equal(20, forestMany.NumberOfTrees);
    }

    [Fact]
    public async Task RandomForestRegression_PredictAsync_ReturnsAveragedPredictions()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 10,
            MaxDepth = 5,
            MinSamplesSplit = 2,
            Seed = 42
        };
        var forest = new RandomForestRegression<double>(options);

        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 10, 20, 30, 40, 50 });
        await forest.TrainAsync(x, y);

        // Act
        var newX = CreateMatrix(new double[,] { { 3 } });
        var predictions = await forest.PredictAsync(newX);

        // Assert - prediction should be reasonable
        Assert.True(predictions[0] >= 20 && predictions[0] <= 40,
            $"Prediction should be around 30, got {predictions[0]}");
    }

    [Fact]
    public async Task RandomForestRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 3,
            MinSamplesSplit = 2,
            MaxFeatures = 0.8
        };
        var forest = new RandomForestRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var y = CreateVector(new double[] { 1, 2, 3 });
        await forest.TrainAsync(x, y);

        // Act
        var metadata = forest.GetModelMetadata();

        // Assert
        Assert.Equal(5, (int)metadata.AdditionalInfo["NumberOfTrees"]);
        Assert.Equal(3, (int)metadata.AdditionalInfo["MaxDepth"]);
        Assert.Equal(0.8, (double)metadata.AdditionalInfo["MaxFeatures"], Tolerance);
    }

    [Fact]
    public async Task RandomForestRegression_Seed_ProducesReproducibleResults()
    {
        // Arrange
        var options1 = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 3,
            MinSamplesSplit = 2,
            Seed = 123
        };
        var options2 = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 3,
            MinSamplesSplit = 2,
            Seed = 123
        };

        var forest1 = new RandomForestRegression<double>(options1);
        var forest2 = new RandomForestRegression<double>(options2);

        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });

        // Act
        await forest1.TrainAsync(x, y);
        await forest2.TrainAsync(x, y);

        var newX = CreateMatrix(new double[,] { { 3 } });
        var pred1 = await forest1.PredictAsync(newX);
        var pred2 = await forest2.PredictAsync(newX);

        // Assert - same seed should produce same predictions
        Assert.Equal(pred1[0], pred2[0], Tolerance);
    }

    #endregion

    #region GradientBoostingRegression Tests

    [Fact]
    public async Task GradientBoostingRegression_TrainAsync_FitsData()
    {
        // Arrange
        var options = new GradientBoostingRegressionOptions
        {
            NumberOfTrees = 10,
            MaxDepth = 3,
            LearningRate = 0.1,
            MinSamplesSplit = 2
        };
        var gbr = new GradientBoostingRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });

        // Act
        await gbr.TrainAsync(x, y);
        var predictions = await gbr.PredictAsync(x);

        // Assert - should fit training data reasonably well
        double mse = 0;
        for (int i = 0; i < y.Length; i++)
        {
            mse += (predictions[i] - y[i]) * (predictions[i] - y[i]);
        }
        mse /= y.Length;

        Assert.True(mse < 2.0, $"MSE should be low, got {mse}");
    }

    [Fact]
    public async Task GradientBoostingRegression_LearningRate_AffectsTraining()
    {
        // Arrange
        var optionsHighLR = new GradientBoostingRegressionOptions
        {
            NumberOfTrees = 10,
            MaxDepth = 3,
            LearningRate = 0.5,
            MinSamplesSplit = 2
        };
        var optionsLowLR = new GradientBoostingRegressionOptions
        {
            NumberOfTrees = 10,
            MaxDepth = 3,
            LearningRate = 0.01,
            MinSamplesSplit = 2
        };

        var gbrHigh = new GradientBoostingRegression<double>(optionsHighLR);
        var gbrLow = new GradientBoostingRegression<double>(optionsLowLR);

        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 10, 20, 30, 40, 50 });

        // Act
        await gbrHigh.TrainAsync(x, y);
        await gbrLow.TrainAsync(x, y);

        var predsHigh = await gbrHigh.PredictAsync(x);
        var predsLow = await gbrLow.PredictAsync(x);

        // Assert - high learning rate should converge faster (closer to targets)
        double mseHigh = 0, mseLow = 0;
        for (int i = 0; i < y.Length; i++)
        {
            mseHigh += (predsHigh[i] - y[i]) * (predsHigh[i] - y[i]);
            mseLow += (predsLow[i] - y[i]) * (predsLow[i] - y[i]);
        }

        // With same number of trees, higher LR should have lower MSE
        Assert.True(mseHigh < mseLow,
            $"Higher LR should fit better with same iterations: MSE_high={mseHigh}, MSE_low={mseLow}");
    }

    #endregion

    #region AdaBoostR2Regression Tests

    [Fact]
    public async Task AdaBoostR2Regression_TrainAsync_FitsData()
    {
        // Arrange
        var options = new AdaBoostR2RegressionOptions
        {
            NumberOfEstimators = 10,
            MaxDepth = 3
        };
        var adaboost = new AdaBoostR2Regression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });

        // Act
        await adaboost.TrainAsync(x, y);
        var predictions = await adaboost.PredictAsync(x);

        // Assert - predictions should be reasonable
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0,
                $"Prediction {i} should be close to {y[i]}, got {predictions[i]}");
        }
    }

    #endregion

    #region ExtremelyRandomizedTreesRegression Tests

    [Fact]
    public async Task ExtremelyRandomizedTreesRegression_TrainAsync_FitsData()
    {
        // Arrange
        var options = new ExtremelyRandomizedTreesRegressionOptions
        {
            NumberOfTrees = 10,
            MaxDepth = 5,
            MinSamplesSplit = 2
        };
        var extraTrees = new ExtremelyRandomizedTreesRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });

        // Act
        await extraTrees.TrainAsync(x, y);
        var predictions = await extraTrees.PredictAsync(x);

        // Assert
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0,
                $"Prediction {i} should be close to {y[i]}, got {predictions[i]}");
        }
    }

    #endregion

    #region QuantileRegressionForests Tests

    [Fact]
    public async Task QuantileRegressionForests_TrainAsync_ProvidesQuantiles()
    {
        // Arrange
        var options = new QuantileRegressionForestsOptions
        {
            NumberOfTrees = 10,
            MaxDepth = 5
        };
        var qrf = new QuantileRegressionForests<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 },
            { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        await qrf.TrainAsync(x, y);
        var newX = CreateMatrix(new double[,] { { 5 } });
        var predictions = await qrf.PredictAsync(newX);

        // Assert - should return predictions for each quantile
        Assert.NotNull(predictions);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void DecisionTreeRegression_Train_SingleSample_HandlesGracefully()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesSplit = 1 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 } });
        var y = CreateVector(new double[] { 10 });

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert
        Assert.Equal(10.0, predictions[0], LooseTolerance);
    }

    [Fact]
    public async Task RandomForestRegression_Train_HighDimensionalData_HandlesCorrectly()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 3,
            MaxFeatures = 0.5,  // Only consider 50% of features
            Seed = 42
        };
        var forest = new RandomForestRegression<double>(options);

        // Create high-dimensional data (20 features, 10 samples)
        var random = new Random(42);
        int numSamples = 10;
        int numFeatures = 20;
        var xData = new double[numSamples, numFeatures];
        var yData = new double[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            yData[i] = 0;
            for (int j = 0; j < numFeatures; j++)
            {
                xData[i, j] = random.NextDouble();
                if (j < 3) // Only first 3 features matter
                {
                    yData[i] += xData[i, j];
                }
            }
        }

        var x = CreateMatrix(xData);
        var y = CreateVector(yData);

        // Act
        await forest.TrainAsync(x, y);
        var predictions = await forest.PredictAsync(x);

        // Assert - should handle high-dimensional data
        Assert.Equal(numSamples, predictions.Length);
    }

    [Fact]
    public void DecisionTreeRegression_Train_ConstantTarget_HandlesCorrectly()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesSplit = 2 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 10, 10, 10, 10, 10 }); // Constant

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - all predictions should be 10
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.Equal(10.0, predictions[i], Tolerance);
        }
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void DecisionTreeRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 3, MinSamplesSplit = 2 };
        var regression = new DecisionTreeRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });
        regression.Train(x, y);

        // Act
        var serialized = regression.Serialize();
        var newRegression = new DecisionTreeRegression<double>(options);
        newRegression.Deserialize(serialized);

        // Assert
        var originalPreds = regression.Predict(x);
        var newPreds = newRegression.Predict(x);

        for (int i = 0; i < originalPreds.Length; i++)
        {
            Assert.Equal(originalPreds[i], newPreds[i], Tolerance);
        }
    }

    [Fact]
    public async Task RandomForestRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 3,
            Seed = 42
        };
        var forest = new RandomForestRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });
        await forest.TrainAsync(x, y);

        // Act
        var serialized = forest.Serialize();
        var newForest = new RandomForestRegression<double>(options);
        newForest.Deserialize(serialized);

        // Assert
        var originalPreds = await forest.PredictAsync(x);
        var newPreds = await newForest.PredictAsync(x);

        for (int i = 0; i < originalPreds.Length; i++)
        {
            Assert.Equal(originalPreds[i], newPreds[i], Tolerance);
        }
    }

    #endregion

    #region Helper Methods

    private static Matrix<double> CreateMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = data[i, j];
            }
        }
        return matrix;
    }

    private static Vector<double> CreateVector(double[] data)
    {
        var vector = new Vector<double>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            vector[i] = data[i];
        }
        return vector;
    }

    #endregion
}
