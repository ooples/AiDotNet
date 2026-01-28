using Xunit;
using AiDotNet.DataProcessor;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Normalizers;
using AiDotNet.FeatureSelectors;
using AiDotNet.AnomalyDetection;

namespace AiDotNet.Tests.IntegrationTests.DataProcessor;

/// <summary>
/// Integration tests for the DataProcessor module.
/// Tests DefaultDataPreprocessor with various configurations for data preprocessing and splitting.
/// </summary>
public class DataProcessorIntegrationTests
{
    #region DataProcessorOptions Tests

    [Fact]
    public void DataProcessorOptions_DefaultValues_AreCorrect()
    {
        var options = new DataProcessorOptions();

        Assert.Equal(0.7, options.TrainingSplitPercentage);
        Assert.Equal(0.15, options.ValidationSplitPercentage);
        Assert.Equal(0.15, options.TestingSplitPercentage);
        Assert.Equal(42, options.RandomSeed);
        Assert.True(options.ShuffleBeforeSplit);
        Assert.True(options.NormalizeBeforeFeatureSelection);
    }

    [Fact]
    public void DataProcessorOptions_CanSetAllProperties()
    {
        var options = new DataProcessorOptions
        {
            TrainingSplitPercentage = 0.8,
            ValidationSplitPercentage = 0.1,
            TestingSplitPercentage = 0.1,
            RandomSeed = 123,
            ShuffleBeforeSplit = false,
            NormalizeBeforeFeatureSelection = false
        };

        Assert.Equal(0.8, options.TrainingSplitPercentage);
        Assert.Equal(0.1, options.ValidationSplitPercentage);
        Assert.Equal(0.1, options.TestingSplitPercentage);
        Assert.Equal(123, options.RandomSeed);
        Assert.False(options.ShuffleBeforeSplit);
        Assert.False(options.NormalizeBeforeFeatureSelection);
    }

    [Fact]
    public void DataProcessorOptions_SplitPercentages_SumToOne()
    {
        var options = new DataProcessorOptions();

        double total = options.TrainingSplitPercentage +
                      options.ValidationSplitPercentage +
                      options.TestingSplitPercentage;

        Assert.Equal(1.0, total, 0.0001);
    }

    #endregion

    #region DefaultDataPreprocessor Construction Tests

    [Fact]
    public void DefaultDataPreprocessor_CanBeConstructed_WithPassThroughDependencies()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        Assert.NotNull(preprocessor);
    }

    [Fact]
    public void DefaultDataPreprocessor_CanBeConstructed_WithOptions()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var options = new DataProcessorOptions
        {
            TrainingSplitPercentage = 0.6,
            ValidationSplitPercentage = 0.2,
            TestingSplitPercentage = 0.2
        };

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options);

        Assert.NotNull(preprocessor);
    }

    #endregion

    #region PreprocessData Tests

    [Fact]
    public void PreprocessData_WithPassThroughComponents_ReturnsUnchangedData()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Create test data
        var X = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var y = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });

        var (processedX, processedY, normInfo) = preprocessor.PreprocessData(X, y);

        // Verify data shape is preserved
        Assert.Equal(X.Rows, processedX.Rows);
        Assert.Equal(X.Columns, processedX.Columns);
        Assert.Equal(y.Length, processedY.Length);
        Assert.NotNull(normInfo);
    }

    [Fact]
    public void PreprocessData_ReturnsNormalizationInfo()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        var X = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var y = new Vector<double>(new double[] { 5.0, 6.0 });

        var (_, _, normInfo) = preprocessor.PreprocessData(X, y);

        Assert.NotNull(normInfo);
        Assert.NotNull(normInfo.Normalizer);
        Assert.NotNull(normInfo.XParams);
        Assert.NotNull(normInfo.YParams);
    }

    [Fact]
    public void PreprocessData_WithNormalizeBeforeFeatureSelection_True()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var options = new DataProcessorOptions { NormalizeBeforeFeatureSelection = true };

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options);

        var X = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var y = new Vector<double>(new double[] { 5.0, 6.0 });

        var (processedX, processedY, normInfo) = preprocessor.PreprocessData(X, y);

        // With pass-through components, data should be unchanged
        Assert.Equal(X.Rows, processedX.Rows);
        Assert.Equal(X.Columns, processedX.Columns);
    }

    [Fact]
    public void PreprocessData_WithNormalizeBeforeFeatureSelection_False()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var options = new DataProcessorOptions { NormalizeBeforeFeatureSelection = false };

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options);

        var X = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var y = new Vector<double>(new double[] { 5.0, 6.0 });

        var (processedX, processedY, normInfo) = preprocessor.PreprocessData(X, y);

        // With pass-through components, data should be unchanged
        Assert.Equal(X.Rows, processedX.Rows);
        Assert.Equal(X.Columns, processedX.Columns);
    }

    #endregion

    #region SplitData Tests - Matrix/Vector

    [Fact]
    public void SplitData_WithDefaultOptions_SplitsCorrectly()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Create dataset with 100 samples
        var X = new Matrix<double>(100, 3);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;
            X[i, 2] = i * 3;
            y[i] = i * 10;
        }

        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(X, y);

        // Default: 70% train, 15% val, 15% test
        Assert.Equal(70, XTrain.Rows);
        Assert.Equal(70, yTrain.Length);
        Assert.Equal(15, XVal.Rows);
        Assert.Equal(15, yVal.Length);
        Assert.Equal(15, XTest.Rows);
        Assert.Equal(15, yTest.Length);
    }

    [Fact]
    public void SplitData_WithCustomSplits_SplitsCorrectly()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var options = new DataProcessorOptions
        {
            TrainingSplitPercentage = 0.6,
            ValidationSplitPercentage = 0.2,
            TestingSplitPercentage = 0.2
        };

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options);

        // Create dataset with 100 samples
        var X = new Matrix<double>(100, 3);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            X[i, 0] = i;
            y[i] = i * 10;
        }

        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(X, y);

        Assert.Equal(60, XTrain.Rows);
        Assert.Equal(20, XVal.Rows);
        Assert.Equal(20, XTest.Rows);
    }

    [Fact]
    public void SplitData_PreservesColumnCount()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        var X = new Matrix<double>(50, 5); // 5 features
        var y = new Vector<double>(50);

        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                X[i, j] = i + j;
            }
            y[i] = i;
        }

        var (XTrain, _, XVal, _, XTest, _) = preprocessor.SplitData(X, y);

        // All splits should have same number of columns (features)
        Assert.Equal(5, XTrain.Columns);
        Assert.Equal(5, XVal.Columns);
        Assert.Equal(5, XTest.Columns);
    }

    [Fact]
    public void SplitData_WithNoShuffling_PreservesOrder()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var options = new DataProcessorOptions
        {
            ShuffleBeforeSplit = false,
            TrainingSplitPercentage = 0.5,
            ValidationSplitPercentage = 0.25,
            TestingSplitPercentage = 0.25
        };

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options);

        var X = new Matrix<double>(20, 1);
        var y = new Vector<double>(20);

        for (int i = 0; i < 20; i++)
        {
            X[i, 0] = i;
            y[i] = i * 10;
        }

        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(X, y);

        // Without shuffling, training should contain first 10 samples (0-9)
        Assert.Equal(0.0, XTrain[0, 0]);
        Assert.Equal(0.0, yTrain[0]);

        // Validation should start from index 10
        Assert.Equal(10.0, XVal[0, 0]);
        Assert.Equal(100.0, yVal[0]);

        // Test should start from index 15
        Assert.Equal(15.0, XTest[0, 0]);
        Assert.Equal(150.0, yTest[0]);
    }

    [Fact]
    public void SplitData_WithShuffling_ReproducibleWithSameSeed()
    {
        var options1 = new DataProcessorOptions
        {
            ShuffleBeforeSplit = true,
            RandomSeed = 42
        };
        var options2 = new DataProcessorOptions
        {
            ShuffleBeforeSplit = true,
            RandomSeed = 42
        };

        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor1 = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options1);
        var preprocessor2 = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options2);

        var X = new Matrix<double>(50, 2);
        var y = new Vector<double>(50);

        for (int i = 0; i < 50; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;
            y[i] = i;
        }

        var (XTrain1, yTrain1, _, _, _, _) = preprocessor1.SplitData(X, y);
        var (XTrain2, yTrain2, _, _, _, _) = preprocessor2.SplitData(X, y);

        // Same seed should produce same shuffle order
        for (int i = 0; i < XTrain1.Rows; i++)
        {
            Assert.Equal(XTrain1[i, 0], XTrain2[i, 0]);
            Assert.Equal(yTrain1[i], yTrain2[i]);
        }
    }

    [Fact]
    public void SplitData_WithDifferentSeeds_ProducesDifferentResults()
    {
        var options1 = new DataProcessorOptions
        {
            ShuffleBeforeSplit = true,
            RandomSeed = 42
        };
        var options2 = new DataProcessorOptions
        {
            ShuffleBeforeSplit = true,
            RandomSeed = 123
        };

        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor1 = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options1);
        var preprocessor2 = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval, options2);

        var X = new Matrix<double>(50, 2);
        var y = new Vector<double>(50);

        for (int i = 0; i < 50; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;
            y[i] = i;
        }

        var (XTrain1, _, _, _, _, _) = preprocessor1.SplitData(X, y);
        var (XTrain2, _, _, _, _, _) = preprocessor2.SplitData(X, y);

        // Different seeds should produce different shuffle order (at least some differences)
        bool anyDifference = false;
        for (int i = 0; i < XTrain1.Rows && !anyDifference; i++)
        {
            if (XTrain1[i, 0] != XTrain2[i, 0])
            {
                anyDifference = true;
            }
        }
        Assert.True(anyDifference, "Different seeds should produce different shuffle orders");
    }

    [Fact]
    public void SplitData_TotalSamplesPreserved()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        int totalSamples = 73; // Odd number to test rounding
        var X = new Matrix<double>(totalSamples, 2);
        var y = new Vector<double>(totalSamples);

        for (int i = 0; i < totalSamples; i++)
        {
            X[i, 0] = i;
            y[i] = i;
        }

        var (XTrain, _, XVal, _, XTest, _) = preprocessor.SplitData(X, y);

        int totalAfterSplit = XTrain.Rows + XVal.Rows + XTest.Rows;
        Assert.Equal(totalSamples, totalAfterSplit);
    }

    #endregion

    #region SplitData Tests - Tensor

    [Fact]
    public void SplitData_WithTensor_SplitsCorrectly()
    {
        var normalizer = new NoNormalizer<double, Tensor<double>, Tensor<double>>();
        var featureSelector = new NoFeatureSelector<double, Tensor<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Tensor<double>, Tensor<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Tensor<double>, Tensor<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Create 2D tensor (100 samples, 3 features)
        var X = new Tensor<double>(new int[] { 100, 3 });
        var y = new Tensor<double>(new int[] { 100, 1 });

        for (int i = 0; i < 100; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;
            X[i, 2] = i * 3;
            y[i, 0] = i * 10;
        }

        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(X, y);

        // Default: 70% train, 15% val, 15% test
        Assert.Equal(70, XTrain.Shape[0]);
        Assert.Equal(70, yTrain.Shape[0]);
        Assert.Equal(15, XVal.Shape[0]);
        Assert.Equal(15, yVal.Shape[0]);
        Assert.Equal(15, XTest.Shape[0]);
        Assert.Equal(15, yTest.Shape[0]);
    }

    [Fact]
    public void SplitData_WithTensor_PreservesFeatureDimensions()
    {
        var normalizer = new NoNormalizer<double, Tensor<double>, Tensor<double>>();
        var featureSelector = new NoFeatureSelector<double, Tensor<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Tensor<double>, Tensor<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Tensor<double>, Tensor<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Create tensor with 5 features
        var X = new Tensor<double>(new int[] { 50, 5 });
        var y = new Tensor<double>(new int[] { 50, 2 }); // Multi-output

        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                X[i, j] = i + j;
            }
            y[i, 0] = i;
            y[i, 1] = i * 2;
        }

        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(X, y);

        // Feature dimension should be preserved
        Assert.Equal(5, XTrain.Shape[1]);
        Assert.Equal(5, XVal.Shape[1]);
        Assert.Equal(5, XTest.Shape[1]);
        Assert.Equal(2, yTrain.Shape[1]);
        Assert.Equal(2, yVal.Shape[1]);
        Assert.Equal(2, yTest.Shape[1]);
    }

    [Fact]
    public void SplitData_WithUnsupportedTypes_ThrowsException()
    {
        var normalizer = new NoNormalizer<double, double[], double>();
        var featureSelector = new NoFeatureSelector<double, double[]>();
        var outlierRemoval = new NoOutlierRemoval<double, double[], double>();

        var preprocessor = new DefaultDataPreprocessor<double, double[], double>(
            normalizer, featureSelector, outlierRemoval);

        var X = new double[] { 1.0, 2.0, 3.0 };
        var y = 4.0;

        Assert.Throws<InvalidOperationException>(() => preprocessor.SplitData(X, y));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SplitData_SmallDataset_HandledCorrectly()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Very small dataset (10 samples)
        var X = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;
            y[i] = i;
        }

        var (XTrain, _, XVal, _, XTest, _) = preprocessor.SplitData(X, y);

        // Should not crash, and totals should add up
        Assert.Equal(10, XTrain.Rows + XVal.Rows + XTest.Rows);
    }

    [Fact]
    public void PreprocessData_EmptyMatrix_HandledGracefully()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        var X = new Matrix<double>(0, 3); // 0 rows
        var y = new Vector<double>(0);

        var (processedX, processedY, normInfo) = preprocessor.PreprocessData(X, y);

        Assert.Equal(0, processedX.Rows);
        Assert.Equal(0, processedY.Length);
    }

    [Fact]
    public void SplitData_SingleFeature_Works()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Single feature
        var X = new Matrix<double>(100, 1);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            X[i, 0] = i;
            y[i] = i * 2;
        }

        var (XTrain, _, XVal, _, XTest, _) = preprocessor.SplitData(X, y);

        Assert.Equal(1, XTrain.Columns);
        Assert.Equal(1, XVal.Columns);
        Assert.Equal(1, XTest.Columns);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void FullPipeline_PreprocessAndSplit_WorksTogether()
    {
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

        var preprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
            normalizer, featureSelector, outlierRemoval);

        // Create realistic dataset
        var X = new Matrix<double>(100, 4);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            X[i, 0] = i * 1.5;
            X[i, 1] = i * 2.3;
            X[i, 2] = i * 0.8;
            X[i, 3] = i * 4.2;
            y[i] = i * 3.14;
        }

        // Preprocess
        var (processedX, processedY, normInfo) = preprocessor.PreprocessData(X, y);

        // Then split
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(processedX, processedY);

        // Verify pipeline worked
        Assert.NotNull(normInfo);
        Assert.Equal(4, XTrain.Columns);
        Assert.Equal(100, XTrain.Rows + XVal.Rows + XTest.Rows);
    }

    [Fact]
    public void SplitData_WithFloat_WorksCorrectly()
    {
        var normalizer = new NoNormalizer<float, Matrix<float>, Vector<float>>();
        var featureSelector = new NoFeatureSelector<float, Matrix<float>>();
        var outlierRemoval = new NoOutlierRemoval<float, Matrix<float>, Vector<float>>();

        var preprocessor = new DefaultDataPreprocessor<float, Matrix<float>, Vector<float>>(
            normalizer, featureSelector, outlierRemoval);

        var X = new Matrix<float>(50, 2);
        var y = new Vector<float>(50);

        for (int i = 0; i < 50; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;
            y[i] = i * 10;
        }

        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = preprocessor.SplitData(X, y);

        Assert.Equal(35, XTrain.Rows); // 70% of 50
        Assert.Equal(7, XVal.Rows);    // 15% of 50 (rounded)
        Assert.Equal(8, XTest.Rows);   // Remaining
    }

    #endregion
}
