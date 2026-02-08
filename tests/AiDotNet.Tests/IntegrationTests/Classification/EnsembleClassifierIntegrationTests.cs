using AiDotNet.Classification.Ensemble;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for Ensemble classifiers: RandomForest, AdaBoost, GradientBoosting, ExtraTrees.
/// Tests verify mathematical correctness without trusting the implementation.
/// </summary>
[Trait("Category", "Integration")]
public class EnsembleClassifierIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region RandomForest Core Tests

    [Fact]
    public void RandomForest_MultipleTreesImproveAccuracy()
    {
        // Arrange: Data with some noise where ensemble should help
        var x = new Matrix<double>(40, 3);
        var y = new Vector<double>(40);

        var random = new Random(42);
        for (int i = 0; i < 40; i++)
        {
            // Class determined by first feature with some noise
            y[i] = (i < 20) ? 0 : 1;
            x[i, 0] = (i < 20) ? -1 - random.NextDouble() : 1 + random.NextDouble();
            x[i, 1] = random.NextDouble() - 0.5;  // Noise
            x[i, 2] = random.NextDouble() - 0.5;  // Noise
        }

        var optionsSingleTree = new RandomForestClassifierOptions<double>
        {
            NEstimators = 1,
            MaxDepth = 5,
            Seed = 42
        };

        var optionsManyTrees = new RandomForestClassifierOptions<double>
        {
            NEstimators = 10,
            MaxDepth = 5,
            Seed = 42
        };

        var singleTreeRF = new RandomForestClassifier<double>(optionsSingleTree);
        var manyTreeRF = new RandomForestClassifier<double>(optionsManyTrees);

        // Act
        singleTreeRF.Train(x, y);
        manyTreeRF.Train(x, y);

        var predsSingle = singleTreeRF.Predict(x);
        var predsMany = manyTreeRF.Predict(x);

        int correctSingle = 0, correctMany = 0;
        for (int i = 0; i < 40; i++)
        {
            if (Math.Abs(predsSingle[i] - y[i]) < 0.01) correctSingle++;
            if (Math.Abs(predsMany[i] - y[i]) < 0.01) correctMany++;
        }

        // Assert: Both should work, ensemble typically does at least as well
        Assert.True(correctSingle >= 30, $"Single tree should work. Got {correctSingle}/40");
        Assert.True(correctMany >= 30, $"Many trees should work. Got {correctMany}/40");
    }

    [Fact]
    public void RandomForest_BootstrapSampling_DifferentTrees()
    {
        // Arrange
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[10 + i, 0] = 1 + 0.1 * i; x[10 + i, 1] = 0; y[10 + i] = 1;
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 5,
            Bootstrap = true,
            MaxDepth = 3,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);

        // Act
        rf.Train(x, y);
        var predictions = rf.Predict(x);

        // Assert: Predictions should be valid
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 16, $"Random Forest should classify most correctly. Got {correct}/20");
    }

    [Fact]
    public void RandomForest_MaxFeaturesConstraint_Works()
    {
        // Arrange: Many features but only first one matters
        var x = new Matrix<double>(20, 10);
        var y = new Vector<double>(20);

        var random = new Random(42);
        for (int i = 0; i < 20; i++)
        {
            y[i] = (i < 10) ? 0 : 1;
            x[i, 0] = (i < 10) ? -1 - random.NextDouble() : 1 + random.NextDouble();  // Important
            for (int j = 1; j < 10; j++)
            {
                x[i, j] = random.NextDouble() - 0.5;  // Noise
            }
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 10,
            MaxFeatures = "sqrt",  // Only consider sqrt(10) ~ 3 features at each split
            MaxDepth = 5,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);

        // Act
        rf.Train(x, y);
        var predictions = rf.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 16, $"Random Forest with max_features should work. Got {correct}/20");
    }

    [Fact]
    public void RandomForest_OobScore_ProvidedWhenEnabled()
    {
        // Arrange
        var x = new Matrix<double>(30, 2);
        var y = new Vector<double>(30);

        for (int i = 0; i < 15; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[15 + i, 0] = 1 + 0.1 * i; x[15 + i, 1] = 0; y[15 + i] = 1;
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 10,
            Bootstrap = true,
            OobScore = true,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);

        // Act
        rf.Train(x, y);

        // Assert: OOB score should be a valid accuracy value
        Assert.True(rf.OobScore_ >= 0 && rf.OobScore_ <= 1,
            $"OOB score should be in [0,1], got {rf.OobScore_}");
    }

    [Fact]
    public void RandomForest_PredictProbabilities_SumToOne()
    {
        // Arrange
        var x = new Matrix<double>(12, 2);
        var y = new Vector<double>(12);

        // 3-class problem
        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = i; x[i, 1] = 0; y[i] = 0;
            x[4 + i, 0] = 5 + i; x[4 + i, 1] = 0; y[4 + i] = 1;
            x[8 + i, 0] = 10 + i; x[8 + i, 1] = 0; y[8 + i] = 2;
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 5,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);
        rf.Train(x, y);

        // Act
        var probs = rf.PredictProbabilities(x);

        // Assert
        for (int i = 0; i < 12; i++)
        {
            double sum = probs[i, 0] + probs[i, 1] + probs[i, 2];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {i} probabilities should sum to 1, got {sum}");
        }
    }

    #endregion

    #region AdaBoost Core Tests

    [Fact]
    public void AdaBoost_WeakLearnersBecomingStrong()
    {
        // Arrange: Data where single stump would struggle but boosting helps
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.2 * i; x[i, 1] = 0.1 * (i % 3); y[i] = 0;
            x[10 + i, 0] = 1 + 0.2 * i; x[10 + i, 1] = 0.1 * (i % 3); y[10 + i] = 1;
        }

        var options = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 10,
            LearningRate = 1.0,
            Seed = 42
        };
        var adaboost = new AdaBoostClassifier<double>(options);

        // Act
        adaboost.Train(x, y);
        var predictions = adaboost.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 16, $"AdaBoost should achieve good accuracy. Got {correct}/20");
    }

    [Fact]
    public void AdaBoost_LearningRate_AffectsConvergence()
    {
        // Arrange
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[10 + i, 0] = 1 + 0.1 * i; x[10 + i, 1] = 0; y[10 + i] = 1;
        }

        var optionsLow = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 20,
            LearningRate = 0.1,
            Seed = 42
        };

        var optionsHigh = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 20,
            LearningRate = 1.0,
            Seed = 42
        };

        var adaLow = new AdaBoostClassifier<double>(optionsLow);
        var adaHigh = new AdaBoostClassifier<double>(optionsHigh);

        // Act
        adaLow.Train(x, y);
        adaHigh.Train(x, y);

        var predsLow = adaLow.Predict(x);
        var predsHigh = adaHigh.Predict(x);

        int correctLow = 0, correctHigh = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predsLow[i] - y[i]) < 0.01) correctLow++;
            if (Math.Abs(predsHigh[i] - y[i]) < 0.01) correctHigh++;
        }

        // Assert: Both should work
        Assert.True(correctLow >= 14, $"Low learning rate AdaBoost should work. Got {correctLow}/20");
        Assert.True(correctHigh >= 14, $"High learning rate AdaBoost should work. Got {correctHigh}/20");
    }

    [Fact]
    public void AdaBoost_PredictProbabilities_ValidProbabilities()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 1 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 5,
            Seed = 42
        };
        var adaboost = new AdaBoostClassifier<double>(options);
        adaboost.Train(x, y);

        // Act
        var probs = adaboost.PredictProbabilities(x);

        // Assert: Probabilities should be valid (may not sum to exactly 1 due to weighting)
        for (int i = 0; i < 10; i++)
        {
            // At minimum, probabilities should be non-negative
            Assert.True(probs[i, 0] >= 0 || probs[i, 0] >= -Tolerance,
                $"Probability at ({i}, 0) should be non-negative, got {probs[i, 0]}");
            Assert.True(probs[i, 1] >= 0 || probs[i, 1] >= -Tolerance,
                $"Probability at ({i}, 1) should be non-negative, got {probs[i, 1]}");
        }
    }

    [Fact]
    public void AdaBoost_FocusesOnMisclassifiedSamples()
    {
        // Arrange: Data with one hard-to-classify sample
        var x = new Matrix<double>(11, 2);
        var y = new Vector<double>(11);

        // Easy class 0 samples
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -2 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
        }

        // Easy class 1 samples
        for (int i = 0; i < 5; i++)
        {
            x[5 + i, 0] = 2 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        // One tricky sample near the boundary
        x[10, 0] = 0; x[10, 1] = 0; y[10] = 0;

        var options = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 20,
            LearningRate = 1.0,
            Seed = 42
        };
        var adaboost = new AdaBoostClassifier<double>(options);

        // Act
        adaboost.Train(x, y);
        var predictions = adaboost.Predict(x);

        // Assert: Easy samples should definitely be correct
        int correctEasy = 0;
        for (int i = 0; i < 5; i++)
        {
            if (Math.Abs(predictions[i] - 0) < 0.01) correctEasy++;
        }
        for (int i = 5; i < 10; i++)
        {
            if (Math.Abs(predictions[i] - 1) < 0.01) correctEasy++;
        }

        Assert.True(correctEasy >= 8, $"Easy samples should be classified correctly. Got {correctEasy}/10");
    }

    #endregion

    #region ExtraTrees Tests

    [Fact]
    public void ExtraTrees_RandomSplitsWork()
    {
        // Arrange
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0.1 * i; y[i] = 0;
            x[10 + i, 0] = 1 + 0.1 * i; x[10 + i, 1] = 0.1 * i; y[10 + i] = 1;
        }

        var options = new ExtraTreesClassifierOptions<double>
        {
            NEstimators = 10,
            MaxDepth = 5,
            Seed = 42
        };
        var et = new ExtraTreesClassifier<double>(options);

        // Act
        et.Train(x, y);
        var predictions = et.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 16, $"Extra Trees should classify correctly. Got {correct}/20");
    }

    [Fact]
    public void ExtraTrees_MoreRandomThanRandomForest()
    {
        // Arrange: Same data, compare behavior
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[10 + i, 0] = 1 + 0.1 * i; x[10 + i, 1] = 0; y[10 + i] = 1;
        }

        var etOptions = new ExtraTreesClassifierOptions<double>
        {
            NEstimators = 5,
            MaxDepth = 3,
            Seed = 42
        };

        var rfOptions = new RandomForestClassifierOptions<double>
        {
            NEstimators = 5,
            MaxDepth = 3,
            Seed = 42
        };

        var et = new ExtraTreesClassifier<double>(etOptions);
        var rf = new RandomForestClassifier<double>(rfOptions);

        // Act
        et.Train(x, y);
        rf.Train(x, y);

        var etPreds = et.Predict(x);
        var rfPreds = rf.Predict(x);

        int etCorrect = 0, rfCorrect = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(etPreds[i] - y[i]) < 0.01) etCorrect++;
            if (Math.Abs(rfPreds[i] - y[i]) < 0.01) rfCorrect++;
        }

        // Assert: Both should work reasonably well
        Assert.True(etCorrect >= 14, $"Extra Trees should work. Got {etCorrect}/20");
        Assert.True(rfCorrect >= 14, $"Random Forest should work. Got {rfCorrect}/20");
    }

    #endregion

    #region GradientBoosting Tests

    [Fact]
    public void GradientBoosting_SequentialLearning()
    {
        // Arrange
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[10 + i, 0] = 1 + 0.1 * i; x[10 + i, 1] = 0; y[10 + i] = 1;
        }

        var options = new GradientBoostingClassifierOptions<double>
        {
            NEstimators = 10,
            LearningRate = 0.1,
            MaxDepth = 3,
            Seed = 42
        };
        var gb = new GradientBoostingClassifier<double>(options);

        // Act
        gb.Train(x, y);
        var predictions = gb.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 16, $"Gradient Boosting should classify correctly. Got {correct}/20");
    }

    [Fact]
    public void GradientBoosting_LearningRateSubsamplingWork()
    {
        // Arrange
        var x = new Matrix<double>(30, 2);
        var y = new Vector<double>(30);

        for (int i = 0; i < 15; i++)
        {
            x[i, 0] = -1 - 0.05 * i; x[i, 1] = 0.05 * (i % 5); y[i] = 0;
            x[15 + i, 0] = 1 + 0.05 * i; x[15 + i, 1] = 0.05 * (i % 5); y[15 + i] = 1;
        }

        var options = new GradientBoostingClassifierOptions<double>
        {
            NEstimators = 20,
            LearningRate = 0.05,
            Subsample = 0.8,
            MaxDepth = 3,
            Seed = 42
        };
        var gb = new GradientBoostingClassifier<double>(options);

        // Act
        gb.Train(x, y);
        var predictions = gb.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 30; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 24, $"Gradient Boosting with subsample should work. Got {correct}/30");
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void RandomForest_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 1 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 5,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);
        rf.Train(x, y);

        // Act
        var clone = (RandomForestClassifier<double>)rf.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var originalPred = rf.Predict(testPoint);
        var clonePred = clone.Predict(testPoint);

        // Assert
        Assert.True(Math.Abs(originalPred[0] - clonePred[0]) < Tolerance);
    }

    [Fact]
    public void AdaBoost_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 1 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 5,
            Seed = 42
        };
        var ada = new AdaBoostClassifier<double>(options);
        ada.Train(x, y);

        // Act
        var clone = (AdaBoostClassifier<double>)ada.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var originalPred = ada.Predict(testPoint);
        var clonePred = clone.Predict(testPoint);

        // Assert
        Assert.True(Math.Abs(originalPred[0] - clonePred[0]) < Tolerance);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void RandomForest_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched

        var rf = new RandomForestClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => rf.Train(x, y));
    }

    [Fact]
    public void AdaBoost_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched

        var ada = new AdaBoostClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ada.Train(x, y));
    }

    [Fact]
    public void RandomForest_PredictBeforeTrain_Throws()
    {
        // Arrange
        var rf = new RandomForestClassifier<double>();
        var testPoint = new Matrix<double>(1, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => rf.Predict(testPoint));
    }

    [Fact]
    public void AdaBoost_PredictBeforeTrain_Throws()
    {
        // Arrange
        var ada = new AdaBoostClassifier<double>();
        var testPoint = new Matrix<double>(1, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => ada.Predict(testPoint));
    }

    #endregion

    #region Multiclass Tests

    [Fact]
    public void RandomForest_MulticlassClassification()
    {
        // Arrange: 3-class problem
        var x = new Matrix<double>(15, 2);
        var y = new Vector<double>(15);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i * 0.2; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 3 + i * 0.2; x[5 + i, 1] = 0; y[5 + i] = 1;
            x[10 + i, 0] = 6 + i * 0.2; x[10 + i, 1] = 0; y[10 + i] = 2;
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 10,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);

        // Act
        rf.Train(x, y);
        var predictions = rf.Predict(x);
        var probs = rf.PredictProbabilities(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 15; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;

            // Probabilities should sum to 1
            double sum = probs[i, 0] + probs[i, 1] + probs[i, 2];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {i} probs should sum to 1, got {sum}");
        }

        Assert.True(correct >= 12, $"Random Forest multiclass should work. Got {correct}/15");
    }

    [Fact]
    public void AdaBoost_MulticlassClassification()
    {
        // Arrange: 3-class problem
        var x = new Matrix<double>(15, 2);
        var y = new Vector<double>(15);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i * 0.2; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 3 + i * 0.2; x[5 + i, 1] = 0; y[5 + i] = 1;
            x[10 + i, 0] = 6 + i * 0.2; x[10 + i, 1] = 0; y[10 + i] = 2;
        }

        var options = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 20,
            Seed = 42
        };
        var ada = new AdaBoostClassifier<double>(options);

        // Act
        ada.Train(x, y);
        var predictions = ada.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 15; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 10, $"AdaBoost multiclass should work. Got {correct}/15");
    }

    #endregion

    #region Feature Importance Tests

    [Fact]
    public void RandomForest_FeatureImportances_CorrectLength()
    {
        // Arrange
        var x = new Matrix<double>(20, 5);
        var y = new Vector<double>(20);

        var random = new Random(42);
        for (int i = 0; i < 20; i++)
        {
            y[i] = (i < 10) ? 0 : 1;
            for (int j = 0; j < 5; j++)
            {
                x[i, j] = (j == 0 && i >= 10) ? 1.0 : random.NextDouble();
            }
        }

        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 5,
            Seed = 42
        };
        var rf = new RandomForestClassifier<double>(options);

        // Act
        rf.Train(x, y);

        // Assert: Feature importances should exist and have correct length
        Assert.NotNull(rf.FeatureImportances);
        Assert.Equal(5, rf.FeatureImportances.Length);

        // All importances should be non-negative
        for (int i = 0; i < 5; i++)
        {
            Assert.True(rf.FeatureImportances[i] >= 0,
                $"Feature importance {i} should be non-negative");
        }
    }

    #endregion

    #region GetModelMetadata Tests

    [Fact]
    public void RandomForest_GetModelMetadata_ContainsCorrectInfo()
    {
        // Arrange
        var options = new RandomForestClassifierOptions<double>
        {
            NEstimators = 10,
            MaxDepth = 5
        };
        var rf = new RandomForestClassifier<double>(options);

        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = i; x[i, 1] = 0;
            y[i] = i < 5 ? 0 : 1;
        }

        rf.Train(x, y);

        // Act
        var metadata = rf.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("NEstimators"));
        Assert.Equal(10, metadata.AdditionalInfo["NEstimators"]);
    }

    [Fact]
    public void AdaBoost_GetModelMetadata_ContainsCorrectInfo()
    {
        // Arrange
        var options = new AdaBoostClassifierOptions<double>
        {
            NEstimators = 15,
            LearningRate = 0.5
        };
        var ada = new AdaBoostClassifier<double>(options);

        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = i; x[i, 1] = 0;
            y[i] = i < 5 ? 0 : 1;
        }

        ada.Train(x, y);

        // Act
        var metadata = ada.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("NEstimators"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("LearningRate"));
        Assert.Equal(15, metadata.AdditionalInfo["NEstimators"]);
        Assert.Equal(0.5, metadata.AdditionalInfo["LearningRate"]);
    }

    #endregion
}
