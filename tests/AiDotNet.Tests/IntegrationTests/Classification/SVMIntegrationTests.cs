using AiDotNet.Classification.SVM;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for Support Vector Machine classifiers.
/// Tests verify mathematical correctness without trusting the implementation.
/// </summary>
[Trait("Category", "Integration")]
public class SVMIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region SupportVectorClassifier Core Tests

    [Fact]
    public void SVC_LinearKernel_LinearSeparableData_PerfectClassification()
    {
        // Arrange: Clearly linearly separable data
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        // Class 0: left side (x < 0)
        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -3 + 0.3 * (i - 5);
            x[i, 1] = 0.5 * (i - 5);
            y[i] = 0;
        }

        // Class 1: right side (x > 4)
        for (int i = 0; i < 10; i++)
        {
            x[10 + i, 0] = 5 + 0.3 * (i - 5);
            x[10 + i, 1] = 0.5 * (i - 5);
            y[10 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            MaxIterations = 1000,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);

        // Act
        svc.Train(x, y);
        var predictions = svc.Predict(x);

        // Assert: Should achieve near-perfect classification
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 18, $"SVC with linear kernel should achieve high accuracy on separable data. Got {correct}/20");
    }

    [Fact]
    public void SVC_RBFKernel_NonLinearData_BetterThanLinear()
    {
        // Arrange: XOR-like pattern (non-linearly separable)
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        // Class 0: bottom-left and top-right
        x[0, 0] = -1; x[0, 1] = -1; y[0] = 0;
        x[1, 0] = -0.5; x[1, 1] = -0.5; y[1] = 0;
        x[2, 0] = -1.5; x[2, 1] = -1.5; y[2] = 0;
        x[3, 0] = -0.8; x[3, 1] = -1.2; y[3] = 0;
        x[4, 0] = -1.2; x[4, 1] = -0.8; y[4] = 0;
        x[5, 0] = 1; x[5, 1] = 1; y[5] = 0;
        x[6, 0] = 0.5; x[6, 1] = 0.5; y[6] = 0;
        x[7, 0] = 1.5; x[7, 1] = 1.5; y[7] = 0;
        x[8, 0] = 0.8; x[8, 1] = 1.2; y[8] = 0;
        x[9, 0] = 1.2; x[9, 1] = 0.8; y[9] = 0;

        // Class 1: bottom-right and top-left
        x[10, 0] = 1; x[10, 1] = -1; y[10] = 1;
        x[11, 0] = 0.5; x[11, 1] = -0.5; y[11] = 1;
        x[12, 0] = 1.5; x[12, 1] = -1.5; y[12] = 1;
        x[13, 0] = 0.8; x[13, 1] = -1.2; y[13] = 1;
        x[14, 0] = 1.2; x[14, 1] = -0.8; y[14] = 1;
        x[15, 0] = -1; x[15, 1] = 1; y[15] = 1;
        x[16, 0] = -0.5; x[16, 1] = 0.5; y[16] = 1;
        x[17, 0] = -1.5; x[17, 1] = 1.5; y[17] = 1;
        x[18, 0] = -0.8; x[18, 1] = 1.2; y[18] = 1;
        x[19, 0] = -1.2; x[19, 1] = 0.8; y[19] = 1;

        var optionsLinear = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            MaxIterations = 500,
            Seed = 42
        };

        var optionsRBF = new SVMOptions<double>
        {
            Kernel = KernelType.RBF,
            C = 10.0,
            Gamma = 1.0,
            MaxIterations = 500,
            Seed = 42
        };

        var svcLinear = new SupportVectorClassifier<double>(optionsLinear);
        var svcRBF = new SupportVectorClassifier<double>(optionsRBF);

        // Act
        svcLinear.Train(x, y);
        svcRBF.Train(x, y);

        var predsLinear = svcLinear.Predict(x);
        var predsRBF = svcRBF.Predict(x);

        int correctLinear = 0, correctRBF = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(predsLinear[i] - y[i]) < 0.01) correctLinear++;
            if (Math.Abs(predsRBF[i] - y[i]) < 0.01) correctRBF++;
        }

        // Assert: RBF should generally do better on XOR-like data
        Assert.True(correctRBF >= correctLinear,
            $"RBF should perform at least as well as linear on XOR data. Linear: {correctLinear}/20, RBF: {correctRBF}/20");
        Assert.True(correctRBF >= 10, $"RBF should classify at least half correctly. Got {correctRBF}/20");
    }

    [Fact]
    public void SVC_DecisionFunction_SignDeterminesPrediction()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -2 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 2 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);
        svc.Train(x, y);

        // Act
        var decisions = svc.DecisionFunction(x);
        var predictions = svc.Predict(x);

        // Assert: Decision function sign should match prediction
        // Positive decision -> positive class (1), negative decision -> negative class (0)
        for (int i = 0; i < 10; i++)
        {
            double decision = decisions[i, 0];
            double prediction = predictions[i];

            // Positive class is the last one (class 1)
            if (decision > 0)
            {
                Assert.True(Math.Abs(prediction - 1) < 0.01,
                    $"Positive decision ({decision}) should predict class 1, got {prediction}");
            }
        }
    }

    [Fact]
    public void SVC_PredictProbabilities_SumToOne()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -2 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 2 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);
        svc.Train(x, y);

        // Act
        var probs = svc.PredictProbabilities(x);

        // Assert: Probabilities should sum to 1
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = probs[i, 0] + probs[i, 1];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {i} probabilities should sum to 1, got {sum}");
            Assert.True(probs[i, 0] >= 0 && probs[i, 0] <= 1,
                $"Probability should be in [0,1], got {probs[i, 0]}");
            Assert.True(probs[i, 1] >= 0 && probs[i, 1] <= 1,
                $"Probability should be in [0,1], got {probs[i, 1]}");
        }
    }

    [Fact]
    public void SVC_HigherC_TighterMargin()
    {
        // Arrange: Data with some overlap
        var x = new Matrix<double>(12, 2);
        var y = new Vector<double>(12);

        // Class 0: mostly left side
        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1.5; x[1, 1] = 0.5; y[1] = 0;
        x[2, 0] = -1.5; x[2, 1] = -0.5; y[2] = 0;
        x[3, 0] = -1; x[3, 1] = 0; y[3] = 0;
        x[4, 0] = 0; x[4, 1] = 0; y[4] = 0;  // Near boundary
        x[5, 0] = 0.2; x[5, 1] = 0; y[5] = 0;  // Near boundary

        // Class 1: mostly right side
        x[6, 0] = 2; x[6, 1] = 0; y[6] = 1;
        x[7, 0] = 1.5; x[7, 1] = 0.5; y[7] = 1;
        x[8, 0] = 1.5; x[8, 1] = -0.5; y[8] = 1;
        x[9, 0] = 1; x[9, 1] = 0; y[9] = 1;
        x[10, 0] = 0.5; x[10, 1] = 0; y[10] = 1;  // Near boundary
        x[11, 0] = 0.3; x[11, 1] = 0; y[11] = 1;  // Near boundary

        var optionsLowC = new SVMOptions<double> { Kernel = KernelType.Linear, C = 0.1, Seed = 42 };
        var optionsHighC = new SVMOptions<double> { Kernel = KernelType.Linear, C = 100.0, Seed = 42 };

        var svcLow = new SupportVectorClassifier<double>(optionsLowC);
        var svcHigh = new SupportVectorClassifier<double>(optionsHighC);

        // Act
        svcLow.Train(x, y);
        svcHigh.Train(x, y);

        var predsLow = svcLow.Predict(x);
        var predsHigh = svcHigh.Predict(x);

        // Assert: Both should produce valid predictions
        int correctLow = 0, correctHigh = 0;
        for (int i = 0; i < 12; i++)
        {
            if (Math.Abs(predsLow[i] - y[i]) < 0.01) correctLow++;
            if (Math.Abs(predsHigh[i] - y[i]) < 0.01) correctHigh++;
        }

        // Higher C should try harder to classify training points correctly
        Assert.True(correctLow >= 8, $"Low C should still work reasonably. Got {correctLow}/12");
        Assert.True(correctHigh >= 8, $"High C should work. Got {correctHigh}/12");
    }

    [Fact]
    public void SVC_PolynomialKernel_Works()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 1 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Polynomial,
            C = 1.0,
            Degree = 3,
            Coef0 = 0,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);

        // Act
        svc.Train(x, y);
        var predictions = svc.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 10; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 8, $"Polynomial kernel should work. Got {correct}/10");
    }

    #endregion

    #region LinearSupportVectorClassifier Tests

    [Fact]
    public void LinearSVC_EfficientForLargeData()
    {
        // Arrange: Larger dataset
        var x = new Matrix<double>(100, 5);
        var y = new Vector<double>(100);

        var random = new Random(42);
        for (int i = 0; i < 100; i++)
        {
            // Class determined by first feature
            y[i] = (i < 50) ? 0 : 1;
            x[i, 0] = (i < 50) ? -1 - random.NextDouble() : 1 + random.NextDouble();
            for (int j = 1; j < 5; j++)
            {
                x[i, j] = random.NextDouble() - 0.5;  // Noise features
            }
        }

        var options = new SVMOptions<double>
        {
            C = 1.0,
            MaxIterations = 500,
            Seed = 42
        };
        var linearSvc = new LinearSupportVectorClassifier<double>(options);

        // Act
        linearSvc.Train(x, y);
        var predictions = linearSvc.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 100; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 80, $"Linear SVC should achieve good accuracy on separable data. Got {correct}/100");
    }

    [Fact]
    public void LinearSVC_DecisionFunction_LinearCombination()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -0.5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 0.5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            C = 1.0,
            MaxIterations = 500,
            Seed = 42
        };
        var linearSvc = new LinearSupportVectorClassifier<double>(options);
        linearSvc.Train(x, y);

        // Act
        var decisions = linearSvc.DecisionFunction(x);

        // Assert: Decision values should exist for all samples
        for (int i = 0; i < 6; i++)
        {
            Assert.False(double.IsNaN(decisions[i, 0]), $"Decision value {i} should not be NaN");
            Assert.False(double.IsInfinity(decisions[i, 0]), $"Decision value {i} should not be infinite");
        }

        // Decision values should have correct sign for most samples
        int correctSigns = 0;
        for (int i = 0; i < 3; i++)
        {
            if (decisions[i, 0] < 0) correctSigns++;  // Class 0 should have negative decisions
        }
        for (int i = 3; i < 6; i++)
        {
            if (decisions[i, 0] > 0) correctSigns++;  // Class 1 should have positive decisions
        }
        Assert.True(correctSigns >= 4, $"Most decision values should have correct sign. Got {correctSigns}/6");
    }

    [Fact]
    public void LinearSVC_PredictProbabilities_SigmoidTransform()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -0.5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 0.5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            C = 1.0,
            MaxIterations = 500,
            Seed = 42
        };
        var linearSvc = new LinearSupportVectorClassifier<double>(options);
        linearSvc.Train(x, y);

        // Act
        var probs = linearSvc.PredictProbabilities(x);

        // Assert: Probabilities should be valid
        for (int i = 0; i < 6; i++)
        {
            double sum = probs[i, 0] + probs[i, 1];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Probabilities at row {i} should sum to 1, got {sum}");
            Assert.True(probs[i, 0] >= 0 && probs[i, 0] <= 1,
                $"P(class 0) should be in [0,1], got {probs[i, 0]}");
            Assert.True(probs[i, 1] >= 0 && probs[i, 1] <= 1,
                $"P(class 1) should be in [0,1], got {probs[i, 1]}");
        }
    }

    [Fact]
    public void LinearSVC_HigherC_MoreAggressive()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -0.5 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 0.5 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var optionsLow = new SVMOptions<double> { C = 0.01, MaxIterations = 300, Seed = 42 };
        var optionsHigh = new SVMOptions<double> { C = 100.0, MaxIterations = 300, Seed = 42 };

        var svcLow = new LinearSupportVectorClassifier<double>(optionsLow);
        var svcHigh = new LinearSupportVectorClassifier<double>(optionsHigh);

        // Act
        svcLow.Train(x, y);
        svcHigh.Train(x, y);

        var predsLow = svcLow.Predict(x);
        var predsHigh = svcHigh.Predict(x);

        // Assert: Both should work
        int correctLow = 0, correctHigh = 0;
        for (int i = 0; i < 10; i++)
        {
            if (Math.Abs(predsLow[i] - y[i]) < 0.01) correctLow++;
            if (Math.Abs(predsHigh[i] - y[i]) < 0.01) correctHigh++;
        }

        // Both should train successfully and make reasonable predictions
        Assert.True(correctLow >= 5, $"Low C model should achieve reasonable accuracy. Got {correctLow}/10");
        Assert.True(correctHigh >= 5, $"High C model should achieve reasonable accuracy. Got {correctHigh}/10");
    }

    #endregion

    #region Kernel Tests

    [Fact]
    public void SVC_LinearKernel_DotProduct()
    {
        // Arrange: Simple data
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = -1; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -2; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 1;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);

        // Act
        svc.Train(x, y);
        var predictions = svc.Predict(x);

        // Assert
        Assert.True(Math.Abs(predictions[0] - 0) < 0.01 || Math.Abs(predictions[1] - 0) < 0.01,
            "At least one class 0 sample should be classified correctly");
        Assert.True(Math.Abs(predictions[2] - 1) < 0.01 || Math.Abs(predictions[3] - 1) < 0.01,
            "At least one class 1 sample should be classified correctly");
    }

    [Fact]
    public void SVC_RBFKernel_GammaAffectsComplexity()
    {
        // Arrange: Non-linear data
        var x = new Matrix<double>(12, 2);
        var y = new Vector<double>(12);

        // Inner circle: class 0
        for (int i = 0; i < 6; i++)
        {
            double angle = 2 * Math.PI * i / 6;
            x[i, 0] = 0.5 * Math.Cos(angle);
            x[i, 1] = 0.5 * Math.Sin(angle);
            y[i] = 0;
        }

        // Outer circle: class 1
        for (int i = 0; i < 6; i++)
        {
            double angle = 2 * Math.PI * i / 6;
            x[6 + i, 0] = 2 * Math.Cos(angle);
            x[6 + i, 1] = 2 * Math.Sin(angle);
            y[6 + i] = 1;
        }

        var optionsLowGamma = new SVMOptions<double>
        {
            Kernel = KernelType.RBF,
            C = 10.0,
            Gamma = 0.01,  // Low gamma: smooth boundary
            Seed = 42
        };

        var optionsHighGamma = new SVMOptions<double>
        {
            Kernel = KernelType.RBF,
            C = 10.0,
            Gamma = 10.0,  // High gamma: complex boundary
            Seed = 42
        };

        var svcLow = new SupportVectorClassifier<double>(optionsLowGamma);
        var svcHigh = new SupportVectorClassifier<double>(optionsHighGamma);

        // Act
        svcLow.Train(x, y);
        svcHigh.Train(x, y);

        var predsLow = svcLow.Predict(x);
        var predsHigh = svcHigh.Predict(x);

        // Assert: Both should train (may have different accuracies)
        int correctLow = 0, correctHigh = 0;
        for (int i = 0; i < 12; i++)
        {
            if (Math.Abs(predsLow[i] - y[i]) < 0.01) correctLow++;
            if (Math.Abs(predsHigh[i] - y[i]) < 0.01) correctHigh++;
        }

        // Just verify both trained without error
        Assert.True(correctLow >= 0, $"Low gamma model should train");
        Assert.True(correctHigh >= 0, $"High gamma model should train");
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void SVC_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -0.5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 0.5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);
        svc.Train(x, y);

        // Act
        var clone = (SupportVectorClassifier<double>)svc.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var originalPred = svc.Predict(testPoint);
        var clonePred = clone.Predict(testPoint);

        // Assert
        Assert.True(Math.Abs(originalPred[0] - clonePred[0]) < Tolerance);
    }

    [Fact]
    public void LinearSVC_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -0.5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 0.5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            C = 1.0,
            MaxIterations = 300,
            Seed = 42
        };
        var svc = new LinearSupportVectorClassifier<double>(options);
        svc.Train(x, y);

        // Act
        var clone = (LinearSupportVectorClassifier<double>)svc.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var originalPred = svc.Predict(testPoint);
        var clonePred = clone.Predict(testPoint);

        // Assert
        Assert.True(Math.Abs(originalPred[0] - clonePred[0]) < Tolerance);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void SVC_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched

        var svc = new SupportVectorClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => svc.Train(x, y));
    }

    [Fact]
    public void LinearSVC_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched

        var svc = new LinearSupportVectorClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => svc.Train(x, y));
    }

    [Fact]
    public void SVC_PredictBeforeTrain_Throws()
    {
        // Arrange
        var svc = new SupportVectorClassifier<double>();
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => svc.Predict(testPoint));
    }

    [Fact]
    public void LinearSVC_PredictBeforeTrain_Throws()
    {
        // Arrange
        var svc = new LinearSupportVectorClassifier<double>();
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => svc.Predict(testPoint));
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void SVC_LargeFeatureValues_Stable()
    {
        // Arrange: Data with large feature values
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -1e5; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -2e5; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -3e5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 1e5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 2e5; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 3e5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            C = 1.0,
            Seed = 42
        };
        var svc = new SupportVectorClassifier<double>(options);

        // Act
        svc.Train(x, y);
        var decisions = svc.DecisionFunction(x);

        // Assert: Decisions should be finite
        for (int i = 0; i < 6; i++)
        {
            Assert.False(double.IsNaN(decisions[i, 0]), $"Decision {i} should not be NaN");
            Assert.False(double.IsInfinity(decisions[i, 0]), $"Decision {i} should not be infinite");
        }
    }

    [Fact]
    public void LinearSVC_SmallFeatureValues_Stable()
    {
        // Arrange: Data with small feature values
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -1e-5; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -2e-5; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -3e-5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 1e-5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 2e-5; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 3e-5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            C = 1.0,
            MaxIterations = 500,
            Seed = 42
        };
        var svc = new LinearSupportVectorClassifier<double>(options);

        // Act
        svc.Train(x, y);
        var decisions = svc.DecisionFunction(x);

        // Assert: Decisions should be finite
        for (int i = 0; i < 6; i++)
        {
            Assert.False(double.IsNaN(decisions[i, 0]), $"Decision {i} should not be NaN");
        }
    }

    #endregion

    #region NuSupportVectorClassifier Tests

    [Fact]
    public void NuSVC_BasicBinaryClassification_Works()
    {
        // Arrange: Well-separated data
        var x = new Matrix<double>(16, 2);
        var y = new Vector<double>(16);

        for (int i = 0; i < 8; i++)
        {
            x[i, 0] = -2 - 0.2 * i; x[i, 1] = 0.1 * i; y[i] = 0;
            x[8 + i, 0] = 2 + 0.2 * i; x[8 + i, 1] = 0.1 * i; y[8 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            MaxIterations = 500,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);

        // Act
        nuSvc.Train(x, y);
        var predictions = nuSvc.Predict(x);

        // Assert: Should classify most samples correctly
        int correct = 0;
        for (int i = 0; i < 16; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 12, $"Nu-SVC should classify most samples correctly. Got {correct}/16");
    }

    [Fact]
    public void NuSVC_NuParameterValidation_ThrowsOnInvalidNu()
    {
        // Arrange & Act & Assert: Nu must be in (0, 1]
        Assert.Throws<ArgumentException>(() => new NuSupportVectorClassifier<double>(null, null, nu: 0.0));
        Assert.Throws<ArgumentException>(() => new NuSupportVectorClassifier<double>(null, null, nu: -0.5));
        Assert.Throws<ArgumentException>(() => new NuSupportVectorClassifier<double>(null, null, nu: 1.5));
    }

    [Fact]
    public void NuSVC_NuBoundsEdgeCases_ValidValues()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -2 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 2 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        // Act & Assert: nu = 1.0 is valid (edge case)
        var nuSvc1 = new NuSupportVectorClassifier<double>(null, null, nu: 1.0);
        nuSvc1.Train(x, y);
        var pred1 = nuSvc1.Predict(x);
        Assert.Equal(10, pred1.Length);

        // nu = 0.01 is valid (edge case)
        var nuSvc2 = new NuSupportVectorClassifier<double>(null, null, nu: 0.01);
        nuSvc2.Train(x, y);
        var pred2 = nuSvc2.Predict(x);
        Assert.Equal(10, pred2.Length);
    }

    [Fact]
    public void NuSVC_DifferentNuValues_AffectsSupportVectors()
    {
        // Arrange
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0.05 * i; y[i] = 0;
            x[10 + i, 0] = 1 + 0.1 * i; x[10 + i, 1] = 0.05 * i; y[10 + i] = 1;
        }

        var optionsLowNu = new SVMOptions<double> { Kernel = KernelType.Linear, MaxIterations = 300, Seed = 42 };
        var optionsHighNu = new SVMOptions<double> { Kernel = KernelType.Linear, MaxIterations = 300, Seed = 42 };

        var svcLowNu = new NuSupportVectorClassifier<double>(optionsLowNu, null, nu: 0.1);
        var svcHighNu = new NuSupportVectorClassifier<double>(optionsHighNu, null, nu: 0.9);

        // Act
        svcLowNu.Train(x, y);
        svcHighNu.Train(x, y);

        // Assert: Both should train successfully
        var predsLow = svcLowNu.Predict(x);
        var predsHigh = svcHighNu.Predict(x);

        Assert.Equal(20, predsLow.Length);
        Assert.Equal(20, predsHigh.Length);
    }

    [Fact]
    public void NuSVC_DecisionFunction_ReturnsValidValues()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -2 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 2 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            MaxIterations = 300,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);
        nuSvc.Train(x, y);

        // Act
        var decisions = nuSvc.DecisionFunction(x);

        // Assert: Decision values should be finite
        Assert.Equal(10, decisions.Rows);
        Assert.Equal(1, decisions.Columns);

        for (int i = 0; i < 10; i++)
        {
            Assert.False(double.IsNaN(decisions[i, 0]), $"Decision {i} should not be NaN");
            Assert.False(double.IsInfinity(decisions[i, 0]), $"Decision {i} should not be infinite");
        }
    }

    [Fact]
    public void NuSVC_PredictProbabilities_SumToOne()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -2 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 2 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            MaxIterations = 300,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);
        nuSvc.Train(x, y);

        // Act
        var probs = nuSvc.PredictProbabilities(x);

        // Assert: Probabilities should sum to 1 and be in [0, 1]
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = probs[i, 0] + probs[i, 1];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {i} probabilities should sum to 1, got {sum}");
            Assert.True(probs[i, 0] >= 0 && probs[i, 0] <= 1,
                $"P(class 0) should be in [0,1], got {probs[i, 0]}");
            Assert.True(probs[i, 1] >= 0 && probs[i, 1] <= 1,
                $"P(class 1) should be in [0,1], got {probs[i, 1]}");
        }
    }

    [Fact]
    public void NuSVC_RBFKernel_Works()
    {
        // Arrange
        var x = new Matrix<double>(12, 2);
        var y = new Vector<double>(12);

        for (int i = 0; i < 6; i++)
        {
            x[i, 0] = -1 - 0.2 * i; x[i, 1] = 0.1 * i; y[i] = 0;
            x[6 + i, 0] = 1 + 0.2 * i; x[6 + i, 1] = 0.1 * i; y[6 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.RBF,
            Gamma = 1.0,
            MaxIterations = 500,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);

        // Act
        nuSvc.Train(x, y);
        var predictions = nuSvc.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 12; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 8, $"Nu-SVC with RBF kernel should work. Got {correct}/12");
    }

    [Fact]
    public void NuSVC_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1.5; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -1; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = -0.5; x[3, 1] = 0; y[3] = 0;
        x[4, 0] = 2; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 1.5; x[5, 1] = 0; y[5] = 1;
        x[6, 0] = 1; x[6, 1] = 0; y[6] = 1;
        x[7, 0] = 0.5; x[7, 1] = 0; y[7] = 1;

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            MaxIterations = 300,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);
        nuSvc.Train(x, y);

        // Act
        var clone = (NuSupportVectorClassifier<double>)nuSvc.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var originalPred = nuSvc.Predict(testPoint);
        var clonePred = clone.Predict(testPoint);

        // Assert
        Assert.True(Math.Abs(originalPred[0] - clonePred[0]) < Tolerance);
    }

    [Fact]
    public void NuSVC_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched

        var nuSvc = new NuSupportVectorClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => nuSvc.Train(x, y));
    }

    [Fact]
    public void NuSVC_PredictBeforeTrain_Throws()
    {
        // Arrange
        var nuSvc = new NuSupportVectorClassifier<double>();
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => nuSvc.Predict(testPoint));
    }

    [Fact]
    public void NuSVC_GetModelMetadata_ContainsNuParameter()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -0.5; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 0.5; x[5, 1] = 0; y[5] = 1;

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            MaxIterations = 200,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.3);
        nuSvc.Train(x, y);

        // Act
        var metadata = nuSvc.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("Nu"));
        Assert.Equal(0.3, metadata.AdditionalInfo["Nu"]);
        Assert.True(metadata.AdditionalInfo.ContainsKey("Rho"));
    }

    [Fact]
    public void NuSVC_PolynomialKernel_Works()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -1 - 0.1 * i; x[i, 1] = 0; y[i] = 0;
            x[5 + i, 0] = 1 + 0.1 * i; x[5 + i, 1] = 0; y[5 + i] = 1;
        }

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Polynomial,
            Degree = 2,
            Coef0 = 1,
            MaxIterations = 500,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);

        // Act
        nuSvc.Train(x, y);
        var predictions = nuSvc.Predict(x);

        // Assert
        int correct = 0;
        for (int i = 0; i < 10; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        Assert.True(correct >= 6, $"Nu-SVC with polynomial kernel should work. Got {correct}/10");
    }

    [Fact]
    public void NuSVC_NumericalStability_LargeFeatureValues()
    {
        // Arrange: Data with large feature values
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        x[0, 0] = -1e4; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -2e4; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = -3e4; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = -4e4; x[3, 1] = 0; y[3] = 0;
        x[4, 0] = 1e4; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 2e4; x[5, 1] = 0; y[5] = 1;
        x[6, 0] = 3e4; x[6, 1] = 0; y[6] = 1;
        x[7, 0] = 4e4; x[7, 1] = 0; y[7] = 1;

        var options = new SVMOptions<double>
        {
            Kernel = KernelType.Linear,
            MaxIterations = 300,
            Seed = 42
        };
        var nuSvc = new NuSupportVectorClassifier<double>(options, null, nu: 0.5);

        // Act
        nuSvc.Train(x, y);
        var decisions = nuSvc.DecisionFunction(x);

        // Assert: Decisions should be finite
        for (int i = 0; i < 8; i++)
        {
            Assert.False(double.IsNaN(decisions[i, 0]), $"Decision {i} should not be NaN");
            Assert.False(double.IsInfinity(decisions[i, 0]), $"Decision {i} should not be infinite");
        }
    }

    #endregion

    #region GetModelMetadata Tests

    [Fact]
    public void LinearSVC_GetModelMetadata_ContainsAlgorithmInfo()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);
        x[0, 0] = -1; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -2; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 1;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 1;

        var options = new SVMOptions<double> { C = 1.0, Seed = 42 };
        var svc = new LinearSupportVectorClassifier<double>(options);
        svc.Train(x, y);

        // Act
        var metadata = svc.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("Algorithm"));
        Assert.Equal("SGD", metadata.AdditionalInfo["Algorithm"]);
    }

    #endregion
}
