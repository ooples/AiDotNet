using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

/// <summary>
/// Unit tests for the ConfusionMatrix class.
/// </summary>
public class ConfusionMatrixTests
{
    #region Binary Classification Tests (Backward Compatibility)

    [Fact]
    public void BinaryConstructor_InitializesCorrectly()
    {
        // Arrange & Act
        var matrix = new ConfusionMatrix<double>(10, 20, 5, 3);

        // Assert
        Assert.Equal(10.0, matrix.TruePositives);
        Assert.Equal(20.0, matrix.TrueNegatives);
        Assert.Equal(5.0, matrix.FalsePositives);
        Assert.Equal(3.0, matrix.FalseNegatives);
        Assert.Equal(2, matrix.ClassCount);
    }

    [Fact]
    public void BinaryConstructor_AccuracyCalculation()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(10, 20, 5, 3);

        // Act
        double accuracy = matrix.Accuracy;

        // Assert
        // Accuracy = (TP + TN) / (TP + TN + FP + FN) = (10 + 20) / (10 + 20 + 5 + 3) = 30/38
        Assert.Equal(30.0 / 38.0, accuracy, precision: 10);
    }

    [Fact]
    public void BinaryConstructor_PrecisionCalculation()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(10, 20, 5, 3);

        // Act
        double precision = matrix.Precision;

        // Assert
        // Precision = TP / (TP + FP) = 10 / (10 + 5) = 10/15
        Assert.Equal(10.0 / 15.0, precision, precision: 10);
    }

    [Fact]
    public void BinaryConstructor_RecallCalculation()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(10, 20, 5, 3);

        // Act
        double recall = matrix.Recall;

        // Assert
        // Recall = TP / (TP + FN) = 10 / (10 + 3) = 10/13
        Assert.Equal(10.0 / 13.0, recall, precision: 10);
    }

    [Fact]
    public void BinaryConstructor_F1ScoreCalculation()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(10, 20, 5, 3);

        // Act
        double f1 = matrix.F1Score;

        // Assert
        // Precision = 10/15, Recall = 10/13
        // F1 = 2 * (Precision * Recall) / (Precision + Recall)
        double precision = 10.0 / 15.0;
        double recall = 10.0 / 13.0;
        double expectedF1 = 2 * (precision * recall) / (precision + recall);
        Assert.Equal(expectedF1, f1, precision: 10);
    }

    [Fact]
    public void BinaryConstructor_SpecificityCalculation()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(10, 20, 5, 3);

        // Act
        double specificity = matrix.Specificity;

        // Assert
        // Specificity = TN / (TN + FP) = 20 / (20 + 5) = 20/25
        Assert.Equal(20.0 / 25.0, specificity, precision: 10);
    }

    #endregion

    #region Multi-Class Constructor Tests

    [Fact]
    public void DimensionConstructor_InitializesWithZeros()
    {
        // Arrange & Act
        var matrix = new ConfusionMatrix<double>(3);

        // Assert
        Assert.Equal(3, matrix.ClassCount);
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(3, matrix.Columns);

        // All elements should be zero
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(0.0, matrix[i, j]);
            }
        }
    }

    [Fact]
    public void DimensionConstructor_ThrowsForInvalidDimension()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(1));
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(0));
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(-1));
    }

    #endregion

    #region Increment Method Tests

    [Fact]
    public void Increment_UpdatesMatrixCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);

        // Act
        matrix.Increment(0, 0);  // Correct prediction: predicted 0, actual 0
        matrix.Increment(1, 1);  // Correct prediction: predicted 1, actual 1
        matrix.Increment(0, 1);  // Wrong prediction: predicted 0, actual 1
        matrix.Increment(2, 1);  // Wrong prediction: predicted 2, actual 1

        // Assert
        Assert.Equal(1.0, matrix[0, 0]);  // True positive for class 0
        Assert.Equal(1.0, matrix[1, 1]);  // True positive for class 1
        Assert.Equal(1.0, matrix[0, 1]);  // Misclassification
        Assert.Equal(1.0, matrix[2, 1]);  // Misclassification
    }

    [Fact]
    public void Increment_ThrowsForInvalidPredictedClass()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.Increment(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.Increment(3, 0));
    }

    [Fact]
    public void Increment_ThrowsForInvalidActualClass()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.Increment(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.Increment(0, 3));
    }

    #endregion

    #region Per-Class Metrics Tests

    [Fact]
    public void GetTruePositives_ReturnsCorrectValue()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);
        matrix.Increment(0, 0);
        matrix.Increment(1, 0);  // False negative for class 0

        // Act
        double tp0 = matrix.GetTruePositives(0);
        double tp1 = matrix.GetTruePositives(1);

        // Assert
        Assert.Equal(2.0, tp0);
        Assert.Equal(0.0, tp1);
    }

    [Fact]
    public void GetFalsePositives_ReturnsCorrectValue()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(0, 1);  // FP for class 0 (predicted 0, actually 1)
        matrix.Increment(0, 2);  // FP for class 0 (predicted 0, actually 2)
        matrix.Increment(1, 1);  // TP for class 1

        // Act
        double fp0 = matrix.GetFalsePositives(0);
        double fp1 = matrix.GetFalsePositives(1);

        // Assert
        Assert.Equal(2.0, fp0);  // Two false positives for class 0
        Assert.Equal(0.0, fp1);  // No false positives for class 1
    }

    [Fact]
    public void GetFalseNegatives_ReturnsCorrectValue()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(1, 0);  // FN for class 0 (predicted 1, actually 0)
        matrix.Increment(2, 0);  // FN for class 0 (predicted 2, actually 0)
        matrix.Increment(1, 1);  // TP for class 1

        // Act
        double fn0 = matrix.GetFalseNegatives(0);
        double fn1 = matrix.GetFalseNegatives(1);

        // Assert
        Assert.Equal(2.0, fn0);  // Two false negatives for class 0
        Assert.Equal(0.0, fn1);  // No false negatives for class 1
    }

    [Fact]
    public void GetTrueNegatives_ReturnsCorrectValue()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        // Class 0 actual: row 0,1,2 / col 0
        // Class 0 predicted: row 0 / col 0,1,2
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(1, 1);  // TN for class 0 (neither predicted nor actual is 0)
        matrix.Increment(2, 2);  // TN for class 0
        matrix.Increment(1, 2);  // TN for class 0

        // Act
        double tn0 = matrix.GetTrueNegatives(0);

        // Assert
        // TN for class 0: all cells except row 0 and column 0
        // In this case: [1,1], [1,2], [2,1], [2,2] = 1 + 1 + 0 + 1 = 3
        Assert.Equal(3.0, tn0);
    }

    #endregion

    #region Overall Metrics Tests

    [Fact]
    public void GetAccuracy_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);  // Correct
        matrix.Increment(1, 1);  // Correct
        matrix.Increment(2, 2);  // Correct
        matrix.Increment(0, 1);  // Wrong
        matrix.Increment(1, 2);  // Wrong

        // Act
        double accuracy = matrix.GetAccuracy();

        // Assert
        // Accuracy = correct / total = 3 / 5
        Assert.Equal(3.0 / 5.0, accuracy, precision: 10);
    }

    [Fact]
    public void GetAccuracy_ReturnsZeroForEmptyMatrix()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);

        // Act
        double accuracy = matrix.GetAccuracy();

        // Assert
        Assert.Equal(0.0, accuracy);
    }

    #endregion

    #region Per-Class Precision/Recall/F1 Tests

    [Fact]
    public void GetPrecision_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(0, 1);  // FP for class 0
        matrix.Increment(0, 2);  // FP for class 0

        // Act
        double precision0 = matrix.GetPrecision(0);

        // Assert
        // Precision = TP / (TP + FP) = 2 / (2 + 2) = 0.5
        Assert.Equal(0.5, precision0, precision: 10);
    }

    [Fact]
    public void GetRecall_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(0, 0);  // TP for class 0
        matrix.Increment(1, 0);  // FN for class 0
        matrix.Increment(2, 0);  // FN for class 0

        // Act
        double recall0 = matrix.GetRecall(0);

        // Assert
        // Recall = TP / (TP + FN) = 2 / (2 + 2) = 0.5
        Assert.Equal(0.5, recall0, precision: 10);
    }

    [Fact]
    public void GetF1Score_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);  // TP
        matrix.Increment(0, 0);  // TP
        matrix.Increment(0, 1);  // FP
        matrix.Increment(1, 0);  // FN

        // Act
        double f1 = matrix.GetF1Score(0);

        // Assert
        // Precision = 2 / (2 + 1) = 2/3
        // Recall = 2 / (2 + 1) = 2/3
        // F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * 4/9 / 4/3 = 8/9 / 4/3 = 8/9 * 3/4 = 2/3
        Assert.Equal(2.0 / 3.0, f1, precision: 10);
    }

    #endregion

    #region Macro Average Tests

    [Fact]
    public void GetMacroPrecision_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        // Class 0: TP=2, FP=1 → Precision = 2/3
        matrix.Increment(0, 0);
        matrix.Increment(0, 0);
        matrix.Increment(0, 1);
        // Class 1: TP=3, FP=0 → Precision = 3/3 = 1.0
        matrix.Increment(1, 1);
        matrix.Increment(1, 1);
        matrix.Increment(1, 1);

        // Act
        double macroPrecision = matrix.GetMacroPrecision();

        // Assert
        // Macro Precision = (2/3 + 1.0) / 2 = 5/6
        Assert.Equal(5.0 / 6.0, macroPrecision, precision: 10);
    }

    [Fact]
    public void GetMacroRecall_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        // Class 0: TP=2, FN=1 → Recall = 2/3
        matrix.Increment(0, 0);
        matrix.Increment(0, 0);
        matrix.Increment(1, 0);
        // Class 1: TP=3, FN=0 → Recall = 3/3 = 1.0
        matrix.Increment(1, 1);
        matrix.Increment(1, 1);
        matrix.Increment(1, 1);

        // Act
        double macroRecall = matrix.GetMacroRecall();

        // Assert
        // Macro Recall = (2/3 + 1.0) / 2 = 5/6
        Assert.Equal(5.0 / 6.0, macroRecall, precision: 10);
    }

    [Fact]
    public void GetMacroF1Score_CalculatesCorrectly()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        // Class 0: F1 = 2/3
        matrix.Increment(0, 0);
        matrix.Increment(0, 0);
        matrix.Increment(0, 1);
        matrix.Increment(1, 0);
        // Class 1: F1 = 1.0
        matrix.Increment(1, 1);
        matrix.Increment(1, 1);

        // Act
        double macroF1 = matrix.GetMacroF1Score();
        double f1Class0 = matrix.GetF1Score(0);
        double f1Class1 = matrix.GetF1Score(1);
        double expected = (f1Class0 + f1Class1) / 2.0;

        // Assert
        Assert.Equal(expected, macroF1, precision: 10);
    }

    #endregion

    #region Micro Average Tests

    [Fact]
    public void GetMicroPrecision_EqualsAccuracy()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);
        matrix.Increment(1, 1);
        matrix.Increment(2, 2);
        matrix.Increment(0, 1);
        matrix.Increment(1, 2);

        // Act
        double microPrecision = matrix.GetMicroPrecision();
        double accuracy = matrix.GetAccuracy();

        // Assert
        Assert.Equal(accuracy, microPrecision);
    }

    [Fact]
    public void GetMicroRecall_EqualsAccuracy()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);
        matrix.Increment(1, 1);
        matrix.Increment(2, 2);
        matrix.Increment(0, 1);
        matrix.Increment(1, 2);

        // Act
        double microRecall = matrix.GetMicroRecall();
        double accuracy = matrix.GetAccuracy();

        // Assert
        Assert.Equal(accuracy, microRecall);
    }

    [Fact]
    public void GetMicroF1Score_EqualsAccuracy()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(3);
        matrix.Increment(0, 0);
        matrix.Increment(1, 1);
        matrix.Increment(2, 2);
        matrix.Increment(0, 1);
        matrix.Increment(1, 2);

        // Act
        double microF1 = matrix.GetMicroF1Score();
        double accuracy = matrix.GetAccuracy();

        // Assert
        Assert.Equal(accuracy, microF1);
    }

    #endregion

    #region Weighted Average Tests

    [Fact]
    public void GetWeightedPrecision_AccountsForClassImbalance()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        // Class 0: 10 samples, Precision = 8/10 = 0.8
        for (int i = 0; i < 8; i++) matrix.Increment(0, 0);  // TP
        for (int i = 0; i < 2; i++) matrix.Increment(0, 1);  // FP
        // Class 1: 5 samples, Precision = 4/5 = 0.8
        for (int i = 0; i < 4; i++) matrix.Increment(1, 1);  // TP
        for (int i = 0; i < 1; i++) matrix.Increment(1, 0);  // FP

        // Act
        double weightedPrecision = matrix.GetWeightedPrecision();

        // Assert
        // Support class 0: 8 + 1 = 9 (actual instances of class 0)
        // Support class 1: 2 + 4 = 6 (actual instances of class 1)
        // Precision class 0: 8/10 = 0.8
        // Precision class 1: 4/5 = 0.8
        // Weighted = (0.8 * 9 + 0.8 * 6) / (9 + 6) = 12/15 = 0.8
        Assert.Equal(0.8, weightedPrecision, precision: 10);
    }

    [Fact]
    public void GetWeightedRecall_AccountsForClassImbalance()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        // Class 0: 10 actual samples, Recall = 8/10 = 0.8
        for (int i = 0; i < 8; i++) matrix.Increment(0, 0);  // TP
        for (int i = 0; i < 2; i++) matrix.Increment(1, 0);  // FN
        // Class 1: 5 actual samples, Recall = 4/5 = 0.8
        for (int i = 0; i < 4; i++) matrix.Increment(1, 1);  // TP
        for (int i = 0; i < 1; i++) matrix.Increment(0, 1);  // FN

        // Act
        double weightedRecall = matrix.GetWeightedRecall();

        // Assert
        // Support class 0: 10
        // Support class 1: 5
        // Weighted = (0.8 * 10 + 0.8 * 5) / 15 = 12/15 = 0.8
        Assert.Equal(0.8, weightedRecall, precision: 10);
    }

    [Fact]
    public void GetWeightedF1Score_AccountsForClassImbalance()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);

        // Build matrix:
        // Predicted\Actual    0    1
        //        0            8    2   (8 TP, 2 FP for class 0)
        //        1            2    4   (2 FN for class 0, 4 TP for class 1)
        // Class 0: support=10 (8+2), TP=8, FP=2, FN=2
        //   Precision = 8/(8+2) = 8/10 = 0.8
        //   Recall = 8/(8+2) = 8/10 = 0.8
        //   F1 = 0.8
        // Class 1: support=6 (2+4), TP=4, FP=2, FN=2
        //   Precision = 4/(4+2) = 4/6 = 0.667
        //   Recall = 4/(4+2) = 4/6 = 0.667
        //   F1 = 0.667
        // Weighted F1 = (0.8*10 + 0.667*6) / 16 = (8 + 4) / 16 = 0.75

        for (int i = 0; i < 8; i++) matrix.Increment(0, 0);  // TP for class 0
        for (int i = 0; i < 2; i++) matrix.Increment(0, 1);  // FP for class 0, FN for class 1
        for (int i = 0; i < 2; i++) matrix.Increment(1, 0);  // FN for class 0, FP for class 1
        for (int i = 0; i < 4; i++) matrix.Increment(1, 1);  // TP for class 1

        // Act
        double weightedF1 = matrix.GetWeightedF1Score();
        double f1Class0 = matrix.GetF1Score(0);
        double f1Class1 = matrix.GetF1Score(1);

        // Calculate expected weighted F1
        double support0 = 10.0;  // Actual instances of class 0
        double support1 = 6.0;   // Actual instances of class 1
        double expected = (f1Class0 * support0 + f1Class1 * support1) / (support0 + support1);

        // Assert
        Assert.Equal(expected, weightedF1, precision: 10);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Precision_ReturnsZeroWhenNoPositivePredictions()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        matrix.Increment(1, 0);  // Only class 1 predictions

        // Act
        double precision0 = matrix.GetPrecision(0);

        // Assert
        Assert.Equal(0.0, precision0);
    }

    [Fact]
    public void Recall_ReturnsZeroWhenNoActualPositives()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        matrix.Increment(0, 1);  // Only class 1 actual

        // Act
        double recall0 = matrix.GetRecall(0);

        // Assert
        Assert.Equal(0.0, recall0);
    }

    [Fact]
    public void F1Score_ReturnsZeroWhenPrecisionAndRecallAreZero()
    {
        // Arrange
        var matrix = new ConfusionMatrix<double>(2);
        matrix.Increment(1, 1);  // Only class 1 predictions and actuals

        // Act
        double f1_0 = matrix.GetF1Score(0);

        // Assert
        Assert.Equal(0.0, f1_0);
    }

    #endregion

    #region Integration Test

    [Fact]
    public void IntegrationTest_ThreeClassProblem()
    {
        // Arrange - Simulating a digit classifier for digits 0, 1, 2
        var matrix = new ConfusionMatrix<double>(3);

        // Digit 0: 10 samples, 8 correct, 1 classified as 1, 1 classified as 2
        for (int i = 0; i < 8; i++) matrix.Increment(0, 0);
        matrix.Increment(1, 0);
        matrix.Increment(2, 0);

        // Digit 1: 15 samples, 12 correct, 2 classified as 0, 1 classified as 2
        for (int i = 0; i < 12; i++) matrix.Increment(1, 1);
        matrix.Increment(0, 1);
        matrix.Increment(0, 1);
        matrix.Increment(2, 1);

        // Digit 2: 5 samples, 4 correct, 1 classified as 1
        for (int i = 0; i < 4; i++) matrix.Increment(2, 2);
        matrix.Increment(1, 2);

        // Act & Assert
        // Overall accuracy: (8 + 12 + 4) / 30 = 24/30 = 0.8
        Assert.Equal(0.8, matrix.GetAccuracy(), precision: 10);

        // Class 0 metrics
        // TP=8, FP=2, FN=2
        Assert.Equal(8.0, matrix.GetTruePositives(0));
        Assert.Equal(2.0, matrix.GetFalsePositives(0));
        Assert.Equal(2.0, matrix.GetFalseNegatives(0));
        Assert.Equal(0.8, matrix.GetPrecision(0), precision: 10);  // 8/(8+2)
        Assert.Equal(0.8, matrix.GetRecall(0), precision: 10);     // 8/(8+2)

        // Class 1 metrics
        // TP=12, FP=2, FN=3
        Assert.Equal(12.0, matrix.GetTruePositives(1));
        Assert.Equal(2.0, matrix.GetFalsePositives(1));
        Assert.Equal(3.0, matrix.GetFalseNegatives(1));
        Assert.Equal(12.0 / 14.0, matrix.GetPrecision(1), precision: 10);  // 12/(12+2)
        Assert.Equal(0.8, matrix.GetRecall(1), precision: 10);              // 12/(12+3)

        // Class 2 metrics
        // TP=4, FP=2, FN=1
        Assert.Equal(4.0, matrix.GetTruePositives(2));
        Assert.Equal(2.0, matrix.GetFalsePositives(2));
        Assert.Equal(1.0, matrix.GetFalseNegatives(2));
        Assert.Equal(4.0 / 6.0, matrix.GetPrecision(2), precision: 10);  // 4/(4+2)
        Assert.Equal(0.8, matrix.GetRecall(2), precision: 10);            // 4/(4+1)

        // Verify weighted metrics account for class imbalance (10, 15, 5 samples)
        double weightedRecall = matrix.GetWeightedRecall();
        Assert.Equal(0.8, weightedRecall, precision: 10);  // All classes have recall 0.8
    }

    #endregion
}
