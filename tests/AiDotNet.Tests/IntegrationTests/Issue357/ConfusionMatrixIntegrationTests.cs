using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the ConfusionMatrix<T> class covering binary and multi-class
/// classification metrics.
/// </summary>
public class ConfusionMatrixIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Binary Classification Basics

    [Fact]
    public void ConfusionMatrix_BinaryConstructor_SetsValuesCorrectly()
    {
        // TP=50, TN=40, FP=10, FN=5
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        Assert.Equal(50.0, cm.TruePositives);
        Assert.Equal(40.0, cm.TrueNegatives);
        Assert.Equal(10.0, cm.FalsePositives);
        Assert.Equal(5.0, cm.FalseNegatives);
    }

    [Fact]
    public void ConfusionMatrix_Accuracy_ComputesCorrectly()
    {
        // TP=50, TN=40, FP=10, FN=5 -> Accuracy = (50+40)/(50+40+10+5) = 90/105
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        double expected = 90.0 / 105.0;
        Assert.Equal(expected, cm.Accuracy, Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_Precision_ComputesCorrectly()
    {
        // Precision = TP / (TP + FP) = 50 / (50 + 10) = 50/60
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        double expected = 50.0 / 60.0;
        Assert.Equal(expected, cm.Precision, Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_Recall_ComputesCorrectly()
    {
        // Recall = TP / (TP + FN) = 50 / (50 + 5) = 50/55
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        double expected = 50.0 / 55.0;
        Assert.Equal(expected, cm.Recall, Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_F1Score_IsHarmonicMeanOfPrecisionAndRecall()
    {
        // F1 = 2 * (Precision * Recall) / (Precision + Recall)
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        double precision = 50.0 / 60.0;
        double recall = 50.0 / 55.0;
        double expected = 2 * (precision * recall) / (precision + recall);

        Assert.Equal(expected, cm.F1Score, Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_Specificity_ComputesCorrectly()
    {
        // Specificity = TN / (TN + FP) = 40 / (40 + 10) = 40/50
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        double expected = 40.0 / 50.0;
        Assert.Equal(expected, cm.Specificity, Tolerance);
    }

    #endregion

    #region Perfect and Worst Case Scenarios

    [Fact]
    public void ConfusionMatrix_PerfectClassifier_HasUnitMetrics()
    {
        // All correct: TP=50, TN=50, FP=0, FN=0
        var cm = new ConfusionMatrix<double>(50, 50, 0, 0);

        Assert.Equal(1.0, cm.Accuracy, Tolerance);
        Assert.Equal(1.0, cm.Precision, Tolerance);
        Assert.Equal(1.0, cm.Recall, Tolerance);
        Assert.Equal(1.0, cm.F1Score, Tolerance);
        Assert.Equal(1.0, cm.Specificity, Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_AllFalsePositives_HasZeroPrecision()
    {
        // TP=0, FP=50 -> Precision = 0
        var cm = new ConfusionMatrix<double>(0, 50, 50, 0);

        Assert.Equal(0.0, cm.Precision, Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_AllFalseNegatives_HasZeroRecall()
    {
        // TP=0, FN=50 -> Recall = 0
        var cm = new ConfusionMatrix<double>(0, 50, 0, 50);

        Assert.Equal(0.0, cm.Recall, Tolerance);
    }

    #endregion

    #region Multi-Class Classification

    [Fact]
    public void ConfusionMatrix_MultiClass_DimensionConstructor()
    {
        var cm = new ConfusionMatrix<double>(3);

        Assert.Equal(3, cm.ClassCount);
        Assert.Equal(3, cm.Rows);
        Assert.Equal(3, cm.Columns);
    }

    [Fact]
    public void ConfusionMatrix_Increment_UpdatesCounts()
    {
        var cm = new ConfusionMatrix<double>(3);

        cm.Increment(0, 0); // Correct prediction for class 0
        cm.Increment(0, 0); // Another correct
        cm.Increment(0, 1); // Predicted 0, actual 1 (false positive for class 0)
        cm.Increment(1, 1); // Correct prediction for class 1

        Assert.Equal(2.0, cm[0, 0]); // TP for class 0
        Assert.Equal(1.0, cm[0, 1]); // FN for class 1 (predicted 0, was 1)
        Assert.Equal(1.0, cm[1, 1]); // TP for class 1
    }

    [Fact]
    public void ConfusionMatrix_MultiClass_GetTruePositives()
    {
        var cm = new ConfusionMatrix<double>(3);
        cm.Increment(0, 0);
        cm.Increment(0, 0);
        cm.Increment(1, 1);
        cm.Increment(2, 2);
        cm.Increment(2, 2);
        cm.Increment(2, 2);

        Assert.Equal(2.0, cm.GetTruePositives(0));
        Assert.Equal(1.0, cm.GetTruePositives(1));
        Assert.Equal(3.0, cm.GetTruePositives(2));
    }

    [Fact]
    public void ConfusionMatrix_MultiClass_GetFalsePositives()
    {
        var cm = new ConfusionMatrix<double>(3);
        cm.Increment(0, 1); // Predicted 0, actual 1 -> FP for class 0
        cm.Increment(0, 2); // Predicted 0, actual 2 -> FP for class 0
        cm.Increment(1, 0); // Predicted 1, actual 0 -> FP for class 1

        Assert.Equal(2.0, cm.GetFalsePositives(0)); // Two FPs for class 0
        Assert.Equal(1.0, cm.GetFalsePositives(1)); // One FP for class 1
        Assert.Equal(0.0, cm.GetFalsePositives(2)); // No FPs for class 2
    }

    [Fact]
    public void ConfusionMatrix_MultiClass_GetFalseNegatives()
    {
        var cm = new ConfusionMatrix<double>(3);
        cm.Increment(1, 0); // Actual 0, predicted 1 -> FN for class 0
        cm.Increment(2, 0); // Actual 0, predicted 2 -> FN for class 0
        cm.Increment(0, 1); // Actual 1, predicted 0 -> FN for class 1

        Assert.Equal(2.0, cm.GetFalseNegatives(0)); // Two FNs for class 0
        Assert.Equal(1.0, cm.GetFalseNegatives(1)); // One FN for class 1
        Assert.Equal(0.0, cm.GetFalseNegatives(2)); // No FNs for class 2
    }

    [Fact]
    public void ConfusionMatrix_MultiClass_GetAccuracy()
    {
        var cm = new ConfusionMatrix<double>(3);
        // 6 correct out of 10
        for (int i = 0; i < 2; i++) cm.Increment(0, 0);
        for (int i = 0; i < 2; i++) cm.Increment(1, 1);
        for (int i = 0; i < 2; i++) cm.Increment(2, 2);
        for (int i = 0; i < 4; i++) cm.Increment(0, 1); // 4 incorrect

        double expected = 6.0 / 10.0;
        Assert.Equal(expected, cm.GetAccuracy(), Tolerance);
    }

    #endregion

    #region Macro/Micro/Weighted Averages

    [Fact]
    public void ConfusionMatrix_MacroPrecision_AveragesClassPrecisions()
    {
        var cm = new ConfusionMatrix<double>(3);
        // Class 0: 2 TP, 1 FP -> Precision = 2/3
        cm.Increment(0, 0); cm.Increment(0, 0); cm.Increment(0, 1);
        // Class 1: 3 TP, 0 FP -> Precision = 1
        cm.Increment(1, 1); cm.Increment(1, 1); cm.Increment(1, 1);
        // Class 2: 1 TP, 1 FP -> Precision = 1/2
        cm.Increment(2, 2); cm.Increment(2, 0);

        double p0 = 2.0 / 3.0;
        double p1 = 1.0;
        double p2 = 1.0 / 2.0;
        double expected = (p0 + p1 + p2) / 3.0;

        Assert.Equal(expected, cm.GetMacroPrecision(), Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_MicroPrecision_EqualsAccuracy()
    {
        var cm = new ConfusionMatrix<double>(3);
        cm.Increment(0, 0); cm.Increment(0, 0);
        cm.Increment(1, 1);
        cm.Increment(2, 2);
        cm.Increment(0, 1); // Misclassification

        Assert.Equal(cm.GetAccuracy(), cm.GetMicroPrecision(), Tolerance);
    }

    #endregion

    #region Matthews Correlation Coefficient

    [Fact]
    public void ConfusionMatrix_MCC_PerfectClassifier_EqualsOne()
    {
        var cm = new ConfusionMatrix<double>(50, 50, 0, 0);

        Assert.Equal(1.0, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_MCC_RandomClassifier_NearZero()
    {
        // Balanced confusion matrix representing random guessing
        var cm = new ConfusionMatrix<double>(25, 25, 25, 25);

        Assert.Equal(0.0, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_MCC_MultiClass_ComputesCorrectly()
    {
        var cm = new ConfusionMatrix<double>(3);
        // Perfect classification
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                cm.Increment(i, i);
            }
        }

        Assert.Equal(1.0, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    #endregion

    #region Cohen's Kappa

    [Fact]
    public void ConfusionMatrix_CohenKappa_PerfectAgreement_EqualsOne()
    {
        var cm = new ConfusionMatrix<double>(50, 50, 0, 0);

        Assert.Equal(1.0, cm.GetCohenKappa(), Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_CohenKappa_ChanceAgreement_NearZero()
    {
        // Balanced confusion that matches expected by chance
        var cm = new ConfusionMatrix<double>(25, 25, 25, 25);

        Assert.Equal(0.0, cm.GetCohenKappa(), Tolerance);
    }

    #endregion

    #region Hamming Loss

    [Fact]
    public void ConfusionMatrix_HammingLoss_PerfectClassifier_IsZero()
    {
        var cm = new ConfusionMatrix<double>(50, 50, 0, 0);

        Assert.Equal(0.0, cm.GetHammingLoss(), Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_HammingLoss_IsOneMinusAccuracy()
    {
        var cm = new ConfusionMatrix<double>(50, 40, 10, 5);

        double expected = 1.0 - cm.Accuracy;
        Assert.Equal(expected, cm.GetHammingLoss(), Tolerance);
    }

    #endregion

    #region Jaccard Score

    [Fact]
    public void ConfusionMatrix_JaccardScore_PerfectClassifier_IsOne()
    {
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                cm.Increment(i, i);
            }
        }

        Assert.Equal(1.0, cm.GetJaccardScore(), Tolerance);
    }

    [Fact]
    public void ConfusionMatrix_JaccardScore_PerClass_ComputesCorrectly()
    {
        var cm = new ConfusionMatrix<double>(3);
        // Class 0: TP=5, FP=2, FN=1 -> Jaccard = 5/(5+2+1) = 5/8
        cm.Increment(0, 0); cm.Increment(0, 0); cm.Increment(0, 0);
        cm.Increment(0, 0); cm.Increment(0, 0);
        cm.Increment(0, 1); cm.Increment(0, 2); // FP
        cm.Increment(1, 0); // FN

        double expected = 5.0 / 8.0;
        Assert.Equal(expected, cm.GetJaccardScore(0), Tolerance);
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void ConfusionMatrix_DimensionLessThanTwo_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(1));
    }

    [Fact]
    public void ConfusionMatrix_Increment_OutOfRange_ThrowsException()
    {
        var cm = new ConfusionMatrix<double>(3);

        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(3, 0));
    }

    [Fact]
    public void ConfusionMatrix_GetMetrics_OutOfRange_ThrowsException()
    {
        var cm = new ConfusionMatrix<double>(3);

        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetTruePositives(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetTruePositives(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetFalsePositives(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetFalseNegatives(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetPrecision(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetRecall(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.GetF1Score(3));
    }

    [Fact]
    public void ConfusionMatrix_EmptyMatrix_ReturnsZeroMetrics()
    {
        var cm = new ConfusionMatrix<double>(3);

        Assert.Equal(0.0, cm.GetAccuracy());
        Assert.Equal(0.0, cm.GetMacroPrecision());
        Assert.Equal(0.0, cm.GetMacroRecall());
    }

    #endregion
}
