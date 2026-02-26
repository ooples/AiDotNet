using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Deep math-correctness integration tests for ConfusionMatrix.
/// Verifies hand-calculated values, mathematical identities, consistency between
/// binary and multi-class APIs, and cross-metric relationships.
/// </summary>
public class ConfusionMatrixDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Binary-MultiClass Consistency

    [Fact]
    public void Binary_Precision_Equals_GetPrecision0()
    {
        // Binary Precision property and GetPrecision(0) must agree
        // TP=50, TN=30, FP=10, FN=5
        var cm = new ConfusionMatrix<double>(50, 30, 10, 5);

        double binaryPrecision = cm.Precision;
        double multiClassPrecision = cm.GetPrecision(0);

        // Binary: P = TP / (TP + FP) = 50 / 60 = 0.8333
        Assert.Equal(50.0 / 60.0, binaryPrecision, Tolerance);
        Assert.Equal(binaryPrecision, multiClassPrecision, Tolerance);
    }

    [Fact]
    public void Binary_Recall_Equals_GetRecall0()
    {
        // Binary Recall property and GetRecall(0) must agree
        var cm = new ConfusionMatrix<double>(50, 30, 10, 5);

        double binaryRecall = cm.Recall;
        double multiClassRecall = cm.GetRecall(0);

        // Binary: R = TP / (TP + FN) = 50 / 55 = 0.9091
        Assert.Equal(50.0 / 55.0, binaryRecall, Tolerance);
        Assert.Equal(binaryRecall, multiClassRecall, Tolerance);
    }

    [Fact]
    public void Binary_F1_Equals_GetF1Score0()
    {
        var cm = new ConfusionMatrix<double>(50, 30, 10, 5);

        double binaryF1 = cm.F1Score;
        double multiClassF1 = cm.GetF1Score(0);

        Assert.Equal(binaryF1, multiClassF1, Tolerance);
    }

    [Fact]
    public void Binary_Specificity_Consistent_With_GetRecall1()
    {
        // Specificity = TN / (TN + FP) is the "recall" for the negative class
        var cm = new ConfusionMatrix<double>(50, 30, 10, 5);

        double specificity = cm.Specificity;
        double recall1 = cm.GetRecall(1);

        // Specificity = TN / (TN + FP) = 30 / 40 = 0.75
        Assert.Equal(30.0 / 40.0, specificity, Tolerance);
        Assert.Equal(specificity, recall1, Tolerance);
    }

    [Fact]
    public void Increment_Built_Binary_Matches_Constructor_Built()
    {
        // A 2x2 matrix built with Increment should give same metrics as constructor-built
        var cmConstructor = new ConfusionMatrix<double>(3, 4, 2, 1);

        var cmIncrement = new ConfusionMatrix<double>(2);
        // TP: predicted 0 (positive), actual 0 (positive) = 3 times
        for (int i = 0; i < 3; i++) cmIncrement.Increment(0, 0);
        // TN: predicted 1 (negative), actual 1 (negative) = 4 times
        for (int i = 0; i < 4; i++) cmIncrement.Increment(1, 1);
        // FP: predicted 0 (positive), actual 1 (negative) = 2 times
        for (int i = 0; i < 2; i++) cmIncrement.Increment(0, 1);
        // FN: predicted 1 (negative), actual 0 (positive) = 1 time
        cmIncrement.Increment(1, 0);

        Assert.Equal(cmConstructor.Accuracy, cmIncrement.GetAccuracy(), Tolerance);
        Assert.Equal(cmConstructor.Precision, cmIncrement.GetPrecision(0), Tolerance);
        Assert.Equal(cmConstructor.Recall, cmIncrement.GetRecall(0), Tolerance);
        Assert.Equal(cmConstructor.F1Score, cmIncrement.GetF1Score(0), Tolerance);
        Assert.Equal(cmConstructor.Specificity, cmIncrement.GetRecall(1), Tolerance);
    }

    #endregion

    #region Hand-Calculated Binary Metrics

    [Fact]
    public void Binary_HandCalculated_AllMetrics()
    {
        // TP=80, TN=10, FP=5, FN=5 → total=100
        var cm = new ConfusionMatrix<double>(80, 10, 5, 5);

        // Accuracy = (80+10)/100 = 0.90
        Assert.Equal(0.90, cm.Accuracy, Tolerance);

        // Precision = 80/(80+5) = 80/85
        Assert.Equal(80.0 / 85.0, cm.Precision, Tolerance);

        // Recall = 80/(80+5) = 80/85
        Assert.Equal(80.0 / 85.0, cm.Recall, Tolerance);

        // Specificity = 10/(10+5) = 10/15
        Assert.Equal(10.0 / 15.0, cm.Specificity, Tolerance);

        // F1 = 2*P*R/(P+R) = 2*(80/85)*(80/85)/((80/85)+(80/85)) = 80/85
        double p = 80.0 / 85.0;
        double r = 80.0 / 85.0;
        Assert.Equal(2 * p * r / (p + r), cm.F1Score, Tolerance);
    }

    [Fact]
    public void Binary_HandCalculated_AsymmetricFPFN()
    {
        // TP=40, TN=30, FP=20, FN=10 → total=100
        var cm = new ConfusionMatrix<double>(40, 30, 20, 10);

        // Accuracy = 70/100 = 0.70
        Assert.Equal(0.70, cm.Accuracy, Tolerance);

        // Precision = 40/(40+20) = 40/60 = 2/3
        Assert.Equal(2.0 / 3.0, cm.Precision, Tolerance);

        // Recall = 40/(40+10) = 40/50 = 0.80
        Assert.Equal(0.80, cm.Recall, Tolerance);

        // Specificity = 30/(30+20) = 30/50 = 0.60
        Assert.Equal(0.60, cm.Specificity, Tolerance);

        // F1 = 2*(2/3)*0.8/((2/3)+0.8) = 2*0.5333/1.4667 = 1.0667/1.4667
        double precision = 2.0 / 3.0;
        double recall = 0.80;
        double expectedF1 = 2 * precision * recall / (precision + recall);
        Assert.Equal(expectedF1, cm.F1Score, Tolerance);
    }

    #endregion

    #region Mathematical Identities

    [Fact]
    public void Identity_Accuracy_Plus_HammingLoss_Equals_One()
    {
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 7; i++) cm.Increment(0, 0);
        for (int i = 0; i < 5; i++) cm.Increment(1, 1);
        for (int i = 0; i < 3; i++) cm.Increment(2, 2);
        cm.Increment(0, 1);
        cm.Increment(1, 2);
        cm.Increment(2, 0);

        Assert.Equal(1.0, cm.GetAccuracy() + cm.GetHammingLoss(), Tolerance);
    }

    [Fact]
    public void Identity_F1_Equals_2Jaccard_Over_1PlusJaccard()
    {
        // F1 = 2*J / (1+J) where J = TP/(TP+FP+FN) and F1 = 2*P*R/(P+R)
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 3; i++) cm.Increment(0, 1);
        for (int i = 0; i < 2; i++) cm.Increment(1, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);

        double f1_0 = cm.GetF1Score(0);
        double j_0 = cm.GetJaccardScore(0);
        double expected = 2 * j_0 / (1 + j_0);

        Assert.Equal(expected, f1_0, Tolerance);
    }

    [Fact]
    public void Identity_F1_HarmonicMean_LessEqual_ArithmeticMean()
    {
        // F1 (harmonic mean of P and R) <= (P + R) / 2 (arithmetic mean)
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 5; i++) cm.Increment(0, 1);
        for (int i = 0; i < 2; i++) cm.Increment(1, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);
        cm.Increment(2, 1);

        for (int cls = 0; cls < 3; cls++)
        {
            double p = cm.GetPrecision(cls);
            double r = cm.GetRecall(cls);
            double f1 = cm.GetF1Score(cls);
            double arithmeticMean = (p + r) / 2.0;

            Assert.True(f1 <= arithmeticMean + Tolerance,
                $"F1({cls})={f1} should be <= arithmetic mean {arithmeticMean}");
        }
    }

    [Fact]
    public void Identity_PerClass_TP_FP_FN_TN_Equals_Total()
    {
        // For each class: TP + FP + FN + TN = total number of samples
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);
        cm.Increment(0, 1);
        cm.Increment(1, 0);
        cm.Increment(2, 1);
        cm.Increment(0, 2);

        double total = 10 + 8 + 6 + 4; // 28 total samples

        for (int cls = 0; cls < 3; cls++)
        {
            double tp = cm.GetTruePositives(cls);
            double fp = cm.GetFalsePositives(cls);
            double fn = cm.GetFalseNegatives(cls);
            double tn = cm.GetTrueNegatives(cls);

            Assert.Equal(total, tp + fp + fn + tn, Tolerance);
        }
    }

    [Fact]
    public void Identity_MicroPrecision_Equals_MicroRecall_Equals_Accuracy()
    {
        // For multi-class: Micro P = Micro R = Micro F1 = Accuracy
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);
        cm.Increment(0, 1);
        cm.Increment(1, 2);

        double accuracy = cm.GetAccuracy();
        Assert.Equal(accuracy, cm.GetMicroPrecision(), Tolerance);
        Assert.Equal(accuracy, cm.GetMicroRecall(), Tolerance);
        Assert.Equal(accuracy, cm.GetMicroF1Score(), Tolerance);
    }

    [Fact]
    public void Identity_Jaccard_LessEqual_F1()
    {
        // Jaccard <= F1 always (since J = TP/(TP+FP+FN) and F1 = 2TP/(2TP+FP+FN))
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 5; i++) cm.Increment(0, 1);
        for (int i = 0; i < 2; i++) cm.Increment(1, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);
        cm.Increment(2, 1);

        for (int cls = 0; cls < 3; cls++)
        {
            double j = cm.GetJaccardScore(cls);
            double f1 = cm.GetF1Score(cls);

            Assert.True(j <= f1 + Tolerance,
                $"Jaccard({cls})={j} should be <= F1({cls})={f1}");
        }
    }

    #endregion

    #region Hand-Calculated Multi-Class MCC

    [Fact]
    public void MCC_Binary_HandCalculated()
    {
        // TP=20, TN=30, FP=5, FN=10
        var cm = new ConfusionMatrix<double>(20, 30, 5, 10);

        // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        // = (20*30 - 5*10) / sqrt(25 * 30 * 35 * 40)
        // = (600 - 50) / sqrt(1050000)
        // = 550 / 1024.695
        double expected = (20.0 * 30 - 5.0 * 10) /
            Math.Sqrt((20.0 + 5) * (20.0 + 10) * (30.0 + 5) * (30.0 + 10));

        Assert.Equal(expected, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    [Fact]
    public void MCC_MultiClass_HandCalculated_3x3()
    {
        // 3x3 confusion matrix:
        //       actual0  actual1  actual2
        // pred0    5       1        0
        // pred1    2       4        1
        // pred2    0       1        6
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 5; i++) cm.Increment(0, 0);
        cm.Increment(0, 1);
        for (int i = 0; i < 2; i++) cm.Increment(1, 0);
        for (int i = 0; i < 4; i++) cm.Increment(1, 1);
        cm.Increment(1, 2);
        cm.Increment(2, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);

        // c = trace = 5 + 4 + 6 = 15
        // s = total = 5+1+0+2+4+1+0+1+6 = 20
        // pk (row sums): p0=6, p1=7, p2=7
        // tk (col sums): t0=7, t1=6, t2=7
        // sum(pk*tk) = 6*7 + 7*6 + 7*7 = 42 + 42 + 49 = 133
        // sum(pk^2) = 36 + 49 + 49 = 134
        // sum(tk^2) = 49 + 36 + 49 = 134
        // MCC = (c*s - sum(pk*tk)) / sqrt((s^2 - sum(pk^2)) * (s^2 - sum(tk^2)))
        //     = (15*20 - 133) / sqrt((400 - 134) * (400 - 134))
        //     = (300 - 133) / sqrt(266 * 266)
        //     = 167 / 266
        double expected = 167.0 / 266.0;

        Assert.Equal(expected, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    [Fact]
    public void MCC_Bounded_Between_Minus1_And_1()
    {
        // Various confusion matrices should all have MCC in [-1, 1]
        var matrices = new[]
        {
            new ConfusionMatrix<double>(10, 10, 0, 0),   // Perfect
            new ConfusionMatrix<double>(0, 0, 10, 10),   // Total disagreement
            new ConfusionMatrix<double>(5, 5, 5, 5),     // Random
            new ConfusionMatrix<double>(100, 1, 50, 1),  // Imbalanced
            new ConfusionMatrix<double>(1, 100, 1, 50),  // Imbalanced reverse
        };

        foreach (var cm in matrices)
        {
            double mcc = cm.GetMatthewsCorrelationCoefficient();
            Assert.True(mcc >= -1.0 - Tolerance && mcc <= 1.0 + Tolerance,
                $"MCC={mcc} should be in [-1, 1]");
        }
    }

    #endregion

    #region Hand-Calculated Cohen's Kappa

    [Fact]
    public void Kappa_Binary_HandCalculated()
    {
        // TP=45, TN=35, FP=10, FN=10 → total=100
        var cm = new ConfusionMatrix<double>(45, 35, 10, 10);

        // Observed agreement = (45+35)/100 = 0.80
        // Row sums: r0 = 45+10 = 55, r1 = 10+35 = 45
        // Col sums: c0 = 45+10 = 55, c1 = 10+35 = 45
        // Expected agreement = (55*55 + 45*45) / (100*100) = (3025 + 2025) / 10000 = 0.505
        // Kappa = (0.80 - 0.505) / (1 - 0.505) = 0.295 / 0.495 = 0.59596...
        double expected = 0.295 / 0.495;

        Assert.Equal(expected, cm.GetCohenKappa(), 1e-4);
    }

    [Fact]
    public void Kappa_Bounded()
    {
        var matrices = new[]
        {
            new ConfusionMatrix<double>(10, 10, 0, 0),
            new ConfusionMatrix<double>(5, 5, 5, 5),
            new ConfusionMatrix<double>(1, 1, 10, 10),
        };

        foreach (var cm in matrices)
        {
            double kappa = cm.GetCohenKappa();
            Assert.True(kappa >= -1.0 - Tolerance && kappa <= 1.0 + Tolerance,
                $"Kappa={kappa} should be in [-1, 1]");
        }
    }

    [Fact]
    public void Kappa_LessThanOrEqual_Accuracy_ForNonPerfect()
    {
        // Kappa <= accuracy for non-perfect classifiers (adjusts for chance)
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 8; i++) cm.Increment(0, 0);
        for (int i = 0; i < 6; i++) cm.Increment(1, 1);
        for (int i = 0; i < 4; i++) cm.Increment(2, 2);
        cm.Increment(0, 1);
        cm.Increment(1, 2);
        cm.Increment(2, 0);

        double kappa = cm.GetCohenKappa();
        double accuracy = cm.GetAccuracy();

        Assert.True(kappa <= accuracy + Tolerance,
            $"Kappa={kappa} should be <= Accuracy={accuracy}");
    }

    #endregion

    #region Jaccard Mathematical Properties

    [Fact]
    public void Jaccard_HandCalculated()
    {
        // Class 0: TP=8, FP=3, FN=2
        // Jaccard = 8 / (8+3+2) = 8/13
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 8; i++) cm.Increment(0, 0);
        for (int i = 0; i < 3; i++) cm.Increment(0, 1);
        for (int i = 0; i < 2; i++) cm.Increment(1, 0);
        for (int i = 0; i < 5; i++) cm.Increment(1, 1);

        Assert.Equal(8.0 / 13.0, cm.GetJaccardScore(0), Tolerance);
    }

    [Fact]
    public void Jaccard_Bounded_0_to_1()
    {
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 5; i++) cm.Increment(0, 1);
        for (int i = 0; i < 3; i++) cm.Increment(1, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);

        for (int cls = 0; cls < 3; cls++)
        {
            double j = cm.GetJaccardScore(cls);
            Assert.True(j >= 0 - Tolerance && j <= 1 + Tolerance,
                $"Jaccard({cls})={j} should be in [0, 1]");
        }
    }

    #endregion

    #region Perfect and Worst-Case Scenarios

    [Fact]
    public void Perfect_Binary_AllMetricsOptimal()
    {
        var cm = new ConfusionMatrix<double>(50, 50, 0, 0);

        Assert.Equal(1.0, cm.Accuracy, Tolerance);
        Assert.Equal(1.0, cm.Precision, Tolerance);
        Assert.Equal(1.0, cm.Recall, Tolerance);
        Assert.Equal(1.0, cm.F1Score, Tolerance);
        Assert.Equal(1.0, cm.Specificity, Tolerance);
        Assert.Equal(1.0, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
        Assert.Equal(1.0, cm.GetCohenKappa(), Tolerance);
        Assert.Equal(0.0, cm.GetHammingLoss(), Tolerance);
        Assert.Equal(1.0, cm.GetJaccardScore(), Tolerance);
    }

    [Fact]
    public void Perfect_MultiClass_AllMetricsOptimal()
    {
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 10; i++) cm.Increment(1, 1);
        for (int i = 0; i < 10; i++) cm.Increment(2, 2);

        Assert.Equal(1.0, cm.GetAccuracy(), Tolerance);
        Assert.Equal(0.0, cm.GetHammingLoss(), Tolerance);
        Assert.Equal(1.0, cm.GetMacroPrecision(), Tolerance);
        Assert.Equal(1.0, cm.GetMacroRecall(), Tolerance);
        Assert.Equal(1.0, cm.GetMacroF1Score(), Tolerance);
        Assert.Equal(1.0, cm.GetWeightedPrecision(), Tolerance);
        Assert.Equal(1.0, cm.GetWeightedRecall(), Tolerance);
        Assert.Equal(1.0, cm.GetWeightedF1Score(), Tolerance);
        Assert.Equal(1.0, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
        Assert.Equal(1.0, cm.GetCohenKappa(), Tolerance);
        Assert.Equal(1.0, cm.GetJaccardScore(), Tolerance);
    }

    [Fact]
    public void AllWrong_Binary_Accuracy_Zero()
    {
        var cm = new ConfusionMatrix<double>(0, 0, 10, 10);

        Assert.Equal(0.0, cm.Accuracy, Tolerance);
        Assert.Equal(1.0, cm.GetHammingLoss(), Tolerance);
        Assert.Equal(-1.0, cm.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    [Fact]
    public void AllWrong_MultiClass_Accuracy_Zero()
    {
        // All predictions wrong: predicted 0→actual 1, 1→actual 2, 2→actual 0
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 1);
        for (int i = 0; i < 10; i++) cm.Increment(1, 2);
        for (int i = 0; i < 10; i++) cm.Increment(2, 0);

        Assert.Equal(0.0, cm.GetAccuracy(), Tolerance);
        Assert.Equal(1.0, cm.GetHammingLoss(), Tolerance);
    }

    #endregion

    #region Weighted Average Properties

    [Fact]
    public void Weighted_Recall_Equals_Accuracy_For_MultiClass()
    {
        // Weighted recall = sum(recall_i * support_i) / total
        // = sum(TP_i / (TP_i + FN_i) * (TP_i + FN_i)) / total
        // = sum(TP_i) / total = accuracy
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);
        cm.Increment(0, 1);
        cm.Increment(1, 2);
        cm.Increment(2, 0);

        Assert.Equal(cm.GetAccuracy(), cm.GetWeightedRecall(), Tolerance);
    }

    [Fact]
    public void Macro_Average_Gives_Equal_Weight_To_Each_Class()
    {
        // Macro average: each class contributes equally regardless of support
        var cm = new ConfusionMatrix<double>(2);
        // Class 0: 100 samples, perfect → Precision = 1.0
        for (int i = 0; i < 100; i++) cm.Increment(0, 0);
        // Class 1: 2 samples, perfect → Precision = 1.0
        cm.Increment(1, 1);
        cm.Increment(1, 1);

        // Macro precision should be (1.0 + 1.0) / 2 = 1.0
        Assert.Equal(1.0, cm.GetMacroPrecision(), Tolerance);
    }

    [Fact]
    public void Macro_F1_Is_Average_Of_PerClass_F1()
    {
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 5; i++) cm.Increment(0, 1);
        for (int i = 0; i < 3; i++) cm.Increment(1, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 6; i++) cm.Increment(2, 2);
        cm.Increment(2, 0);

        double expectedMacroF1 = (cm.GetF1Score(0) + cm.GetF1Score(1) + cm.GetF1Score(2)) / 3.0;
        Assert.Equal(expectedMacroF1, cm.GetMacroF1Score(), Tolerance);
    }

    #endregion

    #region Multi-Class Per-Class Metric Correctness

    [Fact]
    public void MultiClass_PerClass_HandCalculated()
    {
        // 3x3 confusion matrix:
        //       actual0  actual1  actual2
        // pred0    10      2        0       → row sum = 12
        // pred1     1      8        3       → row sum = 12
        // pred2     0      1        5       → row sum = 6
        //          ↓11     ↓11      ↓8       total = 30
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);
        for (int i = 0; i < 2; i++) cm.Increment(0, 1);
        cm.Increment(1, 0);
        for (int i = 0; i < 8; i++) cm.Increment(1, 1);
        for (int i = 0; i < 3; i++) cm.Increment(1, 2);
        cm.Increment(2, 1);
        for (int i = 0; i < 5; i++) cm.Increment(2, 2);

        // Class 0: TP=10, FP=2(row 0, non-diag), FN=1(col 0, non-diag)
        Assert.Equal(10.0, cm.GetTruePositives(0), Tolerance);
        Assert.Equal(2.0, cm.GetFalsePositives(0), Tolerance);
        Assert.Equal(1.0, cm.GetFalseNegatives(0), Tolerance);
        // Precision(0) = 10/(10+2) = 10/12
        Assert.Equal(10.0 / 12.0, cm.GetPrecision(0), Tolerance);
        // Recall(0) = 10/(10+1) = 10/11
        Assert.Equal(10.0 / 11.0, cm.GetRecall(0), Tolerance);

        // Class 1: TP=8, FP=4(row 1 non-diag: 1+3), FN=3(col 1 non-diag: 2+1)
        Assert.Equal(8.0, cm.GetTruePositives(1), Tolerance);
        Assert.Equal(4.0, cm.GetFalsePositives(1), Tolerance);
        Assert.Equal(3.0, cm.GetFalseNegatives(1), Tolerance);
        // Precision(1) = 8/(8+4) = 8/12
        Assert.Equal(8.0 / 12.0, cm.GetPrecision(1), Tolerance);
        // Recall(1) = 8/(8+3) = 8/11
        Assert.Equal(8.0 / 11.0, cm.GetRecall(1), Tolerance);

        // Class 2: TP=5, FP=1(row 2 non-diag: 0+1), FN=3(col 2 non-diag: 0+3)
        Assert.Equal(5.0, cm.GetTruePositives(2), Tolerance);
        Assert.Equal(1.0, cm.GetFalsePositives(2), Tolerance);
        Assert.Equal(3.0, cm.GetFalseNegatives(2), Tolerance);
        // Precision(2) = 5/(5+1) = 5/6
        Assert.Equal(5.0 / 6.0, cm.GetPrecision(2), Tolerance);
        // Recall(2) = 5/(5+3) = 5/8
        Assert.Equal(5.0 / 8.0, cm.GetRecall(2), Tolerance);

        // Accuracy = trace / total = (10+8+5)/30 = 23/30
        Assert.Equal(23.0 / 30.0, cm.GetAccuracy(), Tolerance);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Empty_Matrix_Returns_Zero_ForAll()
    {
        var cm = new ConfusionMatrix<double>(3);

        Assert.Equal(0.0, cm.GetAccuracy());
        Assert.Equal(0.0, cm.GetMatthewsCorrelationCoefficient());
        Assert.Equal(0.0, cm.GetCohenKappa());
        Assert.Equal(0.0, cm.GetJaccardScore());
    }

    [Fact]
    public void SingleClass_AllPredictions_Correct()
    {
        // Only one class has any predictions
        var cm = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) cm.Increment(0, 0);

        Assert.Equal(1.0, cm.GetAccuracy(), Tolerance);
        Assert.Equal(1.0, cm.GetPrecision(0), Tolerance);
        Assert.Equal(1.0, cm.GetRecall(0), Tolerance);
        Assert.Equal(0.0, cm.GetPrecision(1)); // No predictions for class 1
        Assert.Equal(0.0, cm.GetRecall(1));     // No actual class 1 samples
    }

    [Fact]
    public void Dimension_Minimum_2_Enforced()
    {
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(1));
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(0));
    }

    [Fact]
    public void Increment_OutOfRange_Throws()
    {
        var cm = new ConfusionMatrix<double>(3);
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cm.Increment(0, 3));
    }

    #endregion

    #region Metric Ordering and Relationships

    [Fact]
    public void Better_Predictions_Improve_All_Metrics()
    {
        // Good model
        var good = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 18; i++) good.Increment(i % 3, i % 3); // 18 correct
        good.Increment(0, 1); // 2 wrong
        good.Increment(1, 2);

        // Bad model
        var bad = new ConfusionMatrix<double>(3);
        for (int i = 0; i < 10; i++) bad.Increment(i % 3, i % 3); // 10 correct
        for (int i = 0; i < 10; i++) bad.Increment(i % 3, (i + 1) % 3); // 10 wrong

        Assert.True(good.GetAccuracy() > bad.GetAccuracy());
        Assert.True(good.GetMatthewsCorrelationCoefficient() > bad.GetMatthewsCorrelationCoefficient());
        Assert.True(good.GetCohenKappa() > bad.GetCohenKappa());
        Assert.True(good.GetMacroF1Score() > bad.GetMacroF1Score());
        Assert.True(good.GetHammingLoss() < bad.GetHammingLoss());
    }

    [Fact]
    public void MCC_PerfectPrediction_1_TotalDisagreement_Minus1()
    {
        var perfect = new ConfusionMatrix<double>(10, 10, 0, 0);
        var inverted = new ConfusionMatrix<double>(0, 0, 10, 10);

        Assert.Equal(1.0, perfect.GetMatthewsCorrelationCoefficient(), Tolerance);
        Assert.Equal(-1.0, inverted.GetMatthewsCorrelationCoefficient(), Tolerance);
    }

    #endregion

    #region Large Matrix Stress Tests

    [Fact]
    public void LargeMatrix_10Class_MetricsConsistent()
    {
        var cm = new ConfusionMatrix<double>(10);
        var random = new Random(42);

        // Fill with mostly diagonal (correct) with some noise
        for (int i = 0; i < 1000; i++)
        {
            int actual = random.Next(10);
            int predicted = random.NextDouble() < 0.8 ? actual : random.Next(10);
            cm.Increment(predicted, actual);
        }

        // All metrics should be in valid range
        Assert.True(cm.GetAccuracy() >= 0 && cm.GetAccuracy() <= 1);
        Assert.True(cm.GetHammingLoss() >= 0 && cm.GetHammingLoss() <= 1);
        Assert.Equal(1.0, cm.GetAccuracy() + cm.GetHammingLoss(), Tolerance);

        double mcc = cm.GetMatthewsCorrelationCoefficient();
        Assert.True(mcc >= -1 && mcc <= 1);

        double kappa = cm.GetCohenKappa();
        Assert.True(kappa >= -1 && kappa <= 1);

        // With 80% accuracy, MCC and Kappa should be significantly positive
        Assert.True(mcc > 0.5, $"MCC={mcc} should be > 0.5 for 80% accurate 10-class");
        Assert.True(kappa > 0.5, $"Kappa={kappa} should be > 0.5 for 80% accurate 10-class");

        // Micro P = Micro R = Micro F1 = Accuracy
        Assert.Equal(cm.GetAccuracy(), cm.GetMicroPrecision(), Tolerance);
        Assert.Equal(cm.GetAccuracy(), cm.GetMicroRecall(), Tolerance);
        Assert.Equal(cm.GetAccuracy(), cm.GetMicroF1Score(), Tolerance);

        // Weighted recall = accuracy
        Assert.Equal(cm.GetAccuracy(), cm.GetWeightedRecall(), Tolerance);

        // All per-class metrics bounded
        for (int cls = 0; cls < 10; cls++)
        {
            Assert.True(cm.GetPrecision(cls) >= 0 && cm.GetPrecision(cls) <= 1);
            Assert.True(cm.GetRecall(cls) >= 0 && cm.GetRecall(cls) <= 1);
            Assert.True(cm.GetF1Score(cls) >= 0 && cm.GetF1Score(cls) <= 1);
            Assert.True(cm.GetJaccardScore(cls) >= 0 && cm.GetJaccardScore(cls) <= 1);

            // Jaccard <= F1
            Assert.True(cm.GetJaccardScore(cls) <= cm.GetF1Score(cls) + Tolerance);

            // TP + FP + FN + TN = 1000
            double total = cm.GetTruePositives(cls) + cm.GetFalsePositives(cls) +
                          cm.GetFalseNegatives(cls) + cm.GetTrueNegatives(cls);
            Assert.Equal(1000.0, total, Tolerance);
        }
    }

    #endregion
}
