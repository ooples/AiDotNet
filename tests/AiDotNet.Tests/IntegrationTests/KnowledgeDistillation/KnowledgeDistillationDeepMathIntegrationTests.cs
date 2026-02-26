using System;
using AiDotNet.KnowledgeDistillation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.KnowledgeDistillation;

/// <summary>
/// Deep mathematical correctness tests for knowledge distillation loss functions.
/// Verifies softmax with temperature, KL divergence, cross-entropy, and combined
/// distillation loss against hand-calculated expected values.
/// </summary>
public class KnowledgeDistillationDeepMathIntegrationTests
{
    private const double Tol = 1e-6;
    private const double RelaxedTol = 1e-4;

    #region Softmax with Temperature - Verified through Loss

    [Fact]
    public void DistillationLoss_IdenticalStudentTeacher_SoftLossIsZero()
    {
        // When student and teacher have identical logits, KL divergence is 0
        // So soft loss = 0, and without labels, total loss = 0
        var loss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);

        var studentLogits = new Matrix<double>(1, 3);
        studentLogits[0, 0] = 1.0; studentLogits[0, 1] = 2.0; studentLogits[0, 2] = 3.0;

        var teacherLogits = new Matrix<double>(1, 3);
        teacherLogits[0, 0] = 1.0; teacherLogits[0, 1] = 2.0; teacherLogits[0, 2] = 3.0;

        double result = loss.ComputeLoss(studentLogits, teacherLogits);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void DistillationLoss_SoftOnly_HandCalculated()
    {
        // No true labels -> only soft loss
        // student logits = [2, 1], teacher logits = [1, 2], T = 1.0
        // softmax([2, 1]) = [exp(2)/(exp(2)+exp(1)), exp(1)/(exp(2)+exp(1))]
        //                 = [e^2/(e^2+e), e/(e^2+e)]
        //                 = [e/(e+1), 1/(e+1)]
        //                 ~ [0.7311, 0.2689]
        // softmax([1, 2]) = [1/(1+e), e/(1+e)]
        //                 ~ [0.2689, 0.7311]
        //
        // KL(teacher || student) = sum(p * log(p/q))
        //   = 0.2689 * ln(0.2689/0.7311) + 0.7311 * ln(0.7311/0.2689)
        //   = 0.2689 * ln(0.3679) + 0.7311 * ln(2.7183)
        //   = 0.2689 * (-1.0) + 0.7311 * 1.0
        //   = -0.2689 + 0.7311 = 0.4621...
        //
        // T^2 = 1.0, batch_size = 1
        // soft loss = 0.4621 * 1 / 1 = 0.4621
        var loss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.3);

        var studentLogits = new Matrix<double>(1, 2);
        studentLogits[0, 0] = 2.0; studentLogits[0, 1] = 1.0;

        var teacherLogits = new Matrix<double>(1, 2);
        teacherLogits[0, 0] = 1.0; teacherLogits[0, 1] = 2.0;

        double result = loss.ComputeLoss(studentLogits, teacherLogits);

        // Hand-calculated KL divergence
        double e = Math.E;
        double p0 = 1.0 / (1.0 + e);  // teacher softmax class 0
        double p1 = e / (1.0 + e);     // teacher softmax class 1
        double q0 = e / (1.0 + e);     // student softmax class 0
        double q1 = 1.0 / (1.0 + e);   // student softmax class 1

        double kl = p0 * Math.Log(p0 / q0) + p1 * Math.Log(p1 / q1);
        // With T^2 = 1 and batch_size = 1
        double expected = kl;

        Assert.Equal(expected, result, RelaxedTol);
    }

    [Fact]
    public void DistillationLoss_HigherTemperature_SofterDistribution()
    {
        // Higher temperature should produce softer distributions
        // With identical student/teacher, loss is always 0
        // But with different logits, higher temperature should give LOWER soft loss
        // because the distributions become more uniform and thus more similar
        var lowTempLoss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var highTempLoss = new DistillationLoss<double>(temperature: 10.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 3);
        studentLogits[0, 0] = 5.0; studentLogits[0, 1] = 1.0; studentLogits[0, 2] = 0.0;

        var teacherLogits = new Matrix<double>(1, 3);
        teacherLogits[0, 0] = 0.0; teacherLogits[0, 1] = 1.0; teacherLogits[0, 2] = 5.0;

        double lowTempResult = lowTempLoss.ComputeLoss(studentLogits, teacherLogits);
        double highTempResult = highTempLoss.ComputeLoss(studentLogits, teacherLogits);

        // Note: soft loss is scaled by T^2, so the comparison depends on the KL * T^2 product
        // With high temperature, distributions are similar (KL is small) but scaled by large T^2
        // With low temperature, distributions are different (KL is large) but scaled by small T^2
        // The net effect can go either way - what matters is they're both positive
        Assert.True(lowTempResult > 0, $"Low temp loss should be positive: {lowTempResult}");
        Assert.True(highTempResult > 0, $"High temp loss should be positive: {highTempResult}");
    }

    #endregion

    #region KL Divergence Properties

    [Fact]
    public void DistillationLoss_KLDivergence_NonNegative()
    {
        // KL divergence is always >= 0 (Gibbs' inequality)
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);

        // Various different logit pairs
        var testCases = new[]
        {
            (new double[] { 1, 2, 3 }, new double[] { 3, 2, 1 }),
            (new double[] { 0, 0, 0 }, new double[] { 1, 1, 1 }),
            (new double[] { 10, 0, 0 }, new double[] { 0, 0, 10 }),
            (new double[] { 1, 1 }, new double[] { 5, 1 }),
        };

        foreach (var (studentVals, teacherVals) in testCases)
        {
            var student = new Matrix<double>(1, studentVals.Length);
            var teacher = new Matrix<double>(1, teacherVals.Length);
            for (int i = 0; i < studentVals.Length; i++)
            {
                student[0, i] = studentVals[i];
                teacher[0, i] = teacherVals[i];
            }

            double result = loss.ComputeLoss(student, teacher);
            Assert.True(result >= -Tol,
                $"KL divergence should be non-negative, got {result} for student=[{string.Join(",", studentVals)}] teacher=[{string.Join(",", teacherVals)}]");
        }
    }

    [Fact]
    public void DistillationLoss_KLDivergence_ZeroForIdentical()
    {
        var loss = new DistillationLoss<double>(temperature: 5.0, alpha: 0.0);

        double[] logits = { 1.5, -0.5, 3.2, 0.1 };
        var student = new Matrix<double>(1, logits.Length);
        var teacher = new Matrix<double>(1, logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            student[0, i] = logits[i];
            teacher[0, i] = logits[i];
        }

        double result = loss.ComputeLoss(student, teacher);
        Assert.Equal(0.0, result, Tol);
    }

    #endregion

    #region Cross-Entropy and Combined Loss

    [Fact]
    public void DistillationLoss_WithLabels_HandCalculated()
    {
        // student logits = [2, 0], teacher logits = [2, 0], T=1, alpha=0.5
        // true labels = [1, 0] (one-hot)
        //
        // Since student = teacher, soft loss = 0
        // Hard loss: softmax([2, 0]) = [e^2/(e^2+1), 1/(e^2+1)]
        //   ~ [0.8808, 0.1192]
        // CE = -(1*ln(0.8808) + 0*ln(0.1192)) = -ln(0.8808) ~ 0.1269
        //
        // Total = 0.5 * 0.1269 + 0.5 * 0 = 0.0634
        var loss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.5);

        var studentLogits = new Matrix<double>(1, 2);
        studentLogits[0, 0] = 2.0; studentLogits[0, 1] = 0.0;

        var teacherLogits = new Matrix<double>(1, 2);
        teacherLogits[0, 0] = 2.0; teacherLogits[0, 1] = 0.0;

        var labels = new Matrix<double>(1, 2);
        labels[0, 0] = 1.0; labels[0, 1] = 0.0;

        double result = loss.ComputeLoss(studentLogits, teacherLogits, labels);

        // softmax([2, 0]) at T=1
        double e2 = Math.Exp(2);
        double p0 = e2 / (e2 + 1);
        double crossEntropy = -Math.Log(p0 + 1e-10); // +epsilon as in code
        double expected = 0.5 * crossEntropy; // alpha=0.5, soft_loss=0

        Assert.Equal(expected, result, RelaxedTol);
    }

    [Fact]
    public void DistillationLoss_AlphaZero_OnlySoftLoss()
    {
        // alpha = 0 means only soft loss
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 2);
        studentLogits[0, 0] = 3.0; studentLogits[0, 1] = 1.0;

        var teacherLogits = new Matrix<double>(1, 2);
        teacherLogits[0, 0] = 1.0; teacherLogits[0, 1] = 3.0;

        var labels = new Matrix<double>(1, 2);
        labels[0, 0] = 1.0; labels[0, 1] = 0.0;

        double withLabels = loss.ComputeLoss(studentLogits, teacherLogits, labels);
        double withoutLabels = loss.ComputeLoss(studentLogits, teacherLogits);

        // When alpha=0, labels should have no effect (0 * hardLoss + 1 * softLoss)
        Assert.Equal(withoutLabels, withLabels, Tol);
    }

    [Fact]
    public void DistillationLoss_AlphaOne_OnlyHardLoss()
    {
        // alpha = 1 means only hard loss
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 1.0);

        var studentLogits = new Matrix<double>(1, 2);
        studentLogits[0, 0] = 3.0; studentLogits[0, 1] = 1.0;

        var teacherLogits = new Matrix<double>(1, 2);
        teacherLogits[0, 0] = 1.0; teacherLogits[0, 1] = 3.0;

        var labels = new Matrix<double>(1, 2);
        labels[0, 0] = 1.0; labels[0, 1] = 0.0;

        double result = loss.ComputeLoss(studentLogits, teacherLogits, labels);

        // Only hard loss: CE(softmax([3,1]), [1,0])
        // softmax([3,1]) at T=1: [exp(3)/(exp(3)+exp(1)), exp(1)/(exp(3)+exp(1))]
        double e3 = Math.Exp(3);
        double e1 = Math.Exp(1);
        double p0 = e3 / (e3 + e1);
        double hardLoss = -Math.Log(p0 + 1e-10);
        double expected = 1.0 * hardLoss; // alpha=1

        Assert.Equal(expected, result, RelaxedTol);
    }

    [Fact]
    public void DistillationLoss_TSquaredScaling_Effect()
    {
        // T^2 scaling: soft loss should be scaled by T^2
        // For identical student/teacher, soft loss = 0 regardless of T
        // So test with different logits
        var lossT1 = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var lossT2 = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 2);
        studentLogits[0, 0] = 2.0; studentLogits[0, 1] = 0.0;

        var teacherLogits = new Matrix<double>(1, 2);
        teacherLogits[0, 0] = 0.0; teacherLogits[0, 1] = 2.0;

        double resultT1 = lossT1.ComputeLoss(studentLogits, teacherLogits);
        double resultT2 = lossT2.ComputeLoss(studentLogits, teacherLogits);

        // Both should be positive
        Assert.True(resultT1 > 0, $"T=1 loss should be positive: {resultT1}");
        Assert.True(resultT2 > 0, $"T=2 loss should be positive: {resultT2}");
    }

    #endregion

    #region Gradient Verification

    [Fact]
    public void DistillationLoss_Gradient_IdenticalLogits_ZeroGradient()
    {
        // When student = teacher, soft gradient should be zero
        var loss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        var logits = new Matrix<double>(1, 3);
        logits[0, 0] = 1.0; logits[0, 1] = 2.0; logits[0, 2] = 3.0;

        var gradient = loss.ComputeGradient(logits, logits);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(0.0, gradient[0, i], Tol);
        }
    }

    [Fact]
    public void DistillationLoss_Gradient_SoftOnly_HandCalculated()
    {
        // Soft gradient = (student_soft - teacher_soft) * T^2 / batch_size
        // student logits = [2, 0], teacher logits = [0, 2], T=1
        // student soft = softmax([2,0]) = [e^2/(e^2+1), 1/(e^2+1)]
        // teacher soft = softmax([0,2]) = [1/(1+e^2), e^2/(1+e^2)]
        // gradient = (student_soft - teacher_soft) * 1 / 1
        var loss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 2);
        studentLogits[0, 0] = 2.0; studentLogits[0, 1] = 0.0;

        var teacherLogits = new Matrix<double>(1, 2);
        teacherLogits[0, 0] = 0.0; teacherLogits[0, 1] = 2.0;

        var gradient = loss.ComputeGradient(studentLogits, teacherLogits);

        double e2 = Math.Exp(2);
        double studentP0 = e2 / (e2 + 1);
        double teacherP0 = 1.0 / (1 + e2);
        double expectedGrad0 = (studentP0 - teacherP0) * 1.0; // T^2=1, /batchSize=1

        Assert.Equal(expectedGrad0, gradient[0, 0], RelaxedTol);
    }

    [Fact]
    public void DistillationLoss_Gradient_SumsToZero_SoftGradient()
    {
        // For softmax-based gradient, the sum across classes should be approximately 0
        // because softmax probabilities sum to 1 on both sides
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 4);
        studentLogits[0, 0] = 1; studentLogits[0, 1] = 3; studentLogits[0, 2] = 0; studentLogits[0, 3] = 2;

        var teacherLogits = new Matrix<double>(1, 4);
        teacherLogits[0, 0] = 2; teacherLogits[0, 1] = 0; teacherLogits[0, 2] = 3; teacherLogits[0, 3] = 1;

        var gradient = loss.ComputeGradient(studentLogits, teacherLogits);

        double gradSum = 0;
        for (int i = 0; i < 4; i++)
        {
            gradSum += gradient[0, i];
        }

        // Sum of (student_soft - teacher_soft) = (sum of student_soft) - (sum of teacher_soft) = 1 - 1 = 0
        Assert.Equal(0.0, gradSum, RelaxedTol);
    }

    #endregion

    #region Batch Processing

    [Fact]
    public void DistillationLoss_BatchOf2_IsAverageOfIndividualLosses()
    {
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);

        // Sample 1: student=[1, 2], teacher=[2, 1]
        var student1 = new Matrix<double>(1, 2);
        student1[0, 0] = 1; student1[0, 1] = 2;
        var teacher1 = new Matrix<double>(1, 2);
        teacher1[0, 0] = 2; teacher1[0, 1] = 1;
        double loss1 = loss.ComputeLoss(student1, teacher1);

        // Sample 2: student=[3, 0], teacher=[0, 3]
        var student2 = new Matrix<double>(1, 2);
        student2[0, 0] = 3; student2[0, 1] = 0;
        var teacher2 = new Matrix<double>(1, 2);
        teacher2[0, 0] = 0; teacher2[0, 1] = 3;
        double loss2 = loss.ComputeLoss(student2, teacher2);

        // Batch of both
        var studentBatch = new Matrix<double>(2, 2);
        studentBatch[0, 0] = 1; studentBatch[0, 1] = 2;
        studentBatch[1, 0] = 3; studentBatch[1, 1] = 0;
        var teacherBatch = new Matrix<double>(2, 2);
        teacherBatch[0, 0] = 2; teacherBatch[0, 1] = 1;
        teacherBatch[1, 0] = 0; teacherBatch[1, 1] = 3;
        double batchLoss = loss.ComputeLoss(studentBatch, teacherBatch);

        // Batch loss should be average of individual losses
        double expectedAvg = (loss1 + loss2) / 2.0;
        Assert.Equal(expectedAvg, batchLoss, RelaxedTol);
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void DistillationLoss_InvalidTemperature_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DistillationLoss<double>(temperature: 0));
        Assert.Throws<ArgumentException>(() =>
            new DistillationLoss<double>(temperature: -1));
    }

    [Fact]
    public void DistillationLoss_InvalidAlpha_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DistillationLoss<double>(alpha: -0.1));
        Assert.Throws<ArgumentException>(() =>
            new DistillationLoss<double>(alpha: 1.1));
    }

    [Fact]
    public void DistillationLoss_MismatchedDimensions_Throws()
    {
        var loss = new DistillationLoss<double>();

        var student = new Matrix<double>(1, 3);
        var teacher = new Matrix<double>(1, 2); // Different number of classes

        Assert.Throws<ArgumentException>(() =>
            loss.ComputeLoss(student, teacher));
    }

    [Fact]
    public void DistillationLoss_MismatchedBatchSize_Throws()
    {
        var loss = new DistillationLoss<double>();

        var student = new Matrix<double>(2, 3);
        var teacher = new Matrix<double>(1, 3); // Different batch size

        Assert.Throws<ArgumentException>(() =>
            loss.ComputeLoss(student, teacher));
    }

    [Fact]
    public void DistillationLoss_LabelDimensionMismatch_Throws()
    {
        var loss = new DistillationLoss<double>();

        var student = new Matrix<double>(1, 3);
        var teacher = new Matrix<double>(1, 3);
        var labels = new Matrix<double>(1, 2); // Different num classes

        Assert.Throws<ArgumentException>(() =>
            loss.ComputeLoss(student, teacher, labels));
    }

    [Fact]
    public void DistillationLoss_NullStudent_Throws()
    {
        var loss = new DistillationLoss<double>();
        var teacher = new Matrix<double>(1, 3);

        Assert.Throws<ArgumentNullException>(() =>
            loss.ComputeLoss(null!, teacher));
    }

    [Fact]
    public void DistillationLoss_NullTeacher_Throws()
    {
        var loss = new DistillationLoss<double>();
        var student = new Matrix<double>(1, 3);

        Assert.Throws<ArgumentNullException>(() =>
            loss.ComputeLoss(student, null!));
    }

    [Fact]
    public void DistillationLoss_LargeLogits_NumericalStability()
    {
        // Large logit values should not cause overflow due to max subtraction trick
        var loss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 3);
        studentLogits[0, 0] = 100; studentLogits[0, 1] = 200; studentLogits[0, 2] = 300;

        var teacherLogits = new Matrix<double>(1, 3);
        teacherLogits[0, 0] = 300; teacherLogits[0, 1] = 200; teacherLogits[0, 2] = 100;

        double result = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.True(double.IsFinite(result), $"Loss should be finite with large logits, got {result}");
        Assert.True(result >= 0, $"Loss should be non-negative: {result}");
    }

    [Fact]
    public void DistillationLoss_VerySimilarLogits_SmallLoss()
    {
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);

        var studentLogits = new Matrix<double>(1, 3);
        studentLogits[0, 0] = 1.0; studentLogits[0, 1] = 2.0; studentLogits[0, 2] = 3.0;

        var teacherLogits = new Matrix<double>(1, 3);
        teacherLogits[0, 0] = 1.01; teacherLogits[0, 1] = 2.01; teacherLogits[0, 2] = 3.01;

        double result = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.True(result < 0.01, $"Very similar logits should give near-zero loss, got {result}");
    }

    #endregion

    #region Temperature and Alpha Property Tests

    [Fact]
    public void DistillationLoss_DefaultValues()
    {
        var loss = new DistillationLoss<double>();
        Assert.Equal(3.0, loss.Temperature, Tol);
        Assert.Equal(0.3, loss.Alpha, Tol);
    }

    [Fact]
    public void DistillationLoss_TemperatureProperty_CanBeChanged()
    {
        var loss = new DistillationLoss<double>(temperature: 2.0);
        Assert.Equal(2.0, loss.Temperature, Tol);

        loss.Temperature = 5.0;
        Assert.Equal(5.0, loss.Temperature, Tol);

        Assert.Throws<ArgumentException>(() => loss.Temperature = 0);
    }

    [Fact]
    public void DistillationLoss_AlphaProperty_CanBeChanged()
    {
        var loss = new DistillationLoss<double>(alpha: 0.5);
        Assert.Equal(0.5, loss.Alpha, Tol);

        loss.Alpha = 0.8;
        Assert.Equal(0.8, loss.Alpha, Tol);

        Assert.Throws<ArgumentException>(() => loss.Alpha = -0.1);
        Assert.Throws<ArgumentException>(() => loss.Alpha = 1.1);
    }

    #endregion

    #region Gradient Consistency with Loss

    [Fact]
    public void DistillationLoss_GradientShape_MatchesInput()
    {
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.5);

        var student = new Matrix<double>(3, 4);
        var teacher = new Matrix<double>(3, 4);
        var labels = new Matrix<double>(3, 4);

        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
            {
                student[r, c] = r + c;
                teacher[r, c] = r * 2 - c;
                labels[r, c] = (r == c) ? 1.0 : 0.0;
            }

        var gradient = loss.ComputeGradient(student, teacher, labels);

        Assert.Equal(3, gradient.Rows);
        Assert.Equal(4, gradient.Columns);
    }

    [Fact]
    public void DistillationLoss_NumericalGradient_ApproximatesAnalytic()
    {
        // Verify gradient using finite differences
        var loss = new DistillationLoss<double>(temperature: 2.0, alpha: 0.0);
        double eps = 1e-5;

        var studentLogits = new Matrix<double>(1, 3);
        studentLogits[0, 0] = 1.0; studentLogits[0, 1] = 2.0; studentLogits[0, 2] = 0.5;

        var teacherLogits = new Matrix<double>(1, 3);
        teacherLogits[0, 0] = 0.5; teacherLogits[0, 1] = 1.0; teacherLogits[0, 2] = 2.0;

        var analyticGrad = loss.ComputeGradient(studentLogits, teacherLogits);

        for (int c = 0; c < 3; c++)
        {
            // Forward perturbation
            var perturbedPlus = new Matrix<double>(1, 3);
            var perturbedMinus = new Matrix<double>(1, 3);
            for (int j = 0; j < 3; j++)
            {
                perturbedPlus[0, j] = studentLogits[0, j];
                perturbedMinus[0, j] = studentLogits[0, j];
            }
            perturbedPlus[0, c] = studentLogits[0, c] + eps;
            perturbedMinus[0, c] = studentLogits[0, c] - eps;

            double lossPlus = loss.ComputeLoss(perturbedPlus, teacherLogits);
            double lossMinus = loss.ComputeLoss(perturbedMinus, teacherLogits);

            double numericalGrad = (lossPlus - lossMinus) / (2 * eps);
            double analyticGradVal = analyticGrad[0, c];

            Assert.Equal(numericalGrad, analyticGradVal, RelaxedTol);
        }
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void DistillationLoss_Float_ProducesReasonableResult()
    {
        var loss = new DistillationLoss<float>(temperature: 2.0, alpha: 0.3);

        var student = new Matrix<float>(1, 3);
        student[0, 0] = 1; student[0, 1] = 2; student[0, 2] = 3;

        var teacher = new Matrix<float>(1, 3);
        teacher[0, 0] = 3; teacher[0, 1] = 2; teacher[0, 2] = 1;

        var labels = new Matrix<float>(1, 3);
        labels[0, 0] = 0; labels[0, 1] = 0; labels[0, 2] = 1;

        float result = loss.ComputeLoss(student, teacher, labels);

        Assert.True(float.IsFinite(result), $"Float loss should be finite: {result}");
        Assert.True(result > 0, $"Loss should be positive: {result}");
    }

    #endregion
}
