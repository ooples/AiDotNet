using System;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// <see cref="SDCLAlgorithm{T, TInput, TOutput}"/> distills with Hinton's temperature loss
/// tau^2 KL(softmax(t / tau) || softmax(s / tau)); the inner loop backpropagates it instead of a one-sample SPSA guess.
/// </summary>
public class SDCLDistillationTests
{
    [Fact]
    public void DistillationLoss_IsTemperatureScaledKL()
    {
        const double tau = 2.0;
        var algorithm = new SDCLAlgorithm<double, Matrix<double>, Vector<double>>(
            new SDCLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2)) { DistillationTemperature = tau });
        var teacher = new Vector<double>(new[] { 1.0, -0.5, 2.0 });
        var student = new Vector<double>(new[] { 0.2, 0.3, 1.1 });

        double[] Softmax(Vector<double> v)
        {
            var e = new double[v.Length];
            double sum = 0;
            for (int i = 0; i < v.Length; i++) { e[i] = Math.Exp(v[i] / tau); sum += e[i]; }
            for (int i = 0; i < v.Length; i++) e[i] /= sum;
            return e;
        }

        var p = Softmax(teacher);
        var q = Softmax(student);
        double expected = 0;
        for (int i = 0; i < 3; i++) expected += p[i] * Math.Log(p[i] / (q[i] + 1e-10));
        expected *= tau * tau;

        Assert.Equal(expected, algorithm.ComputeKLDivergence(teacher, student), 9);
        Assert.Equal(0.0, algorithm.ComputeKLDivergence(teacher, teacher), 6);
    }

    [Fact]
    public void DistillationLoss_OfABatch_IsTheMeanOfThePerRowLosses()
    {
        // Two examples of three classes: tau^2 KL(softmax(t_n / tau) || softmax(s_n / tau)) per row, averaged over the
        // rows (batchmean). The query term used to flatten the [2, 3] outputs into ONE softmax over all six entries,
        // and the inner-loop loss summed the rows instead of averaging them.
        const double tau = 2.0;
        var algorithm = new SDCLAlgorithm<double, Tensor<double>, Tensor<double>>(
            new SDCLOptions<double, Tensor<double>, Tensor<double>>(new MockFullModel<double>()) { DistillationTemperature = tau });
        double[] t0 = { 1.0, -0.5, 2.0 }, s0 = { 0.2, 0.3, 1.1 };
        double[] t1 = { 0.0, 0.0, 0.0 }, s1 = { 3.0, -1.0, 0.0 };

        double RowLoss(double[] t, double[] s)
        {
            double[] Softmax(double[] v)
            {
                var e = new double[v.Length];
                double sum = 0;
                for (int i = 0; i < v.Length; i++) { e[i] = Math.Exp(v[i] / tau); sum += e[i]; }
                for (int i = 0; i < v.Length; i++) e[i] /= sum;
                return e;
            }

            var p = Softmax(t);
            var q = Softmax(s);
            double kl = 0;
            for (int i = 0; i < p.Length; i++) kl += p[i] * Math.Log(p[i] / (q[i] + 1e-10));
            return tau * tau * kl;
        }

        var teacher = new Tensor<double>(new[] { 1.0, -0.5, 2.0, 0.0, 0.0, 0.0 }, new[] { 2, 3 });
        var student = new Tensor<double>(new[] { 0.2, 0.3, 1.1, 3.0, -1.0, 0.0 }, new[] { 2, 3 });

        double expected = (RowLoss(t0, s0) + RowLoss(t1, s1)) / 2.0;
        Assert.Equal(expected, algorithm.DistillationLoss(teacher, student), 9);
    }

    [Fact]
    public void FallbackDistillationLoss_ScoresFlattenedOutputsPerRow()
    {
        // A model without tape gradients is differentiated by the base fallback, which hands the loss flattened
        // vectors. Read back as [2, 3] rows they must score what the shaped loss does - the per-row batchmean - not
        // one softmax over all six entries.
        const double tau = 2.0;
        var algorithm = new SDCLAlgorithm<double, Tensor<double>, Tensor<double>>(
            new SDCLOptions<double, Tensor<double>, Tensor<double>>(new MockFullModel<double>()) { DistillationTemperature = tau });
        var teacherFlat = new Vector<double>(new[] { 1.0, -0.5, 2.0, 0.0, 0.0, 0.0 });
        var studentFlat = new Vector<double>(new[] { 0.2, 0.3, 1.1, 3.0, -1.0, 0.0 });
        var teacher = Tensor<double>.FromVector(teacherFlat, new[] { 2, 3 });
        var student = Tensor<double>.FromVector(studentFlat, new[] { 2, 3 });

        double shaped = algorithm.DistillationLoss(teacher, student);
        double fallback = algorithm.FallbackDistillationLoss(studentFlat, teacherFlat, rowWidth: 3);
        double oneSoftmax = algorithm.FallbackDistillationLoss(studentFlat, teacherFlat, rowWidth: 6);

        Assert.Equal(shaped, fallback, 12);
        Assert.NotEqual(shaped, oneSoftmax, 6);
    }

    [Fact]
    public void DistillationLoss_IgnoresAnUnderflowedTeacherClassInHalf()
    {
        // Teacher logits / tau = (0, -50): p_T = (1, e^-50), and e^-50 < 1e-10, so that class is masked out and
        // the loss is tau^2 * 1 * ln(1 / 0.5) = 16 ln 2 for a uniform student. In Half the 1e-10 floor rounds to 0:
        // log(p_T) = -inf for the underflowed class and the masked term 0 * -inf poisoned the sum with NaN.
        var algorithm = new SDCLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new SDCLOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>()) { DistillationTemperature = 4.0 });
        var teacher = new Vector<Half>(new[] { (Half)0.0f, (Half)(-200.0f) });
        var student = new Vector<Half>(new[] { (Half)0.0f, (Half)0.0f });

        double loss = (double)(float)algorithm.ComputeKLDivergence(teacher, student);

        Assert.Equal(16.0 * Math.Log(2.0), loss, 1);
    }
}
