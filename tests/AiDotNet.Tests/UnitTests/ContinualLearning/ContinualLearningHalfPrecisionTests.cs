using System;
using System.Reflection;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Half-precision references for the continual-learning reductions that master computed in double.
/// Half rounds 1e-10 to zero and overflows past 65504, so a floor or a square taken directly in T
/// turns these finite master results into infinities.
/// </summary>
public class ContinualLearningHalfPrecisionTests
{
    private static T InvokeDistillation<T>(string method, double[] teacherLogits, double[] studentLogits, Func<double, T> from)
    {
        var lwf = new LearningWithoutForgetting<T, Tensor<T>, Tensor<T>>(new MeanSquaredErrorLoss<T>(), distillationTemperature: 1.0);
        var info = typeof(LearningWithoutForgetting<T, Tensor<T>, Tensor<T>>)
            .GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(info);
        var teacher = new Vector<T>(Array.ConvertAll(teacherLogits, v => from(v)));
        var student = new Vector<T>(Array.ConvertAll(studentLogits, v => from(v)));
        object? result = info?.Invoke(lwf, new object[] { teacher, student });
        Assert.IsType<T>(result);
        return result is T value ? value : from(double.NaN);
    }

    [Fact]
    public void KLDivergence_Half_StudentProbabilityUnderflow_UsesTheFloor()
    {
        // teacher = [0.5, 0.5]; student = [1, e^-100] and e^-100 is 0 in Half.
        // KL = 0.5 (ln 0.5 - ln 1) + 0.5 (ln 0.5 - ln 1e-10) = 10.8198 (master, in double).
        double expected = 0.5 * Math.Log(0.5) + 0.5 * (Math.Log(0.5) - Math.Log(1e-10));
        Half kl = InvokeDistillation<Half>("ComputeKLDivergence", new[] { 0.0, 0.0 }, new[] { 0.0, -100.0 }, v => (Half)v);

        Assert.InRange((double)kl, expected - 0.05, expected + 0.05);
    }

    [Fact]
    public void SoftCrossEntropy_Half_StudentProbabilityUnderflow_UsesTheFloor()
    {
        // -(0.5 ln 1 + 0.5 ln 1e-10) = 11.5129
        double expected = -0.5 * Math.Log(1e-10);
        Half ce = InvokeDistillation<Half>("ComputeSoftCrossEntropy", new[] { 0.0, 0.0 }, new[] { 0.0, -100.0 }, v => (Half)v);

        Assert.InRange((double)ce, expected - 0.05, expected + 0.05);
    }

    [Fact]
    public void KLDivergence_Double_MatchesReference()
    {
        double[] teacherLogits = { 1.0, -0.5, 2.0 };
        double[] studentLogits = { 0.25, 0.75, -1.0 };
        double[] t = Softmax(teacherLogits);
        double[] s = Softmax(studentLogits);
        double expected = 0;
        for (int i = 0; i < t.Length; i++)
        {
            expected += t[i] * Math.Log(t[i] / Math.Max(s[i], 1e-10));
        }

        double kl = InvokeDistillation<double>("ComputeKLDivergence", teacherLogits, studentLogits, v => v);

        Assert.Equal(expected, kl, 10);
    }

    [Fact]
    public void StableL2Norm_Half_DoesNotOverflowOnLargeComponents()
    {
        // 300^2 + 400^2 = 250000 overflows Half; the norm itself, 500, does not.
        var v = new Vector<Half>(new[] { (Half)300.0f, (Half)(-400.0f) });

        Half norm = MemoryAwareSynapses<Half, Tensor<Half>, Tensor<Half>>.StableL2Norm(v);

        Assert.Equal(500.0, (double)norm, 1);
    }

    [Fact]
    public void StableL2Norm_Double_MatchesEuclideanNorm()
    {
        var v = new Vector<double>(new[] { 3.0, -4.0, 12.0 });

        Assert.Equal(13.0, MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>.StableL2Norm(v), 12);
        Assert.Equal(0.0, MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>.StableL2Norm(new Vector<double>(3)), 12);
    }

    [Fact]
    public void ImportanceMeanMetrics_Half_AccumulateBeyondHalfRange()
    {
        // 70000 ones: master's double mean is exactly 1; a Half sum stops growing at 2048.
        var ones = new Vector<Half>(70000);
        ones.Fill((Half)1.0f);
        var loss = new MeanSquaredErrorLoss<Half>();
        object[] strategies =
        {
            new ElasticWeightConsolidation<Half, Tensor<Half>, Tensor<Half>>(loss),
            new ExpectedGradientLength<Half, Tensor<Half>, Tensor<Half>>(loss),
            new MemoryAwareSynapses<Half, Tensor<Half>, Tensor<Half>>(loss),
            new SynapticIntelligence<Half, Tensor<Half>, Tensor<Half>>(loss),
        };

        foreach (var strategy in strategies)
        {
            var method = strategy.GetType().GetMethod("ComputeMean", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(method);
            Assert.Equal(1.0, (double)(method?.Invoke(strategy, new object[] { ones }) ?? double.NaN), 12);
            Assert.True(double.IsNaN((double)(method?.Invoke(strategy, new object[] { new Vector<Half>(0) }) ?? 0.0)),
                $"{strategy.GetType().Name}: the mean of no values is 0 / 0, as in master");
        }
    }

    private static double[] Softmax(double[] logits)
    {
        double max = double.NegativeInfinity;
        foreach (double v in logits)
        {
            max = Math.Max(max, v);
        }

        var result = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = Math.Exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }
}
