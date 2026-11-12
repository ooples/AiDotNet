using System;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
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
}
