using System;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="PEARLAlgorithm{T, TInput, TOutput}"/> against Rakelly et al., "Efficient Off-Policy
/// Meta-Reinforcement Learning via Probabilistic Context Variables" (ICML 2019): the posterior is a product of
/// per-example Gaussian factors (Eq. 4) and tasks are inferred without gradient steps.
/// </summary>
public class PEARLPaperTests
{
    private static PEARLAlgorithm<double, Matrix<double>, Vector<double>> Algorithm()
        => new(new PEARLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
        {
            LatentDim = 2, EncoderHiddenDim = 3, Seed = 21,
        });

    [Fact]
    public void Defaults_InferWithoutGradientSteps()
    {
        Assert.Equal(0, new PEARLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3)).AdaptationSteps);
    }

    [Fact]
    public void Posterior_IsTheProductOfPerExampleFactors()
    {
        // Two identical factors N(mu, s^2) multiply to N(mu, s^2 / 2): same mean, log variance lower by ln 2.
        var algorithm = Algorithm();
        var column = new Vector<double>(new[] { 0.4, -0.9, 1.3 });
        var one = Tensor<double>.FromVector(column, new[] { 3, 1 });
        var two = Tensor<double>.FromVector(new Vector<double>(new[] { 0.4, 0.4, -0.9, -0.9, 1.3, 1.3 }), new[] { 3, 2 });

        var (mu1, logVar1) = algorithm.Encode(one);
        var (mu2, logVar2) = algorithm.Encode(two);

        for (int l = 0; l < 2; l++)
        {
            Assert.Equal(mu1[l], mu2[l], 10);
            Assert.Equal(logVar1[l] - Math.Log(2.0), logVar2[l], 10);
        }
    }

    [Fact]
    public void Posterior_WeightsFactorsByPrecision()
    {
        // mu = (sum_n mu_n / s_n^2) / (sum_n 1 / s_n^2): check it against the single-factor posteriors.
        var algorithm = Algorithm();
        var a = new Vector<double>(new[] { 0.4, -0.9, 1.3 });
        var b = new Vector<double>(new[] { -1.1, 0.2, 0.7 });
        var (muA, lvA) = algorithm.Encode(Tensor<double>.FromVector(a, new[] { 3, 1 }));
        var (muB, lvB) = algorithm.Encode(Tensor<double>.FromVector(b, new[] { 3, 1 }));
        var both = Tensor<double>.FromVector(new Vector<double>(new[] { 0.4, -1.1, -0.9, 0.2, 1.3, 0.7 }), new[] { 3, 2 });
        var (mu, lv) = algorithm.Encode(both);

        for (int l = 0; l < 2; l++)
        {
            double pa = Math.Exp(-lvA[l]);
            double pb = Math.Exp(-lvB[l]);
            Assert.Equal((muA[l] * pa + muB[l] * pb) / (pa + pb), mu[l], 10);
            Assert.Equal(-Math.Log(pa + pb), lv[l], 10);
        }
    }
}
