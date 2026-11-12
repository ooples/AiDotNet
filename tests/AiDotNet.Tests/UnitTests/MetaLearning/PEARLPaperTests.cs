using System;
using System.Reflection;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
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

    [Fact]
    public void Posterior_StaysFiniteInHalfWhenTheFactorPrecisionsOverflow()
    {
        // Every factor has log s_n^2 clamped to -10, so 1 / s_n^2 = e^10 ~ 22026 and four of them sum past Half's
        // 65504: a direct precision sum gives log s^2 = -log(inf) = -inf. The product of four identical factors
        // is N(mu, s^2 / 4), so log s^2 = -10 - ln 4 exactly.
        var algorithm = new PEARLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new PEARLOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 3))
            {
                LatentDim = 1, EncoderHiddenDim = 3, Seed = 21,
            });
        // w1 = 1 (hidden = 3 per unit for an all-ones context), wMu = 0, wLogVar = -2 (pre-clamp log s^2 = -18).
        var encoder = new Half[15];
        for (int i = 0; i < 9; i++) encoder[i] = (Half)1.0f;
        for (int i = 12; i < 15; i++) encoder[i] = (Half)(-2.0f);
        var field = typeof(PEARLAlgorithm<Half, Tensor<Half>, Tensor<Half>>)
            .GetField("_encoderParams", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        field.SetValue(algorithm, new Vector<Half>(encoder));

        var ones = new Half[12];
        for (int i = 0; i < ones.Length; i++) ones[i] = (Half)1.0f;
        var (mu, logVar) = algorithm.Encode(Tensor<Half>.FromVector(new Vector<Half>(ones), new[] { 3, 4 }));

        Assert.Equal(0.0, (double)(float)mu[0], 6);
        Assert.Equal(-10.0 - Math.Log(4.0), (double)(float)logVar[0], 1);
    }
}
