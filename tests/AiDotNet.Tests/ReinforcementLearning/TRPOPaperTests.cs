using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.TRPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ReinforcementLearning;

/// <summary>
/// Schulman et al. 2015, Trust Region Policy Optimization: the single-path update with a conjugate-gradient natural
/// gradient direction scaled to the KL trust region δ and a line search (sections 5-6, appendix C), over a Gaussian
/// policy with state-independent log standard deviations or a categorical policy (appendix D) (#2397).
/// </summary>
public class TRPOPaperTests
{
    private static Vector<double> V(params double[] values) => new(values);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new TRPOOptions<double>();
        Assert.Equal(0.01, o.MaxKL);
        Assert.Equal(1.0, o.GaeLambda);
        Assert.Equal(new[] { 30 }, o.PolicyHiddenLayers);
        Assert.Equal(10, o.ConjugateGradientIterations);
    }

    [Fact(Timeout = 60000)]
    public async Task GaussianPolicy_HasStateIndependentLogStandardDeviations()
    {
        await Task.Yield();
        var options = new TRPOOptions<double> { StateSize = 3, ActionSize = 2, IsContinuous = true, PolicyHiddenLayers = new List<int> { 5 }, Seed = 1 };
        var agent = new TRPOAgent<double>(options);
        // Policy 3 -> 5 -> 2 (means only), the value baseline 3 -> 64 -> 64 -> 1, and two log standard deviations.
        int policy = 3 * 5 + 5 + 5 * 2 + 2, value = 3 * 64 + 64 + 64 * 64 + 64 + 64 + 1;
        Assert.Equal(policy + value + 2, agent.GetParameters().Length);
        Assert.Equal(policy + 2, agent.PolicyParameterCount);
        // The log standard deviations start at zero (unit standard deviation).
        Assert.Equal(new[] { 0.0, 0.0 }, new[] { agent.LogStandardDeviations[0], agent.LogStandardDeviations[1] });
        // The deterministic action is the mean, whatever the state's noise.
        var greedy = agent.SelectAction(V(0.1, 0.2, 0.3), training: false);
        Assert.Equal(greedy[0], agent.SelectAction(V(0.1, 0.2, 0.3), training: false)[0]);
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FisherVectorProduct_IsTheKlHessian(bool continuous)
    {
        await Task.Yield();
        // Appendix C.1: Fv = JᵀMJv, so 1/2 ε² vᵀFv must be the mean KL divergence of a small step εv.
        var agent = new TRPOAgent<double>(new TRPOOptions<double>
        {
            StateSize = 2, ActionSize = 3, IsContinuous = continuous, PolicyHiddenLayers = new List<int> { 8 }, Seed = 2,
        });
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var states = new Tensor<double>(new[] { 16, 2 });
        for (int i = 0; i < states.Length; i++) states[i] = rng.NextDouble() - 0.5;
        var direction = Enumerable.Range(0, agent.PolicyParameterCount).Select(_ => rng.NextDouble() - 0.5).ToArray();
        var (quadratic, measured) = agent.FisherCheck(states, direction, 1e-3);
        Assert.True(quadratic > 0);
        Assert.Equal(1.0, measured / quadratic, 2);
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Step_ImprovesTheSurrogateInsideTheTrustRegion(bool continuous)
    {
        await Task.Yield();
        const double delta = 0.01;
        var options = new TRPOOptions<double>
        {
            StateSize = 2, ActionSize = 3, IsContinuous = continuous, PolicyHiddenLayers = new List<int> { 8 },
            MaxKL = delta, StepsPerUpdate = 1000, Seed = 2,
        };
        var agent = new TRPOAgent<double>(options);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        for (int i = 0; i < 64; i++)
        {
            var state = V(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5);
            var action = agent.SelectAction(state, training: true);
            agent.StoreExperience(state, action, rng.NextDouble(), state, done: true);
        }
        double improvement = agent.Train();
        double kl = agent.GetMetrics()["kl"];
        Assert.True(improvement > 0, $"surrogate improvement {improvement}");
        // The line search accepts only steps that satisfy D_KL <= δ; the natural gradient step uses a good part of it.
        Assert.InRange(kl, 0.05 * delta, delta);
    }

    [Fact(Timeout = 300000)]
    public async Task Categorical_LearnsAContextualBandit()
    {
        await Task.Yield();
        // One-step episodes: the rewarded action is the index of the largest state component.
        var options = new TRPOOptions<double>
        {
            StateSize = 3, ActionSize = 3, PolicyHiddenLayers = new List<int> { 16 }, StepsPerUpdate = 200,
            MaxKL = 0.05, Seed = 4,
        };
        var agent = new TRPOAgent<double>(options);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        Vector<double> State() => V(rng.NextDouble(), rng.NextDouble(), rng.NextDouble());
        int Best(Vector<double> s) => s[0] >= s[1] && s[0] >= s[2] ? 0 : s[1] >= s[2] ? 1 : 2;
        int Taken(Vector<double> a) => Enumerable.Range(0, 3).OrderByDescending(k => a[k]).First();
        double Accuracy() => Enumerable.Range(0, 200).Select(_ => State()).Average(s => Taken(agent.SelectAction(s, training: false)) == Best(s) ? 1.0 : 0.0);

        double before = Accuracy();
        for (int i = 0; i < 200 * 40; i++)
        {
            var s = State();
            var a = agent.SelectAction(s, training: true);
            agent.StoreExperience(s, a, Taken(a) == Best(s) ? 1.0 : 0.0, s, done: true);
        }
        double after = Accuracy();
        Assert.True(after > 0.85 && after > before + 0.2, $"greedy accuracy {before:F2} -> {after:F2}");
    }

    [Fact(Timeout = 300000)]
    public async Task Gaussian_LearnsAContinuousBandit()
    {
        await Task.Yield();
        // One-step episodes rewarding -(a - 2s)^2: the mean must track 2s and the standard deviation shrink.
        var options = new TRPOOptions<double>
        {
            StateSize = 1, ActionSize = 1, IsContinuous = true, PolicyHiddenLayers = new List<int> { 16 },
            StepsPerUpdate = 200, MaxKL = 0.05, Seed = 6,
        };
        var agent = new TRPOAgent<double>(options);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        double Error() => Enumerable.Range(0, 100).Select(_ => rng.NextDouble() - 0.5)
            .Average(s => Math.Abs(agent.SelectAction(V(s), training: false)[0] - 2 * s));

        double before = Error();
        for (int i = 0; i < 200 * 40; i++)
        {
            double s = rng.NextDouble() - 0.5;
            var a = agent.SelectAction(V(s), training: true);
            agent.StoreExperience(V(s), a, -(a[0] - 2 * s) * (a[0] - 2 * s), V(s), done: true);
        }
        double after = Error();
        Assert.True(after < 0.5 * before && after < 0.25, $"mean error {before:F3} -> {after:F3}");
        Assert.True(agent.LogStandardDeviations[0] < 0, $"log std {agent.LogStandardDeviations[0]}");
    }
}
