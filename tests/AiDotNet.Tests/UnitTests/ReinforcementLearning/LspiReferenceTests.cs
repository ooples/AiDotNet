using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.AdvancedRL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// LSPI reference values (Lagoudakis &amp; Parr 2003, Figures 5 and 8): LSTDQ solves ONE system over the joint
/// state-action features, A = sum phi(s,a) (phi(s,a) - gamma phi(s', pi(s')))^T, b = sum phi(s,a) r, and LSPI
/// repeats it with the greedy policy until the weights stop changing.
/// </summary>
/// <remarks>
/// A 2-state, 2-action chain with one-hot state features (s0 = [1, 0], s1 = [0, 1]), gamma = 0.5, no ridge,
/// one sample per pair: (s0, a0) r 0 -> s1; (s0, a1) r 0.5 -> s0; (s1, a0) r 0 -> s0; (s1, a1) r 2 -> s1.
/// With tabular features each LSTDQ solve is exact policy evaluation, Q(s,a) = r + gamma Q(s', pi(s')).
/// <para>Iteration 1, pi = a0 everywhere (all-zero weights tie, first arg-max): Q(s0,a0) = 0.5 Q(s1,a0) and
/// Q(s1,a0) = 0.5 Q(s0,a0) give 0, 0; Q(s0,a1) = 0.5 + 0.5 * 0 = 0.5; Q(s1,a1) = 2 + 0.5 * 0 = 2.
/// Weight rows (action-major): a0 = [0, 0], a1 = [0.5, 2].</para>
/// <para>A per-action solve that bootstraps action a's block off phi(s') in the same block instead gives
/// a1 = [1, 4] here (A = 0.5 I), the deviation this test pins down.</para>
/// <para>Iteration 2, pi = (a1, a1): Q(s0,a1) = 0.5 + 0.5 Q(s0,a1) = 1, Q(s1,a1) = 2 + 0.5 Q(s1,a1) = 4,
/// Q(s0,a0) = 0.5 * 4 = 2, Q(s1,a0) = 0.5 * 1 = 0.5. Iteration 3, pi = (a0, a1): Q(s1,a1) = 4, Q(s0,a0) = 2,
/// Q(s0,a1) = 0.5 + 0.5 * 2 = 1.5, Q(s1,a0) = 0.5 * 2 = 1 = Q*. Iteration 4 keeps that policy, so w is unchanged
/// and LSPI stops: a0 = [2, 1], a1 = [1.5, 4].</para>
/// </remarks>
public class LspiReferenceTests
{
    private const double Tolerance = 1e-9;

    private static Vector<double> Vec(params double[] values) => new Vector<double>(values);

    private static LSPIAgent<double> CreateChainAgent(int maxIterations)
    {
        var agent = new LSPIAgent<double>(new LSPIOptions<double>
        {
            FeatureSize = 2,
            ActionSize = 2,
            MaxIterations = maxIterations,
            ConvergenceThreshold = 1e-9,
            RegularizationParam = 0.0,
            DiscountFactor = 0.5,
            LossFunction = new MeanSquaredErrorLoss<double>()
        });

        var s0 = Vec(1.0, 0.0);
        var s1 = Vec(0.0, 1.0);
        var a0 = Vec(1.0, 0.0);
        var a1 = Vec(0.0, 1.0);
        agent.StoreExperience(s0, a0, 0.0, s1, false);
        agent.StoreExperience(s0, a1, 0.5, s0, false);
        agent.StoreExperience(s1, a0, 0.0, s0, false);
        agent.StoreExperience(s1, a1, 2.0, s1, false);
        return agent;
    }

    private static void AssertWeights(LSPIAgent<double> agent, params double[] expected)
    {
        var weights = agent.GetParameters().ToArray();
        Assert.Equal(expected.Length, weights.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], weights[i], Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async System.Threading.Tasks.Task FirstLstdqSolve_EvaluatesTheGreedyPolicyJointly()
    {
        await System.Threading.Tasks.Task.Yield();
        var agent = CreateChainAgent(maxIterations: 1);

        agent.Train();

        AssertWeights(agent, 0.0, 0.0, 0.5, 2.0);
    }

    [Fact(Timeout = 60000)]
    public async System.Threading.Tasks.Task PolicyIteration_ConvergesToOptimalQInFourIterations()
    {
        await System.Threading.Tasks.Task.Yield();
        var agent = CreateChainAgent(maxIterations: 20);

        agent.Train();

        AssertWeights(agent, 2.0, 1.0, 1.5, 4.0);
        Assert.Equal(4.0, agent.GetMetrics()["iterations"], Tolerance);
        Assert.Equal(0, ArgMaxOf(agent.Predict(Vec(1.0, 0.0))));
        Assert.Equal(1, ArgMaxOf(agent.Predict(Vec(0.0, 1.0))));
    }

    private static int ArgMaxOf(Vector<double> oneHot) => oneHot[0] >= oneHot[1] ? 0 : 1;
}
