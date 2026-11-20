using System.Linq;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.MonteCarlo;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// First-visit Monte Carlo reference values (Sutton &amp; Barto 2018, Figures 5.1, 5.4 and 5.6:
/// "Unless the pair S_t, A_t appears in S_0, A_0, ..., S_{t-1}, A_{t-1}: append G to Returns").
/// </summary>
/// <remarks>
/// The episode visits (s0, a0) at t = 0 and again at t = 2, with rewards 1, 0, 0 and gamma = 0.9.
/// First-visit MC averages only the return after the EARLIEST visit: G_0 = 1 + 0.9 * 0 + 0.81 * 0 = 1.
/// A backward scan that skips pairs it has already seen keeps the LATEST visit instead (G_2 = 0), so
/// every Q-value would stay 0. The pair (s1, a1) at t = 1 has return G_1 = 0.
/// </remarks>
public class FirstVisitMonteCarloReferenceTests
{
    private const int StateSize = 2;
    private const int ActionSize = 2;

    private static Vector<double> Vec(params double[] values) => new Vector<double>(values);

    private static void RunRevisitingEpisode(ReinforcementLearningAgentBase<double> agent)
    {
        var s0 = Vec(0.0, 0.0);
        var s1 = Vec(1.0, 0.0);
        var a0 = Vec(1.0, 0.0);
        var a1 = Vec(0.0, 1.0);

        agent.StoreExperience(s0, a0, 1.0, s1, false);
        agent.StoreExperience(s1, a1, 0.0, s0, false);
        agent.StoreExperience(s0, a0, 0.0, s1, true);
    }

    private static void AssertFirstVisitValues(ReinforcementLearningAgentBase<double> agent)
    {
        var q = agent.GetParameters().ToArray();

        // Q(s0, a0) = G_0 = 1; every other entry (including Q(s1, a1) = G_1 = 0) is 0.
        Assert.Equal(1.0, q.Max(), 12);
        Assert.Equal(1.0, q.Sum(), 12);
        Assert.Equal(1, q.Count(v => v != 0.0));
    }

    [Fact]
    public void FirstVisitMonteCarlo_UsesReturnAfterEarliestVisit()
    {
        var agent = new FirstVisitMonteCarloAgent<double>(new MonteCarloOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            EpsilonStart = 0.0,
            EpsilonEnd = 0.0,
            EpsilonDecay = 1.0,
            LearningRate = 0.1,
            DiscountFactor = 0.9,
            LossFunction = new MeanSquaredErrorLoss<double>()
        });

        RunRevisitingEpisode(agent);
        AssertFirstVisitValues(agent);
    }

    [Fact]
    public void MonteCarloExploringStarts_UsesReturnAfterEarliestVisit()
    {
        var agent = new MonteCarloExploringStartsAgent<double>(new MonteCarloExploringStartsOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            LearningRate = 0.1,
            DiscountFactor = 0.9,
            LossFunction = new MeanSquaredErrorLoss<double>()
        });

        RunRevisitingEpisode(agent);
        AssertFirstVisitValues(agent);
    }

    [Fact]
    public void OnPolicyMonteCarlo_UsesReturnAfterEarliestVisit()
    {
        var agent = new OnPolicyMonteCarloAgent<double>(new OnPolicyMonteCarloOptions<double>
        {
            StateSize = StateSize,
            ActionSize = ActionSize,
            EpsilonStart = 0.0,
            EpsilonEnd = 0.0,
            EpsilonDecay = 1.0,
            LearningRate = 0.1,
            DiscountFactor = 0.9,
            LossFunction = new MeanSquaredErrorLoss<double>()
        });

        RunRevisitingEpisode(agent);
        AssertFirstVisitValues(agent);
    }
}
