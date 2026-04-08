using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.AdvancedRL;
using AiDotNet.ReinforcementLearning.Agents.Bandits;
using AiDotNet.ReinforcementLearning.Agents.DoubleQLearning;
using AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;
using AiDotNet.ReinforcementLearning.Agents.EligibilityTraces;
using AiDotNet.ReinforcementLearning.Agents.ExpectedSARSA;
using AiDotNet.ReinforcementLearning.Agents.MonteCarlo;
using AiDotNet.ReinforcementLearning.Agents.NStepQLearning;
using AiDotNet.ReinforcementLearning.Agents.NStepSARSA;
using AiDotNet.ReinforcementLearning.Agents.Planning;
using AiDotNet.ReinforcementLearning.Agents.SARSA;
using AiDotNet.ReinforcementLearning.Agents.TabularQLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class ClassicAgentsIntegrationTests
{
    private const int StateSize = 2;
    private const int ActionSize = 3;
    private const int FeatureSize = 2;
    private const double LearningRate = 0.1;
    private const double DiscountFactor = 0.9;

    [Fact]
    public void BanditAgents_RunBasicWorkflow()
    {
        ExerciseAgent(
            "EpsilonGreedyBanditAgent",
            new EpsilonGreedyBanditAgent<double>(new EpsilonGreedyBanditOptions<double>
            {
                NumArms = ActionSize,
                Epsilon = 0.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            stateSize: 1,
            actionSize: ActionSize,
            done: true);

        ExerciseAgent(
            "GradientBanditAgent",
            new GradientBanditAgent<double>(new GradientBanditOptions<double>
            {
                NumArms = ActionSize,
                Alpha = 0.1,
                UseBaseline = true,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            stateSize: 1,
            actionSize: ActionSize,
            done: true);

        ExerciseAgent(
            "UCBBanditAgent",
            new UCBBanditAgent<double>(new UCBBanditOptions<double>
            {
                NumArms = ActionSize,
                ExplorationParameter = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            stateSize: 1,
            actionSize: ActionSize,
            done: true);

        ExerciseAgent(
            "ThompsonSamplingAgent",
            new ThompsonSamplingAgent<double>(new ThompsonSamplingOptions<double>
            {
                NumArms = ActionSize,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            stateSize: 1,
            actionSize: ActionSize,
            done: true);
    }

    [Fact]
    public void TabularAgents_RunBasicWorkflow()
    {
        ExerciseAgent(
            "SARSAAgent",
            new SARSAAgent<double>(new SARSAOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "ExpectedSARSAAgent",
            new ExpectedSARSAAgent<double>(new ExpectedSARSAOptions<double>(StateSize, ActionSize)
            {
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "NStepSARSAAgent",
            new NStepSARSAAgent<double>(new NStepSARSAOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                NSteps = 2,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "SARSALambdaAgent",
            new SARSALambdaAgent<double>(new SARSALambdaOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                Lambda = 0.5,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "TabularQLearningAgent",
            new TabularQLearningAgent<double>(new TabularQLearningOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "NStepQLearningAgent",
            new NStepQLearningAgent<double>(new NStepQLearningOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                NSteps = 2,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "QLambdaAgent",
            new QLambdaAgent<double>(new QLambdaOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                Lambda = 0.5,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "WatkinsQLambdaAgent",
            new WatkinsQLambdaAgent<double>(new WatkinsQLambdaOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                Lambda = 0.5,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "DoubleQLearningAgent",
            new DoubleQLearningAgent<double>(new DoubleQLearningOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "DynaQAgent",
            new DynaQAgent<double>(new DynaQOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                PlanningSteps = 1,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "DynaQPlusAgent",
            new DynaQPlusAgent<double>(new DynaQPlusOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                PlanningSteps = 1,
                Kappa = 0.0,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "PrioritizedSweepingAgent",
            new PrioritizedSweepingAgent<double>(new PrioritizedSweepingOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                PlanningSteps = 1,
                PriorityThreshold = 0.0,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);
    }

    [Fact]
    public void MonteCarloAgents_RunBasicWorkflow()
    {
        ExerciseAgent(
            "OnPolicyMonteCarloAgent",
            new OnPolicyMonteCarloAgent<double>(new OnPolicyMonteCarloOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "OffPolicyMonteCarloAgent",
            new OffPolicyMonteCarloAgent<double>(new OffPolicyMonteCarloOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                BehaviorEpsilon = 0.0,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "FirstVisitMonteCarloAgent",
            new FirstVisitMonteCarloAgent<double>(new MonteCarloOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "EveryVisitMonteCarloAgent",
            new EveryVisitMonteCarloAgent<double>(new MonteCarloOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "MonteCarloExploringStartsAgent",
            new MonteCarloExploringStartsAgent<double>(new MonteCarloExploringStartsOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);
    }

    [Fact]
    public void AdvancedClassicAgents_RunBasicWorkflow()
    {
        ExerciseAgent(
            "LinearQLearningAgent",
            new LinearQLearningAgent<double>(new LinearQLearningOptions<double>
            {
                FeatureSize = FeatureSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            FeatureSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "LinearSARSAAgent",
            new LinearSARSAAgent<double>(new LinearSARSAOptions<double>
            {
                FeatureSize = FeatureSize,
                ActionSize = ActionSize,
                EpsilonStart = 0.0,
                EpsilonEnd = 0.0,
                EpsilonDecay = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            FeatureSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "LSTDAgent",
            new LSTDAgent<double>(new LSTDOptions<double>
            {
                FeatureSize = FeatureSize,
                ActionSize = ActionSize,
                RegularizationParam = 0.01,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            FeatureSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "LSPIAgent",
            new LSPIAgent<double>(new LSPIOptions<double>
            {
                FeatureSize = FeatureSize,
                ActionSize = ActionSize,
                MaxIterations = 1,
                ConvergenceThreshold = 1.0,
                RegularizationParam = 0.01,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            FeatureSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "TabularActorCriticAgent",
            new TabularActorCriticAgent<double>(new TabularActorCriticOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                ActorLearningRate = 0.05,
                CriticLearningRate = 0.1,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);
    }

    [Fact]
    public void DynamicProgrammingAgents_RunBasicWorkflow()
    {
        ExerciseAgent(
            "ValueIterationAgent",
            new ValueIterationAgent<double>(new ValueIterationOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                MaxIterations = 1,
                Theta = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "PolicyIterationAgent",
            new PolicyIterationAgent<double>(new PolicyIterationOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                MaxEvaluationIterations = 1,
                Theta = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);

        ExerciseAgent(
            "ModifiedPolicyIterationAgent",
            new ModifiedPolicyIterationAgent<double>(new ModifiedPolicyIterationOptions<double>
            {
                StateSize = StateSize,
                ActionSize = ActionSize,
                MaxEvaluationSweeps = 1,
                Theta = 1.0,
                LearningRate = LearningRate,
                DiscountFactor = DiscountFactor,
                LossFunction = CreateLoss()
            }),
            StateSize,
            ActionSize,
            done: true);
    }

    private static void ExerciseAgent(
        string name,
        ReinforcementLearningAgentBase<double> agent,
        int stateSize,
        int actionSize,
        bool done)
    {
        var state = CreateState(stateSize, 0.1);
        var nextState = CreateState(stateSize, 0.2);
        var action = agent.SelectAction(state, training: true);

        AssertOneHot(action, actionSize, name);

        agent.StoreExperience(state, action, 1.0, nextState, done);

        var loss = agent.Train();
        Assert.False(double.IsNaN(loss), $"{name} Train returned NaN.");

        var metrics = agent.GetMetrics();
        Assert.NotNull(metrics);

        var data = agent.Serialize();
        agent.Deserialize(data);

        var clone = agent.Clone();
        Assert.NotNull(clone);
    }

    private static Vector<double> CreateState(int size, double start)
    {
        var state = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            state[i] = start + i * 0.05;
        }

        return state;
    }

    private static void AssertOneHot(Vector<double> action, int actionSize, string name)
    {
        Assert.Equal(actionSize, action.Length);

        int nonZeroCount = 0;
        for (int i = 0; i < action.Length; i++)
        {
            if (action[i] > 0.0)
            {
                nonZeroCount++;
            }
        }

        Assert.True(nonZeroCount == 1, $"{name} expected a one-hot action.");
    }

    private static MeanSquaredErrorLoss<double> CreateLoss()
    {
        return new MeanSquaredErrorLoss<double>();
    }
}
