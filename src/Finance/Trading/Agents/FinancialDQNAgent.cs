using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Financial Deep Q-Network (DQN) agent for discrete action trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The DQN (Deep Q-Network) trading agent learns to make
/// discrete trading decisions (buy, sell, or hold) by estimating the long-term value of
/// each action. It maintains a "memory" of past experiences and learns from random
/// samples of those memories. A separate target network prevents the learning from
/// becoming unstable. DQN is best suited for trading scenarios with a fixed set of
/// possible actions.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define Q-network architecture for discrete trading (10 state features, 3 actions: buy/hold/sell)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 10, outputSize: 3);
///
/// // Create DQN agent for discrete action stock trading
/// var options = new TradingAgentOptions&lt;double&gt;();
/// var model = new FinancialDQNAgent&lt;double&gt;(architecture, options);
///
/// // Parameterless constructor with default 10-feature, 3-action architecture
/// var defaultModel = new FinancialDQNAgent&lt;double&gt;();
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Playing Atari with Deep Reinforcement Learning", "https://arxiv.org/abs/1312.5602", Year = 2013, Authors = "Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller")]
public partial class FinancialDQNAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    #region Fields

    private readonly TradingAgentOptions<T> _options;
    private readonly INeuralNetwork<T> _qNetwork;
    [Buffer]
    private readonly INeuralNetwork<T> _targetNetwork;
    private readonly ReplayBuffer<T> ReplayBuffer;
    private readonly NeuralNetworkArchitecture<T> _architecture;

    /// <summary>Current exploration rate, decayed from EpsilonStart toward EpsilonEnd on every training step.</summary>
    private double _epsilon;



    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialDQNAgent class.
    /// </summary>
    /// <param name="architecture">The user-provided architecture for the Q-network.</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, FinancialDQNAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public FinancialDQNAgent()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 3),
            options: new FinancialDQNAgentOptions<T> { StateSize = 10, ActionSize = 3 })
    {
    }

    public FinancialDQNAgent(NeuralNetworkArchitecture<T> architecture, TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options;
        _architecture = architecture;

        EnsureDefaultLayers(architecture, options.StateSize, options.ActionSize);

        _qNetwork = new NeuralNetwork<T>(architecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetNetwork = new NeuralNetwork<T>(architecture.CloneForModelConstruction(), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
        _epsilon = TradingOptions.EpsilonStart;
        UpdateTargetNetwork();
    }

    /// <summary>
    /// Current exploration rate. Starts at <c>EpsilonStart</c> and decays toward <c>EpsilonEnd</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Epsilon is how often the agent ignores what it has learned and tries something at
    /// random. It should start high (explore) and fall (exploit what you found). Exposed so a training loop can
    /// record the curve and confirm that is actually happening.
    /// </para>
    /// </remarks>
    public double Epsilon => _epsilon;



    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Compares against the CURRENT epsilon, not EpsilonStart.
        //
        // This read TradingOptions.EpsilonStart directly. EpsilonStart defaults to 1.0 and nothing ever
        // decayed it - EpsilonEnd and EpsilonDecay were declared, validated against each other, and read by
        // nobody - so the behaviour policy was 100% uniform random for the entire run. Every "learning curve"
        // it produced was the return of a random policy, and the network's own Q-values were never once acted
        // on during training.
        if (training && RandomHelper.CreateSecureRandom().NextDouble() < _epsilon)
        {
            var action = new Vector<T>(TradingOptions.ActionSize);
            int randomAction = RandomHelper.CreateSecureRandom().Next(TradingOptions.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        var qValues = _qNetwork.Predict(Tensor<T>.FromVector(state));
        int bestAction = 0;
        T maxQ = qValues.Data.Span[0];

        for (int i = 1; i < TradingOptions.ActionSize; i++)
        {
            if (NumOps.GreaterThan(qValues.Data.Span[i], maxQ))
            {
                maxQ = qValues.Data.Span[i];
                bestAction = i;
            }
        }

        var result = new Vector<T>(TradingOptions.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, Train performs a training step. This updates the FinancialDQNAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        // A supervised one-shot Train(state, target) call bypasses the autonomous-exploration batch
        // gate and trains on the samples gathered so far (clamped to the buffer); autonomous stepping
        // still requires a full minibatch before updating.
        int effectiveBatchSize = SupervisedUpdateRequested
            ? System.Math.Min(TradingOptions.BatchSize, ReplayBuffer.Count)
            : TradingOptions.BatchSize;
        if (effectiveBatchSize <= 0 || ReplayBuffer.Count < effectiveBatchSize)
            return NumOps.Zero;

        var batch = ReplayBuffer.Sample(effectiveBatchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;

        // Batched DQN update: run the online and target Q-networks once over the whole minibatch,
        // build the per-row TD target, then do ONE batched backward — instead of an autograd tape
        // per experience (the per-sample loop dominated RL training time — see profiling).
        int stateDim = batch[0].State.Length;
        var gamma = NumOps.FromDouble(Convert.ToDouble(TradingOptions.DiscountFactor));

        var statesData = new T[n * stateDim];
        var nextStatesData = new T[n * stateDim];
        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            for (int j = 0; j < stateDim; j++)
            {
                statesData[i * stateDim + j] = exp.State[j];
                nextStatesData[i * stateDim + j] = exp.NextState[j];
            }
        }

        var states = new Tensor<T>([n, stateDim], new Vector<T>(statesData));
        var nextStates = new Tensor<T>([n, stateDim], new Vector<T>(nextStatesData));

        var currentQ = _qNetwork.Predict(states).ToVector();        // [n * actionCount], row-major
        var nextQ = _targetNetwork.Predict(nextStates).ToVector();  // [n * actionCount]
        int actionCount = currentQ.Length / n;

        // Targets = current Q with the taken-action slot overwritten by reward + gamma * max_a' Q'.
        var expectedData = currentQ.Clone();
        for (int i = 0; i < n; i++)
        {
            T maxNextQ = nextQ[i * actionCount];
            for (int a = 1; a < actionCount; a++)
            {
                var q = nextQ[i * actionCount + a];
                if (NumOps.GreaterThan(q, maxNextQ))
                {
                    maxNextQ = q;
                }
            }

            var exp = batch[i];
            T target = exp.Done
                ? exp.Reward
                : NumOps.Add(exp.Reward, NumOps.Multiply(gamma, maxNextQ));
            expectedData[i * actionCount + GetActionIndex(exp.Action)] = target;
        }

        var expected = new Tensor<T>([n, actionCount], expectedData);
        _qNetwork.Train(states, expected);

        // The inherited counter, which every other agent advances in its own Train() and which the state
        // generator serialises. This agent never advanced it at all, so nothing downstream could tell how much
        // training had happened.
        TrainingSteps++;

        // Decay epsilon toward EpsilonEnd. Multiplicative, matching what EpsilonDecay (0.995) means and what
        // the reference DQNAgent already does.
        _epsilon = Math.Max(TradingOptions.EpsilonEnd, _epsilon * TradingOptions.EpsilonDecay);

        // Sync the target network on a DETERMINISTIC schedule.
        //
        // This was `rng.Next(TargetUpdateFrequency) == 0` - a coin flip with probability 1/N per step, not
        // "every N steps". At the default N = 1000 a 600-step run expects 0.6 syncs, so the target network
        // usually held its initial random weights for the whole run and the TD target was noise. It is also
        // unreproducible: two runs with the same seed synced at different steps.
        if (TrainingSteps % Math.Max(1, TradingOptions.TargetUpdateFrequency) == 0)
        {
            UpdateTargetNetwork();
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Executes UpdateTargetNetwork for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, UpdateTargetNetwork updates internal parameters or state. This keeps the FinancialDQNAgent architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    private void UpdateTargetNetwork()
    {
        _targetNetwork.UpdateParameters(_qNetwork.GetParameters());
    }

    /// <summary>
    /// Executes GetActionIndex for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetActionIndex performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private static int GetActionIndex(Vector<T> action)
    {
        for (int i = 0; i < action.Length; i++)
        {
            if (Math.Abs(Convert.ToDouble(action[i]) - 1.0) < 1e-5)
                return i;
        }
        return 0;
    }

    /// <summary>
    /// Executes GetMaxQ for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetMaxQ performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private T GetMaxQ(Tensor<T> qValues)
    {
        T max = qValues.Data.Span[0];
        for (int i = 1; i < qValues.Length; i++)
        {
            if (NumOps.GreaterThan(qValues.Data.Span[i], max))
                max = qValues.Data.Span[i];
        }
        return max;
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, reward, nextState, done);
        ReplayBuffer.Add(experience);
    }

    #endregion

    #region Serialization

    #endregion

    #region Model Metadata

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialDQN" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _qNetwork.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _qNetwork.ApplyGradients(gradients, learningRate);
        UpdateTargetNetwork();
    }

    #endregion
}
