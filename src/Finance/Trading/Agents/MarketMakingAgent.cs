using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Specialized market making agent using reinforcement learning for optimal quoting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A market making agent learns to provide liquidity by
/// continuously placing buy and sell orders (quotes) in the market. It earns money from
/// the spread between its buy and sell prices while managing the risk of holding inventory.
/// Using reinforcement learning, it learns when to quote aggressively or conservatively
/// based on market conditions, volatility, and its current position.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for market making policy (10 state features, bid/ask offset output)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 10, outputSize: 2);
///
/// // Create market making agent that learns optimal bid/ask quoting
/// var options = new MarketMakingOptions&lt;double&gt;();
/// var model = new MarketMakingAgent&lt;double&gt;(architecture, options);
///
/// // Parameterless constructor with default architecture
/// var defaultModel = new MarketMakingAgent&lt;double&gt;();
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Extending Deep Reinforcement Learning Frameworks in Cryptocurrency Market Making", "https://arxiv.org/abs/2004.06985")]
public partial class MarketMakingAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    #region Fields

    private readonly INeuralNetwork<T> _policyNetwork;
    private readonly MarketMakingOptions<T> _mmOptions;
    private readonly NeuralNetworkArchitecture<T> _architecture;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _mmOptions;

    private readonly ReplayBuffer<T> ReplayBuffer;

    /// <summary>
    /// Action-value critic Q(s, a) over the quoted bid/ask offsets. Without it the agent had no way to tell
    /// a profitable quote from an unprofitable one: the reward never entered any update.
    /// </summary>
    private readonly INeuralNetwork<T> _critic;

    /// <summary>Slow-moving copy of <see cref="_critic"/> supplying the bootstrap value in the TD target.</summary>
    [Buffer]
    private readonly INeuralNetwork<T> _targetCritic;

    /// <summary>Slow-moving copy of the policy, used to choose the next action in the TD target.</summary>
    [Buffer]
    private readonly INeuralNetwork<T> _targetPolicyNetwork;

    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;

    /// <summary>Gradient updates applied so far; reported through the trading metrics.</summary>
    private int _updateCount;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with paper-default options. Delegates to
    /// the <see cref="MarketMakingAgent(MarketMakingOptions{T})"/> overload
    /// which reads <see cref="MarketMakingOptions{T}.StateSize"/> /
    /// <see cref="MarketMakingOptions{T}.ActionSize"/> (inherited from
    /// <c>TradingAgentOptions</c>: StateSize=64, ActionSize=3) and
    /// constructs an architecture matching those dims. Callers that want
    /// to customize either the network architecture or the
    /// market-making hyperparameters do so by passing their own
    /// <see cref="MarketMakingOptions{T}"/> and / or
    /// <see cref="NeuralNetworkArchitecture{T}"/> to one of the explicit
    /// ctors below.
    /// </summary>
    public MarketMakingAgent() : this(new MarketMakingOptions<T>())
    {
    }

    /// <summary>
    /// Initializes a new instance from <see cref="MarketMakingOptions{T}"/>
    /// alone, building the policy-network architecture from
    /// <see cref="MarketMakingOptions{T}.StateSize"/> and
    /// <see cref="MarketMakingOptions{T}.ActionSize"/>. Equivalent to the
    /// (architecture, options) overload with a sensibly-sized
    /// <see cref="NeuralNetworkArchitecture{T}"/>; lets callers customize
    /// trading hyperparameters without having to hand-construct a
    /// matching architecture.
    /// </summary>
    public MarketMakingAgent(MarketMakingOptions<T> options)
        : this(CreateArchitectureFromOptions(options), options)
    {
    }

    /// <summary>
    /// Static factory so the base-constructor initializer can null-check
    /// <paramref name="options"/> before dereferencing
    /// <c>StateSize</c> / <c>ActionSize</c> — a null at the convenience-ctor
    /// path would otherwise NullReferenceException inside the initializer
    /// rather than throw a clear ArgumentNullException.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateArchitectureFromOptions(MarketMakingOptions<T> options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (options.StateSize <= 0)
            throw new ArgumentOutOfRangeException(
                $"{nameof(options)}.{nameof(MarketMakingOptions<T>.StateSize)}",
                options.StateSize,
                $"{nameof(MarketMakingOptions<T>.StateSize)} must be > 0 (was {options.StateSize}).");
        if (options.ActionSize <= 0)
            throw new ArgumentOutOfRangeException(
                $"{nameof(options)}.{nameof(MarketMakingOptions<T>.ActionSize)}",
                options.ActionSize,
                $"{nameof(MarketMakingOptions<T>.ActionSize)} must be > 0 (was {options.ActionSize}).");
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: options.StateSize,
            outputSize: options.ActionSize);
    }

    public MarketMakingAgent(NeuralNetworkArchitecture<T> architecture, MarketMakingOptions<T> options)
        : base(options)
    {
        _mmOptions = options;
        _architecture = architecture;
        EnsureMarketMakingLayers(architecture, options.StateSize, options.ActionSize);
        _policyNetwork = new NeuralNetwork<T>(architecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());

        // The critic scores a (state, quote) pair, so it takes state and action together and emits one value.
        // It is built here rather than asked of the caller: the public constructor takes a single policy
        // architecture, and widening that would break every existing call site.
        _criticArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: options.StateSize + options.ActionSize,
            outputSize: 1);
        EnsureDefaultLayers(_criticArchitecture, options.StateSize + options.ActionSize, 1);

        var lossFunction = TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>();
        _critic = new NeuralNetwork<T>(_criticArchitecture, lossFunction: lossFunction);

        // CloneForModelConstruction gives each target its OWN layer objects. Building them from the same
        // architecture instance would make the target the SAME network as the online one by reference, and
        // the TD target would then be read from the very network the update is moving.
        _targetCritic = new NeuralNetwork<T>(_criticArchitecture.CloneForModelConstruction(), lossFunction: lossFunction);
        _targetPolicyNetwork = new NeuralNetwork<T>(architecture.CloneForModelConstruction(), lossFunction: lossFunction);

        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);

        SoftUpdateTargetNetwork(_critic, _targetCritic, 1.0);
        SoftUpdateTargetNetwork(_policyNetwork, _targetPolicyNetwork, 1.0);
    }

    /// <summary>Action-space ascent step turning the deterministic policy gradient into a regression target.</summary>
    private const double ActorPolicyGradientStep = 0.05;

    /// <summary>Central-difference half-width used to estimate the critic's action sensitivity.</summary>
    private const double ActionFiniteDifferenceEpsilon = 1e-3;

    /// <summary>
    /// Evaluates the critic at a state-action pair — what the agent believes this quote is worth.
    /// </summary>
    public T EvaluateCritic(Vector<T> state, Vector<T> action)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (action is null) throw new ArgumentNullException(nameof(action));

        return _critic.Predict(Tensor<T>.FromVector(ConcatenateStateAction(state, action))).ToVector()[0];
    }

    private static Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        var combined = new Vector<T>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++) combined[i] = state[i];
        for (int i = 0; i < action.Length; i++) combined[state.Length + i] = action[i];
        return combined;
    }

    /// <summary>
    /// Validates the architecture and creates default market-making layers if needed.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks that the network matches the input and output sizes
    /// for market-making and fills in a sensible default if no layers were provided.</para>
    /// </remarks>
    private void EnsureMarketMakingLayers(NeuralNetworkArchitecture<T> architecture, int stateSize, int actionSize)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        if (architecture.CalculatedInputSize != stateSize)
            throw new ArgumentException($"Architecture input size {architecture.CalculatedInputSize} does not match expected {stateSize}.", nameof(architecture));

        if (architecture.OutputSize != actionSize)
            throw new ArgumentException($"Architecture output size {architecture.OutputSize} does not match expected {actionSize}.", nameof(architecture));

        ApplyNetworkSeed(architecture);

        if (architecture.Layers.Count == 0)
        {
            // ReLU MLP sized by TradingAgentOptions.HiddenLayers (default [64, 64], the same network
            // LayerHelper.CreateDefaultMarketMakingLayers builds).
            var hiddenSizes = GetHiddenLayerSizes();
            AddSeededDefaultLayers(architecture, () => LayerHelper<T>.CreateFeedForwardLayers(
                architecture,
                hiddenSizes,
                actionSize));
        }
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, SelectAction performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _policyNetwork.Predict(Tensor<T>.FromVector(state)).ToVector();

        // Zero-mean Gaussian exploration from the agent's seeded stream. The previous U[0, 0.05) noise was
        // unseeded and one-sided (mean +0.025), so every exploratory quote was skewed the same way and the
        // skew compounded into the policy, which is regressed onto the actions it took.
        return training
            ? AddGaussianExplorationNoise(action, ExplorationNoiseStandardDeviation)
            : action;
    }

    /// <summary>
    /// Standard deviation of the Gaussian exploration noise added to the policy output in training mode.
    /// </summary>
    private const double ExplorationNoiseStandardDeviation = 0.05;

    #endregion

    #region Training

    /// <summary>
    /// Performs a one-shot supervised update for the training/test harness.
    /// </summary>
    /// <remarks>
    /// The shared base adapter decodes <paramref name="target"/> into a discrete one-hot action sized
    /// to the target length, which is incompatible with the market-making policy's continuous,
    /// ActionSize-wide output — the policy-regression loss in <see cref="Train()"/> compares the policy
    /// output against the stored action and would mismatch in length. We therefore build a desired
    /// action of the agent's own ActionSize from the supervised target, store the transition, and run
    /// the policy update.
    /// </remarks>
    public override void Train(Vector<T> state, Vector<T> target)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (target.Length == 0)
            throw new ArgumentException("target must contain at least one element.", nameof(target));

        // Desired continuous action of the agent's own ActionSize, derived from the supervised target.
        int actionSize = TradingOptions.ActionSize;
        var desiredAction = new Vector<T>(actionSize);
        for (int i = 0; i < actionSize; i++)
            desiredAction[i] = target[i % target.Length];

        // Bounded scalar reward signal from the supervised target (mean is dimension-agnostic).
        T reward = NumOps.Zero;
        for (int i = 0; i < target.Length; i++)
            reward = NumOps.Add(reward, target[i]);
        reward = NumOps.Divide(reward, NumOps.FromDouble(target.Length));

        StoreExperience(state, desiredAction, reward, state, done: true);

        SupervisedUpdateRequested = true;
        try
        {
            Train();
        }
        finally
        {
            SupervisedUpdateRequested = false;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, Train performs a training step. This updates the MarketMakingAgent architecture so it learns from data.
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
        if (effectiveBatchSize <= 0 || ReplayBuffer.Count < effectiveBatchSize) return NumOps.Zero;

        // TradingAgentOptions.WarmupSteps: collect this many transitions before the first update.
        if (IsInWarmup(ReplayBuffer.Count)) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(effectiveBatchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;

        int stateDim = batch[0].State.Length;
        int actionDim = batch[0].Action.Length;
        int stateActionDim = stateDim + actionDim;
        double gamma = Convert.ToDouble(TradingOptions.DiscountFactor);

        // ---- 1. Critic target: y = r + gamma * (1 - done) * Q'(s', mu'(s')) ----
        // This is where the REWARD enters the update. The previous implementation regressed the policy
        // onto the actions it had itself taken, so the reward was never read by anything: whatever the
        // agent happened to quote became its own training target and the quotes could not improve.
        var statesData = new T[n * stateDim];
        var nextStatesData = new T[n * stateDim];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < stateDim; j++)
            {
                statesData[(i * stateDim) + j] = batch[i].State[j];
                nextStatesData[(i * stateDim) + j] = batch[i].NextState[j];
            }
        }

        var states = new Tensor<T>([n, stateDim], new Vector<T>(statesData));
        var nextStates = new Tensor<T>([n, stateDim], new Vector<T>(nextStatesData));

        var nextActions = _targetPolicyNetwork.Predict(nextStates).ToVector();
        var nextStateActionsData = new T[n * stateActionDim];
        var stateActionsData = new T[n * stateActionDim];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < stateDim; j++)
            {
                nextStateActionsData[(i * stateActionDim) + j] = batch[i].NextState[j];
                stateActionsData[(i * stateActionDim) + j] = batch[i].State[j];
            }

            for (int j = 0; j < actionDim; j++)
            {
                nextStateActionsData[(i * stateActionDim) + stateDim + j] = nextActions[(i * actionDim) + j];
                stateActionsData[(i * stateActionDim) + stateDim + j] = batch[i].Action[j];
            }
        }

        var nextStateActions = new Tensor<T>([n, stateActionDim], new Vector<T>(nextStateActionsData));
        var nextQ = _targetCritic.Predict(nextStateActions).ToVector();

        var targetData = new T[n];
        for (int i = 0; i < n; i++)
        {
            double bootstrap = batch[i].Done ? 0.0 : gamma * NumOps.ToDouble(nextQ[i]);
            targetData[i] = NumOps.Add(batch[i].Reward, NumOps.FromDouble(bootstrap));
        }

        var stateActions = new Tensor<T>([n, stateActionDim], new Vector<T>(stateActionsData));
        var targets = new Tensor<T>([n, 1], new Vector<T>(targetData));

        _critic.Train(stateActions, targets);
        T criticLoss = _critic.GetLastLoss();

        // ---- 2. Policy update: ascend Q(s, mu(s)) (deterministic policy gradient) ----
        var means = _policyNetwork.Predict(states).ToVector();
        var actionGradients = EstimateActionGradients(batch, means, n, stateDim, actionDim, stateActionDim);

        T maxPosition = TradingOptions.MaxPositionSize;
        T minPosition = NumOps.Negate(maxPosition);
        var policyTargetData = new T[n * actionDim];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < actionDim; j++)
            {
                int flat = (i * actionDim) + j;
                T ascended = NumOps.Add(
                    means[flat],
                    NumOps.FromDouble(ActorPolicyGradientStep * actionGradients[flat]));
                policyTargetData[flat] = MathHelper.Clamp<T>(ascended, minPosition, maxPosition);
            }
        }

        var policyTargets = new Tensor<T>([n, actionDim], new Vector<T>(policyTargetData));
        _policyNetwork.Train(states, policyTargets);
        T policyLoss = _policyNetwork.GetLastLoss();

        // ---- 3. Polyak target updates ----
        double tau = TradingOptions.Tau;
        SoftUpdateTargetNetwork(_critic, _targetCritic, tau);
        SoftUpdateTargetNetwork(_policyNetwork, _targetPolicyNetwork, tau);

        if (_updateCount < int.MaxValue)
        {
            _updateCount++;
        }

        T loss = NumOps.Add(criticLoss, policyLoss);
        LossHistory.Add(loss);
        return loss;
    }

    /// <summary>
    /// Central-difference estimate of <c>dQ/da</c> at the policy's current quote for each sampled state,
    /// returned as a flat <c>[n * actionDim]</c> buffer. One batched critic pass per direction per sign.
    /// </summary>
    private double[] EstimateActionGradients(
        List<Experience<T>> batch, Vector<T> means, int n, int stateDim, int actionDim, int stateActionDim)
    {
        var gradients = new double[n * actionDim];
        double epsilon = ActionFiniteDifferenceEpsilon;

        for (int dimension = 0; dimension < actionDim; dimension++)
        {
            var plusData = new T[n * stateActionDim];
            var minusData = new T[n * stateActionDim];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < stateDim; j++)
                {
                    plusData[(i * stateActionDim) + j] = batch[i].State[j];
                    minusData[(i * stateActionDim) + j] = batch[i].State[j];
                }

                for (int j = 0; j < actionDim; j++)
                {
                    T mean = means[(i * actionDim) + j];
                    int slot = (i * stateActionDim) + stateDim + j;
                    plusData[slot] = j == dimension ? NumOps.Add(mean, NumOps.FromDouble(epsilon)) : mean;
                    minusData[slot] = j == dimension ? NumOps.Subtract(mean, NumOps.FromDouble(epsilon)) : mean;
                }
            }

            var plusQ = _critic.Predict(new Tensor<T>([n, stateActionDim], new Vector<T>(plusData))).ToVector();
            var minusQ = _critic.Predict(new Tensor<T>([n, stateActionDim], new Vector<T>(minusData))).ToVector();
            for (int i = 0; i < n; i++)
            {
                gradients[(i * actionDim) + dimension] =
                    (NumOps.ToDouble(plusQ[i]) - NumOps.ToDouble(minusQ[i])) / (2.0 * epsilon);
            }
        }

        return gradients;
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, LoadModel performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, SaveModel performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, StoreExperience performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, ScaleReward(reward), nextState, done);
        ReplayBuffer.Add(experience);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Every market-making checkpoint saved before this agent gained its critic is now unloadable, and this
    /// is what says so. It is a hard failure with no migration path on purpose: the saved policy was
    /// produced by an update that regressed the policy onto its own quotes and never read the reward, so it
    /// holds no learned value information worth carrying forward. Loading part of it would silently
    /// resurrect a policy that never learned anything — precisely the defect this agent was repaired for.
    /// </para>
    /// </remarks>
    protected override string? DescribeIncompatibleCheckpoint(int savedParameterCount, long currentParameterCount)
    {
        if (savedParameterCount == currentParameterCount)
        {
            return null;
        }

        string shapeHint = savedParameterCount == _policyNetwork.GetParameters().Length
            ? " That is exactly the size of the policy network on its own — the shape this agent had before "
              + "it gained a critic — so this is almost certainly a pre-critic checkpoint."
            : string.Empty;

        return shapeHint
            + " MarketMakingAgent now trains a Q(s,a) critic alongside a target critic and a target policy, "
            + "so the saved parameter layout no longer matches. There is no migration: the saved policy was "
            + "trained by an update that regressed the policy onto its own quotes and never read the reward, "
            + "so it contains no learned value information. Retrain the agent from scratch.";
    }

    #endregion

    #region Model Metadata

    /// <inheritdoc/>
    /// <remarks>Adds the number of gradient updates applied under the key <c>"Updates"</c>.</remarks>
    public override Dictionary<string, T> GetTradingMetrics()
    {
        var metrics = base.GetTradingMetrics();
        metrics["Updates"] = NumOps.FromDouble(_updateCount);
        return metrics;
    }

    /// <summary>
    /// Executes GetModelMetadata for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "MarketMaking" },
                { "MaxInventory", _mmOptions.MaxInventory.HasValue
                    ? _mmOptions.MaxInventory.Value
                    : (object)"unset (the environment's limit binds)" },
                { "BaseSpread", _mmOptions.BaseSpread },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes ComputeGradients for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _policyNetwork.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _policyNetwork.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
