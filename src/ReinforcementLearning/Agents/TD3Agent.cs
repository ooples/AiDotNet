using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.TD3;

/// <summary>
/// Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for continuous control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TD3 improves upon DDPG with three key innovations:
/// 1. Twin Q-Networks: Uses two Q-functions to reduce overestimation bias
/// 2. Delayed Policy Updates: Updates policy less frequently than Q-networks
/// 3. Target Policy Smoothing: Adds noise to target actions for robustness
/// </para>
/// <para><b>For Beginners:</b>
/// TD3 is one of the best algorithms for continuous control tasks (like robot movement).
/// It's more stable and robust than DDPG.
///
/// Key innovations:
/// - **Twin Critics**: Uses two Q-networks and takes the minimum to avoid overoptimism
/// - **Delayed Updates**: Waits before updating the policy to let Q-values stabilize
/// - **Target Smoothing**: Adds noise to target actions to prevent exploitation of errors
///
/// Think of it like getting a second opinion before making decisions, and taking time
/// to verify information before acting on it.
///
/// Used by: Robotic control, autonomous systems, continuous optimization
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TD3 agent for stable continuous control
/// var options = new TD3Options&lt;double&gt; { StateSize = 4, ActionSize = 2};
/// var agent = new TD3Agent&lt;double&gt;(options);
///
/// // Select a continuous action with target policy smoothing
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Addressing Function Approximation Error in Actor-Critic Methods",
    "https://arxiv.org/abs/1802.09477",
    Year = 2018,
    Authors = "Fujimoto, S., van Hoof, H., & Meger, D.")]
public partial class TD3Agent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_actorNetwork);
        RegisterParameterComponent(_targetActorNetwork);
        RegisterParameterComponent(_critic1Network);
        RegisterParameterComponent(_critic2Network);
        RegisterParameterComponent(_targetCritic1Network);
        RegisterParameterComponent(_targetCritic2Network);
    }
    private TD3Options<T> _options;

    // Actor-update hyperparameters (named rather than magic literals). ActorPolicyGradientStep is the
    // deterministic-policy-gradient ascent step on the action; FiniteDifferenceEpsilon is the central
    // finite-difference interval used to estimate ∇a Q1 (the critic exposes no analytic input
    // gradient). Distinct from the actor NETWORK optimizer's learning rate.
    private const double ActorPolicyGradientStep = 0.05;
    private const double FiniteDifferenceEpsilon = 1e-3;

    private readonly INumericOperations<T> _numOps;

    private INeuralNetwork<T> _actorNetwork;
    [Buffer]
    private INeuralNetwork<T> _targetActorNetwork;
    private INeuralNetwork<T> _critic1Network;
    private INeuralNetwork<T> _critic2Network;
    private INeuralNetwork<T> _targetCritic1Network;
    private INeuralNetwork<T> _targetCritic2Network;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private Random _random;
    private int _stepCount;
    private int _updateCount;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public TD3Agent()
        : this(new TD3Options<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public TD3Agent(TD3Options<T> options) : base(CreateBaseOptions(options))
    {
        _options = options;
        Options = _options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = Random;
        _stepCount = 0;
        _updateCount = 0;

        // Initialize networks directly in constructor
        // Actor network: state -> action
        _actorNetwork = CreateActorNetwork();
        _targetActorNetwork = CreateActorNetwork();
        CopyNetworkWeights(_actorNetwork, _targetActorNetwork);

        // Twin Critic networks: (state, action) -> Q-value
        _critic1Network = CreateCriticNetwork();
        _critic2Network = CreateCriticNetwork();
        _targetCritic1Network = CreateCriticNetwork();
        _targetCritic2Network = CreateCriticNetwork();

        // Register every network so DeepReinforcementLearningAgentBase.Dispose releases it.
        Networks.Add(_actorNetwork);
        Networks.Add(_targetActorNetwork);
        Networks.Add(_critic1Network);
        Networks.Add(_critic2Network);
        Networks.Add(_targetCritic1Network);
        Networks.Add(_targetCritic2Network);

        CopyNetworkWeights(_critic1Network, _targetCritic1Network);
        CopyNetworkWeights(_critic2Network, _targetCritic2Network);

        // Initialize replay buffer
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize, _options.Seed);
    }

    private static ReinforcementLearningOptions<T> CreateBaseOptions(TD3Options<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.ActorLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        };
    }

    private NeuralNetwork<T> CreateActorNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.ActorHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer with tanh activation to bound actions to [-1, 1]
        layers.Add(new DenseLayer<T>(_options.ActionSize, (IActivationFunction<T>)new TanhActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_options.ActorLearningRate),
            lossFunction: new MeanSquaredErrorLoss<T>());
    }

    private NeuralNetwork<T> CreateCriticNetwork()
    {
        var layers = new List<ILayer<T>>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int prevSize = inputSize;

        foreach (var hiddenSize in _options.CriticHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output single Q-value
        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_options.CriticLearningRate),
            lossFunction: _options.CriticLossFunction);
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var actionTensor = _actorNetwork.Predict(stateTensor);
        var action = actionTensor.ToVector();

        if (training)
        {
            // Add exploration noise during training: clip(a + N(0, sigma), -1, 1), one draw per dimension in order
            var engine = AiDotNetEngine.Current;
            var noise = RLTensorOps.NormalRandom(_random, action.Length, _numOps.Zero, _numOps.FromDouble(_options.ExplorationNoise));
            action = engine.Clamp(engine.Add(action, noise), _numOps.FromDouble(-1), _numOps.FromDouble(1));
        }

        return action;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
        _stepCount++;
    }

    /// <summary>
    /// Performs a one-shot supervised update for the training/test harness.
    /// </summary>
    /// <remarks>
    /// The shared base adapter decodes <paramref name="target"/> into a discrete one-hot action sized
    /// to the target length, which is incompatible with TD3's continuous critic input
    /// (StateSize + ActionSize). We act in the state to obtain an action of the agent's own ActionSize,
    /// derive a bounded scalar reward from the supervised target, store the transition, and run a TD3
    /// update.
    /// </remarks>
    public override void Train(Vector<T> state, Vector<T> target)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (target.Length == 0)
            throw new ArgumentException("target must contain at least one element.", nameof(target));

        var action = SelectAction(state, training: true);

        // mean of the supervised target
        T reward = NumOps.Divide(AiDotNetEngine.Current.Sum(target), NumOps.FromDouble(target.Length));

        // Treat this one-shot supervised transition as terminal: nextState is fabricated as `state`, so
        // done: false would add a TD3 bootstrap term from the target critics and optimize against an
        // invented transition. done: true makes the critic target just the supplied reward.
        StoreExperience(state, action, reward, state, done: true);

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

    /// <inheritdoc />
    public override T Train()
    {
        // A supervised one-shot Train(state, target) call bypasses the autonomous-exploration warmup
        // and trains on the samples gathered so far (clamped to the buffer); autonomous stepping still
        // respects warmup.
        int effectiveBatchSize = SupervisedUpdateRequested
            ? System.Math.Min(_options.BatchSize, _replayBuffer.Count)
            : _options.BatchSize;
        if ((!SupervisedUpdateRequested && _replayBuffer.Count < _options.WarmupSteps)
            || effectiveBatchSize <= 0
            || _replayBuffer.Count < effectiveBatchSize)
        {
            return _numOps.Zero;
        }

        var batch = _replayBuffer.Sample(effectiveBatchSize);
        int n = batch.Count;
        if (n == 0) return _numOps.Zero;

        int stateDim = _options.StateSize;
        int actionDim = _options.ActionSize;
        int saDim = stateDim + actionDim;
        T gamma = DiscountFactor;

        // --- Critic update (Fujimoto et al. 2018) ---
        // Target action with policy smoothing: a' = clip(mu'(s') + clip(N(0,sigma), -c, c), -1, 1).
        // Clipped double-Q target: y = r + gamma*(1-done)*min(Q'1(s',a'), Q'2(s',a')). Regress both
        // critics toward y.
        // Validate replay-sample shapes before any tensor write/index: a malformed transition
        // (wrong State/Action/NextState length) would otherwise crash tensor construction or
        // train the twin critics on corrupted inputs.
        foreach (var exp in batch)
        {
            if (exp.State.Length != stateDim || exp.NextState.Length != stateDim || exp.Action.Length != actionDim)
            {
                throw new InvalidOperationException(
                    $"TD3 replay experience has wrong dimensions; expected State={stateDim}, " +
                    $"NextState={stateDim}, Action={actionDim} but got State={exp.State.Length}, " +
                    $"NextState={exp.NextState.Length}, Action={exp.Action.Length}.");
            }
        }

        // Target policy smoothing for the whole batch; the smoothing noise is drawn sample by sample, dimension by
        // dimension, as before.
        var engine = AiDotNetEngine.Current;
        var nextStates = RLTensorOps.Batch(batch.Select(exp => exp.NextState), stateDim);
        var targetActions = _targetActorNetwork.Predict(nextStates);
        T noiseClip = _numOps.FromDouble(_options.TargetNoiseClip);
        var smoothing = engine.Clamp(
            RLTensorOps.NormalRandom(_random, n * actionDim, _numOps.Zero, _numOps.FromDouble(_options.TargetPolicyNoise)),
            _numOps.Negate(noiseClip), noiseClip);
        var smoothedFlat = engine.Clamp(
            engine.Add(targetActions.ToVector().GetSubVector(0, n * actionDim), smoothing), _numOps.FromDouble(-1), _numOps.FromDouble(1));
        var nextActions = RLTensorOps.Rows(new Tensor<T>([n, actionDim], smoothedFlat), n);
        var nextSA = RLTensorOps.Batch(batch.Select((exp, row) => ConcatenateStateAction(exp.NextState, nextActions[row])), saDim);

        // Clipped double-Q: min(Q'1, Q'2) (Q'2 when the first is not strictly smaller)
        var q1n = RLTensorOps.FirstColumn(_targetCritic1Network.Predict(nextSA), n).AsTensor();
        var q2n = RLTensorOps.FirstColumn(_targetCritic2Network.Predict(nextSA), n).AsTensor();
        var minN = engine.TensorWhere(engine.TensorLessThan(q1n, q2n), q1n, q2n).ToVector();

        var rewards = Vector<T>.FromEnumerable(batch.Select(exp => exp.Reward));
        var done = batch.Select(exp => exp.Done).ToArray();
        var y = RLTensorOps.TdTargets(rewards, minN, done, gamma);

        var saInputs = RLTensorOps.Batch(batch.Select(exp => ConcatenateStateAction(exp.State, exp.Action)), saDim);
        var yTargets = new Tensor<T>([n, 1], y);
        _critic1Network.Train(saInputs, yTargets);
        _critic2Network.Train(saInputs, yTargets);
        T criticLoss = _numOps.Divide(
            _numOps.Add(_critic1Network.GetLastLoss(), _critic2Network.GetLastLoss()), _numOps.FromDouble(2));

        // --- Delayed actor update via the deterministic policy gradient ---
        // The DPG is ∇θ J = E[∇a Q1(s,a)|a=mu(s) · ∇θ mu(s)]. We obtain ∇a Q1 by central finite
        // differences (the critic exposes no analytic input gradient) and realise the chain rule
        // through the network's MSE Train: regressing the actor toward the Q-ascending target
        // a + step·∇a Q1 makes its parameters move along ∇θ mu in the gradient-ascent direction.
        if (_updateCount % _options.PolicyUpdateFrequency == 0)
        {
            T step = _numOps.FromDouble(ActorPolicyGradientStep);
            T lowerBound = _numOps.FromDouble(-1);
            T upperBound = _numOps.FromDouble(1);
            var actorInputs = RLTensorOps.Batch(batch.Select(exp => exp.State), stateDim);
            var currentActions = RLTensorOps.Rows(_actorNetwork.Predict(actorInputs), n);

            // clip(a + step * dQ1/da, -1, 1) per sample
            var actorTargets = RLTensorOps.Batch(
                batch.Select((exp, row) => engine.Clamp(
                    engine.Add(currentActions[row], engine.Multiply(FiniteDiffActionGradient(_critic1Network, exp.State, currentActions[row]), step)),
                    lowerBound, upperBound)),
                actionDim);
            _actorNetwork.Train(actorInputs, actorTargets);

            // Update target networks with soft updates (delayed, with the policy)
            SoftUpdateTargetNetworks();
        }

        _updateCount++;

        return criticLoss;
    }

    /// <summary>
    /// Central finite-difference estimate of ∇a Q(s, a) — the action-gradient at the heart of the
    /// deterministic policy gradient (Silver et al. 2014; Lillicrap et al. 2015). Used because the
    /// critic network does not expose an analytic gradient w.r.t. its input.
    /// </summary>
    private Vector<T> FiniteDiffActionGradient(INeuralNetwork<T> critic, Vector<T> state, Vector<T> action)
    {
        // dQ/da_i = (Q(s, a + eps e_i) - Q(s, a - eps e_i)) / (2 eps), all 2 |a| probes in one batched forward
        var engine = AiDotNetEngine.Current;
        int width = action.Length;
        if (width == 0)
        {
            return new Vector<T>(0);
        }

        T eps = _numOps.FromDouble(FiniteDifferenceEpsilon);
        T twoEps = _numOps.FromDouble(2.0 * FiniteDifferenceEpsilon);
        var perturbations = Enumerable.Range(0, width).Select(dim => engine.Multiply(RLTensorOps.OneHot<T>(width, dim), eps)).ToList();
        var probes = perturbations.Select(delta => ConcatenateStateAction(state, engine.Add(action, delta)))
            .Concat(perturbations.Select(delta => ConcatenateStateAction(state, engine.Subtract(action, delta))))
            .ToList();
        var q = RLTensorOps.FirstColumn(critic.Predict(RLTensorOps.StackRows(probes)), 2 * width);
        return engine.Divide(engine.Subtract(q.GetSubVector(0, width), q.GetSubVector(width, width)), twoEps);
    }

    private void SoftUpdateTargetNetworks()
    {
        SoftUpdateNetwork(_actorNetwork, _targetActorNetwork);
        SoftUpdateNetwork(_critic1Network, _targetCritic1Network);
        SoftUpdateNetwork(_critic2Network, _targetCritic2Network);
    }

    private void SoftUpdateNetwork(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        // theta_target <- tau * theta + (1 - tau) * theta_target
        target.UpdateParameters(RLTensorOps.SoftUpdate(sourceParams, targetParams, _options.TargetUpdateTau));
    }

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        return RLTensorOps.Concat(state, action);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["steps"] = _numOps.FromDouble(_stepCount),
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_replayBuffer.Count)
        };
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        // TD3 doesn't need per-episode reset
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            "TD3 uses actor-critic training via Train() method. " +
            "Direct gradient computation through this interface is not applicable.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // TD3 uses direct network updates during training, not manual gradient application
    }

    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
