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

namespace AiDotNet.ReinforcementLearning.Agents.DDPG;

/// <summary>
/// Deep Deterministic Policy Gradient (DDPG) agent for continuous control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DDPG is an actor-critic algorithm designed for continuous action spaces. It learns
/// a deterministic policy (actor) and uses an off-policy approach with experience replay
/// and target networks, extending DQN ideas to continuous control.
/// </para>
/// <para><b>For Beginners:</b>
/// DDPG is perfect for controlling things that need precise, continuous adjustments like:
/// - Robot arm angles (not just "left" or "right", but exact degrees)
/// - Car steering and acceleration (smooth continuous values)
/// - Temperature control, volume levels, etc.
///
/// Key components:
/// - **Actor**: Learns the best action to take (deterministic policy)
/// - **Critic**: Evaluates how good that action is (Q-value)
/// - **Target Networks**: Stable copies for training
/// - **Exploration Noise**: Adds randomness during training for exploration
///
/// Think of it like learning to drive: the actor is your decision-making (how much to
/// turn the wheel), the critic is your judgment (was that a good turn?), and noise
/// is trying slight variations to discover better techniques.
/// </para>
/// <para><b>Reference:</b>
/// Lillicrap et al., "Continuous control with deep reinforcement learning", 2015.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DDPG agent for continuous control tasks
/// var options = new DDPGOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, ActorLearningRate = 0.001 };
/// var agent = new DDPGAgent&lt;double&gt;(options);
///
/// // Select a continuous action for the current state
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
[ResearchPaper("Continuous Control with Deep Reinforcement Learning",
    "https://arxiv.org/abs/1509.02971",
    Year = 2016,
    Authors = "Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D.")]
public partial class DDPGAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_actorNetwork);
        RegisterParameterComponent(_criticNetwork);
    }

    /// <inheritdoc />
    /// <remarks>Refreshes what derives from the parameters. This ran at the end of the
    /// hand-written SetParameters; losing it would not fail a test, it would just leave
    /// the agent training against a stale target.</remarks>
    protected override void OnParametersRestored()
    {
        CopyNetworkWeights(_actorNetwork, _actorTargetNetwork);
        CopyNetworkWeights(_criticNetwork, _criticTargetNetwork);
    }
    private DDPGOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private readonly OrnsteinUhlenbeckNoise<T> _noise;

    private INeuralNetwork<T> _actorNetwork;
    private INeuralNetwork<T> _actorTargetNetwork;
    private INeuralNetwork<T> _criticNetwork;
    private INeuralNetwork<T> _criticTargetNetwork;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DDPGAgent()
        : this(new DDPGOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public DDPGAgent(DDPGOptions<T> options)
        : base(CreateBaseOptions(options))
    {
        _options = options;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _noise = new OrnsteinUhlenbeckNoise<T>(options.ActionSize, NumOps, Random, options.ExplorationNoise);
        _steps = 0;

        // Build networks
        _actorNetwork = BuildActorNetwork();
        _actorTargetNetwork = BuildActorNetwork();
        _criticNetwork = BuildCriticNetwork();
        _criticTargetNetwork = BuildCriticNetwork();

        // Initialize targets
        CopyNetworkWeights(_actorNetwork, _actorTargetNetwork);
        CopyNetworkWeights(_criticNetwork, _criticTargetNetwork);

        Networks.Add(_actorNetwork);
        Networks.Add(_actorTargetNetwork);
        Networks.Add(_criticNetwork);
        Networks.Add(_criticTargetNetwork);
    }


    private static ReinforcementLearningOptions<T> CreateBaseOptions(DDPGOptions<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.ActorLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.CriticLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        };
    }

    private NeuralNetwork<T> BuildActorNetwork()
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
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture, optimizer: CreateNetworkOptimizer(_options.ActorLearningRate));
    }

    private NeuralNetwork<T> BuildCriticNetwork()
    {
        // Critic takes state + action as input
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

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var actionTensor = _actorNetwork.Predict(stateTensor);
        var action = actionTensor.ToVector();

        if (training)
        {
            // Add exploration noise
            var noise = _noise.Sample();
            var engine = AiDotNetEngine.Current;
            action = engine.Clamp(engine.Add(action, RLTensorOps.Head(noise, action.Length)), NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
        }

        return action;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
    }

    /// <summary>
    /// Performs a one-shot supervised update for the training/test harness.
    /// </summary>
    /// <remarks>
    /// The shared base adapter decodes <paramref name="target"/> into a discrete one-hot action sized
    /// to the target length, which is incompatible with DDPG's continuous critic input
    /// (StateSize + ActionSize) — the stored experience would have the wrong action dimensionality.
    /// We act in the state to obtain an action of the agent's own ActionSize, derive a bounded scalar
    /// reward from the supervised target, store the transition, and run a DDPG update.
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
        // done: false would inject a γQ'(s, μ'(s)) bootstrap and optimize against an invented target.
        // done: true makes the critic target just the supplied reward.
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

    /// <inheritdoc/>
    public override T Train()
    {
        _steps++;
        TrainingSteps++;

        // A supervised one-shot Train(state, target) call bypasses the autonomous-exploration warmup
        // and trains on the samples gathered so far (clamped to the buffer); autonomous stepping still
        // respects warmup.
        int effectiveBatchSize = SupervisedUpdateRequested
            ? System.Math.Min(_options.BatchSize, _replayBuffer.Count)
            : _options.BatchSize;
        if ((!SupervisedUpdateRequested && _steps < _options.WarmupSteps)
            || effectiveBatchSize <= 0
            || !_replayBuffer.CanSample(effectiveBatchSize))
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(effectiveBatchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;

        int stateDim = _options.StateSize;
        int actionDim = _options.ActionSize;
        int saDim = stateDim + actionDim;
        T gamma = _options.DiscountFactor;

        // --- Critic update (Lillicrap et al. 2015) ---
        // y = r + gamma*(1-done)*Q'(s', mu'(s')); regress the critic toward y.
        // Validate replay-sample shapes before any tensor write: StoreExperience() is a public
        // input boundary, so a malformed transition would otherwise corrupt the critic inputs
        // (saInputs) or throw deep inside the target-network forwards. Reject it with a clear message.
        foreach (var exp in batch)
        {
            if (exp.State.Length != stateDim || exp.NextState.Length != stateDim || exp.Action.Length != actionDim)
            {
                throw new InvalidOperationException(
                    $"DDPG replay experience has wrong dimensions; expected State={stateDim}, " +
                    $"NextState={stateDim}, Action={actionDim} but got State={exp.State.Length}, " +
                    $"NextState={exp.NextState.Length}, Action={exp.Action.Length}.");
            }
        }

        // Q'(s', mu'(s')) for the whole batch: the target actor and target critic each run once on it.
        var nextStates = RLTensorOps.Batch(batch.Select(exp => exp.NextState), stateDim);
        var nextActions = RLTensorOps.Rows(_actorTargetNetwork.Predict(nextStates), n);
        var nextSA = RLTensorOps.Batch(batch.Select((exp, row) => ConcatenateStateAction(exp.NextState, nextActions[row])), saDim);
        var qNext = RLTensorOps.FirstColumn(_criticTargetNetwork.Predict(nextSA), n);

        var rewards = Vector<T>.FromEnumerable(batch.Select(exp => exp.Reward));
        var done = batch.Select(exp => exp.Done).ToArray();
        var y = RLTensorOps.TdTargets(rewards, qNext, done, gamma);

        var saInputs = RLTensorOps.Batch(batch.Select(exp => ConcatenateStateAction(exp.State, exp.Action)), saDim);
        var yTargets = new Tensor<T>([n, 1], y);
        _criticNetwork.Train(saInputs, yTargets);
        T criticLoss = _criticNetwork.GetLastLoss();

        // --- Actor update via the deterministic policy gradient ---
        // ∇θ J = E[∇a Q(s,a)|a=mu(s) · ∇θ mu(s)]. ComputeDDPGPolicyGradient returns −∇a Q (a loss
        // gradient for gradient ASCENT on Q); the Q-ascending action target is therefore
        // a − step·(−∇a Q) = a + step·∇a Q. Training the actor toward it realises the chain rule
        // ∇θ mu through the network's MSE Train.
        // DPG action-space ascent step for building the actor's regression target (Lillicrap et al.
        // 2015). Named constant rather than a magic literal; deliberately distinct from the actor
        // NETWORK's optimizer learning rate (_options.ActorLearningRate, already wired into
        // _actorNetwork) — they play different roles and must not be conflated.
        const double actorPolicyGradientStep = 0.05;
        T stepSize = NumOps.FromDouble(actorPolicyGradientStep);
        var engine = AiDotNetEngine.Current;
        var actorInputs = RLTensorOps.Batch(batch.Select(exp => exp.State), stateDim);
        var currentActions = RLTensorOps.Rows(_actorNetwork.Predict(actorInputs), n);

        // clip(a - step * (-dQ/da), -1, 1) per sample
        T lowerBound = NumOps.FromDouble(-1.0);
        T upperBound = NumOps.FromDouble(1.0);
        var actorTargets = RLTensorOps.Batch(
            batch.Select((exp, row) => engine.Clamp(
                engine.Subtract(currentActions[row], engine.Multiply(ComputeDDPGPolicyGradient(exp.State, currentActions[row]), stepSize)),
                lowerBound, upperBound)),
            actionDim);
        _actorNetwork.Train(actorInputs, actorTargets);
        T actorLoss = _actorNetwork.GetLastLoss();

        // Soft update target networks
        SoftUpdateTargets();

        var totalLoss = NumOps.Add(criticLoss, actorLoss);
        LossHistory.Add(totalLoss);

        return totalLoss;
    }


    private Vector<T> ComputeDDPGPolicyGradient(Vector<T> state, Vector<T> action)
    {
        // DDPG uses deterministic policy gradient: ∇θ J = E[∇a Q(s,a)|a=μ(s) * ∇θ μ(s)]
        //
        // Step 1: Compute ∇a Q(s,a) - gradient of Q-value w.r.t. actions
        // We approximate this using central finite differences, ∂Q/∂a_i ≈ (Q(s,a+ε e_i) - Q(s,a-ε e_i)) / (2ε),
        // evaluating all 2 * |a| perturbed state-actions in one batched critic forward.
        //
        // Step 2: This gradient is then backpropagated through the actor network
        // to compute ∇θ μ(s) via the chain rule
        var engine = AiDotNetEngine.Current;
        int width = action.Length;
        if (width == 0)
        {
            return new Vector<T>(0);
        }

        T epsilon = NumOps.FromDouble(0.001); // Small perturbation for finite differences
        var perturbations = Enumerable.Range(0, width).Select(dim => engine.Multiply(RLTensorOps.OneHot<T>(width, dim), epsilon)).ToList();
        var probes = perturbations.Select(delta => ConcatenateStateAction(state, engine.Add(action, delta)))
            .Concat(perturbations.Select(delta => ConcatenateStateAction(state, engine.Subtract(action, delta))))
            .ToList();
        var q = RLTensorOps.FirstColumn(_criticNetwork.Predict(RLTensorOps.StackRows(probes)), 2 * width);
        var dQda = engine.Divide(
            engine.Subtract(q.GetSubVector(0, width), q.GetSubVector(width, width)),
            NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));

        // We negate because we want to maximize Q (gradient ascent)
        return engine.Negate(dQda);
    }

    private void SoftUpdateTargets()
    {
        SoftUpdateNetwork(_actorNetwork, _actorTargetNetwork);
        SoftUpdateNetwork(_criticNetwork, _criticTargetNetwork);
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

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["ReplayBufferSize"] = NumOps.FromDouble(_replayBuffer.Count);
        baseMetrics["Steps"] = NumOps.FromDouble(_steps);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _options.StateSize,
        };
    }

    private void MaterializeNetworks()
    {
        var state = new Tensor<T>([_options.StateSize]);
        var stateAction = new Tensor<T>([_options.StateSize + _options.ActionSize]);
        _ = _actorNetwork.Predict(state);
        _ = _actorTargetNetwork.Predict(state);
        _ = _criticNetwork.Predict(stateAction);
        _ = _criticTargetNetwork.Predict(stateAction);
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            "DDPG uses actor-critic training via Train() method. " +
            "Direct gradient computation through this interface is not applicable.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        throw new NotSupportedException(
            "DDPG uses actor-critic training via Train() method. " +
            "Direct gradient application through this interface is not applicable.");
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

/// <summary>
/// Ornstein-Uhlenbeck process for temporally correlated exploration noise.
/// </summary>
internal class OrnsteinUhlenbeckNoise<T>
{
    private readonly int _size;
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private readonly double _theta;
    private readonly double _sigma;
    private Vector<T> _state;

    public OrnsteinUhlenbeckNoise(int size, INumericOperations<T> numOps, Random random, double sigma, double theta = 0.15)
    {
        _size = size;
        _numOps = numOps;
        _random = random;
        _theta = theta;
        _sigma = sigma;
        _state = new Vector<T>(size);
    }

    public Vector<T> Sample()
    {
        // dx = -theta x + sigma N(0, 1), one draw per dimension in index order
        var engine = AiDotNetEngine.Current;
        var drift = engine.Multiply(_state, _numOps.FromDouble(-_theta));
        var diffusion = engine.Multiply(RLTensorOps.NormalRandom<T>(_random, _size), _numOps.FromDouble(_sigma));
        _state = engine.Add(_state, engine.Add(drift, diffusion));
        return _state.Clone();
    }

    public void Reset()
    {
        _state = new Vector<T>(_size);
    }
}
