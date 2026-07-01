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
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.MADDPG;

/// <summary>
/// Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent for cooperative
/// and competitive multi-agent reinforcement learning with continuous action spaces.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MADDPG extends DDPG to multi-agent settings with centralized training
/// and decentralized execution.
/// </para>
/// <para><b>For Beginners:</b>
/// MADDPG enables multiple agents to learn together in shared environments.
/// During training, critics can "see" all agents' actions (centralized),
/// but during execution, each agent acts independently (decentralized).
///
/// Key features:
/// - **Centralized Critics**: Observe all agents during training
/// - **Decentralized Actors**: Independent policies per agent
/// - **Continuous Actions**: Based on DDPG
/// - **Cooperative or Competitive**: Handles both settings
///
/// Think of it like: Team sports where players practice together seeing
/// everyone's moves, but during games each makes independent decisions.
///
/// Examples: Robot swarms, traffic control, multi-player games
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a multi-agent DDPG system with 3 cooperative agents
/// var options = new MADDPGOptions&lt;double&gt; { NumAgents = 3, ActorLearningRate = 0.001 };
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 8, outputSize: 2);
/// var agent = new MADDPGAgent&lt;double&gt;(arch, options);
///
/// // Each agent selects a continuous action from its observation
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2, 0.8, -0.1, 0.4, 0.6 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
    "https://arxiv.org/abs/1706.02275",
    Year = 2017,
    Authors = "Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I.")]
public class MADDPGAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private MADDPGOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    // Networks for each agent
    private List<INeuralNetwork<T>> _actorNetworks;
    private List<INeuralNetwork<T>> _targetActorNetworks;
    private List<INeuralNetwork<T>> _criticNetworks;
    private List<INeuralNetwork<T>> _targetCriticNetworks;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _stepCount;

    // Actor-update hyperparameters (named rather than magic literals). ActorGradientStepSize is the
    // deterministic-policy-gradient ascent step; ActorFiniteDifferenceEpsilon is the central
    // finite-difference interval used to estimate ∇_{a_i} Q_i (Lowe et al. 2017).
    private const double ActorGradientStepSize = 0.05;
    private const double ActorFiniteDifferenceEpsilon = 1e-3;

    // Track per-agent rewards for competitive/mixed-motive scenarios
    // Maps experience index to array of per-agent rewards
    private Dictionary<int, List<T>> _perAgentRewards;

    public MADDPGAgent() : this(new MADDPGOptions<T>()) { }

    public MADDPGAgent(MADDPGOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _options.Validate();
        // Issue #3 fix: Use configured actor learning rate for default optimizer
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = NumOps.ToDouble(_options.ActorLearningRate),
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _stepCount = 0;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);
        _perAgentRewards = new Dictionary<int, List<T>>();

        // Initialize networks directly in constructor
        _actorNetworks = new List<INeuralNetwork<T>>();
        _targetActorNetworks = new List<INeuralNetwork<T>>();
        _criticNetworks = new List<INeuralNetwork<T>>();
        _targetCriticNetworks = new List<INeuralNetwork<T>>();

        for (int i = 0; i < _options.NumAgents; i++)
        {
            // Actor: state -> action (per agent)
            var actor = CreateActorNetwork();
            var targetActor = CreateActorNetwork();
            CopyNetworkWeights(actor, targetActor);

            _actorNetworks.Add(actor);
            _targetActorNetworks.Add(targetActor);

            // Critic: (all states, all actions) -> Q-value (centralized)
            var critic = CreateCriticNetwork();
            var targetCritic = CreateCriticNetwork();
            CopyNetworkWeights(critic, targetCritic);

            _criticNetworks.Add(critic);
            _targetCriticNetworks.Add(targetCritic);

            // Register networks with base class
            Networks.Add(actor);
            Networks.Add(targetActor);
            Networks.Add(critic);
            Networks.Add(targetCritic);
        }
    }

    private INeuralNetwork<T> CreateActorNetwork()
    {
        // Create layers
        var layers = new List<ILayer<T>>();

        // Input layer
        layers.Add(new DenseLayer<T>(_options.ActorHiddenLayers[0], (IActivationFunction<T>)new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < _options.ActorHiddenLayers.Count; i++)
        {
            layers.Add(new DenseLayer<T>(_options.ActorHiddenLayers[i], (IActivationFunction<T>)new ReLUActivation<T>()));
        }

        // Output layer with Tanh for continuous actions
        // Issue #1 fix: DenseLayer constructor automatically applies Xavier/Glorot weight initialization
        layers.Add(new DenseLayer<T>(_options.ActionSize, (IActivationFunction<T>)new TanhActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture, lossFunction: _options.LossFunction);
    }

    private INeuralNetwork<T> CreateCriticNetwork()
    {
        // Centralized critic: observes all agents' states and actions
        int inputSize = (_options.StateSize + _options.ActionSize) * _options.NumAgents;

        // Create layers
        var layers = new List<ILayer<T>>();

        // Input layer
        layers.Add(new DenseLayer<T>(_options.CriticHiddenLayers[0], (IActivationFunction<T>)new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < _options.CriticHiddenLayers.Count; i++)
        {
            layers.Add(new DenseLayer<T>(_options.CriticHiddenLayers[i], (IActivationFunction<T>)new ReLUActivation<T>()));
        }

        // Output layer (Q-value)
        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture, lossFunction: _options.LossFunction);
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);
    }

    /// <summary>
    /// Select action for a specific agent.

    public Vector<T> SelectActionForAgent(int agentId, Vector<T> state, bool training = true)
    {
        if (agentId < 0 || agentId >= _options.NumAgents)
        {
            throw new ArgumentException($"Invalid agent ID: {agentId}");
        }

        var inputTensor = Tensor<T>.FromVector(state);
        var actionTensor = _actorNetworks[agentId].Predict(inputTensor);
        var action = actionTensor.ToVector();

        if (training)
        {
            // Add exploration noise
            for (int i = 0; i < action.Length; i++)
            {
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.FromDouble(_options.ExplorationNoise));
                action[i] = NumOps.Add(action[i], noise);
                action[i] = MathHelper.Clamp<T>(action[i], NumOps.FromDouble(-1), NumOps.FromDouble(1));
            }
        }

        return action;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Default to agent 0
        return SelectActionForAgent(0, state, training);
    }

    /// <summary>
    /// Store multi-agent experience with per-agent reward tracking.
    /// </summary>
    /// <remarks>
    /// Stores individual rewards for each agent to support both cooperative and
    /// competitive/mixed-motive scenarios. For backward compatibility, also stores
    /// an averaged reward in the experience.
    ///
    /// The per-agent rewards are stored keyed by the buffer index where the experience
    /// will be placed. This accounts for the circular buffer behavior when capacity is reached.
    /// </remarks>
    public void StoreMultiAgentExperience(
        List<Vector<T>> states,
        List<Vector<T>> actions,
        List<T> rewards,
        List<Vector<T>> nextStates,
        bool done)
    {
        // Concatenate all agents' observations for centralized storage
        var jointState = ConcatenateVectors(states);
        var jointAction = ConcatenateVectors(actions);
        var jointNextState = ConcatenateVectors(nextStates);

        // Compute the buffer index where this experience will be stored
        // This accounts for circular buffer behavior: if buffer is not full, index = Count
        // If buffer is full, the circular position is used (which we approximate here)
        int bufferIndex;
        if (_replayBuffer.Count < _replayBuffer.Capacity)
        {
            // Buffer not full yet, experience goes at the end
            bufferIndex = _replayBuffer.Count;
        }
        else
        {
            // Buffer is full, circular overwrite - use modulo to find position
            // Note: We approximate the position since we don't have access to internal _position field
            // This works because experiences are added sequentially
            bufferIndex = _stepCount % _replayBuffer.Capacity;
        }

        // Store per-agent rewards at the buffer index for competitive/mixed-motive scenarios
        _perAgentRewards[bufferIndex] = new List<T>(rewards);

        // Also compute average reward for cooperative scenarios (backward compatibility)
        T avgReward = NumOps.Zero;
        foreach (var reward in rewards)
        {
            avgReward = NumOps.Add(avgReward, reward);
        }
        avgReward = NumOps.Divide(avgReward, NumOps.FromDouble(rewards.Count));

        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(jointState, jointAction, avgReward, jointNextState, done));
        _stepCount++;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Clear any per-agent reward override left at this buffer position by an earlier joint
        // transition. _perAgentRewards is keyed only by replay index, so without this a scalar
        // transition that reuses a circular-buffer slot would inherit the previous occupant's STALE
        // per-agent rewards in Train(). Index computed exactly as StoreMultiAgentExperience does.
        int bufferIndex = _replayBuffer.Count < _replayBuffer.Capacity
            ? _replayBuffer.Count
            : _stepCount % _replayBuffer.Capacity;
        _perAgentRewards.Remove(bufferIndex);

        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
        _stepCount++;
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.WarmupSteps || _replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        // Sample batch with indices to retrieve per-agent rewards
        var (batch, indices) = _replayBuffer.SampleWithIndices(_options.BatchSize);
        int n = batch.Count;
        int numAgents = _options.NumAgents;
        int sd = _options.StateSize;
        int ad = _options.ActionSize;
        int criticIn = (sd + ad) * numAgents;
        T gamma = DiscountFactor;

        // MADDPG trains on JOINT transitions stored by StoreMultiAgentExperience: the state is the
        // concatenation of every agent's observation and the action of every agent's action. Validate
        // EVERY sampled item (not just batch[0]) and fail loudly on a non-joint transition (e.g. a
        // single-agent vector passed through the generic StoreExperience) rather than silently
        // no-op'ing or indexing out of bounds deep in the per-agent loops below.
        int expectedJointState = numAgents * sd;
        int expectedJointAction = numAgents * ad;
        if (n == 0)
        {
            return NumOps.Zero;
        }
        for (int i = 0; i < n; i++)
        {
            var item = batch[i];
            if (item.State.Length != expectedJointState ||
                item.NextState.Length != expectedJointState ||
                item.Action.Length != expectedJointAction)
            {
                throw new InvalidOperationException(
                    $"{nameof(MADDPGAgent<T>)}.{nameof(Train)} requires joint transitions stored via " +
                    $"{nameof(StoreMultiAgentExperience)}. Batch item {i} has state/action/next-state " +
                    $"lengths {item.State.Length}/{item.Action.Length}/{item.NextState.Length}; expected " +
                    $"{expectedJointState}/{expectedJointAction}/{expectedJointState}.");
            }
        }

        T totalLoss = NumOps.Zero;

        // Per-agent centralized-critic + decentralized-actor update (Lowe et al. 2017).
        for (int agentId = 0; agentId < numAgents; agentId++)
        {
            // --- Centralized critic update: y = r_i + gamma*(1-done)*Q'_i(x', a'_1..a'_N),
            // a'_j = mu'_j(o'_j). Regress critic_i toward y on the stored (x, a_1..a_N). ---
            var criticInputs = new Tensor<T>([n, criticIn]);
            var yTargets = new Tensor<T>([n, 1]);
            for (int i = 0; i < n; i++)
            {
                var exp = batch[i];
                var jointNextAction = new Vector<T>(numAgents * ad);
                for (int j = 0; j < numAgents; j++)
                {
                    var nextObsJ = SliceVector(exp.NextState, j * sd, sd);
                    var aJ = _targetActorNetworks[j].Predict(Tensor<T>.FromVector(nextObsJ)).ToVector();
                    for (int k = 0; k < ad; k++) jointNextAction[j * ad + k] = aJ[k];
                }
                var targetCriticInput = ConcatTwo(exp.NextState, jointNextAction);
                T qNext = _targetCriticNetworks[agentId].Predict(Tensor<T>.FromVector(targetCriticInput)).ToVector()[0];

                T rI = exp.Reward;
                if (_perAgentRewards.TryGetValue(indices[i], out var pr))
                {
                    // A per-agent reward override must cover EVERY agent; a partial list would
                    // silently fall back to the averaged scalar reward for the later agents and
                    // train their critics on inconsistent targets. Reject it loudly instead.
                    if (pr.Count != numAgents)
                    {
                        throw new InvalidOperationException(
                            $"Per-agent reward override for replay index {indices[i]} contains {pr.Count} " +
                            $"rewards; expected {numAgents}.");
                    }
                    rI = pr[agentId];
                }
                T y = rI;
                if (!exp.Done) y = NumOps.Add(y, NumOps.Multiply(gamma, qNext));

                var criticInput = ConcatTwo(exp.State, exp.Action);
                for (int c = 0; c < criticIn; c++) criticInputs[i, c] = criticInput[c];
                yTargets[i, 0] = y;
            }
            _criticNetworks[agentId].Train(criticInputs, yTargets);
            T criticLoss = _criticNetworks[agentId].GetLastLoss();

            // --- Decentralized actor update via the deterministic policy gradient: improve a_i in
            // the direction ∇_{a_i} Q_i (estimated by central finite differences, holding the other
            // agents' actions fixed), then train actor_i toward the improved action. ---
            var actorInputs = new Tensor<T>([n, sd]);
            var actorTargets = new Tensor<T>([n, ad]);
            T step = NumOps.FromDouble(ActorGradientStepSize);
            T eps = NumOps.FromDouble(ActorFiniteDifferenceEpsilon);
            T twoEps = NumOps.FromDouble(2.0 * ActorFiniteDifferenceEpsilon);
            for (int i = 0; i < n; i++)
            {
                var exp = batch[i];
                var ownObs = SliceVector(exp.State, agentId * sd, sd);
                var aI = _actorNetworks[agentId].Predict(Tensor<T>.FromVector(ownObs)).ToVector();

                // Joint action with agent i's slot replaced by the current actor output.
                var jointAction = exp.Action.Clone();
                for (int k = 0; k < ad; k++) jointAction[agentId * ad + k] = aI[k];

                var grad = new Vector<T>(ad);
                for (int k = 0; k < ad; k++)
                {
                    var jaPlus = jointAction.Clone();
                    var jaMinus = jointAction.Clone();
                    jaPlus[agentId * ad + k] = NumOps.Add(jaPlus[agentId * ad + k], eps);
                    jaMinus[agentId * ad + k] = NumOps.Subtract(jaMinus[agentId * ad + k], eps);
                    T qPlus = _criticNetworks[agentId].Predict(Tensor<T>.FromVector(ConcatTwo(exp.State, jaPlus))).ToVector()[0];
                    T qMinus = _criticNetworks[agentId].Predict(Tensor<T>.FromVector(ConcatTwo(exp.State, jaMinus))).ToVector()[0];
                    grad[k] = NumOps.Divide(NumOps.Subtract(qPlus, qMinus), twoEps);
                }

                for (int j = 0; j < sd; j++) actorInputs[i, j] = ownObs[j];
                for (int k = 0; k < ad; k++)
                    actorTargets[i, k] = MathHelper.Clamp<T>(
                        NumOps.Add(aI[k], NumOps.Multiply(step, grad[k])),
                        NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
            }
            _actorNetworks[agentId].Train(actorInputs, actorTargets);
            T actorLoss = _actorNetworks[agentId].GetLastLoss();

            totalLoss = NumOps.Add(totalLoss, NumOps.Add(criticLoss, actorLoss));

            // Soft update target networks
            SoftUpdateTargetNetwork(_actorNetworks[agentId], _targetActorNetworks[agentId]);
            SoftUpdateTargetNetwork(_criticNetworks[agentId], _targetCriticNetworks[agentId]);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(_options.NumAgents * 2));
    }

    private static Vector<T> SliceVector(Vector<T> v, int start, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++) result[i] = v[start + i];
        return result;
    }

    private static Vector<T> ConcatTwo(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }

    private void SoftUpdateTargetNetwork(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        var oneMinusTau = NumOps.Subtract(NumOps.One, _options.TargetUpdateTau);

        for (int i = 0; i < sourceParams.Length; i++)
        {
            var sourceContrib = NumOps.Multiply(_options.TargetUpdateTau, sourceParams[i]);
            var targetContrib = NumOps.Multiply(oneMinusTau, targetParams[i]);
            targetParams[i] = NumOps.Add(sourceContrib, targetContrib);
        }

        target.UpdateParameters(targetParams);
    }

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
    }

    private Vector<T> ConcatenateVectors(List<Vector<T>> vectors)
    {
        int totalSize = 0;
        foreach (var vec in vectors)
        {
            totalSize += vec.Length;
        }

        var result = new Vector<T>(totalSize);
        int offset = 0;

        foreach (var vec in vectors)
        {
            for (int i = 0; i < vec.Length; i++)
            {
                result[offset + i] = vec[i];
            }
            offset += vec.Length;
        }

        return result;
    }

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        var result = new Vector<T>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++)
        {
            result[i] = state[i];
        }
        for (int i = 0; i < action.Length; i++)
        {
            result[state.Length + i] = action[i];
        }
        return result;
    }

    private Vector<T> ComputeJointTargetActions(Vector<T> jointNextState)
    {
        // Decompose joint state into individual agent states
        var individualActions = new List<Vector<T>>();

        for (int i = 0; i < _options.NumAgents; i++)
        {
            // Extract this agent's next state from joint state
            int stateOffset = i * _options.StateSize;
            var agentNextState = new Vector<T>(_options.StateSize);
            for (int j = 0; j < _options.StateSize; j++)
            {
                agentNextState[j] = jointNextState[stateOffset + j];
            }

            // Compute action using target actor network
            var agentNextStateTensor = Tensor<T>.FromVector(agentNextState);
            var targetActionTensor = _targetActorNetworks[i].Predict(agentNextStateTensor);
            var targetAction = targetActionTensor.ToVector();

            individualActions.Add(targetAction);
        }

        // Concatenate all target actions into joint action vector
        return ConcatenateVectors(individualActions);
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["steps"] = NumOps.FromDouble(_stepCount),
            ["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
    }

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

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
        };
    }

    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Serializes the MADDPG agent to a byte array.
    /// </summary>
    /// <returns>Byte array containing the serialized agent data.</returns>
    /// <remarks>
    /// Serializes configuration values and all actor/critic network weights.
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.NumAgents);
        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);

        void WriteNetwork(INeuralNetwork<T> network)
        {
            var bytes = network.Serialize();
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        foreach (var network in _actorNetworks)
        {
            WriteNetwork(network);
        }

        foreach (var network in _targetActorNetworks)
        {
            WriteNetwork(network);
        }

        foreach (var network in _criticNetworks)
        {
            WriteNetwork(network);
        }

        foreach (var network in _targetCriticNetworks)
        {
            WriteNetwork(network);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a MADDPG agent from a byte array.
    /// </summary>
    /// <param name="data">Byte array containing the serialized agent data.</param>
    /// <remarks>
    /// Expects data created by <see cref="Serialize"/> with a compatible configuration.
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        var numAgents = reader.ReadInt32();
        var stateSize = reader.ReadInt32();
        var actionSize = reader.ReadInt32();

        if (numAgents != _options.NumAgents || stateSize != _options.StateSize || actionSize != _options.ActionSize)
        {
            throw new InvalidOperationException("Serialized MADDPG configuration does not match current agent options.");
        }

        void ReadNetwork(INeuralNetwork<T> network)
        {
            var length = reader.ReadInt32();
            var bytes = reader.ReadBytes(length);
            network.Deserialize(bytes);
        }

        foreach (var network in _actorNetworks)
        {
            ReadNetwork(network);
        }

        foreach (var network in _targetActorNetworks)
        {
            ReadNetwork(network);
        }

        foreach (var network in _criticNetworks)
        {
            ReadNetwork(network);
        }

        foreach (var network in _targetCriticNetworks)
        {
            ReadNetwork(network);
        }
    }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Collect actor network parameters
        foreach (var network in _actorNetworks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        // Collect critic network parameters
        foreach (var network in _criticNetworks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        // Collect target actor network parameters
        foreach (var network in _targetActorNetworks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        // Collect target critic network parameters
        foreach (var network in _targetCriticNetworks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        var paramVector = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            paramVector[i] = allParams[i];
        }

        return paramVector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Load actor network parameters
        foreach (var network in _actorNetworks)
        {
            int paramCount = checked((int)network.ParameterCount);
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }

        // Load critic network parameters
        foreach (var network in _criticNetworks)
        {
            int paramCount = checked((int)network.ParameterCount);
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }

        // Load target actor network parameters
        foreach (var network in _targetActorNetworks)
        {
            int paramCount = checked((int)network.ParameterCount);
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }

        // Load target critic network parameters
        foreach (var network in _targetCriticNetworks)
        {
            int paramCount = checked((int)network.ParameterCount);
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }

        // Synchronize target networks from main networks
        // This ensures targets match main networks after loading
        for (int i = 0; i < _actorNetworks.Count; i++)
        {
            var actorParams = _actorNetworks[i].GetParameters();
            _targetActorNetworks[i].UpdateParameters(actorParams);
        }

        for (int i = 0; i < _criticNetworks.Count; i++)
        {
            var criticParams = _criticNetworks[i].GetParameters();
            _targetCriticNetworks[i].UpdateParameters(criticParams);
        }
    }

    /// <summary>
    /// Creates a deep copy of this MADDPG agent including all trained network weights.
    /// </summary>
    /// <returns>A new MADDPG agent with the same configuration and trained parameters.</returns>
    /// <remarks>
    /// Issue #5 fix: Clone now properly copies all trained weights from actor and critic networks
    /// using GetParameters() and SetParameters(), ensuring the cloned agent has the same learned behavior.
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clonedAgent = new MADDPGAgent<T>(_options, _optimizer);

        // Copy all trained parameters to the cloned agent
        var currentParams = GetParameters();
        clonedAgent.SetParameters(currentParams);

        return clonedAgent;
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(prediction, target);

        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    /// <summary>
    /// Not supported for MADDPGAgent. Use the agent's internal Train() loop instead.
    /// </summary>
    /// <param name="gradients">Not used.</param>
    /// <param name="learningRate">Not used.</param>
    /// <exception cref="NotSupportedException">
    /// Always thrown. MADDPG manages gradient computation and parameter updates internally through backpropagation.
    /// </exception>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        throw new NotSupportedException(
            "ApplyGradients is not supported for MADDPGAgent; use the agent's internal Train() loop. " +
            "MADDPG manages gradient computation and parameter updates internally through backpropagation.");
    }

    /// <summary>
    /// Saves the trained model to a file.
    /// </summary>
    /// <param name="filepath">Path to save the model.</param>
    /// <remarks>
    /// Uses <see cref="Serialize"/> to persist network weights.
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Loads a trained model from a file.
    /// </summary>
    /// <param name="filepath">Path to load the model from.</param>
    /// <remarks>
    /// Uses <see cref="Deserialize"/> to restore network weights.
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
