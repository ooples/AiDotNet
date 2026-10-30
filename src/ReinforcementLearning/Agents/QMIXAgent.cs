using AiDotNet.LearningRateSchedulers;
using System.Linq;
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

namespace AiDotNet.ReinforcementLearning.Agents.QMIX;

/// <summary>
/// QMIX agent for multi-agent value-based reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// QMIX factorizes joint action-values into per-agent values using a mixing network
/// that monotonically combines them.
/// </para>
/// <para><b>For Beginners:</b>
/// QMIX solves multi-agent problems by letting each agent learn its own Q-values,
/// then using a "mixing network" to combine them into a team Q-value.
///
/// Key innovation:
/// - **Value Factorization**: Team value = mix(agent1_Q, agent2_Q, ...)
/// - **Mixing Network**: Ensures individual and joint actions are consistent
/// - **Monotonicity**: If one agent improves, team improves
/// - **Decentralized Execution**: Each agent acts independently
///
/// Think of it like: Each player estimates their contribution, and a coach
/// combines these to determine the team's overall score.
///
/// Famous for: StarCraft II micromanagement, cooperative games
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a QMIX agent for cooperative multi-agent tasks
/// var options = new QMIXOptions&lt;double&gt; { NumAgents = 3, StateSize = 8, ActionSize = 4 };
/// var agent = new QMIXAgent&lt;double&gt;(options);
///
/// // Each agent selects an action from its local observation
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2, 0.8, -0.1, 0.4, 0.6 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
    "https://arxiv.org/abs/1803.11485",
    Year = 2018,
    Authors = "Rashid, T., Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J., & Whiteson, S.")]
[PaperOptimizer(OptimizerKind.RmsProp, LearningRate = 5e-4,
                Source = "Rashid et al. 2018, Sec. 5: the learning rate for RMSprop is set to 5e-4.")]
public partial class QMIXAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        foreach (var network in _agentNetworks) RegisterParameterComponent(network);
        RegisterParameterComponent(_mixingNetwork);
    }
    private QMIXOptions<T> _options;

    // Per-agent Q-networks
    private List<INeuralNetwork<T>> _agentNetworks;
    [Buffer]
    private List<INeuralNetwork<T>> _targetAgentNetworks;

    // Mixing network (combines agent Q-values)
    private INeuralNetwork<T> _mixingNetwork;
    [Buffer]
    private INeuralNetwork<T> _targetMixingNetwork;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private double _epsilon;
    private int _stepCount;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public QMIXAgent()
        : this(new QMIXOptions<T> { NumAgents = 2, StateSize = 4, ActionSize = 2, GlobalStateSize = 8 })
    {
    }

    public QMIXAgent(QMIXOptions<T> options)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _epsilon = options.EpsilonStart;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);
        _stepCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    // The constructor used to build a first pair of mixing networks here and then call this method, which
    // built a second pair and registered only that one; the first pair was dropped without ever being
    // disposed. This method is now the only place the networks are created.
    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_agentNetworks), nameof(_targetAgentNetworks), nameof(_mixingNetwork), nameof(_targetMixingNetwork))]
    private void InitializeNetworks()
    {
        _agentNetworks = new List<INeuralNetwork<T>>();
        _targetAgentNetworks = new List<INeuralNetwork<T>>();
        _mixingNetwork = CreateMixingNetwork();
        _targetMixingNetwork = CreateMixingNetwork();

        // Create Q-network for each agent
        for (int i = 0; i < _options.NumAgents; i++)
        {
            var agentNet = CreateAgentNetwork();
            var targetAgentNet = CreateAgentNetwork();
            CopyNetworkWeights(agentNet, targetAgentNet);

            _agentNetworks.Add(agentNet);
            _targetAgentNetworks.Add(targetAgentNet);

            // Register agent networks with base class
            Networks.Add(agentNet);
            Networks.Add(targetAgentNet);
        }

        CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);

        // Register mixing networks with base class
        Networks.Add(_mixingNetwork);
        Networks.Add(_targetMixingNetwork);
    }

    private INeuralNetwork<T> CreateAgentNetwork()
    {
        // Create layers
        var layers = new List<ILayer<T>>();

        // Use configured hidden layer sizes or defaults
        var hiddenSizes = _options.AgentHiddenLayers;
        if (hiddenSizes is null || hiddenSizes.Count == 0)
        {
            hiddenSizes = new List<int> { 64, 64 };
        }

        // Input layer
        layers.Add(new DenseLayer<T>(hiddenSizes[0], (IActivationFunction<T>)new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < hiddenSizes.Count; i++)
        {
            layers.Add(new DenseLayer<T>(hiddenSizes[i], (IActivationFunction<T>)new ReLUActivation<T>()));
        }

        // Output layer (Q-values for each action)
        int lastHiddenSize = hiddenSizes[hiddenSizes.Count - 1];
        layers.Add(new DenseLayer<T>(_options.ActionSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture, lossFunction: _options.LossFunction);
    }

    private INeuralNetwork<T> CreateMixingNetwork()
    {
        // Mixing network: (agent Q-values, global state) -> team Q-value
        int inputSize = _options.NumAgents + _options.GlobalStateSize;

        // Create layers
        var layers = new List<ILayer<T>>();

        // Use configured hidden layer sizes or defaults
        var hiddenSizes = _options.MixingHiddenLayers;
        if (hiddenSizes is null || hiddenSizes.Count == 0)
        {
            hiddenSizes = new List<int> { 64 };
        }

        // Input layer
        layers.Add(new DenseLayer<T>(hiddenSizes[0], (IActivationFunction<T>)new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < hiddenSizes.Count; i++)
        {
            layers.Add(new DenseLayer<T>(hiddenSizes[i], (IActivationFunction<T>)new ReLUActivation<T>()));
        }

        // Output layer (team Q-value) - connect from last hidden layer
        int lastHiddenSize = hiddenSizes[hiddenSizes.Count - 1];
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
    /// Select action for a specific agent using epsilon-greedy.
    /// </summary>
    public Vector<T> SelectActionForAgent(int agentId, Vector<T> state, bool training = true)
    {
        if (agentId < 0 || agentId >= _options.NumAgents)
        {
            throw new ArgumentException($"Invalid agent ID: {agentId}");
        }

        if (training && Random.NextDouble() < _epsilon)
        {
            // Random exploration
            int randomAction = Random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Greedy action
        var stateTensor = Tensor<T>.FromVector(state);
        var qValuesTensor = _agentNetworks[agentId].Predict(stateTensor);
        var qValues = qValuesTensor.ToVector();
        int bestAction = ArgMax(qValues);

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Default to agent 0
        return SelectActionForAgent(0, state, training);
    }

    /// <summary>
    /// Store multi-agent experience with global state.
    /// </summary>
    public void StoreMultiAgentExperience(
        List<Vector<T>> agentStates,
        List<Vector<T>> agentActions,
        T teamReward,
        List<Vector<T>> nextAgentStates,
        Vector<T> globalState,
        Vector<T> nextGlobalState,
        bool done)
    {
        // Concatenate for storage
        var jointState = ConcatenateWithGlobal(agentStates, globalState);
        var jointAction = ConcatenateVectors(agentActions);
        var jointNextState = ConcatenateWithGlobal(nextAgentStates, nextGlobalState);

        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(jointState, jointAction, teamReward, jointNextState, done));
        _stepCount++;

        // Decay epsilon
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
        _stepCount++;
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;

        // QMIX trains on JOINT observations: each stored state must be a concatenation of
        // every agent's observation plus the global state. If a stored transition is not
        // joint-sized (e.g. a single-agent state vector was passed), the joint decomposition
        // below would read out of bounds. Filter those malformed transitions OUT of the batch
        // and train on the rest, rather than aborting the whole update on one bad experience —
        // one malformed transition shouldn't block learning on the valid majority.
        int expectedJointLength = _options.NumAgents * _options.StateSize + _options.GlobalStateSize;
        var validBatch = batch
            .Where(e => e.State.Length >= expectedJointLength && e.NextState.Length >= expectedJointLength)
            .ToList();
        if (validBatch.Count == 0)
        {
            return NumOps.Zero;
        }

        foreach (var experience in validBatch)
        {
            // Decompose joint experience
            var (agentStates, globalState, agentActions) = DecomposeJointState(experience.State, experience.Action);
            var (nextAgentStates, nextGlobalState, _) = DecomposeJointState(experience.NextState, experience.Action);

            // Compute individual agent Q-values
            var agentQValues = new List<T>();
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var stateTensor = Tensor<T>.FromVector(agentStates[i]);
                var qValuesTensor = _agentNetworks[i].Predict(stateTensor);
                var qValues = qValuesTensor.ToVector();
                int actionIdx = ArgMax(agentActions[i]);
                agentQValues.Add(qValues[actionIdx]);
            }

            // Mix agent Q-values to get team Q-value
            var mixingInput = ConcatenateMixingInput(agentQValues, globalState);
            var mixingInputTensor = Tensor<T>.FromVector(mixingInput);
            var teamQTensor = _mixingNetwork.Predict(mixingInputTensor);
            var teamQ = teamQTensor.ToVector()[0];

            // Compute target team Q-value
            var nextAgentQValues = new List<T>();
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var nextStateTensor = Tensor<T>.FromVector(nextAgentStates[i]);
                var nextQValuesTensor = _targetAgentNetworks[i].Predict(nextStateTensor);
                var nextQValues = nextQValuesTensor.ToVector();
                nextAgentQValues.Add(MaxValue(nextQValues));
            }

            var targetMixingInput = ConcatenateMixingInput(nextAgentQValues, nextGlobalState);
            var targetMixingInputTensor = Tensor<T>.FromVector(targetMixingInput);
            var targetTeamQTensor = _targetMixingNetwork.Predict(targetMixingInputTensor);
            var targetTeamQ = targetTeamQTensor.ToVector()[0];

            T target;
            if (experience.Done)
            {
                target = experience.Reward;
            }
            else
            {
                target = NumOps.Add(experience.Reward, NumOps.Multiply(DiscountFactor, targetTeamQ));
            }

            // TD error for monitoring
            var tdError = NumOps.Subtract(target, teamQ);
            var loss = NumOps.Multiply(tdError, tdError);
            totalLoss = NumOps.Add(totalLoss, loss);

            // --- End-to-end training: agents → mixing → loss (Rashid et al. 2018) ---
            // One step over the agent networks and the mixing network together, on the shared training step (plain SGD
            // at the agent's learning rate, as before). Each agent's chosen Q-value is selected with a one-hot dot
            // product and the mixing input is a recorded concatenation, so the TD error reaches the agents: the chosen
            // value used to be copied by host indexing into a fresh tensor, which cut the agents off and left only the
            // mixing network training. The states, the one-hot actions, the global state and the detached TD target
            // travel in the step's data.
            var stepData = new List<Tensor<T>>();
            for (int i = 0; i < _options.NumAgents; i++)
            {
                stepData.Add(Tensor<T>.FromVector(agentStates[i], [1, _options.StateSize]));
                stepData.Add(Tensor<T>.FromVector(OneHotAction(agentActions[i]), [1, _options.ActionSize]));
            }
            stepData.Add(Tensor<T>.FromVector(globalState, [1, globalState.Length]));
            var targetScalar = new Tensor<T>([1, 1]);
            targetScalar[0] = target;
            stepData.Add(targetScalar);
            var packed = Training.StepInputPacking<T>.Pack(stepData.ToArray());
            TrainJointStep(packed, globalState.Length);

            // Enforce QMIX monotonicity: the mixing weights stay non-negative. Clamped in place, in the live tensors,
            // so the optimizer and any compiled plan keep training the same ones.
            ClampMixingWeightsNonNegative();        }

        // Update target networks
        if (_stepCount % _options.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);
            for (int i = 0; i < _options.NumAgents; i++)
            {
                CopyNetworkWeights(_agentNetworks[i], _targetAgentNetworks[i]);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(validBatch.Count));
    }

    private Training.TapeTrainingStepper<T>? _jointStepper;
    // The joint step's optimizer, built on first use: Options.OptimizerFactory's when set, else plain SGD.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _jointOptimizer;

    /// <summary>One joint-optimizer step on the TD loss over the agent and mixing networks, from a packed step.</summary>
    private void TrainJointStep(Tensor<T> packed, int globalStateLength)
    {
        var agentBases = _agentNetworks.Cast<NeuralNetworkBase<T>>().ToList();
        var mixBase = (NeuralNetworkBase<T>)_mixingNetwork;
        var layers = agentBases.SelectMany(n => n.Layers).Concat(mixBase.Layers).ToList();
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(layers);
        if (parameters.Count == 0) return;

        int stateSize = _options.StateSize;
        int actionSize = _options.ActionSize;
        int numAgents = _options.NumAgents;
        Tensor<T> TdLoss(Tensor<T> step)
        {
            var mixingInputs = new List<Tensor<T>>(numAgents + 1);
            int offset = 0;
            for (int i = 0; i < numAgents; i++)
            {
                var state = Training.StepInputPacking<T>.Slice(step, offset, new[] { 1, stateSize });
                offset += stateSize;
                var oneHot = Training.StepInputPacking<T>.Slice(step, offset, new[] { 1, actionSize });
                offset += actionSize;
                var qOut = Engine.Reshape(agentBases[i].ForwardForTraining(state), new[] { 1, actionSize });
                mixingInputs.Add(Engine.ReduceSum(Engine.TensorMultiply(qOut, oneHot), new[] { 1 }, keepDims: true)); // [1, 1]
            }
            mixingInputs.Add(Training.StepInputPacking<T>.Slice(step, offset, new[] { 1, globalStateLength }));
            offset += globalStateLength;
            var qTotal = Engine.Reshape(mixBase.ForwardForTraining(Engine.TensorConcatenate(mixingInputs.ToArray(), 1)), new[] { 1, 1 });
            var diff = Engine.TensorSubtract(qTotal, Training.StepInputPacking<T>.Slice(step, offset, new[] { 1, 1 }));
            return Engine.ReduceMean(Engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
        }

        _jointOptimizer ??= _options.OptimizerFactory?.Invoke() ?? new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(null,
            new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = Convert.ToDouble(LearningRate),
                EnableGradientClipping = false,
            });
        var stepper = _jointStepper ??= new Training.TapeTrainingStepper<T>(this);
        stepper.Step(new Training.FusedTrainingStepRequest<T>
        {
            Layers = layers.OfType<ITrainableLayer<T>>().ToList(),
            Input = packed,
            Target = packed,
            Forward = TdLoss,
            ComputeLoss = (loss, _) => loss,
            Optimizer = _jointOptimizer,
            ExtraParameters = parameters,
        });
    }

    /// <summary>The action as a one-hot row over ActionSize: the chosen index's arg-max.</summary>
    private Vector<T> OneHotAction(Vector<T> action)
    {
        var oneHot = new Vector<T>(_options.ActionSize);
        oneHot[ArgMax(action)] = NumOps.One;
        return oneHot;
    }

    /// <summary>Clamps the mixing network's trainable weights to be non-negative, in place.</summary>
    private void ClampMixingWeightsNonNegative()
    {
        var mixBase = (NeuralNetworkBase<T>)_mixingNetwork;
        foreach (var parameter in Training.TapeTrainingStep<T>.CollectParameters(mixBase.Layers))
        {
            for (int j = 0; j < parameter.Length; j++)
            {
                if (NumOps.LessThan(parameter.GetFlat(j), NumOps.Zero))
                    parameter.SetFlat(j, NumOps.Zero);
            }
        }
    }
    private (List<Vector<T>> agentStates, Vector<T> globalState, List<Vector<T>> agentActions) DecomposeJointState(
        Vector<T> jointState, Vector<T> jointAction)
    {
        var agentStates = new List<Vector<T>>();
        for (int i = 0; i < _options.NumAgents; i++)
        {
            var state = new Vector<T>(_options.StateSize);
            for (int j = 0; j < _options.StateSize; j++)
            {
                state[j] = jointState[i * _options.StateSize + j];
            }
            agentStates.Add(state);
        }

        // Extract global state
        int globalOffset = _options.NumAgents * _options.StateSize;
        var globalState = new Vector<T>(_options.GlobalStateSize);
        for (int i = 0; i < _options.GlobalStateSize; i++)
        {
            globalState[i] = jointState[globalOffset + i];
        }

        // Decompose actions
        var agentActions = new List<Vector<T>>();
        for (int i = 0; i < _options.NumAgents; i++)
        {
            var action = new Vector<T>(_options.ActionSize);
            for (int j = 0; j < _options.ActionSize; j++)
            {
                action[j] = jointAction[i * _options.ActionSize + j];
            }
            agentActions.Add(action);
        }

        return (agentStates, globalState, agentActions);
    }

    private Vector<T> ConcatenateMixingInput(List<T> agentQValues, Vector<T> globalState)
    {
        var input = new Vector<T>(agentQValues.Count + globalState.Length);
        for (int i = 0; i < agentQValues.Count; i++)
        {
            input[i] = agentQValues[i];
        }
        for (int i = 0; i < globalState.Length; i++)
        {
            input[agentQValues.Count + i] = globalState[i];
        }
        return input;
    }

    private Vector<T> ComputeMixingInputGradient(Vector<T> mixingInput, T tdError)
    {
        // Compute gradient w.r.t. agent Q-values from mixing network
        // For QMIX, the mixing network is monotonic, so gradients flow through
        // Approximate: each agent gets gradient proportional to TD error
        // Better approach would use actual backprop through mixing network layers

        var gradient = new Vector<T>(_options.NumAgents);
        T baseGradient = NumOps.Multiply(NumOps.FromDouble(-2.0), tdError);
        T perAgentGradient = NumOps.Divide(baseGradient, NumOps.FromDouble(_options.NumAgents));

        for (int i = 0; i < _options.NumAgents; i++)
        {
            gradient[i] = perAgentGradient;
        }

        return gradient;
    }

    private Vector<T> ConcatenateWithGlobal(List<Vector<T>> agentVectors, Vector<T> globalVector)
    {
        if (agentVectors is null || agentVectors.Count == 0)
        {
            throw new ArgumentException("Agent vectors list cannot be null or empty.", nameof(agentVectors));
        }

        if (globalVector is null)
        {
            throw new ArgumentNullException(nameof(globalVector));
        }

        // Validate all agent vectors have the same length
        int vectorLength = agentVectors[0].Length;
        for (int i = 1; i < agentVectors.Count; i++)
        {
            if (agentVectors[i] is null)
            {
                throw new ArgumentException($"Agent vector at index {i} is null.", nameof(agentVectors));
            }
            if (agentVectors[i].Length != vectorLength)
            {
                throw new ArgumentException(
                    $"All agent vectors must have the same length. Expected {vectorLength} but got {agentVectors[i].Length} at index {i}.",
                    nameof(agentVectors));
            }
        }

        int totalSize = agentVectors.Count * vectorLength + globalVector.Length;
        var result = new Vector<T>(totalSize);
        int offset = 0;

        foreach (var vec in agentVectors)
        {
            for (int i = 0; i < vec.Length; i++)
            {
                result[offset + i] = vec[i];
            }
            offset += vec.Length;
        }

        for (int i = 0; i < globalVector.Length; i++)
        {
            result[offset + i] = globalVector[i];
        }

        return result;
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

    private int ArgMax(Vector<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            throw new ArgumentException("Cannot compute ArgMax of an empty vector.", nameof(values));
        }

        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private T MaxValue(Vector<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            throw new ArgumentException("Cannot compute MaxValue of an empty vector.", nameof(values));
        }

        T maxValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
            }
        }
        return maxValue;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["steps"] = NumOps.FromDouble(_stepCount),
            ["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count),
            ["epsilon"] = NumOps.FromDouble(_epsilon)
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
    /// Computes gradients of the loss with respect to this agent's parameters, without updating them.
    /// </summary>
    /// <param name="input">The input state.</param>
    /// <param name="target">The target output.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the same length as <c>GetParameters()</c>.</returns>
    /// <remarks>
    /// Returns PARAMETER-space gradients, as <c>IGradientComputable</c> documents and as
    /// <c>ApplyGradients</c> requires. The gradient comes from the network's own tape pass, so no
    /// derivative is written by hand here.
    /// </remarks>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        return ComputeGradientsForNetwork(
            _mixingNetwork,
            new[] { _mixingNetwork },
            input,
            target,
            lossFunction);
    }

    public override void SaveModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        if (!System.IO.File.Exists(filepath))
        {
            throw new System.IO.FileNotFoundException($"Model file not found: {filepath}", filepath);
        }

        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    

}
}
