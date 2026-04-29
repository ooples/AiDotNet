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
public class QMIXAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private QMIXOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    // Per-agent Q-networks
    private List<INeuralNetwork<T>> _agentNetworks;
    private List<INeuralNetwork<T>> _targetAgentNetworks;

    // Mixing network (combines agent Q-values)
    private INeuralNetwork<T> _mixingNetwork;
    private INeuralNetwork<T> _targetMixingNetwork;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private double _epsilon;
    private int _stepCount;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public QMIXAgent()
        : this(new QMIXOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public QMIXAgent(QMIXOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _epsilon = options.EpsilonStart;
        _agentNetworks = new List<INeuralNetwork<T>>();
        _targetAgentNetworks = new List<INeuralNetwork<T>>();
        _mixingNetwork = CreateMixingNetwork();
        _targetMixingNetwork = CreateMixingNetwork();
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);
        _stepCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

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

        foreach (var experience in batch)
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
            // Collect ALL trainable parameters from agent networks + mixing network
            var allParams = new List<Tensor<T>>();
            foreach (var agentNet in _agentNetworks)
            {
                if (agentNet is NeuralNetworkBase<T> nnBase)
                    allParams.AddRange(Training.TapeTrainingStep<T>.CollectParameters(nnBase.Layers));
            }
            if (_mixingNetwork is NeuralNetworkBase<T> mixBase)
                allParams.AddRange(Training.TapeTrainingStep<T>.CollectParameters(mixBase.Layers));

            var paramArray = allParams.ToArray();
            if (paramArray.Length > 0)
            {
                // Single tape recording: agents forward → mixing forward → MSE loss
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>();

                // Forward each agent under tape to get Q-values
                var agentQTensor = new Tensor<T>([_options.NumAgents]);
                for (int i = 0; i < _options.NumAgents; i++)
                {
                    var agentBase = (NeuralNetworkBase<T>)_agentNetworks[i];
                    var stateTensor = Tensor<T>.FromVector(agentStates[i], [1, _options.StateSize]);
                    var qOut = agentBase.ForwardForTraining(stateTensor);
                    int actionIdx = ArgMax(agentActions[i]);
                    agentQTensor[i] = qOut[actionIdx];
                }

                // Concatenate agent Q-values with global state for mixing input
                var mixInput = new Tensor<T>([1, _options.NumAgents + globalState.Length]);
                for (int i = 0; i < _options.NumAgents; i++)
                    mixInput[0, i] = agentQTensor[i];
                for (int i = 0; i < globalState.Length; i++)
                    mixInput[0, _options.NumAgents + i] = globalState[i];

                // Forward through mixing network under tape
                var mixBase2 = (NeuralNetworkBase<T>)_mixingNetwork;
                var qTotal = mixBase2.ForwardForTraining(mixInput);

                // MSE loss vs TD target via engine ops
                var targetScalar = new Tensor<T>([1]);
                targetScalar[0] = target;
                var diff = Engine.TensorSubtract(qTotal, targetScalar);
                var squared = Engine.TensorMultiply(diff, diff);
                var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
                var mseLoss = Engine.ReduceMean(squared, allAxes, keepDims: false);

                // Compute gradients for ALL parameters (agents + mixing) in one pass
                var grads = tape.ComputeGradients(mseLoss, paramArray);

                // Apply gradients in-place using learning rate
                foreach (var param in paramArray)
                {
                    if (grads.TryGetValue(param, out var grad))
                    {
                        for (int j = 0; j < param.Length && j < grad.Length; j++)
                            param[j] = NumOps.Subtract(param[j], NumOps.Multiply(LearningRate, grad[j]));
                    }
                }
            }

            // Enforce QMIX monotonicity: clamp mixing network weights to non-negative
            var mixingParams = _mixingNetwork.GetParameters();
            for (int j = 0; j < mixingParams.Length; j++)
            {
                if (NumOps.LessThan(mixingParams[j], NumOps.Zero))
                    mixingParams[j] = NumOps.Zero;
            }
            _mixingNetwork.UpdateParameters(mixingParams);
        }

        // Update target networks
        if (_stepCount % _options.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);
            for (int i = 0; i < _options.NumAgents; i++)
            {
                CopyNetworkWeights(_agentNetworks[i], _targetAgentNetworks[i]);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
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

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
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

    public override byte[] Serialize()
    {
        var parameters = GetParameters();
        var state = new
        {
            Parameters = parameters,
            NumAgents = _options.NumAgents,
            StateSize = _options.StateSize,
            ActionSize = _options.ActionSize
        };
        string json = JsonConvert.SerializeObject(state);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    public override void Deserialize(byte[] data)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Serialized data cannot be null or empty", nameof(data));
        }

        string json = System.Text.Encoding.UTF8.GetString(data);
        var state = JsonConvert.DeserializeObject<dynamic>(json);
        if (state is null)
        {
            throw new InvalidOperationException("Deserialization returned null");
        }

        var parameters = JsonConvert.DeserializeObject<Vector<T>>(state.Parameters.ToString());
        if (parameters is not null)
        {
            SetParameters(parameters);
        }
    }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var network in _agentNetworks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        var mixingParams = _mixingNetwork.GetParameters();
        for (int i = 0; i < mixingParams.Length; i++)
        {
            allParams.Add(mixingParams[i]);
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
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // Calculate expected parameter count
        int expectedParamCount = 0;
        foreach (var network in _agentNetworks)
        {
            expectedParamCount += network.ParameterCount;
        }
        expectedParamCount += _mixingNetwork.ParameterCount;

        if (parameters.Length != expectedParamCount)
        {
            throw new ArgumentException(
                $"Parameter vector length mismatch. Expected {expectedParamCount} parameters but got {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;

        foreach (var network in _agentNetworks)
        {
            int paramCount = network.ParameterCount;
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }

        int mixingParamCount = _mixingNetwork.ParameterCount;
        var mixingParams = new Vector<T>(mixingParamCount);
        for (int i = 0; i < mixingParamCount; i++)
        {
            mixingParams[i] = parameters[offset + i];

            // Enforce QMIX monotonicity: all mixing network weights must be non-negative
            if (NumOps.LessThan(mixingParams[i], NumOps.Zero))
            {
                mixingParams[i] = NumOps.Zero;
            }
        }
        _mixingNetwork.UpdateParameters(mixingParams);
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clonedAgent = new QMIXAgent<T>(_options, _optimizer);

        // Copy trained network parameters to the cloned agent
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
