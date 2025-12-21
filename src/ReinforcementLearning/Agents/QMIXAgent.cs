using AiDotNet.ActivationFunctions;
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
public class QMIXAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private QMIXOptions<T> _options;
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

    public QMIXAgent(QMIXOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            LearningRate = 0.001,
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
        layers.Add(new DenseLayer<T>(_options.StateSize, hiddenSizes[0], (IActivationFunction<T>)new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < hiddenSizes.Count; i++)
        {
            layers.Add(new DenseLayer<T>(hiddenSizes[i - 1], hiddenSizes[i], (IActivationFunction<T>)new ReLUActivation<T>()));
        }

        // Output layer (Q-values for each action)
        int lastHiddenSize = hiddenSizes[hiddenSizes.Count - 1];
        layers.Add(new DenseLayer<T>(lastHiddenSize, _options.ActionSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _options.LossFunction);
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
        layers.Add(new DenseLayer<T>(inputSize, hiddenSizes[0], (IActivationFunction<T>)new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < hiddenSizes.Count; i++)
        {
            layers.Add(new DenseLayer<T>(hiddenSizes[i - 1], hiddenSizes[i], (IActivationFunction<T>)new ReLUActivation<T>()));
        }

        // Output layer (team Q-value) - connect from last hidden layer
        int lastHiddenSize = hiddenSizes[hiddenSizes.Count - 1];
        layers.Add(new DenseLayer<T>(lastHiddenSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _options.LossFunction);
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

            // TD error
            var tdError = NumOps.Subtract(target, teamQ);
            var loss = NumOps.Multiply(tdError, tdError);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagate through mixing network
            // TD loss gradient: d/dQ[loss] = d/dQ[(target - Q)^2] = -2 * (target - Q) = -2 * tdError
            var mixingGradientVec = new Vector<T>(1);
            mixingGradientVec[0] = NumOps.Multiply(NumOps.FromDouble(-2.0), tdError);
            var mixingGradient = Tensor<T>.FromVector(mixingGradientVec);
            ((NeuralNetwork<T>)_mixingNetwork).Backpropagate(mixingGradient);

            // Get gradient w.r.t. mixing network inputs (agent Q-values) for gradient flow
            // This should be obtained from the mixing network's input gradient after backprop
            // For now, approximate using chain rule: dL/dQ_i = dL/dQ_total * dQ_total/dQ_i
            // In QMIX, mixing network is monotonic, so gradient flows proportionally
            var mixingInputGradient = ComputeMixingInputGradient(mixingInput, tdError);

            // Manual parameter update for mixing network
            var mixingParams = _mixingNetwork.GetParameters();
            var mixingGrads = ((NeuralNetwork<T>)_mixingNetwork).GetGradients();
            for (int j = 0; j < mixingParams.Length; j++)
            {
                mixingParams[j] = NumOps.Subtract(mixingParams[j],
                    NumOps.Multiply(LearningRate, mixingGrads[j]));

                // Enforce QMIX monotonicity: all mixing network weights must be non-negative
                // This ensures that increasing any agent's Q-value increases the team Q-value
                if (NumOps.LessThan(mixingParams[j], NumOps.Zero))
                {
                    mixingParams[j] = NumOps.Zero;
                }
            }
            _mixingNetwork.UpdateParameters(mixingParams);

            // Backpropagate through agent networks using gradient from mixing network
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var agentGradientVec = new Vector<T>(_options.ActionSize);
                int actionIdx = ArgMax(agentActions[i]);
                // Use gradient flow from mixing network, not just tdError / NumAgents
                T agentQGradient = mixingInputGradient[i];
                agentGradientVec[actionIdx] = agentQGradient;

                var stateTensor = Tensor<T>.FromVector(agentStates[i]);
                var agentGradient = Tensor<T>.FromVector(agentGradientVec);
                ((NeuralNetwork<T>)_agentNetworks[i]).Backpropagate(agentGradient);

                // Manual parameter update with learning rate
                var agentParams = _agentNetworks[i].GetParameters();
                var agentGrads = ((NeuralNetwork<T>)_agentNetworks[i]).GetGradients();
                for (int j = 0; j < agentParams.Length; j++)
                {
                    agentParams[j] = NumOps.Subtract(agentParams[j],
                        NumOps.Multiply(LearningRate, agentGrads[j]));
                }
                _agentNetworks[i].UpdateParameters(agentParams);
            }
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
            ModelType = ModelType.ReinforcementLearning,
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

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var gradientsTensor = Tensor<T>.FromVector(gradients);
        ((NeuralNetwork<T>)_agentNetworks[0]).Backpropagate(gradientsTensor);

        // Manual parameter update with learning rate
        var agentParams = _agentNetworks[0].GetParameters();
        var agentGrads = ((NeuralNetwork<T>)_agentNetworks[0]).GetGradients();
        for (int i = 0; i < agentParams.Length; i++)
        {
            agentParams[i] = NumOps.Subtract(agentParams[i],
                NumOps.Multiply(learningRate, agentGrads[i]));
        }
        _agentNetworks[0].UpdateParameters(agentParams);
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
