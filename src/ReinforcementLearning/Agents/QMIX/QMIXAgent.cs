using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;

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

    private UniformReplayBuffer<T> _replayBuffer;
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
        _stepCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        _agentNetworks = new List<INeuralNetwork<T>>();
        _targetAgentNetworks = new List<INeuralNetwork<T>>();

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

        // Create mixing networks
        _mixingNetwork = CreateMixingNetwork();
        _targetMixingNetwork = CreateMixingNetwork();
        CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);

        // Register mixing networks with base class
        Networks.Add(_mixingNetwork);
        Networks.Add(_targetMixingNetwork);
    }

    private INeuralNetwork<T> CreateAgentNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create Q-network layers
        var layers = LayerHelper<T>.CreateDefaultDeepQNetworkLayers(architecture);

        architecture.Layers = layers.ToList();
        return new NeuralNetwork<T>(architecture, _options.LossFunction);
    }

    private INeuralNetwork<T> CreateMixingNetwork()
    {
        // Mixing network: (agent Q-values, global state) -> team Q-value
        int inputSize = _options.NumAgents + _options.GlobalStateSize;

        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = inputSize,
            OutputSize = 1,
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            architecture,
            hiddenLayerCount: _options.MixingHiddenLayers.Count,
            hiddenLayerSize: _options.MixingHiddenLayers.FirstOrDefault() > 0 ? _options.MixingHiddenLayers.First() : 64
        );

        architecture.Layers = layers.ToList();
        return new NeuralNetwork<T>(architecture, _options.LossFunction);
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new UniformReplayBuffer<T>(_options.ReplayBufferSize);
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
        var qValues = _agentNetworks[agentId].Forward(state);
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

        _replayBuffer.Add(jointState, jointAction, teamReward, jointNextState, done);
        _stepCount++;

        // Decay epsilon
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(state, action, reward, nextState, done);
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
            var (agentStates, globalState, agentActions) = DecomposeJointState(experience.state, experience.action);
            var (nextAgentStates, nextGlobalState, _) = DecomposeJointState(experience.nextState, experience.action);

            // Compute individual agent Q-values
            var agentQValues = new List<T>();
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var qValues = _agentNetworks[i].Forward(agentStates[i]);
                int actionIdx = ArgMax(agentActions[i]);
                agentQValues.Add(qValues[actionIdx]);
            }

            // Mix agent Q-values to get team Q-value
            var mixingInput = ConcatenateMixingInput(agentQValues, globalState);
            var teamQ = _mixingNetwork.Predict(mixingInput)[0];

            // Compute target team Q-value
            var nextAgentQValues = new List<T>();
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var nextQValues = _targetAgentNetworks[i].Forward(nextAgentStates[i]);
                nextAgentQValues.Add(MaxValue(nextQValues));
            }

            var targetMixingInput = ConcatenateMixingInput(nextAgentQValues, nextGlobalState);
            var targetTeamQ = _targetMixingNetwork.Predict(targetMixingInput)[0];

            T target;
            if (experience.done)
            {
                target = experience.reward;
            }
            else
            {
                target = NumOps.Add(experience.reward, NumOps.Multiply(_options.DiscountFactor, targetTeamQ));
            }

            // TD error
            var tdError = NumOps.Subtract(target, teamQ);
            var loss = NumOps.Multiply(tdError, tdError);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagate through mixing network
            var mixingGradient = new Vector<T>(1);
            mixingGradient[0] = tdError;
            _mixingNetwork.Backward(mixingGradient);
            _mixingNetwork.UpdateWeights(_options.LearningRate);

            // Backpropagate through agent networks
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var agentGradient = new Vector<T>(_options.ActionSize);
                int actionIdx = ArgMax(agentActions[i]);
                agentGradient[actionIdx] = NumOps.Divide(tdError, NumOps.FromDouble(_options.NumAgents));

                _agentNetworks[i].Backward(agentGradient);
                _agentNetworks[i].UpdateWeights(_options.LearningRate);
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

    private Vector<T> ConcatenateWithGlobal(List<Vector<T>> agentVectors, Vector<T> globalVector)
    {
        int totalSize = agentVectors.Count * agentVectors[0].Length + globalVector.Length;
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
            ModelType = "QMIX",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("QMIX serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("QMIX deserialization not yet implemented");
    }

    public override Matrix<T> GetParameters()
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

        return new Matrix<T>(new[] { paramVector });
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        int offset = 0;

        foreach (var network in _agentNetworks)
        {
            int paramCount = network.ParameterCount;
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[0, offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }

        int mixingParamCount = _mixingNetwork.ParameterCount;
        var mixingParams = new Vector<T>(mixingParamCount);
        for (int i = 0; i < mixingParamCount; i++)
        {
            mixingParams[i] = parameters[0, offset + i];
        }
        _mixingNetwork.UpdateParameters(mixingParams);
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new QMIXAgent<T>(_options, _optimizer);
    }

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));

        var gradient = usedLossFunction.CalculateDerivative(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));
        return (gradient, loss);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        _agentNetworks[0].Backward(new Vector<T>(gradients.GetRow(0)));
        _agentNetworks[0].UpdateWeights(learningRate);
    }

    public override void Save(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void Load(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
