using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

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
public class QMIXAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly QMIXOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // Per-agent Q-networks
    private List<NeuralNetwork<T>> _agentNetworks;
    private List<NeuralNetwork<T>> _targetAgentNetworks;

    // Mixing network (combines agent Q-values)
    private NeuralNetwork<T> _mixingNetwork;
    private NeuralNetwork<T> _targetMixingNetwork;

    private ReplayBuffer<T> _replayBuffer;
    private Random _random;
    private double _epsilon;
    private int _stepCount;

    public QMIXAgent(QMIXOptions<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _epsilon = options.EpsilonStart;
        _stepCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        _agentNetworks = new List<NeuralNetwork<T>>();
        _targetAgentNetworks = new List<NeuralNetwork<T>>();

        // Create Q-network for each agent
        for (int i = 0; i < _options.NumAgents; i++)
        {
            var agentNet = CreateAgentNetwork();
            var targetAgentNet = CreateAgentNetwork();
            CopyNetworkWeights(agentNet, targetAgentNet);

            _agentNetworks.Add(agentNet);
            _targetAgentNetworks.Add(targetAgentNet);
        }

        // Create mixing networks
        _mixingNetwork = CreateMixingNetwork();
        _targetMixingNetwork = CreateMixingNetwork();
        CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);
    }

    private NeuralNetwork<T> CreateAgentNetwork()
    {
        var network = new NeuralNetwork<T>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.AgentHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, _options.ActionSize));

        return network;
    }

    private NeuralNetwork<T> CreateMixingNetwork()
    {
        // Mixing network: (agent Q-values, global state) -> team Q-value
        var network = new NeuralNetwork<T>();
        int inputSize = _options.NumAgents + _options.GlobalStateSize;
        int previousSize = inputSize;

        foreach (var layerSize in _options.MixingHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        // Output: single team Q-value
        network.AddLayer(new DenseLayer<T>(previousSize, 1));

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new ReplayBuffer<T>(_options.ReplayBufferSize);
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

        if (training && _random.NextDouble() < _epsilon)
        {
            // Random exploration
            int randomAction = _random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = _numOps.One;
            return action;
        }

        // Greedy action
        var qValues = _agentNetworks[agentId].Forward(state);
        int bestAction = ArgMax(qValues);

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = _numOps.One;
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
            return _numOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = _numOps.Zero;

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
            var teamQ = _mixingNetwork.Forward(mixingInput)[0];

            // Compute target team Q-value
            var nextAgentQValues = new List<T>();
            for (int i = 0; i < _options.NumAgents; i++)
            {
                var nextQValues = _targetAgentNetworks[i].Forward(nextAgentStates[i]);
                nextAgentQValues.Add(MaxValue(nextQValues));
            }

            var targetMixingInput = ConcatenateMixingInput(nextAgentQValues, nextGlobalState);
            var targetTeamQ = _targetMixingNetwork.Forward(targetMixingInput)[0];

            T target;
            if (experience.done)
            {
                target = experience.reward;
            }
            else
            {
                target = _numOps.Add(experience.reward, _numOps.Multiply(_options.DiscountFactor, targetTeamQ));
            }

            // TD error
            var tdError = _numOps.Subtract(target, teamQ);
            var loss = _numOps.Multiply(tdError, tdError);
            totalLoss = _numOps.Add(totalLoss, loss);

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
                agentGradient[actionIdx] = _numOps.Divide(tdError, _numOps.FromDouble(_options.NumAgents));

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

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
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

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceLayers = source.GetLayers();
        var targetLayers = target.GetLayers();

        for (int i = 0; i < sourceLayers.Count; i++)
        {
            if (sourceLayers[i] is DenseLayer<T> sourceLayer && targetLayers[i] is DenseLayer<T> targetLayer)
            {
                targetLayer.SetWeights(sourceLayer.GetWeights().Clone());
                targetLayer.SetBiases(sourceLayer.GetBiases().Clone());
            }
        }
    }

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (_numOps.Compare(values[i], maxValue) > 0)
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
            if (_numOps.Compare(values[i], maxValue) > 0)
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
            ["steps"] = _numOps.FromDouble(_stepCount),
            ["buffer_size"] = _numOps.FromDouble(_replayBuffer.Count),
            ["epsilon"] = _numOps.FromDouble(_epsilon)
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

    public override Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public override Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }
}
