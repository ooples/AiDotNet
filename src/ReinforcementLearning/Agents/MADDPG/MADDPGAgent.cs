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

namespace AiDotNet.ReinforcementLearning.Agents.MADDPG;

/// <summary>
/// Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent.
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
public class MADDPGAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private MADDPGOptions<T> _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    // Networks for each agent
    private List<INeuralNetwork<T>> _actorNetworks;
    private List<INeuralNetwork<T>> _targetActorNetworks;
    private List<INeuralNetwork<T>> _criticNetworks;
    private List<INeuralNetwork<T>> _targetCriticNetworks;

    private UniformReplayBuffer<T> _replayBuffer;
    private int _stepCount;

    public MADDPGAgent(MADDPGOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
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
        _stepCount = 0;
        _replayBuffer = new UniformReplayBuffer<T>(_options.ReplayBufferSize);

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
        layers.Add(new DenseLayer<T>(_options.StateSize, _options.ActorHiddenLayers.First(), new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < _options.ActorHiddenLayers.Count; i++)
        {
            layers.Add(new DenseLayer<T>(_options.ActorHiddenLayers[i - 1], _options.ActorHiddenLayers[i], new ReLUActivation<T>()));
        }

        // Output layer with Tanh for continuous actions
        layers.Add(new DenseLayer<T>(_options.ActorHiddenLayers.Last(), _options.ActionSize, new TanhActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _options.LossFunction);
    }

    private INeuralNetwork<T> CreateCriticNetwork()
    {
        // Centralized critic: observes all agents' states and actions
        int inputSize = (_options.StateSize + _options.ActionSize) * _options.NumAgents;

        // Create layers
        var layers = new List<ILayer<T>>();

        // Input layer
        layers.Add(new DenseLayer<T>(inputSize, _options.CriticHiddenLayers.First(), new ReLUActivation<T>()));

        // Hidden layers
        for (int i = 1; i < _options.CriticHiddenLayers.Count; i++)
        {
            layers.Add(new DenseLayer<T>(_options.CriticHiddenLayers[i - 1], _options.CriticHiddenLayers[i], new ReLUActivation<T>()));
        }

        // Output layer (Q-value)
        layers.Add(new DenseLayer<T>(_options.CriticHiddenLayers.Last(), 1, new LinearActivation<T>()));

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
        _replayBuffer = new UniformReplayBuffer<T>(_options.ReplayBufferSize);
    }

    /// <summary>
    /// Select action for a specific agent.
    /// </summary>
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
    /// Store multi-agent experience.
    /// </summary>
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

        // Use average reward (or could be agent-specific)
        T avgReward = NumOps.Zero;
        foreach (var reward in rewards)
        {
            avgReward = NumOps.Add(avgReward, reward);
        }
        avgReward = NumOps.Divide(avgReward, NumOps.FromDouble(rewards.Count));

        _replayBuffer.Add(jointState, jointAction, avgReward, jointNextState, done);
        _stepCount++;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(state, action, reward, nextState, done);
        _stepCount++;
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.WarmupSteps || _replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;

        // Update each agent's critic and actor
        for (int agentId = 0; agentId < _options.NumAgents; agentId++)
        {
            T criticLoss = UpdateCritic(agentId, batch);
            T actorLoss = UpdateActor(agentId, batch);

            totalLoss = NumOps.Add(totalLoss, NumOps.Add(criticLoss, actorLoss));

            // Soft update target networks
            SoftUpdateTargetNetwork(_actorNetworks[agentId], _targetActorNetworks[agentId]);
            SoftUpdateTargetNetwork(_criticNetworks[agentId], _targetCriticNetworks[agentId]);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(_options.NumAgents * 2));
    }

    private T UpdateCritic(int agentId, List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target using target networks (centralized)
            var nextStateActionInput = ConcatenateStateAction(experience.nextState, experience.action);
            var nextStateActionTensor = Tensor<T>.FromVector(nextStateActionInput);
            var targetQTensor = _targetCriticNetworks[agentId].Predict(nextStateActionTensor);
            var targetQ = targetQTensor.ToVector()[0];

            T target;
            if (experience.done)
            {
                target = experience.reward;
            }
            else
            {
                target = NumOps.Add(experience.reward, NumOps.Multiply(_options.DiscountFactor, targetQ));
            }

            // Current Q-value
            var currentStateActionInput = ConcatenateStateAction(experience.state, experience.action);
            var currentStateActionTensor = Tensor<T>.FromVector(currentStateActionInput);
            var currentQTensor = _criticNetworks[agentId].Predict(currentStateActionTensor);
            var currentQ = currentQTensor.ToVector()[0];

            // TD error
            var error = NumOps.Subtract(target, currentQ);
            var loss = NumOps.Multiply(error, error);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagate
            var gradientVec = new Vector<T>(1);
            gradientVec[0] = error;
            var gradientTensor = Tensor<T>.FromVector(gradientVec);
            _criticNetworks[agentId].Backpropagate(currentStateActionTensor, gradientTensor);
            _criticNetworks[agentId].UpdateParameters(_options.CriticLearningRate);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T UpdateActor(int agentId, List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Decompose joint state to get this agent's state
            int stateOffset = agentId * _options.StateSize;
            var agentState = new Vector<T>(_options.StateSize);
            for (int i = 0; i < _options.StateSize; i++)
            {
                agentState[i] = experience.state[stateOffset + i];
            }

            // Compute action from actor
            var agentStateTensor = Tensor<T>.FromVector(agentState);
            var actionTensor = _actorNetworks[agentId].Predict(agentStateTensor);
            var action = actionTensor.ToVector();

            // Reconstruct joint action with this agent's new action
            var jointAction = experience.action.Clone();
            for (int i = 0; i < _options.ActionSize; i++)
            {
                jointAction[agentId * _options.ActionSize + i] = action[i];
            }

            // Compute Q-value (policy gradient)
            var jointStateAction = ConcatenateStateAction(experience.state, jointAction);
            var jointStateActionTensor = Tensor<T>.FromVector(jointStateAction);
            var qValueTensor = _criticNetworks[agentId].Predict(jointStateActionTensor);
            var qValue = qValueTensor.ToVector()[0];

            // Actor loss: maximize Q-value
            totalLoss = NumOps.Add(totalLoss, NumOps.Negate(qValue));

            // Simplified gradient for actor
            var actorGradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                actorGradient[i] = NumOps.Divide(qValue, NumOps.FromDouble(_options.ActionSize));
            }

            var actorGradientTensor = Tensor<T>.FromVector(actorGradient);
            _actorNetworks[agentId].Backpropagate(agentStateTensor, actorGradientTensor);
            _actorNetworks[agentId].UpdateParameters(_options.ActorLearningRate);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
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
            ModelType = Enums.ModelType.ReinforcementLearning,
        };
    }

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("MADDPG serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("MADDPG deserialization not yet implemented");
    }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var network in _actorNetworks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        foreach (var network in _criticNetworks)
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

        foreach (var network in _actorNetworks)
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

        foreach (var network in _criticNetworks)
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
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new MADDPGAgent<T>(_options, _optimizer);
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(prediction, target);

        var gradient = usedLossFunction.CalculateGradient(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var currentParams = GetParameters();
        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var update = NumOps.Multiply(learningRate, gradients[i]);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        SetParameters(newParams);
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
