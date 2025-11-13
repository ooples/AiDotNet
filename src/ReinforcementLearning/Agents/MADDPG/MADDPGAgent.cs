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

    private ReplayBuffer<T> _replayBuffer;
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

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
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
        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            architecture,
            hiddenLayerCount: _options.ActorHiddenLayers.Count,
            hiddenLayerSize: _options.ActorHiddenLayers.FirstOrDefault() > 0 ? _options.ActorHiddenLayers.First() : 128
        ).ToList();

        // Override final activation to Tanh for continuous actions
        var lastLayer = layers[layers.Count - 1];
        if (lastLayer is DenseLayer<T> denseLayer)
        {
            layers[layers.Count - 1] = new DenseLayer<T>(
                denseLayer.GetWeights().Rows,
                _options.ActionSize,
                new TanhActivation<T>()
            );
        }

        architecture.Layers = layers;
        return new NeuralNetwork<T>(architecture, _options.LossFunction);
    }

    private INeuralNetwork<T> CreateCriticNetwork()
    {
        // Centralized critic: observes all agents' states and actions
        int inputSize = (_options.StateSize + _options.ActionSize) * _options.NumAgents;

        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = inputSize,
            OutputSize = 1,
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            architecture,
            hiddenLayerCount: _options.CriticHiddenLayers.Count,
            hiddenLayerSize: _options.CriticHiddenLayers.FirstOrDefault() > 0 ? _options.CriticHiddenLayers.First() : 128
        );

        architecture.Layers = layers.ToList();
        return new NeuralNetwork<T>(architecture, _options.LossFunction);
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new ReplayBuffer<T>(_options.ReplayBufferSize);
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

        var action = _actorNetworks[agentId].Forward(state);

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
            var targetQ = _targetCriticNetworks[agentId].Forward(ConcatenateStateAction(experience.nextState, experience.action))[0];

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
            var currentQ = _criticNetworks[agentId].Forward(ConcatenateStateAction(experience.state, experience.action))[0];

            // TD error
            var error = NumOps.Subtract(target, currentQ);
            var loss = NumOps.Multiply(error, error);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagate
            var gradient = new Vector<T>(1);
            gradient[0] = error;
            _criticNetworks[agentId].Backward(gradient);
            _criticNetworks[agentId].UpdateWeights(_options.CriticLearningRate);
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
            var action = _actorNetworks[agentId].Forward(agentState);

            // Reconstruct joint action with this agent's new action
            var jointAction = experience.action.Clone();
            for (int i = 0; i < _options.ActionSize; i++)
            {
                jointAction[agentId * _options.ActionSize + i] = action[i];
            }

            // Compute Q-value (policy gradient)
            var qValue = _criticNetworks[agentId].Forward(ConcatenateStateAction(experience.state, jointAction))[0];

            // Actor loss: maximize Q-value
            totalLoss = NumOps.Add(totalLoss, NumOps.Negate(qValue));

            // Simplified gradient for actor
            var actorGradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                actorGradient[i] = NumOps.Divide(qValue, NumOps.FromDouble(_options.ActionSize));
            }

            _actorNetworks[agentId].Backward(actorGradient);
            _actorNetworks[agentId].UpdateWeights(_options.ActorLearningRate);
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

    public override Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public override Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = "MADDPG",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
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

    public override Matrix<T> GetParameters()
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

        return new Matrix<T>(new[] { paramVector });
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        int offset = 0;

        foreach (var network in _actorNetworks)
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

        foreach (var network in _criticNetworks)
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
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new MADDPGAgent<T>(_options, _optimizer);
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
        _actorNetworks[0].Backward(new Vector<T>(gradients.GetRow(0)));
        _actorNetworks[0].UpdateWeights(learningRate);
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
