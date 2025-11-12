using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

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
public class MADDPGAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly MADDPGOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // Networks for each agent
    private List<NeuralNetwork<T>> _actorNetworks;
    private List<NeuralNetwork<T>> _targetActorNetworks;
    private List<NeuralNetwork<T>> _criticNetworks;
    private List<NeuralNetwork<T>> _targetCriticNetworks;

    private ReplayBuffer<T> _replayBuffer;
    private Random _random;
    private int _stepCount;

    public MADDPGAgent(MADDPGOptions<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _stepCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        _actorNetworks = new List<NeuralNetwork<T>>();
        _targetActorNetworks = new List<NeuralNetwork<T>>();
        _criticNetworks = new List<NeuralNetwork<T>>();
        _targetCriticNetworks = new List<NeuralNetwork<T>>();

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
        }
    }

    private NeuralNetwork<T> CreateActorNetwork()
    {
        var network = new NeuralNetwork<T>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.ActorHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, _options.ActionSize));
        network.AddLayer(new ActivationLayer<T>(new Tanh<T>()));

        return network;
    }

    private NeuralNetwork<T> CreateCriticNetwork()
    {
        var network = new NeuralNetwork<T>();
        // Centralized critic: observes all agents' states and actions
        int inputSize = (_options.StateSize + _options.ActionSize) * _options.NumAgents;
        int previousSize = inputSize;

        foreach (var layerSize in _options.CriticHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, 1));

        return network;
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
                var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.FromDouble(_options.ExplorationNoise));
                action[i] = _numOps.Add(action[i], noise);
                action[i] = MathHelper.Clamp<T>(action[i], _numOps.FromDouble(-1), _numOps.FromDouble(1));
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
        T avgReward = _numOps.Zero;
        foreach (var reward in rewards)
        {
            avgReward = _numOps.Add(avgReward, reward);
        }
        avgReward = _numOps.Divide(avgReward, _numOps.FromDouble(rewards.Count));

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
            return _numOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = _numOps.Zero;

        // Update each agent's critic and actor
        for (int agentId = 0; agentId < _options.NumAgents; agentId++)
        {
            T criticLoss = UpdateCritic(agentId, batch);
            T actorLoss = UpdateActor(agentId, batch);

            totalLoss = _numOps.Add(totalLoss, _numOps.Add(criticLoss, actorLoss));

            // Soft update target networks
            SoftUpdateTargetNetwork(_actorNetworks[agentId], _targetActorNetworks[agentId]);
            SoftUpdateTargetNetwork(_criticNetworks[agentId], _targetCriticNetworks[agentId]);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(_options.NumAgents * 2));
    }

    private T UpdateCritic(int agentId, List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

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
                target = _numOps.Add(experience.reward, _numOps.Multiply(_options.DiscountFactor, targetQ));
            }

            // Current Q-value
            var currentQ = _criticNetworks[agentId].Forward(ConcatenateStateAction(experience.state, experience.action))[0];

            // TD error
            var error = _numOps.Subtract(target, currentQ);
            var loss = _numOps.Multiply(error, error);
            totalLoss = _numOps.Add(totalLoss, loss);

            // Backpropagate
            var gradient = new Vector<T>(1);
            gradient[0] = error;
            _criticNetworks[agentId].Backward(gradient);
            _criticNetworks[agentId].UpdateWeights(_options.CriticLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private T UpdateActor(int agentId, List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

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
            totalLoss = _numOps.Add(totalLoss, _numOps.Negate(qValue));

            // Simplified gradient for actor
            var actorGradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                actorGradient[i] = _numOps.Divide(qValue, _numOps.FromDouble(_options.ActionSize));
            }

            _actorNetworks[agentId].Backward(actorGradient);
            _actorNetworks[agentId].UpdateWeights(_options.ActorLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private void SoftUpdateTargetNetwork(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceLayers = source.GetLayers();
        var targetLayers = target.GetLayers();

        for (int i = 0; i < sourceLayers.Count; i++)
        {
            if (sourceLayers[i] is DenseLayer<T> sourceLayer && targetLayers[i] is DenseLayer<T> targetLayer)
            {
                var sourceWeights = sourceLayer.GetWeights();
                var sourceBiases = sourceLayer.GetBiases();
                var targetWeights = targetLayer.GetWeights();
                var targetBiases = targetLayer.GetBiases();

                var oneMinusTau = _numOps.Subtract(_numOps.One, _options.TargetUpdateTau);

                for (int r = 0; r < targetWeights.Rows; r++)
                {
                    for (int c = 0; c < targetWeights.Columns; c++)
                    {
                        var sourceContrib = _numOps.Multiply(_options.TargetUpdateTau, sourceWeights[r, c]);
                        var targetContrib = _numOps.Multiply(oneMinusTau, targetWeights[r, c]);
                        targetWeights[r, c] = _numOps.Add(sourceContrib, targetContrib);
                    }
                }

                for (int i = 0; i < targetBiases.Length; i++)
                {
                    var sourceContrib = _numOps.Multiply(_options.TargetUpdateTau, sourceBiases[i]);
                    var targetContrib = _numOps.Multiply(oneMinusTau, targetBiases[i]);
                    targetBiases[i] = _numOps.Add(sourceContrib, targetContrib);
                }

                targetLayer.SetWeights(targetWeights);
                targetLayer.SetBiases(targetBiases);
            }
        }
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
            ["steps"] = _numOps.FromDouble(_stepCount),
            ["buffer_size"] = _numOps.FromDouble(_replayBuffer.Count)
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
