using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.TD3;

/// <summary>
/// Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for continuous control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TD3 improves upon DDPG with three key innovations:
/// 1. Twin Q-Networks: Uses two Q-functions to reduce overestimation bias
/// 2. Delayed Policy Updates: Updates policy less frequently than Q-networks
/// 3. Target Policy Smoothing: Adds noise to target actions for robustness
/// </para>
/// <para><b>For Beginners:</b>
/// TD3 is one of the best algorithms for continuous control tasks (like robot movement).
/// It's more stable and robust than DDPG.
///
/// Key innovations:
/// - **Twin Critics**: Uses two Q-networks and takes the minimum to avoid overoptimism
/// - **Delayed Updates**: Waits before updating the policy to let Q-values stabilize
/// - **Target Smoothing**: Adds noise to target actions to prevent exploitation of errors
///
/// Think of it like getting a second opinion before making decisions, and taking time
/// to verify information before acting on it.
///
/// Used by: Robotic control, autonomous systems, continuous optimization
/// </para>
/// </remarks>
public class TD3Agent<T> : DeepReinforcementLearningAgentBase<T>
{
    private TD3Options<T> _options;
    private readonly INumericOperations<T> _numOps;

    private NeuralNetwork<T> _actorNetwork;
    private NeuralNetwork<T> _targetActorNetwork;
    private NeuralNetwork<T> _critic1Network;
    private NeuralNetwork<T> _critic2Network;
    private NeuralNetwork<T> _targetCritic1Network;
    private NeuralNetwork<T> _targetCritic2Network;

    private ReplayBuffer<T> _replayBuffer;
    private Random _random;
    private int _stepCount;
    private int _updateCount;

    public TD3Agent(TD3Options<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _stepCount = 0;
        _updateCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        // Actor network: state -> action
        _actorNetwork = CreateActorNetwork();
        _targetActorNetwork = CreateActorNetwork();
        CopyNetworkWeights(_actorNetwork, _targetActorNetwork);

        // Twin Critic networks: (state, action) -> Q-value
        _critic1Network = CreateCriticNetwork();
        _critic2Network = CreateCriticNetwork();
        _targetCritic1Network = CreateCriticNetwork();
        _targetCritic2Network = CreateCriticNetwork();

        CopyNetworkWeights(_critic1Network, _targetCritic1Network);
        CopyNetworkWeights(_critic2Network, _targetCritic2Network);
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
        int inputSize = _options.StateSize + _options.ActionSize;
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

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _actorNetwork.Predict(state);

        if (training)
        {
            // Add exploration noise during training
            for (int i = 0; i < action.Length; i++)
            {
                var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.FromDouble(_options.ExplorationNoise));
                action[i] = _numOps.Add(action[i], noise);
                action[i] = MathHelper.Clamp<T>(action[i], _numOps.FromDouble(-1), _numOps.FromDouble(1));
            }
        }

        return action;
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

        // Update critics
        T criticLoss = UpdateCritics(batch);

        // Delayed policy update
        if (_updateCount % _options.PolicyUpdateFrequency == 0)
        {
            UpdateActor(batch);

            // Update target networks with soft updates
            SoftUpdateTargetNetworks();
        }

        _updateCount++;

        return criticLoss;
    }

    private T UpdateCritics(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target Q-value with target policy smoothing
            var nextAction = _targetActorNetwork.Predict(experience.nextState);

            // Add clipped noise to target action (target policy smoothing)
            for (int i = 0; i < nextAction.Length; i++)
            {
                var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.FromDouble(_options.TargetPolicyNoise));
                noise = MathHelper.Clamp<T>(noise, _numOps.FromDouble(-_options.TargetNoiseClip), _numOps.FromDouble(_options.TargetNoiseClip));
                nextAction[i] = _numOps.Add(nextAction[i], noise);
                nextAction[i] = MathHelper.Clamp<T>(nextAction[i], _numOps.FromDouble(-1), _numOps.FromDouble(1));
            }

            // Concatenate next state and next action for critic input
            var nextStateAction = ConcatenateStateAction(experience.nextState, nextAction);

            // Compute twin Q-targets and take minimum (clipped double Q-learning)
            var q1Target = _targetCritic1Network.Predict(nextStateAction)[0];
            var q2Target = _targetCritic2Network.Predict(nextStateAction)[0];
            var minQTarget = MathHelper.Min<T>(q1Target, q2Target);

            // Compute TD target
            T targetQ;
            if (experience.done)
            {
                targetQ = experience.reward;
            }
            else
            {
                var discountedQ = _numOps.Multiply(_options.DiscountFactor, minQTarget);
                targetQ = _numOps.Add(experience.reward, discountedQ);
            }

            // Concatenate state and action for critic input
            var stateAction = ConcatenateStateAction(experience.state, experience.action);

            // Update Critic 1
            var q1Value = _critic1Network.Predict(stateAction)[0];
            var q1Error = _numOps.Subtract(targetQ, q1Value);
            var q1ErrorVec = new Vector<T>(1);
            q1ErrorVec[0] = q1Error;
            _critic1Network.Backward(q1ErrorVec);
            _critic1Network.UpdateWeights(_options.CriticLearningRate);

            // Update Critic 2
            var q2Value = _critic2Network.Predict(stateAction)[0];
            var q2Error = _numOps.Subtract(targetQ, q2Value);
            var q2ErrorVec = new Vector<T>(1);
            q2ErrorVec[0] = q2Error;
            _critic2Network.Backward(q2ErrorVec);
            _critic2Network.UpdateWeights(_options.CriticLearningRate);

            // Accumulate loss (MSE)
            var loss1 = _numOps.Multiply(q1Error, q1Error);
            var loss2 = _numOps.Multiply(q2Error, q2Error);
            totalLoss = _numOps.Add(totalLoss, _numOps.Add(loss1, loss2));
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count * 2));
    }

    private void UpdateActor(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        foreach (var experience in batch)
        {
            // Compute action from current policy
            var action = _actorNetwork.Predict(experience.state);

            // Concatenate state and action
            var stateAction = ConcatenateStateAction(experience.state, action);

            // Compute Q-value from critic 1 (use only one critic for policy gradient)
            var qValue = _critic1Network.Predict(stateAction)[0];

            // Policy gradient: maximize Q-value, so negate for gradient ascent
            var policyGradient = new Vector<T>(1);
            policyGradient[0] = _numOps.Negate(qValue);

            // Backpropagate through critic to get gradient w.r.t. actions
            var actionGradient = _critic1Network.Backward(policyGradient);

            // Extract action gradients (remove state part)
            var actorGradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                actorGradient[i] = actionGradient[_options.StateSize + i];
            }

            // Backpropagate through actor
            _actorNetwork.Backward(actorGradient);
            _actorNetwork.UpdateWeights(_options.ActorLearningRate);
        }
    }

    private void SoftUpdateTargetNetworks()
    {
        SoftUpdateNetwork(_actorNetwork, _targetActorNetwork);
        SoftUpdateNetwork(_critic1Network, _targetCritic1Network);
        SoftUpdateNetwork(_critic2Network, _targetCritic2Network);
    }

    private void SoftUpdateNetwork(NeuralNetwork<T> source, NeuralNetwork<T> target)
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

                // θ_target = τ * θ_source + (1 - τ) * θ_target
                var oneMinusTau = _numOps.Subtract(_numOps.One, _options.TargetUpdateTau);

                for (int r = 0; r < targetWeights.Rows; r++)
                {
                    for (int c = 0; c < targetWeights.Columns; c++)
                    {
                        var sourceContribution = _numOps.Multiply(_options.TargetUpdateTau, sourceWeights[r, c]);
                        var targetContribution = _numOps.Multiply(oneMinusTau, targetWeights[r, c]);
                        targetWeights[r, c] = _numOps.Add(sourceContribution, targetContribution);
                    }
                }

                for (int i = 0; i < targetBiases.Length; i++)
                {
                    var sourceContribution = _numOps.Multiply(_options.TargetUpdateTau, sourceBiases[i]);
                    var targetContribution = _numOps.Multiply(oneMinusTau, targetBiases[i]);
                    targetBiases[i] = _numOps.Add(sourceContribution, targetContribution);
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
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_replayBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        // TD3 doesn't need per-episode reset
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
