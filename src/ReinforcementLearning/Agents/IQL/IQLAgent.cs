using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.IQL;

/// <summary>
/// Implicit Q-Learning (IQL) agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IQL uses expectile regression to learn a value function that focuses on
/// high-return trajectories, enabling effective offline policy learning without
/// explicit conservative penalties like CQL.
/// </para>
/// <para><b>For Beginners:</b>
/// IQL is an offline RL algorithm that learns from fixed datasets.
/// It uses a clever statistical technique (expectile regression) to avoid
/// overestimating values of unseen actions.
///
/// Key features:
/// - **Expectile Regression**: Asymmetric loss that focuses on upper quantiles
/// - **Three Networks**: V(s), Q(s,a), and π(a|s)
/// - **Simpler than CQL**: No conservative penalties or Lagrangian multipliers
/// - **Advantage-Weighted Regression**: Extracts policy from Q and V functions
///
/// Think of expectiles like percentiles - focusing on "typically good" outcomes
/// rather than "best possible" outcomes helps avoid overoptimism.
///
/// Advantages:
/// - Simpler hyperparameter tuning than CQL
/// - Often more stable
/// - Good for offline datasets with diverse quality
/// </para>
/// </remarks>
public class IQLAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private IQLOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    private NeuralNetwork<T> _policyNetwork;
    private NeuralNetwork<T> _valueNetwork;
    private NeuralNetwork<T> _q1Network;
    private NeuralNetwork<T> _q2Network;
    private NeuralNetwork<T> _targetValueNetwork;

    private ReplayBuffer<T> _offlineBuffer;
    private Random _random;
    private int _updateCount;

    public IQLAgent(IQLOptions<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _updateCount = 0;

        InitializeNetworks();
        InitializeBuffer();
    }

    private void InitializeNetworks()
    {
        _policyNetwork = CreatePolicyNetwork();
        _valueNetwork = CreateValueNetwork();
        _q1Network = CreateQNetwork();
        _q2Network = CreateQNetwork();
        _targetValueNetwork = CreateValueNetwork();

        CopyNetworkWeights(_valueNetwork, _targetValueNetwork);
    }

    private NeuralNetwork<T> CreatePolicyNetwork()
    {
        var network = new NeuralNetwork<T>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.PolicyHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        // Output: mean and log_std for Gaussian policy
        network.AddLayer(new DenseLayer<T>(previousSize, _options.ActionSize * 2));

        return network;
    }

    private NeuralNetwork<T> CreateValueNetwork()
    {
        var network = new NeuralNetwork<T>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.ValueHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, 1));

        return network;
    }

    private NeuralNetwork<T> CreateQNetwork()
    {
        var network = new NeuralNetwork<T>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int previousSize = inputSize;

        foreach (var layerSize in _options.QHiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, 1));

        return network;
    }

    private void InitializeBuffer()
    {
        _offlineBuffer = new ReplayBuffer<T>(_options.BufferSize);
    }

    /// <summary>
    /// Load offline dataset into the replay buffer.
    /// </summary>
    public void LoadOfflineData(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> dataset)
    {
        foreach (var transition in dataset)
        {
            _offlineBuffer.Add(transition.state, transition.action, transition.reward, transition.nextState, transition.done);
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var policyOutput = _policyNetwork.Forward(state);

        // Extract mean and log_std
        var mean = new Vector<T>(_options.ActionSize);
        var logStd = new Vector<T>(_options.ActionSize);

        for (int i = 0; i < _options.ActionSize; i++)
        {
            mean[i] = policyOutput[i];
            logStd[i] = policyOutput[_options.ActionSize + i];
            logStd[i] = MathHelper.Clamp<T>(logStd[i], _numOps.FromDouble(-20), _numOps.FromDouble(2));
        }

        if (!training)
        {
            // Return mean action during evaluation
            for (int i = 0; i < mean.Length; i++)
            {
                mean[i] = MathHelper.Tanh<T>(mean[i]);
            }
            return mean;
        }

        // Sample from Gaussian policy
        var action = new Vector<T>(_options.ActionSize);
        for (int i = 0; i < _options.ActionSize; i++)
        {
            var std = MathHelper.Exp(logStd[i]);
            var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.One);
            var rawAction = _numOps.Add(mean[i], _numOps.Multiply(std, noise));
            action[i] = MathHelper.Tanh<T>(rawAction);
        }

        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // IQL is offline - data is loaded beforehand
        _offlineBuffer.Add(state, action, reward, nextState, done);
    }

    public override T Train()
    {
        if (_offlineBuffer.Count < _options.BatchSize)
        {
            return _numOps.Zero;
        }

        var batch = _offlineBuffer.Sample(_options.BatchSize);

        T totalLoss = _numOps.Zero;

        // 1. Update value function with expectile regression
        T valueLoss = UpdateValueFunction(batch);
        totalLoss = _numOps.Add(totalLoss, valueLoss);

        // 2. Update Q-functions
        T qLoss = UpdateQFunctions(batch);
        totalLoss = _numOps.Add(totalLoss, qLoss);

        // 3. Update policy with advantage-weighted regression
        T policyLoss = UpdatePolicy(batch);
        totalLoss = _numOps.Add(totalLoss, policyLoss);

        // 4. Soft update target value network
        SoftUpdateTargetNetwork();

        _updateCount++;

        return _numOps.Divide(totalLoss, _numOps.FromDouble(3));
    }

    private T UpdateValueFunction(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute Q-values for current state-action
            var stateAction = ConcatenateStateAction(experience.state, experience.action);
            var q1Value = _q1Network.Forward(stateAction)[0];
            var q2Value = _q2Network.Forward(stateAction)[0];
            var qValue = MathHelper.Min<T>(q1Value, q2Value);

            // Compute current value estimate
            var vValue = _valueNetwork.Forward(experience.state)[0];

            // Expectile regression loss
            var diff = _numOps.Subtract(qValue, vValue);
            var loss = ComputeExpectileLoss(diff, _options.Expectile);

            totalLoss = _numOps.Add(totalLoss, loss);

            // Backpropagate
            var gradient = new Vector<T>(1);
            gradient[0] = diff;
            _valueNetwork.Backward(gradient);
            _valueNetwork.UpdateWeights(_options.ValueLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private T ComputeExpectileLoss(T diff, double expectile)
    {
        // Expectile loss: |tau - I(diff < 0)| * diff^2
        var diffSquared = _numOps.Multiply(diff, diff);
        var isNegative = _numOps.Compare(diff, _numOps.Zero) < 0;

        T weight;
        if (isNegative)
        {
            weight = _numOps.FromDouble(1.0 - expectile);
        }
        else
        {
            weight = _numOps.FromDouble(expectile);
        }

        return _numOps.Multiply(weight, diffSquared);
    }

    private T UpdateQFunctions(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target: r + gamma * V(s')
            T targetQ;
            if (experience.done)
            {
                targetQ = experience.reward;
            }
            else
            {
                var nextValue = _targetValueNetwork.Forward(experience.nextState)[0];
                targetQ = _numOps.Add(experience.reward, _numOps.Multiply(_options.DiscountFactor, nextValue));
            }

            var stateAction = ConcatenateStateAction(experience.state, experience.action);

            // Update Q1
            var q1Value = _q1Network.Forward(stateAction)[0];
            var q1Error = _numOps.Subtract(targetQ, q1Value);
            var q1Loss = _numOps.Multiply(q1Error, q1Error);

            var q1ErrorVec = new Vector<T>(1);
            q1ErrorVec[0] = q1Error;
            _q1Network.Backward(q1ErrorVec);
            _q1Network.UpdateWeights(_options.QLearningRate);

            // Update Q2
            var q2Value = _q2Network.Forward(stateAction)[0];
            var q2Error = _numOps.Subtract(targetQ, q2Value);
            var q2Loss = _numOps.Multiply(q2Error, q2Error);

            var q2ErrorVec = new Vector<T>(1);
            q2ErrorVec[0] = q2Error;
            _q2Network.Backward(q2ErrorVec);
            _q2Network.UpdateWeights(_options.QLearningRate);

            totalLoss = _numOps.Add(totalLoss, _numOps.Add(q1Loss, q2Loss));
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count * 2));
    }

    private T UpdatePolicy(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute advantage: A(s,a) = Q(s,a) - V(s)
            var stateAction = ConcatenateStateAction(experience.state, experience.action);
            var q1Value = _q1Network.Forward(stateAction)[0];
            var q2Value = _q2Network.Forward(stateAction)[0];
            var qValue = MathHelper.Min<T>(q1Value, q2Value);

            var vValue = _valueNetwork.Forward(experience.state)[0];
            var advantage = _numOps.Subtract(qValue, vValue);

            // Advantage-weighted regression: exp(advantage / temperature) * log_prob(a|s)
            var weight = MathHelper.Exp(_numOps.Divide(advantage, _options.Temperature));
            weight = MathHelper.Clamp<T>(weight, _numOps.FromDouble(0.0), _numOps.FromDouble(100.0));

            // Simplified policy loss (weighted MSE to match action)
            var predictedAction = SelectAction(experience.state, training: false);
            T actionDiff = _numOps.Zero;
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var diff = _numOps.Subtract(experience.action[i], predictedAction[i]);
                actionDiff = _numOps.Add(actionDiff, _numOps.Multiply(diff, diff));
            }

            var policyLoss = _numOps.Multiply(weight, actionDiff);
            totalLoss = _numOps.Add(totalLoss, policyLoss);

            // Backpropagate
            var gradient = new Vector<T>(_options.ActionSize * 2);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var diff = _numOps.Subtract(predictedAction[i], experience.action[i]);
                gradient[i] = _numOps.Multiply(weight, diff);
            }

            _policyNetwork.Backward(gradient);
            _policyNetwork.UpdateWeights(_options.PolicyLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private void SoftUpdateTargetNetwork()
    {
        var sourceLayers = _valueNetwork.GetLayers();
        var targetLayers = _targetValueNetwork.GetLayers();

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
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_offlineBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        // IQL is offline - no episode reset needed
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
