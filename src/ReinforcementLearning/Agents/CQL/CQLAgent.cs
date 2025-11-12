using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.CQL;

/// <summary>
/// Conservative Q-Learning (CQL) agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CQL is designed for offline RL, learning from fixed datasets without environment interaction.
/// It prevents overestimation by adding a conservative penalty that pushes down Q-values
/// for out-of-distribution actions while maintaining accuracy on in-distribution actions.
/// </para>
/// <para><b>For Beginners:</b>
/// Unlike online RL (which tries actions and learns), CQL learns only from recorded data.
/// This is crucial for domains where exploration is dangerous or expensive.
///
/// Key features:
/// - **Conservative Penalty**: Lowers Q-values for unseen state-action pairs
/// - **Offline Learning**: No environment interaction needed
/// - **Safe Policy Improvement**: Guarantees improvement over behavior policy
///
/// Example use cases:
/// - Learning from medical records (can't experiment on patients)
/// - Autonomous driving from dashcam data
/// - Robotics from demonstration datasets
/// </para>
/// </remarks>
public class CQLAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly CQLOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    private NeuralNetwork<T> _policyNetwork;
    private NeuralNetwork<T> _q1Network;
    private NeuralNetwork<T> _q2Network;
    private NeuralNetwork<T> _targetQ1Network;
    private NeuralNetwork<T> _targetQ2Network;

    private ReplayBuffer<T> _offlineBuffer;  // Fixed offline dataset
    private Random _random;
    private T _logAlpha;
    private T _alpha;
    private int _updateCount;

    public CQLAgent(CQLOptions<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _updateCount = 0;

        _logAlpha = MathHelper.Log(_options.InitialTemperature);
        _alpha = _options.InitialTemperature;

        InitializeNetworks();
        InitializeBuffer();
    }

    private void InitializeNetworks()
    {
        _policyNetwork = CreatePolicyNetwork();
        _q1Network = CreateQNetwork();
        _q2Network = CreateQNetwork();
        _targetQ1Network = CreateQNetwork();
        _targetQ2Network = CreateQNetwork();

        CopyNetworkWeights(_q1Network, _targetQ1Network);
        CopyNetworkWeights(_q2Network, _targetQ2Network);
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

            // Clamp log_std for numerical stability
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

        // Sample action from Gaussian policy
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
        // CQL is offline - data is loaded beforehand
        // This method is kept for interface compliance but not used in offline setting
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

        // Update Q-networks with CQL penalty
        T qLoss = UpdateQNetworks(batch);
        totalLoss = _numOps.Add(totalLoss, qLoss);

        // Update policy
        T policyLoss = UpdatePolicy(batch);
        totalLoss = _numOps.Add(totalLoss, policyLoss);

        // Update temperature
        if (_options.AutoTuneTemperature)
        {
            UpdateTemperature(batch);
        }

        // Soft update target networks
        SoftUpdateTargetNetworks();

        _updateCount++;

        return _numOps.Divide(totalLoss, _numOps.FromDouble(2));
    }

    private T UpdateQNetworks(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target Q-value
            var nextAction = SelectAction(experience.nextState, training: true);
            var nextStateAction = ConcatenateStateAction(experience.nextState, nextAction);

            var q1TargetValue = _targetQ1Network.Forward(nextStateAction)[0];
            var q2TargetValue = _targetQ2Network.Forward(nextStateAction)[0];
            var minQTarget = MathHelper.Min<T>(q1TargetValue, q2TargetValue);

            // Compute entropy term (simplified)
            var entropyTerm = _numOps.Multiply(_alpha, _numOps.FromDouble(0.1));  // Simplified entropy

            T targetQ;
            if (experience.done)
            {
                targetQ = experience.reward;
            }
            else
            {
                var futureValue = _numOps.Subtract(minQTarget, entropyTerm);
                targetQ = _numOps.Add(experience.reward, _numOps.Multiply(_options.DiscountFactor, futureValue));
            }

            // Compute current Q-values
            var stateAction = ConcatenateStateAction(experience.state, experience.action);
            var q1Value = _q1Network.Forward(stateAction)[0];
            var q2Value = _q2Network.Forward(stateAction)[0];

            // CQL Conservative penalty: penalize Q-values for random/OOD actions
            var cqlPenalty = ComputeCQLPenalty(experience.state, experience.action, q1Value, q2Value);

            // Q-learning loss + CQL penalty
            var q1Error = _numOps.Subtract(targetQ, q1Value);
            var q1Loss = _numOps.Multiply(q1Error, q1Error);
            q1Loss = _numOps.Add(q1Loss, cqlPenalty);

            var q2Error = _numOps.Subtract(targetQ, q2Value);
            var q2Loss = _numOps.Multiply(q2Error, q2Error);
            q2Loss = _numOps.Add(q2Loss, cqlPenalty);

            // Backpropagate Q1
            var q1ErrorVec = new Vector<T>(1);
            q1ErrorVec[0] = q1Error;
            _q1Network.Backward(q1ErrorVec);
            _q1Network.UpdateWeights(_options.QLearningRate);

            // Backpropagate Q2
            var q2ErrorVec = new Vector<T>(1);
            q2ErrorVec[0] = q2Error;
            _q2Network.Backward(q2ErrorVec);
            _q2Network.UpdateWeights(_options.QLearningRate);

            totalLoss = _numOps.Add(totalLoss, _numOps.Add(q1Loss, q2Loss));
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count * 2));
    }

    private T ComputeCQLPenalty(Vector<T> state, Vector<T> dataAction, T q1Value, T q2Value)
    {
        // CQL penalty: E[Q(s, a_random)] - Q(s, a_data)
        // This pushes down Q-values for random actions while keeping data actions accurate

        T randomQSum = _numOps.Zero;
        int numSamples = _options.CQLNumActions;

        for (int i = 0; i < numSamples; i++)
        {
            // Sample random action
            var randomAction = new Vector<T>(_options.ActionSize);
            for (int j = 0; j < _options.ActionSize; j++)
            {
                randomAction[j] = _numOps.FromDouble(_random.NextDouble() * 2 - 1);  // [-1, 1]
            }

            var stateAction = ConcatenateStateAction(state, randomAction);
            var q1Random = _q1Network.Forward(stateAction)[0];
            var q2Random = _q2Network.Forward(stateAction)[0];

            var avgQRandom = _numOps.Divide(_numOps.Add(q1Random, q2Random), _numOps.FromDouble(2));
            randomQSum = _numOps.Add(randomQSum, avgQRandom);
        }

        var avgRandomQ = _numOps.Divide(randomQSum, _numOps.FromDouble(numSamples));
        var avgDataQ = _numOps.Divide(_numOps.Add(q1Value, q2Value), _numOps.FromDouble(2));

        // Penalty = alpha * (E[Q(s, a_random)] - Q(s, a_data))
        var gap = _numOps.Subtract(avgRandomQ, avgDataQ);
        return _numOps.Multiply(_options.CQLAlpha, gap);
    }

    private T UpdatePolicy(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            var action = SelectAction(experience.state, training: true);
            var stateAction = ConcatenateStateAction(experience.state, action);

            var q1Value = _q1Network.Forward(stateAction)[0];
            var q2Value = _q2Network.Forward(stateAction)[0];
            var minQ = MathHelper.Min<T>(q1Value, q2Value);

            // Policy loss: -Q(s,a) + alpha * entropy (simplified)
            var policyLoss = _numOps.Negate(minQ);

            totalLoss = _numOps.Add(totalLoss, _numOps.Multiply(policyLoss, policyLoss));

            // Backprop through Q-network to get action gradient
            var qGrad = new Vector<T>(1);
            qGrad[0] = _numOps.One;
            var actionGrad = _q1Network.Backward(qGrad);

            // Extract action part of gradient
            var policyGrad = new Vector<T>(_options.ActionSize * 2);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                policyGrad[i] = actionGrad[_options.StateSize + i];
            }

            _policyNetwork.Backward(policyGrad);
            _policyNetwork.UpdateWeights(_options.PolicyLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private void UpdateTemperature(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch)
    {
        // Simplified temperature update
        _alpha = MathHelper.Exp(_logAlpha);
    }

    private void SoftUpdateTargetNetworks()
    {
        SoftUpdateNetwork(_q1Network, _targetQ1Network);
        SoftUpdateNetwork(_q2Network, _targetQ2Network);
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
            ["buffer_size"] = _numOps.FromDouble(_offlineBuffer.Count),
            ["alpha"] = _alpha
        };
    }

    public override void ResetEpisode()
    {
        // CQL is offline - no episode reset needed
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
