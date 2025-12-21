using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

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
public class CQLAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private CQLOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    private NeuralNetwork<T> _policyNetwork;
    private NeuralNetwork<T> _q1Network;
    private NeuralNetwork<T> _q2Network;
    private NeuralNetwork<T> _targetQ1Network;
    private NeuralNetwork<T> _targetQ2Network;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _offlineBuffer;  // Fixed offline dataset
    private Random _random;
    private T _logAlpha;
    private T _alpha;
    private int _updateCount;

    public CQLAgent(CQLOptions<T> options) : base(CreateBaseOptions(options))
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = options.Seed.HasValue ? RandomHelper.CreateSeededRandom(options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _updateCount = 0;

        _logAlpha = NumOps.Log(_options.InitialTemperature);
        _alpha = _options.InitialTemperature;

        // Initialize networks directly in constructor
        _policyNetwork = CreatePolicyNetwork();
        _q1Network = CreateQNetwork();
        _q2Network = CreateQNetwork();
        _targetQ1Network = CreateQNetwork();
        _targetQ2Network = CreateQNetwork();

        CopyNetworkWeights(_q1Network, _targetQ1Network);
        CopyNetworkWeights(_q2Network, _targetQ2Network);

        // Initialize offline buffer
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize, _options.Seed);
    }

    private static ReinforcementLearningOptions<T> CreateBaseOptions(CQLOptions<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.QLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.QLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.BufferSize
        };
    }

    private NeuralNetwork<T> CreatePolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(previousSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        // Output: mean and log_std for Gaussian policy
        layers.Add(new DenseLayer<T>(previousSize, _options.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize * 2,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, null);
    }

    private NeuralNetwork<T> CreateQNetwork()
    {
        var layers = new List<ILayer<T>>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int previousSize = inputSize;

        foreach (var layerSize in _options.QHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(previousSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(previousSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, _options.QLossFunction);
    }

    private void InitializeBuffer()
    {
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize);
    }

    /// <summary>
    /// Load offline dataset into the replay buffer.
    /// </summary>
    public void LoadOfflineData(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> dataset)
    {
        foreach (var transition in dataset)
        {
            var experience = new Experience<T, Vector<T>, Vector<T>>(
                transition.state,
                transition.action,
                transition.reward,
                transition.nextState,
                transition.done);
            _offlineBuffer.Add(experience);
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

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
            var std = NumOps.Exp(logStd[i]);
            var noise = GetSeededNormalRandom(_numOps.Zero, _numOps.One, _random);
            var rawAction = _numOps.Add(mean[i], _numOps.Multiply(std, noise));
            action[i] = MathHelper.Tanh<T>(rawAction);
        }

        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // CQL is offline - data is loaded beforehand
        // This method is kept for interface compliance but not used in offline setting
        var experience = new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done);
        _offlineBuffer.Add(experience);
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

    private T UpdateQNetworks(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target Q-value
            var nextAction = SelectAction(experience.NextState, training: true);
            var nextStateAction = ConcatenateStateAction(experience.NextState, nextAction);
            var nextStateActionTensor = Tensor<T>.FromVector(nextStateAction);

            var q1TargetTensor = _targetQ1Network.Predict(nextStateActionTensor);
            var q2TargetTensor = _targetQ2Network.Predict(nextStateActionTensor);
            var q1TargetValue = q1TargetTensor.ToVector()[0];
            var q2TargetValue = q2TargetTensor.ToVector()[0];
            var minQTarget = MathHelper.Min<T>(q1TargetValue, q2TargetValue);

            // Compute actual policy entropy from log probabilities
            // For Gaussian policy: entropy = 0.5 * log(2 * pi * e * sigma^2)
            var policyOutputTensor = _policyNetwork.Predict(Tensor<T>.FromVector(experience.NextState));
            var policyOutput = policyOutputTensor.ToVector();
            T policyEntropy = _numOps.Zero;
            for (int entropyIdx = 0; entropyIdx < _options.ActionSize; entropyIdx++)
            {
                var logStd = policyOutput[_options.ActionSize + entropyIdx];
                logStd = MathHelper.Clamp<T>(logStd, _numOps.FromDouble(-20), _numOps.FromDouble(2));
                // Gaussian entropy: 0.5 * (1 + log(2*pi)) + log(sigma)
                var gaussianConst = _numOps.FromDouble(0.5 * (1.0 + System.Math.Log(2.0 * System.Math.PI)));
                policyEntropy = _numOps.Add(policyEntropy, _numOps.Add(gaussianConst, logStd));
            }
            var entropyTerm = _numOps.Multiply(_alpha, policyEntropy);

            T targetQ;
            if (experience.Done)
            {
                targetQ = experience.Reward;
            }
            else
            {
                var futureValue = _numOps.Subtract(minQTarget, entropyTerm);
                targetQ = _numOps.Add(experience.Reward, _numOps.Multiply(_options.DiscountFactor, futureValue));
            }

            // Compute current Q-values
            var stateAction = ConcatenateStateAction(experience.State, experience.Action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var q1Tensor = _q1Network.Predict(stateActionTensor);
            var q2Tensor = _q2Network.Predict(stateActionTensor);
            var q1Value = q1Tensor.ToVector()[0];
            var q2Value = q2Tensor.ToVector()[0];

            // CQL Conservative penalty: penalize Q-values for random/OOD actions
            var cqlPenalty = ComputeCQLPenalty(experience.State, experience.Action, q1Value, q2Value);

            // Q-learning loss + CQL penalty
            var q1Error = _numOps.Subtract(targetQ, q1Value);
            var q1Loss = _numOps.Multiply(q1Error, q1Error);
            q1Loss = _numOps.Add(q1Loss, cqlPenalty);

            var q2Error = _numOps.Subtract(targetQ, q2Value);
            var q2Loss = _numOps.Multiply(q2Error, q2Error);
            q2Loss = _numOps.Add(q2Loss, cqlPenalty);

            // Backpropagate Q1: MSE gradient + CQL penalty gradient
            // MSE: -2 * (target - pred), CQL penalty: -alpha/2 (derivative of -Q(s,a_data))
            var q1MseGrad = _numOps.Multiply(_numOps.FromDouble(-2.0), q1Error);
            var q1CqlGrad = _numOps.Multiply(_numOps.FromDouble(-0.5), _options.CQLAlpha);
            var q1TotalGrad = _numOps.Add(q1MseGrad, q1CqlGrad);
            var q1ErrorTensor = Tensor<T>.FromVector(new Vector<T>(new[] { q1TotalGrad }));
            _q1Network.Backpropagate(q1ErrorTensor);

            // Apply gradients manually
            var q1Params = _q1Network.GetParameters();
            for (int i = 0; i < q1Params.Length; i++)
            {
                q1Params[i] = _numOps.Add(q1Params[i], _numOps.Multiply(_options.QLearningRate, q1TotalGrad));
            }
            _q1Network.UpdateParameters(q1Params);

            // Backpropagate Q2: MSE gradient + CQL penalty gradient
            var q2MseGrad = _numOps.Multiply(_numOps.FromDouble(-2.0), q2Error);
            var q2CqlGrad = _numOps.Multiply(_numOps.FromDouble(-0.5), _options.CQLAlpha);
            var q2TotalGrad = _numOps.Add(q2MseGrad, q2CqlGrad);
            var q2ErrorTensor = Tensor<T>.FromVector(new Vector<T>(new[] { q2TotalGrad }));
            _q2Network.Backpropagate(q2ErrorTensor);

            // Apply gradients manually
            var q2Params = _q2Network.GetParameters();
            for (int i = 0; i < q2Params.Length; i++)
            {
                q2Params[i] = _numOps.Add(q2Params[i], _numOps.Multiply(_options.QLearningRate, q2TotalGrad));
            }
            _q2Network.UpdateParameters(q2Params);

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
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var q1RandomTensor = _q1Network.Predict(stateActionTensor);
            var q2RandomTensor = _q2Network.Predict(stateActionTensor);
            var q1Random = q1RandomTensor.ToVector()[0];
            var q2Random = q2RandomTensor.ToVector()[0];

            var avgQRandom = _numOps.Divide(_numOps.Add(q1Random, q2Random), _numOps.FromDouble(2));
            randomQSum = _numOps.Add(randomQSum, avgQRandom);
        }

        var avgRandomQ = _numOps.Divide(randomQSum, _numOps.FromDouble(numSamples));
        var avgDataQ = _numOps.Divide(_numOps.Add(q1Value, q2Value), _numOps.FromDouble(2));

        // Penalty = alpha * (E[Q(s, a_random)] - Q(s, a_data))
        var gap = _numOps.Subtract(avgRandomQ, avgDataQ);
        return _numOps.Multiply(_options.CQLAlpha, gap);
    }

    private T UpdatePolicy(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            var action = SelectAction(experience.State, training: true);
            var stateAction = ConcatenateStateAction(experience.State, action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);

            var q1Tensor = _q1Network.Predict(stateActionTensor);
            var q2Tensor = _q2Network.Predict(stateActionTensor);
            var q1Value = q1Tensor.ToVector()[0];
            var q2Value = q2Tensor.ToVector()[0];
            var minQ = MathHelper.Min<T>(q1Value, q2Value);

            // Policy loss: -Q(s,a) + alpha * entropy (simplified)
            var policyLoss = _numOps.Negate(minQ);

            totalLoss = _numOps.Add(totalLoss, policyLoss);

            // Backprop through Q-network to get action gradient
            var qGradTensor = Tensor<T>.FromVector(new Vector<T>(new[] { _numOps.One }));
            var actionGradTensor = _q1Network.Backpropagate(qGradTensor);
            var actionGrad = actionGradTensor.ToVector();

            // Compute policy gradients for both mean and log-sigma
            // CRITICAL FIX: We want to MAXIMIZE Q, so we need to negate actionGrad
            // Policy loss is -Q(s,a), gradient is d(-Q)/dθ = -dQ/dθ
            var policyStateTensor = Tensor<T>.FromVector(experience.State);
            var policyOutTensor = _policyNetwork.Predict(policyStateTensor);
            var policyOut = policyOutTensor.ToVector();

            var policyGrad = new Vector<T>(_options.ActionSize * 2);
            for (int policyGradIdx = 0; policyGradIdx < _options.ActionSize; policyGradIdx++)
            {
                // Mean (mu) gradient: Negate actionGrad because policy loss is -Q(s,a)
                // actionGrad contains dQ/da, but we want d(-Q)/da = -dQ/da
                policyGrad[policyGradIdx] = _numOps.Negate(actionGrad[_options.StateSize + policyGradIdx]);

                // Log-sigma gradient: Combine action gradient and entropy regularization
                // The variance affects both Q-value and entropy
                var varianceActionGrad = actionGrad.Length > _options.StateSize + _options.ActionSize + policyGradIdx
                    ? actionGrad[_options.StateSize + _options.ActionSize + policyGradIdx]
                    : _numOps.Zero;
                var entropyGrad = _alpha; // Gradient of entropy w.r.t. log_sigma
                policyGrad[_options.ActionSize + policyGradIdx] = _numOps.Add(_numOps.Negate(varianceActionGrad), entropyGrad);
            }

            var policyGradTensor = Tensor<T>.FromVector(policyGrad);
            _policyNetwork.Backpropagate(policyGradTensor);

            // Apply gradients manually
            var policyParams = _policyNetwork.GetParameters();
            for (int i = 0; i < policyParams.Length; i++)
            {
                policyParams[i] = _numOps.Add(policyParams[i], _numOps.Multiply(_options.PolicyLearningRate, policyGrad[i % policyGrad.Length]));
            }
            _policyNetwork.UpdateParameters(policyParams);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private void UpdateTemperature(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        // Temperature update using entropy target
        // Loss: alpha * (entropy - target_entropy)
        // Gradient: d_loss/d_log_alpha = alpha * (entropy - target_entropy)

        T avgEntropy = _numOps.Zero;
        foreach (var experience in batch)
        {
            var policyOutputTensor = _policyNetwork.Predict(Tensor<T>.FromVector(experience.State));
            var policyOutput = policyOutputTensor.ToVector();

            T entropy = _numOps.Zero;
            for (int tempIdx = 0; tempIdx < _options.ActionSize; tempIdx++)
            {
                var logStd = policyOutput[_options.ActionSize + tempIdx];
                logStd = MathHelper.Clamp<T>(logStd, _numOps.FromDouble(-20), _numOps.FromDouble(2));
                var gaussianConst = _numOps.FromDouble(0.5 * (1.0 + System.Math.Log(2.0 * System.Math.PI)));
                entropy = _numOps.Add(entropy, _numOps.Add(gaussianConst, logStd));
            }
            avgEntropy = _numOps.Add(avgEntropy, entropy);
        }
        avgEntropy = _numOps.Divide(avgEntropy, _numOps.FromDouble(batch.Count));

        // Target entropy: -dim(action_space)
        var targetEntropy = _numOps.FromDouble(-_options.ActionSize);
        var entropyGap = _numOps.Subtract(avgEntropy, targetEntropy);

        // Update log_alpha: log_alpha -= lr * alpha * entropy_gap
        var alphaLr = _numOps.FromDouble(0.0003);
        var alphaGrad = _numOps.Multiply(_alpha, entropyGap);
        var alphaUpdate = _numOps.Multiply(alphaLr, alphaGrad);
        _logAlpha = _numOps.Subtract(_logAlpha, alphaUpdate);

        // Update alpha from log_alpha
        _alpha = NumOps.Exp(_logAlpha);
    }

    private void SoftUpdateTargetNetworks()
    {
        SoftUpdateNetwork(_q1Network, _targetQ1Network);
        SoftUpdateNetwork(_q2Network, _targetQ2Network);
    }

    private void SoftUpdateNetwork(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();
        var oneMinusTau = _numOps.Subtract(_numOps.One, _options.TargetUpdateTau);

        var updatedParams = new Vector<T>(targetParams.Length);
        for (int softUpdateIdx = 0; softUpdateIdx < targetParams.Length; softUpdateIdx++)
        {
            var sourceContrib = _numOps.Multiply(_options.TargetUpdateTau, sourceParams[softUpdateIdx]);
            var targetContrib = _numOps.Multiply(oneMinusTau, targetParams[softUpdateIdx]);
            updatedParams[softUpdateIdx] = _numOps.Add(sourceContrib, targetContrib);
        }

        target.UpdateParameters(updatedParams);
    }

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
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

    private T GetSeededNormalRandom(T mean, T stdDev, Random random)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        double result = randStdNormal * Convert.ToDouble(stdDev) + Convert.ToDouble(mean);
        return _numOps.FromDouble(result);
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

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.CQLAgent,
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Combine parameters from policy network and both Q-networks
        var policyParams = _policyNetwork.GetParameters();
        var q1Params = _q1Network.GetParameters();
        var q2Params = _q2Network.GetParameters();

        var total = policyParams.Length + q1Params.Length + q2Params.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in policyParams) vector[idx++] = p;
        foreach (var p in q1Params) vector[idx++] = p;
        foreach (var p in q2Params) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var policyParams = _policyNetwork.GetParameters();
        var q1Params = _q1Network.GetParameters();
        var q2Params = _q2Network.GetParameters();

        int idx = 0;
        var policyVec = new Vector<T>(policyParams.Length);
        var q1Vec = new Vector<T>(q1Params.Length);
        var q2Vec = new Vector<T>(q2Params.Length);

        for (int i = 0; i < policyParams.Length; i++) policyVec[i] = parameters[idx++];
        for (int i = 0; i < q1Params.Length; i++) q1Vec[i] = parameters[idx++];
        for (int i = 0; i < q2Params.Length; i++) q2Vec[i] = parameters[idx++];

        _policyNetwork.UpdateParameters(policyVec);
        _q1Network.UpdateParameters(q1Vec);
        _q2Network.UpdateParameters(q2Vec);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new CQLAgent<T>(_options);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // CQL uses custom gradient computation - return zero gradients as placeholder
        var parameters = GetParameters();
        var gradients = new Vector<T>(parameters.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = _numOps.Zero;
        }
        return gradients;
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // CQL uses direct network updates - not directly applicable
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_updateCount);
        writer.Write(Convert.ToDouble(_alpha));

        var policyBytes = _policyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        var q1Bytes = _q1Network.Serialize();
        writer.Write(q1Bytes.Length);
        writer.Write(q1Bytes);

        var q2Bytes = _q2Network.Serialize();
        writer.Write(q2Bytes.Length);
        writer.Write(q2Bytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        _updateCount = reader.ReadInt32();
        _alpha = _numOps.FromDouble(reader.ReadDouble());

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _policyNetwork.Deserialize(policyBytes);

        var q1Length = reader.ReadInt32();
        var q1Bytes = reader.ReadBytes(q1Length);
        _q1Network.Deserialize(q1Bytes);

        var q2Length = reader.ReadInt32();
        var q2Bytes = reader.ReadBytes(q2Length);
        _q2Network.Deserialize(q2Bytes);
    }

    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
