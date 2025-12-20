using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

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

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _offlineBuffer;
    private Random _random;
    private int _updateCount;

    public IQLAgent(IQLOptions<T> options) : base(new ReinforcementLearningOptions<T>
    {
        LearningRate = options.PolicyLearningRate,
        DiscountFactor = options.DiscountFactor,
        LossFunction = new MeanSquaredErrorLoss<T>(),
        Seed = options.Seed,
        BatchSize = options.BatchSize
    })
    {
        _options = options;
        _options.Validate();
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = options.Seed.HasValue ? RandomHelper.CreateSeededRandom(options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _updateCount = 0;

        // Initialize networks directly in constructor
        _policyNetwork = CreatePolicyNetwork();
        _valueNetwork = CreateValueNetwork();
        _q1Network = CreateQNetwork();
        _q2Network = CreateQNetwork();
        _targetValueNetwork = CreateValueNetwork();

        CopyNetworkWeights(_valueNetwork, _targetValueNetwork);

        // Initialize offline buffer
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize, _options.Seed);
    }

    private NeuralNetwork<T> CreatePolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var layerSize in _options.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = layerSize;
        }

        // Output: mean and log_std for Gaussian policy
        layers.Add(new DenseLayer<T>(prevSize, _options.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize * 2,
            layers: layers);

        return new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());
    }

    private NeuralNetwork<T> CreateValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var layerSize in _options.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture);
    }

    private NeuralNetwork<T> CreateQNetwork()
    {
        var layers = new List<ILayer<T>>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int prevSize = inputSize;

        foreach (var layerSize in _options.QHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture);
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
            _offlineBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(transition.state, transition.action, transition.reward, transition.nextState, transition.done));
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
            var std = NumOps.Exp(logStd[i]);
            var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.One, _random);
            var rawAction = _numOps.Add(mean[i], _numOps.Multiply(std, noise));
            action[i] = MathHelper.Tanh<T>(rawAction);
        }

        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // IQL is offline - data is loaded beforehand
        _offlineBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
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

    private T UpdateValueFunction(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute Q-values for current state-action
            var stateAction = ConcatenateStateAction(experience.State, experience.Action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var q1OutputTensor = _q1Network.Predict(stateActionTensor);
            var q1Value = q1OutputTensor.ToVector()[0];
            var q2OutputTensor = _q2Network.Predict(stateActionTensor);
            var q2Value = q2OutputTensor.ToVector()[0];
            var qValue = MathHelper.Min<T>(q1Value, q2Value);

            // Compute current value estimate
            var stateTensor = Tensor<T>.FromVector(experience.State);
            var vOutputTensor = _valueNetwork.Predict(stateTensor);
            var vValue = vOutputTensor.ToVector()[0];

            // Expectile regression loss
            var diff = _numOps.Subtract(qValue, vValue);
            var loss = ComputeExpectileLoss(diff, _options.Expectile);

            totalLoss = _numOps.Add(totalLoss, loss);

            // Backpropagate: derivative of expectile loss w.r.t. v is -2 * weight * (q - v)
            var isNegative = _numOps.ToDouble(diff) < 0.0;
            var weight = isNegative ? _numOps.FromDouble(1.0 - _options.Expectile) : _numOps.FromDouble(_options.Expectile);
            var gradValue = _numOps.Multiply(_numOps.FromDouble(-2.0), _numOps.Multiply(weight, diff));

            var gradientVec = new Vector<T>(1);
            gradientVec[0] = gradValue;
            var gradientTensor = Tensor<T>.FromVector(gradientVec);
            _valueNetwork.Backpropagate(gradientTensor);

            var gradients = _valueNetwork.GetParameterGradients();
            _valueNetwork.ApplyGradients(gradients, _options.ValueLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private T ComputeExpectileLoss(T diff, double expectile)
    {
        // Expectile loss: |tau - I(diff < 0)| * diff^2
        var diffSquared = _numOps.Multiply(diff, diff);
        var isNegative = _numOps.ToDouble(diff) < 0.0;

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

    private T UpdateQFunctions(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target: r + gamma * V(s')
            T targetQ;
            if (experience.Done)
            {
                targetQ = experience.Reward;
            }
            else
            {
                var nextStateTensor = Tensor<T>.FromVector(experience.NextState);
                var nextValueTensor = _targetValueNetwork.Predict(nextStateTensor);
                var nextValue = nextValueTensor.ToVector()[0];
                targetQ = _numOps.Add(experience.Reward, _numOps.Multiply(_options.DiscountFactor, nextValue));
            }

            var stateAction = ConcatenateStateAction(experience.State, experience.Action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);

            // Update Q1
            var q1OutputTensor = _q1Network.Predict(stateActionTensor);
            var q1Value = q1OutputTensor.ToVector()[0];
            var q1Error = _numOps.Subtract(targetQ, q1Value);
            var q1Loss = _numOps.Multiply(q1Error, q1Error);

            // MSE gradient: -2 * (target - prediction)
            var q1Grad = _numOps.Multiply(_numOps.FromDouble(-2.0), q1Error);
            var q1ErrorVec = new Vector<T>(1);
            q1ErrorVec[0] = q1Grad;
            var q1GradTensor = Tensor<T>.FromVector(q1ErrorVec);
            _q1Network.Backpropagate(q1GradTensor);

            var q1Gradients = _q1Network.GetParameterGradients();
            _q1Network.ApplyGradients(q1Gradients, _options.QLearningRate);

            // Update Q2
            var q2OutputTensor = _q2Network.Predict(stateActionTensor);
            var q2Value = q2OutputTensor.ToVector()[0];
            var q2Error = _numOps.Subtract(targetQ, q2Value);
            var q2Loss = _numOps.Multiply(q2Error, q2Error);

            // MSE gradient: -2 * (target - prediction)
            var q2Grad = _numOps.Multiply(_numOps.FromDouble(-2.0), q2Error);
            var q2ErrorVec = new Vector<T>(1);
            q2ErrorVec[0] = q2Grad;
            var q2GradTensor = Tensor<T>.FromVector(q2ErrorVec);
            _q2Network.Backpropagate(q2GradTensor);

            var q2Gradients = _q2Network.GetParameterGradients();
            _q2Network.ApplyGradients(q2Gradients, _options.QLearningRate);

            totalLoss = _numOps.Add(totalLoss, _numOps.Add(q1Loss, q2Loss));
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count * 2));
    }

    private T UpdatePolicy(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Compute advantage: A(s,a) = Q(s,a) - V(s)
            var stateAction = ConcatenateStateAction(experience.State, experience.Action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var q1OutputTensor = _q1Network.Predict(stateActionTensor);
            var q1Value = q1OutputTensor.ToVector()[0];
            var q2OutputTensor = _q2Network.Predict(stateActionTensor);
            var q2Value = q2OutputTensor.ToVector()[0];
            var qValue = MathHelper.Min<T>(q1Value, q2Value);

            var stateTensor = Tensor<T>.FromVector(experience.State);
            var vOutputTensor = _valueNetwork.Predict(stateTensor);
            var vValue = vOutputTensor.ToVector()[0];
            var advantage = _numOps.Subtract(qValue, vValue);

            // Advantage-weighted regression: exp(advantage / temperature) * log_prob(a|s)
            var weight = NumOps.Exp(_numOps.Divide(advantage, _options.Temperature));
            weight = MathHelper.Clamp<T>(weight, _numOps.FromDouble(0.0), _numOps.FromDouble(100.0));

            // Simplified policy loss (weighted MSE to match action)
            var predictedAction = SelectAction(experience.State, training: false);
            T actionDiff = _numOps.Zero;
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var diff = _numOps.Subtract(experience.Action[i], predictedAction[i]);
                actionDiff = _numOps.Add(actionDiff, _numOps.Multiply(diff, diff));
            }

            var policyLoss = _numOps.Multiply(weight, actionDiff);
            totalLoss = _numOps.Add(totalLoss, policyLoss);

            // Backpropagate
            var gradientVec = new Vector<T>(_options.ActionSize * 2);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var diff = _numOps.Subtract(predictedAction[i], experience.Action[i]);
                gradientVec[i] = _numOps.Multiply(weight, diff);
            }

            var gradientTensor = Tensor<T>.FromVector(gradientVec);
            _policyNetwork.Backpropagate(gradientTensor);

            var gradients = _policyNetwork.GetParameterGradients();
            _policyNetwork.ApplyGradients(gradients, _options.PolicyLearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private void SoftUpdateTargetNetwork()
    {
        var sourceParams = _valueNetwork.GetParameters();
        var targetParams = _targetValueNetwork.GetParameters();

        var oneMinusTau = _numOps.Subtract(_numOps.One, _options.TargetUpdateTau);
        var updatedParams = new Vector<T>(targetParams.Length);

        for (int i = 0; i < targetParams.Length; i++)
        {
            var sourceContrib = _numOps.Multiply(_options.TargetUpdateTau, sourceParams[i]);
            var targetContrib = _numOps.Multiply(oneMinusTau, targetParams[i]);
            updatedParams[i] = _numOps.Add(sourceContrib, targetContrib);
        }

        _targetValueNetwork.SetParameters(updatedParams);
    }

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.SetParameters(sourceParams.Clone());
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
            ModelType = ModelType.IQLAgent,
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var policyParams = ExtractNetworkParameters(_policyNetwork);
        var valueParams = ExtractNetworkParameters(_valueNetwork);
        var q1Params = ExtractNetworkParameters(_q1Network);
        var q2Params = ExtractNetworkParameters(_q2Network);

        var total = policyParams.Length + valueParams.Length + q1Params.Length + q2Params.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in policyParams) vector[idx++] = p;
        foreach (var p in valueParams) vector[idx++] = p;
        foreach (var p in q1Params) vector[idx++] = p;
        foreach (var p in q2Params) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var policyParams = ExtractNetworkParameters(_policyNetwork);
        var valueParams = ExtractNetworkParameters(_valueNetwork);
        var q1Params = ExtractNetworkParameters(_q1Network);
        var q2Params = ExtractNetworkParameters(_q2Network);

        int idx = 0;
        var policyVec = new Vector<T>(policyParams.Length);
        var valueVec = new Vector<T>(valueParams.Length);
        var q1Vec = new Vector<T>(q1Params.Length);
        var q2Vec = new Vector<T>(q2Params.Length);

        for (int i = 0; i < policyParams.Length; i++) policyVec[i] = parameters[idx++];
        for (int i = 0; i < valueParams.Length; i++) valueVec[i] = parameters[idx++];
        for (int i = 0; i < q1Params.Length; i++) q1Vec[i] = parameters[idx++];
        for (int i = 0; i < q2Params.Length; i++) q2Vec[i] = parameters[idx++];

        UpdateNetworkParameters(_policyNetwork, policyVec);
        UpdateNetworkParameters(_valueNetwork, valueVec);
        UpdateNetworkParameters(_q1Network, q1Vec);
        UpdateNetworkParameters(_q2Network, q2Vec);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new IQLAgent<T>(_options);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // IQL uses offline training with separate network updates
        // Gradient application is handled by individual network updates
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_updateCount);

        var policyBytes = SerializeNetwork(_policyNetwork);
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        var valueBytes = SerializeNetwork(_valueNetwork);
        writer.Write(valueBytes.Length);
        writer.Write(valueBytes);

        var q1Bytes = SerializeNetwork(_q1Network);
        writer.Write(q1Bytes.Length);
        writer.Write(q1Bytes);

        var q2Bytes = SerializeNetwork(_q2Network);
        writer.Write(q2Bytes.Length);
        writer.Write(q2Bytes);

        var targetValueBytes = SerializeNetwork(_targetValueNetwork);
        writer.Write(targetValueBytes.Length);
        writer.Write(targetValueBytes);

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

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        DeserializeNetwork(_policyNetwork, policyBytes);

        var valueLength = reader.ReadInt32();
        var valueBytes = reader.ReadBytes(valueLength);
        DeserializeNetwork(_valueNetwork, valueBytes);

        var q1Length = reader.ReadInt32();
        var q1Bytes = reader.ReadBytes(q1Length);
        DeserializeNetwork(_q1Network, q1Bytes);

        var q2Length = reader.ReadInt32();
        var q2Bytes = reader.ReadBytes(q2Length);
        DeserializeNetwork(_q2Network, q2Bytes);

        var targetValueLength = reader.ReadInt32();
        var targetValueBytes = reader.ReadBytes(targetValueLength);
        DeserializeNetwork(_targetValueNetwork, targetValueBytes);
    }

    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    private Vector<T> ExtractNetworkParameters(NeuralNetwork<T> network)
    {
        return network.GetParameters();
    }

    private void UpdateNetworkParameters(NeuralNetwork<T> network, Vector<T> parameters)
    {
        network.SetParameters(parameters);
    }

    private byte[] SerializeNetwork(NeuralNetwork<T> network)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        var parameters = network.GetParameters();
        writer.Write(parameters.Length);

        foreach (var param in parameters)
        {
            writer.Write(MathHelper.GetNumericOperations<T>().ToDouble(param));
        }

        return ms.ToArray();
    }

    private void DeserializeNetwork(NeuralNetwork<T> network, byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        int paramCount = reader.ReadInt32();
        var parameters = new Vector<T>(paramCount);

        for (int i = 0; i < paramCount; i++)
        {
            parameters[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        network.SetParameters(parameters);
    }
}
