using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

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

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private Random _random;
    private int _stepCount;
    private int _updateCount;

    public TD3Agent(TD3Options<T> options) : base(CreateBaseOptions(options))
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = options.Seed.HasValue ? RandomHelper.CreateSeededRandom(options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _stepCount = 0;
        _updateCount = 0;

        // Initialize networks directly in constructor
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

        // Initialize replay buffer
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize, _options.Seed);
    }

    private static ReinforcementLearningOptions<T> CreateBaseOptions(TD3Options<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.ActorLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        };
    }

    private NeuralNetwork<T> CreateActorNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.ActorHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer with tanh activation to bound actions to [-1, 1]
        layers.Add(new DenseLayer<T>(prevSize, _options.ActionSize, (IActivationFunction<T>)new TanhActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());
    }

    private NeuralNetwork<T> CreateCriticNetwork()
    {
        var layers = new List<ILayer<T>>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int prevSize = inputSize;

        foreach (var hiddenSize in _options.CriticHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output single Q-value
        layers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var actionTensor = _actorNetwork.Predict(stateTensor);
        var action = actionTensor.ToVector();

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
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
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

    private T UpdateCritics(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = _numOps.Zero;

        // CRITICAL FIX: Sample() returns List<Experience<T>>, not tuple
        foreach (var experience in batch)
        {
            // Compute target Q-value with target policy smoothing
            var nextStateTensor = Tensor<T>.FromVector(experience.NextState);
            var nextActionTensor = _targetActorNetwork.Predict(nextStateTensor);
            var nextAction = nextActionTensor.ToVector();

            // Add clipped noise to target action (target policy smoothing)
            for (int i = 0; i < nextAction.Length; i++)
            {
                var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.FromDouble(_options.TargetPolicyNoise));
                noise = MathHelper.Clamp<T>(noise, _numOps.FromDouble(-_options.TargetNoiseClip), _numOps.FromDouble(_options.TargetNoiseClip));
                nextAction[i] = _numOps.Add(nextAction[i], noise);
                nextAction[i] = MathHelper.Clamp<T>(nextAction[i], _numOps.FromDouble(-1), _numOps.FromDouble(1));
            }

            // Concatenate next state and next action for critic input
            var nextStateAction = ConcatenateStateAction(experience.NextState, nextAction);

            // Compute twin Q-targets and take minimum (clipped double Q-learning)
            var nextStateActionTensor = Tensor<T>.FromVector(nextStateAction);
            var q1TargetTensor = _targetCritic1Network.Predict(nextStateActionTensor);
            var q2TargetTensor = _targetCritic2Network.Predict(nextStateActionTensor);
            var q1Target = q1TargetTensor.ToVector()[0];
            var q2Target = q2TargetTensor.ToVector()[0];
            var minQTarget = MathHelper.Min<T>(q1Target, q2Target);

            // Compute TD target
            T targetQ;
            if (experience.Done)
            {
                targetQ = experience.Reward;
            }
            else
            {
                // Ensure both DiscountFactor and minQTarget are not null before using in arithmetic operations
                if (_options.DiscountFactor is not null && minQTarget is not null)
                {
                    var discountedQ = _numOps.Multiply(_options.DiscountFactor, minQTarget);
                    targetQ = _numOps.Add(experience.Reward, discountedQ);
                }
                else
                {
                    targetQ = experience.Reward;
                }
            }

            // Concatenate state and action for critic input
            var stateAction = ConcatenateStateAction(experience.State, experience.Action);

            // Update Critic 1
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var q1ValueTensor = _critic1Network.Predict(stateActionTensor);
            var q1Values = q1ValueTensor.ToVector();
            var q1Value = q1Values[0];

            // Create target vector for loss computation
            var targetVec = new Vector<T>(1);
            targetVec[0] = targetQ;

            // Compute loss and gradients
            var loss1 = _options.CriticLossFunction.CalculateLoss(q1Values, targetVec);
            var gradients1 = _options.CriticLossFunction.CalculateDerivative(q1Values, targetVec);
            var gradientsTensor1 = Tensor<T>.FromVector(gradients1);
            _critic1Network.Backpropagate(gradientsTensor1);

            // Update weights
            var params1 = _critic1Network.GetParameters();
            for (int i = 0; i < params1.Length; i++)
            {
                var update = _numOps.Multiply(_options.CriticLearningRate, gradients1[i % gradients1.Length]);
                params1[i] = _numOps.Subtract(params1[i], update);
            }
            _critic1Network.UpdateParameters(params1);

            // Update Critic 2
            var q2ValueTensor = _critic2Network.Predict(stateActionTensor);
            var q2Values = q2ValueTensor.ToVector();
            var q2Value = q2Values[0];

            var loss2 = _options.CriticLossFunction.CalculateLoss(q2Values, targetVec);
            var gradients2 = _options.CriticLossFunction.CalculateDerivative(q2Values, targetVec);
            var gradientsTensor2 = Tensor<T>.FromVector(gradients2);
            _critic2Network.Backpropagate(gradientsTensor2);

            // Update weights
            var params2 = _critic2Network.GetParameters();
            for (int i = 0; i < params2.Length; i++)
            {
                var update = _numOps.Multiply(_options.CriticLearningRate, gradients2[i % gradients2.Length]);
                params2[i] = _numOps.Subtract(params2[i], update);
            }
            _critic2Network.UpdateParameters(params2);

            // Accumulate loss (already computed above)
            totalLoss = _numOps.Add(totalLoss, _numOps.Add(loss1, loss2));
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count * 2));
    }

    private void UpdateActor(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        foreach (var experience in batch)
        {
            // Compute action from current policy
            var stateTensor = Tensor<T>.FromVector(experience.State);
            var actionTensor = _actorNetwork.Predict(stateTensor);
            var action = actionTensor.ToVector();

            // Concatenate state and action
            var stateAction = ConcatenateStateAction(experience.State, action);

            // Compute Q-value from critic 1 (use only one critic for policy gradient)
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var qValueTensor = _critic1Network.Predict(stateActionTensor);
            var qValue = qValueTensor.ToVector()[0];

            // Policy gradient: maximize Q-value, so negate for gradient ascent
            var policyGradient = new Vector<T>(1);
            policyGradient[0] = _numOps.Negate(qValue);

            // Backpropagate through critic to get gradient w.r.t. actions
            var policyGradientTensor = Tensor<T>.FromVector(policyGradient);
            var actionGradientTensor = _critic1Network.Backpropagate(policyGradientTensor);
            var actionGradient = actionGradientTensor.ToVector();

            // Extract action gradients (remove state part)
            var actorGradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                actorGradient[i] = actionGradient[_options.StateSize + i];
            }

            // Backpropagate through actor
            var actorGradientTensor = Tensor<T>.FromVector(actorGradient);
            _actorNetwork.Backpropagate(actorGradientTensor);

            // Update actor weights
            var actorParams = _actorNetwork.GetParameters();
            for (int i = 0; i < actorParams.Length; i++)
            {
                var update = _numOps.Multiply(_options.ActorLearningRate, actorGradient[i % actorGradient.Length]);
                actorParams[i] = _numOps.Subtract(actorParams[i], update);
            }
            _actorNetwork.UpdateParameters(actorParams);
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
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        var tau = _options.TargetUpdateTau;
        var oneMinusTau = _numOps.Subtract(_numOps.One, tau);

        for (int i = 0; i < targetParams.Length; i++)
        {
            targetParams[i] = _numOps.Add(
                _numOps.Multiply(tau, sourceParams[i]),
                _numOps.Multiply(oneMinusTau, targetParams[i])
            );
        }

        target.UpdateParameters(targetParams);
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
            ModelType = ModelType.TD3Agent,
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var actorParams = _actorNetwork.GetParameters();
        var targetActorParams = _targetActorNetwork.GetParameters();
        var critic1Params = _critic1Network.GetParameters();
        var critic2Params = _critic2Network.GetParameters();
        var targetCritic1Params = _targetCritic1Network.GetParameters();
        var targetCritic2Params = _targetCritic2Network.GetParameters();

        var total = actorParams.Length + targetActorParams.Length + critic1Params.Length +
                    critic2Params.Length + targetCritic1Params.Length + targetCritic2Params.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in actorParams) vector[idx++] = p;
        foreach (var p in targetActorParams) vector[idx++] = p;
        foreach (var p in critic1Params) vector[idx++] = p;
        foreach (var p in critic2Params) vector[idx++] = p;
        foreach (var p in targetCritic1Params) vector[idx++] = p;
        foreach (var p in targetCritic2Params) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var actorParams = _actorNetwork.GetParameters();
        var targetActorParams = _targetActorNetwork.GetParameters();
        var critic1Params = _critic1Network.GetParameters();
        var critic2Params = _critic2Network.GetParameters();
        var targetCritic1Params = _targetCritic1Network.GetParameters();
        var targetCritic2Params = _targetCritic2Network.GetParameters();

        int idx = 0;
        var actorVec = new Vector<T>(actorParams.Length);
        var targetActorVec = new Vector<T>(targetActorParams.Length);
        var critic1Vec = new Vector<T>(critic1Params.Length);
        var critic2Vec = new Vector<T>(critic2Params.Length);
        var targetCritic1Vec = new Vector<T>(targetCritic1Params.Length);
        var targetCritic2Vec = new Vector<T>(targetCritic2Params.Length);

        for (int i = 0; i < actorParams.Length; i++) actorVec[i] = parameters[idx++];
        for (int i = 0; i < targetActorParams.Length; i++) targetActorVec[i] = parameters[idx++];
        for (int i = 0; i < critic1Params.Length; i++) critic1Vec[i] = parameters[idx++];
        for (int i = 0; i < critic2Params.Length; i++) critic2Vec[i] = parameters[idx++];
        for (int i = 0; i < targetCritic1Params.Length; i++) targetCritic1Vec[i] = parameters[idx++];
        for (int i = 0; i < targetCritic2Params.Length; i++) targetCritic2Vec[i] = parameters[idx++];

        _actorNetwork.UpdateParameters(actorVec);
        _targetActorNetwork.UpdateParameters(targetActorVec);
        _critic1Network.UpdateParameters(critic1Vec);
        _critic2Network.UpdateParameters(critic2Vec);
        _targetCritic1Network.UpdateParameters(targetCritic1Vec);
        _targetCritic2Network.UpdateParameters(targetCritic2Vec);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new TD3Agent<T>(_options);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            "TD3 uses actor-critic training via Train() method. " +
            "Direct gradient computation through this interface is not applicable.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // TD3 uses direct network updates during training, not manual gradient application
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_stepCount);
        writer.Write(_updateCount);

        var actorBytes = _actorNetwork.Serialize();
        writer.Write(actorBytes.Length);
        writer.Write(actorBytes);

        var targetActorBytes = _targetActorNetwork.Serialize();
        writer.Write(targetActorBytes.Length);
        writer.Write(targetActorBytes);

        var critic1Bytes = _critic1Network.Serialize();
        writer.Write(critic1Bytes.Length);
        writer.Write(critic1Bytes);

        var critic2Bytes = _critic2Network.Serialize();
        writer.Write(critic2Bytes.Length);
        writer.Write(critic2Bytes);

        var targetCritic1Bytes = _targetCritic1Network.Serialize();
        writer.Write(targetCritic1Bytes.Length);
        writer.Write(targetCritic1Bytes);

        var targetCritic2Bytes = _targetCritic2Network.Serialize();
        writer.Write(targetCritic2Bytes.Length);
        writer.Write(targetCritic2Bytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        _stepCount = reader.ReadInt32();
        _updateCount = reader.ReadInt32();

        var actorLength = reader.ReadInt32();
        var actorBytes = reader.ReadBytes(actorLength);
        _actorNetwork.Deserialize(actorBytes);

        var targetActorLength = reader.ReadInt32();
        var targetActorBytes = reader.ReadBytes(targetActorLength);
        _targetActorNetwork.Deserialize(targetActorBytes);

        var critic1Length = reader.ReadInt32();
        var critic1Bytes = reader.ReadBytes(critic1Length);
        _critic1Network.Deserialize(critic1Bytes);

        var critic2Length = reader.ReadInt32();
        var critic2Bytes = reader.ReadBytes(critic2Length);
        _critic2Network.Deserialize(critic2Bytes);

        var targetCritic1Length = reader.ReadInt32();
        var targetCritic1Bytes = reader.ReadBytes(targetCritic1Length);
        _targetCritic1Network.Deserialize(targetCritic1Bytes);

        var targetCritic2Length = reader.ReadInt32();
        var targetCritic2Bytes = reader.ReadBytes(targetCritic2Length);
        _targetCritic2Network.Deserialize(targetCritic2Bytes);
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
