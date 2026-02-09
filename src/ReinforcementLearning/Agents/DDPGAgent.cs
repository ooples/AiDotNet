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

namespace AiDotNet.ReinforcementLearning.Agents.DDPG;

/// <summary>
/// Deep Deterministic Policy Gradient (DDPG) agent for continuous control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DDPG is an actor-critic algorithm designed for continuous action spaces. It learns
/// a deterministic policy (actor) and uses an off-policy approach with experience replay
/// and target networks, extending DQN ideas to continuous control.
/// </para>
/// <para><b>For Beginners:</b>
/// DDPG is perfect for controlling things that need precise, continuous adjustments like:
/// - Robot arm angles (not just "left" or "right", but exact degrees)
/// - Car steering and acceleration (smooth continuous values)
/// - Temperature control, volume levels, etc.
///
/// Key components:
/// - **Actor**: Learns the best action to take (deterministic policy)
/// - **Critic**: Evaluates how good that action is (Q-value)
/// - **Target Networks**: Stable copies for training
/// - **Exploration Noise**: Adds randomness during training for exploration
///
/// Think of it like learning to drive: the actor is your decision-making (how much to
/// turn the wheel), the critic is your judgment (was that a good turn?), and noise
/// is trying slight variations to discover better techniques.
/// </para>
/// <para><b>Reference:</b>
/// Lillicrap et al., "Continuous control with deep reinforcement learning", 2015.
/// </para>
/// </remarks>
public class DDPGAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DDPGOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private readonly OrnsteinUhlenbeckNoise<T> _noise;

    private NeuralNetwork<T> _actorNetwork;
    private NeuralNetwork<T> _actorTargetNetwork;
    private NeuralNetwork<T> _criticNetwork;
    private NeuralNetwork<T> _criticTargetNetwork;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    public DDPGAgent(DDPGOptions<T> options)
        : base(CreateBaseOptions(options))
    {
        _options = options;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _noise = new OrnsteinUhlenbeckNoise<T>(options.ActionSize, NumOps, Random, options.ExplorationNoise);
        _steps = 0;

        // Build networks
        _actorNetwork = BuildActorNetwork();
        _actorTargetNetwork = BuildActorNetwork();
        _criticNetwork = BuildCriticNetwork();
        _criticTargetNetwork = BuildCriticNetwork();

        // Initialize targets
        CopyNetworkWeights(_actorNetwork, _actorTargetNetwork);
        CopyNetworkWeights(_criticNetwork, _criticTargetNetwork);

        Networks.Add(_actorNetwork);
        Networks.Add(_actorTargetNetwork);
        Networks.Add(_criticNetwork);
        Networks.Add(_criticTargetNetwork);
    }


    private static ReinforcementLearningOptions<T> CreateBaseOptions(DDPGOptions<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.ActorLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.CriticLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        };
    }

    private NeuralNetwork<T> BuildActorNetwork()
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
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture);
    }

    private NeuralNetwork<T> BuildCriticNetwork()
    {
        // Critic takes state + action as input
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
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, _options.CriticLossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var actionTensor = _actorNetwork.Predict(stateTensor);
        var action = actionTensor.ToVector();

        if (training)
        {
            // Add exploration noise
            var noise = _noise.Sample();
            for (int i = 0; i < action.Length; i++)
            {
                action[i] = MathHelper.Clamp<T>(
                    NumOps.Add(action[i], noise[i]),
                    NumOps.FromDouble(-1.0),
                    NumOps.FromDouble(1.0)
                );
            }
        }

        return action;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
    }

    /// <inheritdoc/>
    public override T Train()
    {
        _steps++;
        TrainingSteps++;

        if (_steps < _options.WarmupSteps || !_replayBuffer.CanSample(_options.BatchSize))
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);

        // Update critic
        var criticLoss = UpdateCritic(batch);

        // Update actor
        var actorLoss = UpdateActor(batch);

        // Soft update target networks
        SoftUpdateTargets();

        var totalLoss = NumOps.Add(criticLoss, actorLoss);
        LossHistory.Add(totalLoss);

        return totalLoss;
    }

    private T UpdateCritic(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            // Compute target Q-value
            var nextStateTensor = Tensor<T>.FromVector(exp.NextState);
            var nextActionTensor = _actorTargetNetwork.Predict(nextStateTensor);
            var nextAction = nextActionTensor.ToVector();
            var nextStateAction = ConcatenateStateAction(exp.NextState, nextAction);
            var nextStateActionTensor = Tensor<T>.FromVector(nextStateAction);
            var nextQTensor = _criticTargetNetwork.Predict(nextStateActionTensor);
            var nextQ = nextQTensor.ToVector()[0];

            T targetQ;
            if (exp.Done)
            {
                targetQ = exp.Reward;
            }
            else
            {
                targetQ = NumOps.Add(exp.Reward, NumOps.Multiply(DiscountFactor, nextQ));
            }

            // Compute current Q-value
            var stateAction = ConcatenateStateAction(exp.State, exp.Action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var currentQTensor = _criticNetwork.Predict(stateActionTensor);
            var currentQ = currentQTensor.ToVector()[0];

            // Compute loss
            var target = new Vector<T>(1) { [0] = targetQ };
            var prediction = new Vector<T>(1) { [0] = currentQ };
            var loss = _options.CriticLossFunction.CalculateLoss(prediction, target);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagate gradient through critic network
            var gradient = _options.CriticLossFunction.CalculateDerivative(prediction, target);
            var gradientTensor = Tensor<T>.FromVector(gradient);
            _criticNetwork.Backpropagate(gradientTensor);
        }

        // Update critic weights using accumulated gradients
        UpdateNetworkParameters(_criticNetwork, _options.CriticLearningRate);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T UpdateActor(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            // Compute action from actor
            var stateTensor = Tensor<T>.FromVector(exp.State);
            var actionTensor = _actorNetwork.Predict(stateTensor);
            var action = actionTensor.ToVector();

            // Compute Q-value for this action
            var stateAction = ConcatenateStateAction(exp.State, action);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var qTensor = _criticNetwork.Predict(stateActionTensor);
            var q = qTensor.ToVector()[0];

            // Actor loss is negative Q-value (we want to maximize Q)
            totalLoss = NumOps.Subtract(totalLoss, q);

            // Compute deterministic policy gradient
            // DDPG gradient: ∇θ J = E[∇a Q(s,a)|a=μ(s) * ∇θ μ(s)]
            // This is the chain rule: gradient of Q w.r.t. actions times gradient of policy w.r.t. parameters
            var outputGradient = ComputeDDPGPolicyGradient(exp.State, action);
            var outputGradientTensor = Tensor<T>.FromVector(outputGradient);
            _actorNetwork.Backpropagate(outputGradientTensor);
        }

        // Update actor weights
        UpdateNetworkParameters(_actorNetwork, _options.ActorLearningRate);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }


    private Vector<T> ComputeDDPGPolicyGradient(Vector<T> state, Vector<T> action)
    {
        // DDPG uses deterministic policy gradient: ∇θ J = E[∇a Q(s,a)|a=μ(s) * ∇θ μ(s)]
        //
        // Step 1: Compute ∇a Q(s,a) - gradient of Q-value w.r.t. actions
        // We approximate this using finite differences since we don't have direct access to critic gradients
        //
        // Step 2: This gradient is then backpropagated through the actor network
        // to compute ∇θ μ(s) via the chain rule

        var gradient = new Vector<T>(action.Length);
        T epsilon = NumOps.FromDouble(0.001); // Small perturbation for finite differences

        // Compute gradient of Q w.r.t. each action dimension using finite differences
        for (int i = 0; i < action.Length; i++)
        {
            // Create perturbed action: a + ε
            var actionPlus = new Vector<T>(action.Length);
            for (int j = 0; j < action.Length; j++)
            {
                actionPlus[j] = action[j];
            }
            actionPlus[i] = NumOps.Add(action[i], epsilon);

            // Create perturbed action: a - ε
            var actionMinus = new Vector<T>(action.Length);
            for (int j = 0; j < action.Length; j++)
            {
                actionMinus[j] = action[j];
            }
            actionMinus[i] = NumOps.Subtract(action[i], epsilon);

            // Compute Q(s, a+ε)
            var stateActionPlus = ConcatenateStateAction(state, actionPlus);
            var qPlus = _criticNetwork.Predict(Tensor<T>.FromVector(stateActionPlus)).ToVector()[0];

            // Compute Q(s, a-ε)
            var stateActionMinus = ConcatenateStateAction(state, actionMinus);
            var qMinus = _criticNetwork.Predict(Tensor<T>.FromVector(stateActionMinus)).ToVector()[0];

            // Finite difference approximation: ∂Q/∂a_i ≈ (Q(s,a+ε) - Q(s,a-ε)) / (2ε)
            var dQda = NumOps.Divide(
                NumOps.Subtract(qPlus, qMinus),
                NumOps.Multiply(NumOps.FromDouble(2.0), epsilon)
            );

            // This gradient will be backpropagated through the actor network
            // We negate because we want to maximize Q (gradient ascent)
            gradient[i] = NumOps.Negate(dQda);
        }

        return gradient;
    }

    private void SoftUpdateTargets()
    {
        SoftUpdateNetwork(_actorNetwork, _actorTargetNetwork);
        SoftUpdateNetwork(_criticNetwork, _criticTargetNetwork);
    }

    private void SoftUpdateNetwork(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        var tau = _options.TargetUpdateTau;
        var oneMinusTau = NumOps.Subtract(NumOps.One, tau);

        for (int i = 0; i < targetParams.Length; i++)
        {
            targetParams[i] = NumOps.Add(
                NumOps.Multiply(tau, sourceParams[i]),
                NumOps.Multiply(oneMinusTau, targetParams[i])
            );
        }

        target.UpdateParameters(targetParams);
    }

    private void UpdateNetworkParameters(NeuralNetwork<T> network, T learningRate)
    {
        // Apply accumulated gradients from Backpropagate() calls
        var parameters = network.GetParameters();
        var gradients = network.GetGradients();

        if (gradients.Length > 0)
        {
            for (int i = 0; i < parameters.Length && i < gradients.Length; i++)
            {
                // Gradient descent: θ ← θ - α * ∇θ J
                var update = NumOps.Multiply(learningRate, gradients[i]);
                parameters[i] = NumOps.Subtract(parameters[i], update);
            }

            network.UpdateParameters(parameters);
            // Gradients are managed internally by the network
        }
    }

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        var combined = new Vector<T>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++)
            combined[i] = state[i];
        for (int i = 0; i < action.Length; i++)
            combined[state.Length + i] = action[i];
        return combined;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["ReplayBufferSize"] = NumOps.FromDouble(_replayBuffer.Count);
        baseMetrics["Steps"] = NumOps.FromDouble(_steps);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DDPGAgent,
            FeatureCount = _options.StateSize,
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);

        void WriteNetwork(NeuralNetwork<T> net)
        {
            var bytes = net.Serialize();
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        WriteNetwork(_actorNetwork);
        WriteNetwork(_actorTargetNetwork);
        WriteNetwork(_criticNetwork);
        WriteNetwork(_criticTargetNetwork);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize

        void ReadNetwork(NeuralNetwork<T> net)
        {
            var len = reader.ReadInt32();
            var bytes = reader.ReadBytes(len);
            net.Deserialize(bytes);
        }

        ReadNetwork(_actorNetwork);
        ReadNetwork(_actorTargetNetwork);
        ReadNetwork(_criticNetwork);
        ReadNetwork(_criticTargetNetwork);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var actorParams = _actorNetwork.GetParameters();
        var criticParams = _criticNetwork.GetParameters();

        var total = actorParams.Length + criticParams.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in actorParams) vector[idx++] = p;
        foreach (var p in criticParams) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var actorParams = _actorNetwork.GetParameters();
        var criticParams = _criticNetwork.GetParameters();

        int idx = 0;
        var actorVec = new Vector<T>(actorParams.Length);
        var criticVec = new Vector<T>(criticParams.Length);

        for (int i = 0; i < actorParams.Length; i++) actorVec[i] = parameters[idx++];
        for (int i = 0; i < criticParams.Length; i++) criticVec[i] = parameters[idx++];

        _actorNetwork.UpdateParameters(actorVec);
        _criticNetwork.UpdateParameters(criticVec);

        CopyNetworkWeights(_actorNetwork, _actorTargetNetwork);
        CopyNetworkWeights(_criticNetwork, _criticTargetNetwork);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new DDPGAgent<T>(_options);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            "DDPG uses actor-critic training via Train() method. " +
            "Direct gradient computation through this interface is not applicable.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        throw new NotSupportedException(
            "DDPG uses actor-critic training via Train() method. " +
            "Direct gradient application through this interface is not applicable.");
    }

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        target.UpdateParameters(source.GetParameters());
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

/// <summary>
/// Ornstein-Uhlenbeck process for temporally correlated exploration noise.
/// </summary>
internal class OrnsteinUhlenbeckNoise<T>
{
    private readonly int _size;
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private readonly double _theta;
    private readonly double _sigma;
    private Vector<T> _state;

    public OrnsteinUhlenbeckNoise(int size, INumericOperations<T> numOps, Random random, double sigma, double theta = 0.15)
    {
        _size = size;
        _numOps = numOps;
        _random = random;
        _theta = theta;
        _sigma = sigma;
        _state = new Vector<T>(size);
    }

    public Vector<T> Sample()
    {
        var noise = new Vector<T>(_size);

        for (int i = 0; i < _size; i++)
        {
            var drift = _numOps.Multiply(_numOps.FromDouble(-_theta), _state[i]);
            var diffusion = _numOps.Multiply(_numOps.FromDouble(_sigma),
                MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.One));
            var dx = _numOps.Add(drift, diffusion);

            _state[i] = _numOps.Add(_state[i], dx);
            noise[i] = _state[i];
        }

        return noise;
    }

    public void Reset()
    {
        _state = new Vector<T>(_size);
    }
}
