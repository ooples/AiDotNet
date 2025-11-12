using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

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
public class DDPGAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly DDPGOptions<T> _options;
    private readonly UniformReplayBuffer<T> _replayBuffer;
    private readonly OrnsteinUhlenbeckNoise<T> _noise;

    private NeuralNetwork<T> _actorNetwork;
    private NeuralNetwork<T> _actorTargetNetwork;
    private NeuralNetwork<T> _criticNetwork;
    private NeuralNetwork<T> _criticTargetNetwork;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    public DDPGAgent(DDPGOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.ActorLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.CriticLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        })
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _replayBuffer = new UniformReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
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

    private NeuralNetwork<T> BuildActorNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.ActorHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer with tanh activation to bound actions to [-1, 1]
        layers.Add(new DenseLayer<T>(prevSize, _options.ActionSize, new TanhActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            Layers = layers,
            TaskType = TaskType.Regression
        };

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
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output single Q-value
        layers.Add(new DenseLayer<T>(prevSize, 1, new LinearActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = inputSize,
            OutputSize = 1,
            Layers = layers,
            TaskType = TaskType.Regression
        };

        return new NeuralNetwork<T>(architecture, _options.CriticLossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _actorNetwork.Forward(state);

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
        _replayBuffer.Add(new Experience<T>(state, action, reward, nextState, done));
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

    private T UpdateCritic(List<Experience<T>> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            // Compute target Q-value
            var nextAction = _actorTargetNetwork.Forward(exp.NextState);
            var nextStateAction = ConcatenateStateAction(exp.NextState, nextAction);
            var nextQ = _criticTargetNetwork.Forward(nextStateAction)[0];

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
            var currentQ = _criticNetwork.Forward(stateAction)[0];

            // Compute loss
            var target = new Vector<T>(1) { [0] = targetQ };
            var prediction = new Vector<T>(1) { [0] = currentQ };
            var loss = _options.CriticLossFunction.ComputeLoss(prediction, target);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backprop
            var gradient = _options.CriticLossFunction.ComputeGradient(prediction, target);
            _criticNetwork.Backward(gradient);
        }

        // Update critic weights
        UpdateNetworkParameters(_criticNetwork, _options.CriticLearningRate);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T UpdateActor(List<Experience<T>> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            // Compute action from actor
            var action = _actorNetwork.Forward(exp.State);

            // Compute Q-value for this action
            var stateAction = ConcatenateStateAction(exp.State, action);
            var q = _criticNetwork.Forward(stateAction)[0];

            // Actor loss is negative Q-value (we want to maximize Q)
            totalLoss = NumOps.Subtract(totalLoss, q);

            // Simplified gradient computation for actor
            var gradOutput = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < gradOutput.Length; i++)
            {
                gradOutput[i] = NumOps.Multiply(NumOps.FromDouble(-0.01), q);
            }

            _actorNetwork.Backward(gradOutput);
        }

        // Update actor weights
        UpdateNetworkParameters(_actorNetwork, _options.ActorLearningRate);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private void SoftUpdateTargets()
    {
        SoftUpdateNetwork(_actorNetwork, _actorTargetNetwork);
        SoftUpdateNetwork(_criticNetwork, _criticTargetNetwork);
    }

    private void SoftUpdateNetwork(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceParams = source.GetFlattenedParameters();
        var targetParams = target.GetFlattenedParameters();

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
        var params_ = network.GetFlattenedParameters();
        var grads = network.GetFlattenedGradients();

        for (int i = 0; i < params_.Length; i++)
        {
            var update = NumOps.Multiply(learningRate, grads[i]);
            params_[i] = NumOps.Subtract(params_[i], update);
        }

        network.UpdateParameters(params_);
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
            TrainingSampleCount = _replayBuffer.Count,
            Parameters = GetParameters()
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
    public override Matrix<T> GetParameters()
    {
        var actorParams = _actorNetwork.GetFlattenedParameters();
        var criticParams = _criticNetwork.GetFlattenedParameters();

        var total = actorParams.Length + criticParams.Length;
        var matrix = new Matrix<T>(total, 1);

        int idx = 0;
        foreach (var p in actorParams) matrix[idx++, 0] = p;
        foreach (var p in criticParams) matrix[idx++, 0] = p;

        return matrix;
    }

    /// <inheritdoc/>
    public override void SetParameters(Matrix<T> parameters)
    {
        var actorParams = _actorNetwork.GetFlattenedParameters();
        var criticParams = _criticNetwork.GetFlattenedParameters();

        int idx = 0;
        var actorVec = new Vector<T>(actorParams.Length);
        var criticVec = new Vector<T>(criticParams.Length);

        for (int i = 0; i < actorParams.Length; i++) actorVec[i] = parameters[idx++, 0];
        for (int i = 0; i < criticParams.Length; i++) criticVec[i] = parameters[idx++, 0];

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
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return (GetParameters(), NumOps.Zero);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        // Not directly applicable for DDPG
    }

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        target.UpdateParameters(source.GetFlattenedParameters());
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
            var dx = _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(-_theta), _state[i]),
                _numOps.Multiply(_numOps.FromDouble(_sigma),
                    MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.One))
            );

            _state[i] = NumOps.Add(_state[i], dx);
            noise[i] = _state[i];
        }

        return noise;
    }

    public void Reset()
    {
        _state = new Vector<T>(_size);
    }
}
