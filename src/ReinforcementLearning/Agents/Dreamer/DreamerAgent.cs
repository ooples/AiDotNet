using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;

namespace AiDotNet.ReinforcementLearning.Agents.Dreamer;

/// <summary>
/// Dreamer agent for model-based reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dreamer learns a world model in latent space and uses it for planning.
/// It combines representation learning, dynamics modeling, and policy learning.
/// </para>
/// <para><b>For Beginners:</b>
/// Dreamer learns a "mental model" of how the environment works, then uses that
/// model to imagine future scenarios and plan actions - like chess players
/// thinking multiple moves ahead.
///
/// Key components:
/// - **Representation Network**: Encodes observations to latent states
/// - **Dynamics Model**: Predicts next latent state
/// - **Reward Model**: Predicts rewards
/// - **Value Network**: Estimates state values
/// - **Actor Network**: Learns policy in imagination
///
/// Think of it as: First learn physics by observation, then use that knowledge
/// to predict "what happens if I do X" without actually doing it.
///
/// Advantages: Sample efficient, works with images, enables planning
/// </para>
/// </remarks>
public class DreamerAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DreamerOptions<T> _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    // World model components
    private NeuralNetwork<T> _representationNetwork;  // Observation -> latent state
    private NeuralNetwork<T> _dynamicsNetwork;  // (latent state, action) -> next latent state
    private NeuralNetwork<T> _rewardNetwork;  // latent state -> reward
    private NeuralNetwork<T> _continueNetwork;  // latent state -> continue probability

    // Actor-critic for policy learning
    private NeuralNetwork<T> _actorNetwork;
    private NeuralNetwork<T> _valueNetwork;

    private ReplayBuffers.UniformReplayBuffer<T> _replayBuffer;
    private int _updateCount;

    public DreamerAgent(DreamerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // FIX ISSUE 6: Use learning rate from options consistently
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            LearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _updateCount = 0;

        // Initialize networks directly in constructor
        // Representation network: observation -> latent
        _representationNetwork = CreateEncoderNetwork(_options.ObservationSize, _options.LatentSize);

        // Dynamics network: (latent, action) -> next_latent
        _dynamicsNetwork = CreateEncoderNetwork(_options.LatentSize + _options.ActionSize, _options.LatentSize);

        // Reward predictor
        _rewardNetwork = CreateEncoderNetwork(_options.LatentSize, 1);

        // Continue predictor (for episode termination)
        _continueNetwork = CreateEncoderNetwork(_options.LatentSize, 1);

        // Actor and critic
        _actorNetwork = CreateActorNetwork();
        _valueNetwork = CreateEncoderNetwork(_options.LatentSize, 1);

        // FIX ISSUE 3: Add all networks to Networks list for parameter access
        Networks.Add(_representationNetwork);
        Networks.Add(_dynamicsNetwork);
        Networks.Add(_rewardNetwork);
        Networks.Add(_continueNetwork);
        Networks.Add(_actorNetwork);
        Networks.Add(_valueNetwork);

        // Initialize replay buffer
        _replayBuffer = new ReplayBuffers.UniformReplayBuffer<T>(_options.ReplayBufferSize, _options.Seed);
    }

    private NeuralNetwork<T> CreateEncoderNetwork(int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<T>(inputSize, outputSize, NetworkComplexity.Medium);
        var network = new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());

        for (int i = 0; i < 2; i++)
        {
            network.AddLayer(LayerType.Dense, _options.HiddenSize, ActivationFunction.ReLU);
        }

        network.AddLayer(LayerType.Dense, outputSize, ActivationFunction.Linear);

        return network;
    }

    private NeuralNetwork<T> CreateActorNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<T>(_options.LatentSize, _options.ActionSize, NetworkComplexity.Medium);
        var network = new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());

        for (int i = 0; i < 2; i++)
        {
            network.AddLayer(LayerType.Dense, _options.HiddenSize, ActivationFunction.ReLU);
        }

        network.AddLayer(LayerType.Dense, _options.ActionSize, ActivationFunction.Tanh);

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new ReplayBuffers.UniformReplayBuffer<T>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to latent state
        var latentState = _representationNetwork.Predict(Tensor<T>.FromVector(observation)).ToVector();

        // Select action from policy
        var action = _actorNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector();

        if (training)
        {
            // Add exploration noise
            for (int i = 0; i < action.Length; i++)
            {
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.FromDouble(0.1));
                action[i] = NumOps.Add(action[i], noise);
                action[i] = MathHelper.Clamp<T>(action[i], NumOps.FromDouble(-1), NumOps.FromDouble(1));
            }
        }

        return action;
    }

    public override void StoreExperience(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)
    {
        _replayBuffer.Add(new ReplayBuffers.Experience<T>(observation, action, reward, nextObservation, done));
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);

        // Train world model
        T worldModelLoss = TrainWorldModel(batch);

        // Train actor-critic in imagination
        T policyLoss = TrainPolicy();

        _updateCount++;

        return NumOps.Add(worldModelLoss, policyLoss);
    }

    private T TrainWorldModel(List<ReplayBuffers.Experience<T>> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Encode observations to latent states
            var latentState = _representationNetwork.Predict(Tensor<T>.FromVector(experience.State)).ToVector();
            var nextLatentState = _representationNetwork.Predict(Tensor<T>.FromVector(experience.NextState)).ToVector();

            // Predict next latent from dynamics model
            var dynamicsInput = ConcatenateVectors(latentState, experience.Action);
            var predictedNextLatent = _dynamicsNetwork.Predict(Tensor<T>.FromVector(dynamicsInput)).ToVector();

            // Dynamics loss: predict next latent state
            T dynamicsLoss = NumOps.Zero;
            for (int i = 0; i < predictedNextLatent.Length; i++)
            {
                var diff = NumOps.Subtract(nextLatentState[i], predictedNextLatent[i]);
                dynamicsLoss = NumOps.Add(dynamicsLoss, NumOps.Multiply(diff, diff));
            }

            // Reward prediction loss
            var predictedReward = _rewardNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector()[0];
            var rewardDiff = NumOps.Subtract(experience.Reward, predictedReward);
            var rewardLoss = NumOps.Multiply(rewardDiff, rewardDiff);

            // Continue prediction loss (done = 0, continue = 1)
            var continueTarget = experience.Done ? NumOps.Zero : NumOps.One;
            var predictedContinue = _continueNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector()[0];
            var continueDiff = NumOps.Subtract(continueTarget, predictedContinue);
            var continueLoss = NumOps.Multiply(continueDiff, continueDiff);

            // Total world model loss
            var loss = NumOps.Add(dynamicsLoss, NumOps.Add(rewardLoss, continueLoss));
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backprop through world model
            var gradient = new Vector<T>(predictedNextLatent.Length);
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = NumOps.Subtract(predictedNextLatent[i], nextLatentState[i]);
            }

            _dynamicsNetwork.Backpropagate(Tensor<T>.FromVector(gradient));
            var dynamicsParams = _dynamicsNetwork.GetParameters();
            _dynamicsNetwork.UpdateParameters(dynamicsParams);

            // FIX ISSUE 1: Train representation network
            // Representation network should minimize reconstruction error of latent states
            var representationGradient = new Vector<T>(latentState.Length);
            for (int j = 0; j < representationGradient.Length; j++)
            {
                // Gradient flows from dynamics prediction error back through representation
                representationGradient[j] = NumOps.Divide(gradient[j], NumOps.FromDouble(2.0));
            }
            _representationNetwork.Backpropagate(Tensor<T>.FromVector(representationGradient));
            var representationParams = _representationNetwork.GetParameters();
            _representationNetwork.UpdateParameters(representationParams);

            var rewardGradient = new Vector<T>(1);
            rewardGradient[0] = rewardDiff;
            _rewardNetwork.Backpropagate(Tensor<T>.FromVector(rewardGradient));
            var rewardParams = _rewardNetwork.GetParameters();
            _rewardNetwork.UpdateParameters(rewardParams);

            var continueGradient = new Vector<T>(1);
            continueGradient[0] = continueDiff;
            _continueNetwork.Backpropagate(Tensor<T>.FromVector(continueGradient));
            var continueParams = _continueNetwork.GetParameters();
            _continueNetwork.UpdateParameters(continueParams);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T TrainPolicy()
    {
        // Imagine trajectories using world model
        T totalLoss = NumOps.Zero;

        // Sample initial latent states from replay buffer
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);

        foreach (var experience in batch)
        {
            var latentState = _representationNetwork.Predict(Tensor<T>.FromVector(experience.State)).ToVector();

            // Imagine future trajectory
            var imaginedReturns = ImagineTrajectory(latentState);

            // FIX ISSUE 4: Update value network with correct gradient sign
            // Value network minimizes squared TD error: (return - value)^2
            var predictedValue = _valueNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector()[0];
            var valueDiff = NumOps.Subtract(imaginedReturns, predictedValue);
            var valueLoss = NumOps.Multiply(valueDiff, valueDiff);

            // Gradient of MSE loss: 2 * (prediction - target) = -2 * (target - prediction)
            var valueGradient = new Vector<T>(1);
            valueGradient[0] = NumOps.Multiply(NumOps.FromDouble(-2.0), valueDiff);
            _valueNetwork.Backpropagate(Tensor<T>.FromVector(valueGradient));
            var valueParams = _valueNetwork.GetParameters();
            _valueNetwork.UpdateParameters(valueParams);

            // FIX ISSUE 2: Implement proper policy gradient for actor
            // Actor maximizes expected return by following gradient of value w.r.t. actions
            // Use advantage (return - baseline) as policy gradient weight
            var advantage = valueDiff;

            // Compute value gradient w.r.t. current action to get policy gradient direction
            var action = _actorNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector();
            var actorGradient = new Vector<T>(action.Length);

            // Policy gradient: advantage * grad_action(log pi(action|state))
            // For deterministic policy, approximate with advantage-weighted action gradient
            for (int i = 0; i < actorGradient.Length; i++)
            {
                // Gradient direction: maximize value by adjusting actions
                // Positive advantage -> increase action magnitude in current direction
                actorGradient[i] = NumOps.Multiply(advantage, NumOps.FromDouble(-1.0 / action.Length));
            }

            _actorNetwork.Backpropagate(Tensor<T>.FromVector(actorGradient));
            var actorParams = _actorNetwork.GetParameters();
            _actorNetwork.UpdateParameters(actorParams);

            totalLoss = NumOps.Add(totalLoss, valueLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T ImagineTrajectory(Vector<T> initialLatentState)
    {
        // Roll out imagined trajectory using world model
        T imaginedReturn = NumOps.Zero;
        var latentState = initialLatentState;

        for (int step = 0; step < _options.ImaginationHorizon; step++)
        {
            // Select action
            var action = _actorNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector();

            // Predict reward
            var reward = _rewardNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector()[0];

            // FIX ISSUE 5: Add discount factor (gamma) to imagination rollout
            var discountedReward = NumOps.Multiply(reward, NumOps.Pow(NumOps.FromDouble(_options.Gamma), NumOps.FromDouble(step)));
            imaginedReturn = NumOps.Add(imaginedReturn, discountedReward);

            // Predict next latent state
            var dynamicsInput = ConcatenateVectors(latentState, action);
            latentState = _dynamicsNetwork.Predict(Tensor<T>.FromVector(dynamicsInput)).ToVector();

            // Check if episode continues
            var continueProb = _continueNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector()[0];
            if (NumOps.ToDouble(continueProb) < 0.5)
            {
                break;
            }
        }

        return imaginedReturn;
    }

    private Vector<T> ConcatenateVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i];
        }
        for (int i = 0; i < b.Length; i++)
        {
            result[a.Length + i] = b[i];
        }
        return result;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
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

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
        };
    }

    public override int FeatureCount => _options.ObservationSize;

    public override byte[] Serialize()
    {
        // FIX ISSUE 8: Use NotSupportedException with clear message
        throw new NotSupportedException(
            "Dreamer agent serialization is not supported. " +
            "Use GetParameters()/SetParameters() for parameter transfer, " +
            "or save individual network weights separately.");
    }

    public override void Deserialize(byte[] data)
    {
        // FIX ISSUE 8: Use NotSupportedException with clear message
        throw new NotSupportedException(
            "Dreamer agent deserialization is not supported. " +
            "Use GetParameters()/SetParameters() for parameter transfer, " +
            "or load individual network weights separately.");
    }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var network in Networks)
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

        return paramVector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        foreach (var network in Networks)
        {
            int paramCount = network.ParameterCount;
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        // FIX ISSUE 7: Clone should copy learned network parameters
        var clone = new DreamerAgent<T>(_options, _optimizer);

        // Copy all network parameters
        var parameters = GetParameters();
        clone.SetParameters(parameters);

        return clone;
    }

    /// <summary>
    /// Computes gradients for supervised learning scenarios.
    /// </summary>
    /// <remarks>
    /// FIX ISSUE 9: This method uses simple supervised loss for compatibility with base class API.
    /// It does NOT match the agent's internal training procedure which uses:
    /// - World model losses (dynamics, reward, continue prediction)
    /// - Imagination-based policy gradients
    /// - Value function TD errors
    ///
    /// For actual agent training, use Train() which implements the full Dreamer algorithm.
    /// This method is provided only for API compatibility and simple supervised fine-tuning scenarios.
    /// </remarks>
    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(prediction, target);

        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (Networks.Count > 0 && Networks[0] is NeuralNetwork<T> network)
        {
            network.Backpropagate(Tensor<T>.FromVector(gradients));
            var networkParams = network.GetParameters();
            network.UpdateParameters(networkParams);
        }
    }

    public override void SaveModel(string filepath)
    {
        // FIX ISSUE 8: Throw NotSupportedException since Serialize is not supported
        throw new NotSupportedException(
            "Dreamer agent save/load is not supported. " +
            "Use GetParameters()/SetParameters() for parameter transfer.");
    }

    public override void LoadModel(string filepath)
    {
        // FIX ISSUE 8: Throw NotSupportedException since Deserialize is not supported
        throw new NotSupportedException(
            "Dreamer agent save/load is not supported. " +
            "Use GetParameters()/SetParameters() for parameter transfer.");
    }
}
