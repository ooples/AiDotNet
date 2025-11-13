using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;
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
    private INeuralNetwork<T> _representationNetwork;  // Observation -> latent state
    private INeuralNetwork<T> _dynamicsNetwork;  // (latent state, action) -> next latent state
    private INeuralNetwork<T> _rewardNetwork;  // latent state -> reward
    private INeuralNetwork<T> _continueNetwork;  // latent state -> continue probability

    // Actor-critic for policy learning
    private INeuralNetwork<T> _actorNetwork;
    private INeuralNetwork<T> _valueNetwork;

    private ReplayBuffer<T> _replayBuffer;
    private int _updateCount;

    public DreamerAgent(DreamerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            LearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _updateCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
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
    }

    private NeuralNetwork<T> CreateEncoderNetwork(int inputSize, int outputSize)
    {
        var network = new NeuralNetwork<T>();
        int previousSize = inputSize;

        for (int i = 0; i < 2; i++)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, _options.HiddenSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = _options.HiddenSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, outputSize));

        return network;
    }

    private NeuralNetwork<T> CreateActorNetwork()
    {
        var network = new NeuralNetwork<T>();
        int previousSize = _options.LatentSize;

        for (int i = 0; i < 2; i++)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, _options.HiddenSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = _options.HiddenSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, _options.ActionSize));
        network.AddLayer(new ActivationLayer<T>(new Tanh<T>()));

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new ReplayBuffer<T>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to latent state
        var latentState = _representationNetwork.Forward(observation);

        // Select action from policy
        var action = _actorNetwork.Forward(latentState);

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
        _replayBuffer.Add(observation, action, reward, nextObservation, done);
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

    private T TrainWorldModel(List<(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)> batch)
    {
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Encode observations to latent states
            var latentState = _representationNetwork.Forward(experience.observation);
            var nextLatentState = _representationNetwork.Forward(experience.nextObservation);

            // Predict next latent from dynamics model
            var dynamicsInput = ConcatenateVectors(latentState, experience.action);
            var predictedNextLatent = _dynamicsNetwork.Forward(dynamicsInput);

            // Dynamics loss: predict next latent state
            T dynamicsLoss = NumOps.Zero;
            for (int i = 0; i < predictedNextLatent.Length; i++)
            {
                var diff = NumOps.Subtract(nextLatentState[i], predictedNextLatent[i]);
                dynamicsLoss = NumOps.Add(dynamicsLoss, NumOps.Multiply(diff, diff));
            }

            // Reward prediction loss
            var predictedReward = _rewardNetwork.Forward(latentState)[0];
            var rewardDiff = NumOps.Subtract(experience.reward, predictedReward);
            var rewardLoss = NumOps.Multiply(rewardDiff, rewardDiff);

            // Continue prediction loss (done = 0, continue = 1)
            var continueTarget = experience.done ? NumOps.Zero : NumOps.One;
            var predictedContinue = _continueNetwork.Forward(latentState)[0];
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

            _dynamicsNetwork.Backward(gradient);
            _dynamicsNetwork.UpdateWeights(_options.LearningRate);

            var rewardGradient = new Vector<T>(1);
            rewardGradient[0] = rewardDiff;
            _rewardNetwork.Backward(rewardGradient);
            _rewardNetwork.UpdateWeights(_options.LearningRate);

            var continueGradient = new Vector<T>(1);
            continueGradient[0] = continueDiff;
            _continueNetwork.Backward(continueGradient);
            _continueNetwork.UpdateWeights(_options.LearningRate);
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
            var latentState = _representationNetwork.Forward(experience.observation);

            // Imagine future trajectory
            var imaginedReturns = ImagineTrajectory(latentState);

            // Update value network
            var predictedValue = _valueNetwork.Forward(latentState)[0];
            var valueDiff = NumOps.Subtract(imaginedReturns, predictedValue);
            var valueLoss = NumOps.Multiply(valueDiff, valueDiff);

            var valueGradient = new Vector<T>(1);
            valueGradient[0] = valueDiff;
            _valueNetwork.Backward(valueGradient);
            _valueNetwork.UpdateWeights(_options.LearningRate);

            // Update actor to maximize value
            var action = _actorNetwork.Forward(latentState);
            var actorGradient = new Vector<T>(action.Length);
            for (int i = 0; i < actorGradient.Length; i++)
            {
                actorGradient[i] = NumOps.Divide(valueDiff, NumOps.FromDouble(action.Length));
            }

            _actorNetwork.Backward(actorGradient);
            _actorNetwork.UpdateWeights(_options.LearningRate);

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
            var action = _actorNetwork.Forward(latentState);

            // Predict reward
            var reward = _rewardNetwork.Forward(latentState)[0];
            imaginedReturn = NumOps.Add(imaginedReturn, reward);

            // Predict next latent state
            var dynamicsInput = ConcatenateVectors(latentState, action);
            latentState = _dynamicsNetwork.Forward(dynamicsInput);

            // Check if episode continues
            var continueProb = _continueNetwork.Forward(latentState)[0];
            if (NumOps.Compare(continueProb, NumOps.FromDouble(0.5)) < 0)
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
