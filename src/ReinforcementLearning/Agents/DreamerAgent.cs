using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Validation;

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
/// <example>
/// <code>
/// // Create a Dreamer agent that learns a world model for planning
/// var options = new DreamerOptions&lt;double&gt; { StateSize = 64, ActionSize = 4, ImagineHorizon = 15 };
/// var agent = new DreamerAgent&lt;double&gt;(options);
///
/// // Select an action by imagining future trajectories
/// var state = new Vector&lt;double&gt;(new double[64]);
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Dream to Control: Learning Behaviors by Latent Imagination",
    "https://arxiv.org/abs/1912.01603",
    Year = 2020,
    Authors = "Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M.")]
public class DreamerAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DreamerOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    // World model components
    private INeuralNetwork<T> _representationNetwork;  // Observation -> latent state
    private INeuralNetwork<T> _dynamicsNetwork;  // (latent state, action) -> next latent state
    private INeuralNetwork<T> _rewardNetwork;  // latent state -> reward
    private INeuralNetwork<T> _continueNetwork;  // latent state -> continue probability

    // Actor-critic for policy learning
    private INeuralNetwork<T> _actorNetwork;
    private INeuralNetwork<T> _valueNetwork;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _updateCount;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DreamerAgent()
        : this(new DreamerOptions<T> { ActionSize = 2 })
    {
    }

    public DreamerAgent(DreamerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;

        // FIX ISSUE 6: Use learning rate from options consistently
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = _options.LearningRate is not null ? NumOps.ToDouble(_options.LearningRate) : 0.001,
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
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize, _options.Seed);
    }

    private NeuralNetwork<T> CreateEncoderNetwork(int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize);
        var network = new NeuralNetwork<T>(architecture, lossFunction: new MeanSquaredErrorLoss<T>());

        for (int i = 0; i < 2; i++)
        {
            network.AddLayer(LayerType.Dense, _options.HiddenSize, ActivationFunction.ReLU);
        }

        network.AddLayer(LayerType.Dense, outputSize, ActivationFunction.Linear);

        return network;
    }

    private NeuralNetwork<T> CreateActorNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.LatentSize,
            outputSize: _options.ActionSize);
        var network = new NeuralNetwork<T>(architecture, lossFunction: new MeanSquaredErrorLoss<T>());

        for (int i = 0; i < 2; i++)
        {
            network.AddLayer(LayerType.Dense, _options.HiddenSize, ActivationFunction.ReLU);
        }

        network.AddLayer(LayerType.Dense, _options.ActionSize, ActivationFunction.Tanh);

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);
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
        // Validate transition shapes at this public ingestion boundary so a malformed experience
        // can't enter the replay buffer and later cause indexing / network-shape failures deep in
        // Train() (building dynIn/repIn/rewIn/contIn).
        if (observation.Length != _options.ObservationSize)
            throw new ArgumentException($"Observation length must be {_options.ObservationSize}, got {observation.Length}.", nameof(observation));
        if (nextObservation.Length != _options.ObservationSize)
            throw new ArgumentException($"Next observation length must be {_options.ObservationSize}, got {nextObservation.Length}.", nameof(nextObservation));
        if (action.Length != _options.ActionSize)
            throw new ArgumentException($"Action length must be {_options.ActionSize}, got {action.Length}.", nameof(action));

        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(observation, action, reward, nextObservation, done));
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;

        int obsSize = _options.ObservationSize;
        int latentSize = _options.LatentSize;
        int actionDim = _options.ActionSize;
        T gamma = _options.DiscountFactor is not null ? _options.DiscountFactor : NumOps.FromDouble(0.99);

        // ===== World-model learning (Hafner et al. 2020) =====
        // Encode each observation to a latent, and train the predictive heads:
        //   dynamics(z_t, a_t) -> z_{t+1};  reward(z_t) -> r_t;  continue(z_t) -> 1-done;
        // and keep the encoder consistent with the dynamics (representation(o_{t+1}) -> z_pred).
        var dynIn = new Tensor<T>([n, latentSize + actionDim]);
        var dynTgt = new Tensor<T>([n, latentSize]);
        var repIn = new Tensor<T>([n, obsSize]);
        var repTgt = new Tensor<T>([n, latentSize]);
        var rewIn = new Tensor<T>([n, latentSize]);
        var rewTgt = new Tensor<T>([n, 1]);
        var contIn = new Tensor<T>([n, latentSize]);
        var contTgt = new Tensor<T>([n, 1]);
        var latents = new Vector<T>[n];

        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            var z = _representationNetwork.Predict(Tensor<T>.FromVector(exp.State)).ToVector();
            var zNext = _representationNetwork.Predict(Tensor<T>.FromVector(exp.NextState)).ToVector();
            var dynInput = ConcatenateVectors(z, exp.Action);
            var zPred = _dynamicsNetwork.Predict(Tensor<T>.FromVector(dynInput)).ToVector();
            latents[i] = z;

            for (int j = 0; j < latentSize + actionDim; j++) dynIn[i, j] = dynInput[j];
            for (int j = 0; j < latentSize; j++) dynTgt[i, j] = zNext[j];       // dynamics -> next latent
            for (int j = 0; j < obsSize; j++) repIn[i, j] = exp.NextState[j];
            for (int j = 0; j < latentSize; j++) repTgt[i, j] = zPred[j];        // encoder <-> dynamics consistency
            for (int j = 0; j < latentSize; j++) rewIn[i, j] = z[j];
            rewTgt[i, 0] = exp.Reward;                                           // reward(z) -> r
            for (int j = 0; j < latentSize; j++) contIn[i, j] = z[j];
            contTgt[i, 0] = exp.Done ? NumOps.Zero : NumOps.One;                 // continue(z) -> 1-done
        }
        _dynamicsNetwork.Train(dynIn, dynTgt);
        _representationNetwork.Train(repIn, repTgt);
        _rewardNetwork.Train(rewIn, rewTgt);
        _continueNetwork.Train(contIn, contTgt);
        T worldModelLoss = NumOps.Add(
            NumOps.Add(_dynamicsNetwork.GetLastLoss(), _representationNetwork.GetLastLoss()),
            NumOps.Add(_rewardNetwork.GetLastLoss(), _continueNetwork.GetLastLoss()));

        // ===== Behaviour learning in imagination =====
        // Value regresses toward the imagined discounted return; the actor is improved toward the
        // action that increases the one-step imagined value q(z,a) = gamma * V(dynamics(z,a)) via the
        // deterministic policy gradient (finite-difference ∇a q).
        var valIn = new Tensor<T>([n, latentSize]);
        var valTgt = new Tensor<T>([n, 1]);
        var actIn = new Tensor<T>([n, latentSize]);
        var actTgt = new Tensor<T>([n, actionDim]);
        // Named constants for the behaviour-learning hyperparameters (no magic literals). `step` is
        // the deterministic-policy-gradient ascent step on the imagined value; `eps`/`twoEps` are the
        // central finite-difference interval used to estimate ∇a q(z,a).
        const double behaviorUpdateStep = 0.05;
        const double finiteDifferenceEpsilon = 1e-3;
        T step = NumOps.FromDouble(behaviorUpdateStep);
        T eps = NumOps.FromDouble(finiteDifferenceEpsilon);
        T twoEps = NumOps.FromDouble(2 * finiteDifferenceEpsilon);
        for (int i = 0; i < n; i++)
        {
            var z = latents[i];
            T imaginedReturn = ImagineTrajectory(z);
            for (int j = 0; j < latentSize; j++) valIn[i, j] = z[j];
            valTgt[i, 0] = imaginedReturn;

            var a = _actorNetwork.Predict(Tensor<T>.FromVector(z)).ToVector();
            var grad = new Vector<T>(actionDim);
            for (int k = 0; k < actionDim; k++)
            {
                var aPlus = a.Clone();
                var aMinus = a.Clone();
                aPlus[k] = NumOps.Add(a[k], eps);
                aMinus[k] = NumOps.Subtract(a[k], eps);
                var zPlus = _dynamicsNetwork.Predict(Tensor<T>.FromVector(ConcatenateVectors(z, aPlus))).ToVector();
                var zMinus = _dynamicsNetwork.Predict(Tensor<T>.FromVector(ConcatenateVectors(z, aMinus))).ToVector();
                T vPlus = _valueNetwork.Predict(Tensor<T>.FromVector(zPlus)).ToVector()[0];
                T vMinus = _valueNetwork.Predict(Tensor<T>.FromVector(zMinus)).ToVector()[0];
                grad[k] = NumOps.Divide(NumOps.Multiply(gamma, NumOps.Subtract(vPlus, vMinus)), twoEps);
            }
            for (int j = 0; j < latentSize; j++) actIn[i, j] = z[j];
            for (int k = 0; k < actionDim; k++)
                actTgt[i, k] = MathHelper.Clamp<T>(
                    NumOps.Add(a[k], NumOps.Multiply(step, grad[k])),
                    NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
        }
        _valueNetwork.Train(valIn, valTgt);
        _actorNetwork.Train(actIn, actTgt);
        T policyLoss = NumOps.Add(_valueNetwork.GetLastLoss(), _actorNetwork.GetLastLoss());

        _updateCount++;

        return NumOps.Add(worldModelLoss, policyLoss);
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
            var gamma = _options.DiscountFactor is not null ? NumOps.ToDouble(_options.DiscountFactor) : 0.99;
            var discountedReward = NumOps.Multiply(reward, NumOps.FromDouble(Math.Pow(gamma, step)));
            imaginedReturn = NumOps.Add(imaginedReturn, discountedReward);

            // Predict next latent state
            var dynamicsInput = ConcatenateVectors(latentState, action);
            latentState = _dynamicsNetwork.Predict(Tensor<T>.FromVector(dynamicsInput)).ToVector();

            // Check if episode continues
            var continueProb = _continueNetwork.Predict(Tensor<T>.FromVector(latentState)).ToVector()[0];
            if (NumOps.LessThan(continueProb, NumOps.FromDouble(0.5)))
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
            int paramCount = checked((int)network.ParameterCount);
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
        throw new NotSupportedException(
            "Dreamer agent requires per-network gradient distribution for six networks " +
            "(VAE encoder/decoder, RNN world model, reward/continue/value predictors). " +
            "The current signature cannot distribute gradients appropriately. " +
            "Use the internal Train() method for training, which handles multi-network updates correctly.");
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
