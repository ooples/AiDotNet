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
public partial class DreamerAgent<T> : DeepReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>Every network this agent owns, in the order Networks yields them, which is the
    /// order the hand-written concatenation used.</remarks>
    protected override void RegisterComponents()
    {
        foreach (var network in Networks) RegisterParameterComponent(network);
    }
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
    /// <remarks>
    /// The zero-argument default builds a small, self-consistent continuous-control toy
    /// agent: a 4-dimensional observation and a 4-dimensional continuous action. Four is the
    /// canonical low-dimensional control state (e.g. CartPole's [cart position, cart velocity,
    /// pole angle, pole angular velocity]) and keeps <see cref="ObservationSize"/> and
    /// <see cref="DreamerOptions{T}.ActionSize"/> aligned so that a plain
    /// <c>(state, target)</c> transition of that width is accepted by
    /// <see cref="StoreExperience"/> without any environment-specific configuration.
    /// Real tasks should pass a fully-specified <see cref="DreamerOptions{T}"/>.
    /// </remarks>
    public DreamerAgent()
        : this(new DreamerOptions<T> { ObservationSize = 4, ActionSize = 4 })
    {
    }

    public DreamerAgent(DreamerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;

        // Use the rate the BASE class already resolved, not the raw option. `options.LearningRate`
        // is declared `T?`, but for a value-type T — float and double, i.e. every real use — that
        // is not Nullable<T>, so an unconfigured option reads as default(T) == 0 and `is not null`
        // is TRUE. This ctor therefore handed Adam InitialLearningRate = 0 whenever the caller did
        // not set one, which is every default-constructed DreamerAgent.
        //
        // ReinforcementLearningAgentBase already solves this and says so at length: it treats
        // default(T) as "not configured" and substitutes 0.001, precisely because "a zero learning
        // rate ... is meaningless for Bellman updates (every Q-update collapses to Q <- Q + 0 = Q,
        // which is the symptom that surfaced as the entire RL test family failing
        // Training_ShouldChangeParameters)". The base runs first, so LearningRate is resolved by
        // the time this body executes; re-deriving it here reintroduced the bug the base fixed.
        //
        // The zero was silent until an optimizer validated its rate at construction, at which point
        // it became "Base learning rate must be positive" and took DreamerAgent's whole suite plus
        // AllDefaultConstructableModels_ShouldConstructWithoutException with it. Silent was worse.
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = NumOps.ToDouble(LearningRate),
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
        // Use the rate the BASE class already resolved, exactly as this type's constructor does for
        // LearningRate and for exactly the same reason: `DiscountFactor` is declared `T?`, but for a
        // value-type T -- float and double, i.e. every real use -- that is NOT Nullable<T>, so an
        // unconfigured option reads as default(T) == 0 while `is not null` is still TRUE. This line
        // therefore evaluated to gamma = 0 for every default-constructed DreamerAgent, and the 0.99
        // fallback never ran.
        //
        // Zero gamma made the entire behaviour-learning objective identically zero: the actor loss is
        // gamma * V(dynamics(z,a)), so its gradient was exactly zero for the actor, the dynamics head
        // and the value head alike -- measured as 0 of 12 reachable tensors in each, and 203408
        // published gradients with max |g| = 0. The previous finite-difference formulation was dead
        // for the same reason (grad = gamma * (vPlus - vMinus) / 2eps), which is why replacing it with
        // a taped gradient changed nothing until this line was fixed.
        //
        // The base resolves this properly, treating default(T) as "not configured": see
        // ReinforcementLearningAgentBase, which checks `is null || == zero` before falling back.
        T gamma = DiscountFactor;

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
        // Value regresses toward the imagined discounted return; the actor ascends the imagined value
        // q(z,a) = gamma * V(dynamics(z,a)) by the deterministic policy gradient, taken ON THE TAPE.
        //
        // This previously estimated dq/da by CENTRAL FINITE DIFFERENCES and then fitted the actor to a
        // nudged copy of its own output. That degenerates silently: when the value head is locally flat
        // -- which it is through most of early training, and effectively always on a short run -- vPlus
        // and vMinus are equal, the estimated gradient is exactly zero, the regression target equals the
        // actor's current output, and the supervised step applies NO update whatsoever. The actor's
        // weight-bearing layers then receive nothing while normalization statistics keep moving, so the
        // agent still looks alive to any parameter-movement check while its policy never improves at
        // all. A per-component reachability check is what surfaced it: 6 of 36 components, every one of
        // them in _actorNetwork, received no update across 800 steps.
        //
        // Differentiating through the dynamics and value heads instead yields the exact gradient in a
        // single backward pass, with no epsilon to tune and no dependence on the value surface being
        // locally non-flat. Both heads are forwarded with ForwardForTraining rather than Predict:
        // Predict runs inside a NoGradScope and would hand back a detached constant, reintroducing the
        // very failure this replaces. TrainWithCustomLoss collects only the actor's tensors, so the
        // world model supplies dq/da here without being updated by the actor's step.
        var valIn = new Tensor<T>([n, latentSize]);
        var valTgt = new Tensor<T>([n, 1]);
        var actIn = new Tensor<T>([n, latentSize]);
        for (int i = 0; i < n; i++)
        {
            var z = latents[i];
            T imaginedReturn = ImagineTrajectory(z);
            for (int j = 0; j < latentSize; j++)
            {
                valIn[i, j] = z[j];
                actIn[i, j] = z[j];
            }
            valTgt[i, 0] = imaginedReturn;
        }
        _valueNetwork.Train(valIn, valTgt);

        var tapedActor = (NeuralNetworkBase<T>)_actorNetwork;
        var tapedDynamics = (NeuralNetworkBase<T>)_dynamicsNetwork;
        var tapedValue = (NeuralNetworkBase<T>)_valueNetwork;
        // Ascending gamma * V means minimising its negation; fold the sign into the scalar.
        T negatedGamma = NumOps.Multiply(gamma, NumOps.FromDouble(-1.0));
        T actorLoss = tapedActor.TrainWithCustomLoss(actIn, actorOutput =>
        {
            // [z | a] for the dynamics head, built with an engine op. Filling a fresh tensor element by
            // element would detach the action and strand the actor with no gradient path once more.
            var latentAction = Engine.TensorConcatenate([actIn, actorOutput], axis: 1);
            var imaginedNext = tapedDynamics.ForwardForTraining(latentAction);
            var imaginedValue = tapedValue.ForwardForTraining(imaginedNext);
            var flatValue = Engine.ReduceSum(imaginedValue, new[] { 1 }, keepDims: false);
            var objective = Engine.TensorMultiplyScalar(flatValue, negatedGamma);
            return Engine.ReduceMean(objective, new[] { 0 }, keepDims: false);
        });
        T policyLoss = NumOps.Add(_valueNetwork.GetLastLoss(), actorLoss);

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
            // Base-resolved, not re-derived from the raw option: `is not null` is always true for a
            // value-type T, so this read zero and every imagined return collapsed to reward * 0^step
            // -- degenerate for every step past the first, which corrupted the value targets too.
            var gamma = NumOps.ToDouble(DiscountFactor);
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
        var metadata = new ModelMetadata<T>
        {
            Name = "Dreamer",
            Description = "Dreamer model-based RL agent: learns a latent world model (representation, " +
                "dynamics, reward and continue heads) and an actor-critic trained in latent imagination " +
                "(Hafner et al. 2020, 'Dream to Control').",
            FeatureCount = _options.ObservationSize,
            Complexity = ParameterCount,
        };

        metadata.SetProperty("ObservationSize", _options.ObservationSize);
        metadata.SetProperty("ActionSize", _options.ActionSize);
        metadata.SetProperty("LatentSize", _options.LatentSize);
        metadata.SetProperty("HiddenSize", _options.HiddenSize);
        metadata.SetProperty("ImaginationHorizon", _options.ImaginationHorizon);
        metadata.SetProperty("UpdateCount", _updateCount);

        return metadata;
    }

    public override int FeatureCount => _options.ObservationSize;

}
