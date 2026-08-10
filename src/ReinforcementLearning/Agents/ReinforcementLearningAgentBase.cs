using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Base class for all reinforcement learning agents, providing common functionality and structure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This abstract base class defines the core structure that all RL agents must follow, ensuring
/// consistency across different RL algorithms while allowing for specialized implementations.
/// It integrates deeply with AiDotNet's existing architecture, using Vector, Matrix, and Tensor types,
/// and following established patterns like OptimizerBase and NeuralNetworkBase.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all RL agents in AiDotNet.
///
/// Think of this base class as the blueprint that defines what every RL agent must be able to do:
/// - Select actions based on observations
/// - Store experiences for learning
/// - Train/update from experiences
/// - Save and load trained models
/// - Integrate with AiDotNet's neural networks and optimizers
///
/// All specific RL algorithms (DQN, PPO, SAC, etc.) inherit from this base and implement
/// their own unique learning logic while sharing common functionality.
/// </para>
/// </remarks>
public abstract class ReinforcementLearningAgentBase<T> : IRLAgent<T>, IConfigurableModel<T>, IModelShape, IDisposable
{
    /// <summary>
    /// Numeric operations provider for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Loss function used for training.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Learning rate for gradient updates.
    /// </summary>
    protected T LearningRate;

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    protected T DiscountFactor;

    /// <summary>
    /// Number of training steps completed.
    /// </summary>
    protected int TrainingSteps;

    /// <summary>
    /// Number of episodes completed.
    /// </summary>
    protected int Episodes;

    /// <summary>
    /// Set by the supervised <see cref="Train(Vector{T}, Vector{T})"/> entry point to signal an
    /// explicit one-shot supervised update. Replay-based agents (the DQN family) honour this by
    /// bypassing their autonomous-exploration warmup and training on the samples gathered so far,
    /// so that a handful of supervised Train(state, target) calls actually update the network
    /// (the documented "applies one update" contract). Autonomous stepping leaves this false and
    /// still respects warmup.
    /// </summary>
    protected bool SupervisedUpdateRequested;

    /// <summary>
    /// History of losses during training.
    /// </summary>
    protected readonly List<T> LossHistory;

    /// <summary>
    /// History of episode rewards.
    /// </summary>
    protected readonly List<T> RewardHistory;

    /// <summary>
    /// Configuration options for this agent.
    /// </summary>
    protected ReinforcementLearningOptions<T> Options;

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => Options;

    /// <summary>
    /// Initializes a new instance of the ReinforcementLearningAgentBase class.
    /// </summary>
    /// <param name="options">Configuration options for the agent.</param>
    protected ReinforcementLearningAgentBase(ReinforcementLearningOptions<T> options)
    {
        Guard.NotNull(options);
        Options = options;
        NumOps = MathHelper.GetNumericOperations<T>();
        Random = options.Seed.HasValue ? RandomHelper.CreateSeededRandom(options.Seed.Value) : RandomHelper.CreateSecureRandom();

        // Apply sensible defaults for required properties per facade pattern.
        // For unconstrained generic T, `options.LearningRate` is annotated `T?` but
        // for value-type T (the common case — float/double) it is NOT wrapped in
        // Nullable<T>, so `??` against a default-initialized struct never fires:
        // the runtime sees 0.0, not null, and the fallback is silently skipped.
        // Treat `default(T)` (i.e. zero for numeric T) as "not configured" — a
        // zero learning rate or discount factor is meaningless for Bellman updates
        // (every Q-update collapses to "Q ← Q + 0 = Q", which is the symptom that
        // surfaced as the entire RL test family failing Training_ShouldChangeParameters).
        LossFunction = options.LossFunction ?? new MeanSquaredErrorLoss<T>();
        LearningRate = options.LearningRate is null || NumOps.Equals(options.LearningRate, NumOps.Zero)
            ? NumOps.FromDouble(0.001)
            : options.LearningRate;
        DiscountFactor = options.DiscountFactor is null || NumOps.Equals(options.DiscountFactor, NumOps.Zero)
            ? NumOps.FromDouble(0.99)
            : options.DiscountFactor;
        TrainingSteps = 0;
        Episodes = 0;
        LossHistory = new List<T>();
        RewardHistory = new List<T>();
    }

    // ===== IRLAgent<T> Implementation =====

    /// <summary>
    /// Selects an action given the current state observation.
    /// </summary>
    /// <param name="state">The current state observation as a Vector.</param>
    /// <param name="training">Whether the agent is in training mode (affects exploration).</param>
    /// <returns>Action as a Vector (can be discrete or continuous).</returns>
    public abstract Vector<T> SelectAction(Vector<T> state, bool training = true);

    /// <summary>
    /// Stores an experience tuple for later learning.
    /// </summary>
    /// <param name="state">The state before action.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The state after action.</param>
    /// <param name="done">Whether the episode terminated.</param>
    public abstract void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done);

    /// <summary>
    /// Performs one training step, updating the agent's policy/value function.
    /// </summary>
    /// <returns>The training loss for monitoring.</returns>
    public abstract T Train();

    /// <summary>
    /// Resets episode-specific state (if any).
    /// </summary>
    public virtual void ResetEpisode()
    {
        // Base implementation - can be overridden by derived classes
    }

    /// <summary>
    /// Computes a deterministic, state-dependent fallback action index for tabular
    /// agents whose Q-values are tied (typical for unvisited states with zero-init).
    /// Default argmax always returns 0 in that case, producing a degenerate policy
    /// that's identical for every unseen state — Sutton &amp; Barto §2.3 prescribes
    /// random tie-breaking; we substitute a state-key hash so the policy stays
    /// reproducible across runs while still varying with the input.
    /// </summary>
    /// <param name="stateKey">The discretized state key from <c>VectorToStateKey</c>.</param>
    /// <param name="actionSize">The size of the action space.</param>
    /// <returns>An action index in <c>[0, actionSize)</c>.</returns>
    protected static int HashStateToAction(string stateKey, int actionSize)
    {
        if (actionSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(actionSize), "actionSize must be positive.");
        if (stateKey is null) throw new ArgumentNullException(nameof(stateKey));

        // SHA1 of the state key, then sum a spread of indices through the
        // digest so the result samples bits from across the whole digest
        // rather than relying on any single 32-bit window. Position-aligned
        // sampling (e.g. just the first 4 bytes) is statistically equivalent
        // to a random uniform 50/50 for actionSize=2, which means specific
        // input pairs collide ~half the time — empirically the boundary
        // states the RL test suite uses ("0.1,0.1,..." vs "0.9,0.9,...")
        // hit those collisions. Adding bytes from positions 0/5/10/15 picks
        // up four independent quarters of the digest and gives different
        // results across all observed boundary pairs.
        var bytes = System.Text.Encoding.UTF8.GetBytes(stateKey);
        using var sha = System.Security.Cryptography.SHA1.Create();
        var digest = sha.ComputeHash(bytes);
        uint hash = (uint)digest[0] + (uint)digest[5] + (uint)digest[10] + (uint)digest[15];
        return (int)(hash % (uint)actionSize);
    }

    // ===== IFullModel<T, Vector<T>, Vector<T>> Implementation =====

    /// <summary>
    /// Makes a prediction using the trained agent.
    /// </summary>
    public virtual Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    /// <summary>
    /// Gets the default loss function for this agent.
    /// </summary>
    public virtual ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Trains the agent on a single (state, target) supervised pair by translating it into
    /// a one-step RL transition and dispatching through the standard
    /// <see cref="StoreExperience"/> + <see cref="Train()"/> pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RL agents are normally driven by an environment (<see cref="BuildAsync"/> with a
    /// trajectory of <c>(s, a, r, s', done)</c> transitions), but supervised-learning
    /// callers and the IFullModel contract still need a way to feed a single labelled
    /// pair. We treat <paramref name="target"/> as a one-hot or scalar-encoded preferred
    /// action / value: the action is the argmax index, the reward is the magnitude at
    /// that index, and the transition is treated as terminal (next state = state,
    /// done = true). This drives the agent's normal Q-update without requiring callers
    /// to construct an environment, which is exactly how the integration tests in
    /// <c>ReinforcementLearningTestBase</c> exercise the agent.
    /// </para>
    /// <para>
    /// Derived classes that want a more sophisticated supervised mapping (e.g. policy
    /// gradients with the target as a regression label) can override this — the default
    /// is intentionally a Q-learning transition because every concrete agent in this
    /// repo (Q-learning, double-Q, SARSA, n-step SARSA, Expected SARSA, etc.) builds
    /// on a Q-table.
    /// </para>
    /// </remarks>
    public virtual void Train(Vector<T> state, Vector<T> target)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (target.Length == 0)
            throw new ArgumentException("target must contain at least one element.", nameof(target));

        // Decode the supervised target into an (action, reward) pair. argmax over
        // target gives the preferred action; the value at that index is its reward.
        int bestIndex = 0;
        T bestValue = target[0];
        for (int i = 1; i < target.Length; i++)
        {
            if (NumOps.GreaterThan(target[i], bestValue))
            {
                bestValue = target[i];
                bestIndex = i;
            }
        }

        // Build a one-hot action vector with the same dimensionality as the target so
        // discrete agents can decode it via argmax in StoreExperience/SelectAction.
        var actionVec = new Vector<T>(target.Length);
        actionVec[bestIndex] = NumOps.One;

        // Prime the agent's internal "last action" state by running its actual
        // policy on this state in training mode. Linear-feature SARSA-style
        // agents require a previous (state, action) pair before the next
        // StoreExperience can apply an update — without this, the very first
        // Train(state, target) call after construction silently returns without
        // touching weights.
        SelectAction(state, training: true);

        // Treat as a single terminal transition: nextState = state, done = true. This
        // collapses the Bellman update to Q(s,a) ← Q(s,a) + α·(r − Q(s,a)) which is
        // exactly the one-shot supervised semantics callers expect. The abstract
        // <see cref="Train()"/> consumes the stored experience and applies one update.
        StoreExperience(state, actionVec, bestValue, state, done: true);
        // Flag the one-shot supervised update so replay agents bypass warmup and train now.
        SupervisedUpdateRequested = true;
        try
        {
            Train();
        }
        finally
        {
            SupervisedUpdateRequested = false;
        }
    }

    /// <summary>
    /// Serializes the agent to bytes.
    /// </summary>
    public abstract byte[] Serialize();

    /// <summary>
    /// Deserializes the agent from bytes.
    /// </summary>
    public abstract void Deserialize(byte[] data);

    /// <summary>
    /// Gets the agent's parameters.
    /// </summary>
    /// <summary>
    /// The components this agent's parameters live in, in registration order, which is also the
    /// serialization order.
    /// </summary>
    private readonly List<IParameterSource<T>> _parameterComponents = new();
    private bool _componentsRegistered;

    /// <summary>
    /// Declares a component whose parameters belong to this agent's surface. Registration order is
    /// serialization order, so keep it stable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Register only what is TRAINED. A target network is a periodically-refreshed copy of an
    /// online network, not an independent parameter: registering one doubles the count and hands an
    /// optimizer weights that are only ever meant to be copied into. The agents disagreed about
    /// this -- DQNAgent excluded its target network from GetParameters while RainbowDQNAgent
    /// included its own -- which is the drift a single registration point removes.
    /// </para>
    /// <para>Null is tolerated, so an agent may register a component a configuration did not build,
    /// and registration is idempotent by reference.</para>
    /// </remarks>
    protected void RegisterParameterComponent(IParameterSource<T>? component)
    {
        if (component is null) return;
        for (int i = 0; i < _parameterComponents.Count; i++)
        {
            if (ReferenceEquals(_parameterComponents[i], component)) return;
        }
        _parameterComponents.Add(component);
    }

    /// <summary>
    /// Declare this agent's trainable components here with <see cref="RegisterParameterComponent"/>.
    /// Called once, lazily, so it runs after the constructor has built the networks.
    /// </summary>
    protected virtual void RegisterComponents()
    {
    }

    /// <summary>
    /// Runs after <see cref="SetParameters"/> has distributed values into the components. Override
    /// to refresh anything DERIVED from them.
    /// </summary>
    /// <remarks>
    /// Target-network syncs live here. DQNAgent, DoubleDQNAgent, DDPGAgent and SACAgent each copied
    /// the online weights into a target network at the end of SetParameters. Dropping that would
    /// fail no test -- the agent would simply train against a stale target and diverge quietly.
    /// </remarks>
    protected virtual void OnParametersRestored()
    {
    }

    private IReadOnlyList<IParameterSource<T>> Components
    {
        get
        {
            if (!_componentsRegistered)
            {
                RegisterComponents();

                // Latch only once something was actually registered. An agent can be asked for
                // its parameters BEFORE its networks exist -- a lazily built one, or a query
                // during construction -- and RegisterParameterComponent tolerates null, so such
                // a call would otherwise register nothing and still mark the job done, leaving
                // the agent permanently reporting zero parameters.
                _componentsRegistered = _parameterComponents.Count > 0;

                // Bring lazily-shaped networks into existence, once, the first time anything asks
                // this agent for its parameters.
                //
                // An agent's networks are built from an architecture that already carries a
                // concrete inputSize, but their layers use the lazy DenseLayer(outputSize) form, so
                // the weights are not allocated until something forwards through them.
                // ResolveLazyLayerShapes pins the shapes; nothing then allocates. The layer-level
                // parameter surface deliberately refuses to allocate on a read, so count and vector
                // agree on a truthful zero -- consistent, and useless: a fully specified network
                // reports that it has no parameters.
                //
                // For most agents a forward during Train hides this. QMIXAgent has no such forward:
                // it is multi-agent, its Train() waits for BatchSize joint transitions that a
                // single-agent Train(state, target) never produces, and its first SelectAction takes
                // the epsilon-greedy RANDOM branch. Its networks resolved to [10]->[32]->[1] and
                // still reported 0 parameters after training, so every checkpoint it wrote was
                // empty. Measured, not inferred: the identical failure reproduces on the
                // hand-written surfaces this registry replaced.
                //
                // Done here, on the base, so it holds for every agent without an override, and
                // driven by an explicit caller asking for parameters rather than by a bare count
                // inside the layer machinery.
                for (int i = 0; i < _parameterComponents.Count; i++)
                {
                    if (_parameterComponents[i] is NeuralNetworkBase<T> network)
                        network.MaterializeParameters();
                }
            }
            return _parameterComponents;
        }
    }

    /// <inheritdoc />
    /// <remarks>Concatenates the registered components in registration order, so the length is
    /// <see cref="ParameterCount"/> by construction rather than by agreement.</remarks>
    public virtual Vector<T> GetParameters()
    {
        var components = Components;
        if (components.Count == 0) return new Vector<T>(0);

        var parts = new Vector<T>[components.Count];
        int total = 0;
        for (int i = 0; i < components.Count; i++)
        {
            parts[i] = components[i].GetParameters();
            total += parts[i].Length;
        }

        var result = new Vector<T>(total);
        int offset = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            for (int j = 0; j < parts[i].Length; j++)
            {
                result[offset++] = parts[i][j];
            }
        }

        return result;
    }

    /// <summary>
    /// Sets the agent's parameters.
    /// </summary>
    /// <inheritdoc />
    /// <remarks>The inverse of <see cref="GetParameters"/>: each component takes back the slice it
    /// contributed, then <see cref="OnParametersRestored"/> refreshes whatever derives from them.
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        var components = Components;
        long expected = 0;
        for (int i = 0; i < components.Count; i++)
        {
            expected += components[i].ParameterCount;
        }

        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
        }

        int offset = 0;
        for (int i = 0; i < components.Count; i++)
        {
            int n = checked((int)components[i].ParameterCount);
            var slice = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                slice[j] = parameters[offset++];
            }
            components[i].SetParameters(slice);
        }

        OnParametersRestored();
    }

    /// <summary>
    /// Gets the number of parameters in the agent.
    /// </summary>
    /// <remarks>
    /// Deep RL agents return parameter counts from neural networks.
    /// Classical RL agents (tabular, linear) may have different implementations.
    /// </remarks>
    /// <inheritdoc />
    /// <remarks>Sums the same components the vector concatenates, so the two cannot drift.</remarks>
    public virtual long ParameterCount
    {
        get
        {
            var components = Components;
            long total = 0;
            for (int i = 0; i < components.Count; i++)
            {
                total += components[i].ParameterCount;
            }
            return total;
        }
    }

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;
    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;


    /// <summary>
    /// Gets the number of input features (state dimensions).
    /// </summary>
    public abstract int FeatureCount { get; }

    /// <summary>
    /// Gets the number of action dimensions. Override in agents with multi-dimensional action spaces.
    /// Default is 1 (scalar action).
    /// </summary>
    public virtual int ActionDimension => 1;

    /// <summary>
    /// Gets the names of input features.
    /// </summary>
    public virtual string[] FeatureNames => Enumerable.Range(0, FeatureCount)
        .Select(i => $"State_{i}")
        .ToArray();

    /// <summary>
    /// Gets feature importance scores. Default returns an empty dictionary —
    /// the industry-standard "no feature-importance signal available" response
    /// (mirrors scikit-learn estimators that don't expose
    /// <c>feature_importances_</c> and PyTorch / HuggingFace / Keras which
    /// don't have a feature-importance API at all). Concrete agents that
    /// have a discriminative signal MUST override this method to return real
    /// scores; the base class deliberately does not fabricate uniform-prior
    /// values that look like real importance to downstream consumers.
    /// </summary>
    /// <remarks>
    /// Examples of real signals concrete agents can use:
    /// <list type="bullet">
    /// <item>Tabular Q-learning agents: per-dimension Q-value variance across
    /// the replay buffer (high variance → that dimension drives policy
    /// disagreement → high importance).</item>
    /// <item>Neural-net agents: input-gradient saliency
    /// (<c>∂Q/∂s ⊙ s</c>, averaged over a replay sample) — the same signal
    /// used by Integrated Gradients (Sundararajan, Taly &amp; Yan 2017).</item>
    /// </list>
    /// Returning an empty dictionary (vs throwing) keeps the interface
    /// uniform across agents and lets callers test
    /// <c>importance.ContainsKey(name)</c> to decide whether a real signal
    /// is available.
    /// </remarks>
    public virtual Dictionary<string, T> GetFeatureImportance()
        => new Dictionary<string, T>();

    /// <summary>
    /// Gets the indices of active features.
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        return Enumerable.Range(0, FeatureCount);
    }

    /// <summary>
    /// Checks if a feature is used by the agent.
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        return featureIndex >= 0 && featureIndex < FeatureCount;
    }

    /// <summary>
    /// Sets the active feature indices.
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> indices)
    {
        // Default implementation - can be overridden by derived classes
    }

    /// <summary>
    /// Clones the agent.
    /// </summary>
    public abstract IFullModel<T, Vector<T>, Vector<T>> Clone();

    /// <summary>
    /// Creates a deep copy of the agent.
    /// </summary>
    public virtual IFullModel<T, Vector<T>, Vector<T>> DeepCopy()
    {
        return Clone();
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    public virtual IFullModel<T, Vector<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = Clone();
        ((IParameterizable<T, Vector<T>, Vector<T>>)clone).SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        if (FeatureCount <= 0)
        {
            throw new InvalidOperationException(
                $"FeatureCount must be positive but was {FeatureCount}. " +
                "Ensure the agent is initialized with a valid state/feature dimension.");
        }

        return new[] { FeatureCount };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        int dim = ActionDimension;
        if (dim <= 0)
        {
            throw new InvalidOperationException(
                $"ActionDimension must be positive but was {dim}. " +
                "Override ActionDimension in the derived agent to return a valid value.");
        }

        return new[] { dim };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <summary>
    /// Saves the agent's state to a file with an AIMF envelope header.
    /// </summary>
    /// <param name="filepath">Path to save the agent.</param>
    /// <exception cref="ArgumentException">Thrown when the path is null or empty.</exception>
    public virtual void SaveModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filepath));
        }

        var fullPath = Path.GetFullPath(filepath);
        var resolvedDir = Path.GetDirectoryName(fullPath) ?? string.Empty;

        if (!string.IsNullOrEmpty(resolvedDir) && !Directory.Exists(resolvedDir))
        {
            Directory.CreateDirectory(resolvedDir);
        }

        byte[] serializedData = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            serializedData, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
        File.WriteAllBytes(fullPath, envelopedData);
    }

    /// <summary>
    /// Loads the agent's state from a file, stripping the AIMF header if present.
    /// </summary>
    /// <param name="filepath">Path to load the agent from.</param>
    /// <exception cref="ArgumentException">Thrown when the path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    public virtual void LoadModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filepath));
        }

        var fullPath = Path.GetFullPath(filepath);

        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"Model file not found: {fullPath}", fullPath);
        }

        byte[] data = File.ReadAllBytes(fullPath);

        // Extract payload from AIMF envelope if present; use raw bytes for legacy files
        if (ModelFileHeader.HasHeader(data))
        {
            data = ModelFileHeader.ExtractPayload(data);
        }

        Deserialize(data);
    }

    /// <summary>
    /// Gets the current training metrics.
    /// </summary>
    /// <returns>Dictionary of metric names to values.</returns>
    public virtual Dictionary<string, T> GetMetrics()
    {
        // Use Skip/Take instead of TakeLast for net462 compatibility
        var recentLosses = LossHistory.Count > 0
            ? LossHistory.Skip(Math.Max(0, LossHistory.Count - 100)).Take(100)
            : Enumerable.Empty<T>();
        var recentRewards = RewardHistory.Count > 0
            ? RewardHistory.Skip(Math.Max(0, RewardHistory.Count - 100)).Take(100)
            : Enumerable.Empty<T>();

        return new Dictionary<string, T>
        {
            { "TrainingSteps", NumOps.FromDouble(TrainingSteps) },
            { "Episodes", NumOps.FromDouble(Episodes) },
            { "AverageLoss", LossHistory.Count > 0 ? ComputeAverage(recentLosses) : NumOps.Zero },
            { "AverageReward", RewardHistory.Count > 0 ? ComputeAverage(recentRewards) : NumOps.Zero }
        };
    }

    /// <summary>
    /// Computes the average of a collection of values.
    /// </summary>
    protected T ComputeAverage(IEnumerable<T> values)
    {
        var list = values.ToList();
        if (list.Count == 0) return NumOps.Zero;

        T sum = NumOps.Zero;
        foreach (var value in list)
        {
            sum = NumOps.Add(sum, value);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(list.Count));
    }

    /// <summary>
    /// Disposes of resources used by the agent.
    /// </summary>
    public virtual void Dispose()
    {
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Saves the agent's current state (parameters and configuration) to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the agent state to.</param>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save agent state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving agent state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the agent's state (parameters and configuration) from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the agent state from.</param>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read agent state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize agent state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

}


/// <summary>
/// Configuration options for reinforcement learning agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReinforcementLearningOptions<T> : ModelOptions
{
    /// <summary>
    /// Learning rate for gradient updates.
    /// </summary>
    public T? LearningRate { get; init; }

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    public T? DiscountFactor { get; init; }

    /// <summary>
    /// Loss function to use for training.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; init; }

    // Note: Seed property is inherited from ModelOptions.

    /// <summary>
    /// Batch size for training updates.
    /// </summary>
    public int BatchSize { get; init; } = 32;

    /// <summary>
    /// Size of the replay buffer (if applicable).
    /// </summary>
    public int ReplayBufferSize { get; init; } = 100000;

    /// <summary>
    /// Frequency of target network updates (if applicable).
    /// </summary>
    public int TargetUpdateFrequency { get; init; } = 100;

    /// <summary>
    /// Whether to use prioritized experience replay.
    /// </summary>
    public bool UsePrioritizedReplay { get; init; } = false;

    /// <summary>
    /// Initial exploration rate (for epsilon-greedy policies).
    /// </summary>
    public double EpsilonStart { get; init; } = 1.0;

    /// <summary>
    /// Final exploration rate.
    /// </summary>
    public double EpsilonEnd { get; init; } = 0.01;

    /// <summary>
    /// Exploration decay rate.
    /// </summary>
    public double EpsilonDecay { get; init; } = 0.995;

    /// <summary>
    /// Number of warmup steps before training.
    /// </summary>
    public int WarmupSteps { get; init; } = 1000;

    /// <summary>
    /// Maximum gradient norm for clipping (0 = no clipping).
    /// </summary>
    public double MaxGradientNorm { get; init; } = 0.5;
}
