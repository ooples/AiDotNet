using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Validation;
using AiDotNet.LossFunctions;
using System.Runtime.CompilerServices;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Financial Advantage Actor-Critic (A2C) agent for fast trading policy learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The A2C (Advantage Actor-Critic) trading agent uses two
/// neural networks working together: an "actor" that decides what trades to make, and a
/// "critic" that evaluates how good those decisions are. The advantage of A2C is that it
/// learns quickly because the critic provides immediate feedback to the actor after each
/// trade, rather than waiting for the end result. It is well-suited for fast-paced trading
/// environments where quick adaptation is important.</para>
/// <para>
/// This is a one-step, on-policy categorical actor-critic. Collect actions with
/// <c>SelectAction(state, training: true)</c> and store them before changing the actor. Each update
/// consumes the current rollout once; it never replays transitions from an older actor.
/// Agent parameter/gradient/checkpoint updates discard pending behavior. Ordinary writes to stable
/// live actor tensors are also detected by storage identity and mutation version.
/// </para>
/// <para>
/// Custom layers with opaque/detached parameter storage, retained raw writable spans and mutations
/// under tensor inference mode must use the agent's explicit update boundary. Do not mutate a policy
/// concurrently with collection. Manually constructed or copied one-hot actions carry no verifiable
/// selection provenance; the caller must supply actions sampled from the current actor. Known greedy,
/// changed or stale actions returned by this agent are rejected. Pending behavior is runtime-only
/// and is not restored from checkpoints.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define actor and critic architectures for A2C trading (30 state features, 3 actions)
/// var actorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 3);
/// var criticArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 1);
///
/// // Create A2C agent for fast-adapting trading policy
/// var options = new TradingAgentOptions&lt;double&gt;();
/// var model = new FinancialA2CAgent&lt;double&gt;(actorArch, criticArch, options);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("Asynchronous Methods for Deep Reinforcement Learning", "https://arxiv.org/abs/1602.01783")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public partial class FinancialA2CAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    #region Fields

    private readonly TradingAgentOptions<T> _options;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic;
    private readonly PolicyRuntimeState _policyRuntime = new();
    private bool _initialWarmupComplete;
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialA2CAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the value (critic).</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, FinancialA2CAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialA2CAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        Guard.NotNull(actorArchitecture);
        Guard.NotNull(criticArchitecture);
        _options = options;
        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;

        EnsureDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize);
        EnsureDefaultLayers(criticArchitecture, options.StateSize, 1);

        _actor = new NeuralNetwork<T>(actorArchitecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic = new NeuralNetwork<T>(criticArchitecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The actor emits one LOGIT per discrete action (its output layer is linear, so the values are
    /// unbounded and do not sum to one). The policy is the categorical distribution
    /// <c>pi(a|s) = softmax(logits)[a]</c>: in training mode an action is sampled from it (using the agent's
    /// seeded random stream), otherwise the most probable action is returned. The same distribution is the
    /// one <see cref="Train()"/> differentiates, so exploration and learning agree.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The actor scores every trade option; softmax turns the scores into
    /// probabilities. While training the agent rolls a weighted die over those probabilities (so it keeps
    /// trying every option in proportion to how good it currently thinks it is); when trading for real it
    /// picks the highest-probability option.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var logits = _actor.Predict(Tensor<T>.FromVector(state)).ToVector();
        SynchronizePolicyStorage();
        var probabilities = SoftmaxProbabilities(logits);

        int actionIndex = training ? SampleCategorical(probabilities) : ArgMaxIndex(probabilities);
        var action = new Vector<T>(TradingOptions.ActionSize);
        action[actionIndex] = NumOps.One;
        _policyRuntime.Selections.Add(action, new SelectionStamp(_policyRuntime.Epoch, training, actionIndex));
        return action;
    }

    /// <summary>
    /// Numerically stable softmax of the actor logits (max-subtracted before exponentiation, computed in
    /// double precision so float agents do not overflow or lose the tail probabilities).
    /// </summary>
    private static double[] SoftmaxProbabilities(Vector<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var probabilities = new double[logits.Length];
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = numOps.ToDouble(logits[i]);
            if (probabilities[i] > max)
            {
                max = probabilities[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = Math.Exp(probabilities[i] - max);
            sum += probabilities[i];
        }

        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    /// <summary>
    /// Samples an action index from a categorical distribution using the agent's seeded random stream.
    /// </summary>
    private int SampleCategorical(double[] probabilities)
    {
        double r = Random.NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (r < cumulative) return i;
        }

        // Only reachable through floating-point round-off in the cumulative sum (or non-finite logits).
        return probabilities.Length - 1;
    }

    private static int ArgMaxIndex(double[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static int ArgMaxIndex(Vector<T> values)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (numOps.GreaterThan(values[i], values[best]))
            {
                best = i;
            }
        }

        return best;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Consumes all pending current-policy transitions once, fits the critic to the one-step TD target
    /// <c>r + gamma * V(s')</c>, and takes one advantage-weighted policy-gradient step on the actor's softmax
    /// policy (plus an <see cref="TradingAgentOptions{T}.EntropyCoefficient"/> entropy bonus). Returns the
    /// policy loss plus <see cref="TradingAgentOptions{T}.ValueCoefficient"/> times the critic loss.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, Train performs a training step. This updates the FinancialA2CAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        SynchronizePolicyStorage();
        int pendingCount = _policyRuntime.Pending.Count;
        if (pendingCount == 0 || !SupervisedUpdateRequested && pendingCount < TradingOptions.BatchSize)
            return NumOps.Zero;
        if (!_initialWarmupComplete && IsInWarmup(pendingCount)) return NumOps.Zero;

        // No random sampling, replacement or old-policy leftovers. Consume before the first
        // forward/update: a partially failed critic/actor update must never retry this behavior.
        var batch = _policyRuntime.Pending.ToArray();
        _initialWarmupComplete = true;
        InvalidatePolicy();
        int n = batch.Length;

        // Batched advantage-actor-critic update: one batched forward/backward for the critic and
        // the actor instead of one autograd tape per experience (the per-sample loop dominated RL
        // training time — see profiling). Standard mini-batch update.
        int stateDim = batch[0].State.Length;
        var gamma = NumOps.FromDouble(Convert.ToDouble(TradingOptions.DiscountFactor));

        var statesData = new T[n * stateDim];
        var nextStatesData = new T[n * stateDim];
        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            for (int j = 0; j < stateDim; j++)
            {
                statesData[i * stateDim + j] = exp.State[j];
                nextStatesData[i * stateDim + j] = exp.NextState[j];
            }
        }

        var states = new Tensor<T>([n, stateDim], new Vector<T>(statesData));
        var nextStates = new Tensor<T>([n, stateDim], new Vector<T>(nextStatesData));

        // One-step advantage A(s,a) = r + gamma * V(s') - V(s), evaluated with the critic BEFORE this
        // step's critic update (the standard A2C ordering) and treated as a constant for the actor.
        var vCurrent = _critic.Predict(states).ToVector();
        var vNext = _critic.Predict(nextStates).ToVector();
        var targetData = new T[n];
        var advantageData = new T[n];
        for (int i = 0; i < n; i++)
        {
            var bootstrap = batch[i].Done ? NumOps.Zero : NumOps.Multiply(gamma, vNext[i]);
            targetData[i] = NumOps.Add(batch[i].Reward, bootstrap);
            advantageData[i] = NumOps.Subtract(targetData[i], vCurrent[i]);
        }

        var targets = new Tensor<T>([n, 1], new Vector<T>(targetData));
        var advantages = new Tensor<T>([n], new Vector<T>(advantageData));

        _critic.Train(states, targets);
        T valueLoss = _critic.GetLastLoss();

        // Policy-gradient step on the SAME distribution SelectAction samples from:
        //   L = -mean_i( A_i * log softmax(z_i)[a_i] ) - beta * mean_i( H(softmax(z_i)) ).
        // The previous update regressed the logits onto the sampled one-hot action with MSE, which ignores
        // the advantage entirely (a punished action was reinforced exactly like a rewarded one) and treats
        // unbounded logits as probabilities.
        var actionIndices = new int[n];
        for (int i = 0; i < n; i++)
        {
            actionIndices[i] = ArgMaxIndex(batch[i].Action);
        }

        var entropyCoefficient = NumOps.FromDouble(TradingOptions.EntropyCoefficient);
        var trainableActor = (NeuralNetworkBase<T>)_actor;
        T policyLoss = trainableActor.TrainWithCustomLoss(states, logits =>
        {
            var engine = AiDotNetEngine.Current;
            var logProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(engine, logits, actionIndices);
            var policyObjective = engine.TensorMultiply(logProbs, advantages);
            var entropy = PolicyDistributionHelper<T>.ComputeDiscreteEntropy(engine, logits);
            var objective = engine.TensorAdd(policyObjective, engine.TensorMultiplyScalar(entropy, entropyCoefficient));
            var allAxes = Enumerable.Range(0, objective.Shape.Length).ToArray();
            return engine.TensorNegate(engine.ReduceMean(objective, allAxes, keepDims: false));
        });

        T loss = NumOps.Add(policyLoss, NumOps.Multiply(NumOps.FromDouble(TradingOptions.ValueCoefficient), valueLoss));
        LossHistory.Add(loss);
        return loss;
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Copies one current-policy transition into the bounded pending rollout.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Store an exact one-hot sampled action (one 1, all other entries 0).
    /// The vectors are copied so later environment updates cannot change an earlier transition.
    /// <see cref="TradingAgentOptions{T}.ReplayBufferSize"/> bounds this current rollout; at capacity
    /// the oldest pending transition is dropped. It is not an off-policy replay history.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        Guard.NotNull(state);
        Guard.NotNull(action);
        Guard.NotNull(nextState);
        if (state.Length != TradingOptions.StateSize)
            throw new ArgumentException("State length must match StateSize.", nameof(state));
        if (nextState.Length != TradingOptions.StateSize)
            throw new ArgumentException("Next-state length must match StateSize.", nameof(nextState));
        int selected = ValidateOneHotAction(action);
        SynchronizePolicyStorage();
        if (_policyRuntime.Selections.TryGetValue(action, out var selection) &&
            (!ReferenceEquals(selection.Epoch, _policyRuntime.Epoch) || !selection.Sampled || selected != selection.ActionIndex))
        {
            throw new InvalidOperationException("This action was not sampled from the current unchanged actor policy.");
        }

        var experience = new Experience<T>(state.Clone(), action.Clone(), reward, nextState.Clone(), done);
        if (_policyRuntime.Pending.Count == TradingOptions.ReplayBufferSize)
            _policyRuntime.Pending.Dequeue();
        _policyRuntime.Pending.Enqueue(experience);
    }

    private int ValidateOneHotAction(Vector<T> action)
    {
        if (action.Length != TradingOptions.ActionSize)
            throw new ArgumentException("Action length must match ActionSize.", nameof(action));
        int selected = -1;
        for (int i = 0; i < action.Length; i++)
        {
            // Compare in T, before any floating conversion: decimal values adjacent to one
            // can round to 1.0 as double but are not valid one-hot values.
            if (NumOps.Equals(action[i], NumOps.One) && selected < 0) selected = i;
            else if (!NumOps.Equals(action[i], NumOps.Zero))
                throw new ArgumentException("Action must contain exactly one 1 and otherwise only 0.", nameof(action));
        }
        if (selected < 0)
            throw new ArgumentException("Action must contain exactly one 1 and otherwise only 0.", nameof(action));
        return selected;
    }

    /// <inheritdoc/>
    protected override void OnParametersRestoring() => InvalidatePolicy();

    private void InvalidatePolicy()
    {
        _policyRuntime.Pending.Clear();
        _policyRuntime.Epoch = new object();
        _policyRuntime.Storage.Clear();
        _policyRuntime.NextStorage.Clear();
        _policyRuntime.HasSnapshot = false;
    }

    private void SynchronizePolicyStorage()
    {
        var actor = (NeuralNetworkBase<T>)_actor;
        // Read physical storage, not checkpoint payloads: opaque children and fp16 slots can
        // produce newly allocated value snapshots on every checkpoint enumeration. Those copies
        // are neither cheap mutation stamps nor evidence that an unchanged actor was replaced.
        var current = _policyRuntime.NextStorage;
        current.Clear();
        foreach (var layer in actor.Layers)
            if (layer is LayerBase<T> knownLayer)
                foreach (var stamp in knownLayer.GetParameterStorageVersions())
                    current[stamp.Storage] = stamp.Version;
        bool changed = current.Count != _policyRuntime.Storage.Count;
        foreach (var stamp in current)
            changed |= !_policyRuntime.Storage.TryGetValue(stamp.Key, out int previous) || previous != stamp.Value;
        if (_policyRuntime.HasSnapshot && changed)
        {
            _policyRuntime.Pending.Clear();
            _policyRuntime.Epoch = new object();
        }
        _policyRuntime.NextStorage = _policyRuntime.Storage;
        _policyRuntime.Storage = current;
        _policyRuntime.HasSnapshot = true;
    }

    // Runtime-only ownership state: never serialize an action's provenance or pending rollout.
    private sealed class PolicyRuntimeState
    {
        public Queue<Experience<T>> Pending { get; } = new();
        public ConditionalWeakTable<Vector<T>, SelectionStamp> Selections { get; } = new();
        public Dictionary<object, int> Storage { get; set; } = new(TensorReferenceComparer<object>.Instance);
        public Dictionary<object, int> NextStorage { get; set; } = new(TensorReferenceComparer<object>.Instance);
        public object Epoch { get; set; } = new();
        public bool HasSnapshot { get; set; }
    }

    private sealed class SelectionStamp
    {
        public SelectionStamp(object epoch, bool sampled, int actionIndex)
        { Epoch = epoch; Sampled = sampled; ActionIndex = actionIndex; }
        public object Epoch { get; }
        public bool Sampled { get; }
        public int ActionIndex { get; }
    }

    #endregion

    #region Serialization

    #endregion

    #region Model Metadata

    /// <summary>
    /// Executes GetModelMetadata for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialA2C" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        OnParametersRestoring();
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
