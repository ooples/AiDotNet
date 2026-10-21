using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Proximal Policy Optimization (Schulman et al. 2017) for trading: the clipped surrogate objective trained over a
/// fixed-length rollout, as the paper and its reference code (OpenAI Baselines <c>ppo1</c>) define it.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>The algorithm (paper, Algorithm 1 and §5):</b> act for <c>T</c> steps (<see cref="FinancialPPOAgentOptions{T}.RolloutLength"/>,
/// across episode boundaries), estimate advantages with GAE(λ), then optimize
/// <c>L = L^CLIP − c₂·S + c₁·L^VF</c> for K epochs of minibatch Adam over that rollout.
/// <c>L^CLIP = E[min(r·Â, clip(r, 1 − ε, 1 + ε)·Â)]</c> with <c>r = π(a|s) / π_old(a|s)</c>.
/// </para>
/// <para>
/// <b>Defaults are the paper's MuJoCo setting (Table 3)</b>, the one for separate fully-connected policy and value
/// networks, which this agent builds: T = 2048, Adam step size 3e-4, 10 epochs, minibatch 64, γ = 0.99, λ = 0.95,
/// ε = 0.2, no entropy bonus. Baselines' <c>ppo1</c> fills what the paper leaves open: two 64-unit tanh layers
/// with normc initialization (0.01 on the policy head), a Gaussian policy with a learned state-independent log
/// standard deviation for continuous actions, observations standardized by running statistics (updated once per
/// rollout) and clipped to ±5, advantages standardized over the rollout, one Adam (ε = 1e-5) over both networks, and
/// the value loss added at weight 1. CleanRL's additions are options: an entropy bonus
/// (<see cref="TradingAgentOptions{T}.EntropyCoefficient"/>) and global gradient-norm clipping
/// (<see cref="FinancialPPOAgentOptions{T}.MaxGradientNorm"/>).
/// </para>
/// <para><b>For Beginners:</b> PPO improves its trading policy in small, safe steps: it collects a batch of experience,
/// then updates the policy several times on that batch, but stops any action's probability from moving too far from
/// where it was when the batch was collected. That keeps one unlucky batch from wrecking a policy that was working.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new FinancialPPOAgentOptions&lt;double&gt; { StateSize = 30, ActionSize = 3 };
/// var agent = new FinancialPPOAgent&lt;double&gt;(
///     new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 30, outputSize: 3),
///     new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 30, outputSize: 1),
///     options);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Proximal Policy Optimization Algorithms", "https://arxiv.org/abs/1707.06347", Year = 2017, Authors = "John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, ReferenceBatchSize = 64,
                Source = "Schulman et al. 2017, Table 3: the Adam stepsize is 3e-4 with a minibatch "
                        + "size of 64 over 10 epochs at a horizon of 2048, for the MuJoCo "
                        + "one-million-timestep benchmark. The Roboschool table leaves the stepsize "
                        + "blank, so only the MuJoCo row is declared.")]
public partial class FinancialPPOAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>,
    IMaskableAgent<T>
{
    #region Fields

    // Baselines' RunningMeanStd: the sum starts at 0, the sum of squares and the count at epsilon = 1e-2, and the
    // variance is floored at 1e-2. Before any update the statistics are mean 0, standard deviation 1.
    private const double ObservationStatisticsEpsilon = 1e-2;

    private readonly TradingAgentOptions<T> _options;
    private readonly FinancialPPOAgentOptions<T> _ppoOptions;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic;
    // The constructor's arguments, kept under their names so a clone can replay the constructor.
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;
    // The continuous policy's log standard deviation, one per action dimension (Baselines' "logstd", zeros at init);
    // null for a discrete policy. Trained with the networks by the same optimizer.
    [TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private readonly Tensor<T>? _logStd;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly Random _random;

    private double[]? _observationSum;
    private double[]? _observationSumOfSquares;
    private double _observationCount = ObservationStatisticsEpsilon;

    // The rollout: raw observations (normalized at update time, under the statistics the update uses), the actions,
    // rewards, value predictions made when each step was taken, episode ends and legal-action masks.
    [Scratch] private readonly List<Vector<T>> _states = new();
    [Scratch] private readonly List<Vector<T>> _actions = new();
    [Scratch] private readonly List<double> _rewards = new();
    [Scratch] private readonly List<double> _values = new();
    [Scratch] private readonly List<bool> _dones = new();
    [Scratch] private readonly List<bool[]?> _stepMasks = new();
    [Scratch] private Vector<T>? _lastNextState;

    [Scratch] private Vector<T>? _lastActionRawState;
    [Scratch] private Vector<T>? _lastActionVector;
    [Scratch] private bool[]? _lastActionMask;

    private bool _maskingInUse;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <summary>The number of steps collected since the last update.</summary>
    public int RolloutCount => _states.Count;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PPO agent from its policy (actor) and value (critic) architectures.
    /// </summary>
    /// <param name="actorArchitecture">The policy network: <c>StateSize</c> inputs and <c>ActionSize</c> outputs (action
    /// logits, or the Gaussian's means for continuous actions). Left without layers, the agent builds Baselines' policy:
    /// the <see cref="TradingAgentOptions{T}.HiddenLayers"/> widths with tanh, normc-initialized, 0.01 on the head.</param>
    /// <param name="criticArchitecture">The value network: <c>StateSize</c> inputs and one output. Left without layers,
    /// the same hidden widths with a normc(1.0) head.</param>
    /// <param name="options">The agent's options; pass <see cref="FinancialPPOAgentOptions{T}"/> for PPO's own.</param>
    public FinancialPPOAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options;
        _ppoOptions = options as FinancialPPOAgentOptions<T> ?? new FinancialPPOAgentOptions<T>();
        ValidatePpoOptions(_ppoOptions);

        EnsurePpoDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize, headColumnNorm: 0.01);
        EnsurePpoDefaultLayers(criticArchitecture, options.StateSize, 1, headColumnNorm: 1.0);

        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;
        // The networks do not clip their own gradients: the update clips only when MaxGradientNorm asks it to.
        var networkOptions = new AiDotNet.NeuralNetworks.Options.NeuralNetworkDefaultOptions { MaxGradNorm = 0.0 };
        var actor = new NeuralNetwork<T>(actorArchitecture, lossFunction: new MeanSquaredErrorLoss<T>(), options: networkOptions);
        _actor = actor;
        _critic = new NeuralNetwork<T>(criticArchitecture, lossFunction: new MeanSquaredErrorLoss<T>(), options: networkOptions);
        if (options.ContinuousActions) _logStd = new Tensor<T>([options.ActionSize]);
        _optimizer = CreatePpoOptimizer(actor, options, _ppoOptions);

        Networks.Add(_actor);
        Networks.Add(_critic);
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    private static void ValidatePpoOptions(FinancialPPOAgentOptions<T> o)
    {
        if (o.RolloutLength < 2)
            throw new ArgumentOutOfRangeException(nameof(o.RolloutLength), o.RolloutLength,
                "The rollout must hold at least two steps: its advantages are standardized by their standard deviation.");
        if (o.MiniBatchSize <= 0 || o.MiniBatchSize > o.RolloutLength)
            throw new ArgumentOutOfRangeException(nameof(o.MiniBatchSize), o.MiniBatchSize,
                $"The minibatch must hold between 1 and RolloutLength ({o.RolloutLength}) steps.");
        if (o.NumEpochs <= 0) throw new ArgumentOutOfRangeException(nameof(o.NumEpochs), o.NumEpochs, "At least one epoch is needed.");
        if (!(o.AdamEpsilon > 0)) throw new ArgumentOutOfRangeException(nameof(o.AdamEpsilon), o.AdamEpsilon, "Adam's epsilon must be positive.");
        if (!(o.ObservationClipRange > 0)) throw new ArgumentOutOfRangeException(nameof(o.ObservationClipRange), o.ObservationClipRange, "The clip range must be positive.");
        if (o.MaxGradientNorm is double norm && !(norm > 0))
            throw new ArgumentOutOfRangeException(nameof(o.MaxGradientNorm), norm, "The gradient-norm bound must be positive, or null for none.");
    }

    private void EnsurePpoDefaultLayers(NeuralNetworkArchitecture<T> architecture, int expectedInputSize, int expectedOutputSize,
        double headColumnNorm)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        if (architecture.CalculatedInputSize != expectedInputSize)
            throw new ArgumentException($"Architecture input size {architecture.CalculatedInputSize} does not match expected {expectedInputSize}.", nameof(architecture));
        if (architecture.OutputSize != expectedOutputSize)
            throw new ArgumentException($"Architecture output size {architecture.OutputSize} does not match expected {expectedOutputSize}.", nameof(architecture));

        // Derive a reproducible init seed from options.Seed (when the architecture has none), so identically seeded
        // agents start from identical weights.
        ApplyNetworkSeed(architecture);
        if (architecture.Layers.Count > 0) return;

        var hiddenSizes = GetHiddenLayerSizes();
        // One generator per network, drawn in layer order as each layer materializes.
        var initialization = architecture.RandomSeed is int seed ? RandomHelper.CreateSeededRandom(seed) : null;
        AddSeededDefaultLayers(architecture, () =>
        {
            var layers = new List<ILayer<T>>(hiddenSizes.Length + 1);
            foreach (int width in hiddenSizes)
                layers.Add(new DenseLayer<T>(width, (IActivationFunction<T>)new TanhActivation<T>(),
                    new NormalizedColumnsInitializationStrategy<T>(1.0, initialization)));
            layers.Add(new DenseLayer<T>(expectedOutputSize, (IActivationFunction<T>)new IdentityActivation<T>(),
                new NormalizedColumnsInitializationStrategy<T>(headColumnNorm, initialization)));
            return layers;
        });
    }

    private static IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePpoOptimizer(
        IFullModel<T, Tensor<T>, Tensor<T>> model, TradingAgentOptions<T> options, FinancialPPOAgentOptions<T> ppo)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(model, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = numOps.ToDouble(options.LearningRate),
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = ppo.AdamEpsilon,
            UseAdaptiveBetas = false,
            UseAMSGrad = false,
            // Baselines' ppo1 does not clip; CleanRL clips the global norm at 0.5 (MaxGradientNorm).
            EnableGradientClipping = ppo.MaxGradientNorm.HasValue,
            GradientClippingMethod = GradientClippingMethod.ByNorm,
            MaxGradientNorm = ppo.MaxGradientNorm ?? GradientClippingHelper.DefaultMaxNorm,
        });
    }

    #endregion

    #region Observation statistics

    // Baselines' RunningMeanStd over the raw observations: mean = sum / count, std = sqrt(max(sumsq / count - mean², 1e-2)).
    private (double[] Mean, double[] Std) ObservationStatistics(int size)
    {
        var mean = new double[size];
        var std = new double[size];
        for (int i = 0; i < size; i++)
        {
            double sum = _observationSum is { } s && s.Length == size ? s[i] : 0.0;
            double squares = _observationSumOfSquares is { } q && q.Length == size ? q[i] : ObservationStatisticsEpsilon;
            double count = _observationSum is { } c && c.Length == size ? _observationCount : ObservationStatisticsEpsilon;
            mean[i] = sum / count;
            std[i] = Math.Sqrt(Math.Max(squares / count - mean[i] * mean[i], 1e-2));
        }
        return (mean, std);
    }

    private void UpdateObservationStatistics(IReadOnlyList<Vector<T>> states)
    {
        int size = states[0].Length;
        if (_observationSum is null || _observationSum.Length != size)
        {
            _observationSum = new double[size];
            _observationSumOfSquares = new double[size];
            for (int i = 0; i < size; i++) _observationSumOfSquares[i] = ObservationStatisticsEpsilon;
            _observationCount = ObservationStatisticsEpsilon;
        }
        foreach (var state in states)
            for (int i = 0; i < size; i++)
            {
                double value = NumOps.ToDouble(state[i]);
                _observationSum[i] += value;
                _observationSumOfSquares![i] += value * value;
            }
        _observationCount += states.Count;
    }

    // clip((x - mean) / std, -c, c), Baselines' "obz".
    private void Normalize(Vector<T> state, double[] mean, double[] std, Span<T> destination)
    {
        double clip = _ppoOptions.ObservationClipRange;
        for (int i = 0; i < state.Length; i++)
            destination[i] = NumOps.FromDouble(Math.Max(-clip, Math.Min(clip, (NumOps.ToDouble(state[i]) - mean[i]) / std[i])));
    }

    private Tensor<T> NormalizedRow(Vector<T> state)
    {
        var (mean, std) = ObservationStatistics(state.Length);
        var row = new Tensor<T>([1, state.Length]);
        Normalize(state, mean, std, row.AsWritableSpan());
        return row;
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        => SelectAction(state, training, legalActions: null);

    /// <inheritdoc cref="IMaskableAgent{T}.SelectAction(Vector{T}, bool, bool[])"/>
    /// <remarks>
    /// <para>Training samples from the policy (Baselines' <c>stochastic=True</c>); evaluation takes its mode, the most
    /// probable action or the Gaussian's mean.</para>
    /// <para><b>The mask is applied to the LOGITS, before the softmax.</b> Huang and Ontañón (2020) showed masking at
    /// this point keeps the policy gradient unbiased, and the update re-applies the same mask to the current policy, so
    /// the ratio compares the distributions the action was actually drawn from.</para>
    /// <para>The continuous branch REFUSES a mask: a real-valued action has no index set to restrict, and this agent
    /// advertises <see cref="IMaskableAgent{T}"/> whatever its <c>ContinuousActions</c> setting.</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">A non-null mask was supplied to a continuous-action policy.</exception>
    public Vector<T> SelectAction(Vector<T> state, bool training, bool[]? legalActions)
    {
        var output = _actor.Predict(NormalizedRow(state)).ToVector();
        int actionSize = TradingOptions.ActionSize;

        if (_options.ContinuousActions)
        {
            if (legalActions is not null)
            {
                throw new InvalidOperationException(
                    "A legal-action mask was supplied, but this FinancialPPOAgent is configured with "
                    + "ContinuousActions = true and emits a real-valued action vector, which has no index set "
                    + "to restrict. Constraining a continuous action is projection onto a feasible region, not "
                    + "masking. Either configure the agent for discrete actions, or apply the constraint in "
                    + "the environment.");
            }

            var continuous = new Vector<T>(actionSize);
            for (int j = 0; j < actionSize; j++)
            {
                double mean = NumOps.ToDouble(output[j]);
                continuous[j] = NumOps.FromDouble(training
                    ? mean + Math.Exp(NumOps.ToDouble(_logStd![j])) * StandardNormal()
                    : mean);
            }
            CacheSelectedAction(state, continuous, mask: null);
            return continuous;
        }

        var mask = ActionMasking.Validate(legalActions, actionSize);
        _maskingInUse |= mask is not null;
        var logits = ActionMasking.MaskLogits(output, mask, NumOps);
        int index;
        if (training)
        {
            var probabilities = AiDotNetEngine.Current.Softmax(Tensor<T>.FromVector(logits)).ToVector();
            index = SampleCategorical(probabilities);
        }
        else
        {
            index = ActionMasking.ArgMaxLegal(logits, mask, NumOps);
        }

        var action = new Vector<T>(actionSize);
        action[index] = NumOps.One;
        CacheSelectedAction(state, action, mask);
        return action;
    }

    private int SampleCategorical(Vector<T> probabilities)
    {
        double r = _random.NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += NumOps.ToDouble(probabilities[i]);
            if (r < cumulative) return i;
        }
        // Rounding can leave the cumulative sum a hair under 1; the last legal entry takes the remainder.
        for (int i = probabilities.Length - 1; i >= 0; i--)
            if (NumOps.ToDouble(probabilities[i]) > 0) return i;
        return probabilities.Length - 1;
    }

    private double StandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private void CacheSelectedAction(Vector<T> rawState, Vector<T> action, bool[]? mask)
    {
        _lastActionRawState = CopyVector(rawState);
        _lastActionVector = CopyVector(action);
        // A copy: the caller owns the array it passed and may reuse it for the next state.
        _lastActionMask = mask is null ? null : (bool[])mask.Clone();
    }

    #endregion

    #region Experience

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState,
        bool done, bool[]? nextLegalActions)
    {
        if (!done && nextLegalActions is not null)
        {
            if (_options.ContinuousActions)
                throw new InvalidOperationException("Continuous PPO does not support next-state action masks.");
            ActionMasking.Validate(nextLegalActions, TradingOptions.ActionSize);
        }
        // PPO bootstraps V(s'); its action distribution uses the captured current-state mask.
        StoreExperience(state, action, reward, nextState, done);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Records one step of the rollout with the value prediction for <paramref name="state"/> (Baselines' <c>vpred</c>,
    /// taken under the statistics the action was chosen with). A step whose state does not continue the previous
    /// step's next state ends the previous episode, so a caller that resets without passing <c>done</c> does not
    /// bootstrap across the reset.
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        if (action.Length != TradingOptions.ActionSize)
            throw new ArgumentException($"Expected an action of {TradingOptions.ActionSize} entries, got {action.Length}.", nameof(action));
        MarkHiddenEpisodeBoundaryIfNeeded(state);

        bool[]? mask = null;
        if (_lastActionRawState is not null && _lastActionVector is not null
            && VectorsApproximatelyEqual(_lastActionRawState, state) && VectorsApproximatelyEqual(_lastActionVector, action))
        {
            mask = _lastActionMask;
        }
        else if (_maskingInUse)
        {
            // Refused rather than degraded: the cache is the only record of which actions were legal when this
            // action was chosen, and the update must re-apply that mask, or PPO's ratio would divide two different
            // policies. That does not fail - it trains, and the numbers stay plausible.
            throw new InvalidOperationException(
                "StoreExperience could not match this (state, action) to the selection that produced "
                + "it, and this agent has selected under a legal-action mask, so the mask for this step "
                + "cannot be recovered. PPO needs the mask to compute the behaviour log-probability "
                + "from the same distribution the action was drawn from. Call StoreExperience with the "
                + "state and action from the immediately preceding SelectAction call.");
        }

        _states.Add(CopyVector(state));
        _actions.Add(CopyVector(action));
        _rewards.Add(NumOps.ToDouble(reward));
        _values.Add(NumOps.ToDouble(_critic.Predict(NormalizedRow(state)).ToVector()[0]));
        _dones.Add(done);
        _stepMasks.Add(mask);
        _lastNextState = CopyVector(nextState);
    }

    private void MarkHiddenEpisodeBoundaryIfNeeded(Vector<T> currentRawState)
    {
        int previous = _states.Count - 1;
        if (previous < 0 || _lastNextState is null || _dones[previous]) return;
        if (!VectorsApproximatelyEqual(_lastNextState, currentRawState)) _dones[previous] = true;
    }

    private bool VectorsApproximatelyEqual(Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
        {
            double a = NumOps.ToDouble(left[i]);
            double b = NumOps.ToDouble(right[i]);
            double tolerance = 1e-6 * Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(b)));
            if (Math.Abs(a - b) > tolerance) return false;
        }
        return true;
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) copy[i] = source[i];
        return copy;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Returns zero until <see cref="FinancialPPOAgentOptions{T}.RolloutLength"/> steps have been stored, then runs
    /// one PPO update over them (Baselines' <c>pposgd_simple.learn</c> iteration) and returns its mean loss:</para>
    /// <list type="number">
    /// <item>GAE(λ) advantages and λ-returns from the stored values, bootstrapping the last step from V(s_T);</item>
    /// <item>advantages standardized over the rollout;</item>
    /// <item>the observation statistics updated with the rollout, and the old log-probabilities taken under them;</item>
    /// <item><see cref="FinancialPPOAgentOptions{T}.NumEpochs"/> passes over a fresh shuffle in minibatches of
    /// <see cref="FinancialPPOAgentOptions{T}.MiniBatchSize"/> (a final partial minibatch is dropped), each one Adam
    /// step on <c>−L^CLIP − c₂·S + c₁·L^VF</c> over both networks.</item>
    /// </list>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Every advantage in the rollout is equal, so standardizing them divides
    /// by zero (the reference divides regardless and trains on NaN).</exception>
    public override T Train()
    {
        int n = _states.Count;
        if (n < _ppoOptions.RolloutLength) return NumOps.Zero;
        TrainingSteps++;

        var (advantages, returns) = AdvantagesAndReturns();
        Standardize(advantages);
        UpdateObservationStatistics(_states);
        var order = Enumerable.Range(0, n).ToArray();
        var oldLogProbs = OldLogProbabilities(order);

        double total = 0;
        int updates = 0;
        int batch = _ppoOptions.MiniBatchSize;
        Actor.SetTrainingMode(true);
        Critic.SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < _ppoOptions.NumEpochs; epoch++)
            {
                Shuffle(order);
                for (int start = 0; start + batch <= n; start += batch)
                {
                    var indices = new int[batch];
                    Array.Copy(order, start, indices, 0, batch);
                    total += NumOps.ToDouble(UpdateMiniBatch(indices, oldLogProbs, advantages, returns));
                    updates++;
                }
            }
        }
        finally
        {
            Actor.SetTrainingMode(false);
            Critic.SetTrainingMode(false);
        }

        var averageLoss = NumOps.FromDouble(updates == 0 ? 0 : total / updates);
        LossHistory.Add(averageLoss);
        ClearRollout();
        return averageLoss;
    }

    private void ClearRollout()
    {
        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _values.Clear();
        _dones.Clear();
        _stepMasks.Clear();
    }

    // Baselines' add_vtarg_and_adv.
    private (double[] Advantages, double[] Returns) AdvantagesAndReturns()
    {
        int n = _states.Count;
        double gamma = NumOps.ToDouble(TradingOptions.DiscountFactor), lambda = TradingOptions.GAELambda;
        double bootstrap = _dones[n - 1] ? 0.0 : NumOps.ToDouble(_critic.Predict(NormalizedRow(_lastNextState!)).ToVector()[0]);
        var advantages = new double[n];
        var returns = new double[n];
        double last = 0;
        for (int t = n - 1; t >= 0; t--)
        {
            double nonterminal = _dones[t] ? 0.0 : 1.0;
            double next = t == n - 1 ? bootstrap : _values[t + 1];
            double delta = _rewards[t] + gamma * next * nonterminal - _values[t];
            last = delta + gamma * lambda * nonterminal * last;
            advantages[t] = last;
            returns[t] = last + _values[t];
        }
        return (advantages, returns);
    }

    // (A - mean) / std, the population standard deviation as numpy's default.
    private static void Standardize(double[] values)
    {
        double mean = values.Average();
        double variance = values.Sum(v => (v - mean) * (v - mean)) / values.Length;
        double std = Math.Sqrt(variance);
        if (!(std > 0))
            throw new InvalidOperationException(
                $"Every advantage in this rollout of {values.Length} steps equals {mean}, so standardizing them divides by "
                + "zero. PPO normalizes advantages by their standard deviation over the rollout; a rollout with no "
                + "variation in reward or value gives it nothing to learn from.");
        for (int i = 0; i < values.Length; i++) values[i] = (values[i] - mean) / std;
    }

    // Fisher-Yates over the agent's generator (numpy.random.shuffle in the reference).
    private void Shuffle(int[] order)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    // log π_old(a|s) for every step, under the statistics just updated and the weights before this update
    // (Baselines assigns oldpi = pi after the observation filter's update).
    private double[] OldLogProbabilities(int[] order)
    {
        var (states, actions, bias) = BatchTensors(order);
        using var _ = new NoGradScope<T>();
        var logProbs = LogProbabilityAndEntropy(_actor.Predict(states), actions, bias).LogProb.ToVector();
        var result = new double[order.Length];
        for (int i = 0; i < result.Length; i++) result[i] = NumOps.ToDouble(logProbs[i]);
        return result;
    }

    // The minibatch's normalized states [B, S], its actions [B, A] (one-hot, or the continuous values), and the
    // additive legal-action bias [B, A] (0 or −∞; null when no step carried a mask, or for continuous actions).
    private (Tensor<T> States, Tensor<T> Actions, Tensor<T>? Bias) BatchTensors(int[] indices)
    {
        int b = indices.Length, stateSize = _states[indices[0]].Length, actionSize = TradingOptions.ActionSize;
        var (mean, std) = ObservationStatistics(stateSize);
        var states = new Tensor<T>([b, stateSize]);
        var actions = new Tensor<T>([b, actionSize]);
        var stateSpan = states.AsWritableSpan();
        var actionSpan = actions.AsWritableSpan();
        bool anyMask = false;
        for (int i = 0; i < b; i++)
        {
            var state = _states[indices[i]];
            if (state.Length != stateSize)
                throw new InvalidOperationException($"The rollout mixes state sizes ({stateSize} and {state.Length}). Use a stable observation schema.");
            Normalize(state, mean, std, stateSpan.Slice(i * stateSize, stateSize));
            var action = _actions[indices[i]];
            if (_options.ContinuousActions)
            {
                for (int j = 0; j < actionSize; j++) actionSpan[i * actionSize + j] = action[j];
            }
            else
            {
                actionSpan[i * actionSize + ArgMax(action)] = NumOps.One;
            }
            anyMask |= _stepMasks[indices[i]] is not null;
        }
        if (!anyMask || _options.ContinuousActions) return (states, actions, null);

        var bias = new Tensor<T>([b, actionSize]);
        var biasSpan = bias.AsWritableSpan();
        var blocked = ActionMasking.NegativeInfinity(NumOps);
        for (int i = 0; i < b; i++)
        {
            var mask = _stepMasks[indices[i]];
            if (mask is null) continue;
            for (int j = 0; j < actionSize; j++)
                if (!mask[j]) biasSpan[i * actionSize + j] = blocked;
        }
        return (states, actions, bias);
    }

    private T UpdateMiniBatch(int[] indices, double[] oldLogProbs, double[] advantages, double[] returns)
    {
        int b = indices.Length, actionSize = TradingOptions.ActionSize;
        var (states, actions, bias) = BatchTensors(indices);
        var old = new Tensor<T>([b]);
        var adv = new Tensor<T>([b]);
        var ret = new Tensor<T>([b, 1]);
        for (int i = 0; i < b; i++)
        {
            old[i] = NumOps.FromDouble(oldLogProbs[indices[i]]);
            adv[i] = NumOps.FromDouble(advantages[indices[i]]);
            ret[i, 0] = NumOps.FromDouble(returns[indices[i]]);
        }
        int stateSize = states.Shape[1];
        // Everything the objective reads per step travels in the packed input, which a compiled step rebinds on
        // every replay; the bias is packed as zeros when no step was masked.
        var packed = StepInputPacking<T>.Pack(states, actions, bias ?? new Tensor<T>([b, actionSize]), old, adv, ret);
        double clip = TradingOptions.PPOClipRange;
        double entropyCoefficient = TradingOptions.EntropyCoefficient, valueCoefficient = TradingOptions.ValueCoefficient;

        Tensor<T> Objective(Tensor<T> data)
        {
            var engine = AiDotNetEngine.Current;
            int offset = 0;
            Tensor<T> Next(int[] shape)
            {
                var part = StepInputPacking<T>.Slice(data, offset, shape);
                offset += shape.Aggregate(1, (x, y) => x * y);
                return part;
            }
            var s = Next([b, stateSize]);
            var a = Next([b, actionSize]);
            var maskBias = Next([b, actionSize]);
            var oldLogProb = Next([b]);
            var advantage = Next([b]);
            var target = Next([b, 1]);

            var (logProb, entropy) = LogProbabilityAndEntropy(Actor.ForwardForTraining(s), a, maskBias);
            var ratio = engine.TensorExp(engine.TensorSubtract(logProb, oldLogProb));
            var surrogate = engine.TensorMultiply(ratio, advantage);
            var clipped = engine.TensorMultiply(
                engine.TensorClamp(ratio, NumOps.FromDouble(1.0 - clip), NumOps.FromDouble(1.0 + clip)), advantage);
            // min(x, y) = −max(−x, −y)
            var pessimistic = engine.TensorNegate(engine.TensorMax(engine.TensorNegate(surrogate), engine.TensorNegate(clipped)));
            var policyLoss = engine.TensorNegate(engine.ReduceMean(pessimistic, [0], keepDims: false));
            var difference = engine.TensorSubtract(Critic.ForwardForTraining(s), target);
            var valueLoss = engine.ReduceMean(engine.TensorMultiply(difference, difference), [0, 1], keepDims: false);
            var loss = engine.TensorAdd(policyLoss, engine.TensorMultiplyScalar(valueLoss, NumOps.FromDouble(valueCoefficient)));
            if (entropyCoefficient != 0)
                loss = engine.TensorSubtract(loss, engine.TensorMultiplyScalar(
                    engine.ReduceMean(entropy, [0], keepDims: false), NumOps.FromDouble(entropyCoefficient)));
            return loss;
        }

        return Actor.TrainObjectiveFused(packed, Objective, _optimizer, TrainedTensors);
    }

    private NeuralNetworkBase<T> Actor => (NeuralNetworkBase<T>)_actor;
    private NeuralNetworkBase<T> Critic => (NeuralNetworkBase<T>)_critic;

    // Both networks and the Gaussian's log standard deviation, under one optimizer (Baselines' var_list).
    private IReadOnlyList<Tensor<T>> TrainedTensors()
    {
        var tensors = new List<Tensor<T>>(Actor.CollectModelTrainableTensors());
        tensors.AddRange(Critic.CollectModelTrainableTensors());
        if (_logStd is not null) tensors.Add(_logStd);
        return tensors;
    }

    /// <summary>
    /// The log-probability of each row's action under the policy output, and the policy's entropy, exactly.
    /// </summary>
    /// <remarks>
    /// Discrete: <c>log π(a) = logit_a − logsumexp(logits + bias)</c> and <c>H = logsumexp − Σ p·logit</c>, which stay
    /// finite with −∞ biases (a masked entry has p = 0 and never meets its −∞ in a product). Continuous (Baselines'
    /// DiagGaussianPd): <c>log π(a) = −½ Σ ((a − μ)/σ)² − ½ log(2π)·A − Σ log σ</c> and <c>H = Σ (log σ + ½ log(2πe))</c>.
    /// </remarks>
    private (Tensor<T> LogProb, Tensor<T> Entropy) LogProbabilityAndEntropy(Tensor<T> output, Tensor<T> actions, Tensor<T>? bias)
    {
        var engine = AiDotNetEngine.Current;
        int b = actions.Shape[0], actionSize = actions.Shape[1];
        var policy = engine.Reshape(output, [b, actionSize]);
        if (_logStd is not null)
        {
            var logStd = engine.TensorTile(engine.Reshape(_logStd, [1, actionSize]), [b, 1]);
            var z = engine.TensorDivide(engine.TensorSubtract(actions, policy), engine.TensorExp(logStd));
            var squares = engine.ReduceSum(engine.TensorMultiply(z, z), [1], keepDims: false);
            var logStdSum = engine.ReduceSum(logStd, [1], keepDims: false);
            var logProb = engine.TensorAddScalar(
                engine.TensorSubtract(engine.TensorMultiplyScalar(squares, NumOps.FromDouble(-0.5)), logStdSum),
                NumOps.FromDouble(-0.5 * Math.Log(2.0 * Math.PI) * actionSize));
            var entropy = engine.TensorAddScalar(logStdSum, NumOps.FromDouble(0.5 * Math.Log(2.0 * Math.PI * Math.E) * actionSize));
            return (logProb, entropy);
        }

        var masked = bias is null ? policy : engine.TensorAdd(policy, bias);
        var max = engine.ReduceMax(masked, [1], keepDims: true);
        var logSumExp = engine.TensorAdd(
            engine.TensorLog(engine.ReduceSum(engine.TensorExp(engine.TensorSubtract(masked, max)), [1], keepDims: true)), max);
        var chosen = engine.ReduceSum(engine.TensorMultiply(actions, policy), [1], keepDims: false);
        var lse = engine.Reshape(logSumExp, [b]);
        var probabilities = engine.TensorExp(engine.TensorSubtract(masked, logSumExp));
        var expectedLogit = engine.ReduceSum(engine.TensorMultiply(probabilities, policy), [1], keepDims: false);
        return (engine.TensorSubtract(chosen, lse), engine.TensorSubtract(lse, expectedLogit));
    }

    private int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        for (int i = 1; i < vector.Length; i++)
            if (NumOps.GreaterThan(vector[i], vector[maxIndex])) maxIndex = i;
        return maxIndex;
    }

    #endregion

    #region Base Implementation

    /// <inheritdoc/>
    public override void LoadModel(string filepath) => Deserialize(File.ReadAllBytes(filepath));

    /// <inheritdoc/>
    public override void SaveModel(string filepath) => File.WriteAllBytes(filepath, Serialize());

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialPPO" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>The policy network's gradients of <paramref name="lossFunction"/> on one normalized observation.</summary>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
        => _actor.ComputeGradients(NormalizedRow(input), Tensor<T>.FromVector(target), lossFunction);

    /// <summary>Applies gradients to the policy network.</summary>
    public void ApplyGradients(Vector<T> gradients, T learningRate) => _actor.ApplyGradients(gradients, learningRate);

    #endregion
}
