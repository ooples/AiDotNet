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
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Soft Actor-Critic for trading, as Haarnoja et al. define it: by default the ICML paper (2018a), with a state-value
/// network and a fixed temperature, or the follow-up (2018b), with target Q-networks and a learned temperature.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>The policy</b> is a Gaussian over pre-squash actions whose mean and log standard deviation the actor predicts
/// from the state (log σ clipped to [−20, 2], as both authors' codes do), squashed into the position limit:
/// <c>a = MaxPositionSize · tanh(u)</c>, <c>u = μ(s) + σ(s)·ε</c>. Its log-density carries the change of variables of
/// 2018a Appendix C, computed exactly: <c>log π(a|s) = log N(u; μ, σ) − Σ log(1 − tanh²u) − dim(A)·log MaxPositionSize</c>.
/// </para>
/// <para>
/// <b>2018a (<see cref="SacVariant.ValueNetwork"/>, the default)</b>, each gradient step in the order of Algorithm 1:
/// V toward <c>min Q(s, a) − α log π(a|s)</c> with a ~ π (eq. 5); both Q-functions toward <c>r + γ(1 − d)V̄(s')</c>
/// (eq. 8); the policy on <c>E[α log π(ã|s) − min Q(s, ã)]</c> with the reparameterized ã (eq. 12); then
/// <c>V̄ ← τV + (1 − τ)V̄</c>. Every loss on the value functions is <c>½(prediction − target)²</c>.
/// </para>
/// <para>
/// <b>2018b (<see cref="SacVariant.AutomaticTemperature"/>, <see cref="FinancialSACAgentOptions{T}.Sac2018b"/>)</b>:
/// both Q-functions toward <c>r + γ(1 − d)(min Q̄(s', a') − α log π(a'|s'))</c> with a' ~ π(s'); the policy as above;
/// the temperature on <c>J(α) = E[−α(log π(a|s) + H̄)]</c> (eq. 18) with H̄ = −dim(A); then both target Q-functions
/// track by Polyak averaging.
/// </para>
/// <para>
/// Hyperparameters are the papers' Table 1: Adam (Kingma and Ba, ε = 1e-8) at 3e-4 for every network, two hidden
/// layers of 256 ReLU units with Glorot-uniform weights (the authors' TensorFlow default), minibatches of 256,
/// γ = 0.99, τ = 0.005. The authors' code differs from its papers in three places, each an option: the 2018a code
/// computes every step from the same parameters (<see cref="FinancialSACAgentOptions{T}.SimultaneousUpdates"/>) and
/// uses Q1 alone in the policy loss, softlearning averages the Q-functions there
/// (<see cref="FinancialSACAgentOptions{T}.PolicyCriticAggregate"/>) and learns log α
/// (<see cref="FinancialSACAgentOptions{T}.TemperatureParameter"/>).
/// </para>
/// <para><b>For Beginners:</b> SAC learns a policy that earns reward while staying as random as it can afford to. The
/// randomness keeps it exploring, and the "temperature" sets how much reward it will give up to stay random.</para>
/// </remarks>
/// <example>
/// <code>
/// // 2018a (the default); FinancialSACAgentOptions&lt;double&gt;.Sac2018b() gives the learned-temperature version.
/// var options = new FinancialSACAgentOptions&lt;double&gt; { StateSize = 30, ActionSize = 3, ContinuousActions = true };
/// var agent = new FinancialSACAgent&lt;double&gt;(
///     new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 30, outputSize: 6),   // 3 means, 3 log standard deviations
///     new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 33, outputSize: 1),  // Q(s, a)
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
[ResearchPaper("Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", "https://arxiv.org/abs/1801.01290", Year = 2018, Authors = "Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine")]
[ResearchPaper("Soft Actor-Critic Algorithms and Applications", "https://arxiv.org/abs/1812.05905", Year = 2018, Authors = "Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine")]
public partial class FinancialSACAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{
    #region Fields

    private const double MinLogStandardDeviation = -20.0;
    private const double MaxLogStandardDeviation = 2.0;

    private readonly TradingAgentOptions<T> _options;
    private readonly FinancialSACAgentOptions<T> _sacOptions;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic1;
    private readonly INeuralNetwork<T> _critic2;
    // 2018b: the Q-functions' Polyak-averaged targets. Null under 2018a, which bootstraps from V̄ instead.
    [Buffer]
    private readonly INeuralNetwork<T>? _targetCritic1;
    [Buffer]
    private readonly INeuralNetwork<T>? _targetCritic2;
    // 2018a: the state-value network and its Polyak-averaged target. Null under 2018b.
    private readonly INeuralNetwork<T>? _value;
    [Buffer]
    private readonly INeuralNetwork<T>? _targetValue;
    private readonly ReplayBuffer<T> ReplayBuffer;
    // The constructor's arguments, kept under their names so a clone can replay the constructor.
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;

    // The state-independent head's log standard deviation (one per action dimension); null for the paper's head.
    [TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private readonly Tensor<T>? _logStd;

    // A learned temperature: α itself or log α (TemperatureParameter). Null when the temperature is fixed.
    [TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private readonly Tensor<T>? _temperature;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _actorOptimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _critic1Optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _critic2Optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _valueOptimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _temperatureOptimizer;

    private int _updateCount;

    private readonly SacActorHead _actorHead;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SAC agent from its policy (actor) and Q-function (critic) architectures.
    /// </summary>
    /// <param name="actorArchitecture">The policy: <c>StateSize</c> inputs, and <c>2 × ActionSize</c> outputs (means then
    /// log standard deviations) for the paper's head or <c>ActionSize</c> for the state-independent one. Left without
    /// layers, the agent builds the papers' two 256-unit ReLU layers.</param>
    /// <param name="criticArchitecture">A soft Q-function: <c>StateSize + ActionSize</c> inputs, one output. Both
    /// Q-functions (and their targets) are built from it with independent initializations.</param>
    /// <param name="options">The agent's options; pass <see cref="FinancialSACAgentOptions{T}"/> for SAC's own.</param>
    /// <param name="valueArchitecture">2018a's state-value network: <c>StateSize</c> inputs, one output. Null builds
    /// it from <see cref="TradingAgentOptions{T}.HiddenLayers"/>; ignored under 2018b.</param>
    public FinancialSACAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options,
        NeuralNetworkArchitecture<T>? valueArchitecture = null)
        : base(options)
    {
        _options = options;
        _sacOptions = options as FinancialSACAgentOptions<T> ?? new FinancialSACAgentOptions<T>();
        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;
        _actorHead = _sacOptions.ActorHead;

        int actionSize = options.ActionSize;
        int actorOutputWidth = UsesStateConditionedHead ? actionSize * 2 : actionSize;
        if (actorArchitecture.OutputSize != actorOutputWidth)
        {
            throw new ArgumentException(
                UsesStateConditionedHead
                    ? $"The paper's Gaussian head needs an actor architecture with {actorOutputWidth} outputs (2 * ActionSize: "
                      + $"{actionSize} means, then {actionSize} log standard deviations), but this one has {actorArchitecture.OutputSize}. "
                      + "Widen the actor architecture, or set FinancialSACAgentOptions.ActorHead to StateIndependentLogStd."
                    : $"The state-independent head needs an actor architecture with ActionSize ({actionSize}) outputs, but this one "
                      + $"has {actorArchitecture.OutputSize}.",
                nameof(actorArchitecture));
        }

        EnsureSacDefaultLayers(actorArchitecture, options.StateSize, actorOutputWidth);
        EnsureSacDefaultLayers(criticArchitecture, options.StateSize + actionSize, 1);

        // The networks do not clip their own gradients; neither paper clips.
        var networkOptions = new AiDotNet.NeuralNetworks.Options.NeuralNetworkDefaultOptions { MaxGradNorm = 0.0 };
        NeuralNetwork<T> Build(NeuralNetworkArchitecture<T> architecture)
            => new(architecture, lossFunction: new MeanSquaredErrorLoss<T>(), options: networkOptions);

        var actor = Build(actorArchitecture);
        _actor = actor;
        _critic1 = Build(criticArchitecture);
        _critic2 = Build(IndependentlyInitialisedCritic(criticArchitecture, ordinal: 1));
        Networks.Add(_actor);
        Networks.Add(_critic1);
        Networks.Add(_critic2);
        if (_sacOptions.Variant == SacVariant.AutomaticTemperature)
        {
            _targetCritic1 = Build(IndependentlyInitialisedCritic(criticArchitecture, ordinal: 2));
            _targetCritic2 = Build(IndependentlyInitialisedCritic(criticArchitecture, ordinal: 3));
            Networks.Add(_targetCritic1);
            Networks.Add(_targetCritic2);
        }
        else
        {
            var valueArch = valueArchitecture ?? new NeuralNetworkArchitecture<T>(inputFeatures: options.StateSize, outputSize: 1);
            EnsureSacDefaultLayers(valueArch, options.StateSize, 1);
            _value = Build(valueArch);
            _targetValue = Build(IndependentlyInitialisedCritic(valueArch, ordinal: 4));
            Networks.Add(_value);
            Networks.Add(_targetValue);
        }
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);

        if (!UsesStateConditionedHead) _logStd = new Tensor<T>([actionSize]);
        if (options.AutoTuneAlpha)
        {
            _temperature = new Tensor<T>([1]);
            _temperature[0] = NumOps.FromDouble(_sacOptions.TemperatureParameter == SacTemperatureParameter.LogAlpha
                ? _sacOptions.InitialLogAlpha
                : Math.Exp(_sacOptions.InitialLogAlpha));
        }

        _actorOptimizer = CreateAdam(actor);
        _critic1Optimizer = CreateAdam(_critic1);
        _critic2Optimizer = CreateAdam(_critic2);
        if (_value is not null) _valueOptimizer = CreateAdam(_value);
        if (_temperature is not null) _temperatureOptimizer = CreateAdam(actor);

        // The targets start as copies of what they track (both papers' codes copy with τ = 1 at initialization).
        if (_targetValue is not null) SoftUpdateTargetNetwork(RequireValueNetwork(), _targetValue, 1.0);
        if (_targetCritic1 is not null) SoftUpdateTargetNetwork(_critic1, _targetCritic1, 1.0);
        if (_targetCritic2 is not null) SoftUpdateTargetNetwork(_critic2, _targetCritic2, 1.0);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateAdam(INeuralNetwork<T> model)
        => new AdamOptimizer<T, Tensor<T>, Tensor<T>>((IFullModel<T, Tensor<T>, Tensor<T>>)model,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = NumOps.ToDouble(TradingOptions.LearningRate),
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = false,
            });

    // The papers' networks: the HiddenLayers widths with ReLU and a linear output, Glorot-uniform weights and zero
    // biases (the authors' TensorFlow default); each layer draws from its own seed (LayerBase.InitializeLayerWeights).
    private void EnsureSacDefaultLayers(NeuralNetworkArchitecture<T> architecture, int expectedInputSize, int expectedOutputSize)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        if (architecture.CalculatedInputSize != expectedInputSize)
            throw new ArgumentException($"Architecture input size {architecture.CalculatedInputSize} does not match expected {expectedInputSize}.", nameof(architecture));
        if (architecture.OutputSize != expectedOutputSize)
            throw new ArgumentException($"Architecture output size {architecture.OutputSize} does not match expected {expectedOutputSize}.", nameof(architecture));
        ApplyNetworkSeed(architecture);
        if (architecture.Layers.Count > 0) return;

        var hiddenSizes = GetHiddenLayerSizes();
        AddSeededDefaultLayers(architecture, () =>
        {
            var layers = new List<ILayer<T>>(hiddenSizes.Length + 1);
            foreach (int width in hiddenSizes)
                layers.Add(new DenseLayer<T>(width, (IActivationFunction<T>)new ReLUActivation<T>(),
                    new GlorotUniformInitializationStrategy<T>()));
            layers.Add(new DenseLayer<T>(expectedOutputSize, (IActivationFunction<T>)new IdentityActivation<T>(),
                new GlorotUniformInitializationStrategy<T>()));
            return layers;
        });
    }

    private NeuralNetworkArchitecture<T> IndependentlyInitialisedCritic(
        NeuralNetworkArchitecture<T> criticArchitecture, int ordinal)
    {
        // Save and restore the ambient scope around the clone. Reseeding it here is deliberate, but leaving it
        // reseeded is not: the sequence would stay armed with THIS agent's critic seed, and the next model built on
        // this thread whose layers arrive as constructor arguments would silently initialize from it.
        var previousScope = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.CaptureScope();
        try
        {
            int? distinctSeed = TradingOptions.Seed is int seed
                ? unchecked((int)(((uint)seed * 2246822519u) ^ ((uint)(ordinal + 1) * 3266489917u)))
                : null;
            if (distinctSeed is int scopeSeed)
                AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(scopeSeed);

            var clone = criticArchitecture.CloneForModelConstruction();
            // The clone rebuilds each layer from its metadata, seed included, so without this the twin would draw
            // exactly its original's weights. Each cloned layer gets its own seed from the clone's.
            if (distinctSeed is int cloneSeed)
            {
                for (int i = 0; i < clone.Layers.Count; i++)
                    if (clone.Layers[i] is LayerBase<T> layer)
                        layer.RandomSeed = unchecked((int)(((uint)cloneSeed * 2654435761u) ^ ((uint)(i + 1) * 40503u)));
            }
            return clone;
        }
        finally
        {
            AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.RestoreScope(previousScope);
        }
    }

    #endregion

    #region Policy

    internal SacActorHead ActorHead => _actorHead;

    private INeuralNetwork<T> RequireValueNetwork() => _value ?? throw VariantMismatch("value network");
    private INeuralNetwork<T> RequireTargetValueNetwork() => _targetValue ?? throw VariantMismatch("target value network");
    private INeuralNetwork<T> RequireTargetCritic1() => _targetCritic1 ?? throw VariantMismatch("first target Q-function");
    private INeuralNetwork<T> RequireTargetCritic2() => _targetCritic2 ?? throw VariantMismatch("second target Q-function");
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> RequireValueOptimizer() => _valueOptimizer ?? throw VariantMismatch("value optimizer");
    private Tensor<T> RequireTemperature() => _temperature ?? throw new InvalidOperationException("The temperature is fixed (AutoTuneAlpha is off).");
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> RequireTemperatureOptimizer() =>
        _temperatureOptimizer ?? throw new InvalidOperationException("The temperature is fixed (AutoTuneAlpha is off).");
    private Tensor<T> RequireSharedLogStd() => _logStd ?? throw new InvalidOperationException("The paper's head predicts the spread per state.");

    private InvalidOperationException VariantMismatch(string component)
        => new($"The {_sacOptions.Variant} variant has no {component}.");

    private bool UsesStateConditionedHead => _actorHead == SacActorHead.StateConditionedGaussian;

    /// <summary>The temperature α in force: the learned one, or SACAlpha when it is fixed.</summary>
    internal double CurrentAlpha => _temperature is null
        ? TradingOptions.SACAlpha
        : _sacOptions.TemperatureParameter == SacTemperatureParameter.LogAlpha
            ? Math.Exp(NumOps.ToDouble(_temperature[0]))
            : NumOps.ToDouble(_temperature[0]);

    /// <summary>The policy's standard deviations (before the squash) for one state.</summary>
    internal double[] PolicyStandardDeviationsFor(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        using var _ = new NoGradScope<T>();
        var (_, logStd) = PolicyParameters(_actor.Predict(Row(state)), 1);
        return logStd.ToArray().Select(v => Math.Exp(NumOps.ToDouble(v))).ToArray();
    }

    internal (T Q1, T Q2) EvaluateCritics(Vector<T> state, Vector<T> action)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (action is null) throw new ArgumentNullException(nameof(action));
        var stateAction = AiDotNetEngine.Current.TensorConcatenate([Row(state), Row(action)], axis: 1);
        return (_critic1.Predict(stateAction).ToVector()[0], _critic2.Predict(stateAction).ToVector()[0]);
    }

    private static Tensor<T> Row(Vector<T> vector)
    {
        var row = new Tensor<T>([1, vector.Length]);
        for (int i = 0; i < vector.Length; i++) row[0, i] = vector[i];
        return row;
    }

    // The policy's mean and clipped log standard deviation for n states, [n, A] each.
    private (Tensor<T> Mean, Tensor<T> LogStd) PolicyParameters(Tensor<T> actorOutput, int n)
    {
        var engine = AiDotNetEngine.Current;
        int a = TradingOptions.ActionSize;
        var output = engine.Reshape(actorOutput, [n, UsesStateConditionedHead ? 2 * a : a]);
        var mean = engine.TensorSlice(output, [0, 0], [n, a]);
        var rawLogStd = UsesStateConditionedHead
            ? engine.TensorSlice(output, [0, a], [n, a])
            : engine.TensorTile(engine.Reshape(RequireSharedLogStd(), [1, a]), [n, 1]);
        var logStd = engine.TensorClamp(rawLogStd, NumOps.FromDouble(MinLogStandardDeviation), NumOps.FromDouble(MaxLogStandardDeviation));
        return (mean, logStd);
    }

    // a = M·tanh(μ + σε) and log π(a|s), exactly: log N(u; μ, σ) − Σ log(1 − tanh²u) − A·log M, with
    // log(1 − tanh²u) = 2(log 2 − |u| − log(1 + e^{−2|u|})) and (u − μ)/σ = ε.
    private (Tensor<T> Action, Tensor<T> LogProb) SquashedSample(Tensor<T> mean, Tensor<T> logStd, Tensor<T> noise)
    {
        var engine = AiDotNetEngine.Current;
        int n = mean.Shape[0], a = mean.Shape[1];
        double maxPosition = Convert.ToDouble(TradingOptions.MaxPositionSize);
        var u = engine.TensorAdd(mean, engine.TensorMultiply(engine.TensorExp(logStd), noise));
        var action = engine.TensorMultiplyScalar(engine.TensorTanh(u), NumOps.FromDouble(maxPosition));
        var gaussian = engine.TensorSubtract(
            engine.TensorMultiplyScalar(engine.TensorMultiply(noise, noise), NumOps.FromDouble(-0.5)), logStd);
        var absU = engine.TensorAbs(u);
        var logOneMinusTanhSquared = engine.TensorMultiplyScalar(
            engine.TensorSubtract(engine.TensorAddScalar(engine.TensorNegate(absU), NumOps.FromDouble(Math.Log(2.0))),
                engine.TensorLog(engine.TensorAddScalar(engine.TensorExp(engine.TensorMultiplyScalar(absU, NumOps.FromDouble(-2.0))), NumOps.One))),
            NumOps.FromDouble(2.0));
        var perDimension = engine.TensorSubtract(gaussian, logOneMinusTanhSquared);
        var logProb = engine.TensorAddScalar(engine.ReduceSum(perDimension, [1], keepDims: false),
            NumOps.FromDouble(-a * (0.5 * Math.Log(2.0 * Math.PI) + Math.Log(maxPosition))));
        return (action, engine.Reshape(logProb, [n]));
    }

    private Tensor<T> StandardNormals(int n)
    {
        int a = TradingOptions.ActionSize;
        var noise = new Tensor<T>([n, a]);
        var span = noise.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = NumOps.FromDouble(NextStandardNormal());
        return noise;
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>Training samples <c>a = MaxPositionSize · tanh(μ + σε)</c>; evaluation takes the squashed mean,
    /// <c>MaxPositionSize · tanh(μ)</c>, as the authors' deterministic policy does.</remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        using var _ = new NoGradScope<T>();
        var (mean, logStd) = PolicyParameters(_actor.Predict(Row(state)), 1);
        if (!training)
        {
            double maxPosition = Convert.ToDouble(TradingOptions.MaxPositionSize);
            return AiDotNetEngine.Current.TensorMultiplyScalar(AiDotNetEngine.Current.TensorTanh(mean), NumOps.FromDouble(maxPosition))
                .ToVector();
        }
        return SquashedSample(mean, logStd, StandardNormals(1)).Action.ToVector();
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// One gradient step (the papers' one step per environment step) on a minibatch drawn uniformly with replacement
    /// from the replay buffer, once the buffer holds a minibatch and the warm-up has passed. The order is the papers'
    /// Algorithm 1, or one shared parameter snapshot with <see cref="FinancialSACAgentOptions{T}.SimultaneousUpdates"/>.
    /// Returns the mean of the Q losses plus the policy loss.
    /// </remarks>
    /// <exception cref="InvalidOperationException">A learned α stepped on directly
    /// (<see cref="SacTemperatureParameter.Alpha"/>) became non-positive.</exception>
    public override T Train()
    {
        // A supervised one-shot Train(state, target) call trains on the samples gathered so far (clamped to the
        // buffer); autonomous stepping requires a full minibatch.
        int batchSize = SupervisedUpdateRequested
            ? Math.Min(TradingOptions.BatchSize, ReplayBuffer.Count)
            : TradingOptions.BatchSize;
        if (batchSize <= 0 || ReplayBuffer.Count < batchSize) return NumOps.Zero;
        if (IsInWarmup(ReplayBuffer.Count)) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(batchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;
        int stateDim = batch[0].State.Length, actionDim = batch[0].Action.Length;
        var states = Rows(batch.Select(e => e.State).ToList(), stateDim);
        var nextStates = Rows(batch.Select(e => e.NextState).ToList(), stateDim);
        var actions = Rows(batch.Select(e => e.Action).ToList(), actionDim);
        var rewards = new double[n];
        var notDone = new double[n];
        for (int i = 0; i < n; i++)
        {
            rewards[i] = NumOps.ToDouble(batch[i].Reward);
            notDone[i] = batch[i].Done ? 0.0 : 1.0;
        }

        double alpha = CurrentAlpha;
        double gamma = NumOps.ToDouble(TradingOptions.DiscountFactor);
        bool v1 = _sacOptions.Variant == SacVariant.ValueNetwork;
        T qLoss1, qLoss2, policyLoss;
        SetTrainingModes(true);
        try
        {
            if (_sacOptions.SimultaneousUpdates)
            {
                // Every target from the pre-step parameters, and one policy sample at s shared by the value target,
                // the policy loss and the temperature loss (the 2018a code's single session call).
                var noise = StandardNormals(n);
                var (valueTarget, logProbs) = PolicyTargets(states, noise);
                var qTargets = QTargets(nextStates, rewards, notDone, gamma, alpha);
                policyLoss = UpdatePolicy(states, noise, alpha);
                qLoss1 = UpdateQ(_critic1, _critic1Optimizer, states, actions, qTargets);
                qLoss2 = UpdateQ(_critic2, _critic2Optimizer, states, actions, qTargets);
                if (v1) UpdateValue(states, valueTarget);
                if (_temperature is not null) UpdateTemperature(logProbs);
            }
            else if (v1)
            {
                // 2018a, Algorithm 1: V, then both Q-functions, then the policy.
                var (valueTarget, _) = PolicyTargets(states, StandardNormals(n));
                UpdateValue(states, valueTarget);
                var qTargets = QTargets(nextStates, rewards, notDone, gamma, alpha);
                qLoss1 = UpdateQ(_critic1, _critic1Optimizer, states, actions, qTargets);
                qLoss2 = UpdateQ(_critic2, _critic2Optimizer, states, actions, qTargets);
                policyLoss = UpdatePolicy(states, StandardNormals(n), alpha);
                if (_temperature is not null) UpdateTemperature(PolicyTargets(states, StandardNormals(n)).LogProbs);
            }
            else
            {
                // 2018b, Algorithm 1: both Q-functions, then the policy, then the temperature.
                var qTargets = QTargets(nextStates, rewards, notDone, gamma, alpha);
                qLoss1 = UpdateQ(_critic1, _critic1Optimizer, states, actions, qTargets);
                qLoss2 = UpdateQ(_critic2, _critic2Optimizer, states, actions, qTargets);
                policyLoss = UpdatePolicy(states, StandardNormals(n), alpha);
                if (_temperature is not null) UpdateTemperature(PolicyTargets(states, StandardNormals(n)).LogProbs);
            }
        }
        finally
        {
            SetTrainingModes(false);
        }

        double tau = TradingOptions.Tau;
        if (v1)
        {
            SoftUpdateTargetNetwork(RequireValueNetwork(), RequireTargetValueNetwork(), tau);
        }
        else
        {
            SoftUpdateTargetNetwork(_critic1, RequireTargetCritic1(), tau);
            SoftUpdateTargetNetwork(_critic2, RequireTargetCritic2(), tau);
        }
        if (_updateCount < int.MaxValue) _updateCount++;

        T loss = NumOps.Add(NumOps.Divide(NumOps.Add(qLoss1, qLoss2), NumOps.FromDouble(2.0)), policyLoss);
        LossHistory.Add(loss);
        return loss;
    }

    private void SetTrainingModes(bool training)
    {
        foreach (var network in new[] { _actor, _critic1, _critic2, _value })
            if (network is not null) ((NeuralNetworkBase<T>)network).SetTrainingMode(training);
    }

    private static Tensor<T> Rows(IReadOnlyList<Vector<T>> vectors, int width)
    {
        var rows = new Tensor<T>([vectors.Count, width]);
        var span = rows.AsWritableSpan();
        for (int i = 0; i < vectors.Count; i++)
            for (int j = 0; j < width; j++) span[i * width + j] = vectors[i][j];
        return rows;
    }

    private static NeuralNetworkBase<T> Trainable(INeuralNetwork<T> network) => (NeuralNetworkBase<T>)network;

    private Tensor<T> MinQ(Tensor<T> q1, Tensor<T> q2) => AiDotNetEngine.Current.TensorMin(q1, q2);

    // With a sample a ~ π(s) (no gradient): the 2018a value target min Q(s, a) − α log π(a|s), and log π(a|s).
    private (Tensor<T> ValueTarget, Tensor<T> LogProbs) PolicyTargets(Tensor<T> states, Tensor<T> noise)
    {
        var engine = AiDotNetEngine.Current;
        int n = states.Shape[0];
        using var _ = new NoGradScope<T>();
        var (mean, logStd) = PolicyParameters(_actor.Predict(states), n);
        var (action, logProb) = SquashedSample(mean, logStd, noise);
        var stateAction = engine.TensorConcatenate([states, action], axis: 1);
        var minQ = engine.Reshape(MinQ(_critic1.Predict(stateAction), _critic2.Predict(stateAction)), [n]);
        var target = engine.TensorSubtract(minQ, engine.TensorMultiplyScalar(logProb, NumOps.FromDouble(CurrentAlpha)));
        return (engine.Reshape(target, [n, 1]), logProb);
    }

    // The Q-functions' target: 2018a r + γ(1 − d)V̄(s'); 2018b r + γ(1 − d)(min Q̄(s', a') − α log π(a'|s')), a' ~ π(s').
    private Tensor<T> QTargets(Tensor<T> nextStates, double[] rewards, double[] notDone, double gamma, double alpha)
    {
        var engine = AiDotNetEngine.Current;
        int n = nextStates.Shape[0];
        using var _ = new NoGradScope<T>();
        Tensor<T> next;
        if (_sacOptions.Variant == SacVariant.ValueNetwork)
        {
            next = engine.Reshape(RequireTargetValueNetwork().Predict(nextStates), [n]);
        }
        else
        {
            var (mean, logStd) = PolicyParameters(_actor.Predict(nextStates), n);
            var (action, logProb) = SquashedSample(mean, logStd, StandardNormals(n));
            var stateAction = engine.TensorConcatenate([nextStates, action], axis: 1);
            var minQ = engine.Reshape(MinQ(RequireTargetCritic1().Predict(stateAction), RequireTargetCritic2().Predict(stateAction)), [n]);
            next = engine.TensorSubtract(minQ, engine.TensorMultiplyScalar(logProb, NumOps.FromDouble(alpha)));
        }
        var values = next.ToVector();
        var targets = new Tensor<T>([n, 1]);
        for (int i = 0; i < n; i++)
            targets[i, 0] = NumOps.FromDouble(rewards[i] + gamma * notDone[i] * NumOps.ToDouble(values[i]));
        return targets;
    }

    // ½ mean((Q(s, a) − y)²) on one Q-function.
    private T UpdateQ(INeuralNetwork<T> critic, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Tensor<T> states, Tensor<T> actions, Tensor<T> targets)
    {
        int n = states.Shape[0], width = states.Shape[1] + actions.Shape[1];
        var packed = StepInputPacking<T>.Pack(AiDotNetEngine.Current.TensorConcatenate([states, actions], axis: 1), targets);
        var network = Trainable(critic);
        return network.TrainObjectiveFused(packed, data =>
        {
            var engine = AiDotNetEngine.Current;
            var stateAction = StepInputPacking<T>.Slice(data, 0, [n, width]);
            var target = StepInputPacking<T>.Slice(data, n * width, [n, 1]);
            var difference = engine.TensorSubtract(network.ForwardForTraining(stateAction), target);
            return engine.TensorMultiplyScalar(engine.ReduceMean(engine.TensorMultiply(difference, difference), [0, 1], keepDims: false),
                NumOps.FromDouble(0.5));
        }, optimizer);
    }

    // 2018a eq. 5: ½ mean((V(s) − target)²).
    private void UpdateValue(Tensor<T> states, Tensor<T> targets)
    {
        int n = states.Shape[0], width = states.Shape[1];
        var packed = StepInputPacking<T>.Pack(states, targets);
        var network = Trainable(RequireValueNetwork());
        network.TrainObjectiveFused(packed, data =>
        {
            var engine = AiDotNetEngine.Current;
            var s = StepInputPacking<T>.Slice(data, 0, [n, width]);
            var target = StepInputPacking<T>.Slice(data, n * width, [n, 1]);
            var difference = engine.TensorSubtract(network.ForwardForTraining(s), target);
            return engine.TensorMultiplyScalar(engine.ReduceMean(engine.TensorMultiply(difference, difference), [0, 1], keepDims: false),
                NumOps.FromDouble(0.5));
        }, RequireValueOptimizer());
    }

    // mean(α log π(ã|s) − Q_agg(s, ã)), ã = M·tanh(μ + σε) reparameterized; only the policy (and its log-std) trains.
    private T UpdatePolicy(Tensor<T> states, Tensor<T> noise, double alpha)
    {
        int n = states.Shape[0], width = states.Shape[1], a = TradingOptions.ActionSize;
        // α travels in the packed input: a compiled step rebinds its input on every replay but keeps any host value
        // the objective captures, and a learned α changes every step.
        var temperature = new Tensor<T>([1, 1]);
        temperature[0, 0] = NumOps.FromDouble(alpha);
        var packed = StepInputPacking<T>.Pack(states, noise, temperature);
        var actor = Trainable(_actor);
        var critic1 = Trainable(_critic1);
        var critic2 = Trainable(_critic2);
        var aggregate = _sacOptions.PolicyCriticAggregate;
        return actor.TrainObjectiveFused(packed, data =>
        {
            var engine = AiDotNetEngine.Current;
            var s = StepInputPacking<T>.Slice(data, 0, [n, width]);
            var epsilon = StepInputPacking<T>.Slice(data, n * width, [n, a]);
            var alphaTensor = StepInputPacking<T>.Slice(data, n * width + n * a, [1]);
            var (mean, logStd) = PolicyParameters(actor.ForwardForTraining(s), n);
            var (action, logProb) = SquashedSample(mean, logStd, epsilon);
            var stateAction = engine.TensorConcatenate([s, action], axis: 1);
            var q1 = critic1.ForwardForTraining(stateAction);
            var q = aggregate switch
            {
                SacCriticAggregate.FirstCritic => q1,
                SacCriticAggregate.Mean => engine.TensorMultiplyScalar(engine.TensorAdd(q1, critic2.ForwardForTraining(stateAction)), NumOps.FromDouble(0.5)),
                _ => engine.TensorMin(q1, critic2.ForwardForTraining(stateAction)),
            };
            var objective = engine.TensorSubtract(engine.TensorMultiply(logProb, engine.TensorTile(alphaTensor, [n])), engine.Reshape(q, [n]));
            return engine.ReduceMean(objective, [0], keepDims: false);
        }, _actorOptimizer, () => PolicyTensors(actor));
    }

    private IReadOnlyList<Tensor<T>> PolicyTensors(NeuralNetworkBase<T> actor)
    {
        var tensors = new List<Tensor<T>>(actor.CollectModelTrainableTensors());
        if (_logStd is not null) tensors.Add(_logStd);
        return tensors;
    }

    // 2018b eq. 18: J = −mean(α (log π(a|s) + H̄)), on α itself (Algorithm 1) or on log α (softlearning).
    private void UpdateTemperature(Tensor<T> logProbs)
    {
        double targetEntropy = _sacOptions.TargetEntropyRatio * TradingOptions.ActionSize;
        double mean = logProbs.ToArray().Average(v => NumOps.ToDouble(v)) + targetEntropy;
        var temperature = RequireTemperature();
        bool logParameter = _sacOptions.TemperatureParameter == SacTemperatureParameter.LogAlpha;
        var unit = new Tensor<T>([1]);
        var request = new FusedTrainingStepRequest<T>
        {
            Layers = Array.Empty<ITrainableLayer<T>>(),
            Input = unit,
            Target = unit,
            Forward = _ =>
            {
                var engine = AiDotNetEngine.Current;
                var alphaValue = logParameter ? engine.TensorExp(temperature) : temperature;
                return engine.TensorMultiplyScalar(engine.ReduceSum(alphaValue, [0], keepDims: false), NumOps.FromDouble(-mean));
            },
            ComputeLoss = (loss, _) => loss,
            Optimizer = RequireTemperatureOptimizer(),
            ExtraParameters = [temperature],
            Selection = [temperature],
        };
        TapeTrainingStepper<T>.EagerStep(request);
        if (!logParameter && !(NumOps.ToDouble(temperature[0]) > 0))
            throw new InvalidOperationException(
                $"The learned temperature α reached {NumOps.ToDouble(temperature[0])}. The 2018b paper steps α directly on an objective "
                + "linear in α, which nothing keeps positive; set FinancialSACAgentOptions.TemperatureParameter to LogAlpha "
                + "(softlearning's parameterization) to keep it positive.");
    }

    #endregion

    #region Base Implementation

    /// <inheritdoc/>
    public override void LoadModel(string filepath) => Deserialize(File.ReadAllBytes(filepath));

    /// <inheritdoc/>
    public override void SaveModel(string filepath) => File.WriteAllBytes(filepath, Serialize());

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        ValidateTransitionShape(state, action, nextState);
        ReplayBuffer.Add(new Experience<T>(state, action, ScaleReward(reward), nextState, done));
    }

    #endregion

    #region Model Metadata

    /// <inheritdoc/>
    public override Dictionary<string, T> GetTradingMetrics()
    {
        var metrics = base.GetTradingMetrics();
        metrics["Alpha"] = NumOps.FromDouble(CurrentAlpha);
        if (_logStd is not null)
            metrics["PolicyStdDev"] = NumOps.FromDouble(_logStd.ToArray().Average(v => Math.Exp(MathPolyfill.Clamp(
                NumOps.ToDouble(v), MinLogStandardDeviation, MaxLogStandardDeviation))));
        metrics["Updates"] = NumOps.FromDouble(_updateCount);
        return metrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialSAC" },
                { "Variant", _sacOptions.Variant.ToString() },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>The policy network's gradients of <paramref name="lossFunction"/> on one state.</summary>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
        => _actor.ComputeGradients(Row(input), Tensor<T>.FromVector(target), lossFunction);

    /// <summary>Applies gradients to the policy network.</summary>
    public void ApplyGradients(Vector<T> gradients, T learningRate) => _actor.ApplyGradients(gradients, learningRate);

    #endregion
}
