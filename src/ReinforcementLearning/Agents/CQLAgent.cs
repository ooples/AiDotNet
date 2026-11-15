using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.CQL;

/// <summary>
/// Conservative Q-Learning (CQL) agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CQL is designed for offline RL, learning from fixed datasets without environment interaction.
/// It prevents overestimation by adding a conservative penalty that pushes down Q-values
/// for out-of-distribution actions while maintaining accuracy on in-distribution actions.
/// </para>
/// <para><b>For Beginners:</b>
/// Unlike online RL (which tries actions and learns), CQL learns only from recorded data.
/// This is crucial for domains where exploration is dangerous or expensive.
///
/// Key features:
/// - **Conservative Penalty**: Lowers Q-values for unseen state-action pairs
/// - **Offline Learning**: No environment interaction needed
/// - **Safe Policy Improvement**: Guarantees improvement over behavior policy
///
/// Example use cases:
/// - Learning from medical records (can't experiment on patients)
/// - Autonomous driving from dashcam data
/// - Robotics from demonstration datasets
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Conservative Q-Learning agent for offline RL from recorded data
/// var options = new CQLOptions&lt;double&gt; { StateSize = 4, ActionSize = 2};
/// var agent = new CQLAgent&lt;double&gt;(options);
///
/// // Select an action given the current state
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Conservative Q-Learning for Offline Reinforcement Learning",
    "https://arxiv.org/abs/2006.04779",
    Year = 2020,
    Authors = "Kumar, A., Zhou, A., Tucker, G., & Levine, S.")]
public partial class CQLAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_policyNetwork);
        RegisterParameterComponent(_q1Network);
        RegisterParameterComponent(_q2Network);
    }
    private CQLOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly INumericOperations<T> _numOps;

    private INeuralNetwork<T> _policyNetwork;
    private INeuralNetwork<T> _q1Network;
    private INeuralNetwork<T> _q2Network;
    private INeuralNetwork<T> _targetQ1Network;
    private INeuralNetwork<T> _targetQ2Network;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _offlineBuffer;  // Fixed offline dataset
    private Random _random;
    private T _logAlpha;
    private T _alpha;
    private int _updateCount;

    // CQL-Lagrange multiplier α' in log space, initialised to log 1 as in the reference implementation.
    private double _logAlphaPrime;
    private readonly ScalarAdam _alphaPrimeOptimizer;
    // Built on the first temperature update, so a zero AlphaLearningRate is fine while AutoTuneTemperature is off.
    private ScalarAdam? _alphaOptimizer;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public CQLAgent()
        : this(new CQLOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public CQLAgent(CQLOptions<T> options) : base(CreateBaseOptions(options))
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = Random;
        _updateCount = 0;

        _logAlpha = NumOps.Log(_options.InitialTemperature);
        _alpha = _options.InitialTemperature;
        _logAlphaPrime = 0.0;
        _alphaPrimeOptimizer = new ScalarAdam(_numOps.ToDouble(_options.QLearningRate));

        // Initialize networks directly in constructor
        _policyNetwork = CreatePolicyNetwork();
        _q1Network = CreateQNetwork();
        _q2Network = CreateQNetwork();
        _targetQ1Network = CreateQNetwork();
        _targetQ2Network = CreateQNetwork();

        // Register every network so DeepReinforcementLearningAgentBase.Dispose releases it.
        Networks.Add(_policyNetwork);
        Networks.Add(_q1Network);
        Networks.Add(_q2Network);
        Networks.Add(_targetQ1Network);
        Networks.Add(_targetQ2Network);

        CopyNetworkWeights(_q1Network, _targetQ1Network);
        CopyNetworkWeights(_q2Network, _targetQ2Network);

        // Initialize offline buffer
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize, _options.Seed);
    }

    private static ReinforcementLearningOptions<T> CreateBaseOptions(CQLOptions<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.QLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.QLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.BufferSize
        };
    }

    private NeuralNetwork<T> CreatePolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        // Output: mean and log_std for Gaussian policy
        layers.Add(new DenseLayer<T>(_options.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize * 2,
            layers: layers
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture, optimizer: CreateNetworkOptimizer(_options.PolicyLearningRate));
    }

    private NeuralNetwork<T> CreateQNetwork()
    {
        var layers = new List<ILayer<T>>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int previousSize = inputSize;

        foreach (var layerSize in _options.QHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_options.QLearningRate),
            lossFunction: _options.QLossFunction);
    }

    private void InitializeBuffer()
    {
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize, Random.Next());
    }

    /// <summary>
    /// Load offline dataset into the replay buffer.
    /// </summary>
    public void LoadOfflineData(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> dataset)
    {
        foreach (var transition in dataset)
        {
            var experience = new Experience<T, Vector<T>, Vector<T>>(
                transition.state,
                transition.action,
                transition.reward,
                transition.nextState,
                transition.done);
            _offlineBuffer.Add(experience);
        }
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        // Extract mean and log_std (clamped to [-20, 2] for numerical stability)
        var engine = AiDotNetEngine.Current;
        var mean = policyOutput.GetSubVector(0, _options.ActionSize);
        var logStd = engine.Clamp(
            policyOutput.GetSubVector(_options.ActionSize, _options.ActionSize), _numOps.FromDouble(-20), _numOps.FromDouble(2));

        if (!training)
        {
            // Return mean action during evaluation
            return engine.Tanh(mean);
        }

        // Sample action from Gaussian policy: tanh(mean + exp(log_std) * N(0, 1)), one draw per dimension in order
        var noise = StandardNormalDraws(_options.ActionSize);
        return engine.Tanh(engine.Add(mean, engine.Multiply(engine.Exp(logStd), noise)));
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // CQL is offline - data is loaded beforehand
        // This method is kept for interface compliance but not used in offline setting
        var experience = new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done);
        _offlineBuffer.Add(experience);
    }

    /// <inheritdoc />
    public override T Train()
    {
        if (_offlineBuffer.Count < _options.BatchSize)
        {
            return _numOps.Zero;
        }

        int gradientSteps = Math.Max(1, _options.GradientSteps);
        var stepLosses = new List<T>(gradientSteps);
        for (int step = 0; step < gradientSteps; step++)
        {
            stepLosses.Add(TrainStep());
        }

        T totalLoss = AiDotNetEngine.Current.Sum(Vector<T>.FromEnumerable(stepLosses));

        _updateCount++;
        return _numOps.Divide(totalLoss, _numOps.FromDouble(gradientSteps));
    }

    /// <summary>The Lagrange multiplier α' of the conservative penalty (fixed at 1 unless CQLLagrange is set).</summary>
    internal double ConservativeMultiplier => Math.Exp(_logAlphaPrime);

    /// <summary>The SAC entropy temperature α.</summary>
    internal double EntropyTemperature => _numOps.ToDouble(_alpha);

    /// <summary>
    /// The CQL(H) regulariser E_s[log Σ_a exp Q(s,a)] − E_(s,a)~D[Q(s,a)] measured on the most recent step,
    /// averaged over the two critics, before the CQLAlpha weight.
    /// </summary>
    internal double LastConservativeGap { get; private set; }

    /// <summary>
    /// One CQL(H) gradient step (Kumar et al. 2020, Eq. 4 with the Lagrangian of Eq. 30 when CQLLagrange is set),
    /// built on SAC as the paper's continuous-control experiments are.
    /// </summary>
    private T TrainStep()
    {
        var batch = _offlineBuffer.Sample(_options.BatchSize);
        int n = batch.Count;
        if (n == 0) return _numOps.Zero;

        int stateDim = _options.StateSize;
        int actionDim = _options.ActionSize;
        int saDim = stateDim + actionDim;
        int samples = Math.Max(1, _options.CQLNumActions);
        T gamma = _options.DiscountFactor;

        // Rows: the n dataset pairs, then for every state its `samples` uniform actions followed by its `samples`
        // policy actions, so the sampled block reshapes to [n, 2·samples] one state per row.
        var engine = AiDotNetEngine.Current;

        // log of the proposal density each sampled action is divided by in the importance-sampled log-sum-exp
        // (Kumar et al. 2020, Appendix F): Unif([-1,1]^d) has density 2^-d, and a policy sample has π(a|s).
        double uniformLogDensity = -actionDim * Math.Log(2.0);

        var batchStates = StatesTensor(batch, useNext: false);
        var policyOutputs = _policyNetwork.Predict(batchStates).ToVector();
        var nextPolicyOutputs = _policyNetwork.Predict(StatesTensor(batch, useNext: true)).ToVector();
        int policyWidth = actionDim * 2;

        // Per state, in the order the random stream is consumed: a' ~ π(·|s'), then `samples` uniform actions,
        // then `samples` policy actions.
        var blocks = batch.Select((exp, i) => SampleBlock(exp, i * policyWidth, policyOutputs, nextPolicyOutputs, samples, uniformLogDensity)).ToList();

        // Rows: the n dataset pairs, then for every state its `samples` uniform actions followed by its `samples`
        // policy actions, so the sampled block reshapes to [n, 2·samples] one state per row.
        var datasetRows = RLTensorOps.Batch(batch.Select(exp => ConcatenateStateAction(RLTensorOps.Head(exp.State, stateDim), RLTensorOps.Head(exp.Action, actionDim))), saDim);
        var qInput = engine.TensorConcatenate(new[] { datasetRows }.Concat(blocks.Select(block => block.Rows)).ToArray(), 0);
        var densityCorrection = RLTensorOps.StackRows(blocks.Select(block => block.Density).ToList());
        var nextStateActions = RLTensorOps.Batch(
            batch.Select((exp, i) => ConcatenateStateAction(RLTensorOps.Head(exp.NextState, stateDim), blocks[i].NextAction)), saDim);

        // Clipped double-Q TD target with a' ~ π(·|s') and no entropy bonus in the backup (the reference
        // implementation's deterministic backup): y = r + γ(1 − done)·min(Q'_1, Q'_2)(s', a').
        var q1Next = RLTensorOps.Head(_targetQ1Network.Predict(nextStateActions).ToVector(), n).AsTensor();
        var q2Next = RLTensorOps.Head(_targetQ2Network.Predict(nextStateActions).ToVector(), n).AsTensor();
        var minNext = engine.TensorWhere(engine.TensorLessThan(q1Next, q2Next), q1Next, q2Next).ToVector();
        var rewards = Vector<T>.FromEnumerable(batch.Select(exp => exp.Reward));
        var done = batch.Select(exp => exp.Done).ToArray();
        var tdTargets = new Tensor<T>([n, 1], RLTensorOps.TdTargets(rewards, minNext, done, gamma));

        double alphaPrime = Math.Min(Math.Exp(_logAlphaPrime), 1e6);
        double penaltyWeight = _numOps.ToDouble(_options.CQLAlpha);
        double targetGap = _options.CQLLagrange ? _numOps.ToDouble(_options.CQLTargetActionGap) : 0.0;

        double gap1 = TrainConservativeCritic(_q1Network, qInput, tdTargets, densityCorrection, n, samples, alphaPrime * penaltyWeight);
        double gap2 = TrainConservativeCritic(_q2Network, qInput, tdTargets, densityCorrection, n, samples, alphaPrime * penaltyWeight);
        LastConservativeGap = 0.5 * (gap1 + gap2);

        // Lagrangian dual ascent on α' (Kumar et al. 2020, Eq. 30): max_{α' ≥ 0} α'·(CQLAlpha·gap − τ). The
        // reference implementation descends L(α') = −½·α'·Σ_i (CQLAlpha·gap_i − τ) in log α' with the critics'
        // optimizer settings, which is what QLearningRate is.
        if (_options.CQLLagrange)
        {
            double alphaPrimeGradient = -0.5 * alphaPrime * ((penaltyWeight * gap1 - targetGap) + (penaltyWeight * gap2 - targetGap));
            _logAlphaPrime = _alphaPrimeOptimizer.Step(_logAlphaPrime, alphaPrimeGradient);
        }

        T policyLoss = TrainPolicy(batchStates, n);

        if (_options.AutoTuneTemperature)
        {
            UpdateTemperature(policyOutputs, n);
        }

        SoftUpdateTargetNetworks();

        T criticLoss = _numOps.Add(_q1Network.GetLastLoss(), _q2Network.GetLastLoss());
        return _numOps.Divide(_numOps.Add(criticLoss, policyLoss), _numOps.FromDouble(3));
    }

    private Tensor<T> StatesTensor(List<Experience<T, Vector<T>, Vector<T>>> batch, bool useNext)
    {
        return RLTensorOps.Batch(batch.Select(exp => useNext ? exp.NextState : exp.State), _options.StateSize);
    }

    /// <summary>
    /// One critic step on the QLossFunction Bellman error plus <paramref name="penaltyScale"/> times the CQL(H)
    /// regulariser; returns the regulariser measured before the step.
    /// </summary>
    private double TrainConservativeCritic(INeuralNetwork<T> critic, Tensor<T> qInput, Tensor<T> tdTargets,
        Tensor<T> densityCorrection, int n, int samples, double penaltyScale)
    {
        int width = 2 * samples;

        // log Σ exp is shift-invariant, so subtracting each row's maximum (a constant taken from the current
        // critic) changes nothing mathematically and keeps exp from overflowing.
        var engine = AiDotNetEngine.Current;
        var current = critic.Predict(qInput).ToVector();
        var currentDoubles = current.AsTensor().Cast<double>().ToVector();

        // Per state: m = max_k (Q_k + c_k), log Σ_k exp(Q_k + c_k − m) + m − log(width) − Q(s, a_data), in double.
        var scores = engine.TensorAdd(
            engine.Reshape(currentDoubles.GetSubVector(n, n * width).AsTensor(), [n, width]),
            densityCorrection.Cast<double>());
        var rowMax = engine.ReduceMax(scores, [1], keepDims: false);
        var rowSum = engine.ReduceSum(
            engine.TensorExp(engine.TensorSubtract(scores, engine.TensorTile(engine.Reshape(rowMax, [n, 1]), [1, width]))),
            [1], keepDims: false);
        var perState = engine.TensorSubtract(
            engine.TensorAddScalar(engine.TensorAdd(rowMax, engine.TensorLog(rowSum)), -Math.Log(width)),
            currentDoubles.GetSubVector(0, n).AsTensor());
        double gap = engine.TensorSum(perState) / n;
        var rowShiftVector = rowMax.Cast<T>();
        var rowShift = engine.TensorTile(engine.Reshape(rowShiftVector, [n, 1]), [1, width]);

        var trainable = (NeuralNetworkBase<T>)critic;
        trainable.TrainWithCustomLoss(qInput, q =>
        {
            var qData = Engine.TensorSlice(q, [0, 0], [n, 1]);
            // Bellman error under QLossFunction (mean squared error by default, as in the paper).
            var bellman = Engine.Reshape(_options.QLossFunction.ComputeTapeLoss(qData, tdTargets), [1]);

            var sampled = Engine.Reshape(Engine.TensorSlice(q, [n, 0], [n * width, 1]), [n, width]);
            var shifted = Engine.TensorSubtract(Engine.TensorAdd(sampled, densityCorrection), rowShift);
            var logSumExp = Engine.TensorAdd(
                Engine.TensorLog(Engine.ReduceSum(Engine.TensorExp(shifted), [1], keepDims: false)),
                rowShiftVector);
            var conservative = Engine.TensorSubtract(
                Engine.TensorAddScalar(
                    Engine.ReduceMean(logSumExp, [0], keepDims: false), _numOps.FromDouble(-Math.Log(width))),
                Engine.ReduceMean(qData, [0, 1], keepDims: false));

            return Engine.TensorAdd(bellman,
                Engine.TensorMultiplyScalar(Engine.Reshape(conservative, [1]), _numOps.FromDouble(penaltyScale)));
        });

        return gap;
    }

    /// <summary>
    /// SAC policy improvement (Haarnoja et al. 2018) against the conservative critics: minimise
    /// E[α·log π(ã|s) − min_i Q_i(s, ã)] with ã = tanh(μ + σ·ε) reparameterised on the tape.
    /// </summary>
    private T TrainPolicy(Tensor<T> batchStates, int n)
    {
        int actionSize = _options.ActionSize;
        T alpha = _alpha;
        var trainableActor = (NeuralNetworkBase<T>)_policyNetwork;
        return trainableActor.TrainWithCustomLoss(batchStates, actorOutput =>
        {
            var means = Engine.TensorSlice(actorOutput, [0, 0], [n, actionSize]);
            var logStds = Engine.TensorClamp(
                Engine.TensorSlice(actorOutput, [0, actionSize], [n, actionSize]),
                _numOps.FromDouble(-20), _numOps.FromDouble(2));
            var noise = new Tensor<T>([n, actionSize], StandardNormalDraws(n * actionSize));
            var preSquash = Engine.TensorAdd(means, Engine.TensorMultiply(Engine.TensorExp(logStds), noise));
            var actions = PolicyDistributionHelper<T>.SquashAction(Engine, preSquash, 1.0);
            var logProbs = PolicyDistributionHelper<T>.ComputeSquashedGaussianLogProb(Engine, means, logStds, preSquash, 1.0);

            // The critics are forwarded on the tape so dQ/dã reaches the actor; TrainWithCustomLoss updates only
            // the actor's own tensors.
            var stateAction = Engine.TensorConcatenate([batchStates, actions], axis: 1);
            var minQ = Engine.TensorMin(
                ((NeuralNetworkBase<T>)_q1Network).ForwardForTraining(stateAction),
                ((NeuralNetworkBase<T>)_q2Network).ForwardForTraining(stateAction));
            var objective = Engine.TensorSubtract(
                Engine.TensorMultiplyScalar(logProbs, alpha),
                Engine.ReduceSum(minQ, [1], keepDims: false));
            return Engine.ReduceMean(objective, [0], keepDims: false);
        });
    }

    /// <summary>
    /// Samples ã = tanh(μ + σ·ε) from the policy head at <paramref name="offset"/> and returns it with
    /// log π(ã|s), including the tanh change-of-variables term (Haarnoja et al. 2018, Appendix C).
    /// </summary>
    private (Vector<T> Action, double LogProb) SampleSquashedAction(Vector<T> policyOutputs, int offset)
    {
        // ã = tanh(μ + σ·ε), log π = Σ_k [−½ε_k² − log σ_k − ½ log 2π − log(1 − ã_k² + 1e-6)], in double precision
        int actionSize = _options.ActionSize;
        var engine = AiDotNetEngine.Current;
        var mean = policyOutputs.GetSubVector(offset, actionSize).AsTensor().Cast<double>().ToVector();
        var logStd = engine.Clamp(policyOutputs.GetSubVector(offset + actionSize, actionSize).AsTensor().Cast<double>().ToVector(), -20.0, 2.0);
        var epsilon = StandardNormalDraws(actionSize).AsTensor().Cast<double>().ToVector();
        var squashed = engine.Tanh(engine.Add(mean, engine.Multiply(engine.Exp(logStd), epsilon)));
        var oneMinusSquare = engine.Subtract(Vector<double>.CreateDefault(actionSize, 1.0), engine.Multiply(squashed, squashed));
        var perDimension = engine.Subtract(
            engine.Subtract(
                engine.Subtract(engine.Multiply(engine.Multiply(epsilon, epsilon), -0.5), logStd),
                Vector<double>.CreateDefault(actionSize, 0.5 * Math.Log(2.0 * Math.PI))),
            engine.TensorLog(engine.TensorAddScalar(oneMinusSquare.AsTensor(), 1e-6)).ToVector());
        return (squashed.AsTensor().Cast<T>().ToVector(), engine.Sum(perDimension));
    }

    /// <summary>
    /// <paramref name="count"/> standard-normal draws from this agent's seeded stream, in order.
    /// </summary>
    private Vector<T> StandardNormalDraws(int count)
    {
        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        T zero = _numOps.Zero;
        T one = _numOps.One;
        return new Vector<T>(Enumerable.Range(0, count).Select(_ => GetSeededNormalRandom(zero, one, _random)));
    }

    /// <summary>
    /// One state's sampled critic rows, drawn in the order the scalar loop drew them: a' ~ π(·|s') for the TD
    /// target, then <paramref name="samples"/> uniform actions on [-1, 1]^d, then <paramref name="samples"/> policy
    /// actions; with each sampled row's log proposal-density correction.
    /// </summary>
    private (Vector<T> NextAction, Tensor<T> Rows, Vector<T> Density) SampleBlock(
        Experience<T, Vector<T>, Vector<T>> exp, int offset, Vector<T> policyOutputs, Vector<T> nextPolicyOutputs,
        int samples, double uniformLogDensity)
    {
        var engine = AiDotNetEngine.Current;
        int stateDim = _options.StateSize;
        int actionDim = _options.ActionSize;

        var (nextAction, _) = SampleSquashedAction(nextPolicyOutputs, offset);

        var uniform = new Vector<double>(Enumerable.Range(0, samples * actionDim).Select(_ => _random.NextDouble()));
        var uniformActions = engine.Subtract(engine.Multiply(uniform, 2.0), Vector<double>.CreateDefault(samples * actionDim, 1.0));
        var uniformRows = engine.Reshape(uniformActions.AsTensor().Cast<T>(), [samples, actionDim]);

        var policySamples = Enumerable.Range(0, samples).Select(_ => SampleSquashedAction(policyOutputs, offset)).ToList();
        var policyRows = RLTensorOps.StackRows(policySamples.Select(sample => sample.Action).ToList());

        var states = engine.TensorTile(engine.Reshape(RLTensorOps.Head(exp.State, stateDim).AsTensor(), [1, stateDim]), [2 * samples, 1]);
        var actions = engine.TensorConcatenate([uniformRows, policyRows], 0);
        var rows = engine.TensorConcatenate([states, actions], 1);

        var density = RLTensorOps.Concat(
            Vector<double>.CreateDefault(samples, -uniformLogDensity).AsTensor().Cast<T>().ToVector(),
            new Vector<double>(policySamples.Select(sample => -sample.LogProb)).AsTensor().Cast<T>().ToVector());
        return (nextAction, rows, density);
    }

    /// <summary>
    /// SAC automatic entropy tuning (Haarnoja et al. 2018b, Eq. 18), in the form the reference implementations
    /// optimise: descend L(log α) = −log α·(log π(a|s) + H̄) at AlphaLearningRate, so the step on log α is
    /// −(log π + H̄). The target entropy H̄ is TargetEntropy, or −dim(A) when it is not configured.
    /// </summary>
    private void UpdateTemperature(Vector<T> policyOutputs, int n)
    {
        double meanLogProb = 0.0;
        for (int i = 0; i < n; i++)
        {
            meanLogProb += SampleSquashedAction(policyOutputs, i * _options.ActionSize * 2).LogProb;
        }
        meanLogProb /= n;

        double gradient = -(meanLogProb + ResolvedTargetEntropy);
        _alphaOptimizer ??= new ScalarAdam(_numOps.ToDouble(_options.AlphaLearningRate));
        double logAlpha = _alphaOptimizer.Step(_numOps.ToDouble(_logAlpha), gradient);
        _logAlpha = _numOps.FromDouble(logAlpha);
        _alpha = _numOps.FromDouble(Math.Exp(logAlpha));
    }

    /// <summary>TargetEntropy, or the −dim(A) heuristic of Haarnoja et al. 2018b when it is null.</summary>
    internal double ResolvedTargetEntropy => _options.TargetEntropy ?? -_options.ActionSize;


    private void SoftUpdateTargetNetworks()
    {
        SoftUpdateNetwork(_q1Network, _targetQ1Network);
        SoftUpdateNetwork(_q2Network, _targetQ2Network);
    }

    private void SoftUpdateNetwork(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        // theta_target <- tau * theta + (1 - tau) * theta_target
        target.UpdateParameters(RLTensorOps.SoftUpdate(sourceParams, targetParams, _options.TargetUpdateTau));
    }

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        return RLTensorOps.Concat(state, action);
    }

    private T GetSeededNormalRandom(T mean, T stdDev, Random random)
    {
        double result = random.NextGaussian() * Convert.ToDouble(stdDev) + Convert.ToDouble(mean);
        return _numOps.FromDouble(result);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_offlineBuffer.Count),
            ["alpha"] = _alpha
        };
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        // CQL is offline - no episode reset needed
    }

    /// <inheritdoc />
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

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // CQL uses custom gradient computation - return zero gradients as placeholder
        var parameters = GetParameters();
        return Vector<T>.CreateDefault(parameters.Length, _numOps.Zero);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // CQL uses direct network updates - not directly applicable
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
