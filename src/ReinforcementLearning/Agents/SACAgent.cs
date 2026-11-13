using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.SAC;

/// <summary>
/// Soft Actor-Critic (SAC) agent for continuous control reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SAC is a state-of-the-art off-policy actor-critic algorithm that achieves high sample
/// efficiency and robustness by incorporating maximum entropy reinforcement learning.
/// It's particularly effective for continuous control tasks.
/// </para>
/// <para><b>For Beginners:</b>
/// SAC is one of the best algorithms for continuous control (robot movement, etc.).
///
/// Key innovations:
/// - **Maximum Entropy**: Learns to be both effective AND diverse
/// - **Twin Q-Networks**: Two critics prevent overestimation
/// - **Automatic Tuning**: Adjusts exploration automatically
/// - **Off-Policy**: Very sample efficient
///
/// Think of it like learning to drive: you want to reach your destination (high reward)
/// but also maintain flexibility in how you drive (high entropy). This makes the policy
/// more robust and adaptable.
///
/// Used by: Boston Dynamics robots, autonomous vehicles, dexterous manipulation
/// </para>
/// <para><b>Reference:</b>
/// Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor", 2018.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Soft Actor-Critic agent for continuous control
/// var options = new SACOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, AutoTuneTemperature = true };
/// var agent = new SACAgent&lt;double&gt;(options);
///
/// // Select a continuous action with entropy-driven exploration
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
    "https://arxiv.org/abs/1801.01290",
    Year = 2018,
    Authors = "Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S.")]
public partial class SACAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
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

    /// <inheritdoc />
    /// <remarks>Refreshes what derives from the parameters. This ran at the end of the
    /// hand-written SetParameters; losing it would not fail a test, it would just leave
    /// the agent training against a stale target.</remarks>
    protected override void OnParametersRestored()
    {
        CopyNetworkWeights(_q1Network, _q1TargetNetwork);
        CopyNetworkWeights(_q2Network, _q2TargetNetwork);
    }
    private SACOptions<T> _sacOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _sacOptions;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;

    private INeuralNetwork<T> _policyNetwork;      // Actor (stochastic policy)
    private INeuralNetwork<T> _q1Network;          // First Q-network (critic 1)
    private INeuralNetwork<T> _q2Network;          // Second Q-network (critic 2)
    private INeuralNetwork<T> _q1TargetNetwork;    // Target for Q1
    private INeuralNetwork<T> _q2TargetNetwork;    // Target for Q2

    private T _logAlpha;                          // Log of temperature parameter
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _sacOptions.StateSize;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public SACAgent()
        : this(new SACOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    /// <summary>
    /// Initializes a new instance of the SACAgent class.
    /// </summary>
    public SACAgent(SACOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.PolicyLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.QLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        })
    {
        Guard.NotNull(options);
        _sacOptions = options;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _steps = 0;
        _logAlpha = NumOps.FromDouble(Math.Log(NumOps.ToDouble(options.InitialTemperature)));

        // Build networks
        _policyNetwork = BuildPolicyNetwork();
        _q1Network = BuildQNetwork();
        _q2Network = BuildQNetwork();
        _q1TargetNetwork = BuildQNetwork();
        _q2TargetNetwork = BuildQNetwork();

        // Initialize target networks
        CopyNetworkWeights(_q1Network, _q1TargetNetwork);
        CopyNetworkWeights(_q2Network, _q2TargetNetwork);

        // Register networks
        Networks.Add(_policyNetwork);
        Networks.Add(_q1Network);
        Networks.Add(_q2Network);
        Networks.Add(_q1TargetNetwork);
        Networks.Add(_q2TargetNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        // Policy network outputs mean and log_std for Gaussian policy
        var layers = new List<ILayer<T>>();
        int prevSize = _sacOptions.StateSize;

        foreach (var hiddenSize in _sacOptions.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output: mean and log_std for each action dimension
        layers.Add(new DenseLayer<T>(_sacOptions.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _sacOptions.StateSize,
            outputSize: _sacOptions.ActionSize * 2,
            layers: layers
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture, optimizer: CreateNetworkOptimizer(_sacOptions.PolicyLearningRate));
    }

    private NeuralNetwork<T> BuildQNetwork()
    {
        // Q-network takes state and action as input
        var layers = new List<ILayer<T>>();
        int inputSize = _sacOptions.StateSize + _sacOptions.ActionSize;
        int prevSize = inputSize;

        foreach (var hiddenSize in _sacOptions.QHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output: single Q-value
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
            optimizer: CreateNetworkOptimizer(_sacOptions.QLearningRate),
            lossFunction: _sacOptions.QLossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();
        var (action, _) = SampleAction(policyOutput, training);
        return action;
    }

    private (Vector<T> Action, T LogProb) SampleAction(Vector<T> policyOutput, bool training)
    {
        int actionSize = _sacOptions.ActionSize;
        var engine = AiDotNetEngine.Current;
        var mean = policyOutput.GetSubVector(0, actionSize);

        if (!training)
        {
            // Deterministic: use mean with tanh
            return (engine.Tanh(mean), NumOps.Zero);
        }

        // Clip log_std for numerical stability
        var logStd = engine.Clamp(policyOutput.GetSubVector(actionSize, actionSize), NumOps.FromDouble(-20), NumOps.FromDouble(2));
        var std = engine.Exp(logStd);

        // Reparameterized sample a = tanh(mu + sigma * eps), one N(0, 1) draw per dimension in order
        var noise = RLTensorOps.NormalRandom<T>(Random, actionSize);
        var action = engine.Tanh(engine.Add(mean, engine.Multiply(std, noise)));

        // log pi = sum_i [-0.5 log(2 pi) - log sigma_i - 0.5 eps_i^2] - sum_i log(1 - a_i^2 + 1e-6)
        var gaussianLogProb = engine.Subtract(
            engine.Subtract(Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(-0.5 * Math.Log(2 * Math.PI))), logStd),
            engine.Multiply(engine.Multiply(noise, noise), NumOps.FromDouble(0.5)));
        var oneMinusSquare = engine.Subtract(Vector<T>.CreateDefault(actionSize, NumOps.One), engine.Multiply(action, action));
        var tanhCorrection = engine.TensorLog(engine.TensorAddScalar(oneMinusSquare.AsTensor(), NumOps.FromDouble(1e-6))).ToVector();
        T totalLogProb = engine.Sum(engine.Subtract(gaussianLogProb, tanhCorrection));

        return (action, totalLogProb);
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Supervised <see cref="IFullModel{T, TInput, TOutput}"/> entry point. Performs
    /// a single behavior-cloning step that regresses the actor's mean head against
    /// <paramref name="target"/> (interpreted as a demonstration action of length
    /// <c>ActionSize</c> — first elements are taken when <paramref name="target"/>
    /// is longer, missing tail elements are zero-padded). This is actor-only —
    /// the implementation deliberately does NOT enqueue a synthetic transition
    /// into the replay buffer (an earlier version did, but the fabricated
    /// <c>(reward=1, nextState=state, done=true)</c> tuple poisoned the critic
    /// targets after warmup). Critic updates still come solely from real
    /// env-driven <c>StoreExperience</c> calls.
    /// </para>
    /// <para>
    /// This override is necessary because the SAC paper-faithful defaults gate
    /// gradient updates behind <c>WarmupSteps=10000</c> and <c>BatchSize=256</c>
    /// (Haarnoja et al. 2018), which means a few supervised calls would silently
    /// no-op. Behavior-cloning pretraining is a standard SAC trick — see
    /// Rajeswaran et al. 2018 "Learning Complex Dexterous Manipulation with Deep
    /// Reinforcement Learning and Demonstrations" — so this override is paper-
    /// adjacent rather than a contract hack.
    /// </para>
    /// </remarks>
    public override void Train(Vector<T> state, Vector<T> target)
    {
        Guard.NotNull(state);
        Guard.NotNull(target);
        if (target.Length == 0)
            throw new ArgumentException("target must contain at least one element.", nameof(target));

        int stateSize = _sacOptions.StateSize;
        int actionSize = _sacOptions.ActionSize;

        // Pack target into an ActionSize-shaped demonstration action: take the first
        // ActionSize components, zero-pad the tail if target is shorter. Length match
        // means we can hand the vector straight to Tensor<T>.FromVector below; the
        // padding path only allocates a fresh vector when reshaping is needed.
        Vector<T> demoActionVec;
        if (target.Length == actionSize)
        {
            demoActionVec = target;
        }
        else
        {
            demoActionVec = new Vector<T>(actionSize);
            int copyLen = Math.Min(actionSize, target.Length);
            target.AsSpan().Slice(0, copyLen).CopyTo(demoActionVec.AsWritableSpan());
        }

        // Reshape via Tensor<T>.FromVector — this shares the underlying buffer with
        // the source vector instead of running an indexer-copy loop, so the input
        // is set up in O(1) regardless of state size. The engine then takes over
        // for the forward/backward pass (BLAS / GPU when enabled). FromVector
        // validates length against the requested shape, so a mis-sized state fails
        // fast here rather than silently corrupting the forward pass downstream.
        var batchState = Tensor<T>.FromVector(state, [1, stateSize]);
        var batchDemoAction = Tensor<T>.FromVector(demoActionVec, [1, actionSize]);

        // Behavior-cloning loss: MSE between actor mean head and the demo action.
        // Engine.TensorMSELoss is the tape-tracked, BLAS-accelerated fused op —
        // equivalent to PyTorch's F.mse_loss — instead of a manual
        // diff/multiply/reduce chain that allocates two intermediate tensors.
        var trainableActor = (NeuralNetworkBase<T>)_policyNetwork;
        trainableActor.TrainWithCustomLoss(batchState, actorOutput =>
        {
            var means = Engine.TensorSlice(actorOutput, [0, 0], [1, actionSize]);
            return Engine.TensorMSELoss(means, batchDemoAction);
        });

        // DO NOT synthesize an RL transition from a supervised label. The
        // previous code enqueued `(state, demoActionVec, reward=1,
        // nextState=state, done=true)` into the replay buffer. That fabricates
        // a critic target (reward=1, V(s')=0) that was never observed from
        // the environment — after warmup, both Q-networks would train on
        // those synthetic returns and drift away from the actual task
        // objective. Behavior cloning here is supervised actor-only;
        // critic updates remain the responsibility of real env-driven
        // (s, a, r, s', done) transitions stored via StoreExperience from
        // the rollout loop.
    }

    /// <inheritdoc/>
    public override T Train()
    {
        _steps++;
        TrainingSteps++;

        if (_steps < _sacOptions.WarmupSteps || !_replayBuffer.CanSample(_sacOptions.BatchSize))
        {
            return NumOps.Zero;
        }

        var stepLosses = new List<T>();

        // Multiple gradient steps per environment step (Haarnoja et al. 2018)
        for (int g = 0; g < _sacOptions.GradientSteps; g++)
        {
            stepLosses.Add(GradientStep());
        }

        T totalLoss = stepLosses.Count == 0 ? NumOps.Zero : AiDotNetEngine.Current.Sum(Vector<T>.FromEnumerable(stepLosses));
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_sacOptions.GradientSteps));
        LossHistory.Add(avgLoss);

        return avgLoss;
    }


    /// <summary>One SAC gradient step on a fresh minibatch; returns the critic loss plus the actor loss.</summary>
    private T GradientStep()
    {
        var batch = _replayBuffer.Sample(_sacOptions.BatchSize);
        int stateSize = _sacOptions.StateSize;
        int actionSize = _sacOptions.ActionSize;
        int batchCount = batch.Count;
        T alpha = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_logAlpha)));

        // Build batched state-action tensors for critic training
        var batchStates = RLTensorOps.Batch(batch.Select(exp => exp.State), stateSize);
        var batchStateActions = RLTensorOps.Batch(
            batch.Select(exp => RLTensorOps.Concat(RLTensorOps.Head(exp.State, stateSize), RLTensorOps.Head(exp.Action, actionSize))),
            stateSize + actionSize);

        // --- Compute Q targets (outside tape — target networks not trained) ---
        // y = r + gamma * (1-done) * (min(Q_tgt1, Q_tgt2) - alpha * log_pi_next)
        // SampleAction expects the policy network's output (mean+log_std pair of length 2*ActionSize);
        // the next states go through the policy as one batch, then each row is sampled in order (stochastic
        // branch, so nextLogProb is the real log pi(a'|s') the entropy term needs).
        var engine = AiDotNetEngine.Current;
        var nextPolicyOutputs = RLTensorOps.Rows(
            _policyNetwork.Predict(RLTensorOps.Batch(batch.Select(exp => exp.NextState), stateSize)), batchCount);
        var nextSamples = nextPolicyOutputs.Select(output => SampleAction(output, training: true)).ToList();
        var nextSA = RLTensorOps.Batch(
            batch.Select((exp, row) => ConcatenateStateAction(exp.NextState, nextSamples[row].Action)), stateSize + actionSize);
        var q1Next = RLTensorOps.FirstColumn(_q1TargetNetwork.Predict(nextSA), batchCount).AsTensor();
        var q2Next = RLTensorOps.FirstColumn(_q2TargetNetwork.Predict(nextSA), batchCount).AsTensor();
        var minQNext = engine.TensorWhere(engine.TensorLessThan(q1Next, q2Next), q1Next, q2Next).ToVector();
        var nextLogProbs = Vector<T>.FromEnumerable(nextSamples.Select(sample => sample.LogProb));
        var softV = engine.Subtract(minQNext, engine.Multiply(nextLogProbs, alpha));
        var notDone = Vector<T>.FromEnumerable(batch.Select(exp => exp.Done ? NumOps.Zero : NumOps.One));
        var rewards = Vector<T>.FromEnumerable(batch.Select(exp => exp.Reward));
        var targetValues = engine.Add(rewards, engine.Multiply(engine.Multiply(notDone, softV), DiscountFactor));
        var targetQ = new Tensor<T>([batchCount, 1], targetValues);

        // --- Update both Q-networks (MSE on TD targets) ---
        _q1Network.Train(batchStateActions, targetQ);
        T q1Loss = _q1Network.GetLastLoss();
        _q2Network.Train(batchStateActions, targetQ);
        T q2Loss = _q2Network.GetLastLoss();
        T qLoss = NumOps.Divide(NumOps.Add(q1Loss, q2Loss), NumOps.FromDouble(2.0));

        // --- Update actor (minimize alpha * log_pi - Q via engine ops) ---
        var trainableActor = (NeuralNetworkBase<T>)_policyNetwork;
        T policyLoss = trainableActor.TrainWithCustomLoss(batchStates, actorOutput =>
        {
            // actorOutput = [batchCount, actionSize*2] for continuous = [means, logStds].
            // TensorSlice takes (start, length) — length on axis 1 must be
            // actionSize for the logStds half, not actionSize*2 (which is
            // the FULL output width and would over-slice past the tensor's
            // upper bound, throwing "Slice length 2*actionSize starting at
            // actionSize exceeds axis 1 size 2*actionSize").
            var means = Engine.TensorSlice(actorOutput, [0, 0], [batchCount, actionSize]);
            var logStds = Engine.TensorSlice(actorOutput, [0, actionSize], [batchCount, actionSize]);

            // Reparameterized sample: u = mu + sigma * eps, eps ~ N(0, I). The previous noise was
            // rng.NextDouble() * 2 - 1, which is UNIFORM on [-1, 1) rather than Gaussian, so neither the
            // sample nor the log-probability below described the distribution this agent samples from.
            // It also came from an unseeded CreateSecureRandom, making the update irreproducible under a
            // configured seed; GetNormalRandom is what SampleAction already draws with.
            var stds = Engine.TensorExp(logStds);
            var noise = new Tensor<T>([batchCount, actionSize], RLTensorOps.NormalRandom<T>(Random, batchCount * actionSize));
            var preSquash = Engine.TensorAdd(means, Engine.TensorMultiply(stds, noise));

            // SampleAction squashes with tanh, so the replay buffer holds squashed actions and the
            // critics were trained on them. Handing the RAW Gaussian sample to the critics here queried
            // Q off the distribution it learned, and scored it with the wrong density: a squashed policy
            // needs the change-of-variables correction, log pi = log N(u) - sum_i log(1 - tanh^2(u_i))
            // (Haarnoja et al. 2018, Appendix C). This agent's tanh is unscaled, hence scale 1.
            var sampledActions = PolicyDistributionHelper<T>.SquashAction(Engine, preSquash, 1.0);
            var logProbs = PolicyDistributionHelper<T>.ComputeSquashedGaussianLogProb(
                Engine, means, logStds, preSquash, 1.0);

            // Build [state | action] with engine ops. Filling a fresh tensor element by element detaches
            // it from the tape, so the critics below would have been handed a constant no matter how
            // they were called.
            var stateActionForQ = Engine.TensorConcatenate([batchStates, sampledActions], axis: 1);

            // Forward the critics ON THE TAPE. Predict runs inside a NoGradScope, so reading Q through it
            // yields a detached constant and the actor ascends nothing but its own entropy — the Q term
            // contributes no gradient at all. TrainWithCustomLoss collects only the actor's tensors, so
            // the critics supply dQ/da here without being updated by this step.
            var tapedQ1 = (NeuralNetworkBase<T>)_q1Network;
            var tapedQ2 = (NeuralNetworkBase<T>)_q2Network;
            var minQ = Engine.TensorMin(
                tapedQ1.ForwardForTraining(stateActionForQ),
                tapedQ2.ForwardForTraining(stateActionForQ));

            // policy loss = mean(alpha * log_pi - min_Q). min_Q stays on the tape rather than being
            // copied element by element into a new tensor, which severed it a second time.
            var alphaLogPi = Engine.TensorMultiplyScalar(logProbs, alpha);
            var flatMinQ = Engine.ReduceSum(minQ, [1], keepDims: false);
            var loss = Engine.TensorSubtract(alphaLogPi, flatMinQ);
            return Engine.ReduceMean(loss, [0], keepDims: false);
        });

        // Update temperature (alpha)
        if (_sacOptions.AutoTuneTemperature)
        {
            UpdateTemperature(batch);
        }

        // Soft update target networks (Polyak averaging)
        SoftUpdateTargets();

        return NumOps.Add(qLoss, policyLoss);
    }

    private Vector<T> ComputeSACPolicyGradient(
        Vector<T> policyOutput, Vector<T> action, T alpha, T logProb, T qValue)
    {
        // SAC gradient: ∇θ [α log π(a|s) - Q(s,a)]
        // Split into two terms:
        // 1. Entropy term: α * ∇θ log π(a|s)
        // 2. Q term: -∇θ Q(s, f_θ(s, ε)) via reparameterization
        int actionSize = _sacOptions.ActionSize;
        var engine = AiDotNetEngine.Current;
        var gradient = new Vector<T>(actionSize * 2);
        var mean = policyOutput.GetSubVector(0, actionSize);
        var clippedLogStd = engine.Clamp(policyOutput.GetSubVector(actionSize, actionSize), NumOps.FromDouble(-20), NumOps.FromDouble(2));
        var std = engine.Exp(clippedLogStd);

        // Reconstruct the noise used: ε = (atanh(a) - μ) / σ, atanh(a) = log((1 + a) / (1 - a + 1e-6)) / 2
        // Note: action is already tanh-squashed
        var taken = RLTensorOps.Head(action, actionSize);
        var ones = Vector<T>.CreateDefault(actionSize, NumOps.One);
        var ratio = engine.Divide(
            engine.Add(ones, taken),
            engine.TensorAddScalar(engine.Subtract(ones, taken).AsTensor(), NumOps.FromDouble(1e-6)).ToVector());
        var rawAction = engine.Divide(engine.TensorLog(ratio.AsTensor()).ToVector(), NumOps.FromDouble(2.0));

        // Gradient of log probability w.r.t. mean and log_std
        // For Gaussian: ∂log_π/∂μ = (a - μ) / σ²
        //               ∂log_π/∂log_σ = -1 - (a - μ)² / σ²   (as implemented)
        var actionDiff = engine.Subtract(rawAction, mean);
        var stdSquared = engine.Multiply(std, std);
        var dLogPiDMean = engine.Divide(actionDiff, stdSquared);
        var dLogPiDLogStd = engine.Subtract(
            Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(-1.0)),
            engine.Divide(engine.Multiply(actionDiff, actionDiff), stdSquared));

        // Entropy gradient: α * ∇θ log π
        var entropyGradMean = engine.Multiply(dLogPiDMean, alpha);
        var entropyGradLogStd = engine.Multiply(dLogPiDLogStd, alpha);

        // Q-value gradient, simplified: -Q / (σ + 1e-6) through the mean, scaled by σ for log_std
        var qGradScale = engine.Negate(engine.Divide(
            Vector<T>.CreateDefault(actionSize, qValue),
            engine.TensorAddScalar(std.AsTensor(), NumOps.FromDouble(1e-6)).ToVector()));
        var qGradLogStd = engine.Multiply(qGradScale, std);

        // Total gradient: α * ∇θ log π - ∇θ Q, negated because networks minimize loss
        var meanGradient = engine.Negate(engine.Add(entropyGradMean, qGradScale));
        var logStdGradient = engine.Negate(engine.Add(entropyGradLogStd, qGradLogStd));
        meanGradient.AsSpan().CopyTo(gradient.AsWritableSpan().Slice(0, actionSize));
        logStdGradient.AsSpan().CopyTo(gradient.AsWritableSpan().Slice(actionSize, actionSize));
        return gradient;
    }

    private void UpdateTemperature(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        if (!_sacOptions.AutoTuneTemperature) return;

        // sum of log pi(a|s) over the batch: one batched policy forward, then each row sampled in order
        var policyOutputs = RLTensorOps.Rows(
            _policyNetwork.Predict(RLTensorOps.Batch(batch.Select(exp => exp.State), _sacOptions.StateSize)), batch.Count);
        var logProbs = Vector<T>.FromEnumerable(policyOutputs.Select(output => SampleAction(output, training: true).LogProb));
        T totalEntropy = batch.Count == 0 ? NumOps.Zero : AiDotNetEngine.Current.Sum(logProbs);

        var avgEntropy = NumOps.Divide(totalEntropy, NumOps.FromDouble(batch.Count));
        var targetEntropy = NumOps.FromDouble(ResolvedTargetEntropy);

        // Alpha loss: -alpha * (log_prob + target_entropy)
        var alphaLoss = NumOps.Multiply(
            NumOps.FromDouble(-Math.Exp(NumOps.ToDouble(_logAlpha))),
            NumOps.Add(avgEntropy, targetEntropy)
        );

        // Update log_alpha
        var alphaGrad = NumOps.Multiply(_sacOptions.AlphaLearningRate, alphaLoss);
        _logAlpha = NumOps.Subtract(_logAlpha, alphaGrad);
    }

    /// <summary>TargetEntropy, or the −dim(A) heuristic of Haarnoja et al. 2018b when it is null.</summary>
    internal double ResolvedTargetEntropy => _sacOptions.TargetEntropy ?? -_sacOptions.ActionSize;

    private void SoftUpdateTargets()
    {
        SoftUpdateNetwork(_q1Network, _q1TargetNetwork);
        SoftUpdateNetwork(_q2Network, _q2TargetNetwork);
    }

    private void SoftUpdateNetwork(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        // theta_target <- tau * theta + (1 - tau) * theta_target
        target.UpdateParameters(RLTensorOps.SoftUpdate(sourceParams, targetParams, _sacOptions.TargetUpdateTau));
    }

    // UpdateNetworkParameters removed — networks now trained via Train() and
    // TrainWithCustomLoss which use configured optimizers and tape-based gradients.

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        return RLTensorOps.Concat(state, action);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["Alpha"] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_logAlpha)));
        baseMetrics["ReplayBufferSize"] = NumOps.FromDouble(_replayBuffer.Count);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _sacOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // SAC uses actor-critic architecture with separate policy and value networks.
        // Gradients are computed and applied internally during Train() for each network.
        // External gradient application is not applicable for this algorithm.
        throw new NotSupportedException("SAC applies gradients internally during training. Use Train() method instead.");
    }

    // Helper methods
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
