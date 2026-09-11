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
using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Financial Soft Actor-Critic (SAC) agent for high-performance continuous trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The SAC (Soft Actor-Critic) trading agent is designed for
/// continuous trading decisions, like choosing exact position sizes (e.g., buy 37% of
/// portfolio capacity). It encourages exploration by maximizing both returns and the
/// "entropy" (randomness) of its strategy, which prevents it from getting stuck in a
/// suboptimal trading pattern. SAC is considered state-of-the-art for continuous action
/// spaces and adapts well to changing market conditions.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define actor and critic architectures for SAC continuous trading (30 features, 5 position sizes)
/// var actorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 5);
/// var criticArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 1);
///
/// // Create SAC agent for entropy-regularized continuous portfolio allocation
/// var options = new TradingAgentOptions&lt;double&gt;();
/// var model = new FinancialSACAgent&lt;double&gt;(actorArch, criticArch, options);
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
public partial class FinancialSACAgent<T> : TradingAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    #region Fields

    private readonly TradingAgentOptions<T> _options;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic1;
    private readonly INeuralNetwork<T> _critic2;
    [Buffer]
    private readonly INeuralNetwork<T> _targetCritic1;
    [Buffer]
    private readonly INeuralNetwork<T> _targetCritic2;
    private readonly ReplayBuffer<T> ReplayBuffer;
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;

    /// <summary>
    /// Log of the entropy temperature alpha. Persisted so a reloaded agent resumes its tuned temperature
    /// instead of restarting from the configured one.
    /// </summary>
    private T _logAlpha;

    /// <summary>
    /// Per-action-dimension log standard deviation of the policy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The actor head is deterministic (one output per action dimension — the architecture contract callers
    /// already build against), so the policy's spread lives here instead: pi(.|s) = N(mu(s), diag(exp(logStd)^2)),
    /// a state-independent diagonal Gaussian. This is the same state-independent log-std used by the reference
    /// PPO/SAC implementations in OpenAI Baselines, and it is what makes the entropy term real: without a
    /// learned spread the policy entropy is a constant and <see cref="TradingAgentOptions{T}.AutoTuneAlpha"/>
    /// could only push alpha monotonically to zero or infinity.
    /// </para>
    /// </remarks>
    private Vector<T> _logStd;

    /// <summary>
    /// Gradient updates applied so far. Exposed through the trading metrics and persisted with the agent.
    /// </summary>
    private int _updateCount;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialSACAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the critics.</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, FinancialSACAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialSACAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options;
        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;

        EnsureDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize);
        EnsureDefaultLayers(criticArchitecture, options.StateSize + options.ActionSize, 1);

        _actor = new NeuralNetwork<T>(actorArchitecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic1 = new NeuralNetwork<T>(criticArchitecture, lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic2 = new NeuralNetwork<T>(IndependentlyInitialisedCritic(criticArchitecture, ordinal: 1), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetCritic1 = new NeuralNetwork<T>(IndependentlyInitialisedCritic(criticArchitecture, ordinal: 2), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetCritic2 = new NeuralNetwork<T>(IndependentlyInitialisedCritic(criticArchitecture, ordinal: 3), lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);

        // Policy spread starts at the historical exploration width so behaviour is unchanged on step 0.
        _logStd = new Vector<T>(options.ActionSize);
        for (int i = 0; i < options.ActionSize; i++)
        {
            _logStd[i] = NumOps.FromDouble(Math.Log(InitialExplorationStandardDeviation));
        }

        // Temperature: the configured SACAlpha is the operating temperature; when auto-tuning is on, a
        // SAC-specific options object may instead name the starting point for the tuned value.
        double initialAlpha = options.SACAlpha;
        if (options.AutoTuneAlpha && options is FinancialSACAgentOptions<T> sacOptions)
        {
            initialAlpha = Math.Exp(sacOptions.InitialLogAlpha);
        }

        _logAlpha = NumOps.FromDouble(Math.Log(Math.Max(initialAlpha, 1e-8)));

        // Hard sync: the targets must start equal to the critics they track.
        SoftUpdateTargetNetwork(_critic1, _targetCritic1, 1.0);
        SoftUpdateTargetNetwork(_critic2, _targetCritic2, 1.0);
    }

    /// <summary>
    /// Current entropy temperature alpha (<c>exp(logAlpha)</c>).
    /// </summary>
    public double CurrentAlpha => Math.Exp(NumOps.ToDouble(_logAlpha));

    /// <summary>
    /// Current per-dimension policy standard deviations, <c>exp(logStd)</c>.
    /// </summary>
    public double[] CurrentPolicyStandardDeviations => CurrentLogStandardDeviations()
        .Select(Math.Exp)
        .ToArray();

    /// <summary>
    /// Evaluates the twin critics at a state-action pair, returning <c>(Q1, Q2)</c>.
    /// </summary>
    /// <remarks>
    /// Exposed so callers (and tests) can inspect what the critics actually learned — the twin values are
    /// separately meaningful, and their agreement is the diagnostic that says the pair is genuinely
    /// independent rather than two references to one network.
    /// </remarks>
    public (T Q1, T Q2) EvaluateCritics(Vector<T> state, Vector<T> action)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (action is null) throw new ArgumentNullException(nameof(action));

        var stateAction = Tensor<T>.FromVector(ConcatenateStateAction(state, action));
        return (_critic1.Predict(stateAction).ToVector()[0], _critic2.Predict(stateAction).ToVector()[0]);
    }

    /// <summary>
    /// Clones the critic architecture so the new network's weights are drawn INDEPENDENTLY of the critic it
    /// is cloned from.
    /// </summary>
    /// <param name="criticArchitecture">The architecture <see cref="_critic1"/> was built from.</param>
    /// <param name="ordinal">Distinguishes the clones, so no two draw the same weights.</param>
    /// <remarks>
    /// <para>
    /// <see cref="NeuralNetworkArchitecture{T}.CloneForModelConstruction"/> gives the clone its own layer
    /// OBJECTS — which is what stops two networks sharing mutable layers — but it rebuilds each layer through
    /// the layer's constructor, so the new weights come from the ambient initialization-seed scope, and it
    /// copies the source architecture's <c>RandomSeed</c>. Under a fixed
    /// <see cref="TradingAgentOptions{T}.Seed"/> every clone therefore re-initializes from the SAME seed and
    /// the twin critics come out BIT-IDENTICAL. Nothing fails: <c>min(Q1, Q2)</c> simply equals <c>Q1</c>, and
    /// the overestimation control that twin critics exist to provide is silently gone. Re-seeding the scope
    /// per clone is what makes the pair genuinely independent.
    /// </para>
    /// <para>
    /// With no seed configured the initialization is already independent (each layer draws from the shared
    /// non-deterministic RNG), so the scope is left alone and behaviour is unchanged.
    /// </para>
    /// </remarks>
    private NeuralNetworkArchitecture<T> IndependentlyInitialisedCritic(
        NeuralNetworkArchitecture<T> criticArchitecture, int ordinal)
    {
        if (TradingOptions.Seed is int seed)
        {
            int distinctSeed = unchecked((int)(((uint)seed * 2246822519u) ^ ((uint)(ordinal + 1) * 3266489917u)));
            AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(distinctSeed);
        }

        return criticArchitecture.CloneForModelConstruction();
    }

    /// <summary>Clamped log standard deviations, as plain doubles.</summary>
    private double[] CurrentLogStandardDeviations()
    {
        var logStds = new double[_logStd.Length];
        for (int i = 0; i < logStds.Length; i++)
        {
            logStds[i] = Math.Clamp(NumOps.ToDouble(_logStd[i]), MinLogStandardDeviation, MaxLogStandardDeviation);
        }

        return logStds;
    }

    private static Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        var combined = new Vector<T>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++) combined[i] = state[i];
        for (int i = 0; i < action.Length; i++) combined[state.Length + i] = action[i];
        return combined;
    }

    #endregion

    #region Action Selection

    /// <summary>
    /// Standard deviation the policy's learned spread starts from, so untrained behaviour is unchanged.
    /// </summary>
    private const double InitialExplorationStandardDeviation = 0.1;

    /// <summary>Lower clamp on log-std, keeping the policy from collapsing to a delta (and log-prob to -inf).</summary>
    private const double MinLogStandardDeviation = -20.0;

    /// <summary>Upper clamp on log-std, keeping exploration from diverging.</summary>
    private const double MaxLogStandardDeviation = 2.0;

    /// <summary>Action-space ascent step used to turn the deterministic policy gradient into a regression target.</summary>
    private const double ActorPolicyGradientStep = 0.05;

    /// <summary>Central-difference half-width used to estimate the critic's action sensitivity.</summary>
    private const double ActionFiniteDifferenceEpsilon = 1e-3;

    /// <summary>Step size for the log-std ascent and the temperature update.</summary>
    private const double PolicySpreadLearningRate = 1e-3;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The actor has a single deterministic head (one output per action dimension, no log-std head), so
    /// there is no policy Gaussian to reparameterize. Training-mode exploration therefore adds independent
    /// zero-mean Gaussian noise, <c>a = mu(s) + 0.1 * eps, eps ~ N(0, I)</c>, drawn from the agent's seeded
    /// random stream. The previous noise was <c>U[0, 0.1)</c> — mean +0.05 and never negative — which biased
    /// every exploratory position long and, because the actor is regressed onto the actions it took, pushed
    /// the policy's output upward on every update.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> While training, the agent jitters its chosen position sizes a little in both
    /// directions so it can discover better ones; at evaluation time it uses the actor's output as-is.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _actor.Predict(Tensor<T>.FromVector(state)).ToVector();

        return training
            ? AddGaussianExplorationNoise(action, CurrentPolicyStandardDeviations)
            : action;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, Train performs a training step. This updates the FinancialSACAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        // A supervised one-shot Train(state, target) call bypasses the autonomous-exploration batch
        // gate and trains on the samples gathered so far (clamped to the buffer); autonomous stepping
        // still requires a full minibatch before updating.
        int effectiveBatchSize = SupervisedUpdateRequested
            ? System.Math.Min(TradingOptions.BatchSize, ReplayBuffer.Count)
            : TradingOptions.BatchSize;
        if (effectiveBatchSize <= 0 || ReplayBuffer.Count < effectiveBatchSize) return NumOps.Zero;

        // TradingAgentOptions.WarmupSteps: collect this many transitions before the first update.
        if (IsInWarmup(ReplayBuffer.Count)) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(effectiveBatchSize);
        int n = batch.Count;
        if (n == 0) return NumOps.Zero;

        // Every network pass below is batched over the whole minibatch rather than run per experience
        // (the per-sample loop dominated RL training time — see profiling).
        int stateDim = batch[0].State.Length;
        int actionDim = batch[0].Action.Length;

        var statesData = new T[n * stateDim];
        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            for (int j = 0; j < stateDim; j++)
            {
                statesData[i * stateDim + j] = exp.State[j];
            }
        }

        var nextStatesData = new T[n * stateDim];
        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            for (int j = 0; j < stateDim; j++)
            {
                nextStatesData[i * stateDim + j] = exp.NextState[j];
            }
        }

        var states = new Tensor<T>([n, stateDim], new Vector<T>(statesData));
        var nextStates = new Tensor<T>([n, stateDim], new Vector<T>(nextStatesData));

        int stateActionDim = stateDim + actionDim;
        var logStds = CurrentLogStandardDeviations();
        double alpha = CurrentAlpha;
        double gamma = Convert.ToDouble(TradingOptions.DiscountFactor);

        // ---- 1. Soft TD target (no gradients: the target critics are not trained) ----
        //   a' ~ pi(.|s') = N(mu(s'), diag(exp(logStd)^2))
        //   y  = r + gamma * (1 - done) * ( min(Q1'(s',a'), Q2'(s',a')) - alpha * log pi(a'|s') )
        // The entropy term is what makes this SAC rather than TD3: the bootstrap is the SOFT value, so
        // the critics learn the value of acting AND staying stochastic.
        var nextMeans = _actor.Predict(nextStates).ToVector();
        var nextStateActionsData = new T[n * stateActionDim];
        var nextLogProbs = new double[n];
        for (int i = 0; i < n; i++)
        {
            var nextMean = new Vector<T>(actionDim);
            var nextAction = new Vector<T>(actionDim);
            for (int j = 0; j < actionDim; j++)
            {
                nextMean[j] = nextMeans[(i * actionDim) + j];
                double noise = Math.Exp(logStds[j]) * NextStandardNormal();
                nextAction[j] = NumOps.Add(nextMean[j], NumOps.FromDouble(noise));
            }

            nextLogProbs[i] = GaussianLogProbability(nextAction, nextMean, logStds);
            for (int j = 0; j < stateDim; j++)
            {
                nextStateActionsData[(i * stateActionDim) + j] = batch[i].NextState[j];
            }

            for (int j = 0; j < actionDim; j++)
            {
                nextStateActionsData[(i * stateActionDim) + stateDim + j] = nextAction[j];
            }
        }

        var nextStateActions = new Tensor<T>([n, stateActionDim], new Vector<T>(nextStateActionsData));
        var targetQ1 = _targetCritic1.Predict(nextStateActions).ToVector();
        var targetQ2 = _targetCritic2.Predict(nextStateActions).ToVector();

        var stateActionsData = new T[n * stateActionDim];
        var targetData = new T[n];
        for (int i = 0; i < n; i++)
        {
            double q1 = NumOps.ToDouble(targetQ1[i]);
            double q2 = NumOps.ToDouble(targetQ2[i]);
            double softValue = Math.Min(q1, q2) - (alpha * nextLogProbs[i]);
            double bootstrap = batch[i].Done ? 0.0 : gamma * softValue;
            targetData[i] = NumOps.Add(batch[i].Reward, NumOps.FromDouble(bootstrap));

            for (int j = 0; j < stateDim; j++)
            {
                stateActionsData[(i * stateActionDim) + j] = batch[i].State[j];
            }

            for (int j = 0; j < actionDim; j++)
            {
                stateActionsData[(i * stateActionDim) + stateDim + j] = batch[i].Action[j];
            }
        }

        var stateActions = new Tensor<T>([n, stateActionDim], new Vector<T>(stateActionsData));
        var targets = new Tensor<T>([n, 1], new Vector<T>(targetData));

        // ---- 2. Train BOTH critics on the same target ----
        // They differ only by their independent initialization, which is exactly what makes min(Q1, Q2)
        // a pessimistic estimate instead of a redundant one.
        _critic1.Train(stateActions, targets);
        T critic1Loss = _critic1.GetLastLoss();
        _critic2.Train(stateActions, targets);
        T critic2Loss = _critic2.GetLastLoss();

        // ---- 3. Actor update: ascend min(Q1, Q2) at a = mu(s) ----
        // The deterministic policy gradient is grad_theta J = E[ grad_a Q(s,a)|a=mu(s) * grad_theta mu(s) ].
        // grad_a Q is estimated by central differences over the ONLINE critics (the library's DDPG agent
        // uses the same construction), turned into a regression target a + step * grad_a Q, and realised
        // through the actor's own MSE step, which supplies grad_theta mu by backpropagation.
        var means = _actor.Predict(states).ToVector();
        var actionGradients = EstimateActionGradients(batch, means, n, stateDim, actionDim, stateActionDim);

        T maxPosition = TradingOptions.MaxPositionSize;
        T minPosition = NumOps.Negate(maxPosition);
        var actorTargetData = new T[n * actionDim];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < actionDim; j++)
            {
                int flat = (i * actionDim) + j;
                T ascended = NumOps.Add(
                    means[flat],
                    NumOps.FromDouble(ActorPolicyGradientStep * actionGradients[flat]));
                actorTargetData[flat] = MathHelper.Clamp<T>(ascended, minPosition, maxPosition);
            }
        }

        var actorTargets = new Tensor<T>([n, actionDim], new Vector<T>(actorTargetData));
        _actor.Train(states, actorTargets);
        T actorLoss = _actor.GetLastLoss();

        // ---- 4. Policy spread: ascend Q + alpha * H over log-std ----
        // With the reparameterization a = mu + exp(logStd) * eps,
        //   d/dlogStd_j  E[Q] = E[ dQ/da_j * exp(logStd_j) * eps_j ]   and   d/dlogStd_j  H = 1.
        // Estimating the first term at eps = 0 leaves the entropy term, which is the part that actually
        // trades spread against value; alpha sets the exchange rate.
        UpdatePolicySpread(actionGradients, logStds, alpha, n, actionDim);

        // ---- 5. Temperature ----
        if (TradingOptions.AutoTuneAlpha)
        {
            UpdateTemperature(nextLogProbs, actionDim);
        }

        // ---- 6. Polyak target updates ----
        double tau = TradingOptions.Tau;
        SoftUpdateTargetNetwork(_critic1, _targetCritic1, tau);
        SoftUpdateTargetNetwork(_critic2, _targetCritic2, tau);

        if (_updateCount < int.MaxValue)
        {
            _updateCount++;
        }

        T criticLoss = NumOps.Divide(NumOps.Add(critic1Loss, critic2Loss), NumOps.FromDouble(2.0));
        T loss = NumOps.Add(criticLoss, actorLoss);
        LossHistory.Add(loss);
        return loss;
    }

    /// <summary>
    /// Central-difference estimate of <c>d min(Q1, Q2) / da</c> at each sampled state, evaluated at the
    /// actor's current mean action. Returns a flat <c>[n * actionDim]</c> buffer of doubles.
    /// </summary>
    /// <remarks>
    /// Every perturbation is packed into ONE batched forward per critic per direction rather than a
    /// forward per element, so the cost is 2 * actionDim batched passes instead of 2 * n * actionDim
    /// single-sample ones.
    /// </remarks>
    private double[] EstimateActionGradients(
        List<Experience<T>> batch, Vector<T> means, int n, int stateDim, int actionDim, int stateActionDim)
    {
        var gradients = new double[n * actionDim];
        double epsilon = ActionFiniteDifferenceEpsilon;

        for (int dimension = 0; dimension < actionDim; dimension++)
        {
            var plusData = new T[n * stateActionDim];
            var minusData = new T[n * stateActionDim];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < stateDim; j++)
                {
                    plusData[(i * stateActionDim) + j] = batch[i].State[j];
                    minusData[(i * stateActionDim) + j] = batch[i].State[j];
                }

                for (int j = 0; j < actionDim; j++)
                {
                    T mean = means[(i * actionDim) + j];
                    int slot = (i * stateActionDim) + stateDim + j;
                    plusData[slot] = j == dimension ? NumOps.Add(mean, NumOps.FromDouble(epsilon)) : mean;
                    minusData[slot] = j == dimension ? NumOps.Subtract(mean, NumOps.FromDouble(epsilon)) : mean;
                }
            }

            var plus = new Tensor<T>([n, stateActionDim], new Vector<T>(plusData));
            var minus = new Tensor<T>([n, stateActionDim], new Vector<T>(minusData));
            var plusQ1 = _critic1.Predict(plus).ToVector();
            var plusQ2 = _critic2.Predict(plus).ToVector();
            var minusQ1 = _critic1.Predict(minus).ToVector();
            var minusQ2 = _critic2.Predict(minus).ToVector();

            for (int i = 0; i < n; i++)
            {
                double forward = Math.Min(NumOps.ToDouble(plusQ1[i]), NumOps.ToDouble(plusQ2[i]));
                double backward = Math.Min(NumOps.ToDouble(minusQ1[i]), NumOps.ToDouble(minusQ2[i]));
                gradients[(i * actionDim) + dimension] = (forward - backward) / (2.0 * epsilon);
            }
        }

        return gradients;
    }

    /// <summary>
    /// One ascent step on the state-independent log standard deviations against <c>Q + alpha * H</c>.
    /// </summary>
    private void UpdatePolicySpread(double[] actionGradients, double[] logStds, double alpha, int n, int actionDim)
    {
        for (int j = 0; j < actionDim; j++)
        {
            // Mean |dQ/da_j| measures how sharply value depends on this dimension; widening a dimension the
            // critic is sensitive to costs value, so that term pushes the spread DOWN while entropy pushes up.
            double sensitivity = 0.0;
            for (int i = 0; i < n; i++)
            {
                sensitivity += Math.Abs(actionGradients[(i * actionDim) + j]);
            }

            sensitivity /= Math.Max(1, n);

            double std = Math.Exp(logStds[j]);
            double gradient = alpha - (sensitivity * std);
            double updated = Math.Clamp(
                logStds[j] + (PolicySpreadLearningRate * gradient),
                MinLogStandardDeviation,
                MaxLogStandardDeviation);
            _logStd[j] = NumOps.FromDouble(updated);
        }
    }

    /// <summary>
    /// Automatic temperature tuning (Haarnoja et al. 2018 §5): descend
    /// <c>L(alpha) = -alpha * (log pi + H_target)</c>, so alpha rises while the policy is more
    /// deterministic than the target entropy and falls once it is more random.
    /// </summary>
    private void UpdateTemperature(double[] logProbs, int actionDim)
    {
        double averageLogProb = 0.0;
        for (int i = 0; i < logProbs.Length; i++)
        {
            averageLogProb += logProbs[i];
        }

        averageLogProb /= Math.Max(1, logProbs.Length);

        // Target entropy defaults to -ActionSize (the SAC convention); TargetEntropyRatio scales it.
        double ratio = TradingOptions is FinancialSACAgentOptions<T> sacOptions
            ? sacOptions.TargetEntropyRatio
            : -1.0;
        double targetEntropy = ratio * actionDim;

        double logAlpha = NumOps.ToDouble(_logAlpha);
        double gradient = -Math.Exp(logAlpha) * (averageLogProb + targetEntropy);
        logAlpha -= PolicySpreadLearningRate * gradient;
        _logAlpha = NumOps.FromDouble(Math.Clamp(logAlpha, -20.0, 4.0));
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, ScaleReward(reward), nextState, done);
        ReplayBuffer.Add(experience);
    }

    #endregion

    #region Serialization

    #endregion

    #region Model Metadata

    /// <inheritdoc/>
    /// <remarks>
    /// Adds the entropy temperature (<c>"Alpha"</c>), the mean policy standard deviation
    /// (<c>"PolicyStdDev"</c>) and the number of gradient updates applied (<c>"Updates"</c>), so a caller can
    /// see the exploration/exploitation trade-off the agent has actually settled on.
    /// </remarks>
    public override Dictionary<string, T> GetTradingMetrics()
    {
        var metrics = base.GetTradingMetrics();
        metrics["Alpha"] = NumOps.FromDouble(CurrentAlpha);
        metrics["PolicyStdDev"] = NumOps.FromDouble(CurrentPolicyStandardDeviations.Average());
        metrics["Updates"] = NumOps.FromDouble(_updateCount);
        return metrics;
    }

    /// <summary>
    /// Executes GetModelMetadata for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialSAC" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
