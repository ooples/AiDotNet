using AiDotNet.ActivationFunctions;
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
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Financial Proximal Policy Optimization (PPO) agent for robust trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The PPO (Proximal Policy Optimization) trading agent is one
/// of the most reliable RL algorithms for trading. It prevents the agent from making too
/// large a policy change in any single update, which keeps learning stable. Think of it
/// as a cautious trader who adjusts their strategy gradually rather than making radical
/// shifts. PPO balances exploration (trying new strategies) with exploitation (sticking
/// with what works), making it robust for financial applications.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define actor and critic architectures for PPO trading (30 state features, 5 continuous actions)
/// var actorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 5);
/// var criticArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 1);
///
/// // Create PPO agent for stable, robust trading policy optimization
/// var options = new TradingAgentOptions&lt;double&gt;();
/// var model = new FinancialPPOAgent&lt;double&gt;(actorArch, criticArch, options);
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
public class FinancialPPOAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private const string ObservationNormalizerMarker = "AiDotNet.FinancialPPOAgent.ObservationNormalizer.v1";
    private const double ObservationNormalizerEpsilon = 1e-8;
    private const double ObservationClipRange = 10.0;

    private readonly FinancialPPOAgentOptions<T> _options;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic;
    private readonly Trajectory<T> _trajectory;
    private readonly List<Vector<T>> _nextStates;
    private readonly Random _random;
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;
    private double[]? _observationMean;
    private double[]? _observationM2;
    private long _observationCount;
    private Vector<T>? _lastActionRawState;
    private Vector<T>? _lastActionNormalizedState;
    private Vector<T>? _lastActionVector;
    private Vector<T>? _lastStoredNextRawState;
    private T _lastActionLogProb = default!;
    private bool _hasLastActionLogProb;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override long ParameterCount => _actor.ParameterCount + _critic.ParameterCount;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialPPOAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the value (critic).</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, FinancialPPOAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialPPOAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options as FinancialPPOAgentOptions<T> ?? new FinancialPPOAgentOptions<T>();
        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;

        EnsurePpoDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize);
        EnsurePpoDefaultLayers(criticArchitecture, options.StateSize, 1);

        var actor = new NeuralNetwork<T>(
            actorArchitecture,
            optimizer: CreatePpoOptimizer(null, options),
            lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        var critic = new NeuralNetwork<T>(
            criticArchitecture,
            optimizer: CreatePpoOptimizer(null, options),
            lossFunction: TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        actor.SetBaseTrainOptimizer(CreatePpoOptimizer(actor, options));
        critic.SetBaseTrainOptimizer(CreatePpoOptimizer(critic, options));

        _actor = actor;
        _critic = critic;
        _trajectory = new Trajectory<T>();
        _nextStates = new List<Vector<T>>();
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #endregion

    private static void EnsurePpoDefaultLayers(
        NeuralNetworkArchitecture<T> architecture,
        int expectedInputSize,
        int expectedOutputSize)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        if (architecture.CalculatedInputSize != expectedInputSize)
            throw new ArgumentException($"Architecture input size {architecture.CalculatedInputSize} does not match expected {expectedInputSize}.", nameof(architecture));

        if (architecture.OutputSize != expectedOutputSize)
            throw new ArgumentException($"Architecture output size {architecture.OutputSize} does not match expected {expectedOutputSize}.", nameof(architecture));

        if (architecture.Layers.Count == 0)
        {
            architecture.Layers.Add(new DenseLayer<T>(64, (IActivationFunction<T>)new TanhActivation<T>()));
            architecture.Layers.Add(new DenseLayer<T>(64, (IActivationFunction<T>)new TanhActivation<T>()));
            architecture.Layers.Add(new DenseLayer<T>(expectedOutputSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        }
    }

    private static AdamOptimizer<T, Tensor<T>, Tensor<T>> CreatePpoOptimizer(
        IFullModel<T, Tensor<T>, Tensor<T>>? model,
        TradingAgentOptions<T> options)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            model,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = numOps.ToDouble(options.LearningRate),
                UseAdaptiveBetas = false,
                UseAMSGrad = false
            });
    }

    private Vector<T> NormalizeObservation(Vector<T> state, bool updateStatistics)
    {
        if (updateStatistics)
        {
            UpdateObservationStatistics(state);
        }

        if (_observationMean is null ||
            _observationM2 is null ||
            _observationMean.Length != state.Length ||
            _observationCount == 0)
        {
            return CopyVector(state);
        }

        var normalized = new Vector<T>(state.Length);
        if (_observationCount < 2)
        {
            return normalized;
        }

        double denominator = Math.Max(1, _observationCount - 1);
        for (int i = 0; i < state.Length; i++)
        {
            double variance = _observationM2[i] / denominator;
            double scale = Math.Sqrt(Math.Max(variance, ObservationNormalizerEpsilon));
            double value = (NumOps.ToDouble(state[i]) - _observationMean[i]) / scale;
            value = Math.Max(-ObservationClipRange, Math.Min(ObservationClipRange, value));
            normalized[i] = NumOps.FromDouble(value);
        }

        return normalized;
    }

    private void UpdateObservationStatistics(Vector<T> state)
    {
        EnsureObservationStatistics(state.Length);

        _observationCount++;
        for (int i = 0; i < state.Length; i++)
        {
            double value = NumOps.ToDouble(state[i]);
            double delta = value - _observationMean![i];
            _observationMean[i] += delta / _observationCount;
            double delta2 = value - _observationMean[i];
            _observationM2![i] += delta * delta2;
        }
    }

    private void EnsureObservationStatistics(int observationSize)
    {
        if (_observationMean is not null && _observationMean.Length == observationSize)
        {
            return;
        }

        _observationMean = new double[observationSize];
        _observationM2 = new double[observationSize];
        _observationCount = 0;
    }

    private void CacheSelectedAction(Vector<T> rawState, Vector<T> normalizedState, Vector<T> action, T logProb)
    {
        _lastActionRawState = CopyVector(rawState);
        _lastActionNormalizedState = CopyVector(normalizedState);
        _lastActionVector = CopyVector(action);
        _lastActionLogProb = logProb;
        _hasLastActionLogProb = true;
    }

    private bool TryGetCachedPolicyState(
        Vector<T> state,
        Vector<T> action,
        out Vector<T> normalizedState,
        out T logProb)
    {
        if (!_hasLastActionLogProb ||
            _lastActionRawState is null ||
            _lastActionNormalizedState is null ||
            _lastActionVector is null ||
            !VectorsApproximatelyEqual(_lastActionRawState, state) ||
            !VectorsApproximatelyEqual(_lastActionVector, action))
        {
            normalizedState = null!;
            logProb = NumOps.Zero;
            return false;
        }

        normalizedState = CopyVector(_lastActionNormalizedState);
        logProb = _lastActionLogProb;
        return true;
    }

    private void MarkHiddenEpisodeBoundaryIfNeeded(Vector<T> currentRawState)
    {
        if (_trajectory.Length == 0 || _lastStoredNextRawState is null)
        {
            return;
        }

        int previousIndex = _trajectory.Length - 1;
        if (!_trajectory.Dones[previousIndex] &&
            !VectorsApproximatelyEqual(_lastStoredNextRawState, currentRawState))
        {
            _trajectory.Dones[previousIndex] = true;
        }
    }

    private bool VectorsApproximatelyEqual(Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
        {
            return false;
        }

        for (int i = 0; i < left.Length; i++)
        {
            double a = NumOps.ToDouble(left[i]);
            double b = NumOps.ToDouble(right[i]);
            double tolerance = 1e-6 * Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(b)));
            if (Math.Abs(a - b) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            copy[i] = source[i];
        }

        return copy;
    }

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var normalizedState = NormalizeObservation(state, updateStatistics: training);
        var logits = _actor.Predict(CreateStateTensor(normalizedState)).ToVector();

        if (_options.ContinuousActions)
        {
            CacheSelectedAction(state, normalizedState, logits, NumOps.Zero);
            return logits;
        }

        var probs = Softmax(logits);
        
        if (training)
        {
            int actionIdx = SampleCategorical(probs);
            var action = new Vector<T>(TradingOptions.ActionSize);
            action[actionIdx] = NumOps.One;
            CacheSelectedAction(state, normalizedState, action, LogProbability(probs, actionIdx));
            return action;
        }

        int bestIdx = 0;
        T maxProb = probs[0];
        for (int i = 1; i < probs.Length; i++)
        {
            if (NumOps.GreaterThan(probs[i], maxProb))
            {
                maxProb = probs[i];
                bestIdx = i;
            }
        }

        var result = new Vector<T>(TradingOptions.ActionSize);
        result[bestIdx] = NumOps.One;
        CacheSelectedAction(state, normalizedState, result, LogProbability(probs, bestIdx));
        return result;
    }

    /// <summary>
    /// Executes SampleAction for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SampleAction performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private int SampleCategorical(Vector<T> probabilities)
    {
        double r = _random.NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += NumOps.ToDouble(probabilities[i]);
            if (r < cumulative) return i;
        }
        return probabilities.Length - 1;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Train performs a training step. This updates the FinancialPPOAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        if (_trajectory.Length == 0)
        {
            return NumOps.Zero;
        }

        if (_trajectory.Length < GetMinimumRolloutSize() && !LatestStepIsTerminal())
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        ComputeAdvantagesAndReturns();

        T totalLoss = NumOps.Zero;
        int updateCount = 0;
        int trajectoryLength = _trajectory.Length;
        int epochCount = GetEffectiveEpochCount(trajectoryLength);
        int minibatchCount = GetEffectiveMiniBatchCount(trajectoryLength);
        int minibatchSize = Math.Max(1, (int)Math.Ceiling((double)trajectoryLength / minibatchCount));

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            var indices = Enumerable.Range(0, trajectoryLength)
                .OrderBy(_ => _random.Next())
                .ToArray();

            for (int start = 0; start < indices.Length; start += minibatchSize)
            {
                int count = Math.Min(minibatchSize, indices.Length - start);
                var batchIndices = new int[count];
                Array.Copy(indices, start, batchIndices, 0, count);

                var loss = UpdatePpoMiniBatch(batchIndices);
                totalLoss = NumOps.Add(totalLoss, loss);
                updateCount++;
            }
        }

        var averageLoss = updateCount == 0
            ? NumOps.Zero
            : NumOps.Divide(totalLoss, NumOps.FromDouble(updateCount));
        LossHistory.Add(averageLoss);

        _trajectory.Clear();
        _nextStates.Clear();

        return averageLoss;
    }

    private int GetEffectiveEpochCount(int trajectoryLength)
    {
        int configuredEpochs = Math.Max(1, _options.NumEpochs);
        return trajectoryLength < 8 ? 1 : configuredEpochs;
    }

    private int GetEffectiveMiniBatchCount(int trajectoryLength)
    {
        if (trajectoryLength < 8)
        {
            return 1;
        }

        return Math.Max(1, Math.Min(_options.NumMiniBatches, trajectoryLength));
    }

    private int GetMinimumRolloutSize()
    {
        return Math.Max(1, _options.BatchSize);
    }

    private bool LatestStepIsTerminal()
    {
        return _trajectory.Length > 0 && _trajectory.Dones[^1];
    }

    private T UpdatePpoMiniBatch(int[] batchIndices)
    {
        int n = batchIndices.Length;
        int stateDim = GetBatchStateSize(batchIndices);
        var returns = _trajectory.Returns ?? throw new InvalidOperationException("Returns not initialized.");
        var advantages = _trajectory.Advantages ?? throw new InvalidOperationException("Advantages not initialized.");

        var states = new Tensor<T>([n, stateDim]);
        var targetReturns = new Tensor<T>([n, 1]);
        var oldLogProbs = new Tensor<T>([n]);
        var advantageTensor = new Tensor<T>([n]);

        for (int i = 0; i < n; i++)
        {
            int idx = batchIndices[i];
            CopyStateToBatch(states, i, _trajectory.States[idx], stateDim);

            targetReturns[i, 0] = returns[idx];
            oldLogProbs[i] = _trajectory.LogProbs[idx];
            advantageTensor[i] = advantages[idx];
        }

        _critic.Train(states, targetReturns);
        T valueLoss = _critic.GetLastLoss();

        if (_options.ContinuousActions)
        {
            var actionTargets = new Tensor<T>([n, TradingOptions.ActionSize]);
            for (int i = 0; i < n; i++)
            {
                var action = _trajectory.Actions[batchIndices[i]];
                for (int j = 0; j < TradingOptions.ActionSize; j++)
                {
                    actionTargets[i, j] = action[j];
                }
            }

            _actor.Train(states, actionTargets);
            return NumOps.Add(valueLoss, _actor.GetLastLoss());
        }

        var actionIndices = new int[n];
        for (int i = 0; i < n; i++)
        {
            actionIndices[i] = ArgMax(_trajectory.Actions[batchIndices[i]]);
        }

        var trainableActor = (NeuralNetworkBase<T>)_actor;
        T policyLoss = trainableActor.TrainWithCustomLoss(states, actorOutput =>
        {
            var engine = AiDotNetEngine.Current;
            var newLogProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(engine, actorOutput, actionIndices);
            var logDiff = engine.TensorSubtract(newLogProbs, oldLogProbs);
            var ratio = engine.TensorExp(logDiff);

            var surr1 = engine.TensorMultiply(ratio, advantageTensor);
            var clippedRatio = engine.TensorClamp(
                ratio,
                NumOps.FromDouble(1.0 - TradingOptions.PPOClipRange),
                NumOps.FromDouble(1.0 + TradingOptions.PPOClipRange));
            var surr2 = engine.TensorMultiply(clippedRatio, advantageTensor);

            var minSurr = engine.TensorNegate(
                engine.TensorMax(
                    engine.TensorNegate(surr1),
                    engine.TensorNegate(surr2)));

            var entropy = PolicyDistributionHelper<T>.ComputeDiscreteEntropy(engine, actorOutput);
            var entropyBonus = engine.TensorMultiplyScalar(entropy, NumOps.FromDouble(TradingOptions.EntropyCoefficient));
            var objective = engine.TensorAdd(minSurr, entropyBonus);
            var allAxes = Enumerable.Range(0, objective.Shape.Length).ToArray();
            var meanObjective = engine.ReduceMean(objective, allAxes, keepDims: false);
            return engine.TensorNegate(meanObjective);
        });

        return NumOps.Add(policyLoss, NumOps.Multiply(NumOps.FromDouble(TradingOptions.ValueCoefficient), valueLoss));
    }

    private Tensor<T> CreateStateTensor(Vector<T> normalizedState)
    {
        return Tensor<T>.FromVector(normalizedState);
    }

    private int GetBatchStateSize(int[] batchIndices)
    {
        if (batchIndices.Length == 0)
        {
            return Math.Max(1, TradingOptions.StateSize);
        }

        int stateSize = _trajectory.States[batchIndices[0]].Length;
        for (int i = 1; i < batchIndices.Length; i++)
        {
            int nextSize = _trajectory.States[batchIndices[i]].Length;
            if (nextSize != stateSize)
            {
                throw new InvalidOperationException(
                    $"PPO trajectory contains mixed state sizes ({stateSize} and {nextSize}). " +
                    "Use a stable observation schema before training.");
            }
        }

        return stateSize;
    }

    private static void CopyStateToBatch(Tensor<T> batch, int row, Vector<T> state, int stateSize)
    {
        for (int j = 0; j < stateSize; j++)
        {
            batch[row, j] = state[j];
        }
    }

    private void ComputeAdvantagesAndReturns()
    {
        var advantages = new List<T>(_trajectory.Length);
        var returns = new List<T>(_trajectory.Length);
        T lastGae = NumOps.Zero;
        var gamma = TradingOptions.DiscountFactor;
        var lambda = NumOps.FromDouble(TradingOptions.GAELambda);

        for (int t = _trajectory.Length - 1; t >= 0; t--)
        {
            T nextValue = _trajectory.Dones[t] ? NumOps.Zero : PredictValueFromNormalized(_nextStates[t]);
            var delta = NumOps.Add(
                _trajectory.Rewards[t],
                NumOps.Subtract(NumOps.Multiply(gamma, nextValue), _trajectory.Values[t]));

            lastGae = NumOps.Add(
                delta,
                NumOps.Multiply(
                    NumOps.Multiply(gamma, lambda),
                    _trajectory.Dones[t] ? NumOps.Zero : lastGae));

            advantages.Insert(0, lastGae);
            returns.Insert(0, NumOps.Add(lastGae, _trajectory.Values[t]));
        }

        NormalizeAdvantagesWhenStable(advantages);

        _trajectory.Advantages = advantages;
        _trajectory.Returns = returns;
    }

    private void NormalizeAdvantagesWhenStable(List<T> advantages)
    {
        if (advantages.Count < 8)
        {
            return;
        }

        var stdAdv = StatisticsHelper<T>.CalculateStandardDeviation(advantages);
        if (Math.Abs(NumOps.ToDouble(stdAdv)) <= 1e-8)
        {
            return;
        }

        T meanAdv = NumOps.Zero;
        foreach (var advantage in advantages)
        {
            meanAdv = NumOps.Add(meanAdv, advantage);
        }

        meanAdv = NumOps.Divide(meanAdv, NumOps.FromDouble(advantages.Count));

        for (int i = 0; i < advantages.Count; i++)
        {
            advantages[i] = NumOps.Divide(
                NumOps.Subtract(advantages[i], meanAdv),
                NumOps.Add(stdAdv, NumOps.FromDouble(1e-8)));
        }
    }

    private T PredictValueFromNormalized(Vector<T> normalizedState)
    {
        return _critic.Predict(CreateStateTensor(normalizedState)).ToVector()[0];
    }

    private T ComputeDiscreteLogProbFromNormalized(Vector<T> normalizedState, Vector<T> action)
    {
        var logits = _actor.Predict(CreateStateTensor(normalizedState)).ToVector();
        var probs = Softmax(logits);
        int actionIndex = ArgMax(action);
        return LogProbability(probs, actionIndex);
    }

    private T LogProbability(Vector<T> probabilities, int actionIndex)
    {
        return NumOps.FromDouble(Math.Log(NumOps.ToDouble(probabilities[actionIndex]) + 1e-10));
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.GreaterThan(logits[i], maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        var probabilities = new Vector<T>(logits.Length);
        T sumExp = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            var exp = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(logits[i], maxLogit))));
            probabilities[i] = exp;
            sumExp = NumOps.Add(sumExp, exp);
        }

        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.Divide(probabilities[i], sumExp);
        }

        return probabilities;
    }

    private int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], vector[maxIndex]))
            {
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        MarkHiddenEpisodeBoundaryIfNeeded(state);

        bool usedCachedPolicy = TryGetCachedPolicyState(state, action, out var normalizedState, out var logProb);
        if (!usedCachedPolicy)
        {
            normalizedState = NormalizeObservation(state, updateStatistics: true);
            logProb = _options.ContinuousActions
                ? NumOps.Zero
                : ComputeDiscreteLogProbFromNormalized(normalizedState, action);
        }

        var normalizedNextState = NormalizeObservation(nextState, updateStatistics: false);
        var value = PredictValueFromNormalized(normalizedState);

        _trajectory.AddStep(normalizedState, CopyVector(action), reward, value, logProb, done);
        _nextStates.Add(normalizedNextState);
        _lastStoredNextRawState = CopyVector(nextState);
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Executes Serialize for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Serialize saves or restores model-specific settings. This lets the FinancialPPOAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        var actorData = _actor.Serialize();
        var criticData = _critic.Serialize();
        writer.Write(actorData.Length);
        writer.Write(actorData);
        writer.Write(criticData.Length);
        writer.Write(criticData);
        writer.Write(ObservationNormalizerMarker);
        WriteObservationNormalizer(writer);
        return ms.ToArray();
    }

    /// <summary>
    /// Executes Deserialize for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Deserialize saves or restores model-specific settings. This lets the FinancialPPOAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        int actorLen = reader.ReadInt32();
        _actor.Deserialize(reader.ReadBytes(actorLen));
        int criticLen = reader.ReadInt32();
        _critic.Deserialize(reader.ReadBytes(criticLen));
        if (ms.Position < ms.Length)
        {
            string marker = reader.ReadString();
            if (marker == ObservationNormalizerMarker)
            {
                ReadObservationNormalizer(reader);
            }
        }
    }

    private void WriteObservationNormalizer(BinaryWriter writer)
    {
        writer.Write(_observationCount);
        int length = _observationMean?.Length ?? 0;
        writer.Write(length);

        for (int i = 0; i < length; i++)
        {
            writer.Write(_observationMean![i]);
            writer.Write(_observationM2![i]);
        }
    }

    private void ReadObservationNormalizer(BinaryReader reader)
    {
        _observationCount = reader.ReadInt64();
        int length = reader.ReadInt32();

        _observationMean = length > 0 ? new double[length] : null;
        _observationM2 = length > 0 ? new double[length] : null;

        for (int i = 0; i < length; i++)
        {
            _observationMean![i] = reader.ReadDouble();
            _observationM2![i] = reader.ReadDouble();
        }
    }

    private void CopyObservationNormalizerFrom(FinancialPPOAgent<T> source)
    {
        _observationCount = source._observationCount;
        _observationMean = source._observationMean is null ? null : (double[])source._observationMean.Clone();
        _observationM2 = source._observationM2 is null ? null : (double[])source._observationM2.Clone();
    }

    /// <summary>
    /// Executes GetParameters for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, GetParameters performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var actorParams = _actor.GetParameters();
        var criticParams = _critic.GetParameters();
        var combined = new Vector<T>(actorParams.Length + criticParams.Length);
        
        for (int i = 0; i < actorParams.Length; i++)
            combined[i] = actorParams[i];
            
        for (int i = 0; i < criticParams.Length; i++)
            combined[actorParams.Length + i] = criticParams[i];
            
        return combined;
    }

    /// <summary>
    /// Executes SetParameters for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SetParameters performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int actorCount = checked((int)_actor.ParameterCount);
        _actor.SetParameters(parameters.Slice(0, actorCount));
        _critic.SetParameters(parameters.Slice(actorCount, (int)_critic.ParameterCount));
    }

    #endregion

    #region Model Metadata

    /// <summary>
    /// Executes GetModelMetadata for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Executes Clone for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Clone performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FinancialPPOAgent<T>(_actorArchitecture, _criticArchitecture, TradingOptions);
        clone.SetParameters(GetParameters());
        clone.CopyObservationNormalizerFrom(this);
        return clone;
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(
            CreateStateTensor(NormalizeObservation(input, updateStatistics: false)),
            Tensor<T>.FromVector(target),
            lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
