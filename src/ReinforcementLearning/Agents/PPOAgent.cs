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
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.PPO;

/// <summary>
/// Proximal Policy Optimization (PPO) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PPO is a policy gradient method that uses a clipped surrogate objective to enable
/// multiple epochs of minibatch updates without destructively large policy changes.
/// It achieves state-of-the-art performance across many RL benchmarks while being
/// simpler and more robust than methods like TRPO.
/// </para>
/// <para><b>For Beginners:</b>
/// PPO is one of the most popular RL algorithms today. It's used by:
/// - OpenAI's ChatGPT (for RLHF training)
/// - Many robotics systems
/// - Game AI (including Dota 2 bots)
///
/// Key idea: Make small, safe policy improvements by clipping updates.
/// Think of it like driving - small steering adjustments work better than jerking the wheel.
///
/// PPO learns two things:
/// - A policy (actor): What action to take in each state
/// - A value function (critic): How good each state is
///
/// The critic helps the actor learn more efficiently.
/// </para>
/// <para><b>Reference:</b>
/// Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a PPO agent for stable policy optimization
/// var options = new PPOOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, ClipEpsilon = 0.2 };
/// var agent = new PPOAgent&lt;double&gt;(options);
///
/// // Select an action using the clipped policy
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
[ResearchPaper("Proximal Policy Optimization Algorithms",
    "https://arxiv.org/abs/1707.06347",
    Year = 2017,
    Authors = "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.")]
public partial class PPOAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_policyNetwork);
        RegisterParameterComponent(_valueNetwork);
    }
    private PPOOptions<T> _ppoOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _ppoOptions;
    private readonly Trajectory<T> _trajectory;

    // s_{t+1} of the latest stored step: GAE bootstraps V(s_T) from it when the rollout ends mid-episode.
    private Vector<T>? _bootstrapState;

    private INeuralNetwork<T> _policyNetwork;
    private INeuralNetwork<T> _valueNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _ppoOptions.StateSize;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public PPOAgent()
        : this(new PPOOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    /// <summary>
    /// Initializes a new instance of the PPOAgent class.
    /// </summary>
    /// <param name="options">Configuration options for the PPO agent.</param>
    public PPOAgent(PPOOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.PolicyLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),  // For policy, though we compute custom loss
            Seed = options.Seed,
            BatchSize = options.MiniBatchSize
        })
    {
        Guard.NotNull(options);
        _ppoOptions = options;
        _trajectory = new Trajectory<T>();

        // Build policy network
        _policyNetwork = BuildPolicyNetwork();

        // Build value network
        _valueNetwork = BuildValueNetwork();

        // Register networks with base class
        Networks.Add(_policyNetwork);
        Networks.Add(_valueNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _ppoOptions.StateSize;

        // Hidden layers
        foreach (var hiddenSize in _ppoOptions.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer
        if (_ppoOptions.IsContinuous)
        {
            // For continuous: output mean and log_std for Gaussian policy
            layers.Add(new DenseLayer<T>(_ppoOptions.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));
        }
        else
        {
            // For discrete: output action logits (softmax applied later)
            layers.Add(new DenseLayer<T>(_ppoOptions.ActionSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        }

        int finalOutputSize = _ppoOptions.IsContinuous ? _ppoOptions.ActionSize * 2 : _ppoOptions.ActionSize;
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _ppoOptions.StateSize,
            outputSize: finalOutputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_ppoOptions.PolicyLearningRate, _ppoOptions.MaxGradNorm),
            options: new NeuralNetworkDefaultOptions { Seed = _ppoOptions.Seed });
    }

    private NeuralNetwork<T> BuildValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _ppoOptions.StateSize;

        // Hidden layers
        foreach (var hiddenSize in _ppoOptions.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer (single value)
        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _ppoOptions.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_ppoOptions.ValueLearningRate, _ppoOptions.MaxGradNorm),
            lossFunction: _ppoOptions.ValueLossFunction,
            options: new NeuralNetworkDefaultOptions { Seed = _ppoOptions.Seed is int seed ? unchecked(seed + 1) : null });
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_ppoOptions.IsContinuous)
        {
            // Continuous action space: sample from Gaussian
            return SampleContinuousAction(policyOutput, training);
        }
        else
        {
            // Discrete action space: sample from categorical
            return SampleDiscreteAction(policyOutput, training);
        }
    }

    private Vector<T> SampleDiscreteAction(Vector<T> logits, bool training)
    {
        // Apply softmax to get probabilities
        var probs = Softmax(logits);

        if (training)
        {
            // Training path: sample a discrete action from the categorical
            // distribution and return a one-hot vector so StoreExperience
            // can record the commitment.
            int actionIndex = SampleCategorical(probs);
            var action = new Vector<T>(_ppoOptions.ActionSize);
            action[actionIndex] = NumOps.One;
            return action;
        }

        // Inference path: return a one-hot argmax(π) action. SelectAction's
        // public contract on discrete envs is "give me the action to take"
        // — callers feed the result directly into env.Step(action). Returning
        // a raw softmax distribution here would change the API from
        // action-selector to policy-diagnostic and break any evaluation
        // loop that expects a deterministic action. The full π(·|s) is
        // still observable via the underlying policy network's Predict.
        int bestIdx = probs.Length == 0 ? 0 : RLTensorOps.ArgMax(probs);
        var oneHot = new Vector<T>(_ppoOptions.ActionSize);
        oneHot[bestIdx] = NumOps.One;
        return oneHot;
    }

    private Vector<T> SampleContinuousAction(Vector<T> output, bool training)
    {
        // Gaussian head [mean | log_std]: mean + exp(log_std) * N(0, 1) when training, the mean otherwise.
        return RLTensorOps.GaussianHeadAction(output, _ppoOptions.ActionSize, training, Random);
    }

    /// <inheritdoc/>
    /// <remarks><paramref name="done"/> is termination: nothing is bootstrapped after the step. Use the
    /// terminated/truncated overload when a time limit cut the episode off.</remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
        => StoreExperience(state, action, reward, nextState, terminated: done, truncated: false);

    /// <inheritdoc/>
    /// <remarks>A truncated step bootstraps V(nextState) into its return (Stable-Baselines3 and CleanRL); a
    /// terminated one does not.</remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState,
        bool terminated, bool truncated)
    {
        var value = RLTensorOps.FirstValue(_valueNetwork.Predict(Tensor<T>.FromVector(state)));
        var logProb = ComputeLogProb(state, action);
        bool bootstraps = truncated && !terminated;
        T bootstrapValue = bootstraps
            ? RLTensorOps.FirstValue(_valueNetwork.Predict(Tensor<T>.FromVector(nextState)))
            : NumOps.Zero;
        _trajectory.AddStep(state, action, reward, value, logProb, terminated, bootstraps, bootstrapValue);
        _bootstrapState = nextState?.Clone();
    }

    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_ppoOptions.IsContinuous)
        {
            return ComputeContinuousLogProb(policyOutput, action);
        }
        else
        {
            return ComputeDiscreteLogProb(policyOutput, action);
        }
    }

    private T ComputeDiscreteLogProb(Vector<T> logits, Vector<T> action)
    {
        var probs = Softmax(logits);
        int actionIndex = ArgMax(action);

        // Log probability of selected action
        var prob = probs[actionIndex];
        return NumOps.FromDouble(Math.Log(NumOps.ToDouble(prob) + 1e-10));
    }

    private T ComputeContinuousLogProb(Vector<T> output, Vector<T> action)
    {
        // sum_i [-0.5 log(2 pi) - log sigma_i - 0.5 (a_i - mu_i)^2 / sigma_i^2], sigma = exp(log_std)
        int actionSize = _ppoOptions.ActionSize;
        var engine = AiDotNetEngine.Current;
        var mean = output.GetSubVector(0, actionSize);
        var logStd = output.GetSubVector(actionSize, actionSize);
        var std = engine.Exp(logStd);
        var diff = engine.Subtract(RLTensorOps.Head(action, actionSize), mean);
        var variance = engine.Multiply(std, std);
        var squaredTerm = engine.Multiply(engine.Divide(engine.Multiply(diff, diff), variance), NumOps.FromDouble(0.5));
        var perDimension = engine.Subtract(
            engine.Subtract(Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(-0.5 * Math.Log(2 * Math.PI))), logStd),
            squaredTerm);
        return engine.Sum(perDimension);
    }

    /// <inheritdoc/>
    public override T Train()
    {
        // PPO is on-policy: it trains on a rollout of StepsPerUpdate
        // steps then discards it (Schulman et al. 2017 §6.1, paper
        // default 2048 for Atari / MuJoCo). With an empty trajectory
        // there's nothing to train on; with fewer than StepsPerUpdate
        // but more than zero we run the same PPO update on the partial
        // rollout rather than skipping. This matches reference
        // implementations (Stable Baselines3, RLLib) that flush at
        // episode end even if shorter than the configured horizon, and
        // ensures short test runs (5 StoreExperience + Train cycles)
        // actually exercise the policy/value update.
        if (_trajectory.Length == 0)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute advantages and returns using GAE
        ComputeAdvantages();

        // Train for multiple epochs on collected data
        var losses = new List<T>();
        int numUpdates = 0;

        for (int epoch = 0; epoch < _ppoOptions.TrainingEpochs; epoch++)
        {
            // Shuffle indices for minibatch sampling
            var indices = Enumerable.Range(0, _trajectory.Length).OrderBy(_ => Random.Next()).ToList();

            for (int start = 0; start < _trajectory.Length; start += _ppoOptions.MiniBatchSize)
            {
                int end = Math.Min(start + _ppoOptions.MiniBatchSize, _trajectory.Length);
                var batchIndices = indices.Skip(start).Take(end - start).ToList();

                losses.Add(UpdateNetworks(batchIndices));
                numUpdates++;
            }
        }

        T totalLoss = losses.Count == 0 ? NumOps.Zero : AiDotNetEngine.Current.Sum(Vector<T>.FromEnumerable(losses));
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(numUpdates));
        LossHistory.Add(avgLoss);

        // Clear trajectory for next collection
        _trajectory.Clear();
        _bootstrapState = null;

        return avgLoss;
    }

    private void ComputeAdvantages()
    {
        // A rollout that stops mid-episode truncates its last step there: V(s_T) of the stored next state is
        // bootstrapped (Schulman et al. 2016; Stable-Baselines3's last_values).
        if (_bootstrapState is not null)
        {
            _trajectory.TruncateLastStep(RLTensorOps.FirstValue(_valueNetwork.Predict(Tensor<T>.FromVector(_bootstrapState))));
        }

        // GAE: V(s_{t+1}) is zero after a terminal step and the stored bootstrap value after a truncated one, and the
        // advantage chain stops at both (SB3's next_non_terminal, with truncation bootstrapped as SB3 and CleanRL do).
        var (advantages, returns) = RLTensorOps.Gae(
            Vector<T>.FromEnumerable(_trajectory.Rewards),
            Vector<T>.FromEnumerable(_trajectory.Values),
            _trajectory.NextValues(),
            _trajectory.ChainCuts(),
            DiscountFactor,
            _ppoOptions.GaeLambda);

        // Normalize advantages: (A - mean) / (std + 1e-8), unbiased std
        _trajectory.Advantages = RLTensorOps.NormalizeAdvantages(advantages.ToList());
        _trajectory.Returns = returns.ToList();
    }

    private T UpdateNetworks(List<int> batchIndices)
    {
        int stateSize = _ppoOptions.StateSize;
        int batchCount = batchIndices.Count;
        var advantages = _trajectory.Advantages ?? throw new InvalidOperationException("Advantages not initialized.");
        var returns = _trajectory.Returns ?? throw new InvalidOperationException("Returns not initialized.");

        // Build batched state tensor [batchCount, stateSize]
        // (gathered row by row from the trajectory in minibatch order)
        var batchStates = RLTensorOps.StackRows(batchIndices.Select(idx => RLTensorOps.Head(_trajectory.States[idx], stateSize)).ToList());
        var batchReturns = RLTensorOps.FromList(batchIndices.Select(idx => returns[idx]).ToList(), batchCount, 1);
        var batchActions = batchIndices.Select(idx => _trajectory.Actions[idx]).ToList();

        // --- Update value network (standard MSE regression) ---
        _valueNetwork.Train(batchStates, batchReturns);
        T valueLoss = _valueNetwork.GetLastLoss();

        // Build batched old log-probs and advantages tensors for engine ops
        var oldLogProbsTensor = RLTensorOps.FromList(batchIndices.Select(idx => _trajectory.LogProbs[idx]).ToList(), batchCount);
        var advantagesTensor = RLTensorOps.FromList(batchIndices.Select(idx => advantages[idx]).ToList(), batchCount);

        // Build action indices/tensors for log-prob computation
        int[] actionIndices = _ppoOptions.IsContinuous ? Array.Empty<int>() : batchActions.Select(ArgMax).ToArray();
        Tensor<T>? actionsTensor = null;

        if (_ppoOptions.IsContinuous)
        {
            int actionSize = _ppoOptions.ActionSize;
            actionsTensor = RLTensorOps.StackRows(batchActions.Select(actionRow => RLTensorOps.Head(actionRow, actionSize)).ToList());
        }

        // --- Update policy network (PPO clipped surrogate via engine ops) ---
        var trainablePolicy = (NeuralNetworkBase<T>)_policyNetwork;
        T policyLoss = trainablePolicy.TrainWithCustomLoss(batchStates, policyOutput =>
        {
            // Compute new log-probs using engine ops via PolicyDistributionHelper
            Tensor<T> newLogProbs;
            Tensor<T> entropy;
            if (_ppoOptions.IsContinuous)
            {
                int actSize = _ppoOptions.ActionSize;
                var means = Engine.TensorSlice(policyOutput, [0, 0], [batchCount, actSize]);
                // TensorSlice takes (start, length): the log-std half is actSize wide, not the full output width.
                var logStds = Engine.TensorSlice(policyOutput, [0, actSize], [batchCount, actSize]);
                newLogProbs = PolicyDistributionHelper<T>.ComputeGaussianLogProb(Engine, means, logStds, (actionsTensor ?? throw new InvalidOperationException("actionsTensor is null where it is required.")));
                entropy = PolicyDistributionHelper<T>.ComputeGaussianEntropy(Engine, logStds);
            }
            else
            {
                newLogProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(Engine, policyOutput, actionIndices);
                entropy = PolicyDistributionHelper<T>.ComputeDiscreteEntropy(Engine, policyOutput);
            }

            // ratio = exp(new_log_prob - old_log_prob) — all engine ops
            var logDiff = Engine.TensorSubtract(newLogProbs, oldLogProbsTensor);
            var ratio = Engine.TensorExp(logDiff);

            // surr1 = ratio * advantage
            var surr1 = Engine.TensorMultiply(ratio, advantagesTensor);

            // clipped_ratio = clamp(ratio, 1-eps, 1+eps)
            var clippedRatio = Engine.TensorClamp(ratio,
                NumOps.Subtract(NumOps.One, _ppoOptions.ClipEpsilon),
                NumOps.Add(NumOps.One, _ppoOptions.ClipEpsilon));
            var surr2 = Engine.TensorMultiply(clippedRatio, advantagesTensor);

            // min(surr1, surr2) = -max(-surr1, -surr2)
            var negSurr1 = Engine.TensorNegate(surr1);
            var negSurr2 = Engine.TensorNegate(surr2);
            var negMin = Engine.TensorMax(negSurr1, negSurr2);
            var minSurr = Engine.TensorNegate(negMin);

            // loss = -mean(min(surr1, surr2)) - c2·mean(H(π)) — the clipped surrogate plus the entropy bonus of
            // Schulman et al. 2017, Eq. 9. The value term of that equation is trained on the value network.
            var allAxes = Enumerable.Range(0, minSurr.Shape.Length).ToArray();
            var mean = Engine.ReduceMean(minSurr, allAxes, keepDims: false);
            var meanEntropy = Engine.ReduceMean(entropy, Enumerable.Range(0, entropy.Shape.Length).ToArray(), keepDims: false);
            return Engine.TensorNegate(Engine.TensorAdd(mean, Engine.TensorMultiplyScalar(meanEntropy, _ppoOptions.EntropyCoefficient)));
        });

        // Reported loss: the two losses each network minimised. They are trained by separate optimizers,
        // so no coefficient weights one against the other.
        var totalLoss = NumOps.Add(policyLoss, valueLoss);

        return totalLoss;
    }

    private T ComputeEntropy(Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();
        var engine = AiDotNetEngine.Current;

        if (_ppoOptions.IsContinuous)
        {
            // sum_i [0.5 log(2 pi e) + log_std_i]
            int actionSize = _ppoOptions.ActionSize;
            var logStd = policyOutput.GetSubVector(actionSize, actionSize);
            var perDimension = engine.Add(
                Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI * Math.E))), logStd);
            return engine.Sum(perDimension);
        }
        else
        {
            // -sum p log p over the probabilities above 1e-10
            var probs = Softmax(policyOutput).AsTensor();
            var zeros = engine.TensorFullLike(probs, NumOps.Zero);
            var kept = engine.TensorGreaterThan(probs, NumOps.FromDouble(1e-10));
            var safeProbs = engine.TensorWhere(kept, probs, engine.TensorFullLike(probs, NumOps.One));
            var plogp = engine.TensorWhere(kept, engine.TensorMultiply(probs, engine.TensorLog(safeProbs)), zeros);
            return NumOps.Negate(engine.TensorSum(plogp));
        }
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["TrajectoryLength"] = NumOps.FromDouble(_trajectory.Length);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _ppoOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        // Returned GetParameters() -- the WEIGHTS -- where this interface promises "gradients
        // with respect to all model parameters". The length matches, so nothing downstream could
        // detect the substitution: ApplyGradients subtracts it element-wise, and Elastic Weight
        // Consolidation / Gradient Episodic Memory / Memory Aware Synapses build Fisher-information
        // estimates from it. A distributed trainer averaging these across workers was averaging
        // parameters and calling the result a gradient.
        //
        // This agent trains through its own loss inside Train(), with its networks driven by
        // separate optimizers, so there is no single parameter-gradient vector here.
        throw new NotSupportedException(
            "PPOAgent trains via its own loss inside the Train() method, so there is no single "
            + "parameter-gradient vector for this interface to return. Call Train() instead.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Not directly applicable for PPO
    }

    // Helper methods
    private Vector<T> Softmax(Vector<T> logits)
    {
        // exp(z - max z) / sum exp(z - max z)
        return RLTensorOps.StableSoftmax(logits);
    }

    private int SampleCategorical(Vector<T> probs)
    {
        double rand = Random.NextDouble();
        return RLTensorOps.SampleFromCdf(probs, rand, probs.Length - 1, strict: true);
    }

    private int ArgMax(Vector<T> vector)
    {
        // An empty vector keeps the scan's index-0 default.
        return vector.Length == 0 ? 0 : RLTensorOps.ArgMax(vector);
    }

    private T Max(Vector<T> vector)
    {
        T max = RLTensorOps.Max(vector);
        return max;
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
