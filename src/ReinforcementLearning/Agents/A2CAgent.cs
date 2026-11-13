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
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.ReinforcementLearning.Agents.A2C;

/// <summary>
/// Advantage Actor-Critic (A2C) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A2C is a synchronous, simpler version of A3C that combines policy gradients with value
/// function learning. It's the foundation for many modern RL algorithms including PPO.
/// </para>
/// <para><b>For Beginners:</b>
/// A2C learns two networks simultaneously:
/// - **Actor**: Decides which action to take (policy)
/// - **Critic**: Evaluates how good the current state is (value function)
///
/// The critic helps the actor learn faster by providing better feedback than rewards alone.
/// Think of it like having a coach (critic) give you targeted advice instead of just
/// saying "good" or "bad" after the game ends.
///
/// A2C is simpler than PPO but still very effective. Good starting point for actor-critic methods.
/// </para>
/// <para><b>Reference:</b>
/// Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016 (describes A3C, A2C is the synchronous version).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Advantage Actor-Critic agent for discrete action spaces
/// var options = new A2COptions&lt;double&gt; { StateSize = 4, ActionSize = 2, PolicyLearningRate = 0.001 };
/// var agent = new A2CAgent&lt;double&gt;(options);
///
/// // Select an action given the current environment state
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
[ResearchPaper("Asynchronous Methods for Deep Reinforcement Learning",
    "https://arxiv.org/abs/1602.01783",
    Year = 2016,
    Authors = "Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K.")]
public partial class A2CAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_policyNetwork);
        RegisterParameterComponent(_valueNetwork);
    }
    private A2COptions<T> _a2cOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _a2cOptions;
    private readonly Trajectory<T> _trajectory;

    // s_{t+1} of the latest stored step: the n-step returns bootstrap V(s_T) from it when the rollout ends
    // mid-episode.
    private Vector<T>? _bootstrapState;

    private INeuralNetwork<T> _policyNetwork;
    private INeuralNetwork<T> _valueNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _a2cOptions.StateSize;

    private static ReinforcementLearningOptions<T> CreateBaseOptions(A2COptions<T> options)
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.PolicyLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed
        };
    }

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public A2CAgent()
        : this(new A2COptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public A2CAgent(A2COptions<T> options)
        : base(CreateBaseOptions(options))
    {
        _a2cOptions = options;
        _trajectory = new Trajectory<T>();

        _policyNetwork = BuildPolicyNetwork();
        _valueNetwork = BuildValueNetwork();

        Networks.Add(_policyNetwork);
        Networks.Add(_valueNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _a2cOptions.StateSize;

        foreach (var hiddenSize in _a2cOptions.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        int outputSize = _a2cOptions.IsContinuous ? _a2cOptions.ActionSize * 2 : _a2cOptions.ActionSize;
        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _a2cOptions.StateSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_a2cOptions.PolicyLearningRate),
            options: new NeuralNetworkDefaultOptions { Seed = _a2cOptions.Seed });
    }

    private NeuralNetwork<T> BuildValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _a2cOptions.StateSize;

        foreach (var hiddenSize in _a2cOptions.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _a2cOptions.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_a2cOptions.ValueLearningRate),
            lossFunction: _a2cOptions.ValueLossFunction,
            options: new NeuralNetworkDefaultOptions { Seed = _a2cOptions.Seed is int seed ? unchecked(seed + 1) : null });
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_a2cOptions.IsContinuous)
        {
            return SampleContinuousAction(policyOutput, training);
        }
        else
        {
            return SampleDiscreteAction(policyOutput, training);
        }
    }

    private Vector<T> SampleDiscreteAction(Vector<T> logits, bool training)
    {
        var probs = Softmax(logits);

        if (training)
        {
            // Training: sample a discrete action and return a one-hot
            // vector so StoreExperience records the commitment.
            int actionIndex = SampleCategorical(probs);
            var action = new Vector<T>(_a2cOptions.ActionSize);
            action[actionIndex] = NumOps.One;
            return action;
        }

        // Inference: return a one-hot argmax(π) action. The IRLAgent<T>
        // public contract on discrete envs is "give me the action to take"
        // — callers feed the result directly into env.Step(action).
        // Returning a raw softmax distribution would change the API from
        // action-selector to policy-diagnostic and break evaluation loops
        // that expect a deterministic action. The full π(·|s) is still
        // observable via the underlying policy network's Predict.
        int bestIdx = probs.Length == 0 ? 0 : RLTensorOps.ArgMax(probs);
        var oneHot = new Vector<T>(_a2cOptions.ActionSize);
        oneHot[bestIdx] = NumOps.One;
        return oneHot;
    }

    private Vector<T> SampleContinuousAction(Vector<T> output, bool training)
    {
        // Gaussian head [mean | log_std]: mean + exp(log_std) * N(0, 1) when training, the mean otherwise.
        return RLTensorOps.GaussianHeadAction(output, _a2cOptions.ActionSize, training, Random);
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

        if (_a2cOptions.IsContinuous)
        {
            // sum_i [-0.5 log(2 pi) - log sigma_i - 0.5 (a_i - mu_i)^2 / sigma_i^2], sigma = exp(log_std)
            int actionSize = _a2cOptions.ActionSize;
            var engine = AiDotNetEngine.Current;
            var mean = policyOutput.GetSubVector(0, actionSize);
            var logStd = policyOutput.GetSubVector(actionSize, actionSize);
            var std = engine.Exp(logStd);
            var diff = engine.Subtract(RLTensorOps.Head(action, actionSize), mean);
            var variance = engine.Multiply(std, std);
            var squaredTerm = engine.Multiply(engine.Divide(engine.Multiply(diff, diff), variance), NumOps.FromDouble(0.5));
            var perDimension = engine.Subtract(
                engine.Subtract(Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(-0.5 * Math.Log(2 * Math.PI))), logStd),
                squaredTerm);
            return engine.Sum(perDimension);
        }
        else
        {
            var probs = Softmax(policyOutput);
            int actionIndex = ArgMax(action);
            // log(p + 1e-10) in double, as before: 1e-10 rounds to zero in Half.
            return RLTensorOps.LogWithFloor(probs[actionIndex], 1e-10);
        }
    }

    /// <inheritdoc/>
    public override T Train()
    {
        // A2C trains after fixed number of steps
        if (_trajectory.Length < _a2cOptions.StepsPerUpdate)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute returns and advantages
        ComputeAdvantages();

        // Build batched tensors from trajectory
        int stateSize = _a2cOptions.StateSize;
        int trajLen = _trajectory.Length;
        var advantages = _trajectory.Advantages ?? throw new InvalidOperationException("Advantages not initialized.");
        var returns = _trajectory.Returns ?? throw new InvalidOperationException("Returns not initialized.");

        // [trajLen, stateSize] states and [trajLen, 1] returns, stacked in bulk.
        var batchStates = RLTensorOps.StackRows(_trajectory.States.Select(stateRow => RLTensorOps.Head(stateRow, stateSize)).ToList());
        var batchReturns = RLTensorOps.FromList(returns, trajLen, 1);

        // --- Update value network (MSE regression on returns) ---
        _valueNetwork.Train(batchStates, batchReturns);
        T valueLoss = _valueNetwork.GetLastLoss();

        // Build advantages tensor and action data for engine ops
        var advantagesTensor = RLTensorOps.FromList(advantages, trajLen);

        int[] actionIndices = _a2cOptions.IsContinuous ? Array.Empty<int>() : new int[trajLen];
        Tensor<T>? actionsTensor = null;

        if (_a2cOptions.IsContinuous)
        {
            int actionSize = _a2cOptions.ActionSize;
            actionsTensor = RLTensorOps.StackRows(_trajectory.Actions.Select(actionRow => RLTensorOps.Head(actionRow, actionSize)).ToList());
        }
        else
        {
            for (int i = 0; i < trajLen; i++)
                actionIndices[i] = ArgMax(_trajectory.Actions[i]);
        }

        // --- Update policy network (policy gradient via engine ops) ---
        var trainablePolicy = (NeuralNetworkBase<T>)_policyNetwork;
        T policyLoss = trainablePolicy.TrainWithCustomLoss(batchStates, policyOutput =>
        {
            Tensor<T> logProbs;
            Tensor<T> entropy;
            if (_a2cOptions.IsContinuous)
            {
                int actSize = _a2cOptions.ActionSize;
                var means = Engine.TensorSlice(policyOutput, [0, 0], [trajLen, actSize]);
                // TensorSlice takes (start, length): the log-std half is actSize wide, not the full output width.
                var logStds = Engine.TensorSlice(policyOutput, [0, actSize], [trajLen, actSize]);
                logProbs = PolicyDistributionHelper<T>.ComputeGaussianLogProb(Engine, means, logStds, (actionsTensor ?? throw new InvalidOperationException("actionsTensor is null where it is required.")));
                entropy = PolicyDistributionHelper<T>.ComputeGaussianEntropy(Engine, logStds);
            }
            else
            {
                logProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(Engine, policyOutput, actionIndices);
                entropy = PolicyDistributionHelper<T>.ComputeDiscreteEntropy(Engine, policyOutput);
            }

            // loss = -mean(logProbs * advantages)
            // Entropy regularisation (Mnih et al. 2016, §4): loss = −mean(log π·A) − β·mean(H(π)).
            var weighted = Engine.TensorMultiply(logProbs, advantagesTensor);
            var allAxes = Enumerable.Range(0, weighted.Shape.Length).ToArray();
            var mean = Engine.ReduceMean(weighted, allAxes, keepDims: false);
            var meanEntropy = Engine.ReduceMean(entropy, Enumerable.Range(0, entropy.Shape.Length).ToArray(), keepDims: false);
            return Engine.TensorNegate(Engine.TensorAdd(mean, Engine.TensorMultiplyScalar(meanEntropy, _a2cOptions.EntropyCoefficient)));
        });

        // Reported loss: the two losses each network minimised. They are trained by separate optimizers,
        // so no coefficient weights one against the other.
        var totalLoss = NumOps.Add(policyLoss, valueLoss);

        LossHistory.Add(totalLoss);
        _trajectory.Clear();
        _bootstrapState = null;

        return totalLoss;
    }

    private void ComputeAdvantages()
    {
        // A rollout that stops mid-episode truncates its last step there: R_T = V(s_T) of the stored next state
        // (Mnih et al. 2016, Algorithm S3; Stable-Baselines3's last_values).
        if (_bootstrapState is not null)
        {
            _trajectory.TruncateLastStep(RLTensorOps.FirstValue(_valueNetwork.Predict(Tensor<T>.FromVector(_bootstrapState))));
        }

        // n-step returns: R_t = r_t + gamma R_{t+1}, restarted after a terminal step (nothing bootstrapped) and after a
        // truncated one (V(s_{t+1}) bootstrapped, as SB3 and CleanRL do on truncation).
        var returns = RLTensorOps.DiscountedReturns(
            Vector<T>.FromEnumerable(_trajectory.Rewards), _trajectory.NextValues(), _trajectory.ChainCuts(), DiscountFactor);
        var advantages = AiDotNetEngine.Current.Subtract(returns, Vector<T>.FromEnumerable(_trajectory.Values));

        // Normalize advantages: (A - mean) / (std + 1e-8), unbiased std
        _trajectory.Advantages = RLTensorOps.NormalizeAdvantages(advantages.ToList());
        _trajectory.Returns = returns.ToList();
    }

    // UpdateValueNetwork removed — value network now trained via _valueNetwork.Train()
    // which uses the configured optimizer and tape-based gradient computation.

    private T ComputeEntropy(Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();
        var engine = AiDotNetEngine.Current;

        if (_a2cOptions.IsContinuous)
        {
            // sum_i [0.5 log(2 pi e) + log_std_i]
            int actionSize = _a2cOptions.ActionSize;
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
            FeatureCount = _a2cOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
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
            "A2CAgent trains via its own loss inside the Train() method, so there is no single "
            + "parameter-gradient vector for this interface to return. Call Train() instead.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Not directly applicable
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

    private Vector<T> ComputePolicyOutputGradient(Vector<T> policyOutput, Vector<T> action, T advantage)
    {
        var gradient = new Vector<T>(policyOutput.Length);
        var scaledAdvantage = NumOps.Divide(advantage, NumOps.FromDouble(_trajectory.Length));
        var engine = AiDotNetEngine.Current;

        if (_a2cOptions.IsContinuous)
        {
            // Continuous: Gaussian policy [mean, log_std]
            int actionSize = _a2cOptions.ActionSize;
            var mean = policyOutput.GetSubVector(0, actionSize);
            var logStd = policyOutput.GetSubVector(actionSize, actionSize);
            var std = engine.Exp(logStd);
            var actionDiff = engine.Subtract(RLTensorOps.Head(action, actionSize), mean);
            var stdSquared = engine.Multiply(std, std);

            // d/d mu: -(a - mu) / sigma^2 * advantage
            var meanGradient = engine.Negate(engine.Multiply(engine.Divide(actionDiff, stdSquared), scaledAdvantage));

            // d/d log sigma: -((a - mu)^2 / sigma^2 - 1) * advantage
            var normalizedDiff = engine.Divide(actionDiff, std);
            var term = engine.Subtract(engine.Multiply(normalizedDiff, normalizedDiff), Vector<T>.CreateDefault(actionSize, NumOps.One));
            var logStdGradient = engine.Negate(engine.Multiply(term, scaledAdvantage));

            meanGradient.AsSpan().CopyTo(gradient.AsWritableSpan().Slice(0, actionSize));
            logStdGradient.AsSpan().CopyTo(gradient.AsWritableSpan().Slice(actionSize, actionSize));
        }
        else
        {
            // Discrete: softmax policy, -(onehot(a) - pi) * advantage
            var softmax = ComputeSoftmax(policyOutput);
            int selectedAction = GetDiscreteAction(action);
            var indicator = RLTensorOps.OneHot<T>(policyOutput.Length, selectedAction);
            gradient = engine.Negate(engine.Multiply(engine.Subtract(indicator, softmax), scaledAdvantage));
        }

        return gradient;
    }

    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        // exp(z - max z) / sum exp(z - max z)
        return RLTensorOps.StableSoftmax(logits);
    }

    private int GetDiscreteAction(Vector<T> action)
    {
        return RLTensorOps.FirstPositiveIndex(action, NumOps.FromDouble(0.5));
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
