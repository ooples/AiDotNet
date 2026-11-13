using System;
using System.Linq;
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
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.REINFORCE;

/// <summary>
/// REINFORCE (Monte Carlo Policy Gradient) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// REINFORCE is the simplest and most fundamental policy gradient algorithm. It directly
/// optimizes the policy by following the gradient of expected returns. Despite its simplicity,
/// it forms the foundation for many modern RL algorithms.
/// </para>
/// <para><b>For Beginners:</b>
/// REINFORCE is the "hello world" of policy gradient methods. The algorithm is beautifully simple:
///
/// 1. Play an entire episode
/// 2. Calculate total rewards for each action
/// 3. Make good actions more likely, bad actions less likely
///
/// Think of it like learning to play a game:
/// - You play a round
/// - At the end, you see your score
/// - You adjust your strategy to do better next time
///
/// **Pros**: Simple, works for any problem, easy to understand
/// **Cons**: High variance, slow learning, requires complete episodes
///
/// Modern algorithms like PPO and A2C improve on REINFORCE's core ideas.
/// </para>
/// <para><b>Reference:</b>
/// Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist RL."
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a REINFORCE agent for Monte Carlo policy gradient learning
/// var options = new REINFORCEOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, LearningRate = 0.001 };
/// var agent = new REINFORCEAgent&lt;double&gt;(options);
///
/// // Select an action sampled from the learned policy
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
[ResearchPaper("Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning",
    "https://link.springer.com/article/10.1007/BF00992696",
    Year = 1992,
    Authors = "Williams, R. J.")]
public partial class REINFORCEAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_policyNetwork);
    }
    private REINFORCEOptions<T> _reinforceOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _reinforceOptions;
    private readonly Trajectory<T> _trajectory;

    private readonly INeuralNetwork<T> _policyNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _reinforceOptions.StateSize;

    public REINFORCEAgent() : this(new REINFORCEOptions<T>()) { }

    public REINFORCEAgent(REINFORCEOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.LearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed
        })
    {
        Guard.NotNull(options);
        _reinforceOptions = options;
        _trajectory = new Trajectory<T>();

        _policyNetwork = BuildPolicyNetwork();
        Networks.Add(_policyNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _reinforceOptions.StateSize;

        foreach (var hiddenSize in _reinforceOptions.HiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        int outputSize = _reinforceOptions.IsContinuous
            ? _reinforceOptions.ActionSize * 2  // Mean and log_std for Gaussian
            : _reinforceOptions.ActionSize;      // Logits for categorical

        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _reinforceOptions.StateSize,
            outputSize: outputSize,
            layers: layers)
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture, optimizer: CreateNetworkOptimizer(LearningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_reinforceOptions.IsContinuous)
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
            var action = new Vector<T>(_reinforceOptions.ActionSize);
            action[actionIndex] = NumOps.One;
            return action;
        }

        // Inference: return a discrete one-hot action by argmax over π(·|s).
        // The public SelectAction contract is "give me the action to take"
        // — callers feed the result directly into env.Step(action). Returning
        // a raw probability vector here would change SelectAction from an
        // action selector to a policy-diagnostic API and break evaluation
        // loops that expect a deterministic discrete action.
        int bestIdx = probs.Length == 0 ? 0 : RLTensorOps.ArgMax(probs);
        var deterministic = new Vector<T>(_reinforceOptions.ActionSize);
        deterministic[bestIdx] = NumOps.One;
        return deterministic;
    }

    private Vector<T> SampleContinuousAction(Vector<T> output, bool training)
    {
        // Gaussian head [mean | log_std]: mean + exp(log_std) * N(0, 1) when training, the mean otherwise.
        return RLTensorOps.GaussianHeadAction(output, _reinforceOptions.ActionSize, training, Random);
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // REINFORCE only needs states, actions, and rewards
        var logProb = ComputeLogProb(state, action);
        _trajectory.AddStep(state, action, reward, NumOps.Zero, logProb, done);
    }

    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_reinforceOptions.IsContinuous)
        {
            // sum_i [-0.5 log(2 pi) - log sigma_i - 0.5 (a_i - mu_i)^2 / sigma_i^2], sigma = exp(log_std)
            int actionSize = _reinforceOptions.ActionSize;
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
    /// <remarks>
    /// One REINFORCE update step (Williams 1992 "Simple statistical
    /// gradient-following algorithms for connectionist reinforcement
    /// learning" Eq. 5; Sutton &amp; Barto 2018 §13.3 Algorithm box):
    /// <para><c>θ ← θ + α · Σ_t G_t · ∇_θ log π(a_t | s_t; θ)</c></para>
    /// where <c>G_t</c> is the discounted return from step t and
    /// <c>π(·|s; θ)</c> is the parameterized policy. Implemented via
    /// tape-based autodiff: build a batched log-prob tensor against the
    /// sampled actions, multiply by returns, sum (NOT mean — that
    /// rescales the per-sample gradient by 1/N and isn't the paper's
    /// objective), negate to convert maximization to minimization,
    /// backprop through the policy network.
    /// </remarks>
    public override T Train()
    {
        if (_trajectory.Length == 0)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute discounted returns
        ComputeReturns();

        int stateSize = _reinforceOptions.StateSize;
        int trajLen = _trajectory.Length;
        var returns = _trajectory.Returns ?? throw new InvalidOperationException("Returns not initialized.");

        // [trajLen, stateSize] states and [trajLen] returns, stacked in bulk.
        var batchStates = RLTensorOps.StackRows(_trajectory.States.Select(stateRow => RLTensorOps.Head(stateRow, stateSize)).ToList());
        var returnsTensor = RLTensorOps.FromList(returns, trajLen);

        // Policy gradient objective: maximize Σ_t G_t · log π(a_t | s_t).
        // Minimize the negation. All ops below are engine-routed so the
        // gradient tape captures them through to the policy network's
        // trainable parameters.
        var trainablePolicy = (NeuralNetworkBase<T>)_policyNetwork;
        T avgLoss = trainablePolicy.TrainWithCustomLoss(batchStates, policyOutput =>
        {
            var eng = Engine;
            Tensor<T> logProbs;

            if (_reinforceOptions.IsContinuous)
            {
                int actionSize = _reinforceOptions.ActionSize;
                var means = eng.TensorSlice(policyOutput, [0, 0], [trajLen, actionSize]);
                // TensorSlice takes (start, length): the log-std half is actionSize wide, not the full output width.
                var logStds = eng.TensorSlice(policyOutput, [0, actionSize], [trajLen, actionSize]);

                var actionsTensor = RLTensorOps.StackRows(_trajectory.Actions.Select(actionRow => RLTensorOps.Head(actionRow, actionSize)).ToList());

                logProbs = PolicyDistributionHelper<T>.ComputeGaussianLogProb(eng, means, logStds, actionsTensor);
            }
            else
            {
                var actionIndices = _trajectory.Actions.Take(trajLen).Select(GetDiscreteAction).ToArray();

                logProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(eng, policyOutput, actionIndices);
            }

            // weighted = G_t · log π(a_t|s_t), shape [trajLen]
            var weighted = eng.TensorMultiply(logProbs, returnsTensor);
            // Sum (NOT mean) per the paper's Σ_t — mean would shrink the
            // gradient by 1/N which silently changes the effective
            // learning rate as trajectory length varies.
            var allAxes = Enumerable.Range(0, weighted.Shape.Length).ToArray();
            var summed = eng.ReduceSum(weighted, allAxes, keepDims: false);
            return eng.TensorNegate(summed);
        });

        LossHistory.Add(avgLoss);
        _trajectory.Clear();

        return avgLoss;
    }

    private void ComputeReturns()
    {
        var returns = new List<T>();
        T runningReturn = NumOps.Zero;

        // Compute discounted returns backwards
        for (int t = _trajectory.Length - 1; t >= 0; t--)
        {
            if (_trajectory.Dones[t])
            {
                runningReturn = _trajectory.Rewards[t];
            }
            else
            {
                runningReturn = NumOps.Add(
                    _trajectory.Rewards[t],
                    NumOps.Multiply(DiscountFactor, runningReturn)
                );
            }

            returns.Insert(0, runningReturn);
        }

        // Baseline subtraction + std normalization is a Sutton & Barto §13.4 variance-reduction
        // heuristic, NOT part of the original Williams 1992 paper objective. Apply it only when
        // there are at least two samples AND the spread is non-trivial — otherwise (single-step
        // trajectory, or all-equal returns) the formula collapses every return to zero and kills
        // the gradient signal. Leaving the raw returns alone in that case is still a valid (just
        // higher-variance) Monte Carlo policy-gradient estimator per Williams Eq. 5.
        if (returns.Count > 1)
        {
            var stdReturn = StatisticsHelper<T>.CalculateStandardDeviation(returns);
            if (NumOps.ToDouble(stdReturn) > 1e-6)
            {
                // (G - mean) / (std + 1e-8)
                T meanReturn = RLTensorOps.Average(returns);
                var engine = AiDotNetEngine.Current;
                var normalized = engine.Divide(
                    engine.Subtract(Vector<T>.FromEnumerable(returns), Vector<T>.CreateDefault(returns.Count, meanReturn)),
                    NumOps.Add(stdReturn, NumOps.FromDouble(1e-8)));
                returns = normalized.ToList();
            }
        }

        _trajectory.Returns = returns;
    }

    // UpdatePolicyNetwork removed — policy network now trained via TrainWithCustomLoss
    // which uses the configured optimizer and tape-based gradient computation.

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
            FeatureCount = _reinforceOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Returned GetParameters() — the WEIGHTS — where the contract promises "gradients with
        // respect to all model parameters". The lengths match, so nothing downstream could detect
        // the substitution: ApplyGradients would subtract the weights from themselves, and any
        // Fisher-information consumer would build its estimate out of parameter magnitudes.
        //
        // REINFORCE accumulates its policy-gradient episode-wise inside Train() against returns that
        // are only known at episode end, so a single (input, target) pair does not define a gradient
        // here at all.
        throw new NotSupportedException(
            "REINFORCE computes its policy gradient over a completed episode's returns inside the "
            + "Train() method, so a single input/target pair does not define a parameter gradient "
            + "for this interface to return. Call Train() instead.");
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
