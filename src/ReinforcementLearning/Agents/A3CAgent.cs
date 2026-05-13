using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.A3C;

/// <summary>
/// Asynchronous Advantage Actor-Critic (A3C) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A3C runs multiple agents in parallel, each exploring different strategies.
/// Workers periodically synchronize with a global network, enabling diverse exploration
/// without replay buffers.
/// </para>
/// <para><b>For Beginners:</b>
/// A3C is like having multiple students learn simultaneously - each has different
/// experiences, and they periodically share knowledge with a "master" network.
/// This parallel learning provides stability and diverse exploration.
///
/// Key features:
/// - **Asynchronous Updates**: Multiple workers update global network independently
/// - **No Replay Buffer**: On-policy learning with parallel exploration
/// - **Actor-Critic**: Learns both policy and value function
/// - **Diverse Exploration**: Each worker explores differently
///
/// Famous for: DeepMind's breakthrough (2016), enables CPU-only training
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an asynchronous A3C agent with parallel workers
/// var options = new A3COptions&lt;double&gt; { StateSize = 4, ActionSize = 2 };
/// var agent = new A3CAgent&lt;double&gt;(options);
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
public class A3CAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private readonly A3COptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    private INeuralNetwork<T> _globalPolicyNetwork;
    private INeuralNetwork<T> _globalValueNetwork;
    private readonly object _globalLock = new();

    private int _globalSteps;

    /// <summary>
    /// On-policy trajectory buffer. A3C is on-policy — it accumulates a
    /// t_max-step rollout of (s, a, r, s', done) tuples between updates and
    /// never reuses old experience (no replay buffer; see Mnih et al. 2016 §3.2).
    /// <see cref="StoreExperience"/> appends and <see cref="Train"/> drains
    /// this list.
    /// </summary>
    private readonly List<(Vector<T> State, Vector<T> Action, T Reward, Vector<T> NextState, bool Done)> _trajectory = new();

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public A3CAgent()
        : this(new A3COptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public A3CAgent(A3COptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _globalSteps = 0;

        // Initialize networks directly in constructor
        _globalPolicyNetwork = CreatePolicyNetwork();
        _globalValueNetwork = CreateValueNetwork();

        // Register networks with base class
        Networks.Add(_globalPolicyNetwork);
        Networks.Add(_globalValueNetwork);
    }

    private INeuralNetwork<T> CreatePolicyNetwork()
    {
        int outputSize = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;

        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer
        if (_options.IsContinuous)
        {
            layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        }
        else
        {
            layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new SoftmaxActivation<T>()));
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture, lossFunction: _options.ValueLossFunction);
    }

    private INeuralNetwork<T> CreateValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture, lossFunction: _options.ValueLossFunction);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        Vector<T> policyOutput;

        lock (_globalLock)
        {
            var stateTensor = Tensor<T>.FromVector(state);
            var policyOutputTensor = _globalPolicyNetwork.Predict(stateTensor);
            policyOutput = policyOutputTensor.ToVector();
        }

        if (_options.IsContinuous)
        {
            // Continuous action space
            var mean = new Vector<T>(_options.ActionSize);
            var logStd = new Vector<T>(_options.ActionSize);

            for (int i = 0; i < _options.ActionSize; i++)
            {
                mean[i] = policyOutput[i];
                logStd[i] = policyOutput[_options.ActionSize + i];
                logStd[i] = MathHelper.Clamp<T>(logStd[i], NumOps.FromDouble(-20), NumOps.FromDouble(2));
            }

            if (!training)
            {
                return mean;
            }

            var action = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var std = NumOps.Exp(logStd[i]);
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
                action[i] = NumOps.Add(mean[i], NumOps.Multiply(std, noise));
            }

            return action;
        }
        else
        {
            // Discrete action space
            if (!training)
            {
                // Inference path: return a discrete one-hot action by argmax
                // over the softmax distribution. SelectAction's contract on
                // discrete envs is "return the action to take" so callers
                // can pipe the result straight into env.Step(action) — a
                // raw probability vector would change the API from action-
                // selector to policy-diagnostic and break any evaluation
                // loop expecting a deterministic action. The full π(·|s)
                // remains observable via the underlying policy network's
                // Predict for diagnostics that genuinely need it.
                int bestIdx = 0;
                for (int i = 1; i < policyOutput.Length; i++)
                {
                    if (NumOps.GreaterThan(policyOutput[i], policyOutput[bestIdx]))
                        bestIdx = i;
                }
                var oneHot = new Vector<T>(_options.ActionSize);
                oneHot[bestIdx] = NumOps.One;
                return oneHot;
            }

            // Sample from distribution
            double[] probs = new double[_options.ActionSize];
            for (int i = 0; i < _options.ActionSize; i++)
            {
                probs[i] = Convert.ToDouble(NumOps.ToDouble(policyOutput[i]));
            }

            double r = Random.NextDouble();
            double cumulative = 0.0;
            int selectedAction = 0;

            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (r <= cumulative)
                {
                    selectedAction = i;
                    break;
                }
            }

            var actionVec = new Vector<T>(_options.ActionSize);
            actionVec[selectedAction] = NumOps.One;
            return actionVec;
        }
    }

    /// <summary>
    /// Train A3C with parallel workers (simplified for single-threaded execution).
    /// In production, this would spawn actual parallel tasks.
    /// </summary>
    public async Task TrainAsync(Interfaces.IEnvironment<T> environment, int maxSteps)
    {
        // Run workers sequentially to avoid concurrent environment access
        // The environment is not thread-safe, so we cannot run workers in parallel
        // In a full implementation, each worker would need its own environment instance
        for (int i = 0; i < _options.NumWorkers; i++)
        {
            await Task.Run(() => RunWorker(environment, maxSteps, i));
        }
    }

    private void RunWorker(Interfaces.IEnvironment<T> environment, int maxSteps, int workerId)
    {
        // Create worker-local networks (not registered with Networks list)
        var localPolicy = CreatePolicyNetwork();
        var localValue = CreateValueNetwork();

        var trajectory = new List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)>();

        while (_globalSteps < maxSteps)
        {
            // Synchronize with global network
            lock (_globalLock)
            {
                CopyNetworkWeights(_globalPolicyNetwork, localPolicy);
                CopyNetworkWeights(_globalValueNetwork, localValue);
            }

            // Collect trajectory
            var state = environment.Reset();
            trajectory.Clear();

            for (int t = 0; t < _options.TMax && _globalSteps < maxSteps; t++)
            {
                var action = SelectActionWithLocalNetwork(state, localPolicy, training: true);
                var stateTensor = Tensor<T>.FromVector(state);
                var valueTensor = localValue.Predict(stateTensor);
                var value = valueTensor.ToVector()[0];
                var (nextState, reward, done, info) = environment.Step(action);

                trajectory.Add((state, action, reward, done, value));

                state = nextState;
                Interlocked.Increment(ref _globalSteps);

                if (done)
                {
                    break;
                }
            }

            // Compute returns and advantages
            var returns = ComputeReturns(trajectory, localValue);
            var advantages = ComputeAdvantages(trajectory, returns);

            // Update global network
            lock (_globalLock)
            {
                UpdateGlobalNetworks(trajectory, returns, advantages, localPolicy, localValue);
            }
        }
    }

    private Vector<T> SelectActionWithLocalNetwork(Vector<T> state, INeuralNetwork<T> policy, bool training)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = policy.Predict(stateTensor);
        // Simplified: reuse SelectAction logic but with local network output
        // In full implementation, would extract to shared method
        return SelectAction(state, training);
    }

    private List<T> ComputeReturns(List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> trajectory, INeuralNetwork<T> valueNetwork)
    {
        var returns = new List<T>();
        T nextValue = NumOps.Zero;

        if (trajectory.Count > 0 && !trajectory[trajectory.Count - 1].done)
        {
            var lastState = trajectory[trajectory.Count - 1].state;
            var lastStateTensor = Tensor<T>.FromVector(lastState);
            var nextValueTensor = valueNetwork.Predict(lastStateTensor);
            nextValue = nextValueTensor.ToVector()[0];
        }

        T runningReturn = nextValue;
        for (int i = trajectory.Count - 1; i >= 0; i--)
        {
            var exp = trajectory[i];
            if (exp.done)
            {
                runningReturn = exp.reward;
            }
            else
            {
                runningReturn = NumOps.Add(exp.reward, NumOps.Multiply(DiscountFactor, runningReturn));
            }
            returns.Insert(0, runningReturn);
        }

        return returns;
    }

    private List<T> ComputeAdvantages(List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> trajectory, List<T> returns)
    {
        var advantages = new List<T>();

        for (int i = 0; i < trajectory.Count; i++)
        {
            var advantage = NumOps.Subtract(returns[i], trajectory[i].value);
            advantages.Add(advantage);
        }

        // Normalize advantages
        var mean = StatisticsHelper<T>.CalculateMean(advantages.ToArray());
        var std = StatisticsHelper<T>.CalculateStandardDeviation(advantages.ToArray());

        if (NumOps.GreaterThan(std, NumOps.Zero))
        {
            for (int i = 0; i < advantages.Count; i++)
            {
                advantages[i] = NumOps.Divide(NumOps.Subtract(advantages[i], mean), std);
            }
        }

        return advantages;
    }

    private void UpdateGlobalNetworks(
        List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> trajectory,
        List<T> returns,
        List<T> advantages,
        INeuralNetwork<T> localPolicy,
        INeuralNetwork<T> localValue)
    {
        int stateSize = _options.StateSize;
        int trajLen = trajectory.Count;

        // Build batched tensors
        var batchStates = new Tensor<T>([trajLen, stateSize]);
        var batchReturns = new Tensor<T>([trajLen, 1]);
        var advantagesTensor = new Tensor<T>([trajLen]);

        for (int i = 0; i < trajLen; i++)
        {
            for (int j = 0; j < stateSize; j++)
                batchStates[i, j] = trajectory[i].state[j];
            batchReturns[i, 0] = returns[i];
            advantagesTensor[i] = advantages[i];
        }

        // --- Train local value network (MSE on returns) ---
        localValue.Train(batchStates, batchReturns);

        // --- Train local policy network (policy gradient via engine ops) ---
        int[] actionIndices = _options.IsContinuous ? Array.Empty<int>() : new int[trajLen];
        Tensor<T>? actionsTensor = null;

        if (_options.IsContinuous)
        {
            actionsTensor = new Tensor<T>([trajLen, _options.ActionSize]);
            for (int i = 0; i < trajLen; i++)
                for (int j = 0; j < _options.ActionSize; j++)
                    actionsTensor[i, j] = trajectory[i].action[j];
        }
        else
        {
            for (int i = 0; i < trajLen; i++)
            {
                var act = trajectory[i].action;
                int bestIdx = 0;
                for (int k = 1; k < act.Length; k++)
                    if (NumOps.GreaterThan(act[k], act[bestIdx]))
                        bestIdx = k;
                actionIndices[i] = bestIdx;
            }
        }

        var trainablePolicy = (NeuralNetworkBase<T>)localPolicy;
        trainablePolicy.TrainWithCustomLoss(batchStates, policyOutput =>
        {
            Tensor<T> logProbs;
            if (_options.IsContinuous)
            {
                int actSize = _options.ActionSize;
                var means = Engine.TensorSlice(policyOutput, [0, 0], [trajLen, actSize]);
                var logStds = Engine.TensorSlice(policyOutput, [0, actSize], [trajLen, actSize * 2]);
                logProbs = PolicyDistributionHelper<T>.ComputeGaussianLogProb(Engine, means, logStds, actionsTensor!);
            }
            else
            {
                logProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(Engine, policyOutput, actionIndices);
            }

            var weighted = Engine.TensorMultiply(logProbs, advantagesTensor);
            var allAxes = Enumerable.Range(0, weighted.Shape.Length).ToArray();
            var mean = Engine.ReduceMean(weighted, allAxes, keepDims: false);
            return Engine.TensorNegate(mean);
        });

        // Copy local network parameters to global networks (A3C async update)
        _globalPolicyNetwork.UpdateParameters(localPolicy.GetParameters());
        _globalValueNetwork.UpdateParameters(localValue.GetParameters());
    }


    private Vector<T> ComputeA3CPolicyGradient(Vector<T> policyOutput, Vector<T> action, T advantage)
    {
        // A3C uses same policy gradient as A2C: ∇θ log π(a|s) * advantage
        // Supports both continuous (Gaussian) and discrete (Softmax) policies

        if (_options.ActionSize == policyOutput.Length)
        {
            // Discrete action space: softmax policy
            var softmax = ComputeSoftmax(policyOutput);
            var selectedAction = GetDiscreteAction(action);

            var gradient = new Vector<T>(policyOutput.Length);
            for (int i = 0; i < policyOutput.Length; i++)
            {
                var indicator = (i == selectedAction) ? NumOps.One : NumOps.Zero;
                var grad = NumOps.Subtract(indicator, softmax[i]);
                gradient[i] = NumOps.Negate(NumOps.Multiply(advantage, grad));
            }
            return gradient;
        }
        else
        {
            // Continuous action space: Gaussian policy
            int actionDim = policyOutput.Length / 2;
            var gradient = new Vector<T>(policyOutput.Length);

            for (int i = 0; i < actionDim; i++)
            {
                var mean = policyOutput[i];
                var logStd = policyOutput[actionDim + i];
                var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));
                var actionDiff = NumOps.Subtract(action[i], mean);
                var stdSquared = NumOps.Multiply(std, std);

                // ∇mean = -(a - μ) / σ² * advantage
                gradient[i] = NumOps.Negate(
                    NumOps.Multiply(advantage, NumOps.Divide(actionDiff, stdSquared)));

                // ∇log_std = -((a - μ)² / σ² - 1) * advantage
                var stdGrad = NumOps.Subtract(
                    NumOps.Divide(NumOps.Multiply(actionDiff, actionDiff), stdSquared),
                    NumOps.One);
                gradient[actionDim + i] = NumOps.Negate(NumOps.Multiply(advantage, stdGrad));
            }
            return gradient;
        }
    }

    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        var max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (NumOps.GreaterThan(logits[i], max))
                max = logits[i];

        var expSum = NumOps.Zero;
        var exps = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            exps[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(logits[i], max))));
            expSum = NumOps.Add(expSum, exps[i]);
        }

        var softmax = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
            softmax[i] = NumOps.Divide(exps[i], expSum);

        return softmax;
    }

    private int GetDiscreteAction(Vector<T> actionVector)
    {
        // Action vector for discrete actions is one-hot encoded
        int maxIdx = 0;
        T maxVal = actionVector[0];
        for (int i = 1; i < actionVector.Length; i++)
        {
            if (NumOps.GreaterThan(actionVector[i], maxVal))
            {
                maxVal = actionVector[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // UpdateNetworkParameters removed — networks trained via Train()/TrainWithCustomLoss,
    // then local parameters copied to global networks.

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
    }

    /// <summary>
    /// Appends one step of (s, a, r, s', done) experience to the on-policy
    /// trajectory buffer. A3C uses no replay (Mnih et al. 2016 §3.1) — the
    /// buffer is drained on every <see cref="Train"/> call and accumulates
    /// only until the next update.
    /// </summary>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Vector<T> is mutable, and many environments reuse the same
        // state / action / nextState instances across step calls (e.g.,
        // an env that maintains a single observation buffer it overwrites
        // each Step). Without snapshotting, those callers would silently
        // overwrite earlier trajectory entries before Train() consumes
        // them. Take defensive copies so the buffered transitions are
        // immutable from the agent's perspective.
        _trajectory.Add((CloneVector(state), CloneVector(action), reward, CloneVector(nextState), done));
    }

    private static Vector<T> CloneVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) copy[i] = source[i];
        return copy;
    }

    /// <summary>
    /// One A3C update step (Mnih et al. 2016 Algorithm 1, inner loop).
    /// Drains <see cref="_trajectory"/>, bootstraps the return from the
    /// value network on the last non-terminal state, and applies a
    /// REINFORCE-with-baseline / advantage-actor-critic update to both
    /// the policy and value heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The supervised <c>NeuralNetwork.Train(input, target)</c> API takes
    /// (state, target) so we translate the policy-gradient and value
    /// objectives into supervised targets:
    /// </para>
    /// <para>
    /// <b>Value head:</b> target is the n-step discounted return R_t. The
    /// network's MSE loss reduces to <c>(R_t − V(s_t))²</c>, identical to
    /// the paper's value loss (Algorithm 1: <c>∂(R − V)²/∂θ_v</c>).
    /// </para>
    /// <para>
    /// <b>Policy head (discrete, softmax output):</b> the paper's policy
    /// gradient is <c>∇ log π(a|s) · A(s,a)</c>. With softmax output and
    /// categorical cross-entropy on a one-hot action target, the gradient
    /// reduces to <c>π(s) − one_hot(a)</c>. The MSE-on-softmax surrogate
    /// produces gradients in the same direction; scale by signed advantage
    /// so policy probability of <c>a</c> moves toward the one-hot target
    /// when <c>A(s,a) &gt; 0</c> and away when <c>A(s,a) &lt; 0</c>.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        if (_trajectory.Count == 0)
            return NumOps.Zero;

        // Snapshot the step count BEFORE the reverse-pass loop. The final
        // divisor below must use this count, not _trajectory.Count after
        // the loop / Clear() because the buffer is drained and Count
        // would be 0.
        int stepCount = _trajectory.Count;

        // Bootstrap the n-step return. For the terminal step, R_T = 0.
        // For non-terminal, R_T = V(s_T') per the paper's "Receive reward
        // r_t and new state s_{t+1}" loop followed by bootstrap.
        var last = _trajectory[_trajectory.Count - 1];
        T R = last.Done
            ? NumOps.Zero
            : _globalValueNetwork.Predict(Tensor<T>.FromVector(last.NextState)).ToVector()[0];

        // DiscountFactor is T? on the base options class so the compiler
        // can't statically tell that A3COptions's ctor seeded it; fall
        // back to the paper-default γ = 0.99 (Mnih et al. 2016 Table 1).
        T discountFactor = _options.DiscountFactor ?? NumOps.FromDouble(0.99);
        T totalLoss = NumOps.Zero;

        // Walk the trajectory in REVERSE accumulating discounted returns,
        // then apply one supervised update per step. Paper Algorithm 1
        // does `for i ∈ {t-1, ..., t_start}: R ← r_i + γR`.
        // Walk trajectory in reverse to compute returns + advantages, then
        // delegate the actual gradient step to UpdateGlobalNetworks which
        // already implements the paper-correct policy-gradient loss
        // (log π(a|s) · A via PolicyDistributionHelper). The previous
        // per-step supervised-target loop here was a surrogate that
        // (a) interpolated π toward a one-hot target weighted by advantage,
        // and (b) silently mis-typed the continuous-action [mean, logStd]
        // head as discrete probabilities — neither matches the A3C
        // objective in Mnih et al. 2016 Algorithm 1.
        var trajectoryForUpdate = new List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)>(_trajectory.Count);
        var returns = new List<T>(_trajectory.Count);
        var advantages = new List<T>(_trajectory.Count);

        // First pass: forward order, snapshot the value-estimate for each step.
        for (int i = 0; i < _trajectory.Count; i++)
        {
            var step = _trajectory[i];
            var stateTensor = Tensor<T>.FromVector(step.State);
            var valuePred = _globalValueNetwork.Predict(stateTensor).ToVector()[0];
            trajectoryForUpdate.Add((step.State, step.Action, step.Reward, step.Done, valuePred));
        }

        // Second pass: reverse, accumulate discounted returns + per-step advantages.
        // Reset R at every terminal transition (step.Done == true) so a
        // multi-episode trajectory doesn't leak rewards backward across
        // episode boundaries — the discounted-return chain only holds
        // within a single episode. Matches the standard implementation of
        // n-step returns in policy-gradient code (Sutton & Barto §13.7).
        var revReturns = new T[_trajectory.Count];
        var revAdvantages = new T[_trajectory.Count];
        for (int i = _trajectory.Count - 1; i >= 0; i--)
        {
            var step = _trajectory[i];
            R = step.Done
                ? step.Reward
                : NumOps.Add(step.Reward, NumOps.Multiply(discountFactor, R));
            T advantage = NumOps.Subtract(R, trajectoryForUpdate[i].value);
            revReturns[i] = R;
            revAdvantages[i] = advantage;
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(advantage, advantage));
        }
        returns.AddRange(revReturns);
        advantages.AddRange(revAdvantages);

        UpdateGlobalNetworks(trajectoryForUpdate, returns, advantages, _globalPolicyNetwork, _globalValueNetwork);

        _trajectory.Clear();
        _globalSteps++;

        // Average squared advantage as a loss proxy. Paper monitors
        // policy + value loss separately; we return a single scalar to fit
        // the abstract Train() : T contract used across the agent base.
        // Divide by the captured step count (stepCount), not _trajectory
        // .Count — the trajectory has been Clear()ed above so reading
        // _trajectory.Count here would always yield 0.
        return NumOps.Divide(totalLoss, NumOps.FromDouble(Math.Max(1, stepCount)));
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["global_steps"] = NumOps.FromDouble(_globalSteps)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-level state to reset
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    /// <summary>
    /// Async wrapper around the sync <see cref="Train"/> step. The actor-
    /// learner threads of the original A3C paper run <see cref="Train"/>
    /// asynchronously against a shared global network; this wrapper
    /// preserves the API contract for callers that await an async update.
    /// </summary>
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
    public override Vector<T> GetParameters()
    {
        var policyParams = _globalPolicyNetwork.GetParameters();
        var valueParams = _globalValueNetwork.GetParameters();

        var total = policyParams.Length + valueParams.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in policyParams) vector[idx++] = p;
        foreach (var p in valueParams) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var policyParams = _globalPolicyNetwork.GetParameters();
        var valueParams = _globalValueNetwork.GetParameters();

        int idx = 0;
        var policyVec = new Vector<T>(policyParams.Length);
        var valueVec = new Vector<T>(valueParams.Length);

        for (int i = 0; i < policyParams.Length; i++) policyVec[i] = parameters[idx++];
        for (int i = 0; i < valueParams.Length; i++) valueVec[i] = parameters[idx++];

        _globalPolicyNetwork.UpdateParameters(policyVec);
        _globalValueNetwork.UpdateParameters(valueVec);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new A3CAgent<T>(_options, _optimizer);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // A3C uses asynchronous updates - not directly applicable
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_globalSteps);

        var policyBytes = _globalPolicyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        var valueBytes = _globalValueNetwork.Serialize();
        writer.Write(valueBytes.Length);
        writer.Write(valueBytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        _globalSteps = reader.ReadInt32();

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _globalPolicyNetwork.Deserialize(policyBytes);

        var valueLength = reader.ReadInt32();
        var valueBytes = reader.ReadBytes(valueLength);
        _globalValueNetwork.Deserialize(valueBytes);
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
