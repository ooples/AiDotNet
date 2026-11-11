using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.TRPO;

/// <summary>
/// Trust Region Policy Optimization (TRPO) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TRPO ensures monotonic improvement by constraining policy updates within a trust region
/// defined by KL divergence. This prevents destructively large updates.
/// </para>
/// <para><b>For Beginners:</b>
/// TRPO is like learning carefully - it never makes changes that are "too big".
/// By limiting how much the policy can change (using KL divergence), it guarantees
/// that performance never degrades (monotonic improvement).
///
/// Key innovations:
/// - **Trust Region**: Constraints on policy change (KL divergence ≤ δ)
/// - **Monotonic Improvement**: Provable performance guarantees
/// - **Conjugate Gradient**: Efficient solution to constrained optimization
/// - **Line Search**: Ensures constraints are satisfied
///
/// Think of it like walking carefully on uncertain terrain - small, safe steps
/// rather than large leaps that might cause you to fall.
///
/// Famous for: OpenAI robotics, predecessor to PPO (which simplified TRPO)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TRPO agent with monotonic improvement guarantees
/// var options = new TRPOOptions&lt;double&gt; { StateSize = 4, ActionSize = 2};
/// var agent = new TRPOAgent&lt;double&gt;(options);
///
/// // Select an action within the trust region
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
[ResearchPaper("Trust Region Policy Optimization",
    "https://arxiv.org/abs/1502.05477",
    Year = 2015,
    Authors = "Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P.")]
public partial class TRPOAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The same components, in the same order, that the hand-written
    /// GetParameters concatenated -- that order is the serialization order.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_policyNetwork);
        RegisterParameterComponent(_valueNetwork);
    }
    private TRPOOptions<T> _options;

    private INeuralNetwork<T> _policyNetwork;
    [Buffer]
    private INeuralNetwork<T> _oldPolicyNetwork;  // For KL divergence
    private INeuralNetwork<T> _valueNetwork;

    private List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _trajectoryBuffer;
    private int _updateCount;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public TRPOAgent()
        : this(new TRPOOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    /// <summary>
    /// Initializes a TRPO agent from <paramref name="options"/>.
    /// </summary>
    /// <param name="options">The agent's configuration.</param>
    /// <remarks>
    /// TRPO takes no optimizer: its policy step is the natural-gradient step solved by conjugate gradient inside the
    /// MaxKL trust region, and its value function regresses with Adam at ValueLearningRate.
    /// </remarks>
    public TRPOAgent(TRPOOptions<T> options)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();

        // Initialize networks directly in constructor
        _policyNetwork = CreatePolicyNetwork();
        _oldPolicyNetwork = CreatePolicyNetwork();
        _valueNetwork = CreateValueNetwork();

        CopyNetworkWeights(_policyNetwork, _oldPolicyNetwork);

        // Register networks with base class
        Networks.Add(_policyNetwork);
        Networks.Add(_oldPolicyNetwork);
        Networks.Add(_valueNetwork);
    }

    private INeuralNetwork<T> CreatePolicyNetwork()
    {
        int outputSize = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;

        if (_options.PolicyHiddenLayers is null || _options.PolicyHiddenLayers.Count == 0)
            throw new ArgumentException("Policy hidden layers must contain at least one positive width.", nameof(_options.PolicyHiddenLayers));
        var layers = new List<ILayer<T>>();
        foreach (int width in _options.PolicyHiddenLayers)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(_options.PolicyHiddenLayers), "Every policy hidden width must be positive.");
            layers.Add(new DenseLayer<T>(width, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        IActivationFunction<T> outputActivation = _options.IsContinuous
            ? new IdentityActivation<T>()
            : new SoftmaxActivation<T>();
        layers.Add(new DenseLayer<T>(outputSize, outputActivation));
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize,
            layers: layers)
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture, lossFunction: _options.ValueLossFunction);
    }

    private INeuralNetwork<T> CreateValueNetwork()
    {
        if (_options.ValueHiddenLayers is null || _options.ValueHiddenLayers.Count == 0)
            throw new ArgumentException("Value hidden layers must contain at least one positive width.", nameof(_options.ValueHiddenLayers));
        var layers = new List<ILayer<T>>();
        foreach (int width in _options.ValueHiddenLayers)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(_options.ValueHiddenLayers), "Every value hidden width must be positive.");
            layers.Add(new DenseLayer<T>(width, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1,
            layers: layers)
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_options.ValueLearningRate),
            lossFunction: _options.ValueLossFunction);
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_options.IsContinuous)
        {
            var engine = AiDotNetEngine.Current;
            var mean = policyOutput.GetSubVector(0, _options.ActionSize);
            var logStd = engine.Clamp(
                policyOutput.GetSubVector(_options.ActionSize, _options.ActionSize), NumOps.FromDouble(-20), NumOps.FromDouble(2));

            if (!training)
            {
                return mean;
            }

            // mean + exp(log_std) * N(0, 1), one draw per dimension in index order
            var noise = RLTensorOps.NormalRandom<T>(Random, _options.ActionSize);
            return engine.Add(mean, engine.Multiply(engine.Exp(logStd), noise));
        }
        else
        {
            // Discrete: sample from distribution
            if (!training)
            {
                int bestAction = ArgMax(policyOutput);
                return RLTensorOps.OneHot<T>(_options.ActionSize, bestAction);
            }

            // First action whose running probability reaches the draw (summed in double precision, as before);
            // action 0 when rounding falls short.
            double r = Random.NextDouble();
            int selectedAction = RLTensorOps.SampleFromCdf(AsDoubles(RLTensorOps.Head(policyOutput, _options.ActionSize)), r, 0);
            return RLTensorOps.OneHot<T>(_options.ActionSize, selectedAction);
        }
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _trajectoryBuffer.Add((state, action, reward, nextState, done));

        if (_trajectoryBuffer.Count >= _options.StepsPerUpdate)
        {
            Train();
            _trajectoryBuffer.Clear();
        }
    }

    /// <inheritdoc />
    public override T Train()
    {
        if (_trajectoryBuffer.Count == 0)
        {
            return NumOps.Zero;
        }

        // OUTSIDE THE STEP ARENA. One update runs a tape per value iteration, one for g and one per Fisher-vector
        // product, and every top-level tape rewinds the active arena when it is disposed. The batch, g, the old
        // policy's outputs and each conjugate-gradient iterate outlive the tape that made them, so under an arena the
        // next tape was handed their buffers: two identically seeded agents diverged from the first update. Same rule
        // as CreditAssignmentGradientComputer: what outlives a tape allocates outside it.
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();

        // Compute returns and advantages
        var (states, actions, advantages, returns) = ComputeAdvantages();
        int n = states.Count;
        int stateSize = _options.StateSize;

        var batchStates = RLTensorOps.StackRows(states.Select(stateRow => RLTensorOps.Head(stateRow, stateSize)).ToList());
        var batchReturns = RLTensorOps.FromList(returns.Take(n).ToList(), n, 1);

        // Value function: a separate regression on the empirical returns (Schulman et al. 2015, §8 and
        // Appendix D), ValueIterations passes at ValueLearningRate. The value network trains with the
        // optimizer it was built with, which carries ValueLearningRate.
        for (int iteration = 0; iteration < _options.ValueIterations; iteration++)
        {
            _valueNetwork.Train(batchStates, batchReturns);
        }

        // Policy: one natural-gradient step inside the KL trust region.
        T surrogateImprovement = UpdatePolicyWithinTrustRegion(batchStates, actions, advantages);

        _updateCount++;

        return surrogateImprovement;
    }

    /// <summary>
    /// The KL divergence between the policy before and after the most recent update, averaged over that
    /// update's states. Zero when the line search rejected every candidate step.
    /// </summary>
    internal double LastPolicyKL { get; private set; }

    /// <summary>
    /// The fraction of the full natural-gradient step the line search accepted on the most recent update
    /// (<c>LineSearchBacktrackCoeff^k</c>), or zero when it accepted none.
    /// </summary>
    internal double LastStepFraction { get; private set; }

    /// <summary>
    /// The policy's action distribution for <paramref name="state"/>: the action probabilities for a discrete
    /// policy, or the mean followed by the clamped log standard deviation for a continuous one.
    /// </summary>
    internal Vector<T> PolicyDistribution(Vector<T> state)
    {
        var output = _policyNetwork.Predict(Tensor<T>.FromVector(state)).ToVector();
        return ToDistributionParameters(output, 0, output.Length);
    }

    /// <summary>The value network's parameters, for tests that check the value regression's settings.</summary>
    internal Vector<T> GetValueNetworkParameters()
    {
        // Lazily sized layers have no weights until their first forward; materialize so the vector's
        // length does not change between two reads.
        if (_valueNetwork is NeuralNetworkBase<T> network) network.MaterializeParameters();
        return _valueNetwork.GetParameters();
    }

    /// <summary>
    /// Trust Region Policy Optimization step (Schulman et al. 2015, §6 and Appendix C):
    /// maximise the surrogate L(θ) = E[π_θ(a|s)/π_old(a|s) · A] subject to E_s[KL(π_old ‖ π_θ)] ≤ MaxKL.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constrained problem is solved as the paper prescribes: the search direction x ≈ F⁻¹g comes from
    /// ConjugateGradientIterations steps of conjugate gradient on Fisher-vector products (with Damping added to
    /// F), the step is scaled so its quadratic KL estimate ½·sᵀFs equals MaxKL, and a backtracking line search
    /// (LineSearchSteps candidates shrinking by LineSearchBacktrackCoeff) accepts the first candidate whose
    /// actual surrogate improvement is at least LineSearchAcceptRatio of the linear prediction AND whose exact KL
    /// stays within MaxKL. When no candidate passes, the policy is left unchanged.
    /// </para>
    /// <para>
    /// The Fisher matrix is the analytic estimator of Appendix C.1, F = (1/N)·Σ Jᵀ M J, where J is the Jacobian
    /// of the network's distribution parameters with respect to θ and M is the Fisher information of the action
    /// distribution in those parameters (diag(1/p) for categorical probabilities; diag(1/σ², 2) for a Gaussian
    /// parameterised by mean and log standard deviation). Jᵀu is an exact reverse-mode product through the
    /// network's tape; Jv is a central difference of the network outputs along v, whose truncation error is
    /// second order in the probe size.
    /// </para>
    /// </remarks>
    private T UpdatePolicyWithinTrustRegion(Tensor<T> batchStates, List<Vector<T>> actions, List<T> advantages)
    {
        int n = actions.Count;
        int outputWidth = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;

        // Distribution parameters of the policy that collected the data; constants for this update. The
        // forward runs before the parameters are read so lazily sized layers already hold their weights.
        var oldOutputs = _policyNetwork.Predict(batchStates).ToVector();
        var thetaOld = _policyNetwork.GetParameters();
        var oldParams = Enumerable.Range(0, n)
            .Select(row => ToDistributionParameters(oldOutputs, row * outputWidth, outputWidth)).ToArray();
        var oldLogProbs = Enumerable.Range(0, n).Select(row => LogProbability(oldParams[row], actions[row])).ToArray();
        var advantage = AsDoubles(Vector<T>.FromEnumerable(advantages.Take(n))).ToArray();
        CopyNetworkWeights(_policyNetwork, _oldPolicyNetwork);

        // g = ∇θ L at θ_old = Jᵀ (∂L/∂output), with ∂L/∂output evaluated in closed form at ratio = 1.
        // Row i: A_i * d log pi / d params / n (in double precision, as before).
        var engine = AiDotNetEngine.Current;
        var surrogateRows = Enumerable.Range(0, n)
            .Select(row => engine.Divide(
                engine.Multiply(Vector<double>.FromArray(LogProbabilityOutputGradient(oldParams[row], actions[row])), advantage[row]),
                (double)n))
            .ToList();
        var surrogateOutputGradient = FromDoubleRows(surrogateRows, n, outputWidth);
        var g = PolicyVectorJacobianProduct(batchStates, surrogateOutputGradient);
        double gNorm = Math.Sqrt(Dot(g, g));
        LastPolicyKL = 0.0;
        LastStepFraction = 0.0;
        if (double.IsNaN(gNorm) || gNorm <= 0.0 || double.IsInfinity(gNorm))
        {
            return NumOps.Zero;
        }

        Vector<T> FisherVectorProduct(Vector<T> v)
        {
            double vNorm = Math.Sqrt(Dot(v, v));
            var result = new Vector<T>(v.Length);
            if (vNorm <= 0.0) return result;

            // Central difference of the outputs along v gives J·v.
            double thetaNorm = Math.Sqrt(Dot(thetaOld, thetaOld));
            double probe = 1e-5 * Math.Max(1.0, thetaNorm) / vNorm;
            _policyNetwork.UpdateParameters(Axpy(thetaOld, probe, v));
            var plus = _policyNetwork.Predict(batchStates).ToVector();
            _policyNetwork.UpdateParameters(Axpy(thetaOld, -probe, v));
            var minus = _policyNetwork.Predict(batchStates).ToVector();
            _policyNetwork.UpdateParameters(thetaOld);

            // Row i: M_i (J v)_i / n, with (J v)_i = (params(theta + h v) - params(theta - h v)) / 2h.
            var weightedRows = Enumerable.Range(0, n)
                .Select(row =>
                {
                    var plusParams = AsDoubles(ToDistributionParameters(plus, row * outputWidth, outputWidth));
                    var minusParams = AsDoubles(ToDistributionParameters(minus, row * outputWidth, outputWidth));
                    var jv = engine.Divide(engine.Subtract(plusParams, minusParams), 2.0 * probe);
                    return engine.Divide(engine.Multiply(DistributionFisherDiagonal(oldParams[row], outputWidth), jv), (double)n);
                })
                .ToList();
            var weighted = FromDoubleRows(weightedRows, n, outputWidth);

            // F v + damping * v
            var fv = PolicyVectorJacobianProduct(batchStates, weighted);
            return Axpy(fv, _options.Damping, v);
        }

        // Conjugate gradient for F x = g.
        var x = ConjugateGradient(FisherVectorProduct, g, _options.ConjugateGradientIterations);
        double xFx = Dot(x, FisherVectorProduct(x));
        double maxKl = NumOps.ToDouble(_options.MaxKL);
        if (double.IsNaN(xFx) || xFx <= 0.0 || double.IsInfinity(xFx) || double.IsNaN(maxKl) || maxKl <= 0.0)
        {
            return NumOps.Zero;
        }

        // Scale so the quadratic KL model of the full step, ½ sᵀ F s, equals MaxKL.
        double stepScale = Math.Sqrt(2.0 * maxKl / xFx);
        double expectedImprovement = stepScale * Dot(g, x);
        double oldSurrogate = Surrogate(batchStates, actions, oldLogProbs, advantage, outputWidth, null);

        double fraction = 1.0;
        double? acceptedImprovement = null;
        for (int step = 0; step < _options.LineSearchSteps; step++, fraction *= _options.LineSearchBacktrackCoeff)
        {
            var candidate = Axpy(thetaOld, fraction * stepScale, x);
            _policyNetwork.UpdateParameters(candidate);
            double kl = 0.0;
            double newSurrogate = Surrogate(batchStates, actions, oldLogProbs, advantage, outputWidth,
                (i, newParams) => kl += KLDivergence(oldParams[i], newParams));
            kl /= n;
            double improvement = newSurrogate - oldSurrogate;
            double expected = fraction * expectedImprovement;

            if (kl <= maxKl && improvement > 0.0 && improvement / expected >= _options.LineSearchAcceptRatio)
            {
                LastPolicyKL = kl;
                LastStepFraction = fraction;
                acceptedImprovement = improvement;
                break;
            }
        }

        if (acceptedImprovement.HasValue)
        {
            return NumOps.FromDouble(acceptedImprovement.Value);
        }

        // No candidate satisfied both conditions: keep the old policy.
        _policyNetwork.UpdateParameters(thetaOld);
        return NumOps.Zero;
    }

    /// <summary>
    /// Mean importance-weighted advantage under the network's current parameters; <paramref name="visit"/>
    /// receives each state's new distribution parameters so the caller can accumulate the KL in the same pass.
    /// </summary>
    private double Surrogate(Tensor<T> batchStates, List<Vector<T>> actions, double[] oldLogProbs, double[] advantage,
        int outputWidth, Action<int, Vector<T>>? visit)
    {
        var outputs = _policyNetwork.Predict(batchStates).ToVector();
        double total = 0.0;
        for (int i = 0; i < actions.Count; i++)
        {
            var parameters = ToDistributionParameters(outputs, i * outputWidth, outputWidth);
            total += Math.Exp(LogProbability(parameters, actions[i]) - oldLogProbs[i]) * advantage[i];
            visit?.Invoke(i, parameters);
        }
        return total / actions.Count;
    }

    /// <summary>
    /// Jᵀw for the policy network over a batch: the parameter gradient of Σ output ⊙ w, laid out as
    /// <c>GetParameters()</c>.
    /// </summary>
    private Vector<T> PolicyVectorJacobianProduct(Tensor<T> batchStates, Tensor<T> outputWeights)
    {
        if (_policyNetwork is not NeuralNetworkBase<T> network)
        {
            throw new NotSupportedException(
                $"TRPO needs parameter gradients from its policy network, which {_policyNetwork.GetType().Name} does not provide.");
        }

        return network.ComputeGradients(batchStates, outputWeights, OutputContractionLoss<T>.Instance);
    }

    private Vector<T> ConjugateGradient(Func<Vector<T>, Vector<T>> multiply, Vector<T> b, int iterations)
    {
        var x = new Vector<T>(b.Length);
        var r = b.Clone();
        var p = b.Clone();
        double rr = Dot(r, r);
        for (int k = 0; k < iterations && rr > 1e-10; k++)
        {
            var ap = multiply(p);
            double pAp = Dot(p, ap);
            if (double.IsNaN(pAp) || pAp <= 0.0) break;
            double alpha = rr / pAp;
            x = Axpy(x, alpha, p);
            r = Axpy(r, -alpha, ap);
            double rrNext = Dot(r, r);
            p = Axpy(r, rrNext / rr, p);
            rr = rrNext;
        }
        return x;
    }

    private double Dot(Vector<T> a, Vector<T> b)
    {
        // sum_i a_i b_i, accumulated in double precision
        return AiDotNetEngine.Current.DotProduct(AsDoubles(a), AsDoubles(b));
    }

    /// <summary>a + scale·b.</summary>
    private Vector<T> Axpy(Vector<T> a, double scale, Vector<T> b)
    {
        // a + scale * b, evaluated in double precision
        var engine = AiDotNetEngine.Current;
        return FromDoubles(engine.Add(AsDoubles(a), engine.Multiply(AsDoubles(b), scale)));
    }

    /// <summary>
    /// Reads one state's distribution parameters out of the network output: probabilities (floored away from
    /// zero) for the softmax head, or the mean then the log standard deviation clamped to [-20, 2] — the same
    /// clamp <see cref="SelectAction"/> samples with.
    /// </summary>
    private Vector<T> ToDistributionParameters(Vector<T> outputs, int offset, int width)
    {
        var parameters = new Vector<T>(width);
        var engine = AiDotNetEngine.Current;
        if (_options.IsContinuous)
        {
            int actionSize = _options.ActionSize;
            var mean = outputs.GetSubVector(offset, actionSize);
            var logStd = engine.Clamp(outputs.GetSubVector(offset + actionSize, actionSize), NumOps.FromDouble(-20), NumOps.FromDouble(2));
            mean.AsSpan().CopyTo(parameters.AsWritableSpan().Slice(0, actionSize));
            logStd.AsSpan().CopyTo(parameters.AsWritableSpan().Slice(actionSize, actionSize));
        }
        else
        {
            // max(p, floor) in double precision; a NaN stays NaN, as Math.Max keeps it.
            var probabilities = AsDoubles(outputs.GetSubVector(offset, width)).AsTensor();
            var floor = engine.TensorFullLike(probabilities, ProbabilityFloor);
            var floored = engine.TensorWhere(engine.TensorLessThan(probabilities, ProbabilityFloor), floor, probabilities);
            parameters = floored.Cast<T>().ToVector();
        }
        return parameters;
    }

    private const double ProbabilityFloor = 1e-8;

    private double LogProbability(Vector<T> parameters, Vector<T> action)
    {
        if (!_options.IsContinuous)
        {
            return Math.Log(NumOps.ToDouble(parameters[ArgMax(action)]));
        }

        // sum_i -0.5 z_i^2 - log sigma_i - 0.5 log(2 pi), z = (a - mu) / sigma, in double precision
        int actionSize = _options.ActionSize;
        var doubles = AsDoubles(parameters);
        return RLTensorOps.GaussianLogProb(
            doubles.GetSubVector(0, actionSize), doubles.GetSubVector(actionSize, actionSize), AsDoubles(RLTensorOps.Head(action, actionSize)));
    }

    /// <summary>∂ log π(a|s) / ∂(distribution parameters).</summary>
    private double[] LogProbabilityOutputGradient(Vector<T> parameters, Vector<T> action)
    {
        var gradient = new double[parameters.Length];
        if (!_options.IsContinuous)
        {
            int index = ArgMax(action);
            gradient[index] = 1.0 / NumOps.ToDouble(parameters[index]);
            return gradient;
        }

        // d/d mu = z / sigma, d/d log sigma = z^2 - 1, z = (a - mu) / sigma (double precision)
        int actionSize = _options.ActionSize;
        var engine = AiDotNetEngine.Current;
        var doubles = AsDoubles(parameters);
        var std = engine.Exp(doubles.GetSubVector(actionSize, actionSize));
        var z = engine.Divide(engine.Subtract(AsDoubles(RLTensorOps.Head(action, actionSize)), doubles.GetSubVector(0, actionSize)), std);
        engine.Divide(z, std).AsSpan().CopyTo(gradient.AsSpan(0, actionSize));
        engine.Subtract(engine.Multiply(z, z), Vector<double>.CreateDefault(actionSize, 1.0)).AsSpan().CopyTo(gradient.AsSpan(actionSize, actionSize));
        return gradient;
    }

    /// <summary>
    /// Diagonal of the action distribution's Fisher information in its parameters: 1/p_k for categorical
    /// probabilities (exact on the simplex, where every admissible direction sums to zero), 1/σ² for a Gaussian
    /// mean and 2 for its log standard deviation.
    /// </summary>
    private Vector<double> DistributionFisherDiagonal(Vector<T> parameters, int width)
    {
        var engine = AiDotNetEngine.Current;
        var doubles = AsDoubles(parameters);
        if (!_options.IsContinuous)
        {
            // 1 / p_k
            return engine.Divide(Vector<double>.CreateDefault(width, 1.0), RLTensorOps.Head(doubles, width));
        }

        // [1 / sigma^2 for the means | 2 for the log standard deviations]
        int actionSize = _options.ActionSize;
        var std = engine.Exp(doubles.GetSubVector(actionSize, actionSize));
        var meanDiagonal = engine.Divide(Vector<double>.CreateDefault(actionSize, 1.0), engine.Multiply(std, std));
        var diagonal = new Vector<double>(width);
        meanDiagonal.AsSpan().CopyTo(diagonal.AsWritableSpan().Slice(0, actionSize));
        diagonal.AsWritableSpan().Slice(actionSize, width - actionSize).Fill(2.0);
        return diagonal;
    }

    /// <summary>KL(π_old ‖ π_new) for one state.</summary>
    private double KLDivergence(Vector<T> oldParameters, Vector<T> newParameters)
    {
        var engine = AiDotNetEngine.Current;
        var oldDoubles = AsDoubles(oldParameters);
        var newDoubles = AsDoubles(newParameters);
        if (!_options.IsContinuous)
        {
            // sum_k p_k log(p_k / q_k)
            var p = oldDoubles.AsTensor();
            var logRatio = engine.TensorLog(engine.TensorDivide(p, RLTensorOps.Head(newDoubles, oldDoubles.Length).AsTensor()));
            return engine.TensorSum(engine.TensorMultiply(p, logRatio));
        }

        // sum_i log(sigma_new / sigma_old) + (sigma_old^2 + (mu_old - mu_new)^2) / (2 sigma_new^2) - 1/2
        int actionSize = _options.ActionSize;
        var oldLogStd = oldDoubles.GetSubVector(actionSize, actionSize);
        var newLogStd = newDoubles.GetSubVector(actionSize, actionSize);
        var oldVar = engine.Exp(engine.Multiply(oldLogStd, 2.0));
        var newVar = engine.Exp(engine.Multiply(newLogStd, 2.0));
        var meanGap = engine.Subtract(oldDoubles.GetSubVector(0, actionSize), newDoubles.GetSubVector(0, actionSize));
        var spread = engine.Divide(engine.Add(oldVar, engine.Multiply(meanGap, meanGap)), engine.Multiply(newVar, 2.0));
        var perDimension = engine.Subtract(engine.Add(engine.Subtract(newLogStd, oldLogStd), spread), Vector<double>.CreateDefault(actionSize, 0.5));
        return engine.Sum(perDimension);
    }

    private (List<Vector<T>> states, List<Vector<T>> actions, List<T> advantages, List<T> returns) ComputeAdvantages()
    {
        var states = new List<Vector<T>>();
        var actions = new List<Vector<T>>();
        var rewards = new List<T>();
        var values = new List<T>();
        var doneFlags = new List<bool>();
        var nextValues = new List<T>();

        // Cache options values to avoid nullable warnings
        T discountFactor = DiscountFactor;
        T gaeLambda = _options.GaeLambda;

        foreach (var (state, action, reward, nextState, done) in _trajectoryBuffer)
        {
            states.Add(state);
            actions.Add(action);
            rewards.Add(reward);
            doneFlags.Add(done);

            // Compute current state value
            var stateTensor = Tensor<T>.FromVector(state);
            var valueTensor = _valueNetwork.Predict(stateTensor);
            values.Add(RLTensorOps.FirstValue(valueTensor));

            // Compute next state value (correctly use nextState from buffer)
            if (done)
            {
                nextValues.Add(NumOps.Zero);
            }
            else
            {
                var nextStateTensor = Tensor<T>.FromVector(nextState);
                var nextValueTensor = _valueNetwork.Predict(nextStateTensor);
                nextValues.Add(RLTensorOps.FirstValue(nextValueTensor));
            }
        }

        // Compute returns
        var returns = new List<T>();
        T runningReturn = NumOps.Zero;

        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            if (doneFlags[i])
            {
                runningReturn = rewards[i];
            }
            else
            {
                runningReturn = NumOps.Add(rewards[i], NumOps.Multiply(discountFactor, runningReturn));
            }
            returns.Insert(0, runningReturn);
        }

        // Compute advantages using GAE
        var advantages = new List<T>();
        T gaeAdvantage = NumOps.Zero;

        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            T nextValue = nextValues[i];
            if (doneFlags[i])
            {
                nextValue = NumOps.Zero;
            }

            var delta = NumOps.Add(rewards[i], NumOps.Multiply(discountFactor, nextValue));
            delta = NumOps.Subtract(delta, values[i]);

            gaeAdvantage = NumOps.Add(delta, NumOps.Multiply(discountFactor,
                NumOps.Multiply(gaeLambda, gaeAdvantage)));

            advantages.Insert(0, gaeAdvantage);
        }

        // Normalize advantages
        var mean = StatisticsHelper<T>.CalculateMean(advantages.ToArray());
        var std = StatisticsHelper<T>.CalculateStandardDeviation(advantages.ToArray());

        if (advantages.Count > 0 && NumOps.GreaterThan(std, NumOps.Zero))
        {
            // (A - mean) / std
            var engine = AiDotNetEngine.Current;
            var advantageVector = Vector<T>.FromEnumerable(advantages);
            advantages = engine.Divide(engine.Subtract(advantageVector, Vector<T>.CreateDefault(advantages.Count, mean)), std).ToList();
        }

        return (states, actions, advantages, returns);
    }

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    // The trust-region arithmetic runs in double precision (conjugate gradient, KL, log-probabilities); these
    // convert whole vectors across that boundary.
    private static Vector<double> AsDoubles(Vector<T> values)
    {
        return values.Length == 0 ? new Vector<double>(0) : values.AsTensor().Cast<double>().ToVector();
    }

    private static Vector<T> FromDoubles(Vector<double> values)
    {
        return values.Length == 0 ? new Vector<T>(0) : values.AsTensor().Cast<T>().ToVector();
    }

    private static Tensor<T> FromDoubleRows(IReadOnlyList<Vector<double>> rows, int count, int width)
    {
        return count == 0 ? new Tensor<T>([0, width]) : RLTensorOps.StackRows(rows).Cast<T>();
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_trajectoryBuffer.Count)
        };
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        // No episode-specific state
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

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
        };
    }

    /// <inheritdoc />
    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Computes gradients of the loss with respect to this agent's parameters, without updating them.
    /// </summary>
    /// <param name="input">The input state.</param>
    /// <param name="target">The target output.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the same length as <c>GetParameters()</c>.</returns>
    /// <remarks>
    /// Returns PARAMETER-space gradients, as <c>IGradientComputable</c> documents and as
    /// <c>ApplyGradients</c> requires. The gradient comes from the network's own tape pass, so no
    /// derivative is written by hand here.
    /// </remarks>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        return ComputeGradientsForNetwork(
            _policyNetwork,
            new[] { _policyNetwork, _valueNetwork },
            input,
            target,
            lossFunction);
    }

    /// <inheritdoc />
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc />
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    

}
}
