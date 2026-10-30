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

        // Create initial architecture for LayerHelper
        var tempArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize);

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            tempArchitecture,
            hiddenLayerCount: _options.PolicyHiddenLayers.Count,
            hiddenLayerSize: _options.PolicyHiddenLayers.FirstOrDefault() > 0 ? _options.PolicyHiddenLayers.First() : 128
        ).ToList();

        // Override output layer activation for continuous vs discrete actions
        if (!_options.IsContinuous)
        {
            // For discrete actions, use softmax activation
            // Note: Just rebuild the last layer with correct activation
            int lastInputSize = _options.PolicyHiddenLayers.LastOrDefault() > 0 ? _options.PolicyHiddenLayers.Last() : 128;
            layers[layers.Count - 1] = new DenseLayer<T>(
                outputSize,
                (IActivationFunction<T>)new SoftmaxActivation<T>()
            );
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
        // Create initial architecture for LayerHelper
        var tempArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1);

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            tempArchitecture,
            hiddenLayerCount: _options.ValueHiddenLayers.Count,
            hiddenLayerSize: _options.ValueHiddenLayers.FirstOrDefault() > 0 ? _options.ValueHiddenLayers.First() : 128
        ).ToList();

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture,
            optimizer: CreateNetworkOptimizer(_options.ValueLearningRate),
            lossFunction: _options.ValueLossFunction);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_options.IsContinuous)
        {
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
            // Discrete: sample from distribution
            if (!training)
            {
                int bestAction = ArgMax(policyOutput);
                var action = new Vector<T>(_options.ActionSize);
                action[bestAction] = NumOps.One;
                return action;
            }

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

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _trajectoryBuffer.Add((state, action, reward, nextState, done));

        if (_trajectoryBuffer.Count >= _options.StepsPerUpdate)
        {
            Train();
            _trajectoryBuffer.Clear();
        }
    }

    public override T Train()
    {
        if (_trajectoryBuffer.Count == 0)
        {
            return NumOps.Zero;
        }

        // Compute returns and advantages
        var (states, actions, advantages, returns) = ComputeAdvantages();
        int n = states.Count;
        int stateSize = _options.StateSize;

        var batchStates = new Tensor<T>([n, stateSize]);
        var batchReturns = new Tensor<T>([n, 1]);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < stateSize; j++) batchStates[i, j] = states[i][j];
            batchReturns[i, 0] = returns[i];
        }

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
        var oldParams = new Vector<T>[n];
        var oldLogProbs = new double[n];
        var advantage = new double[n];
        for (int i = 0; i < n; i++)
        {
            oldParams[i] = ToDistributionParameters(oldOutputs, i * outputWidth, outputWidth);
            oldLogProbs[i] = LogProbability(oldParams[i], actions[i]);
            advantage[i] = NumOps.ToDouble(advantages[i]);
        }
        CopyNetworkWeights(_policyNetwork, _oldPolicyNetwork);

        // g = ∇θ L at θ_old = Jᵀ (∂L/∂output), with ∂L/∂output evaluated in closed form at ratio = 1.
        var surrogateOutputGradient = new Tensor<T>([n, outputWidth]);
        for (int i = 0; i < n; i++)
        {
            var dLogProb = LogProbabilityOutputGradient(oldParams[i], actions[i]);
            for (int k = 0; k < outputWidth; k++)
            {
                surrogateOutputGradient[i, k] = NumOps.FromDouble(advantage[i] * dLogProb[k] / n);
            }
        }
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

            var weighted = new Tensor<T>([n, outputWidth]);
            for (int i = 0; i < n; i++)
            {
                var plusParams = ToDistributionParameters(plus, i * outputWidth, outputWidth);
                var minusParams = ToDistributionParameters(minus, i * outputWidth, outputWidth);
                for (int k = 0; k < outputWidth; k++)
                {
                    double jv = (NumOps.ToDouble(plusParams[k]) - NumOps.ToDouble(minusParams[k])) / (2.0 * probe);
                    weighted[i, k] = NumOps.FromDouble(DistributionFisherDiagonal(oldParams[i], k) * jv / n);
                }
            }

            var fv = PolicyVectorJacobianProduct(batchStates, weighted);
            for (int p = 0; p < fv.Length; p++)
            {
                result[p] = NumOps.FromDouble(NumOps.ToDouble(fv[p]) + _options.Damping * NumOps.ToDouble(v[p]));
            }
            return result;
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
                return NumOps.FromDouble(improvement);
            }
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
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += NumOps.ToDouble(a[i]) * NumOps.ToDouble(b[i]);
        return sum;
    }

    /// <summary>a + scale·b.</summary>
    private Vector<T> Axpy(Vector<T> a, double scale, Vector<T> b)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.FromDouble(NumOps.ToDouble(a[i]) + scale * NumOps.ToDouble(b[i]));
        }
        return result;
    }

    /// <summary>
    /// Reads one state's distribution parameters out of the network output: probabilities (floored away from
    /// zero) for the softmax head, or the mean then the log standard deviation clamped to [-20, 2] — the same
    /// clamp <see cref="SelectAction"/> samples with.
    /// </summary>
    private Vector<T> ToDistributionParameters(Vector<T> outputs, int offset, int width)
    {
        var parameters = new Vector<T>(width);
        if (_options.IsContinuous)
        {
            int actionSize = _options.ActionSize;
            for (int i = 0; i < actionSize; i++)
            {
                parameters[i] = outputs[offset + i];
                parameters[actionSize + i] = MathHelper.Clamp<T>(
                    outputs[offset + actionSize + i], NumOps.FromDouble(-20), NumOps.FromDouble(2));
            }
        }
        else
        {
            for (int i = 0; i < width; i++)
            {
                parameters[i] = NumOps.FromDouble(Math.Max(NumOps.ToDouble(outputs[offset + i]), ProbabilityFloor));
            }
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

        int actionSize = _options.ActionSize;
        double logProb = 0.0;
        for (int i = 0; i < actionSize; i++)
        {
            double mean = NumOps.ToDouble(parameters[i]);
            double logStd = NumOps.ToDouble(parameters[actionSize + i]);
            double z = (NumOps.ToDouble(action[i]) - mean) / Math.Exp(logStd);
            logProb += -0.5 * z * z - logStd - 0.5 * Math.Log(2.0 * Math.PI);
        }
        return logProb;
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

        int actionSize = _options.ActionSize;
        for (int i = 0; i < actionSize; i++)
        {
            double mean = NumOps.ToDouble(parameters[i]);
            double std = Math.Exp(NumOps.ToDouble(parameters[actionSize + i]));
            double z = (NumOps.ToDouble(action[i]) - mean) / std;
            gradient[i] = z / std;
            gradient[actionSize + i] = z * z - 1.0;
        }
        return gradient;
    }

    /// <summary>
    /// Diagonal of the action distribution's Fisher information in its parameters: 1/p_k for categorical
    /// probabilities (exact on the simplex, where every admissible direction sums to zero), 1/σ² for a Gaussian
    /// mean and 2 for its log standard deviation.
    /// </summary>
    private double DistributionFisherDiagonal(Vector<T> parameters, int index)
    {
        if (!_options.IsContinuous)
        {
            return 1.0 / NumOps.ToDouble(parameters[index]);
        }

        int actionSize = _options.ActionSize;
        if (index < actionSize)
        {
            double std = Math.Exp(NumOps.ToDouble(parameters[actionSize + index]));
            return 1.0 / (std * std);
        }
        return 2.0;
    }

    /// <summary>KL(π_old ‖ π_new) for one state.</summary>
    private double KLDivergence(Vector<T> oldParameters, Vector<T> newParameters)
    {
        double kl = 0.0;
        if (!_options.IsContinuous)
        {
            for (int k = 0; k < oldParameters.Length; k++)
            {
                double p = NumOps.ToDouble(oldParameters[k]);
                double q = NumOps.ToDouble(newParameters[k]);
                kl += p * Math.Log(p / q);
            }
            return kl;
        }

        int actionSize = _options.ActionSize;
        for (int i = 0; i < actionSize; i++)
        {
            double oldMean = NumOps.ToDouble(oldParameters[i]);
            double newMean = NumOps.ToDouble(newParameters[i]);
            double oldLogStd = NumOps.ToDouble(oldParameters[actionSize + i]);
            double newLogStd = NumOps.ToDouble(newParameters[actionSize + i]);
            double oldVar = Math.Exp(2.0 * oldLogStd);
            double newVar = Math.Exp(2.0 * newLogStd);
            double meanGap = oldMean - newMean;
            kl += newLogStd - oldLogStd + (oldVar + meanGap * meanGap) / (2.0 * newVar) - 0.5;
        }
        return kl;
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
            values.Add(valueTensor.ToVector()[0]);

            // Compute next state value (correctly use nextState from buffer)
            if (done)
            {
                nextValues.Add(NumOps.Zero);
            }
            else
            {
                var nextStateTensor = Tensor<T>.FromVector(nextState);
                var nextValueTensor = _valueNetwork.Predict(nextStateTensor);
                nextValues.Add(nextValueTensor.ToVector()[0]);
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

        if (NumOps.GreaterThan(std, NumOps.Zero))
        {
            for (int i = 0; i < advantages.Count; i++)
            {
                advantages[i] = NumOps.Divide(NumOps.Subtract(advantages[i], mean), std);
            }
        }

        return (states, actions, advantages, returns);
    }

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_trajectoryBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
    }

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

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
        };
    }

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

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    

}
}
