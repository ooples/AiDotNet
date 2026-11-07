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
using AiDotNet.ReinforcementLearning.Parameters;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.TRPO;

/// <summary>
/// Trust Region Policy Optimization (TRPO) agent for reinforcement learning (Schulman et al. 2015).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The single-path algorithm of section 5.1 with the practical step of section 6 and appendix C: it maximizes the
/// surrogate <c>L(θ) = E[π_θ(a|s) / π_θold(a|s) · A(s, a)]</c> subject to the mean KL divergence
/// <c>D_KL(π_θold || π_θ) ≤ δ</c>. The search direction <c>s ≈ F⁻¹g</c> comes from the conjugate gradient algorithm on
/// Fisher-vector products <c>Fv = JᵀMJv</c> (appendix C.1), the step is scaled to <c>β = sqrt(2δ / sᵀFs)</c>, and a line
/// search shrinks it exponentially until the surrogate improves and the KL constraint holds.
/// </para>
/// <para>
/// The policy (appendix D) is a fully connected network from the state to the mean of a Gaussian with a diagonal,
/// state-independent covariance whose log standard deviations are a separate parameter vector; for discrete actions the
/// network's outputs are the logits of a categorical distribution. Advantages are discounted returns minus a learned
/// value baseline (<see cref="TRPOOptions{T}.GaeLambda"/> 1 is the paper's Monte Carlo estimate).
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
    /// <remarks>The serialization order: the policy network, the value baseline, then (continuous actions) the
    /// policy's log standard deviations.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_policyNetwork);
        RegisterParameterComponent(_valueNetwork);
        if (_options.IsContinuous)
            RegisterParameterComponent(new VectorParameterSource<T>(() => _logStd));
    }
    private TRPOOptions<T> _options;


    private INeuralNetwork<T> _policyNetwork;
    private INeuralNetwork<T> _valueNetwork;
    // Appendix D: "a separate set of parameters specifies the log standard deviation of each element" (continuous).
    private Vector<T> _logStd;

    private List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _trajectoryBuffer;
    private int _updateCount;
    private double _lastKl, _lastImprovement;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public TRPOAgent()
        : this(new TRPOOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public TRPOAgent(TRPOOptions<T> options)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();

        _policyNetwork = CreatePolicyNetwork();
        _valueNetwork = CreateValueNetwork();
        // Standard deviations start at one.
        _logStd = new Vector<T>(_options.IsContinuous ? _options.ActionSize : 0);

        // Size the lazy layers now, so the parameter vector is complete before the first update.
        var warmup = new Tensor<T>(new[] { 1, _options.StateSize });
        _policyNetwork.Predict(warmup);
        _valueNetwork.Predict(warmup);

        Networks.Add(_policyNetwork);
        Networks.Add(_valueNetwork);
    }

    // Appendix D: fully connected layers from the state to the mean of the Gaussian (or the categorical logits).
    private INeuralNetwork<T> CreatePolicyNetwork() => CreateMlp(_options.PolicyHiddenLayers, _options.ActionSize, optimizer: null);

    private INeuralNetwork<T> CreateValueNetwork() => CreateMlp(_options.ValueHiddenLayers, 1,
        new AdamOptimizer<T, Tensor<T>, Tensor<T>>(null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = NumOps.ToDouble(_options.ValueLearningRate),
            UseAdaptiveLearningRate = false,
        }));

    private INeuralNetwork<T> CreateMlp(List<int> hidden, int outputSize, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
    {
        if (hidden is null || hidden.Count == 0 || hidden.Any(width => width <= 0))
            throw new ArgumentException("Hidden layers must contain positive widths.", nameof(hidden));
        var layers = new List<ILayer<T>>();
        foreach (int width in hidden)
            layers.Add(new DenseLayer<T>(width, (IActivationFunction<T>)new TanhActivation<T>()));
        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));
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
        return new NeuralNetwork<T>(architecture, optimizer, lossFunction: _options.ValueLossFunction);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var output = _policyNetwork.Predict(Tensor<T>.FromVector(state)).ToVector();

        if (_options.IsContinuous)
        {
            var action = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                if (!training)
                {
                    action[i] = output[i];
                    continue;
                }
                var std = NumOps.Exp(_logStd[i]);
                var noise = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(Math.Max(Random.NextDouble(), double.Epsilon)))
                    * Math.Cos(2.0 * Math.PI * Random.NextDouble()));
                action[i] = NumOps.Add(output[i], NumOps.Multiply(std, noise));
            }
            return action;
        }

        var probabilities = Softmax(Enumerable.Range(0, _options.ActionSize).Select(i => NumOps.ToDouble(output[i])).ToArray());
        int selected = 0;
        if (!training)
        {
            for (int i = 1; i < probabilities.Length; i++) if (probabilities[i] > probabilities[selected]) selected = i;
        }
        else
        {
            double r = Random.NextDouble(), cumulative = 0;
            selected = probabilities.Length - 1;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (r <= cumulative) { selected = i; break; }
            }
        }
        var oneHot = new Vector<T>(_options.ActionSize);
        oneHot[selected] = NumOps.One;
        return oneHot;
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

    /// <summary>
    /// One TRPO iteration on the collected transitions: fits the value baseline, then takes the trust-region step on
    /// the policy. Returns the surrogate's improvement.
    /// </summary>
    public override T Train()
    {
        if (_trajectoryBuffer.Count == 0)
        {
            return NumOps.Zero;
        }
        // Batch tensors and conjugate-gradient iterates outlive individual tapes and their arena rewinds.
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();

        var (states, advantages, returns) = ComputeAdvantages();
        var actions = _trajectoryBuffer.Select(t => t.action).ToList();

        FitBaseline(states, returns);
        _lastImprovement = PolicyStep(states, actions, advantages);
        _updateCount++;
        return NumOps.FromDouble(_lastImprovement);
    }

    private (Tensor<T> States, double[] Advantages, double[] Returns) ComputeAdvantages()
    {
        int n = _trajectoryBuffer.Count;
        var states = new Tensor<T>(new[] { n, _options.StateSize });
        var nextStates = new Tensor<T>(new[] { n, _options.StateSize });
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _options.StateSize; j++)
            {
                states[i, j] = _trajectoryBuffer[i].state[j];
                nextStates[i, j] = _trajectoryBuffer[i].nextState[j];
            }
        }
        var values = _valueNetwork.Predict(states);
        var nextValues = _valueNetwork.Predict(nextStates);
        double gamma = NumOps.ToDouble(DiscountFactor), lambda = NumOps.ToDouble(_options.GaeLambda);

        // Generalized advantage estimation; lambda = 1 is the paper's discounted return minus the baseline. A
        // trajectory cut off by the buffer is bootstrapped with the baseline's value of its last next state.
        var advantages = new double[n];
        var returns = new double[n];
        double running = 0;
        for (int i = n - 1; i >= 0; i--)
        {
            bool done = _trajectoryBuffer[i].done;
            double value = NumOps.ToDouble(values[i]);
            double next = done ? 0 : NumOps.ToDouble(nextValues[i]);
            if (done || i == n - 1) running = 0;
            double delta = NumOps.ToDouble(_trajectoryBuffer[i].reward) + gamma * next - value;
            running = delta + gamma * lambda * running;
            advantages[i] = running;
            returns[i] = running + value;
        }

        // Standardized advantages (the authors' implementation); the surrogate's direction is unchanged.
        double mean = advantages.Average();
        double std = Math.Sqrt(advantages.Select(a => (a - mean) * (a - mean)).Average());
        if (std > 1e-8) for (int i = 0; i < n; i++) advantages[i] = (advantages[i] - mean) / std;
        return (states, advantages, returns);
    }

    private void FitBaseline(Tensor<T> states, double[] returns)
    {
        var targets = new Tensor<T>(new[] { returns.Length, 1 });
        for (int i = 0; i < returns.Length; i++) targets[i, 0] = NumOps.FromDouble(returns[i]);
        for (int it = 0; it < _options.ValueIterations; it++) _valueNetwork.Train(states, targets);
    }

    // The policy's distribution parameters at every state: means (with the shared log standard deviations) or logits.
    private double[][] Distribution(Tensor<T> states)
    {
        var output = _policyNetwork.Predict(states);
        int n = states.Shape[0], a = _options.ActionSize;
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[a];
            for (int k = 0; k < a; k++) result[i][k] = NumOps.ToDouble(output[i * a + k]);
        }
        return result;
    }

    private static double[] Softmax(double[] logits)
    {
        double max = logits.Max();
        var e = logits.Select(z => Math.Exp(z - max)).ToArray();
        double sum = e.Sum();
        return e.Select(v => v / sum).ToArray();
    }

    private double LogProbability(double[] parameters, double[] logStd, Vector<T> action)
    {
        if (_options.IsContinuous)
        {
            double logp = 0;
            for (int k = 0; k < parameters.Length; k++)
            {
                double z = (NumOps.ToDouble(action[k]) - parameters[k]) / Math.Exp(logStd[k]);
                logp += -0.5 * z * z - logStd[k] - 0.5 * Math.Log(2 * Math.PI);
            }
            return logp;
        }
        return Math.Log(Math.Max(Softmax(parameters)[ArgMax(action)], 1e-300));
    }

    // D_KL(old || new) at one state.
    private double Kl(double[] oldParameters, double[] oldLogStd, double[] newParameters, double[] newLogStd)
    {
        double kl = 0;
        if (_options.IsContinuous)
        {
            for (int k = 0; k < oldParameters.Length; k++)
            {
                double oldVariance = Math.Exp(2 * oldLogStd[k]), newVariance = Math.Exp(2 * newLogStd[k]);
                double diff = oldParameters[k] - newParameters[k];
                kl += newLogStd[k] - oldLogStd[k] + (oldVariance + diff * diff) / (2 * newVariance) - 0.5;
            }
            return kl;
        }
        var p = Softmax(oldParameters);
        var q = Softmax(newParameters);
        for (int k = 0; k < p.Length; k++) if (p[k] > 0) kl += p[k] * Math.Log(p[k] / Math.Max(q[k], 1e-300));
        return kl;
    }

    private double[] LogStdArray() => Enumerable.Range(0, _logStd.Length).Select(k => NumOps.ToDouble(_logStd[k])).ToArray();

    private double[] PolicyParameters()
    {
        var network = _policyNetwork.GetParameters();
        var all = new double[network.Length + _logStd.Length];
        for (int i = 0; i < network.Length; i++) all[i] = NumOps.ToDouble(network[i]);
        for (int k = 0; k < _logStd.Length; k++) all[network.Length + k] = NumOps.ToDouble(_logStd[k]);
        return all;
    }

    private void SetPolicyParameters(double[] all)
    {
        int count = all.Length - _logStd.Length;
        var network = new Vector<T>(count);
        for (int i = 0; i < count; i++) network[i] = NumOps.FromDouble(all[i]);
        _policyNetwork.UpdateParameters(network);
        for (int k = 0; k < _logStd.Length; k++) _logStd[k] = NumOps.FromDouble(all[count + k]);
    }

    // Jᵀu: the policy network's parameter gradient of sum_i <output_i, u_i>, a vector-Jacobian product on the tape.
    private double[] VectorJacobianProduct(Tensor<T> states, double[][] u)
    {
        int n = u.Length, a = _options.ActionSize;
        var weights = new Tensor<T>(new[] { n, a });
        for (int i = 0; i < n; i++) for (int k = 0; k < a; k++) weights[i, k] = NumOps.FromDouble(u[i][k]);
        var network = (IGradientComputable<T, Tensor<T>, Tensor<T>>)_policyNetwork;
        var gradient = network.ComputeGradients(states, weights, new OutputDotLoss());
        return Enumerable.Range(0, gradient.Length).Select(i => NumOps.ToDouble(gradient[i])).ToArray();
    }

    // Jv by a central difference of the network's outputs along v.
    private double[][] JacobianVectorProduct(Tensor<T> states, double[] theta, double[] v)
    {
        int count = theta.Length - _logStd.Length;
        double norm = Math.Sqrt(v.Take(count).Sum(x => x * x));
        int n = states.Shape[0], a = _options.ActionSize;
        var result = new double[n][];
        for (int i = 0; i < n; i++) result[i] = new double[a];
        if (norm == 0) return result;
        double epsilon = 1e-4 * Math.Max(1, Math.Sqrt(theta.Take(count).Sum(x => x * x))) / norm;
        var plus = (double[])theta.Clone();
        var minus = (double[])theta.Clone();
        for (int i = 0; i < count; i++) { plus[i] += epsilon * v[i]; minus[i] -= epsilon * v[i]; }
        SetPolicyParameters(plus);
        var up = Distribution(states);
        SetPolicyParameters(minus);
        var down = Distribution(states);
        SetPolicyParameters(theta);
        for (int i = 0; i < n; i++) for (int k = 0; k < a; k++) result[i][k] = (up[i][k] - down[i][k]) / (2 * epsilon);
        return result;
    }

    // Appendix C.1: Fv = JᵀMJv averaged over the states, M the Fisher information of the distribution parameters: 1/σ²
    // for the Gaussian means and 2 for each log standard deviation (independent of the state), diag(p) - ppᵀ for the
    // logits. Damping adds a multiple of v.
    private double[] FisherVectorProduct(Tensor<T> states, double[] theta, double[][] oldDistribution, double[] logStd, double[] v)
    {
        int n = oldDistribution.Length, a = _options.ActionSize, count = theta.Length - _logStd.Length;
        var jv = JacobianVectorProduct(states, theta, v);
        var mjv = new double[n][];
        for (int i = 0; i < n; i++)
        {
            mjv[i] = new double[a];
            if (_options.IsContinuous)
            {
                for (int k = 0; k < a; k++) mjv[i][k] = jv[i][k] / Math.Exp(2 * logStd[k]) / n;
            }
            else
            {
                var p = Softmax(oldDistribution[i]);
                double dot = 0;
                for (int k = 0; k < a; k++) dot += p[k] * jv[i][k];
                for (int k = 0; k < a; k++) mjv[i][k] = p[k] * (jv[i][k] - dot) / n;
            }
        }
        var network = VectorJacobianProduct(states, mjv);
        var result = new double[theta.Length];
        for (int i = 0; i < count; i++) result[i] = network[i] + _options.Damping * v[i];
        for (int k = 0; k < _logStd.Length; k++) result[count + k] = 2 * v[count + k] + _options.Damping * v[count + k];
        return result;
    }

    private double Surrogate(double[][] distribution, double[] logStd, double[][] oldDistribution, double[] oldLogStd,
        List<Vector<T>> actions, double[] advantages)
    {
        double total = 0;
        for (int i = 0; i < advantages.Length; i++)
        {
            double ratio = Math.Exp(LogProbability(distribution[i], logStd, actions[i])
                - LogProbability(oldDistribution[i], oldLogStd, actions[i]));
            total += ratio * advantages[i];
        }
        return total / advantages.Length;
    }

    private double PolicyStep(Tensor<T> states, List<Vector<T>> actions, double[] advantages)
    {
        LastStepFraction = 0.0;
        _lastKl = 0.0;
        int n = advantages.Length, a = _options.ActionSize;
        var theta = PolicyParameters();
        int count = theta.Length - _logStd.Length;
        var oldDistribution = Distribution(states);
        var oldLogStd = LogStdArray();

        // The surrogate's gradient at θold, where every ratio is 1: the policy-gradient d/dθ mean(log π · A).
        var outputGradient = new double[n][];
        var g = new double[theta.Length];
        for (int i = 0; i < n; i++)
        {
            outputGradient[i] = new double[a];
            if (_options.IsContinuous)
            {
                for (int k = 0; k < a; k++)
                {
                    double variance = Math.Exp(2 * oldLogStd[k]);
                    double diff = NumOps.ToDouble(actions[i][k]) - oldDistribution[i][k];
                    outputGradient[i][k] = advantages[i] * diff / variance / n;
                    g[count + k] += advantages[i] * (diff * diff / variance - 1) / n;
                }
            }
            else
            {
                var p = Softmax(oldDistribution[i]);
                int taken = ArgMax(actions[i]);
                for (int k = 0; k < a; k++) outputGradient[i][k] = advantages[i] * ((k == taken ? 1 : 0) - p[k]) / n;
            }
        }
        var networkGradient = VectorJacobianProduct(states, outputGradient);
        for (int i = 0; i < count; i++) g[i] = networkGradient[i];
        if (g.All(v => v == 0)) { _lastKl = 0; return 0; }

        // Conjugate gradient on F s = g.
        var s = new double[theta.Length];
        var r = (double[])g.Clone();
        var p2 = (double[])g.Clone();
        double rr = Dot(r, r);
        for (int it = 0; it < _options.ConjugateGradientIterations && rr > 1e-10; it++)
        {
            var fp = FisherVectorProduct(states, theta, oldDistribution, oldLogStd, p2);
            double alpha = rr / Math.Max(Dot(p2, fp), 1e-300);
            for (int i = 0; i < s.Length; i++) { s[i] += alpha * p2[i]; r[i] -= alpha * fp[i]; }
            double next = Dot(r, r);
            for (int i = 0; i < p2.Length; i++) p2[i] = r[i] + next / rr * p2[i];
            rr = next;
        }

        // β = sqrt(2δ / sᵀFs): the largest step along s whose quadratic KL estimate is δ.
        double delta = NumOps.ToDouble(_options.MaxKL);
        double sFs = Dot(s, FisherVectorProduct(states, theta, oldDistribution, oldLogStd, s));
        if (!(sFs > 0)) { _lastKl = 0; return 0; }
        double beta = Math.Sqrt(2 * delta / sFs);
        double expected = beta * Dot(g, s);
        double before = Surrogate(oldDistribution, oldLogStd, oldDistribution, oldLogStd, actions, advantages);

        // Line search on L(θ) - X[D_KL ≤ δ]: shrink the step exponentially until the surrogate improves (by at least
        // the accept ratio of its linear prediction) within the trust region.
        double fraction = 1;
        for (int step = 0; step < _options.LineSearchSteps; step++, fraction *= _options.LineSearchBacktrackCoeff)
        {
            var candidate = new double[theta.Length];
            for (int i = 0; i < theta.Length; i++) candidate[i] = theta[i] + fraction * beta * s[i];
            SetPolicyParameters(candidate);
            var distribution = Distribution(states);
            var logStd = LogStdArray();
            double kl = 0;
            for (int i = 0; i < n; i++) kl += Kl(oldDistribution[i], oldLogStd, distribution[i], logStd) / n;
            double improvement = Surrogate(distribution, logStd, oldDistribution, oldLogStd, actions, advantages) - before;
            if (kl <= delta && improvement > 0 && improvement >= _options.LineSearchAcceptRatio * fraction * expected)
            {
                _lastKl = kl;
                LastStepFraction = fraction;
                return improvement;
            }
        }
        SetPolicyParameters(theta);
        _lastKl = 0;
        return 0;
    }

    /// <summary>The measured mean KL of the last accepted policy step.</summary>
    internal double LastPolicyKL => _lastKl;

    /// <summary>The fraction of the full natural-gradient step accepted by line search.</summary>
    internal double LastStepFraction { get; private set; }

    /// <summary>The policy probabilities, or Gaussian means followed by log standard deviations.</summary>
    internal Vector<T> PolicyDistribution(Vector<T> state)
    {
        var output = _policyNetwork.Predict(Tensor<T>.FromVector(state)).ToVector();
        if (!_options.IsContinuous)
            return new Vector<T>(Softmax(Enumerable.Range(0, output.Length)
                .Select(i => NumOps.ToDouble(output[i])).ToArray()).Select(NumOps.FromDouble).ToArray());
        var result = new Vector<T>(output.Length + _logStd.Length);
        for (int i = 0; i < output.Length; i++) result[i] = output[i];
        for (int i = 0; i < _logStd.Length; i++) result[output.Length + i] = _logStd[i];
        return result;
    }

    /// <summary>The value network parameters, materialized by the constructor.</summary>
    internal Vector<T> GetValueNetworkParameters() => _valueNetwork.GetParameters();

    /// <summary>The policy's log standard deviations (empty for discrete actions).</summary>
    internal Vector<T> LogStandardDeviations => _logStd;

    /// <summary>
    /// Checks the Fisher-vector product along <paramref name="direction"/>: the quadratic estimate
    /// <c>1/2 ε² vᵀFv</c> (undamped) against the mean KL divergence measured after moving the policy by <c>εv</c>.
    /// </summary>
    internal (double Quadratic, double Measured) FisherCheck(Tensor<T> states, double[] direction, double epsilon)
    {
        var theta = PolicyParameters();
        var oldDistribution = Distribution(states);
        var oldLogStd = LogStdArray();
        double damping = _options.Damping;
        var fv = FisherVectorProduct(states, theta, oldDistribution, oldLogStd, direction);
        double quadratic = 0.5 * epsilon * epsilon * (Dot(direction, fv) - damping * Dot(direction, direction));
        SetPolicyParameters(theta.Select((value, i) => value + epsilon * direction[i]).ToArray());
        var distribution = Distribution(states);
        var logStd = LogStdArray();
        double measured = 0;
        for (int i = 0; i < oldDistribution.Length; i++) measured += Kl(oldDistribution[i], oldLogStd, distribution[i], logStd) / oldDistribution.Length;
        SetPolicyParameters(theta);
        return (quadratic, measured);
    }

    /// <summary>The policy's parameter count: the network's weights and the log standard deviations.</summary>
    internal int PolicyParameterCount => _policyNetwork.GetParameters().Length + _logStd.Length;

    private static double Dot(double[] x, double[] y)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++) sum += x[i] * y[i];
        return sum;
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

    // sum(predicted ⊙ target): its parameter gradient is the vector-Jacobian product Jᵀ target.
    private sealed class OutputDotLoss : LossFunctionBase<T>
    {
        public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < predicted.Length; i++) sum = NumOps.Add(sum, NumOps.Multiply(predicted[i], actual[i]));
            return sum;
        }

        public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
        {
            var product = Engine.TensorMultiply(predicted, target);
            return Engine.ReduceSum(product, Enumerable.Range(0, product.Shape.Length).ToArray(), keepDims: false);
        }
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_trajectoryBuffer.Count),
            ["kl"] = NumOps.FromDouble(_lastKl),
            ["surrogate_improvement"] = NumOps.FromDouble(_lastImprovement),
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
    /// <c>ApplyGradients</c> requires. The gradient comes from the policy network's own tape pass; the value baseline
    /// and the log standard deviations, which the loss does not reach, get zero slices.
    /// </remarks>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var networks = ComputeGradientsForNetwork(
            _policyNetwork,
            new[] { _policyNetwork, _valueNetwork },
            input,
            target,
            lossFunction);
        if (_logStd.Length == 0) return networks;
        var all = new Vector<T>(networks.Length + _logStd.Length);
        for (int i = 0; i < networks.Length; i++) all[i] = networks[i];
        return all;
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
