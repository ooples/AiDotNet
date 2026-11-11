using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Multi-modal policy using mixture of Gaussians for complex action distributions.
    /// </summary>
    /// <example>
    /// <code>
    /// // Create a multi-modal policy with mixture of Gaussians
    /// var network = new NeuralNetwork&lt;double&gt;();
    /// var policy = new MultiModalPolicy&lt;double&gt;(network, actionSize: 2, numComponents: 3,
    ///     new EpsilonGreedyExploration&lt;double&gt;());
    ///
    /// // Sample an action from the mixture distribution
    /// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
    /// var action = policy.SelectAction(state, training: true);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Mixture Density Networks",
        "https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf",
        Year = 1994,
        Authors = "Bishop, C. M.")]
    public partial class MultiModalPolicy<T> : PolicyBase<T>
    {
        private readonly INeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly int _numComponents;

        /// <summary>
        /// Initializes a new instance with default settings.
        /// </summary>
        public MultiModalPolicy()
            : this(
                new NeuralNetwork<T>(),
                2,
                3,
                new EpsilonGreedyExploration<T>())
        {
        }

        /// <summary>
        /// Builds the policy from <paramref name="options"/>: a dense ReLU network of the options' hidden widths from
        /// StateSize inputs to a mixing logit, ActionSize means and ActionSize log standard deviations per component
        /// (NumComponents × (1 + 2 × ActionSize) outputs).
        /// </summary>
        /// <param name="options">The policy's configuration.</param>
        public MultiModalPolicy(MultiModalPolicyOptions<T> options)
            : this(
                PolicyNetworks.Build((options ?? throw new ArgumentNullException(nameof(options))).StateSize,
                    options.HiddenLayers,
                    options.ActionSize > 0 ? options.NumComponents * (1 + 2 * options.ActionSize) : 0,
                    options.LossFunction, nameof(MultiModalPolicyOptions<T>)),
                options.ActionSize, options.NumComponents, options.ExplorationStrategy,
                PolicyNetworks.RandomFor(options.Seed))
        {
        }

        public MultiModalPolicy(
            INeuralNetwork<T> policyNetwork,
            int actionSize,
            int numComponents,
            IExplorationStrategy<T> explorationStrategy,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
            _numComponents = numComponents;
        }

        /// <summary>
        /// Initializes a new instance of the MultiModalPolicy class from its options: action size, mixture
        /// component count, exploration strategy and <see cref="AiDotNet.Models.Options.ModelOptions.Seed"/>.
        /// </summary>
        /// <param name="policyNetwork">Network producing the mixture parameters.</param>
        /// <param name="options">The policy options.</param>
        public MultiModalPolicy(INeuralNetwork<T> policyNetwork, MultiModalPolicyOptions<T> options)
            : this(
                policyNetwork,
                (options ?? throw new ArgumentNullException(nameof(options))).ActionSize,
                options.NumComponents,
                options.ExplorationStrategy,
                RandomFromSeed(options.Seed))
        {
        }

        /// <summary>
        /// The mode of a mixture of diagonal Gaussians: the highest-density point found by the fixed-point
        /// (mean-shift) iteration started from every component mean (Carreira-Perpinan 2000, "Mode-finding for mixtures
        /// of Gaussian distributions").
        /// </summary>
        /// <remarks>
        /// Each step moves x to the posterior-weighted, precision-weighted mean of the component means; from any start
        /// it climbs the mixture density to a stationary point. Starting it from every component mean is the paper's
        /// procedure for finding the modes; the converged point of highest mixture density is returned.
        /// </remarks>
        internal static double[] MixtureMode(double[] weights, double[][] means, double[][] logStds)
        {
            const int MaxIterations = 500;
            const double Tolerance = 1e-12;
            int components = weights.Length, dims = means[0].Length;
            var precisions = new double[components][];
            for (int k = 0; k < components; k++)
            {
                precisions[k] = new double[dims];
                for (int i = 0; i < dims; i++) precisions[k][i] = Math.Exp(-2.0 * logStds[k][i]);
            }

            double[] best = (double[])means[0].Clone();
            double bestDensity = double.NegativeInfinity;
            var logTerms = new double[components];
            for (int start = 0; start < components; start++)
            {
                var x = (double[])means[start].Clone();
                for (int iteration = 0; iteration < MaxIterations; iteration++)
                {
                    // Posterior responsibilities p(k | x), from log terms for stability.
                    double maxTerm = double.NegativeInfinity;
                    for (int k = 0; k < components; k++)
                    {
                        logTerms[k] = LogComponentDensity(x, weights[k], means[k], logStds[k], precisions[k]);
                        if (logTerms[k] > maxTerm) maxTerm = logTerms[k];
                    }
                    double change = 0.0;
                    for (int i = 0; i < dims; i++)
                    {
                        double numerator = 0.0, denominator = 0.0;
                        for (int k = 0; k < components; k++)
                        {
                            double responsibility = Math.Exp(logTerms[k] - maxTerm);
                            numerator += responsibility * precisions[k][i] * means[k][i];
                            denominator += responsibility * precisions[k][i];
                        }
                        double next = numerator / denominator;
                        change = Math.Max(change, Math.Abs(next - x[i]));
                        x[i] = next;
                    }
                    if (change <= Tolerance * (1.0 + MaxAbs(x))) break;
                }

                double density = MixtureLogDensity(x, weights, means, logStds, precisions);
                if (density > bestDensity)
                {
                    bestDensity = density;
                    best = x;
                }
            }
            return best;
        }

        private static double MaxAbs(double[] x)
        {
            double m = 0.0;
            foreach (double v in x) m = Math.Max(m, Math.Abs(v));
            return m;
        }

        // log(w_k N(x; mu_k, diag(sigma_k^2))).
        private static double LogComponentDensity(double[] x, double weight, double[] mean, double[] logStd, double[] precision)
        {
            double log = Math.Log(Math.Max(weight, double.Epsilon));
            for (int i = 0; i < x.Length; i++)
            {
                double d = x[i] - mean[i];
                log -= 0.5 * d * d * precision[i] + logStd[i] + 0.5 * Math.Log(2.0 * Math.PI);
            }
            return log;
        }

        private static double MixtureLogDensity(double[] x, double[] weights, double[][] means, double[][] logStds,
            double[][] precisions)
        {
            double max = double.NegativeInfinity;
            var terms = new double[weights.Length];
            for (int k = 0; k < weights.Length; k++)
            {
                terms[k] = LogComponentDensity(x, weights[k], means[k], logStds[k], precisions[k]);
                if (terms[k] > max) max = terms[k];
            }
            double sum = 0.0;
            foreach (double t in terms) sum += Math.Exp(t - max);
            return max + Math.Log(sum);
        }

        /// <inheritdoc />
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            int outputSize = _numComponents * (1 + 2 * _actionSize);
            if (output.Length != outputSize)
            {
                throw new InvalidOperationException(
                    string.Format("Network output size {0} does not match expected size {1}",
                        output.Length, outputSize));
            }

            // Output layout: [mixing logits (K) | means (K x A) | log standard deviations (K x A)].
            int meanOffset = _numComponents;
            int logStdOffset = _numComponents + _numComponents * _actionSize;
            var mixingCoefficients = Softmax(output.GetSubVector(0, _numComponents));
            var means = Enumerable.Range(0, _numComponents)
                .Select(component => output.GetSubVector(meanOffset + component * _actionSize, _actionSize)).ToList();
            var logStds = Enumerable.Range(0, _numComponents)
                .Select(component => output.GetSubVector(logStdOffset + component * _actionSize, _actionSize)).ToList();

            var action = new Vector<T>(_actionSize);
            if (!training)
            {
                // Inference takes the mixture's mode: the highest-density point of the whole weighted mixture, which
                // need not be the heaviest component's mean (a broad heavy component can peak below a narrow light one).
                // MixtureMode is the double-precision mean-shift solver; the parameters cross into it in bulk.
                var weights = ToDoubles(mixingCoefficients);
                var componentMeans = means.Select(ToDoubles).ToArray();
                var componentLogStds = logStds.Select(ToDoubles).ToArray();
                var mode = MixtureMode(weights, componentMeans, componentLogStds);
                return Vector<double>.FromArray(mode).AsTensor().Cast<T>().ToVector();
            }

            // Training samples a component, then its Gaussian.
            int selectedComponent = SampleCategoricalIndex(mixingCoefficients);
            for (int i = 0; i < _actionSize; i++)
            {
                double meanValue = NumOps.ToDouble(means[selectedComponent][i]);
                double stdValue = Math.Exp(NumOps.ToDouble(logStds[selectedComponent][i]));

                action[i] = NumOps.FromDouble(meanValue + stdValue * _random.NextGaussian());
            }

            return _explorationStrategy.GetExplorationAction(state, action, _actionSize, _random);
        }

        /// <inheritdoc />
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            ValidateState(state, nameof(state));
            ValidateActionSize(_actionSize, action.Length, nameof(action));

            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            // Output layout: [mixing logits (K) | means (K x A) | log standard deviations (K x A)].
            int components = _numComponents;
            int width = _actionSize;
            var engine = AiDotNetEngine.Current;
            var mixingCoefficients = Softmax(output.GetSubVector(0, components));
            var means = engine.Reshape(output.GetSubVector(components, components * width).AsTensor(), [components, width]);
            var logStds = engine.Reshape(
                output.GetSubVector(components + components * width, components * width).AsTensor(), [components, width]);
            var actions = engine.TensorTile(engine.Reshape(action.AsTensor(), [1, width]), [components, 1]);

            // log N(a; mu_k, sigma_k) = sum_i [-0.5 ((a_i - mu_ki) / sigma_ki)^2 - log sigma_ki] - 0.5 A log(2 pi)
            var z = engine.TensorDivide(engine.TensorSubtract(actions, means), engine.TensorExp(logStds));
            var perDimension = engine.TensorSubtract(engine.TensorMultiplyScalar(engine.TensorMultiply(z, z), NumOps.FromDouble(-0.5)), logStds);
            var componentLogProb = engine.TensorSubtractScalar(
                engine.ReduceSum(perDimension, [1], false), NumOps.FromDouble(0.5 * Math.Log(2.0 * Math.PI) * width));

            // log(sum_k w_k exp(log N_k) + 1e-8)
            T totalProb = engine.TensorSum(engine.TensorMultiply(mixingCoefficients.AsTensor(), engine.TensorExp(componentLogProb)));
            return NumOps.Log(NumOps.Add(totalProb, NumOps.FromDouble(1e-8)));
        }

        /// <inheritdoc />
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        /// <inheritdoc />
        public override void Reset()
        {
            _explorationStrategy.Reset();
        }

        private Vector<T> Softmax(Vector<T> logits)
        {
            // exp(z - max z) / sum exp(z - max z)
            return RLTensorOps.StableSoftmax(logits);
        }

        private static double[] ToDoubles(Vector<T> values)
        {
            return values.AsTensor().Cast<double>().ToVector().ToArray();
        }

        private int SampleCategoricalIndex(Vector<T> probabilities)
        {
            // First index whose running probability reaches the uniform draw; the last when rounding falls short.
            double randomValue = _random.NextDouble();
            return RLTensorOps.SampleFromCdf(probabilities, randomValue, probabilities.Length - 1);
        }
    }
}
