using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Boltzmann (softmax) exploration with temperature-based action selection.
    /// Uses temperature parameter to control exploration: higher temperature = more random.
    /// Action probability: P(a) = exp(Q(a)/τ) / Σ exp(Q(a')/τ)
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// var random = new Random(42);
    /// var state = new Vector&lt;double&gt;(new double[] { 0.1, -0.4, 0.7, 0.2 });
    /// // Create Boltzmann exploration with temperature decay
    /// var exploration = new BoltzmannExploration&lt;double&gt;(temperatureStart: 1.0, temperatureEnd: 0.01, temperatureDecay: 0.995);
    ///
    /// // Get an exploration action weighted by Q-value estimates
    /// var policyAction = new Vector&lt;double&gt;(new double[] { 1.5, 0.8, 2.3 });
    /// var action = exploration.GetExplorationAction(state, policyAction, actionSpaceSize: 3, random);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelTask(ModelTask.Classification)]
    [ModelComplexity(ModelComplexity.Low)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Reinforcement Learning: An Introduction",
        "https://incompleteideas.net/book/the-book-2nd.html",
        Year = 2018,
        Authors = "Sutton, R. S. & Barto, A. G.")]
    public class BoltzmannExploration<T> : ExplorationStrategyBase<T>
    {
        private double _temperature;
        private readonly double _temperatureStart;
        private readonly double _temperatureEnd;
        private readonly double _temperatureDecay;

        /// <summary>
        /// Initializes a new instance of the Boltzmann exploration strategy.
        /// </summary>
        /// <param name="temperatureStart">Initial temperature (default: 1.0).</param>
        /// <param name="temperatureEnd">Minimum temperature (default: 0.01).</param>
        /// <param name="temperatureDecay">Temperature decay rate per update (default: 0.995).</param>
        public BoltzmannExploration(
            double temperatureStart = 1.0,
            double temperatureEnd = 0.01,
            double temperatureDecay = 0.995)
        {
            _temperatureStart = temperatureStart;
            _temperatureEnd = temperatureEnd;
            _temperatureDecay = temperatureDecay;
            _temperature = temperatureStart;
        }

        /// <summary>
        /// Applies Boltzmann (softmax) exploration to select an action.
        /// </summary>
        public override Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random)
        {
            // For discrete actions, apply softmax with temperature to the policy action (assumed to be logits or Q-values)
            // For continuous actions, treat policyAction as mean and sample with temperature-scaled noise

            // Check if this is discrete (one-hot) or continuous
            bool isDiscrete = IsOneHot(policyAction);

            if (isDiscrete)
            {
                // Apply softmax with temperature
                var probabilities = SoftmaxWithTemperature(policyAction);
                return SampleCategorical(probabilities, random);
            }
            else
            {
                // For continuous: add temperature-scaled Gaussian noise
                var engine = AiDotNetEngine.Current;
                var noise = engine.Multiply(RLTensorOps.StandardNormal<T>(random, actionSpaceSize), NumOps.FromDouble(_temperature));
                var noisyAction = engine.Add(RLTensorOps.Head(policyAction, actionSpaceSize), noise);
                return ClampAction(noisyAction);
            }
        }

        /// <summary>
        /// Updates the temperature using exponential decay.
        /// </summary>
        public override void Update()
        {
            _temperature = Math.Max(_temperatureEnd, _temperature * _temperatureDecay);
        }

        /// <summary>
        /// Resets the temperature to its initial value.
        /// </summary>
        public override void Reset()
        {
            _temperature = _temperatureStart;
        }

        /// <summary>
        /// Gets the current temperature value.
        /// </summary>
        public double CurrentTemperature => _temperature;

        // Helper methods

        private Vector<T> SoftmaxWithTemperature(Vector<T> logits)
        {
            // Find max for numerical stability
            T maxLogit = RLTensorOps.Max(logits);

            // exp((logit - max) / temperature), normalized
            var engine = AiDotNetEngine.Current;
            var expValues = engine.TensorExp(engine.TensorDivideScalar(
                engine.TensorSubtractScalar(logits.AsTensor(), maxLogit), NumOps.FromDouble(_temperature)));
            return engine.TensorDivideScalar(expValues, engine.TensorSum(expValues)).ToVector();
        }

        private Vector<T> SampleCategorical(Vector<T> probabilities, Random random)
        {
            // First index whose running probability reaches the draw; the last action when rounding falls short.
            double randomValue = random.NextDouble();
            int index = RLTensorOps.SampleFromCdf(probabilities, randomValue, probabilities.Length - 1);
            return RLTensorOps.OneHot<T>(probabilities.Length, index);
        }

        private bool IsOneHot(Vector<T> action)
        {
            // One-hot: exactly one element within 1e-6 of one and no other element farther than 1e-6 from zero.
            // Every near-one element is also non-zero, so that is one non-zero element which is the near-one one
            // (a NaN is neither, as in the scalar test).
            var engine = AiDotNetEngine.Current;
            var values = action.AsTensor();
            T tolerance = NumOps.FromDouble(1e-6);
            var nearOne = engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtractScalar(values, NumOps.One)), tolerance);
            var nonZero = engine.TensorGreaterThan(engine.TensorAbs(values), tolerance);
            T one = NumOps.One;
            return NumOps.Equals(engine.TensorSum(nearOne), one) && NumOps.Equals(engine.TensorSum(nonZero), one);
        }
    }
}
