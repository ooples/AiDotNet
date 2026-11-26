using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Gumbel-Softmax activation function for neural networks, which enables
/// differentiable sampling from discrete distributions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Gumbel-Softmax is a special activation function that helps neural networks
/// make categorical (multiple-choice) decisions while still allowing for gradient-based learning.
/// 
/// Imagine you want your neural network to choose between several options (like choosing a word
/// from a vocabulary). Normally, this would require a non-differentiable "hard" selection, which
/// makes training difficult. Gumbel-Softmax solves this by:
/// 
/// 1. Adding randomness (Gumbel noise) to the inputs
/// 2. Using a "temperature" parameter to control how "certain" the choices are
/// 3. Producing a probability distribution that can approximate discrete choices
/// 
/// At high temperatures, the output is very "soft" (all options get some probability).
/// At low temperatures, the output becomes more like a one-hot vector (one option gets almost all probability).
/// 
/// This technique is widely used in generative models, reinforcement learning, and any neural network
/// that needs to make discrete choices while remaining differentiable.
/// </para>
/// </remarks>
[ActivationCategory(ActivationCategory.Stochastic)]
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationTask(ActivationTask.NormalizationOutput)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.High)]
public class GumbelSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Controls the "sharpness" of the output distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The temperature parameter controls how "decisive" the function is:
    /// - Low temperature (e.g., 0.1): Makes the function choose one option with high confidence
    /// - High temperature (e.g., 5.0): Makes the function more uncertain, spreading probability across options
    /// 
    /// As training progresses, you might want to gradually lower the temperature to make
    /// the model's decisions more definitive.
    /// </para>
    /// </remarks>
    private readonly T _temperature;

    /// <summary>
    /// Random number generator used for sampling Gumbel noise.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// The noise drawn by the most recent <see cref="Activate(Vector{T})"/>, so that
    /// <see cref="Derivative(Vector{T})"/> describes that forward pass instead of a new sample.
    /// </summary>
    private Vector<T>? _lastVectorNoise;

    /// <summary>
    /// Initializes a new instance of the GumbelSoftmaxActivation class.
    /// </summary>
    /// <param name="temperature">Controls the sharpness of the distribution. Default is 1.0.</param>
    /// <param name="seed">Optional seed for the random number generator to ensure reproducible results.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When creating a Gumbel-Softmax activation:
    /// - The temperature parameter controls how "certain" the choices will be (lower = more certain)
    /// - The seed parameter lets you get consistent results across runs (useful for testing)
    /// 
    /// If you're just starting out, the default temperature of 1.0 is a good starting point.
    /// </para>
    /// </remarks>
    public GumbelSoftmaxActivation(double temperature = 1.0, int? seed = null)
    {
        _temperature = NumOps.FromDouble(temperature); // Store the temperature in the activation's numeric type.
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Indicates whether this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns false as Gumbel-Softmax requires a vector of values to operate on.</returns>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Gumbel-Softmax activation function to a vector of input values.
    /// </summary>
    /// <param name="input">The vector of input logits.</param>
    /// <returns>A vector of probabilities that sum to 1.0.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a vector of numbers (logits) into a probability distribution:
    /// 
    /// 1. It adds random Gumbel noise to each input value
    /// 2. It applies the softmax function to convert these values to probabilities
    /// 
    /// The result is a vector where:
    /// - All values are between 0 and 1
    /// - All values sum to exactly 1
    /// - The largest input value typically gets the highest probability
    /// - The randomness allows for exploration of different options during training
    /// 
    /// This is useful when your neural network needs to make choices between multiple options
    /// while still being able to learn through backpropagation.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> gumbel = SampleGumbel(input.Length);
        _lastVectorNoise = gumbel;
        Vector<T> logits = input.Add(gumbel);

        return Softmax(logits);
    }

    /// <summary>
    /// Applies Gumbel-Softmax to a tensor via engine primitives so the gradient tape
    /// records every step. Decomposes <c>softmax((x + g) / τ)</c> where g is a fresh
    /// sample from Gumbel(0, 1) packaged as a constant tensor so gradients flow through
    /// the add but not into the noise. Softmax is applied along the last axis.
    /// </summary>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        if (input.Shape.Length == 0)
        {
            // One logit: its softmax is 1 whatever the noise. The draw is still taken, so the seeded stream
            // advances as it does for every other shape.
            StandardGumbel(input._shape);
            return Engine.TensorOnesLike(input);
        }

        int lastAxis = input.Shape.Length - 1;

        // Sample Gumbel noise g = -log(-log(u)) with u ~ U(0, 1).
        var noise = StandardGumbel(input._shape);

        var withNoise = Engine.TensorAdd(input, noise);
        var scaled = Engine.TensorMultiplyScalar(withNoise, NumOps.Divide(NumOps.One, _temperature));
        return Engine.Softmax(scaled, axis: lastAxis);
    }

    /// <summary>
    /// Calculates the derivative (Jacobian matrix) of the Gumbel-Softmax function for a vector of input values.
    /// </summary>
    /// <param name="input">The vector of input logits.</param>
    /// <returns>A matrix representing the partial derivatives of each output with respect to each input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how changes in the input affect the output.
    /// 
    /// The result is a matrix where:
    /// - Each row corresponds to an output value
    /// - Each column corresponds to an input value
    /// - Each cell shows how much that particular output changes when the corresponding input changes slightly
    /// 
    /// This is essential for neural network training, as it tells the network how to adjust its weights
    /// to improve performance. The temperature parameter affects these derivatives - lower temperatures
    /// lead to larger derivatives for the winning category and smaller derivatives for others.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        if (input.Length == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // The Jacobian of the forward pass that was just run: reuse its noise. Only when there was none of this
        // length is a sample drawn (and kept).
        Vector<T> gumbel;
        if (_lastVectorNoise is { } last && last.Length == input.Length)
        {
            gumbel = last;
        }
        else
        {
            gumbel = SampleGumbel(input.Length);
            _lastVectorNoise = gumbel;
        }

        // J = (diag(y) - y y^T) / temperature
        var engine = Engine;
        T invTemp = NumOps.Divide(NumOps.One, _temperature);
        if (UseScalarJacobian(input.Length))
        {
            return ScalarSoftmaxJacobian(Softmax(input.Add(gumbel)), invTemp);
        }

        var output = Tensor<T>.FromVector(Softmax(input.Add(gumbel)));
        var negated = engine.TensorNegate(output);
        var oneMinus = engine.TensorAddScalar(negated, NumOps.One);
        return OuterWithDiagonal(
            engine.TensorMultiplyScalar(negated, invTemp), output,
            engine.TensorMultiplyScalar(engine.TensorMultiply(output, oneMinus), invTemp));
    }

    /// <summary>
    /// Generates a vector of random values from the Gumbel distribution.
    /// </summary>
    /// <param name="size">The size of the vector to generate.</param>
    /// <returns>A vector of Gumbel-distributed random values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates the randomness needed for Gumbel-Softmax.
    /// 
    /// The Gumbel distribution is a special probability distribution that, when added to values
    /// before applying softmax, effectively simulates random sampling from a categorical distribution.
    /// 
    /// This randomness is important because:
    /// - It allows the network to explore different options during training
    /// - It makes the process of selecting categories differentiable
    /// - It helps prevent the network from getting stuck in suboptimal solutions
    /// 
    /// The formula used is: -log(-log(u)), where u is a uniform random value between 0 and 1. The temperature is
    /// applied afterwards, when the noisy values are divided by it inside the softmax.
    /// </para>
    /// </remarks>
    private Vector<T> SampleGumbel(int size)
    {
        if (size == 0)
        {
            return new Vector<T>(0);
        }

        // y = softmax((x + g) / temperature) with g standard Gumbel noise (Jang et al., 2017, equation 1):
        // the temperature divides the noisy logits once, in Softmax below, and does not scale the noise
        return StandardGumbel(new[] { size }).ToVector();
    }

    /// <summary>
    /// Draws standard Gumbel noise -log(-log(u)) with u ~ U(0, 1), one draw per element in row-major
    /// order from this activation's generator.
    /// </summary>
    /// <param name="shape">The shape of the noise tensor.</param>
    /// <returns>The noise tensor.</returns>
    private Tensor<T> StandardGumbel(int[] shape)
    {
        // Both logarithms floor their argument at the stability epsilon, as SafeLog does
        var engine = Engine;
        T floor = NumOps.FromDouble(NumericalStabilityHelper.DefaultEpsilon);
        // CreateRandom needs at least one dimension; a rank-0 tensor is one draw
        var uniform = shape.Length == 0
            ? new Tensor<T>(Tensor<T>.CreateRandom(_random, new[] { 1 }).ToArray(), shape)
            : Tensor<T>.CreateRandom(_random, shape);
        var inner = engine.TensorNegate(engine.TensorLog(engine.TensorClampMin(uniform, floor)));
        return engine.TensorNegate(engine.TensorLog(engine.TensorClampMin(inner, floor)));
    }

    /// <summary>
    /// Applies the softmax function to a vector of logits, using the current temperature.
    /// </summary>
    /// <param name="logits">The input logits.</param>
    /// <returns>A vector of probabilities that sum to 1.0.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts a vector of any numbers into a probability distribution.
    /// 
    /// The softmax function works by:
    /// 1. Taking the exponential (e^x) of each input value divided by the temperature
    /// 2. Dividing each result by the sum of all the exponentials
    /// 
    /// This ensures that:
    /// - All output values are positive
    /// - All output values sum to exactly 1
    /// - Larger input values get assigned larger probabilities
    /// - The temperature controls how "peaked" the distribution is
    /// 
    /// Lower temperatures make the largest input much more probable than others,
    /// while higher temperatures make the distribution more uniform.
    /// </para>
    /// </remarks>
    private Vector<T> Softmax(Vector<T> logits)
    {
        if (logits.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Shifted by the largest scaled logit, which leaves the distribution unchanged: at a low temperature the
        // scaled noisy logits reach the thousands, and exp of those overflowed the sum to infinity and the
        // probabilities to NaN.
        Vector<T> scaled = logits.Transform(x => NumOps.Divide(x, _temperature));
        T maximum = Engine.TensorMaxValue(Tensor<T>.FromVector(scaled));
        Vector<T> expValues = scaled.Transform(x => NumOps.Exp(NumOps.Subtract(x, maximum)));
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

}
