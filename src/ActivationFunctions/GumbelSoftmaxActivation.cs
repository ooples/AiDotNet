using AiDotNet.Autodiff;

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
        _temperature = NumOps.FromDouble(temperature);
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
        Vector<T> logits = input.Add(gumbel);

        return Softmax(logits);
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
        Vector<T> output = Activate(input);
        int d = input.Length;
        Matrix<T> jacobian = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(output[i], NumOps.Subtract(NumOps.One, output[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Multiply(NumOps.Negate(output[i]), output[j]);
                }
            }
        }

        // Scale the Jacobian by the inverse temperature
        T invTemp = NumOps.Divide(NumOps.One, _temperature);
        return jacobian.Transform((x, row, col) => NumOps.Multiply(x, invTemp));
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
    /// The formula used is: -log(-log(u)) * temperature, where u is a uniform random value between 0 and 1.
    /// </para>
    /// </remarks>
    private Vector<T> SampleGumbel(int size)
    {
        Vector<T> uniform = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            uniform[i] = NumOps.FromDouble(_random.NextDouble());
        }

        return uniform.Transform(u =>
            NumOps.Multiply(
                NumOps.Negate(
                    NumericalStabilityHelper.SafeLog(
                        NumOps.Negate(
                            NumericalStabilityHelper.SafeLog(u)
                        )
                    )
                ),
                _temperature
            )
        );
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
        Vector<T> expValues = logits.Transform(x => NumOps.Exp(NumOps.Divide(x, _temperature)));
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.GumbelSoftmax provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// Gumbel-Softmax supports JIT compilation with straight-through gradient estimation.
    /// The backward pass computes softmax gradients scaled by inverse temperature.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with GumbelSoftmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.GumbelSoftmax(input) which handles both
    /// forward and backward passes for JIT compilation with differentiable categorical sampling.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double temperature = Convert.ToDouble(_temperature);
        return TensorOperations<T>.GumbelSoftmax(input, temperature, hard: false);
    }
}
