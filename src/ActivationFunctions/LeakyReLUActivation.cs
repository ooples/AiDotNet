namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Leaky ReLU activation function is a variation of the standard ReLU function.
/// 
/// How it works:
/// - For positive inputs (x > 0): It returns the input unchanged (like a straight line)
/// - For negative inputs (x = 0): It returns a small fraction of the input (a * x)
/// 
/// The main advantage of Leaky ReLU over standard ReLU is that it never completely "turns off" 
/// neurons for negative inputs. Instead, it allows a small gradient to flow through, which helps
/// prevent the "dying ReLU" problem where neurons can stop learning during training.
/// 
/// Think of it like a water pipe that:
/// - Allows full flow when the input is positive
/// - Allows a small "leak" when the input is negative (controlled by the alpha parameter)
/// </para>
/// </remarks>
public class LeakyReLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The slope coefficient for negative input values.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the Leaky ReLU activation function with the specified alpha parameter.
    /// </summary>
    /// <param name="alpha">
    /// The slope coefficient for negative input values. Default value is 0.01.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The alpha parameter determines how much of the negative inputs "leak through":
    /// 
    /// - With alpha = 0.01 (default), negative inputs are multiplied by 0.01 (reduced to 1% of their value)
    /// - With alpha = 0.1, negative inputs are multiplied by 0.1 (reduced to 10% of their value)
    /// - With alpha = 0.001, negative inputs are multiplied by 0.001 (reduced to 0.1% of their value)
    /// 
    /// A larger alpha means more information flows through for negative inputs, which can help with learning
    /// but might make the network less focused on positive features. The default value of 0.01 works well
    /// for most applications, but you can adjust it based on your specific needs.
    /// </para>
    /// </remarks>
    public LeakyReLUActivation(double alpha = 0.01)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as the Leaky ReLU function can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Leaky ReLU activation function to a single value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>
    /// The input value if it's positive, or the input value multiplied by alpha if it's negative or zero.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the formula:
    /// 
    /// f(x) = x        if x > 0
    /// f(x) = a * x    if x = 0
    /// 
    /// For example, with the default a = 0.01:
    /// - Input of 5 ? Output of 5 (unchanged)
    /// - Input of 0 ? Output of 0
    /// - Input of -5 ? Output of -0.05 (5 * 0.01)
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? input : NumOps.Multiply(_alpha, input);
    }

    /// <summary>
    /// Applies the Leaky ReLU activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A new vector with the Leaky ReLU function applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method applies the Leaky ReLU function to each value in a collection (vector)
    /// of inputs. It processes each number individually using the same rules as the single-value version.
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => Activate(x));
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Leaky ReLU function for a single value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>1 if the input is positive, or alpha if the input is negative or zero.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// This information is crucial during neural network training.
    /// 
    /// For the Leaky ReLU function, the derivative is very simple:
    /// - For positive inputs (x > 0): The derivative is 1 (output changes at the same rate as input)
    /// - For negative inputs (x = 0): The derivative is alpha (output changes at alpha times the rate of input)
    /// 
    /// Unlike some other activation functions, Leaky ReLU's derivative never becomes zero,
    /// which helps prevent neurons from "dying" during training.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? NumOps.One : _alpha;
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Leaky ReLU function for a vector of values.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A diagonal matrix containing the derivatives for each input value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how the output vector changes when we slightly change each input value.
    /// 
    /// The result is a special matrix called a "Jacobian matrix" where:
    /// - Values on the main diagonal (top-left to bottom-right) are the derivatives for each input
    /// - All other values are 0
    /// 
    /// This diagonal structure indicates that each output is affected only by its corresponding input,
    /// with no cross-interactions between different elements.
    /// 
    /// For Leaky ReLU, each diagonal value will be either:
    /// - 1 (for inputs > 0)
    /// - alpha (for inputs = 0)
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int size = input.Length;
        Matrix<T> jacobian = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = Derivative(input[i]);
                }
                else
                {
                    jacobian[i, j] = NumOps.Zero;
                }
            }
        }

        return jacobian;
    }
}