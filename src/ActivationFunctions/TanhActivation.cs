

using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Hyperbolic Tangent (tanh) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Hyperbolic Tangent (tanh) activation function is a popular choice in neural networks.
/// It transforms any input value to an output between -1 and 1, creating an S-shaped curve that's
/// symmetric around the origin.
///
/// Key properties of tanh:
/// - Outputs values between -1 and 1
/// - An input of 0 produces an output of 0
/// - Large positive inputs approach +1
/// - Large negative inputs approach -1
/// - It's zero-centered, which often helps with learning
///
/// When to use tanh:
/// - When you need outputs centered around zero
/// - For hidden layers in many types of neural networks
/// - When dealing with data that naturally has both positive and negative values
///
/// One limitation is the "vanishing gradient problem" - for very large or small inputs,
/// the function's slope becomes very small, which can slow down learning in deep networks.
/// </para>
/// </remarks>
public class TanhActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates that this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as tanh can be applied to scalar values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the tanh activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms any input number into an output between -1 and 1.
    /// The transformation follows an S-shaped curve that passes through the origin (0,0).
    /// 
    /// For example:
    /// - An input of 0 gives an output of 0
    /// - An input of 2 gives an output of about 0.96 (close to 1)
    /// - An input of -2 gives an output of about -0.96 (close to -1)
    /// 
    /// This "squashing" property makes tanh useful for normalizing outputs in neural networks.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        return MathHelper.Tanh(input);
    }

    /// <summary>
    /// Applies the tanh activation function to a vector of input values using SIMD optimization.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector with tanh applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This implementation uses TensorPrimitivesHelper for SIMD-optimized operations (3-6× speedup for float).
    /// For arrays with fewer than 16 elements, it falls back to manual loops.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an entire vector of numbers at once using hardware
    /// acceleration, making it much faster than processing each number separately.
    ///
    /// For example, if you have a vector [0.5, 1.0, -0.5, -1.0]:
    /// - The output would be approximately [0.46, 0.76, -0.46, -0.76]
    /// - All values are computed in parallel using SIMD instructions
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        // Use SIMD-optimized Tanh (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Tanh(input);
    }

    /// <summary>
    /// Applies the tanh activation function to a tensor of input values.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with tanh applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method transforms an entire tensor of numbers at once using hardware
    /// acceleration, making it much faster than processing each number separately.
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return Engine.Tanh(input);
    }

    /// <summary>
    /// Calculates the derivative of the tanh function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the tanh function's output changes
    /// when its input changes slightly. This is crucial during neural network training to determine
    /// how to adjust weights.
    /// 
    /// The formula is: f'(x) = 1 - tanh²(x)
    /// 
    /// Key properties of this derivative:
    /// - It's highest (equal to 1) at x = 0, where the function is steepest
    /// - It approaches zero for very large positive or negative inputs
    /// - This means the network learns most effectively from inputs near zero
    /// 
    /// The "vanishing gradient problem" occurs when inputs are very large in magnitude,
    /// causing very small derivatives that slow down learning.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        T tanh = MathHelper.Tanh(input);
        return NumOps.Subtract(NumOps.One, NumOps.Multiply(tanh, tanh));
    }

    /// <summary>
    /// Calculates the derivative of the tanh function for a tensor input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor containing the derivative values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates how sensitive the tanh output is to changes in the input
    /// for the entire tensor at once. The derivative is 1 - tanh^2(x).
    /// </remarks>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        var tanh = Activate(input);
        var tanhSquared = Engine.TensorMultiply(tanh, tanh);
        var one = Tensor<T>.CreateDefault(input.Shape, NumOps.One);
        return Engine.TensorSubtract(one, tanhSquared);
    }

    /// <summary>
    /// Calculates the backward pass gradient for Tanh using GPU-accelerated fused operation.
    /// </summary>
    /// <param name="input">The input tensor that was used in the forward pass.</param>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method uses a single GPU kernel to compute the gradient,
    /// which is faster than computing derivative and gradient multiplication separately.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> input, Tensor<T> outputGradient)
    {
        // Tanh backward uses forward output: grad = gradOutput * (1 - output^2)
        var tanhOutput = Activate(input);
        return Engine.TanhBackward(outputGradient, tanhOutput);
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because Tanh gradient computation is fully implemented and tested.</value>
    /// <remarks>
    /// <para>
    /// Tanh supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The operation is well-defined and differentiable
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Tanh activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the Tanh activation to TensorOperations&lt;T&gt;.Tanh(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Tanh(input);
    }
}
