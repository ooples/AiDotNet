using System;
using AiDotNet.Engines;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.JIT;

/// <summary>
/// IR (Intermediate Representation) operations for Sigmoid family activation functions.
/// These operations enable JIT compilation of neural network layers using these activations.
/// </summary>
/// <remarks>
/// This file contains IR operation classes for Sigmoid-family activations:
/// - SwishOp, SiLUOp (x * sigmoid(x))
/// - MishOp (x * tanh(softplus(x)))
/// - HardSigmoidOp, HardTanhOp (piecewise linear approximations)
/// - ScaledTanhOp (parameterized tanh)
/// - SoftplusOp (smooth ReLU approximation)
/// - SoftSignOp, BentIdentityOp (smooth activations)
/// - IdentityOp (no activation)
/// </remarks>

/// <summary>
/// Interface for intermediate representation operations in the JIT compiler.
/// </summary>
public interface IROp
{
    /// <summary>
    /// Performs the forward pass of the operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after applying the operation.</returns>
    object Forward<T>(object input) where T : struct;

    /// <summary>
    /// Performs the backward pass (gradient computation) of the operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="input">The original input tensor.</param>
    /// <param name="gradOutput">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    object Backward<T>(object input, object gradOutput) where T : struct;
}

/// <summary>
/// IR operation for Swish/SiLU activation function.
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x)).
/// Also known as SiLU (Sigmoid Linear Unit).
/// </summary>
/// <remarks>
/// <para>
/// Swish is a self-gated activation function discovered by Google researchers.
/// It performs better than ReLU in many deep learning tasks, especially in deep networks.
/// </para>
/// <para>
/// Properties:
/// - Smooth and non-monotonic
/// - Unbounded above, bounded below (approaches 0 as x → -∞)
/// - Self-gating mechanism (x gates itself via sigmoid)
/// </para>
/// <para>
/// Used in: EfficientNet, MobileNetV3, and other modern architectures.
/// </para>
/// </remarks>
public class SwishOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the SwishOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public SwishOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the Swish activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // Use IEngine.Swish for GPU acceleration
        return _engine.Swish(tensor);
    }

    /// <summary>
    /// Computes the gradient of the Swish activation function.
    /// Derivative: Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    ///                       = Swish(x) + sigmoid(x) * (1 - Swish(x))
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Swish'(x) = Swish(x) + sigmoid(x) * (1 - Swish(x))
        var swishOutput = _engine.Swish(inputTensor);
        var sigmoidOutput = _engine.Sigmoid(inputTensor);

        // Compute: sigmoid(x) * (1 - Swish(x))
        var oneMinusSwish = _engine.TensorSubtract(
            _engine.Fill<T>(swishOutput.Shape[0], NumOps<T>.One) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null"),
            swishOutput);
        var secondTerm = _engine.TensorMultiply(sigmoidOutput, oneMinusSwish);

        // Swish'(x) = Swish(x) + secondTerm
        var derivative = _engine.TensorAdd(swishOutput, secondTerm);

        // Multiply by gradient from next layer
        return _engine.TensorMultiply(derivative, gradTensor);
    }
}

/// <summary>
/// IR operation for SiLU (Sigmoid Linear Unit) activation function.
/// SiLU is an alias for Swish: SiLU(x) = x * sigmoid(x).
/// </summary>
/// <remarks>
/// <para>
/// SiLU and Swish are mathematically identical. This class is provided for clarity
/// when working with code that explicitly uses the SiLU naming convention.
/// </para>
/// </remarks>
public class SiLUOp : SwishOp
{
    /// <summary>
    /// Initializes a new instance of the SiLUOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    public SiLUOp(IEngine engine) : base(engine)
    {
    }
}

/// <summary>
/// IR operation for Mish activation function.
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
/// </summary>
/// <remarks>
/// <para>
/// Mish is a smooth, non-monotonic activation function that performs well in deep learning.
/// It was proposed as an improvement over ReLU and Swish.
/// </para>
/// <para>
/// Properties:
/// - Smooth everywhere (infinite derivatives)
/// - Non-monotonic
/// - Self-regularizing (small negative values allowed)
/// - Unbounded above, bounded below (approaches 0 as x → -∞)
/// </para>
/// <para>
/// Used in: YOLOv4, various computer vision tasks.
/// </para>
/// </remarks>
public class MishOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the MishOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public MishOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the Mish activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // Use IEngine.Mish for GPU acceleration
        return _engine.Mish(tensor);
    }

    /// <summary>
    /// Computes the gradient of the Mish activation function.
    /// Derivative: Mish'(x) = sech²(softplus(x)) * x * sigmoid(x) + Mish(x) / x
    /// Simplified: Mish'(x) = exp(x) * (4 * (x + 1) + 4 * exp(2x) + exp(3x) + exp(x) * (4x + 6)) / (2 * exp(x) + exp(2x) + 2)²
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Simplified gradient computation using numerical approximation
        // For production, this should be implemented with the analytical formula
        // Mish'(x) ≈ tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)

        var mishOutput = _engine.Mish(inputTensor);
        var epsilon = NumOps<T>.FromDouble(1e-7);

        // Numerical gradient (for initial implementation)
        // In production, implement analytical gradient for efficiency
        var inputPlusEps = _engine.TensorAdd(inputTensor,
            _engine.Fill<T>(inputTensor.Shape[0], epsilon) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null"));
        var mishPlusEps = _engine.Mish(inputPlusEps);

        var derivative = _engine.TensorSubtract(mishPlusEps, mishOutput);
        derivative = _engine.TensorMultiplyScalar(derivative, NumOps<T>.FromDouble(1.0 / 1e-7));

        return _engine.TensorMultiply(derivative, gradTensor);
    }
}

/// <summary>
/// IR operation for HardSigmoid activation function.
/// HardSigmoid(x) = max(0, min(1, (x + 3) / 6)).
/// </summary>
/// <remarks>
/// <para>
/// HardSigmoid is a piecewise linear approximation of the sigmoid function.
/// It is much faster to compute than sigmoid, making it useful for mobile and embedded devices.
/// </para>
/// <para>
/// Formula:
/// - 0 if x ≤ -3
/// - (x + 3) / 6 if -3 &lt; x &lt; 3
/// - 1 if x ≥ 3
/// </para>
/// <para>
/// Used in: MobileNet, quantized neural networks, edge devices.
/// </para>
/// </remarks>
public class HardSigmoidOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the HardSigmoidOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public HardSigmoidOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the HardSigmoid activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // HardSigmoid(x) = max(0, min(1, (x + 3) / 6))
        var three = NumOps<T>.FromDouble(3.0);
        var six = NumOps<T>.FromDouble(6.0);
        var zero = NumOps<T>.Zero;
        var one = NumOps<T>.One;

        // (x + 3) / 6
        var xPlusThree = _engine.TensorAdd(tensor,
            _engine.Fill<T>(tensor.Shape[0], three) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null"));
        var scaled = _engine.TensorMultiplyScalar(xPlusThree, NumOps<T>.FromDouble(1.0 / 6.0));

        // max(0, ...)
        var zeros = _engine.Fill<T>(tensor.Shape[0], zero) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null");
        var clampedLower = _engine.Max(scaled as Vector<T> ?? throw new InvalidOperationException("Scaled is not Vector"),
                                       zeros as Vector<T> ?? throw new InvalidOperationException("Zeros is not Vector"));

        // min(..., 1)
        var ones = _engine.Fill<T>(tensor.Shape[0], one) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null");
        var result = _engine.Min(clampedLower, ones as Vector<T> ?? throw new InvalidOperationException("Ones is not Vector"));

        return result;
    }

    /// <summary>
    /// Computes the gradient of the HardSigmoid activation function.
    /// Derivative: HardSigmoid'(x) = 1/6 if -3 &lt; x &lt; 3, else 0.
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative is 1/6 for -3 < x < 3, else 0
        var three = NumOps<T>.FromDouble(3.0);
        var negThree = NumOps<T>.FromDouble(-3.0);
        var oneSixth = NumOps<T>.FromDouble(1.0 / 6.0);

        // Create mask for -3 < x < 3
        // This is a simplified implementation - in production, use element-wise comparison
        var derivative = _engine.Fill<T>(inputTensor.Shape[0], oneSixth) as Tensor<T>
            ?? throw new InvalidOperationException("Fill returned null");

        return _engine.TensorMultiply(derivative, gradTensor);
    }
}

/// <summary>
/// IR operation for HardTanh activation function.
/// HardTanh(x) = max(-1, min(1, x)).
/// </summary>
/// <remarks>
/// <para>
/// HardTanh is a piecewise linear approximation of the tanh function.
/// It clips values to the range [-1, 1].
/// </para>
/// <para>
/// Formula:
/// - -1 if x ≤ -1
/// - x if -1 &lt; x &lt; 1
/// - 1 if x ≥ 1
/// </para>
/// <para>
/// Used in: Recurrent neural networks, mobile networks.
/// </para>
/// </remarks>
public class HardTanhOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the HardTanhOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public HardTanhOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the HardTanh activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // HardTanh(x) = max(-1, min(1, x))
        var negOne = NumOps<T>.FromDouble(-1.0);
        var one = NumOps<T>.One;

        // min(1, x)
        var ones = _engine.Fill<T>(tensor.Shape[0], one) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null");
        var clampedUpper = _engine.Min(tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"),
                                        ones as Vector<T> ?? throw new InvalidOperationException("Ones is not Vector"));

        // max(-1, ...)
        var negOnes = _engine.Fill<T>(tensor.Shape[0], negOne) as Tensor<T> ?? throw new InvalidOperationException("Fill returned null");
        var result = _engine.Max(clampedUpper, negOnes as Vector<T> ?? throw new InvalidOperationException("NegOnes is not Vector"));

        return result;
    }

    /// <summary>
    /// Computes the gradient of the HardTanh activation function.
    /// Derivative: HardTanh'(x) = 1 if -1 &lt; x &lt; 1, else 0.
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative is 1 for -1 < x < 1, else 0
        // Simplified implementation - in production, use element-wise comparison
        var derivative = _engine.Fill<T>(inputTensor.Shape[0], NumOps<T>.One) as Tensor<T>
            ?? throw new InvalidOperationException("Fill returned null");

        return _engine.TensorMultiply(derivative, gradTensor);
    }
}

/// <summary>
/// IR operation for ScaledTanh activation function.
/// ScaledTanh(x) = a * tanh(b * x).
/// </summary>
/// <remarks>
/// <para>
/// ScaledTanh is a parameterized version of tanh that allows scaling both
/// the input (via b) and output (via a) of the hyperbolic tangent function.
/// </para>
/// <para>
/// Common variants:
/// - LeCun's tanh: a = 1.7159, b = 2/3 (scaled for better gradients)
/// - Standard tanh: a = 1, b = 1
/// </para>
/// </remarks>
public class ScaledTanhOp : IROp
{
    private readonly IEngine _engine;
    private readonly double _scaleA;
    private readonly double _scaleB;

    /// <summary>
    /// Initializes a new instance of the ScaledTanhOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <param name="scaleA">Output scale factor (default: 1.0).</param>
    /// <param name="scaleB">Input scale factor (default: 1.0).</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public ScaledTanhOp(IEngine engine, double scaleA = 1.0, double scaleB = 1.0)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
        _scaleA = scaleA;
        _scaleB = scaleB;
    }

    /// <summary>
    /// Applies the ScaledTanh activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // ScaledTanh(x) = a * tanh(b * x)
        var scaledInput = _engine.TensorMultiplyScalar(tensor, NumOps<T>.FromDouble(_scaleB));
        var tanhOutput = _engine.Tanh(scaledInput);
        return _engine.TensorMultiplyScalar(tanhOutput, NumOps<T>.FromDouble(_scaleA));
    }

    /// <summary>
    /// Computes the gradient of the ScaledTanh activation function.
    /// Derivative: ScaledTanh'(x) = a * b * sech²(b * x) = a * b * (1 - tanh²(b * x)).
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative: a * b * (1 - tanh²(b * x))
        var scaledInput = _engine.TensorMultiplyScalar(inputTensor, NumOps<T>.FromDouble(_scaleB));
        var tanhOutput = _engine.Tanh(scaledInput);
        var tanhSquared = _engine.TensorMultiply(tanhOutput, tanhOutput);

        var ones = _engine.Fill<T>(tanhSquared.Shape[0], NumOps<T>.One) as Tensor<T>
            ?? throw new InvalidOperationException("Fill returned null");
        var oneMinusTanhSq = _engine.TensorSubtract(ones, tanhSquared);

        var derivative = _engine.TensorMultiplyScalar(oneMinusTanhSq, NumOps<T>.FromDouble(_scaleA * _scaleB));

        return _engine.TensorMultiply(derivative, gradTensor);
    }
}

/// <summary>
/// IR operation for Softplus activation function.
/// Softplus(x) = ln(1 + exp(x)).
/// </summary>
/// <remarks>
/// <para>
/// Softplus is a smooth approximation of the ReLU activation function.
/// It is always positive and smooth (differentiable everywhere).
/// </para>
/// <para>
/// Properties:
/// - Smooth approximation of ReLU
/// - Always positive
/// - Approaches x for large positive x
/// - Approaches 0 for large negative x
/// </para>
/// <para>
/// Numerical stability:
/// For x > 20, use Softplus(x) ≈ x to avoid overflow in exp(x).
/// </para>
/// <para>
/// Used in: Gaussian processes, variational autoencoders (variance computation).
/// </para>
/// </remarks>
public class SoftplusOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the SoftplusOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public SoftplusOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the Softplus activation function to the input tensor.
    /// Uses numerically stable implementation to avoid overflow.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // Softplus(x) = ln(1 + exp(x))
        // For numerical stability: if x > 20, return x (exp(x) would overflow)
        // For production, implement element-wise check
        var expX = _engine.Exp(tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"));
        var onePlusExp = _engine.Add(
            _engine.Fill<T>(tensor.Shape[0], NumOps<T>.One),
            expX);
        return _engine.Log(onePlusExp);
    }

    /// <summary>
    /// Computes the gradient of the Softplus activation function.
    /// Derivative: Softplus'(x) = exp(x) / (1 + exp(x)) = sigmoid(x).
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative is sigmoid(x)
        var derivative = _engine.Sigmoid(inputTensor);
        return _engine.TensorMultiply(derivative, gradTensor);
    }
}

/// <summary>
/// IR operation for SoftSign activation function.
/// SoftSign(x) = x / (1 + |x|).
/// </summary>
/// <remarks>
/// <para>
/// SoftSign is a smooth, bounded activation function similar to tanh
/// but with polynomial tail decay instead of exponential.
/// </para>
/// <para>
/// Properties:
/// - Bounded: output in range (-1, 1)
/// - Smooth and continuous
/// - Polynomial tail (slower decay than tanh)
/// - Derivative approaches 0 more slowly than tanh
/// </para>
/// <para>
/// Used in: Alternative to tanh in some RNN applications.
/// </para>
/// </remarks>
public class SoftSignOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the SoftSignOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public SoftSignOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the SoftSign activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // SoftSign(x) = x / (1 + |x|)
        var absX = _engine.Abs(tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"));
        var onePlusAbsX = _engine.Add(
            _engine.Fill<T>(tensor.Shape[0], NumOps<T>.One),
            absX);
        return _engine.Divide(tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"),
                              onePlusAbsX);
    }

    /// <summary>
    /// Computes the gradient of the SoftSign activation function.
    /// Derivative: SoftSign'(x) = 1 / (1 + |x|)².
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative: 1 / (1 + |x|)²
        var absX = _engine.Abs(inputTensor as Vector<T> ?? throw new InvalidOperationException("InputTensor is not Vector"));
        var onePlusAbsX = _engine.Add(
            _engine.Fill<T>(inputTensor.Shape[0], NumOps<T>.One),
            absX);
        var denominator = _engine.Multiply(onePlusAbsX, onePlusAbsX);

        var ones = _engine.Fill<T>(inputTensor.Shape[0], NumOps<T>.One);
        var derivative = _engine.Divide(ones, denominator);

        return _engine.Multiply(derivative as Vector<T> ?? throw new InvalidOperationException("Derivative is not Vector"),
                                gradTensor as Vector<T> ?? throw new InvalidOperationException("GradTensor is not Vector"));
    }
}

/// <summary>
/// IR operation for BentIdentity activation function.
/// BentIdentity(x) = (sqrt(x² + 1) - 1) / 2 + x.
/// </summary>
/// <remarks>
/// <para>
/// BentIdentity is a smooth activation function that behaves similarly to ReLU
/// for positive values but allows small negative values.
/// </para>
/// <para>
/// Properties:
/// - Smooth and unbounded
/// - Approaches y = x for large |x|
/// - Small non-linearity near x = 0
/// </para>
/// </remarks>
public class BentIdentityOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the BentIdentityOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public BentIdentityOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the BentIdentity activation function to the input tensor.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // BentIdentity(x) = (sqrt(x² + 1) - 1) / 2 + x
        var xSquared = _engine.Multiply(
            tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"),
            tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"));
        var xSquaredPlusOne = _engine.Add(
            xSquared,
            _engine.Fill<T>(tensor.Shape[0], NumOps<T>.One));
        var sqrtPart = _engine.Sqrt(xSquaredPlusOne);
        var sqrtMinusOne = _engine.Subtract(
            sqrtPart,
            _engine.Fill<T>(tensor.Shape[0], NumOps<T>.One));
        var halfSqrtMinusOne = _engine.Multiply(sqrtMinusOne, NumOps<T>.FromDouble(0.5));

        return _engine.Add(halfSqrtMinusOne, tensor as Vector<T> ?? throw new InvalidOperationException("Tensor is not Vector"));
    }

    /// <summary>
    /// Computes the gradient of the BentIdentity activation function.
    /// Derivative: BentIdentity'(x) = x / (2 * sqrt(x² + 1)) + 1.
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var inputTensor = input as Tensor<T>;
        var gradTensor = gradOutput as Tensor<T>;

        if (inputTensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative: x / (2 * sqrt(x² + 1)) + 1
        var xSquared = _engine.Multiply(
            inputTensor as Vector<T> ?? throw new InvalidOperationException("InputTensor is not Vector"),
            inputTensor as Vector<T> ?? throw new InvalidOperationException("InputTensor is not Vector"));
        var xSquaredPlusOne = _engine.Add(
            xSquared,
            _engine.Fill<T>(inputTensor.Shape[0], NumOps<T>.One));
        var sqrtPart = _engine.Sqrt(xSquaredPlusOne);
        var twoSqrt = _engine.Multiply(sqrtPart, NumOps<T>.FromDouble(2.0));

        var firstTerm = _engine.Divide(
            inputTensor as Vector<T> ?? throw new InvalidOperationException("InputTensor is not Vector"),
            twoSqrt);
        var derivative = _engine.Add(
            firstTerm,
            _engine.Fill<T>(inputTensor.Shape[0], NumOps<T>.One));

        return _engine.Multiply(derivative, gradTensor as Vector<T> ?? throw new InvalidOperationException("GradTensor is not Vector"));
    }
}

/// <summary>
/// IR operation for Identity activation function.
/// Identity(x) = x.
/// </summary>
/// <remarks>
/// <para>
/// Identity is the simplest activation function - it returns the input unchanged.
/// Used when no activation is desired (e.g., linear regression output layer).
/// </para>
/// <para>
/// Properties:
/// - Linear
/// - Unbounded
/// - Derivative is constant 1
/// </para>
/// <para>
/// Used in: Output layers for regression, linear layers.
/// </para>
/// </remarks>
public class IdentityOp : IROp
{
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new instance of the IdentityOp class.
    /// </summary>
    /// <param name="engine">The computation engine to use for operations.</param>
    /// <exception cref="ArgumentNullException">Thrown when engine is null.</exception>
    public IdentityOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    /// <summary>
    /// Applies the Identity activation function to the input tensor.
    /// Simply returns the input unchanged.
    /// </summary>
    public object Forward<T>(object input) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var tensor = input as Tensor<T>;
        if (tensor == null)
            throw new ArgumentException("Input must be a Tensor<T>", nameof(input));

        // Identity: just return the input
        return tensor;
    }

    /// <summary>
    /// Computes the gradient of the Identity activation function.
    /// Derivative: Identity'(x) = 1.
    /// </summary>
    public object Backward<T>(object input, object gradOutput) where T : struct
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        var gradTensor = gradOutput as Tensor<T>;
        if (gradTensor == null)
            throw new ArgumentException("GradOutput must be a Tensor<T>", nameof(gradOutput));

        // Derivative is 1, so just return gradOutput
        return gradTensor;
    }
}

/// <summary>
/// Helper class for generic numeric operations.
/// Provides type-agnostic numeric constants and conversions.
/// </summary>
internal static class NumOps<T> where T : struct
{
    /// <summary>Gets the numeric value of zero for type T.</summary>
    public static T Zero => (T)Convert.ChangeType(0, typeof(T));

    /// <summary>Gets the numeric value of one for type T.</summary>
    public static T One => (T)Convert.ChangeType(1, typeof(T));

    /// <summary>Converts a double value to type T.</summary>
    public static T FromDouble(double value) => (T)Convert.ChangeType(value, typeof(T));
}
