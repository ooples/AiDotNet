using System;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.JIT;

/// <summary>
/// IR operation for ReLU (Rectified Linear Unit) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// ReLU(x) = max(0, x). Most commonly used activation in modern deep learning.
/// </para>
/// </remarks>
public class ReLUOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;

    public ReLUOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return _engine.ReLU(input);
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("ReLU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for GELU (Gaussian Error Linear Unit) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution.
/// Commonly used in transformers (BERT, GPT) and modern architectures.
/// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// </para>
/// </remarks>
public class GeluOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;

    public GeluOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return _engine.GELU(input);
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("GELU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for ELU (Exponential Linear Unit) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise.
/// Helps with vanishing gradient problem and can produce negative outputs.
/// </para>
/// </remarks>
public class EluOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private readonly double _alpha;

    public EluOp(IEngine engine, double alpha = 1.0)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
        _alpha = alpha;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return _engine.ELU(input, _alpha);
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("ELU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for SELU (Scaled Exponential Linear Unit) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) otherwise)
/// where α = 1.6732632423543772848170429916717 and λ = 1.0507009873554804934193349852946.
/// Self-normalizing activation that enables self-normalizing neural networks.
/// </para>
/// </remarks>
public class SeluOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private const double Alpha = 1.6732632423543772848170429916717;
    private const double Lambda = 1.0507009873554804934193349852946;

    public SeluOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // SELU is λ * ELU(x, α)
        var eluResult = _engine.ELU(input, Alpha);
        return _engine.TensorMultiplyScalar(eluResult, _numOps.FromDouble(Lambda));
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("SELU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for CELU (Continuously Differentiable Exponential Linear Unit) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// CELU(x) = max(0, x) + min(0, α * (exp(x/α) - 1))
/// Continuously differentiable variant of ELU with parameterized alpha.
/// </para>
/// </remarks>
public class CeluOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private readonly double _alpha;

    public CeluOp(IEngine engine, double alpha = 1.0)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        if (alpha <= 0)
            throw new ArgumentException("Alpha must be positive", nameof(alpha));
        _engine = engine;
        _alpha = alpha;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // For now, use ELU as approximation - full CELU implementation requires element-wise operations
        throw new NotImplementedException("CELU forward pass requires element-wise tensor operations not yet in IEngine");
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("CELU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for LeakyReLU activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// LeakyReLU(x) = x if x > 0, negativeSlope * x otherwise (default negativeSlope = 0.01).
/// Variant of ReLU that allows small negative values to prevent dying ReLU problem.
/// </para>
/// </remarks>
public class LeakyReLUOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private readonly double _negativeSlope;

    public LeakyReLUOp(IEngine engine, double negativeSlope = 0.01)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        if (negativeSlope < 0)
            throw new ArgumentException("Negative slope must be non-negative", nameof(negativeSlope));
        _engine = engine;
        _negativeSlope = negativeSlope;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        throw new NotImplementedException("LeakyReLU forward pass requires element-wise conditional operations not yet in IEngine");
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("LeakyReLU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for PReLU (Parametric ReLU) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// PReLU(x) = x if x > 0, α * x otherwise where α is a learnable parameter (per channel).
/// Adaptive variant of LeakyReLU where the negative slope is learned during training.
/// </para>
/// </remarks>
public class PReLUOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private Tensor<T> _alpha;

    public PReLUOp(IEngine engine, Tensor<T> alpha)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        if (alpha == null)
            throw new ArgumentNullException(nameof(alpha));
        _engine = engine;
        _alpha = alpha;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        throw new NotImplementedException("PReLU forward pass requires element-wise conditional and broadcasting operations not yet in IEngine");
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("PReLU backward pass requires derivative computation and parameter gradient");
    }
}

/// <summary>
/// IR operation for RReLU (Randomized Leaky ReLU) activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// RReLU(x) = x if x > 0, α * x otherwise where α is randomly sampled from uniform distribution
/// during training (typically [1/8, 1/3]) and fixed to expectation during inference.
/// Helps with regularization by introducing randomness.
/// </para>
/// </remarks>
public class RReLUOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private readonly double _lowerBound;
    private readonly double _upperBound;
    private readonly bool _training;

    public RReLUOp(IEngine engine, double lowerBound = 0.125, double upperBound = 0.333, bool training = false)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        if (lowerBound < 0 || upperBound < lowerBound)
            throw new ArgumentException("Invalid bounds for RReLU");
        _engine = engine;
        _lowerBound = lowerBound;
        _upperBound = upperBound;
        _training = training;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        throw new NotImplementedException("RReLU forward pass requires random number generation and conditional operations not yet in IEngine");
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("RReLU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for Thresholded ReLU activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// ThresholdedReLU(x) = x if x > threshold, 0 otherwise (default threshold = 1.0).
/// Only activates when input exceeds a threshold, creating sparsity in activations.
/// </para>
/// </remarks>
public class ThresholdedReLUOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;
    private readonly double _threshold;

    public ThresholdedReLUOp(IEngine engine, double threshold = 1.0)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
        _threshold = threshold;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        throw new NotImplementedException("ThresholdedReLU forward pass requires element-wise conditional operations not yet in IEngine");
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("ThresholdedReLU backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for Sigmoid activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// Sigmoid(x) = 1 / (1 + exp(-x)).
/// Commonly used for binary classification and gate functions in LSTMs/GRUs.
/// </para>
/// </remarks>
public class SigmoidOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;

    public SigmoidOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return _engine.Sigmoid(input);
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("Sigmoid backward pass requires derivative computation");
    }
}

/// <summary>
/// IR operation for Tanh activation.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor.</typeparam>
/// <remarks>
/// <para>
/// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
/// Commonly used in hidden layers of neural networks.
/// </para>
/// </remarks>
public class TanhOp<T> : IROp where T : struct
{
    private readonly IEngine _engine;

    public TanhOp(IEngine engine)
    {
        if (engine == null)
            throw new ArgumentNullException(nameof(engine));
        _engine = engine;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return _engine.Tanh(input);
    }

    public Tensor<T> Backward(Tensor<T> input, Tensor<T> gradOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        throw new NotImplementedException("Tanh backward pass requires derivative computation");
    }
}
