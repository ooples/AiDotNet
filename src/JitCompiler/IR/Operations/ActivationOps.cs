namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents ReLU (Rectified Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.ReLU().
/// Computes max(0, x) for each element: result[i] = max(0, a[i]).
/// </para>
/// <para><b>For Beginners:</b> Keeps positive values, zeros out negative values.
///
/// Example:
/// ReLU([-2, -1, 0, 1, 2]) = [0, 0, 0, 1, 2]
///
/// Very common in neural networks because it's simple and effective.
/// </para>
/// </remarks>
public class ReLUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Sigmoid activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Sigmoid().
/// Computes sigmoid function: result[i] = 1 / (1 + exp(-a[i])).
/// Output range is (0, 1).
/// </para>
/// <para><b>For Beginners:</b> Squashes values to between 0 and 1.
///
/// Example:
/// Sigmoid([-∞, -2, 0, 2, ∞]) ≈ [0, 0.12, 0.5, 0.88, 1]
///
/// Used for binary classification (outputs can be interpreted as probabilities).
/// </para>
/// </remarks>
public class SigmoidOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Tanh (hyperbolic tangent) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Tanh().
/// Computes tanh function: result[i] = (exp(a[i]) - exp(-a[i])) / (exp(a[i]) + exp(-a[i])).
/// Output range is (-1, 1).
/// </para>
/// <para><b>For Beginners:</b> Squashes values to between -1 and 1.
///
/// Example:
/// Tanh([-∞, -2, 0, 2, ∞]) ≈ [-1, -0.96, 0, 0.96, 1]
///
/// Similar to sigmoid but centered at zero, often works better than sigmoid.
/// </para>
/// </remarks>
public class TanhOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Softmax().
/// Computes softmax along specified axis. Converts logits to probabilities.
/// </para>
/// <para><b>For Beginners:</b> Converts scores to probabilities that sum to 1.
///
/// Example:
/// Softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
/// (notice they sum to 1.0)
///
/// Used for multi-class classification - outputs can be interpreted as
/// class probabilities.
/// </para>
/// </remarks>
public class SoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute softmax. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Softmax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents a generic activation function application in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.ApplyActivation().
/// Applies a named activation function to the input.
/// </para>
/// <para><b>For Beginners:</b> Applies any activation function by name.
///
/// This is a more generic operation that can apply various activations
/// (ReLU, Sigmoid, Tanh, etc.) based on a parameter rather than being
/// hard-coded to one specific activation.
/// </para>
/// </remarks>
public class ApplyActivationOp : IROp
{
    /// <summary>
    /// The name of the activation function to apply.
    /// </summary>
    public string ActivationName { get; set; } = string.Empty;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (string.IsNullOrWhiteSpace(ActivationName)) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ApplyActivation(t{InputIds[0]}, \"{ActivationName}\") : {OutputType} {OutputShape.ShapeToString()}";
    }
}

// ============================================================================
// ADDITIONAL ACTIVATION OPERATIONS
// ============================================================================

/// <summary>
/// Represents ELU (Exponential Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise.
/// Smoother than ReLU for negative values.
/// </para>
/// </remarks>
public class ELUOp : IROp
{
    /// <summary>
    /// The alpha parameter for negative values. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ELU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Leaky ReLU activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LeakyReLU(x) = max(alpha * x, x) where alpha is typically 0.01.
/// Allows small gradients for negative inputs.
/// </para>
/// </remarks>
public class LeakyReLUOp : IROp
{
    /// <summary>
    /// The negative slope. Default is 0.01.
    /// </summary>
    public double Alpha { get; set; } = 0.01;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LeakyReLU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents GELU (Gaussian Error Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes GELU(x) = x * Φ(x) where Φ is the standard normal CDF.
/// Used in modern transformers (BERT, GPT).
/// </para>
/// </remarks>
public class GELUOp : IROp
{
    /// <summary>
    /// Whether to use the approximate formula.
    /// </summary>
    public bool Approximate { get; set; } = false;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GELU(t{InputIds[0]}, approx={Approximate}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Swish/SiLU activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Swish(x) = x * sigmoid(x).
/// Self-gated activation with smooth gradient.
/// </para>
/// </remarks>
public class SwishOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Mish activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Mish(x) = x * tanh(softplus(x)).
/// Smooth, non-monotonic activation that often outperforms ReLU.
/// </para>
/// </remarks>
public class MishOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents SoftPlus activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SoftPlus(x) = ln(1 + exp(x)).
/// Smooth approximation of ReLU.
/// </para>
/// </remarks>
public class SoftPlusOp : IROp
{
    /// <summary>
    /// Scaling factor. Default is 1.0.
    /// </summary>
    public double Beta { get; set; } = 1.0;

    /// <summary>
    /// Threshold for switching to linear. Default is 20.0.
    /// </summary>
    public double Threshold { get; set; } = 20.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents SELU (Scaled Exponential Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1))).
/// Self-normalizing activation with fixed scale and alpha values.
/// </para>
/// </remarks>
public class SELUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Hard Sigmoid activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes HardSigmoid(x) = clip((x + 3) / 6, 0, 1).
/// Faster piecewise linear approximation of sigmoid.
/// </para>
/// </remarks>
public class HardSigmoidOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Hard Tanh activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes HardTanh(x) = clip(x, -1, 1).
/// Faster piecewise linear approximation of tanh.
/// </para>
/// </remarks>
public class HardTanhOp : IROp
{
    /// <summary>
    /// Minimum value. Default is -1.0.
    /// </summary>
    public double MinVal { get; set; } = -1.0;

    /// <summary>
    /// Maximum value. Default is 1.0.
    /// </summary>
    public double MaxVal { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents SoftSign activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SoftSign(x) = x / (1 + |x|).
/// Alternative to tanh with polynomial tails.
/// </para>
/// </remarks>
public class SoftSignOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents CELU (Continuously Differentiable ELU) activation in the IR.
/// </summary>
public class CELUOp : IROp
{
    /// <summary>
    /// The alpha parameter. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents LogSoftmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LogSoftmax(x) = log(softmax(x)).
/// Numerically stable for cross-entropy loss.
/// </para>
/// </remarks>
public class LogSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute log softmax. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LogSoftmax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents PReLU (Parametric ReLU) activation in the IR.
/// </summary>
public class PReLUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input + alpha parameter
        if (InputIds.Length != 2) return false;
        return true;
    }
}

/// <summary>
/// Represents Thresholded ReLU activation in the IR.
/// </summary>
public class ThresholdedReLUOp : IROp
{
    /// <summary>
    /// The threshold value. Default is 1.0.
    /// </summary>
    public double Threshold { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

// ============================================================================
// ADDITIONAL ACTIVATION OPERATIONS - Extended Set
// ============================================================================

/// <summary>
/// Represents LiSHT (Linearly Scaled Hyperbolic Tangent) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LiSHT(x) = x * tanh(x).
/// Similar to Swish but with tanh instead of sigmoid.
/// </para>
/// </remarks>
public class LiSHTOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Bent Identity activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes BentIdentity(x) = (sqrt(x^2 + 1) - 1) / 2 + x.
/// Smooth approximation to ReLU with non-zero gradients everywhere.
/// </para>
/// </remarks>
public class BentIdentityOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Gaussian activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Gaussian(x) = exp(-x^2).
/// Bell-shaped activation centered at zero.
/// </para>
/// </remarks>
public class GaussianOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Scaled Tanh activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ScaledTanh(x) = tanh(beta * x).
/// Tanh with adjustable steepness.
/// </para>
/// </remarks>
public class ScaledTanhOp : IROp
{
    /// <summary>
    /// Scaling factor for input. Default is 1.0.
    /// </summary>
    public double Beta { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ScaledTanh(t{InputIds[0]}, beta={Beta}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Squash activation in the IR (for Capsule Networks).
/// </summary>
/// <remarks>
/// <para>
/// Computes Squash(x) = (||x||^2 / (1 + ||x||^2)) * (x / ||x||).
/// Used in capsule networks to ensure output vectors have length between 0 and 1.
/// </para>
/// </remarks>
public class SquashOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents ISRU (Inverse Square Root Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ISRU(x) = x / sqrt(1 + alpha * x^2).
/// Self-regularizing activation with bounded output.
/// </para>
/// </remarks>
public class ISRUOp : IROp
{
    /// <summary>
    /// The alpha parameter. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ISRU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Sign activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Sign(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0.
/// Hard threshold activation, commonly used in binary networks.
/// </para>
/// </remarks>
public class SignOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Softmin activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Softmin(x) = softmax(-x).
/// Similar to softmax but emphasizes smaller values.
/// </para>
/// </remarks>
public class SoftminOp : IROp
{
    /// <summary>
    /// The axis along which to compute softmin. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Softmin(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Log Softmin activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LogSoftmin(x) = log(softmin(x)).
/// Numerically stable version of softmin in log space.
/// </para>
/// </remarks>
public class LogSoftminOp : IROp
{
    /// <summary>
    /// The axis along which to compute log softmin. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LogSoftmin(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents SQRBF (Squared Radial Basis Function) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SQRBF(x) = 1 - x^2 if |x| <= 1, else 0.
/// Compactly supported activation function.
/// </para>
/// </remarks>
public class SQRBFOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents Maxout activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Maxout(x) = max(x_1, x_2, ..., x_k) over k pieces.
/// Piecewise linear activation that learns its shape.
/// </para>
/// </remarks>
public class MaxoutOp : IROp
{
    /// <summary>
    /// Number of linear pieces. Default is 2.
    /// </summary>
    public int NumPieces { get; set; } = 2;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (NumPieces < 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Maxout(t{InputIds[0]}, pieces={NumPieces}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents RReLU (Randomized Leaky ReLU) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes RReLU(x) = x if x >= 0, else alpha * x where alpha is randomly
/// sampled from uniform(lower, upper) during training.
/// </para>
/// </remarks>
public class RReLUOp : IROp
{
    /// <summary>
    /// Lower bound for random negative slope. Default is 0.125.
    /// </summary>
    public double Lower { get; set; } = 0.125;

    /// <summary>
    /// Upper bound for random negative slope. Default is 0.333.
    /// </summary>
    public double Upper { get; set; } = 0.333;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Lower > Upper) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = RReLU(t{InputIds[0]}, lower={Lower}, upper={Upper}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Spherical Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Projects inputs onto a unit sphere before applying softmax.
/// Useful for directional data or when angular relationships matter.
/// </para>
/// </remarks>
public class SphericalSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute spherical softmax. Default is -1.
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = SphericalSoftmax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Taylor Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Uses Taylor series expansion of exp() for approximate softmax.
/// Can be faster than standard softmax for lower orders.
/// </para>
/// </remarks>
public class TaylorSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute Taylor softmax. Default is -1.
    /// </summary>
    public int Axis { get; set; } = -1;

    /// <summary>
    /// Taylor series expansion order. Default is 2.
    /// </summary>
    public int Order { get; set; } = 2;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Order < 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = TaylorSoftmax(t{InputIds[0]}, axis={Axis}, order={Order}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Sparsemax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Like softmax but produces sparse outputs (some outputs exactly zero).
/// Useful when you want a hard-ish attention mechanism.
/// </para>
/// </remarks>
public class SparsemaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute sparsemax. Default is -1.
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Sparsemax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Hierarchical Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Organizes classes into a tree structure for efficient computation.
/// Reduces complexity from O(V) to O(log V) for large vocabularies.
/// </para>
/// </remarks>
public class HierarchicalSoftmaxOp : IROp
{
    /// <summary>
    /// The hierarchy tree structure (encoded).
    /// </summary>
    public int[] TreeStructure { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
