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

/// <summary>
/// Represents Softmin activation in the IR (min-based variant of softmax).
/// </summary>
/// <remarks>
/// <para>
/// Computes softmin along specified axis: softmin(x) = softmax(-x).
/// Converts negative logits to probabilities that sum to 1.
/// </para>
/// <para><b>For Beginners:</b> Like softmax, but emphasizes smaller values.
///
/// Example:
/// Softmin([1, 2, 3]) approximately equals [0.67, 0.24, 0.09]
/// (notice the smallest value gets the highest probability)
///
/// Less common than softmax, but useful when minimizing is desired.
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
/// Represents LogSoftmax activation in the IR (numerically stable).
/// </summary>
/// <remarks>
/// <para>
/// Computes log(softmax(x)) using log-sum-exp trick for numerical stability.
/// Equivalent to log(softmax(x)) but avoids overflow/underflow.
/// </para>
/// <para><b>For Beginners:</b> Logarithm of softmax probabilities.
///
/// Example:
/// LogSoftmax([1, 2, 3]) approximately equals [-2.41, -1.41, -0.41]
///
/// More numerically stable than computing log(softmax(x)) separately.
/// Often used with negative log-likelihood loss in classification.
/// </para>
/// </remarks>
public class LogSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute log-softmax. Default is -1 (last axis).
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
/// Represents LogSoftmin activation in the IR (numerically stable).
/// </summary>
/// <remarks>
/// <para>
/// Computes log(softmin(x)) using log-sum-exp trick for numerical stability.
/// </para>
/// <para><b>For Beginners:</b> Logarithm of softmin probabilities.
///
/// Example:
/// LogSoftmin([1, 2, 3]) approximately equals [-0.41, -1.41, -2.41]
///
/// Numerically stable version of log(softmin(x)).
/// </para>
/// </remarks>
public class LogSoftminOp : IROp
{
    /// <summary>
    /// The axis along which to compute log-softmin. Default is -1 (last axis).
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
/// Represents Sparsemax activation in the IR (sparse alternative to softmax).
/// </summary>
/// <remarks>
/// <para>
/// Computes sparsemax projection: produces sparse probability distributions.
/// Unlike softmax, can produce exact zeros for low-probability classes.
/// </para>
/// <para><b>For Beginners:</b> Like softmax, but can produce exact zeros.
///
/// Example:
/// Sparsemax([1, 2, 7]) approximately equals [0, 0, 1]
/// (notice exact zeros for unlikely classes)
///
/// Useful when you want sparse predictions (most classes with zero probability).
/// </para>
/// <para><b>TODO:</b> Implement efficient sparsemax algorithm.
/// Current implementation is placeholder - requires O(n log n) projection algorithm.
/// </para>
/// </remarks>
public class SparsemaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute sparsemax. Default is -1 (last axis).
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
/// Represents Spherical Softmax activation in the IR (softmax on unit sphere).
/// </summary>
/// <remarks>
/// <para>
/// Computes softmax after normalizing input vectors to unit sphere.
/// Useful for angular-based representations.
/// </para>
/// <para><b>For Beginners:</b> Softmax applied to normalized vectors.
///
/// First normalizes each vector to unit length, then applies softmax.
/// Useful when direction matters more than magnitude.
/// </para>
/// </remarks>
public class SphericalSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute spherical softmax. Default is -1 (last axis).
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
/// Represents Gumbel-Softmax activation in the IR (stochastic, differentiable).
/// </summary>
/// <remarks>
/// <para>
/// Computes Gumbel-Softmax: softmax((x + Gumbel noise) / temperature).
/// Provides differentiable sampling from categorical distributions.
/// </para>
/// <para><b>For Beginners:</b> Softmax with controllable randomness.
///
/// Adds Gumbel noise before softmax to enable stochastic discrete choices
/// while maintaining differentiability. Temperature controls randomness.
///
/// Used in variational autoencoders and discrete latent variable models.
/// </para>
/// </remarks>
public class GumbelSoftmaxOp : IROp
{
    /// <summary>
    /// Temperature parameter controlling randomness. Lower = more deterministic.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// The axis along which to compute Gumbel-Softmax. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Temperature <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GumbelSoftmax(t{InputIds[0]}, temp={Temperature}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Taylor-Softmax activation in the IR (Taylor series approximation).
/// </summary>
/// <remarks>
/// <para>
/// Approximates softmax using Taylor series expansion.
/// Faster but less accurate than standard softmax.
/// </para>
/// <para><b>For Beginners:</b> Fast approximation of softmax.
///
/// Uses polynomial approximation instead of expensive exponentials.
/// Trades accuracy for speed - good for low-precision applications.
/// </para>
/// <para><b>TODO:</b> Implement Taylor series approximation.
/// Current implementation is placeholder - requires order parameter for series.
/// </para>
/// </remarks>
public class TaylorSoftmaxOp : IROp
{
    /// <summary>
    /// Order of Taylor series approximation. Higher = more accurate, slower.
    /// </summary>
    public int Order { get; set; } = 2;

    /// <summary>
    /// The axis along which to compute Taylor-Softmax. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Order < 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = TaylorSoftmax(t{InputIds[0]}, order={Order}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Hierarchical Softmax activation in the IR (tree-structured).
/// </summary>
/// <remarks>
/// <para>
/// Computes hierarchical softmax using binary tree structure.
/// Reduces computational complexity from O(n) to O(log n).
/// </para>
/// <para><b>For Beginners:</b> Efficient softmax for many classes.
///
/// Instead of computing probabilities for all classes at once,
/// makes binary decisions in a tree structure.
///
/// Much faster when number of classes is very large (e.g., vocabulary in NLP).
/// </para>
/// <para><b>TODO:</b> Implement hierarchical tree structure.
/// Current implementation is placeholder - requires tree specification.
/// </para>
/// </remarks>
public class HierarchicalSoftmaxOp : IROp
{
    /// <summary>
    /// Tree structure specification (placeholder - needs design).
    /// </summary>
    public string TreeStructure { get; set; } = string.Empty;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = HierarchicalSoftmax(t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Maxout activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes max(W1*x + b1, W2*x + b2, ...) across multiple linear projections.
/// Learns the activation function itself through multiple weight sets.
/// </para>
/// <para><b>For Beginners:</b> Takes maximum across multiple linear transformations.
///
/// Instead of applying a fixed function like ReLU, computes several
/// linear functions and takes the max. The network learns which function
/// shape works best.
///
/// More powerful but requires more parameters than standard activations.
/// </para>
/// </remarks>
public class MaxoutOp : IROp
{
    /// <summary>
    /// Number of linear projections to max over.
    /// </summary>
    public int NumProjections { get; set; } = 2;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 1) return false;
        if (NumProjections < 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Maxout(t{InputIds[0]}, projections={NumProjections}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Sign activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes sign function: -1 for negative, 0 for zero, +1 for positive.
/// </para>
/// <para><b>For Beginners:</b> Outputs only -1, 0, or +1.
///
/// Example:
/// Sign([-5.3, -0.1, 0, 0.1, 5.3]) = [-1, -1, 0, 1, 1]
///
/// Used in binary neural networks and sign-based optimization.
/// Not differentiable at zero, so requires special gradient handling.
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
/// Represents Gaussian activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Gaussian function: exp(-x^2).
/// Bell-shaped curve centered at zero.
/// </para>
/// <para><b>For Beginners:</b> Bell curve activation.
///
/// Example:
/// Gaussian([-2, -1, 0, 1, 2]) approximately equals [0.02, 0.37, 1.0, 0.37, 0.02]
///
/// Maximum at zero, decreases towards zero as x moves away from origin.
/// Used in radial basis function networks.
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
/// Represents ISRU (Inverse Square Root Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ISRU: x / sqrt(1 + alpha * x^2).
/// Self-normalizing activation similar to ELU but faster.
/// </para>
/// <para><b>For Beginners:</b> Smooth, bounded activation function.
///
/// Example (alpha=1):
/// ISRU([-2, -1, 0, 1, 2]) approximately equals [-0.89, -0.71, 0, 0.71, 0.89]
///
/// Output range is approximately (-1/sqrt(alpha), 1/sqrt(alpha)).
/// Faster than ELU because it avoids exponentials.
/// </para>
/// </remarks>
public class ISRUOp : IROp
{
    /// <summary>
    /// Alpha parameter controlling the curve shape. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Alpha <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ISRU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents LiSHT (Linearly Scaled Hyperbolic Tangent) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LiSHT: x * tanh(x).
/// Combines linear and tanh properties.
/// </para>
/// <para><b>For Beginners:</b> Smooth, non-monotonic activation.
///
/// Example:
/// LiSHT([-2, -1, 0, 1, 2]) approximately equals [-1.93, -0.76, 0, 0.76, 1.93]
///
/// Similar to Swish but uses tanh instead of sigmoid.
/// Has a small negative region and grows almost linearly for large x.
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
/// Represents SQRBF (Squared Radial Basis Function) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes squared RBF: exp(-beta * x^2).
/// Gaussian-like activation with adjustable width.
/// </para>
/// <para><b>For Beginners:</b> Adjustable bell curve.
///
/// Example (beta=1):
/// SQRBF([-2, -1, 0, 1, 2]) approximately equals [0.02, 0.37, 1.0, 0.37, 0.02]
///
/// Beta controls the width of the bell curve.
/// Used in radial basis function networks for local learning.
/// </para>
/// </remarks>
public class SQRBFOp : IROp
{
    /// <summary>
    /// Beta parameter controlling the RBF width. Default is 1.0.
    /// </summary>
    public double Beta { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Beta <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = SQRBF(t{InputIds[0]}, beta={Beta}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Squash activation in the IR (capsule network squashing).
/// </summary>
/// <remarks>
/// <para>
/// Computes squashing function: (||x||^2 / (1 + ||x||^2)) * (x / ||x||).
/// Squashes vector length to [0, 1) while preserving direction.
/// </para>
/// <para><b>For Beginners:</b> Normalizes vector length to less than 1.
///
/// Used in capsule networks to represent presence of features.
/// - Long vectors stay long (approach length 1)
/// - Short vectors get shorter (approach length 0)
/// - Direction is always preserved
///
/// Unlike softmax, works on vector magnitudes, not individual elements.
/// </para>
/// </remarks>
public class SquashOp : IROp
{
    /// <summary>
    /// The axis along which to compute vector norms. Default is -1 (last axis).
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
        return $"t{OutputId} = Squash(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents Binary Spiking Activation in the IR (for spiking neural networks).
/// </summary>
/// <remarks>
/// <para>
/// Computes binary step function with threshold: output = (x >= threshold) ? 1 : 0.
/// Used in spiking neural networks to model neuron firing.
/// </para>
/// <para><b>For Beginners:</b> Outputs 1 if above threshold, 0 otherwise.
///
/// Example (threshold=0.5):
/// BinarySpike([0.1, 0.5, 0.9, 1.5]) = [0, 1, 1, 1]
///
/// Models biological neurons that fire when membrane potential exceeds threshold.
/// Not differentiable, requires surrogate gradients for training.
/// </para>
/// </remarks>
public class BinarySpikingActivationOp : IROp
{
    /// <summary>
    /// Firing threshold. Default is 0.5.
    /// </summary>
    public double Threshold { get; set; } = 0.5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = BinarySpike(t{InputIds[0]}, threshold={Threshold}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
