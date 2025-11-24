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
