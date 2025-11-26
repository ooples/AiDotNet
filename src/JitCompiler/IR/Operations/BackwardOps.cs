namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Base class for backward (gradient) operations in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Backward operations compute gradients during backpropagation for training.
/// Each forward operation has corresponding backward operation(s) that compute
/// the gradient with respect to its inputs.
/// </para>
/// <para><b>For Beginners:</b> These operations compute gradients for training.
///
/// In neural network training:
/// - Forward pass: Compute outputs from inputs
/// - Backward pass: Compute how to adjust weights to reduce error
///
/// Backward operations implement the chain rule of calculus to flow
/// gradients backward through the network.
/// </para>
/// </remarks>
public abstract class BackwardOp : IROp
{
    /// <summary>
    /// The tensor ID from the forward pass that may be needed for gradient computation.
    /// Many backward operations need the forward pass output or inputs.
    /// </summary>
    public int? SavedForwardTensorId { get; set; }
}

/// <summary>
/// Gradient accumulation operation - sums gradients from multiple paths.
/// </summary>
/// <remarks>
/// <para>
/// When a tensor is used by multiple operations, gradients flow back from
/// multiple paths. These must be summed to get the total gradient.
/// </para>
/// <para><b>For Beginners:</b> Combines gradients from different paths.
///
/// Example: If x is used in both y = x + 2 and z = x * 3
/// The gradient of x needs contributions from both operations:
/// grad_x = grad_from_y + grad_from_z
/// </para>
/// </remarks>
public class GradAccumulateOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Can have 2+ inputs to accumulate
        if (InputIds.Length < 2) return false;
        return true;
    }

    public override string ToString()
    {
        var inputs = string.Join(" + ", InputIds.Select(id => $"t{id}"));
        return $"t{OutputId} = AccumulateGrad({inputs}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for AddOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a + b
/// Backward: grad_a = grad_c, grad_b = grad_c
/// (gradient flows equally to both inputs)
/// </para>
/// </remarks>
public class GradAddOp : BackwardOp
{
    /// <summary>
    /// Which input are we computing the gradient for? (0 = left, 1 = right)
    /// </summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false; // Takes output gradient
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradAdd[input={InputIndex}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SubtractOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a - b
/// Backward: grad_a = grad_c, grad_b = -grad_c
/// </para>
/// </remarks>
public class GradSubtractOp : BackwardOp
{
    /// <summary>
    /// Which input are we computing the gradient for? (0 = left, 1 = right)
    /// </summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSubtract[input={InputIndex}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for ElementwiseMultiplyOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a * b (element-wise)
/// Backward: grad_a = grad_c * b, grad_b = grad_c * a
/// </para>
/// </remarks>
public class GradElementwiseMultiplyOp : BackwardOp
{
    /// <summary>
    /// Which input are we computing the gradient for? (0 = left, 1 = right)
    /// </summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and the other input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradElemMul[input={InputIndex}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for MatMulOp (left input).
/// </summary>
/// <remarks>
/// <para>
/// Forward: C = A @ B (matrix multiplication)
/// Backward for A: grad_A = grad_C @ B^T
/// </para>
/// </remarks>
public class GradMatMulLeftOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and right input (B)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMatMulLeft(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for MatMulOp (right input).
/// </summary>
/// <remarks>
/// <para>
/// Forward: C = A @ B (matrix multiplication)
/// Backward for B: grad_B = A^T @ grad_C
/// </para>
/// </remarks>
public class GradMatMulRightOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // left input (A) and grad_output
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMatMulRight(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for ReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = max(0, x)
/// Backward: grad_x = grad_y * (x > 0)
/// </para>
/// </remarks>
public class GradReLUOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input (x)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradReLU(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SigmoidOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = 1 / (1 + exp(-x))
/// Backward: grad_x = grad_y * y * (1 - y)
/// </para>
/// </remarks>
public class GradSigmoidOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSigmoid(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for TanhOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = tanh(x)
/// Backward: grad_x = grad_y * (1 - y^2)
/// </para>
/// </remarks>
public class GradTanhOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradTanh(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for ExpOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = exp(x)
/// Backward: grad_x = grad_y * y
/// (derivative of exp is exp itself)
/// </para>
/// </remarks>
public class GradExpOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradExp(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for LogOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = log(x)
/// Backward: grad_x = grad_y / x
/// </para>
/// </remarks>
public class GradLogOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input (x)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLog(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SoftmaxOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y_i = exp(x_i) / sum(exp(x_j))
/// Backward: grad_x = y * (grad_y - sum(grad_y * y))
/// (Jacobian computation for softmax)
/// </para>
/// </remarks>
public class GradSoftmaxOp : BackwardOp
{
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSoftmax[axis={Axis}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for Conv2DOp.
/// </summary>
/// <remarks>
/// <para>
/// Computes gradient for convolution inputs (data, filters, or bias).
/// Uses convolution theorems for efficient gradient computation.
/// </para>
/// </remarks>
public class GradConv2DOp : BackwardOp
{
    public int InputIndex { get; set; } // 0 = data, 1 = filters, 2 = bias
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs depend on which gradient we're computing
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradConv2D[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for MaxPool2DOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: Records indices of max elements
/// Backward: Routes gradient only to max elements
/// </para>
/// </remarks>
public class GradMaxPool2DOp : BackwardOp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward indices/input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMaxPool2D(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for AvgPool2DOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: Average values in each window
/// Backward: Distributes gradient equally to all elements in window
/// </para>
/// </remarks>
public class GradAvgPool2DOp : BackwardOp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false; // Only needs grad_output
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradAvgPool2D(t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for BatchNormOp.
/// </summary>
/// <remarks>
/// <para>
/// Batch normalization has complex gradients involving batch statistics.
/// Computes gradients for input, scale, and bias parameters.
/// </para>
/// </remarks>
public class GradBatchNormOp : BackwardOp
{
    public int InputIndex { get; set; } // 0 = input, 1 = scale, 2 = bias
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradBatchNorm[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
