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

/// <summary>
/// Backward operation for ReshapeOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = reshape(x, new_shape)
/// Backward: grad_x = reshape(grad_y, original_shape)
/// Reshape doesn't change data, just view, so gradient just reshapes back.
/// </para>
/// </remarks>
public class GradReshapeOp : BackwardOp
{
    /// <summary>Original shape before reshape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (OriginalShape.Length == 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradReshape[shape={string.Join(",", OriginalShape)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for TransposeOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = transpose(x) or permute(x, axes)
/// Backward: grad_x = transpose(grad_y, inverse_axes)
/// </para>
/// </remarks>
public class GradTransposeOp : BackwardOp
{
    /// <summary>Axes used in forward transpose.</summary>
    public int[]? Axes { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? string.Join(",", Axes) : "default";
        return $"t{OutputId} = GradTranspose[axes={axesStr}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for ConcatOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = concat([x1, x2, ...], axis)
/// Backward: grad_xi = slice(grad_y, start_i, end_i, axis)
/// Each input gets a slice of the output gradient.
/// </para>
/// </remarks>
public class GradConcatOp : BackwardOp
{
    /// <summary>Which input are we computing gradient for.</summary>
    public int InputIndex { get; set; }

    /// <summary>Concatenation axis.</summary>
    public int Axis { get; set; }

    /// <summary>Start index along axis for this input's gradient.</summary>
    public int StartIndex { get; set; }

    /// <summary>Size along axis for this input.</summary>
    public int Size { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradConcat[input={InputIndex}, axis={Axis}, start={StartIndex}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SplitOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: [y1, y2, ...] = split(x, sizes, axis)
/// Backward: grad_x = concat([grad_y1, grad_y2, ...], axis)
/// </para>
/// </remarks>
public class GradSplitOp : BackwardOp
{
    /// <summary>Split axis.</summary>
    public int Axis { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSplit[axis={Axis}]({string.Join(", ", InputIds.Select(id => $"t{id}"))}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for DivideOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a / b
/// Backward: grad_a = grad_c / b, grad_b = -grad_c * a / (b^2)
/// </para>
/// </remarks>
public class GradDivideOp : BackwardOp
{
    /// <summary>Which input: 0 = numerator, 1 = denominator.</summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Needs grad_output and original inputs
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradDivide[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for PowerOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x^p
/// Backward: grad_x = grad_y * p * x^(p-1)
/// </para>
/// </remarks>
public class GradPowerOp : BackwardOp
{
    /// <summary>Exponent used in forward pass.</summary>
    public double Exponent { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradPower[exp={Exponent}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SqrtOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = sqrt(x)
/// Backward: grad_x = grad_y / (2 * sqrt(x)) = grad_y / (2 * y)
/// </para>
/// </remarks>
public class GradSqrtOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSqrt(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SumOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = sum(x, axes)
/// Backward: grad_x = broadcast(grad_y, original_shape)
/// Gradient is broadcasted back to original shape.
/// </para>
/// </remarks>
public class GradSumOp : BackwardOp
{
    /// <summary>Original input shape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Axes that were reduced.</summary>
    public int[]? Axes { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? string.Join(",", Axes) : "all";
        return $"t{OutputId} = GradSum[axes={axesStr}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for MeanOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = mean(x, axes)
/// Backward: grad_x = broadcast(grad_y / count, original_shape)
/// Similar to sum but divided by number of elements.
/// </para>
/// </remarks>
public class GradMeanOp : BackwardOp
{
    /// <summary>Original input shape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Axes that were reduced.</summary>
    public int[]? Axes { get; set; }

    /// <summary>Number of elements that were averaged.</summary>
    public int Count { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMean[count={Count}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for SliceOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = slice(x, start, end)
/// Backward: grad_x = pad_with_zeros(grad_y, original_shape, start_indices)
/// Gradient is zero everywhere except the sliced region.
/// </para>
/// </remarks>
public class GradSliceOp : BackwardOp
{
    /// <summary>Original input shape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Start indices for the slice.</summary>
    public int[] StartIndices { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSlice[start={string.Join(",", StartIndices)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for PadOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = pad(x, padding)
/// Backward: grad_x = slice(grad_y, unpad)
/// Gradient comes from the center (unpadded) region.
/// </para>
/// </remarks>
public class GradPadOp : BackwardOp
{
    /// <summary>Padding that was applied.</summary>
    public int[] Padding { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradPad[padding={string.Join(",", Padding)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for DropoutOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = dropout(x, p, mask)
/// Backward: grad_x = grad_y * mask / (1 - p) (using same mask from forward)
/// </para>
/// </remarks>
public class GradDropoutOp : BackwardOp
{
    /// <summary>Dropout probability.</summary>
    public double Probability { get; set; } = 0.5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and dropout mask
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradDropout[p={Probability}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for LayerNormOp.
/// </summary>
/// <remarks>
/// <para>
/// Layer normalization gradient is complex, involving variance and mean.
/// Computes gradients for input, gamma, and beta.
/// </para>
/// </remarks>
public class GradLayerNormOp : BackwardOp
{
    /// <summary>Which input: 0 = input, 1 = gamma, 2 = beta.</summary>
    public int InputIndex { get; set; }

    /// <summary>Epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Normalized shape.</summary>
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLayerNorm[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for EmbeddingOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = embedding[indices]
/// Backward: grad_embedding = scatter_add(grad_y, indices, embedding_shape)
/// Gradients are scattered back to embedding table positions.
/// </para>
/// </remarks>
public class GradEmbeddingOp : BackwardOp
{
    /// <summary>Shape of the embedding table.</summary>
    public int[] EmbeddingShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and indices
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradEmbedding[shape={string.Join(",", EmbeddingShape)}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for GatherOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = gather(x, indices, axis)
/// Backward: grad_x = scatter(grad_y, indices, axis, shape)
/// </para>
/// </remarks>
public class GradGatherOp : BackwardOp
{
    /// <summary>Gather axis.</summary>
    public int Axis { get; set; }

    /// <summary>Original input shape.</summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and indices
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradGather[axis={Axis}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for LeakyReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = max(alpha * x, x)
/// Backward: grad_x = grad_y * (1 if x > 0 else alpha)
/// </para>
/// </remarks>
public class GradLeakyReLUOp : BackwardOp
{
    /// <summary>Negative slope.</summary>
    public double Alpha { get; set; } = 0.01;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLeakyReLU[alpha={Alpha}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for GELUOp.
/// </summary>
/// <remarks>
/// <para>
/// GELU gradient is computed using the derivative of the GELU function.
/// grad_x = grad_y * (0.5 * (1 + tanh(...)) + 0.5 * x * sech^2(...) * derivative_of_inner)
/// </para>
/// </remarks>
public class GradGELUOp : BackwardOp
{
    /// <summary>Whether approximate GELU was used.</summary>
    public bool Approximate { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradGELU[approx={Approximate}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Backward operation for BroadcastOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = broadcast(x, target_shape)
/// Backward: grad_x = reduce_sum(grad_y, broadcasted_axes)
/// Sum over axes that were broadcasted.
/// </para>
/// </remarks>
public class GradBroadcastOp : BackwardOp
{
    /// <summary>Original shape before broadcast.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Axes that were broadcasted.</summary>
    public int[] BroadcastedAxes { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradBroadcast[axes={string.Join(",", BroadcastedAxes)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
