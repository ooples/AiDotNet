namespace AiDotNet.JitCompiler.IR.Operations;

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

/// <summary>
/// Represents sum reduction in the IR.
/// </summary>
public class SumOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? $"[{string.Join(",", Axes)}]" : "all";
        return $"t{OutputId} = Sum(t{InputIds[0]}, axes={axesStr}, keepDims={KeepDims}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents mean reduction in the IR.
/// </summary>
public class MeanOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents max reduction in the IR.
/// </summary>
public class ReduceMaxOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents mean reduction in the IR.
/// </summary>
public class ReduceMeanOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents log variance reduction in the IR.
/// </summary>
public class ReduceLogVarianceOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

// ============================================================================
// SHAPE OPERATIONS
// ============================================================================

/// <summary>
/// Represents reshape operation in the IR.
/// </summary>
public class ReshapeOp : IROp
{
    public int[] NewShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (NewShape.Length == 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Reshape(t{InputIds[0]}, {NewShape.ShapeToString()}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents concatenation along an axis in the IR.
/// </summary>
public class ConcatOp : IROp
{
    public int Axis { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;  // Need at least 2 inputs to concat
        return true;
    }

    public override string ToString()
    {
        var inputs = string.Join(", ", InputIds.Select(id => $"t{id}"));
        return $"t{OutputId} = Concat([{inputs}], axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents padding operation in the IR.
/// </summary>
public class PadOp : IROp
{
    /// <summary>Padding width per dimension as 2D array [dim, (before, after)].</summary>
    public int[,]? PadWidth { get; set; }

    /// <summary>Simplified padding as 1D array [pad_before_0, pad_after_0, pad_before_1, pad_after_1, ...].</summary>
    public int[] Padding { get; set; } = Array.Empty<int>();

    /// <summary>Input shape for kernel generation.</summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents cropping operation in the IR.
/// </summary>
public class CropOp : IROp
{
    /// <summary>Cropping amounts per dimension.</summary>
    public int[] Cropping { get; set; } = Array.Empty<int>();

    /// <summary>Offset positions for cropping [start indices per dimension].</summary>
    public int[] Offsets { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents upsampling operation in the IR.
/// </summary>
public class UpsampleOp : IROp
{
    /// <summary>Upsampling scale factor.</summary>
    public int Scale { get; set; } = 2;

    /// <summary>Upsampling mode: "nearest" or "bilinear".</summary>
    public string Mode { get; set; } = "nearest";

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Scale <= 0) return false;
        return true;
    }
}

/// <summary>
/// Represents pixel shuffle (depth-to-space) operation in the IR.
/// </summary>
public class PixelShuffleOp : IROp
{
    public int UpscaleFactor { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (UpscaleFactor <= 0) return false;
        return true;
    }
}

// ============================================================================
// CONVOLUTION OPERATIONS
// ============================================================================

/// <summary>
/// Represents 2D convolution in the IR.
/// </summary>
public class Conv2DOp : IROp
{
    /// <summary>Kernel size [height, width].</summary>
    public int[] KernelSize { get; set; } = new int[] { 3, 3 };

    /// <summary>Stride [height, width].</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Padding [height, width].</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>Whether this convolution has a bias term.</summary>
    public bool HasBias { get; set; }

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input + kernel, optionally + bias
        if (InputIds.Length < 2 || InputIds.Length > 3) return false;
        if (InputIds.Length == 3 && !HasBias) return false;
        return true;
    }

    public override string ToString()
    {
        var inputs = HasBias ? $"t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}" : $"t{InputIds[0]}, t{InputIds[1]}";
        return $"t{OutputId} = Conv2D({inputs}, kernel=[{string.Join(",", KernelSize)}], stride=[{string.Join(",", Stride)}], pad=[{string.Join(",", Padding)}]) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents transposed 2D convolution in the IR.
/// </summary>
public class ConvTranspose2DOp : IROp
{
    /// <summary>Kernel size [height, width].</summary>
    public int[] KernelSize { get; set; } = new int[] { 3, 3 };

    /// <summary>Stride [height, width].</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Padding [height, width].</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>Output padding [height, width].</summary>
    public int[] OutputPadding { get; set; } = new int[] { 0, 0 };

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

/// <summary>
/// Represents depthwise 2D convolution in the IR.
/// </summary>
public class DepthwiseConv2DOp : IROp
{
    /// <summary>Kernel size [height, width].</summary>
    public int[] KernelSize { get; set; } = new int[] { 3, 3 };

    /// <summary>Stride [height, width].</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Padding [height, width].</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

/// <summary>
/// Represents dilated 2D convolution in the IR.
/// </summary>
public class DilatedConv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };
    public int[] Dilation { get; set; } = new int[] { 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

/// <summary>
/// Represents locally connected 2D convolution in the IR.
/// </summary>
public class LocallyConnectedConv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

// ============================================================================
// POOLING OPERATIONS
// ============================================================================

/// <summary>
/// Represents 2D max pooling in the IR.
/// </summary>
public class MaxPool2DOp : IROp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents 2D average pooling in the IR.
/// </summary>
public class AvgPool2DOp : IROp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

// ============================================================================
// NORMALIZATION OPERATIONS
// ============================================================================

/// <summary>
/// Represents layer normalization in the IR.
/// </summary>
public class LayerNormOp : IROp
{
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta
        if (InputIds.Length != 3) return false;
        return true;
    }
}

/// <summary>
/// Represents batch normalization in the IR.
/// </summary>
public class BatchNormOp : IROp
{
    public double Epsilon { get; set; } = 1e-5;
    public double Momentum { get; set; } = 0.1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta, running_mean, running_var
        if (InputIds.Length != 5) return false;
        return true;
    }
}

// ============================================================================
// ADVANCED OPERATIONS
// ============================================================================

/// <summary>
/// Represents graph convolution in the IR.
/// </summary>
public class GraphConvOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // features, adjacency_matrix, weights
        if (InputIds.Length != 3) return false;
        return true;
    }
}

/// <summary>
/// Represents affine grid generation for spatial transformer in the IR.
/// </summary>
public class AffineGridOp : IROp
{
    public int[] OutputSize { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;  // theta (affine transformation matrix)
        return true;
    }
}

/// <summary>
/// Represents grid sampling for spatial transformer in the IR.
/// </summary>
public class GridSampleOp : IROp
{
    public string InterpolationMode { get; set; } = "bilinear";
    public string PaddingMode { get; set; } = "zeros";

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;  // input, grid
        return true;
    }
}

/// <summary>
/// Represents RBF (Radial Basis Function) kernel computation in the IR.
/// </summary>
public class RBFKernelOp : IROp
{
    public double Gamma { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;  // x, centers
        return true;
    }
}

// ============================================================================
// RECURRENT NETWORK OPERATIONS
// ============================================================================

/// <summary>
/// Represents a GRU (Gated Recurrent Unit) cell operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// GRU cell computes:
/// - z = sigmoid(Wz @ x + Uz @ h + bz)  // Update gate
/// - r = sigmoid(Wr @ x + Ur @ h + br)  // Reset gate
/// - h_tilde = tanh(Wh @ x + Uh @ (r * h) + bh)  // Candidate hidden state
/// - h_new = (1 - z) * h + z * h_tilde  // New hidden state
/// </para>
/// </remarks>
public class GRUCellOp : IROp
{
    /// <summary>
    /// Size of the hidden state.
    /// </summary>
    public int HiddenSize { get; set; }

    /// <summary>
    /// Whether to include bias terms.
    /// </summary>
    public bool HasBias { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input (x), hidden state (h), weights (W_ih, W_hh), optionally biases (b_ih, b_hh)
        if (InputIds.Length < 4) return false;
        if (HiddenSize <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GRUCell(t{InputIds[0]}, t{InputIds[1]}, hidden={HiddenSize}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents an LSTM (Long Short-Term Memory) cell operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// LSTM cell computes:
/// - i = sigmoid(Wi @ x + Ui @ h + bi)  // Input gate
/// - f = sigmoid(Wf @ x + Uf @ h + bf)  // Forget gate
/// - g = tanh(Wg @ x + Ug @ h + bg)     // Cell candidate
/// - o = sigmoid(Wo @ x + Uo @ h + bo)  // Output gate
/// - c_new = f * c + i * g              // New cell state
/// - h_new = o * tanh(c_new)            // New hidden state
/// </para>
/// </remarks>
public class LSTMCellOp : IROp
{
    /// <summary>
    /// Size of the hidden state.
    /// </summary>
    public int HiddenSize { get; set; }

    /// <summary>
    /// Whether to include bias terms.
    /// </summary>
    public bool HasBias { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input (x), hidden state (h), cell state (c), weights (W_ih, W_hh), optionally biases (b_ih, b_hh)
        if (InputIds.Length < 5) return false;
        if (HiddenSize <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LSTMCell(t{InputIds[0]}, h=t{InputIds[1]}, c=t{InputIds[2]}, hidden={HiddenSize}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

// ============================================================================
// ADDITIONAL SHAPE OPERATIONS
// ============================================================================

/// <summary>
/// Represents split operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Splits a tensor into multiple parts along a specified axis.
/// </para>
/// </remarks>
public class SplitOp : IROp
{
    /// <summary>
    /// The axis along which to split.
    /// </summary>
    public int Axis { get; set; }

    /// <summary>
    /// The sizes of each split section.
    /// </summary>
    public int[] SplitSizes { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Number of equal splits (alternative to SplitSizes).
    /// </summary>
    public int NumSplits { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var sizesStr = SplitSizes.Length > 0 ? $"[{string.Join(",", SplitSizes)}]" : $"num={NumSplits}";
        return $"t{OutputId} = Split(t{InputIds[0]}, axis={Axis}, {sizesStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents slice operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Extracts a contiguous slice from a tensor along specified axes.
/// </para>
/// </remarks>
public class SliceOp : IROp
{
    /// <summary>
    /// Start indices for each axis.
    /// </summary>
    public int[] Starts { get; set; } = Array.Empty<int>();

    /// <summary>
    /// End indices for each axis (exclusive).
    /// </summary>
    public int[] Ends { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Step size for each axis.
    /// </summary>
    public int[] Steps { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Axes to slice on.
    /// </summary>
    public int[] Axes { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Slice(t{InputIds[0]}, starts=[{string.Join(",", Starts)}], ends=[{string.Join(",", Ends)}]) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents square operation in the IR.
/// </summary>
public class SquareOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents L2 norm operation in the IR.
/// </summary>
public class NormOp : IROp
{
    /// <summary>
    /// The axis along which to compute the norm.
    /// </summary>
    public int Axis { get; set; } = -1;

    /// <summary>
    /// Whether to keep the reduced dimension.
    /// </summary>
    public bool KeepDims { get; set; } = false;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

// ============================================================================
// EMBEDDING AND ATTENTION OPERATIONS
// ============================================================================

/// <summary>
/// Represents embedding lookup operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Looks up embeddings for input indices from an embedding table.
/// </para>
/// </remarks>
public class EmbeddingOp : IROp
{
    /// <summary>
    /// Size of the vocabulary.
    /// </summary>
    public int NumEmbeddings { get; set; }

    /// <summary>
    /// Size of each embedding vector.
    /// </summary>
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Optional padding index that will output zeros.
    /// </summary>
    public int? PaddingIdx { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: indices, embedding_weights
        if (InputIds.Length != 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Embedding(t{InputIds[0]}, t{InputIds[1]}, dim={EmbeddingDim}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents scaled dot-product attention in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
/// </para>
/// </remarks>
public class ScaledDotProductAttentionOp : IROp
{
    /// <summary>
    /// Optional scaling factor. If not specified, uses 1/sqrt(d_k).
    /// </summary>
    public double? Scale { get; set; }

    /// <summary>
    /// Whether to apply causal (autoregressive) masking.
    /// </summary>
    public bool IsCausal { get; set; }

    /// <summary>
    /// Dropout probability for attention weights.
    /// </summary>
    public double DropoutProbability { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: query, key, value, optional mask
        if (InputIds.Length < 3 || InputIds.Length > 4) return false;
        return true;
    }

    public override string ToString()
    {
        var causalStr = IsCausal ? ", causal" : "";
        return $"t{OutputId} = ScaledDotProductAttention(q=t{InputIds[0]}, k=t{InputIds[1]}, v=t{InputIds[2]}{causalStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents a simplified attention operation for GPU code generation.
/// </summary>
/// <remarks>
/// <para>
/// This is a simplified version of attention used for GPU kernel generation.
/// Computes Attention(Q, K, V) = softmax(QK^T * scale) * V
/// </para>
/// </remarks>
public class AttentionOp : IROp
{
    /// <summary>
    /// Scaling factor for the attention scores.
    /// Typically 1/sqrt(head_dim).
    /// </summary>
    public double Scale { get; set; } = 1.0;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int NumHeads { get; set; } = 1;

    /// <summary>
    /// Head dimension (d_k).
    /// </summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>
    /// Sequence length.
    /// </summary>
    public int SeqLength { get; set; } = 512;

    /// <summary>
    /// Whether to apply causal (autoregressive) masking.
    /// </summary>
    public bool IsCausal { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: query, key, value (optionally mask)
        if (InputIds.Length < 3 || InputIds.Length > 4) return false;
        return true;
    }

    public override string ToString()
    {
        var causalStr = IsCausal ? ", causal" : "";
        return $"t{OutputId} = Attention(q=t{InputIds[0]}, k=t{InputIds[1]}, v=t{InputIds[2]}, scale={Scale}{causalStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents multi-head attention in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Multi-head attention allows the model to jointly attend to information
/// from different representation subspaces.
/// </para>
/// </remarks>
public class MultiHeadAttentionOp : IROp
{
    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    public int EmbedDim { get; set; }

    /// <summary>
    /// Key dimension per head.
    /// </summary>
    public int KeyDim { get; set; }

    /// <summary>
    /// Value dimension per head.
    /// </summary>
    public int ValueDim { get; set; }

    /// <summary>
    /// Dropout probability.
    /// </summary>
    public double DropoutProbability { get; set; }

    /// <summary>
    /// Whether this is self-attention (Q=K=V from same source).
    /// </summary>
    public bool IsSelfAttention { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: query, key, value, W_q, W_k, W_v, W_o, optional mask
        if (InputIds.Length < 7) return false;
        if (NumHeads <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = MultiHeadAttention(q=t{InputIds[0]}, k=t{InputIds[1]}, v=t{InputIds[2]}, heads={NumHeads}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

// ============================================================================
// FUSED OPERATIONS
// ============================================================================

/// <summary>
/// Represents fused MatMul + Add operation in the IR.
/// </summary>
public class FusedMatMulAddOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: A, B, bias
        if (InputIds.Length != 3) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = FusedMatMulAdd(t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents fused Linear + ReLU operation in the IR.
/// </summary>
public class FusedLinearReLUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input, weights, bias
        if (InputIds.Length != 3) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = FusedLinearReLU(t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents fused Conv + BatchNorm operation in the IR.
/// </summary>
public class FusedConvBatchNormOp : IROp
{
    /// <summary>
    /// Convolution stride.
    /// </summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>
    /// Convolution padding.
    /// </summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>
    /// BatchNorm epsilon.
    /// </summary>
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input, conv_weights, bn_gamma, bn_beta, bn_running_mean, bn_running_var
        if (InputIds.Length != 6) return false;
        return true;
    }
}

/// <summary>
/// Represents fused Add + ReLU operation in the IR.
/// </summary>
public class FusedAddReLUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = FusedAddReLU(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

// ============================================================================
// COMPLEX NUMBER OPERATIONS
// ============================================================================

/// <summary>
/// Represents complex matrix multiplication in the IR.
/// </summary>
public class ComplexMatMulOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: A_real, A_imag, B_real, B_imag
        if (InputIds.Length != 4) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise complex multiplication in the IR.
/// </summary>
public class ComplexMultiplyOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: A_real, A_imag, B_real, B_imag
        if (InputIds.Length != 4) return false;
        return true;
    }
}

// ============================================================================
// DROPOUT OPERATION
// ============================================================================

/// <summary>
/// Represents dropout operation in the IR.
/// </summary>
public class DropoutOp : IROp
{
    /// <summary>
    /// Dropout probability.
    /// </summary>
    public double Probability { get; set; } = 0.5;

    /// <summary>
    /// Whether in training mode.
    /// </summary>
    public bool Training { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
