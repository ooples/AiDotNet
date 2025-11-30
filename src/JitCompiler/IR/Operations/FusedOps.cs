namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused linear operation (MatMul + Add bias).
/// </summary>
/// <remarks>
/// <para>
/// Combines matrix multiplication and bias addition into a single operation.
/// This is the fundamental operation of a neural network dense/linear layer.
/// </para>
/// <para><b>For Beginners:</b> This combines two operations into one.
///
/// Instead of:
///   t1 = MatMul(input, weights)  // Matrix multiply
///   t2 = Add(t1, bias)           // Add bias
///
/// We do:
///   t2 = Linear(input, weights, bias)  // One operation!
///
/// Benefits:
/// - Fewer memory reads/writes
/// - Better cache utilization
/// - Less overhead
/// - Typically 1.5-2x faster
/// </para>
/// </remarks>
public class FusedLinearOp : IROp
{
    /// <summary>
    /// Validates that this operation has correct inputs (3 inputs: input, weights, bias).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;  // input, weights, bias
        return true;
    }
}

/// <summary>
/// Fused linear + activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines linear layer with activation function.
///
/// Instead of:
///   t1 = Linear(input, weights, bias)
///   t2 = ReLU(t1)
///
/// We do:
///   t2 = LinearReLU(input, weights, bias)
///
/// Common in neural networks - almost every layer has an activation!
/// </para>
/// </remarks>
public class FusedLinearActivationOp : IROp
{
    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs.
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}

/// <summary>
/// Fused convolution + batch normalization operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines convolution with batch normalization.
///
/// Batch normalization after convolution is extremely common in CNNs.
/// By fusing them, we can:
/// - Fold BN parameters into conv weights (at inference time)
/// - Skip intermediate tensor storage
/// - Reduce memory bandwidth significantly
///
/// This can be 2-3x faster than separate operations!
/// </para>
/// </remarks>
public class FusedConvBatchNormOp : IROp
{
    /// <summary>
    /// Gets or sets the convolution stride.
    /// </summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>
    /// Gets or sets the convolution padding.
    /// </summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>
    /// Gets or sets the batch norm epsilon value.
    /// </summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the batch norm momentum.
    /// </summary>
    public double Momentum { get; set; } = 0.1;

    /// <summary>
    /// Validates inputs (input, kernel, gamma, beta, running_mean, running_var).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 6) return false;  // input, kernel, gamma, beta, running_mean, running_var
        return true;
    }
}

/// <summary>
/// Fused element-wise operation with activation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines element-wise math with activation.
///
/// Examples:
///   Add + ReLU
///   Multiply + Sigmoid
///   Subtract + Tanh
///
/// Very common in residual connections and skip connections.
/// Saves memory by not storing intermediate results.
/// </para>
/// </remarks>
public class FusedElementwiseActivationOp : IROp
{
    /// <summary>
    /// Gets or sets the element-wise operation type.
    /// </summary>
    public string ElementwiseOp { get; set; } = "Add";

    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs (2 inputs for binary element-wise ops).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (string.IsNullOrEmpty(ElementwiseOp) || string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}

/// <summary>
/// Fused matrix multiply + add + activation (full dense layer).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The ultimate fusion - entire dense layer in one op!
///
/// Combines:
///   MatMul + Add bias + Activation → One operation
///
/// Example:
///   output = activation(input @ weights + bias)
///
/// This is THE most common pattern in neural networks.
/// Can be 3-5x faster than three separate operations!
/// </para>
/// </remarks>
public class FusedDenseLayerOp : IROp
{
    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs (input, weights, bias).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}

/// <summary>
/// Fused residual block operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Fuses a residual/skip connection pattern.
///
/// Residual blocks are everywhere in modern networks (ResNet, Transformers, etc.)
/// Pattern:
///   output = activation(main_path + skip_connection)
///
/// By fusing this, we can:
/// - Optimize the addition and activation together
/// - Reduce memory traffic
/// - Better utilize CPU/GPU resources
/// </para>
/// </remarks>
public class FusedResidualBlockOp : IROp
{
    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs (main_path, skip_connection).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}

/// <summary>
/// Fused batch normalization + activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines batch norm with activation.
///
/// BatchNorm followed by ReLU is extremely common in CNNs.
/// Fusing them reduces memory traffic and improves performance.
///
/// Pattern:
///   x_norm = (x - mean) / sqrt(var + epsilon)
///   output = activation(gamma * x_norm + beta)
/// </para>
/// </remarks>
public class FusedBatchNormActivationOp : IROp
{
    /// <summary>Gets or sets the activation function name.</summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>Gets or sets epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Gets or sets momentum for running statistics.</summary>
    public double Momentum { get; set; } = 0.1;

    /// <summary>Validates inputs.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 5) return false; // input, gamma, beta, running_mean, running_var
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}

/// <summary>
/// Fused layer normalization + add operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines LayerNorm with residual addition.
///
/// Very common in Transformers:
///   output = LayerNorm(x) + residual
///
/// Fusing reduces memory reads/writes.
/// </para>
/// </remarks>
public class FusedLayerNormAddOp : IROp
{
    /// <summary>Gets or sets the normalized shape.</summary>
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();

    /// <summary>Gets or sets epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Validates inputs (x, gamma, beta, residual).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 4) return false;
        return true;
    }
}

/// <summary>
/// Fused add + layer normalization operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines residual addition with LayerNorm.
///
/// Common in Transformer blocks:
///   output = LayerNorm(x + residual)
///
/// Reduces memory traffic by avoiding intermediate storage.
/// </para>
/// </remarks>
public class FusedAddLayerNormOp : IROp
{
    /// <summary>Gets or sets the normalized shape.</summary>
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();

    /// <summary>Gets or sets epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Validates inputs (a, b, gamma, beta).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 4) return false;
        return true;
    }
}

/// <summary>
/// Fused chain of element-wise operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines multiple element-wise ops into one.
///
/// Instead of:
///   t1 = Add(a, b)
///   t2 = ReLU(t1)
///   t3 = Multiply(t2, c)
///
/// One fused operation processes all three steps together.
/// Saves memory by not storing intermediate results.
/// </para>
/// </remarks>
public class FusedElementwiseChainOp : IROp
{
    /// <summary>Gets or sets the list of operations in the chain.</summary>
    public List<string> Operations { get; set; } = new();

    /// <summary>Validates the operation chain.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (Operations.Count < 2) return false;
        return true;
    }
}

/// <summary>
/// Fused attention operation (Q*K^T + softmax + matmul V).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The core of Transformer models!
///
/// Attention:
///   scores = Q @ K^T / sqrt(d_k)
///   weights = softmax(scores)
///   output = weights @ V
///
/// This is the most expensive part of transformers.
/// Fusing allows optimizations like Flash Attention for massive speedups.
/// </para>
/// </remarks>
public class FusedAttentionOp : IROp
{
    /// <summary>Gets or sets the softmax axis.</summary>
    public int SoftmaxAxis { get; set; } = -1;

    /// <summary>Gets or sets the scaling factor (typically 1/sqrt(d_k)).</summary>
    public double Scale { get; set; } = 1.0;

    /// <summary>Gets or sets whether to use causal masking.</summary>
    public bool CausalMask { get; set; } = false;

    /// <summary>Validates inputs (Q, K, V).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        return true;
    }
}

/// <summary>
/// Fused Swish/SiLU activation (x * sigmoid(x)).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A popular activation function.
///
/// Swish(x) = x * sigmoid(x)
///
/// Used in EfficientNet and other modern architectures.
/// Fusing avoids computing sigmoid separately.
/// </para>
/// </remarks>
public class FusedSwishOp : IROp
{
    /// <summary>Validates inputs (single input).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Fused Conv2D + BatchNorm + Activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The complete CNN layer in one operation!
///
/// Combines:
///   1. Convolution
///   2. Batch normalization
///   3. Activation (ReLU, etc.)
///
/// This is THE most common pattern in CNNs.
/// Can be 3-5x faster than separate operations.
/// </para>
/// </remarks>
public class FusedConvBatchNormActivationOp : IROp
{
    /// <summary>Gets or sets the convolution stride.</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Gets or sets the convolution padding.</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>Gets or sets the batch norm epsilon.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Gets or sets the batch norm momentum.</summary>
    public double Momentum { get; set; } = 0.1;

    /// <summary>Gets or sets the activation function name.</summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>Validates inputs.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 6) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}

/// <summary>
/// Fused GELU activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Gaussian Error Linear Unit.
///
/// GELU(x) = x * Phi(x), where Phi is the standard Gaussian CDF.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Very popular in transformers (BERT, GPT, etc.)
/// </para>
/// </remarks>
public class FusedGELUOp : IROp
{
    /// <summary>Whether to use the approximate version.</summary>
    public bool Approximate { get; set; } = true;

    /// <summary>Validates inputs (single input).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Fused multi-head attention operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Multi-head attention for transformers.
///
/// Splits Q, K, V into multiple heads, applies attention, then concatenates.
/// This is the complete attention layer including all projections.
/// </para>
/// </remarks>
public class FusedMultiHeadAttentionOp : IROp
{
    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the head dimension.</summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>Gets or sets whether to use causal masking.</summary>
    public bool CausalMask { get; set; } = false;

    /// <summary>Gets or sets dropout probability.</summary>
    public double Dropout { get; set; } = 0.0;

    /// <summary>Validates inputs (query, key, value).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 3) return false;
        return true;
    }
}

/// <summary>
/// Fused bias + activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Adds bias and applies activation together.
///
/// output = activation(input + bias)
///
/// Common after linear/conv layers without built-in bias.
/// </para>
/// </remarks>
public class FusedBiasActivationOp : IROp
{
    /// <summary>Gets or sets the activation function name.</summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>Validates inputs (input, bias).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
