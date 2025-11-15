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
