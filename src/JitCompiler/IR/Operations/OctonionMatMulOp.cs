namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents octonion matrix multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// This operation performs octonion-valued matrix multiplication for neural network layers.
/// Each element in the result is a sum of octonion products.
/// </para>
/// <para><b>For Beginners:</b> This is like regular matrix multiplication but using
/// 8-dimensional octonion numbers instead of regular numbers. It's used in
/// octonion neural networks for specialized processing tasks.
/// </para>
/// </remarks>
public class OctonionMatMulOp : IROp
{
    /// <summary>
    /// Gets or sets the batch size for the matrix multiplication.
    /// </summary>
    public int BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the number of input features (columns of input).
    /// </summary>
    public int InputFeatures { get; set; }

    /// <summary>
    /// Gets or sets the number of output features (rows of weight matrix).
    /// </summary>
    public int OutputFeatures { get; set; }

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input tensor, weight tensor (both octonion-valued)
        if (InputIds.Length != 2) return false;
        if (BatchSize <= 0 || InputFeatures <= 0 || OutputFeatures <= 0) return false;
        return true;
    }
}
