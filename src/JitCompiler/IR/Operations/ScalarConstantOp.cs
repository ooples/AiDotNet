namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents a scalar constant in the IR (single value).
/// </summary>
/// <remarks>
/// <para>
/// ScalarConstantOp is a specialized version of ConstantOp for single values.
/// It's more efficient for storing scalar values used in operations.
/// </para>
/// <para><b>For Beginners:</b> A ScalarConstantOp holds a single number.
///
/// Examples:
/// - Learning rate: 0.001
/// - Epsilon for numerical stability: 1e-7
/// - Scale factor: 2.0
///
/// These are used in operations like:
/// - result = input * 0.001 (scaling by learning rate)
/// - result = input + 1e-7 (adding epsilon)
/// </para>
/// </remarks>
public class ScalarConstantOp : IROp
{
    /// <summary>
    /// Gets or sets the scalar value.
    /// </summary>
    public double Value { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 0) return false;
        // Scalar should have empty or [1] shape
        if (OutputShape.Length > 1) return false;
        if (OutputShape.Length == 1 && OutputShape[0] != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Scalar({Value:G6}) : {OutputType}";
    }
}
