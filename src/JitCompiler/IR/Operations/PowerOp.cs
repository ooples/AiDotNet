namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise power operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Power().
/// Raises each element to a power: result[i] = a[i] ^ exponent.
/// </para>
/// <para><b>For Beginners:</b> Raises each element to a power.
///
/// Example:
/// [2, 3, 4] ^ 2 = [4, 9, 16]
/// </para>
/// </remarks>
public class PowerOp : IROp
{
    /// <summary>
    /// The exponent to raise elements to.
    /// </summary>
    public double Exponent { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Power(t{InputIds[0]}, {Exponent}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
