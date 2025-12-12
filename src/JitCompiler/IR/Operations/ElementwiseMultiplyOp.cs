namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.ElementwiseMultiply().
/// Performs Hadamard (element-wise) product: result[i] = a[i] * b[i].
/// This is different from matrix multiplication.
/// </para>
/// <para><b>For Beginners:</b> Multiplies tensors element by element.
///
/// Example:
/// [1, 2, 3] * [4, 5, 6] = [4, 10, 18]
///
/// This is NOT matrix multiplication! Each element is multiplied independently.
/// </para>
/// </remarks>
public class ElementwiseMultiplyOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}
