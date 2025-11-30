namespace AiDotNet.JitCompiler.IR.Operations;

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
