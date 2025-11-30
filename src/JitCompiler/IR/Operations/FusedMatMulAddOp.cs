namespace AiDotNet.JitCompiler.IR.Operations;

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
