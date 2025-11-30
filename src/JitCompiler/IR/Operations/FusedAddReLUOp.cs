namespace AiDotNet.JitCompiler.IR.Operations;

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
