namespace AiDotNet.JitCompiler.IR.Operations;

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
