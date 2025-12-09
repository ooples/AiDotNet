namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents reshape operation in the IR.
/// </summary>
public class ReshapeOp : IROp
{
    public int[] NewShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (NewShape.Length == 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Reshape(t{InputIds[0]}, {NewShape.ShapeToString()}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
