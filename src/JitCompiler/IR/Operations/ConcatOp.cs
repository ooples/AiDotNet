namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents concatenation along an axis in the IR.
/// </summary>
public class ConcatOp : IROp
{
    public int Axis { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;  // Need at least 2 inputs to concat
        return true;
    }

    public override string ToString()
    {
        var inputs = string.Join(", ", InputIds.Select(id => $"t{id}"));
        return $"t{OutputId} = Concat([{inputs}], axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
