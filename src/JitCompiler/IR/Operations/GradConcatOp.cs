namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ConcatOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = concat([x1, x2, ...], axis)
/// Backward: grad_xi = slice(grad_y, start_i, end_i, axis)
/// Each input gets a slice of the output gradient.
/// </para>
/// </remarks>
public class GradConcatOp : BackwardOp
{
    /// <summary>Which input are we computing gradient for.</summary>
    public int InputIndex { get; set; }

    /// <summary>Concatenation axis.</summary>
    public int Axis { get; set; }

    /// <summary>Start index along axis for this input's gradient.</summary>
    public int StartIndex { get; set; }

    /// <summary>Size along axis for this input.</summary>
    public int Size { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradConcat[input={InputIndex}, axis={Axis}, start={StartIndex}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
