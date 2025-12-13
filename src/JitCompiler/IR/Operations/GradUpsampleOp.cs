namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for UpsampleOp.
/// </summary>
public class GradUpsampleOp : BackwardOp
{
    /// <summary>Upsampling scale factor.</summary>
    public int Scale { get; set; }

    /// <summary>Interpolation mode used.</summary>
    public string Mode { get; set; } = "nearest";

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Scale <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradUpsample[scale={Scale}, mode={Mode}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
