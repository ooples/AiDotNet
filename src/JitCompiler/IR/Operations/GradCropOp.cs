namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for CropOp.
/// </summary>
public class GradCropOp : BackwardOp
{
    /// <summary>Original shape before cropping.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Crop offsets used in forward.</summary>
    public int[] CropOffsets { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradCrop[offsets={string.Join(",", CropOffsets)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
