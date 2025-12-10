namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents cropping operation in the IR.
/// </summary>
public class CropOp : IROp
{
    /// <summary>Cropping amounts per dimension.</summary>
    public int[] Cropping { get; set; } = Array.Empty<int>();

    /// <summary>Offset positions for cropping [start indices per dimension].</summary>
    public int[] Offsets { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
