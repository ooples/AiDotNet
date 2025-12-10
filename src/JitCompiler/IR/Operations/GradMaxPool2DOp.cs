namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for MaxPool2DOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: Records indices of max elements
/// Backward: Routes gradient only to max elements
/// </para>
/// </remarks>
public class GradMaxPool2DOp : BackwardOp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward indices/input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMaxPool2D(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
