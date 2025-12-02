namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for AvgPool2DOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: Average values in each window
/// Backward: Distributes gradient equally to all elements in window
/// </para>
/// </remarks>
public class GradAvgPool2DOp : BackwardOp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false; // Only needs grad_output
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradAvgPool2D(t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
