namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Gradient accumulation operation - sums gradients from multiple paths.
/// </summary>
/// <remarks>
/// <para>
/// When a tensor is used by multiple operations, gradients flow back from
/// multiple paths. These must be summed to get the total gradient.
/// </para>
/// <para><b>For Beginners:</b> Combines gradients from different paths.
///
/// Example: If x is used in both y = x + 2 and z = x * 3
/// The gradient of x needs contributions from both operations:
/// grad_x = grad_from_y + grad_from_z
/// </para>
/// </remarks>
public class GradAccumulateOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Can have 2+ inputs to accumulate
        if (InputIds.Length < 2) return false;
        return true;
    }

    public override string ToString()
    {
        var inputs = string.Join(" + ", InputIds.Select(id => $"t{id}"));
        return $"t{OutputId} = AccumulateGrad({inputs}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
