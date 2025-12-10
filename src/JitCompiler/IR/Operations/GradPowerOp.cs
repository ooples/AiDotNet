namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for PowerOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x^p
/// Backward: grad_x = grad_y * p * x^(p-1)
/// </para>
/// </remarks>
public class GradPowerOp : BackwardOp
{
    /// <summary>Exponent used in forward pass.</summary>
    public double Exponent { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradPower[exp={Exponent}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
