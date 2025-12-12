namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Spherical Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Projects inputs onto a unit sphere before applying softmax.
/// Useful for directional data or when angular relationships matter.
/// </para>
/// </remarks>
public class SphericalSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute spherical softmax. Default is -1.
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = SphericalSoftmax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
