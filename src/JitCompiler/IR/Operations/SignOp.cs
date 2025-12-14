namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Sign activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Sign(x) = -1 if x &lt; 0, 0 if x == 0, 1 if x &gt; 0.
/// Hard threshold activation, commonly used in binary networks.
/// </para>
/// </remarks>
public class SignOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
