namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused Add (residual) + GroupNorm in a single data traversal.
/// Computes: GroupNorm(a + b, gamma, beta) without materializing the (a + b) intermediate.
/// </summary>
/// <remarks>
/// <para>
/// In a DiffusionResBlock, after the second conv output is computed, it gets added
/// to the skip connection (residual) and the result feeds into the next block's GroupNorm.
/// Fusing the add and norm saves one full tensor allocation and one data traversal.
/// </para>
/// <para>
/// Inputs: [a, b, gamma, beta] where:
/// - a: first operand of the add (e.g., conv output)
/// - b: second operand (e.g., skip connection)
/// - gamma: GroupNorm scale [channels]
/// - beta: GroupNorm shift [channels]
/// Output: GroupNorm(a + b, gamma, beta)
/// </para>
/// </remarks>
public class FusedAddGroupNormOp : IROp
{
    public int NumGroups { get; set; } = 32;
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // a, b, gamma, beta
        if (InputIds.Length != 4) return false;
        if (NumGroups <= 0) return false;
        return true;
    }
}
