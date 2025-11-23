namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise exponential function in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Exp().
/// Computes e^x for each element: result[i] = exp(a[i]).
/// </para>
/// <para><b>For Beginners:</b> Calculates e raised to the power of each element.
///
/// Example:
/// exp([0, 1, 2]) ≈ [1.0, 2.718, 7.389]
/// </para>
/// </remarks>
public class ExpOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise natural logarithm in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Log().
/// Computes natural log for each element: result[i] = ln(a[i]).
/// </para>
/// <para><b>For Beginners:</b> Calculates the natural logarithm of each element.
///
/// Example:
/// log([1, 2.718, 7.389]) ≈ [0, 1, 2]
/// </para>
/// </remarks>
public class LogOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise square root in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Sqrt().
/// Computes square root for each element: result[i] = √a[i].
/// </para>
/// <para><b>For Beginners:</b> Calculates the square root of each element.
///
/// Example:
/// sqrt([1, 4, 9, 16]) = [1, 2, 3, 4]
/// </para>
/// </remarks>
public class SqrtOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
