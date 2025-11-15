namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise addition in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.Add().
/// Performs element-wise addition of two tensors: result[i] = a[i] + b[i].
/// </para>
/// <para><b>For Beginners:</b> Adds two tensors together, element by element.
///
/// Example:
/// [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
///
/// Supports broadcasting:
/// [1, 2, 3] + 5 = [6, 7, 8]
/// </para>
/// </remarks>
public class AddOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise subtraction in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.Subtract().
/// Performs element-wise subtraction: result[i] = a[i] - b[i].
/// </para>
/// <para><b>For Beginners:</b> Subtracts one tensor from another, element by element.
///
/// Example:
/// [5, 7, 9] - [1, 2, 3] = [4, 5, 6]
/// </para>
/// </remarks>
public class SubtractOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.ElementwiseMultiply().
/// Performs Hadamard (element-wise) product: result[i] = a[i] * b[i].
/// This is different from matrix multiplication.
/// </para>
/// <para><b>For Beginners:</b> Multiplies tensors element by element.
///
/// Example:
/// [1, 2, 3] * [4, 5, 6] = [4, 10, 18]
///
/// This is NOT matrix multiplication! Each element is multiplied independently.
/// </para>
/// </remarks>
public class ElementwiseMultiplyOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise division in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.Divide().
/// Performs element-wise division: result[i] = a[i] / b[i].
/// </para>
/// <para><b>For Beginners:</b> Divides one tensor by another, element by element.
///
/// Example:
/// [10, 20, 30] / [2, 4, 5] = [5, 5, 6]
/// </para>
/// </remarks>
public class DivideOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}

/// <summary>
/// Represents element-wise power operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.Power().
/// Raises each element to a power: result[i] = a[i] ^ exponent.
/// </para>
/// <para><b>For Beginners:</b> Raises each element to a power.
///
/// Example:
/// [2, 3, 4] ^ 2 = [4, 9, 16]
/// </para>
/// </remarks>
public class PowerOp : IROp
{
    /// <summary>
    /// The exponent to raise elements to.
    /// </summary>
    public double Exponent { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Power(t{InputIds[0]}, {Exponent}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents element-wise negation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.Negate().
/// Negates each element: result[i] = -a[i].
/// </para>
/// <para><b>For Beginners:</b> Flips the sign of each element.
///
/// Example:
/// -[1, -2, 3] = [-1, 2, -3]
/// </para>
/// </remarks>
public class NegateOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
