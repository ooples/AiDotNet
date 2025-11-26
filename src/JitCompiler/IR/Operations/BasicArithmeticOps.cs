namespace AiDotNet.JitCompiler.IR.Operations;

// ============================================================================
// CONSTANT OPERATIONS
// ============================================================================

/// <summary>
/// Represents a constant tensor in the IR (result of constant folding).
/// </summary>
/// <remarks>
/// <para>
/// ConstantOp stores pre-computed tensor values that were evaluated at compile time.
/// This is the result of constant folding optimization, where expressions with
/// all constant inputs are computed during compilation rather than at runtime.
/// </para>
/// <para><b>For Beginners:</b> A ConstantOp holds a pre-calculated result.
///
/// When the compiler sees:
///   t0 = Constant([2.0])
///   t1 = Constant([3.0])
///   t2 = Add(t0, t1)
///
/// It computes 2.0 + 3.0 = 5.0 at compile time and replaces with:
///   t2 = Constant([5.0])
///
/// Benefits:
/// - No addition happens at runtime
/// - Less memory for intermediate tensors
/// - Faster execution
/// </para>
/// </remarks>
public class ConstantOp : IROp
{
    /// <summary>
    /// Gets or sets the constant values as a flat array.
    /// </summary>
    /// <remarks>
    /// Values are stored as double for precision. They can be cast to the
    /// appropriate type during code generation based on OutputType.
    /// </remarks>
    public double[] Values { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets a flag indicating whether this is a scalar constant.
    /// </summary>
    public bool IsScalar => OutputShape.Length == 0 || (OutputShape.Length == 1 && OutputShape[0] == 1);

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Constants should have no inputs
        if (InputIds.Length != 0) return false;
        // Values should match the shape
        var expectedSize = OutputShape.Length == 0 ? 1 : OutputShape.Aggregate(1, (a, b) => a * b);
        if (Values.Length != expectedSize) return false;
        return true;
    }

    public override string ToString()
    {
        var valueStr = Values.Length <= 4
            ? $"[{string.Join(", ", Values.Take(4).Select(v => v.ToString("G4")))}]"
            : $"[{Values[0]:G4}, ..., {Values[^1]:G4}] ({Values.Length} elements)";
        return $"t{OutputId} = Constant({valueStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents a scalar constant in the IR (single value).
/// </summary>
/// <remarks>
/// <para>
/// ScalarConstantOp is a specialized version of ConstantOp for single values.
/// It's more efficient for storing scalar values used in operations.
/// </para>
/// <para><b>For Beginners:</b> A ScalarConstantOp holds a single number.
///
/// Examples:
/// - Learning rate: 0.001
/// - Epsilon for numerical stability: 1e-7
/// - Scale factor: 2.0
///
/// These are used in operations like:
/// - result = input * 0.001 (scaling by learning rate)
/// - result = input + 1e-7 (adding epsilon)
/// </para>
/// </remarks>
public class ScalarConstantOp : IROp
{
    /// <summary>
    /// Gets or sets the scalar value.
    /// </summary>
    public double Value { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 0) return false;
        // Scalar should have empty or [1] shape
        if (OutputShape.Length > 1) return false;
        if (OutputShape.Length == 1 && OutputShape[0] != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Scalar({Value:G6}) : {OutputType}";
    }
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

/// <summary>
/// Represents element-wise addition in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Add().
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
/// Corresponds to TensorOperations<T>.Subtract().
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
/// Corresponds to TensorOperations<T>.ElementwiseMultiply().
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
/// Corresponds to TensorOperations<T>.Divide().
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
/// Corresponds to TensorOperations<T>.Power().
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
/// Corresponds to TensorOperations<T>.Negate().
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
