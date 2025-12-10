namespace AiDotNet.JitCompiler.IR.Operations;

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
