namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Types of vectorized binary operations.
/// </summary>
public enum VectorizedBinaryOpType
{
    /// <summary>Element-wise addition.</summary>
    Add,
    /// <summary>Element-wise subtraction.</summary>
    Subtract,
    /// <summary>Element-wise multiplication.</summary>
    Multiply,
    /// <summary>Element-wise division.</summary>
    Divide,
    /// <summary>Element-wise maximum.</summary>
    Max,
    /// <summary>Element-wise minimum.</summary>
    Min,
    /// <summary>Element-wise power.</summary>
    Power
}

/// <summary>
/// Types of vectorized unary operations.
/// </summary>
public enum VectorizedUnaryOpType
{
    /// <summary>Negation.</summary>
    Negate,
    /// <summary>Absolute value.</summary>
    Abs,
    /// <summary>Exponential.</summary>
    Exp,
    /// <summary>Natural logarithm.</summary>
    Log,
    /// <summary>Square root.</summary>
    Sqrt,
    /// <summary>Reciprocal square root.</summary>
    Rsqrt,
    /// <summary>Square.</summary>
    Square,
    /// <summary>ReLU activation.</summary>
    ReLU,
    /// <summary>Sigmoid activation.</summary>
    Sigmoid,
    /// <summary>Hyperbolic tangent.</summary>
    Tanh,
    /// <summary>Floor function.</summary>
    Floor,
    /// <summary>Ceiling function.</summary>
    Ceil,
    /// <summary>Round function.</summary>
    Round
}

/// <summary>
/// Types of vectorized reduction operations.
/// </summary>
public enum VectorizedReductionType
{
    /// <summary>Sum reduction.</summary>
    Sum,
    /// <summary>Mean reduction.</summary>
    Mean,
    /// <summary>Maximum reduction.</summary>
    Max,
    /// <summary>Minimum reduction.</summary>
    Min,
    /// <summary>Product reduction.</summary>
    Product
}

/// <summary>
/// Vectorized binary operation (Add, Subtract, Multiply, Divide).
/// </summary>
public class VectorizedBinaryOp : IROp
{
    /// <summary>Gets or sets the operation type.</summary>
    public VectorizedBinaryOpType Operation { get; set; } = VectorizedBinaryOpType.Add;

    /// <summary>Gets or sets the vector width.</summary>
    public int VectorWidth { get; set; } = 4;

    /// <summary>Gets or sets the number of full vectors to process.</summary>
    public int NumVectors { get; set; }

    /// <summary>Gets or sets the number of remaining scalar elements.</summary>
    public int Remainder { get; set; }

    /// <summary>Validates the operation.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (VectorWidth < 1) return false;
        return true;
    }

    /// <summary>Returns a string representation.</summary>
    public override string ToString()
    {
        return $"t{OutputId} = Vectorized{Operation}[width={VectorWidth}, vecs={NumVectors}](t{InputIds[0]}, t{InputIds[1]})";
    }
}

/// <summary>
/// Vectorized unary operation (Negate, Exp, Log, Sqrt, ReLU, etc.).
/// </summary>
public class VectorizedUnaryOp : IROp
{
    /// <summary>Gets or sets the operation type.</summary>
    public VectorizedUnaryOpType Operation { get; set; } = VectorizedUnaryOpType.Negate;

    /// <summary>Gets or sets the vector width.</summary>
    public int VectorWidth { get; set; } = 4;

    /// <summary>Gets or sets the number of full vectors to process.</summary>
    public int NumVectors { get; set; }

    /// <summary>Gets or sets the number of remaining scalar elements.</summary>
    public int Remainder { get; set; }

    /// <summary>Validates the operation.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (VectorWidth < 1) return false;
        return true;
    }

    /// <summary>Returns a string representation.</summary>
    public override string ToString()
    {
        return $"t{OutputId} = Vectorized{Operation}[width={VectorWidth}, vecs={NumVectors}](t{InputIds[0]})";
    }
}

/// <summary>
/// Vectorized reduction operation (Sum, Mean, Max).
/// </summary>
public class VectorizedReductionOp : IROp
{
    /// <summary>Gets or sets the reduction type.</summary>
    public VectorizedReductionType ReductionType { get; set; } = VectorizedReductionType.Sum;

    /// <summary>Gets or sets the vector width.</summary>
    public int VectorWidth { get; set; } = 4;

    /// <summary>Gets or sets the axes to reduce over.</summary>
    public int[]? Axes { get; set; }

    /// <summary>Gets or sets whether to keep reduced dimensions.</summary>
    public bool KeepDims { get; set; } = false;

    /// <summary>Validates the operation.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    /// <summary>Returns a string representation.</summary>
    public override string ToString()
    {
        var axesStr = Axes != null ? $"[{string.Join(",", Axes)}]" : "all";
        return $"t{OutputId} = VectorizedReduce{ReductionType}[width={VectorWidth}, axes={axesStr}](t{InputIds[0]})";
    }
}

/// <summary>
/// Vectorized matrix multiplication operation.
/// </summary>
public class VectorizedMatMulOp : IROp
{
    /// <summary>Gets or sets the vector width.</summary>
    public int VectorWidth { get; set; } = 4;

    /// <summary>Gets or sets whether to use tiling.</summary>
    public bool UseTiling { get; set; } = true;

    /// <summary>Gets or sets the tile size.</summary>
    public int TileSize { get; set; } = 32;

    /// <summary>Validates the operation.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }

    /// <summary>Returns a string representation.</summary>
    public override string ToString()
    {
        return $"t{OutputId} = VectorizedMatMul[width={VectorWidth}, tile={TileSize}](t{InputIds[0]}, t{InputIds[1]})";
    }
}
