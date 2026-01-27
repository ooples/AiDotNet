using System.Numerics;

namespace AiDotNet.InferenceOptimization.IR.Common;

/// <summary>
/// Represents the data type of a tensor in the IR.
/// Exceeds industry standards by supporting all common ML types plus quantized types.
/// </summary>
/// <remarks>
/// <para><b>Industry Comparison:</b></para>
/// <list type="bullet">
/// <item>MLIR: Uses builtin types with explicit bit widths</item>
/// <item>XLA: PrimitiveType enum with similar coverage</item>
/// <item>TVM: DataType class with code/bits/lanes</item>
/// <item>Our approach: Comprehensive enum with quantization support and extension points</item>
/// </list>
/// </remarks>
public enum IRDataType
{
    // Standard floating point
    Float16,
    Float32,
    Float64,
    BFloat16,

    // Standard integers
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,

    // Specialized types
    Bool,
    Complex64,
    Complex128,
    Decimal,

    // Quantized types (exceeds most frameworks)
    QInt8,      // Quantized int8 with scale/zero-point
    QUInt8,     // Quantized uint8 with scale/zero-point
    QInt4,      // 4-bit quantized (for LLMs)
    QInt2,      // 2-bit quantized (extreme compression)

    // Dynamic/unknown
    Unknown
}

/// <summary>
/// Memory layout for tensors. Critical for hardware optimization.
/// </summary>
/// <remarks>
/// <para><b>Industry Comparison:</b></para>
/// <list type="bullet">
/// <item>TVM: Uses layout strings like "NCHW", "NHWC"</item>
/// <item>ONNX: Implicit layouts based on operator</item>
/// <item>Our approach: Explicit enum with all common layouts plus extensibility</item>
/// </list>
/// </remarks>
public enum MemoryLayout
{
    // Standard layouts
    RowMajor,           // C-style, last dimension contiguous
    ColumnMajor,        // Fortran-style, first dimension contiguous

    // Image layouts
    NCHW,               // Batch, Channel, Height, Width (PyTorch default)
    NHWC,               // Batch, Height, Width, Channel (TensorFlow default)
    CHWN,               // For specific hardware optimizations

    // Tiled layouts (for GPU/TPU optimization)
    Tiled4x4,
    Tiled8x8,
    Tiled16x16,
    Tiled32x32,

    // Blocked layouts (for CPU SIMD)
    Blocked,

    // Custom/unknown
    Custom,
    Unknown
}

/// <summary>
/// Execution device target for operations.
/// </summary>
public enum DeviceType
{
    CPU,
    GPU,
    TPU,
    NPU,
    FPGA,
    Auto,       // Let the scheduler decide
    Any         // Can run on any device
}

/// <summary>
/// Quantization parameters for quantized tensor types.
/// </summary>
public class QuantizationParams
{
    public double Scale { get; set; } = 1.0;
    public int ZeroPoint { get; set; } = 0;
    public double Min { get; set; } = double.MinValue;
    public double Max { get; set; } = double.MaxValue;
    public bool PerChannel { get; set; } = false;
    public int QuantizationAxis { get; set; } = -1;
    public double[]? PerChannelScales { get; set; }
    public int[]? PerChannelZeroPoints { get; set; }
}

/// <summary>
/// Comprehensive tensor type information.
/// Exceeds industry standards by combining type, shape, layout, and device info.
/// </summary>
public class TensorType
{
    /// <summary>
    /// Element data type.
    /// </summary>
    public IRDataType DataType { get; set; } = IRDataType.Float32;

    /// <summary>
    /// Tensor shape. Empty for scalars, -1 for dynamic dimensions.
    /// </summary>
    public int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Memory layout.
    /// </summary>
    public MemoryLayout Layout { get; set; } = MemoryLayout.RowMajor;

    /// <summary>
    /// Target device.
    /// </summary>
    public DeviceType Device { get; set; } = DeviceType.Auto;

    /// <summary>
    /// Quantization parameters (if quantized type).
    /// </summary>
    public QuantizationParams? Quantization { get; set; }

    /// <summary>
    /// Strides for each dimension (computed from shape and layout if not specified).
    /// </summary>
    public long[]? Strides { get; set; }

    /// <summary>
    /// Whether this tensor has dynamic (runtime-determined) shape.
    /// </summary>
    public bool HasDynamicShape => Shape.Any(d => d < 0);

    /// <summary>
    /// Total number of elements (returns -1 if dynamic).
    /// </summary>
    public long NumElements
    {
        get
        {
            if (HasDynamicShape) return -1;
            if (Shape.Length == 0) return 1; // scalar
            return Shape.Aggregate(1L, (acc, dim) => acc * dim);
        }
    }

    /// <summary>
    /// Size in bytes of each element.
    /// </summary>
    public int ElementSize => DataType switch
    {
        IRDataType.Bool => 1,
        IRDataType.Int8 or IRDataType.UInt8 or IRDataType.QInt8 or IRDataType.QUInt8 => 1,
        IRDataType.Int16 or IRDataType.UInt16 or IRDataType.Float16 or IRDataType.BFloat16 => 2,
        IRDataType.Int32 or IRDataType.UInt32 or IRDataType.Float32 => 4,
        IRDataType.Int64 or IRDataType.UInt64 or IRDataType.Float64 or IRDataType.Complex64 => 8,
        IRDataType.Complex128 or IRDataType.Decimal => 16,
        IRDataType.QInt4 or IRDataType.QInt2 => 1, // packed
        _ => 8
    };

    /// <summary>
    /// Total memory size in bytes.
    /// </summary>
    public long TotalBytes => NumElements >= 0 ? NumElements * ElementSize : -1;

    /// <summary>
    /// Check if this type is compatible with another for broadcasting.
    /// </summary>
    public bool IsBroadcastCompatible(TensorType other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (HasDynamicShape || other.HasDynamicShape) return true;

        int maxRank = Math.Max(Shape.Length, other.Shape.Length);
        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < Shape.Length ? Shape[Shape.Length - 1 - i] : 1;
            int dim2 = i < other.Shape.Length ? other.Shape[other.Shape.Length - 1 - i] : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) return false;
        }
        return true;
    }

    public TensorType Clone() => new()
    {
        DataType = DataType,
        Shape = (int[])Shape.Clone(),
        Layout = Layout,
        Device = Device,
        Quantization = Quantization,
        Strides = Strides != null ? (long[])Strides.Clone() : null
    };

    public override string ToString()
    {
        var shape = Shape.Length == 0 ? "scalar" : $"[{string.Join(", ", Shape.Select(d => d < 0 ? "?" : d.ToString()))}]";
        return $"{DataType}{shape}@{Device}";
    }
}

/// <summary>
/// Extension methods for IRDataType.
/// </summary>
public static class IRDataTypeExtensions
{
    public static IRDataType FromSystemType(Type type)
    {
        if (type == null)
            throw new ArgumentNullException(nameof(type));

        return type switch
        {
            Type t when t == typeof(float) => IRDataType.Float32,
            Type t when t == typeof(double) => IRDataType.Float64,
            Type t when t == typeof(Half) => IRDataType.Float16,
            Type t when t == typeof(int) => IRDataType.Int32,
            Type t when t == typeof(long) => IRDataType.Int64,
            Type t when t == typeof(short) => IRDataType.Int16,
            Type t when t == typeof(byte) => IRDataType.UInt8,
            Type t when t == typeof(sbyte) => IRDataType.Int8,
            Type t when t == typeof(ushort) => IRDataType.UInt16,
            Type t when t == typeof(uint) => IRDataType.UInt32,
            Type t when t == typeof(ulong) => IRDataType.UInt64,
            Type t when t == typeof(bool) => IRDataType.Bool,
            Type t when t == typeof(decimal) => IRDataType.Decimal,
            Type t when t == typeof(Complex) => IRDataType.Complex128,
            _ => IRDataType.Unknown
        };
    }

    public static Type ToSystemType(this IRDataType type)
    {
        return type switch
        {
            IRDataType.Float16 => typeof(Half),
            IRDataType.Float32 => typeof(float),
            IRDataType.Float64 => typeof(double),
            IRDataType.BFloat16 => typeof(Half), // Approximation
            IRDataType.Int8 or IRDataType.QInt8 => typeof(sbyte),
            IRDataType.Int16 => typeof(short),
            IRDataType.Int32 => typeof(int),
            IRDataType.Int64 => typeof(long),
            IRDataType.UInt8 or IRDataType.QUInt8 or IRDataType.QInt4 or IRDataType.QInt2 => typeof(byte), // Packed quantized types stored as bytes
            IRDataType.UInt16 => typeof(ushort),
            IRDataType.UInt32 => typeof(uint),
            IRDataType.UInt64 => typeof(ulong),
            IRDataType.Bool => typeof(bool),
            IRDataType.Decimal => typeof(decimal),
            IRDataType.Complex64 or IRDataType.Complex128 => typeof(Complex),
            _ => typeof(double)
        };
    }

    public static bool IsFloatingPoint(this IRDataType type) =>
        type is IRDataType.Float16 or IRDataType.Float32 or IRDataType.Float64 or IRDataType.BFloat16;

    public static bool IsInteger(this IRDataType type) =>
        type is IRDataType.Int8 or IRDataType.Int16 or IRDataType.Int32 or IRDataType.Int64 or
               IRDataType.UInt8 or IRDataType.UInt16 or IRDataType.UInt32 or IRDataType.UInt64;

    public static bool IsQuantized(this IRDataType type) =>
        type is IRDataType.QInt8 or IRDataType.QUInt8 or IRDataType.QInt4 or IRDataType.QInt2;

    /// <summary>
    /// Gets the size in bytes of each element for the given data type.
    /// </summary>
    public static int ElementSizeInBytes(this IRDataType type) => type switch
    {
        IRDataType.Bool => 1,
        IRDataType.Int8 or IRDataType.UInt8 or IRDataType.QInt8 or IRDataType.QUInt8 => 1,
        IRDataType.Int16 or IRDataType.UInt16 or IRDataType.Float16 or IRDataType.BFloat16 => 2,
        IRDataType.Int32 or IRDataType.UInt32 or IRDataType.Float32 => 4,
        IRDataType.Int64 or IRDataType.UInt64 or IRDataType.Float64 or IRDataType.Complex64 => 8,
        IRDataType.Complex128 or IRDataType.Decimal => 16,
        IRDataType.QInt4 or IRDataType.QInt2 => 1, // packed representation
        _ => 8 // default to 8 for unknown types
    };
}
