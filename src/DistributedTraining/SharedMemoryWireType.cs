namespace AiDotNet.DistributedTraining;

/// <summary>
/// How a shared-memory communication backend encodes one element in its collective slots. Every rank of a session
/// must use the same encoding.
/// </summary>
internal enum SharedMemoryWireType
{
    /// <summary>32-bit IEEE-754 floats.</summary>
    Float32 = 1,

    /// <summary>32-bit signed integers.</summary>
    Int32 = 2,

    /// <summary>64-bit IEEE-754 doubles.</summary>
    Float64 = 3,

    /// <summary>64-bit signed integers.</summary>
    Int64 = 4,

    /// <summary>Single bytes.</summary>
    Byte = 5,

    /// <summary>Any other numeric type, converted to 64-bit doubles.</summary>
    DoubleFallback = 6,
}
