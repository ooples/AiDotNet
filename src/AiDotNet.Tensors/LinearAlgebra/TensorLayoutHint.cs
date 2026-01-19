using System;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Flags indicating the memory layout and format of a tensor.
/// </summary>
/// <remarks>
/// <para>
/// These hints are used to track the current memory layout of tensors, enabling
/// optimizations that avoid unnecessary format conversions between operations.
/// </para>
/// <para><b>For Beginners:</b> Different operations work best with data organized
/// in different ways. These flags tell us how the data is currently organized,
/// so we can skip reorganizing it if it's already in the right format.
/// </para>
/// </remarks>
[Flags]
internal enum TensorLayoutHint
{
    /// <summary>No specific layout hint.</summary>
    None = 0,

    /// <summary>Data is stored in row-major order (C-style, last dimension varies fastest).</summary>
    RowMajor = 1 << 0,

    /// <summary>Data is stored in column-major order (Fortran-style, first dimension varies fastest).</summary>
    ColumnMajor = 1 << 1,

    /// <summary>Data is packed for optimized matrix multiplication.</summary>
    PackedForMatMul = 1 << 2,

    /// <summary>
    /// Data is in oneDNN blocked format (e.g., nChw16c, nChw8c).
    /// This format enables SIMD-optimized convolution operations.
    /// </summary>
    OneDnnBlocked = 1 << 3,

    /// <summary>
    /// Data format was chosen by oneDNN using format_tag_any.
    /// The specific format is optimal for the current operation and hardware.
    /// </summary>
    OneDnnOptimal = 1 << 4,

    /// <summary>
    /// Data is GPU-resident and should not be accessed from CPU without synchronization.
    /// </summary>
    GpuResident = 1 << 5
}
