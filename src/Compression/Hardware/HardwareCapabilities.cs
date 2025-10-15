using AiDotNet.Enums;

namespace AiDotNet.Compression.Hardware;

/// <summary>
/// Container for hardware capabilities detected by the InferenceOptimizer.
/// </summary>
/// <remarks>
/// <para>
/// This class stores information about the detected hardware capabilities that
/// can be used for optimization.
/// </para>
/// <para><b>For Beginners:</b> This stores information about what your hardware can do.
/// 
/// It includes details like:
/// - What CPU features are available
/// - Whether a GPU is present and what type
/// - Whether special accelerators are available
/// - What memory architecture is being used
/// 
/// This information helps the optimizer choose the right optimizations.
/// </para>
/// </remarks>
public class HardwareCapabilities
{
    /// <summary>
    /// Gets or sets a value indicating whether SIMD instructions are supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SIMD (Single Instruction, Multiple Data) allows performing the same operation
    /// on multiple data elements simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> SIMD lets the CPU process multiple numbers at once.
    /// 
    /// Instead of adding numbers one at a time, SIMD can add 4, 8, or 16 numbers
    /// in a single operation, which can greatly speed up mathematical operations.
    /// </para>
    /// </remarks>
    public bool HasSIMD { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether AVX2 instructions are supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// AVX2 (Advanced Vector Extensions 2) is a SIMD instruction set that provides
    /// 256-bit vector operations.
    /// </para>
    /// <para><b>For Beginners:</b> AVX2 is a specific set of instructions for faster math.
    /// 
    /// It allows processing even more numbers at once than basic SIMD,
    /// and is particularly useful for neural network computations.
    /// </para>
    /// </remarks>
    public bool HasAvx2 { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether int8 operations are accelerated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Many modern CPUs have special instructions for accelerating 8-bit integer operations,
    /// which are commonly used in quantized neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates if your CPU can process 8-bit numbers quickly.
    /// 
    /// Quantized models use 8-bit integers instead of 32-bit floating point,
    /// and some CPUs have special instructions to process these faster.
    /// </para>
    /// </remarks>
    public bool HasInt8Acceleration { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether sparse matrix operations are accelerated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some hardware has special support for accelerating operations on sparse matrices
    /// (matrices with many zero values), which is useful for pruned models.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates if your hardware can handle sparse matrices efficiently.
    /// 
    /// Pruned models have many zero weights, and this special acceleration can
    /// skip computations involving these zeros, making inference faster.
    /// </para>
    /// </remarks>
    public bool HasSparseAcceleration { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether a GPU is available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Graphics Processing Units (GPUs) can significantly accelerate neural network inference.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates if a graphics card is available for acceleration.
    /// 
    /// GPUs have many small processing cores that can perform many calculations in parallel,
    /// which is ideal for neural network operations.
    /// </para>
    /// </remarks>
    public bool HasGPU { get; set; }
    
    /// <summary>
    /// Gets or sets the type of GPU available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different GPU types (NVIDIA, AMD, Intel, etc.) have different capabilities and
    /// require different optimization strategies.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you what brand or type of GPU is available.
    /// 
    /// Different GPU brands have different capabilities and programming interfaces,
    /// so the optimizer needs to know which type it's working with.
    /// </para>
    /// </remarks>
    public GPUType GPUType { get; set; } = GPUType.None;
    
    /// <summary>
    /// Gets or sets a value indicating whether a Neural Processing Unit is available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Neural Processing Units (NPUs) are specialized hardware accelerators designed
    /// specifically for neural network operations.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates if a special AI chip is available.
    /// 
    /// NPUs are dedicated chips designed specifically for AI workloads, like Apple's Neural Engine
    /// or Google's Edge TPU, and can be much more efficient than CPUs or GPUs for certain models.
    /// </para>
    /// </remarks>
    public bool HasNPU { get; set; }
    
    /// <summary>
    /// Gets or sets the type of NPU available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different NPU types have different capabilities and require different optimization approaches.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you what kind of AI chip is available.
    /// 
    /// Different NPUs support different operations and have different programming interfaces,
    /// so the optimizer needs to know which type it's working with.
    /// </para>
    /// </remarks>
    public NPUType NPUType { get; set; } = NPUType.None;
}