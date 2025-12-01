namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Handle to a compiled GPU kernel.
/// </summary>
/// <remarks>
/// <para>
/// Represents a compiled GPU kernel that is ready for execution. The handle
/// encapsulates the compiled binary code and provides information about the kernel.
/// </para>
/// </remarks>
public interface IGPUKernelHandle : IDisposable
{
    /// <summary>
    /// Gets the kernel name.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether the kernel is valid and ready for execution.
    /// </summary>
    bool IsValid { get; }
}
