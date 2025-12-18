namespace AiDotNet.Data.Structures;

/// <summary>
/// Methods for initializing external memory in Neural Turing Machines.
/// </summary>
/// <remarks>
/// <para>
/// Memory initialization strategies determine the starting state of the
/// external memory matrix, which can affect training dynamics and performance.
/// </para>
/// </remarks>
public enum MemoryInitialization
{
    /// <summary>
    /// Initialize memory with all zeros.
    /// </summary>
    Zeros,

    /// <summary>
    /// Initialize memory with small random values.
    /// </summary>
    Random,

    /// <summary>
    /// Initialize memory with learned embeddings.
    /// </summary>
    Learned
}