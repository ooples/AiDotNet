namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides configuration settings for CPU parallel operations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This static class holds settings that control how
/// operations are parallelized across CPU cores. The default values are
/// optimized for most modern systems.
/// </remarks>
public static class CpuParallelSettings
{
    /// <summary>
    /// Gets or sets the maximum degree of parallelism for CPU operations.
    /// </summary>
    /// <remarks>
    /// Default is Environment.ProcessorCount, which uses all available cores.
    /// Set to 1 to disable parallelism.
    /// </remarks>
    public static int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Gets or sets whether SIMD (Single Instruction, Multiple Data) operations are enabled.
    /// </summary>
    /// <remarks>
    /// SIMD allows processing multiple data elements with a single instruction,
    /// significantly speeding up vector and matrix operations on supported hardware.
    /// </remarks>
    public static bool EnableSimd { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum array length before parallelization is applied.
    /// </summary>
    /// <remarks>
    /// For small arrays, the overhead of parallelization may exceed the benefits.
    /// Operations on arrays smaller than this threshold will run sequentially.
    /// </remarks>
    public static int ParallelThreshold { get; set; } = 65536;

    /// <summary>
    /// The minimum chunk size for parallel operations.
    /// </summary>
    /// <remarks>
    /// This ensures each parallel task processes at least this many elements
    /// to avoid excessive task creation overhead.
    /// </remarks>
    public const int MinChunkSize = 4096;

    /// <summary>
    /// Executes a parallel for loop with chunked iterations.
    /// </summary>
    /// <param name="length">Total number of elements to process.</param>
    /// <param name="minChunkSize">Minimum elements per chunk.</param>
    /// <param name="action">Action to execute for each chunk (start index, count).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method divides work into chunks and processes
    /// them in parallel across available CPU cores for better performance.
    /// </remarks>
    public static void ParallelForChunks(int length, int minChunkSize, Action<int, int> action)
    {
        if (length <= 0)
            return;

        int maxThreads = MaxDegreeOfParallelism;
        if (maxThreads <= 1 || length < minChunkSize)
        {
            // Sequential execution
            action(0, length);
            return;
        }

        // Calculate chunk size - ensure at least minChunkSize per thread
        int chunkSize = Math.Max(minChunkSize, (length + maxThreads - 1) / maxThreads);
        int numChunks = (length + chunkSize - 1) / chunkSize;

        Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = maxThreads }, chunkIndex =>
        {
            int start = chunkIndex * chunkSize;
            int count = Math.Min(chunkSize, length - start);
            action(start, count);
        });
    }
}
