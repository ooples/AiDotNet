namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Mock GPU runtime for testing without actual GPU hardware.
/// </summary>
/// <remarks>
/// <para>
/// This implementation simulates GPU operations on the CPU for testing purposes.
/// It allows the JIT compiler to be tested without requiring actual GPU hardware.
/// </para>
/// </remarks>
public class MockGPURuntime : IGPURuntime
{
    private readonly GPUCodeGenerator.GPUDeviceInfo _deviceInfo;
    private bool _disposed;

    /// <summary>
    /// Initializes a new mock GPU runtime.
    /// </summary>
    public MockGPURuntime()
    {
        _deviceInfo = new GPUCodeGenerator.GPUDeviceInfo
        {
            DeviceName = "Mock GPU (CPU Simulation)",
            MaxThreadsPerBlock = 1024,
            MaxSharedMemoryPerBlock = 49152,
            MultiprocessorCount = 1,
            WarpSize = 32,
            ComputeCapability = "Mock",
            GlobalMemory = 8L * 1024 * 1024 * 1024,
            HasTensorCores = false
        };
    }

    /// <inheritdoc/>
    public GPUCodeGenerator.GPUDeviceInfo DeviceInfo => _deviceInfo;

    /// <inheritdoc/>
    public IGPUKernelHandle CompileKernel(string sourceCode, string kernelName)
    {
        // In mock mode, we just store the source code
        return new MockKernelHandle(kernelName, sourceCode);
    }

    /// <inheritdoc/>
    public IGPUMemoryHandle Allocate(long sizeBytes)
    {
        // Allocate on CPU heap
        return new MockMemoryHandle(sizeBytes);
    }

    /// <inheritdoc/>
    public void CopyToDevice<T>(IGPUMemoryHandle destination, T[] source)
    {
        if (destination is MockMemoryHandle mock)
        {
            // Use INumericOperations for type-safe conversion
            var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            int elementSize = GetElementSize<T>();
            var bytes = new byte[source.Length * elementSize];

            // Convert each element to bytes
            for (int i = 0; i < source.Length; i++)
            {
                double value = numOps.ToDouble(source[i]);
                byte[] elementBytes = typeof(T) == typeof(float)
                    ? BitConverter.GetBytes((float)value)
                    : typeof(T) == typeof(double)
                        ? BitConverter.GetBytes(value)
                        : typeof(T) == typeof(int)
                            ? BitConverter.GetBytes((int)value)
                            : BitConverter.GetBytes(value);
                Array.Copy(elementBytes, 0, bytes, i * elementSize, Math.Min(elementSize, elementBytes.Length));
            }
            mock.Data = bytes;
        }
    }

    /// <inheritdoc/>
    public void CopyFromDevice<T>(T[] destination, IGPUMemoryHandle source)
    {
        if (source is MockMemoryHandle mock && mock.Data != null)
        {
            var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            int elementSize = GetElementSize<T>();
            int count = Math.Min(destination.Length, mock.Data.Length / elementSize);

            for (int i = 0; i < count; i++)
            {
                double value = typeof(T) == typeof(float)
                    ? BitConverter.ToSingle(mock.Data.ToArray(), i * elementSize)
                    : typeof(T) == typeof(double)
                        ? BitConverter.ToDouble(mock.Data.ToArray(), i * elementSize)
                        : typeof(T) == typeof(int)
                            ? BitConverter.ToInt32(mock.Data.ToArray(), i * elementSize)
                            : BitConverter.ToDouble(mock.Data.ToArray(), i * elementSize);
                destination[i] = numOps.FromDouble(value);
            }
        }
    }

    private static int GetElementSize<T>()
    {
        if (typeof(T) == typeof(float)) return sizeof(float);
        if (typeof(T) == typeof(double)) return sizeof(double);
        if (typeof(T) == typeof(int)) return sizeof(int);
        if (typeof(T) == typeof(long)) return sizeof(long);
        if (typeof(T) == typeof(byte)) return sizeof(byte);
        if (typeof(T) == typeof(short)) return sizeof(short);
        return sizeof(double); // Default fallback
    }

    /// <inheritdoc/>
    public void LaunchKernel(IGPUKernelHandle kernel, int[] gridSize, int[] blockSize, int sharedMemorySize, params object[] arguments)
    {
        // In mock mode, we would interpret the kernel
        // For now, this is a no-op - actual execution would require a kernel interpreter
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        // No-op in mock mode
    }

    /// <inheritdoc/>
    public void Free(IGPUMemoryHandle memory)
    {
        memory.Dispose();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }

    private class MockKernelHandle : IGPUKernelHandle
    {
        public string Name { get; }
        public string SourceCode { get; }
        public bool IsValid => true;

        public MockKernelHandle(string name, string sourceCode)
        {
            Name = name;
            SourceCode = sourceCode;
        }

        public void Dispose() { }
    }

    private class MockMemoryHandle : IGPUMemoryHandle
    {
        public long SizeBytes { get; }
        public bool IsAllocated { get; private set; } = true;
        public byte[]? Data { get; set; }

        public MockMemoryHandle(long sizeBytes)
        {
            SizeBytes = sizeBytes;
            Data = new byte[sizeBytes];
        }

        public void Dispose()
        {
            IsAllocated = false;
            Data = null;
        }
    }
}
