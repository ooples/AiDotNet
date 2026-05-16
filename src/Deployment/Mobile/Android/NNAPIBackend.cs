using System.Runtime.InteropServices;
using AiDotNet.Deployment.Export;
using AiDotNet.Validation;

namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// NNAPI (Neural Networks API) backend for Android deployment.
/// Provides hardware acceleration on Android devices when libneuralnetworks.so
/// is loadable; falls back to managed CPU execution otherwise.
/// </summary>
/// <typeparam name="T">The numeric type for input/output tensors.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> NNAPI is the Android-side API that exposes hardware
/// accelerators (GPU / DSP / NPU) for neural network inference. This backend
/// binds to the native <c>libneuralnetworks.so</c> via P/Invoke; if the library
/// is not present, the backend reports unavailability and routes execution
/// through a managed CPU fallback when one has been configured.
/// </para>
/// <para>
/// Default values follow the Android NNAPI guidance: relaxed FP32, optional
/// FP16, and CPU fallback support. The <see cref="CpuExecutor"/> hook lets
/// callers plug in a real managed inference path for fallback execution.
/// </para>
/// </remarks>
public class NNAPIBackend<T> : IDisposable
{
    private static readonly OSPlatform AndroidPlatform = OSPlatform.Create("ANDROID");

    private readonly NNAPIConfiguration _config;
    private bool _isInitialized;
    private bool _disposed;

    private IntPtr _model = IntPtr.Zero;
    private IntPtr _compilation = IntPtr.Zero;
    private bool _hasNative;
    private byte[]? _modelBytes;

    /// <summary>
    /// Output element count set by an upstream model-format adapter.
    /// </summary>
    internal int OutputElementCount { get; set; }

    /// <summary>
    /// Managed-CPU inference hook invoked when NNAPI is unavailable or no
    /// compiled native graph is available.
    /// </summary>
    internal Func<T[], T[]>? CpuExecutor { get; set; }

    /// <summary>
    /// Optional graph builder that decodes loaded model bytes into NNAPI
    /// operands and operations before compilation.
    /// </summary>
    internal INNAPIGraphBuilder? GraphBuilder { get; set; }

    public NNAPIBackend(NNAPIConfiguration config, INNAPIGraphBuilder? graphBuilder = null)
    {
        Guard.NotNull(config);
        ValidateElementType();
        _config = config;
        GraphBuilder = graphBuilder;
    }

    /// <summary>
    /// Initializes the NNAPI backend.
    /// </summary>
    public void Initialize()
    {
        ThrowIfDisposed();
        if (_isInitialized) return;

        _hasNative = IsNNAPIAvailable();
        if (!_hasNative && !_config.AllowCpuFallback)
        {
            throw new PlatformNotSupportedException(
                "NNAPI native binding (libneuralnetworks.so) is not loadable on this host, " +
                "and NNAPIConfiguration.AllowCpuFallback is false. Set AllowCpuFallback=true or " +
                "run on Android API 27+ with NNAPI present.");
        }

        if (_hasNative)
        {
            ReleaseNativeHandles();
            int rc = NNAPIInterop.ModelCreate(out _model);
            if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                throw new InvalidOperationException($"ANeuralNetworksModel_create failed (rc={rc}).");
        }

        _isInitialized = true;
    }

    /// <summary>
    /// Loads a model for NNAPI execution.
    /// </summary>
    public void LoadModel(string modelPath)
    {
        ThrowIfDisposed();
        if (!_isInitialized)
            throw new InvalidOperationException("NNAPI backend not initialized. Call Initialize() first.");
        Guard.NotNull(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _modelBytes = File.ReadAllBytes(modelPath);
        CompileForNNAPI(_modelBytes);
    }

    /// <summary>
    /// Executes inference using NNAPI or the managed CPU fallback.
    /// </summary>
    public T[] Execute(T[] input)
    {
        ThrowIfDisposed();
        if (!_isInitialized)
            throw new InvalidOperationException("NNAPI backend not initialized. Call Initialize() first.");
        Guard.NotNull(input);
        return ExecuteOnNNAPI(input);
    }

    /// <summary>Executes inference asynchronously.</summary>
    public async Task<T[]> ExecuteAsync(T[] input)
    {
        Guard.NotNull(input);
        return await Task.Run(() => Execute(input));
    }

    /// <summary>Gets the supported acceleration devices on this Android device.</summary>
    public List<string> GetSupportedDevices()
    {
        ThrowIfDisposed();
        if (IsNNAPIAvailable())
        {
            var names = new List<string>();
            if (NNAPIInterop.GetDeviceCount(out uint count) == NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
            {
                for (uint i = 0; i < count; i++)
                {
                    if (NNAPIInterop.GetDevice(i, out IntPtr device) != NNAPIInterop.ANEURALNETWORKS_NO_ERROR) continue;
                    if (NNAPIInterop.GetDeviceName(device, out IntPtr namePtr) != NNAPIInterop.ANEURALNETWORKS_NO_ERROR) continue;
                    var name = Marshal.PtrToStringAnsi(namePtr);
                    if (!string.IsNullOrEmpty(name)) names.Add(name);
                }
            }
            return names.Count > 0 ? names : ["nnapi-cpu"];
        }

        return ["cpu"];
    }

    /// <summary>
    /// Checks if NNAPI is available on the current device.
    /// </summary>
    public static bool IsNNAPIAvailable()
    {
        if (!RuntimeInformation.IsOSPlatform(AndroidPlatform)
            && !RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return false;
        }

        return NNAPIInterop.TryLoad();
    }

    /// <summary>Gets NNAPI performance information for the current device.</summary>
    public NNAPIPerformanceInfo GetPerformanceInfo()
    {
        ThrowIfDisposed();
        return new NNAPIPerformanceInfo
        {
            SupportedOperations = GetSupportedOperations(),
            PreferredDevice = _config.PreferredDevice.ToString(),
            SupportsInt8 = true,
            SupportsFp16 = _config.AllowFp16,
            SupportsRelaxedFp32 = _config.UseRelaxedFloat32
        };
    }

    private int MapPreference()
    {
        return _config.ExecutionPreference switch
        {
            NNAPIExecutionPreference.SustainedSpeed => NNAPIInterop.ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            NNAPIExecutionPreference.LowPower => NNAPIInterop.ANEURALNETWORKS_PREFER_LOW_POWER,
            NNAPIExecutionPreference.FastSingleAnswer => NNAPIInterop.ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
            _ => NNAPIInterop.ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER
        };
    }

    private void CompileForNNAPI(byte[] modelData)
    {
        Guard.NotNull(modelData);
        if (!_hasNative || _model == IntPtr.Zero) return;

        if (GraphBuilder is not null)
        {
            bool built = GraphBuilder.BuildGraph(_model, modelData);
            if (!built)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "NNAPIBackend.CompileForNNAPI: configured INNAPIGraphBuilder returned false. " +
                    "Execute() will use CpuExecutor when configured.");
                return;
            }
        }

        int rcFinish = NNAPIInterop.ModelFinish(_model);
        if (rcFinish != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"NNAPIBackend.CompileForNNAPI: ANeuralNetworksModel_finish failed (rc={rcFinish}). " +
                "No NNAPI op graph was compiled; Execute() will use CpuExecutor when configured.");
            return;
        }

        ReleaseCompilationHandle();

        int rcCompile = NNAPIInterop.CompilationCreate(_model, out _compilation);
        if (rcCompile != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
        {
            _compilation = IntPtr.Zero;
            throw new InvalidOperationException($"ANeuralNetworksCompilation_create failed (rc={rcCompile}).");
        }

        int rcPref = NNAPIInterop.CompilationSetPreference(_compilation, MapPreference());
        if (rcPref != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
        {
            ReleaseCompilationHandle();
            throw new InvalidOperationException(
                $"ANeuralNetworksCompilation_setPreference failed (rc={rcPref}).");
        }

        int rcFin = NNAPIInterop.CompilationFinish(_compilation);
        if (rcFin != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
        {
            ReleaseCompilationHandle();
            throw new InvalidOperationException($"ANeuralNetworksCompilation_finish failed (rc={rcFin}).");
        }
    }

    private T[] ExecuteOnNNAPI(T[] input)
    {
        if (_hasNative && _compilation != IntPtr.Zero)
        {
            return ExecuteNativeNNAPI(input);
        }

        var executor = CpuExecutor;
        if (executor != null) return executor(input);

        throw new InvalidOperationException(
            "NNAPIBackend.Execute cannot run: no compiled NNAPI graph is available and CpuExecutor is not configured.");
    }

    private T[] ExecuteNativeNNAPI(T[] input)
    {
        int rc = NNAPIInterop.ExecutionCreate(_compilation, out IntPtr execution);
        if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
            throw new InvalidOperationException($"ANeuralNetworksExecution_create failed (rc={rc}).");

        try
        {
            int outElems = OutputElementCount > 0 ? OutputElementCount : input.Length;
            int elemSize = ElementSize();
            int inBytes = input.Length * elemSize;
            int outBytes = outElems * elemSize;
            IntPtr inputPtr = Marshal.AllocHGlobal(inBytes);
            IntPtr outputPtr = Marshal.AllocHGlobal(outBytes);

            try
            {
                CopyManagedToNative(input, inputPtr);

                rc = NNAPIInterop.ExecutionSetInput(execution, 0, IntPtr.Zero, inputPtr, (UIntPtr)(uint)inBytes);
                if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                    throw new InvalidOperationException($"ExecutionSetInput failed (rc={rc}).");

                rc = NNAPIInterop.ExecutionSetOutput(execution, 0, IntPtr.Zero, outputPtr, (UIntPtr)(uint)outBytes);
                if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                    throw new InvalidOperationException($"ExecutionSetOutput failed (rc={rc}).");

                rc = NNAPIInterop.ExecutionStartCompute(execution, out IntPtr evt);
                if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                    throw new InvalidOperationException($"ExecutionStartCompute failed (rc={rc}).");

                try
                {
                    rc = NNAPIInterop.EventWait(evt);
                    if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                        throw new InvalidOperationException($"EventWait failed (rc={rc}).");
                }
                finally
                {
                    NNAPIInterop.EventFree(evt);
                }

                var output = new T[outElems];
                CopyNativeToManaged(outputPtr, output);
                return output;
            }
            finally
            {
                Marshal.FreeHGlobal(inputPtr);
                Marshal.FreeHGlobal(outputPtr);
            }
        }
        finally
        {
            NNAPIInterop.ExecutionFree(execution);
        }
    }

    private static int ElementSize()
    {
        if (typeof(T) == typeof(float)) return sizeof(float);
        if (typeof(T) == typeof(int)) return sizeof(int);
        if (typeof(T) == typeof(byte)) return sizeof(byte);
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half)) return 2;
#endif
        throw new NotSupportedException(UnsupportedElementTypeMessage());
    }

    private static void CopyManagedToNative(T[] src, IntPtr dst)
    {
        if (typeof(T) == typeof(float)) { Marshal.Copy((float[])(object)src, 0, dst, src.Length); return; }
        if (typeof(T) == typeof(int)) { Marshal.Copy((int[])(object)src, 0, dst, src.Length); return; }
        if (typeof(T) == typeof(byte)) { Marshal.Copy((byte[])(object)src, 0, dst, src.Length); return; }
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
        {
            var halfSrc = (Half[])(object)src;
            var bytes = MemoryMarshal.AsBytes(halfSrc.AsSpan()).ToArray();
            Marshal.Copy(bytes, 0, dst, bytes.Length);
            return;
        }
#endif
        throw new NotSupportedException(UnsupportedElementTypeMessage());
    }

    private static void CopyNativeToManaged(IntPtr src, T[] dst)
    {
        if (typeof(T) == typeof(float)) { Marshal.Copy(src, (float[])(object)dst, 0, dst.Length); return; }
        if (typeof(T) == typeof(int)) { Marshal.Copy(src, (int[])(object)dst, 0, dst.Length); return; }
        if (typeof(T) == typeof(byte)) { Marshal.Copy(src, (byte[])(object)dst, 0, dst.Length); return; }
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
        {
            var halfDst = (Half[])(object)dst;
            var bytes = new byte[halfDst.Length * 2];
            Marshal.Copy(src, bytes, 0, bytes.Length);
            MemoryMarshal.Cast<byte, Half>(bytes.AsSpan()).CopyTo(halfDst.AsSpan());
            return;
        }
#endif
        throw new NotSupportedException(UnsupportedElementTypeMessage());
    }

    private static void ValidateElementType()
    {
        if (typeof(T) == typeof(float) ||
            typeof(T) == typeof(int) ||
            typeof(T) == typeof(byte))
        {
            return;
        }

#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half)) return;
#endif

        throw new NotSupportedException(UnsupportedElementTypeMessage());
    }

    private static string UnsupportedElementTypeMessage() =>
        "NNAPI element type " + typeof(T).Name + " is not supported. Supported types: float, int, byte" +
#if NET5_0_OR_GREATER
        ", Half" +
#endif
        ".";

    public void Dispose()
    {
        if (_disposed) return;
        ReleaseNativeHandles();
        _hasNative = false;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~NNAPIBackend()
    {
        ReleaseNativeHandles();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(NNAPIBackend<T>));
    }

    private void ReleaseNativeHandles()
    {
        ReleaseCompilationHandle();
        ReleaseModelHandle();
    }

    private void ReleaseCompilationHandle()
    {
        if (_compilation == IntPtr.Zero) return;
        NNAPIInterop.CompilationFree(_compilation);
        _compilation = IntPtr.Zero;
    }

    private void ReleaseModelHandle()
    {
        if (_model == IntPtr.Zero) return;
        NNAPIInterop.ModelFree(_model);
        _model = IntPtr.Zero;
    }

    private List<string> GetSupportedOperations()
    {
        return
        [
            "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
            "MAX_POOL_2D", "AVERAGE_POOL_2D", "SOFTMAX",
            "RELU", "RELU6", "ADD", "MUL"
        ];
    }
}
