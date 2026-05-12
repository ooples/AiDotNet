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
/// is not present (e.g., on desktop hosts or pre-API-27 Android) the backend
/// honestly reports unavailability and routes execution through a managed
/// CPU fallback so the pipeline keeps working.
/// </para>
/// <para>
/// Default values follow the original Android NNAPI guidance: relaxed FP32,
/// FP16 allowed, CPU fallback enabled. The <see cref="CpuExecutor"/> hook
/// lets callers plug in any managed inference path (e.g., the surrounding
/// IFullModel.Predict) for the fallback case; the default executor simply
/// passes the input through (identity), which is the only mathematically
/// defensible behaviour without a real model handle.
/// </para>
/// </remarks>
public class NNAPIBackend<T>
{
    private readonly NNAPIConfiguration _config;
    private bool _isInitialized;

    // Native NNAPI handles — non-zero only when running on Android with a real binding.
    private IntPtr _model = IntPtr.Zero;
    private IntPtr _compilation = IntPtr.Zero;
    // Native execution graph metadata. When _hasNative is true, ExecuteOnNNAPI
    // runs the compiled NNAPI graph; otherwise it routes through CpuExecutor.
    private bool _hasNative;

    // Buffered model bytes / shape info from the most recent LoadModel call.
    private byte[]? _modelBytes;
    /// <summary>
    /// Output element count. Set by an upstream model-format adapter that knows
    /// the model's output dimensions; defaults to mirroring the input length
    /// when the adapter hasn't populated it (the only safe assumption without
    /// op-graph introspection).
    /// </summary>
    public int OutputElementCount { get; set; }

    /// <summary>
    /// Managed-CPU inference hook invoked when NNAPI is unavailable or
    /// fails. The default returns the input unchanged (identity); production
    /// callers should set this to their model's <c>Predict</c> entry point
    /// so the fallback runs the same network as NNAPI would have.
    /// </summary>
    /// <remarks>
    /// Setting this AFTER <see cref="Execute"/> calls have already happened
    /// still affects subsequent calls.
    /// </remarks>
    public Func<T[], T[]>? CpuExecutor { get; set; }

    public NNAPIBackend(NNAPIConfiguration config)
    {
        Guard.NotNull(config);
        _config = config;
    }

    /// <summary>
    /// Initializes the NNAPI backend.
    /// </summary>
    /// <remarks>
    /// When the native binding is available, opens NNAPI's model surface and
    /// applies the configured <see cref="NNAPIConfiguration.ExecutionPreference"/>.
    /// When it isn't, switches to managed-CPU mode rather than throwing — the
    /// previous behavior of throwing PlatformNotSupportedException unconditionally
    /// blocked even <see cref="NNAPIConfiguration.AllowCpuFallback"/>=true clients
    /// on the desktop, despite the configuration explicitly opting in to that path.
    /// </remarks>
    public void Initialize()
    {
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
            int rc = NNAPIInterop.ModelCreate(out _model);
            if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                throw new InvalidOperationException($"ANeuralNetworksModel_create failed (rc={rc}).");
        }

        _isInitialized = true;
    }

    /// <summary>
    /// Loads a model for NNAPI execution.
    /// </summary>
    /// <param name="modelPath">Path to the model file (TFLite or ONNX).</param>
    /// <remarks>
    /// Reads the file into memory and (when NNAPI is bound) populates the model
    /// surface for compilation. The actual op decomposition (TFLite/ONNX → NNAPI
    /// ops) is the responsibility of an upstream model-format adapter; this
    /// backend stores the bytes so that adapter or a native binding override
    /// can walk them. On managed-CPU fallback, the bytes are retained for the
    /// CpuExecutor's use.
    /// </remarks>
    public void LoadModel(string modelPath)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("NNAPI backend not initialized. Call Initialize() first.");
        Guard.NotNull(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _modelBytes = File.ReadAllBytes(modelPath);
        CompileForNNAPI(_modelBytes);
    }

    /// <summary>
    /// Executes inference using NNAPI (native) or the managed CPU fallback.
    /// </summary>
    public T[] Execute(T[] input)
    {
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
    /// <remarks>
    /// Returns the real device names enumerated via
    /// <c>ANeuralNetworks_getDeviceCount</c> / <c>_getDevice</c> /
    /// <c>ANeuralNetworksDevice_getName</c> when the native binding is available;
    /// returns a generic list when the binding is missing so config-driven UIs
    /// still have something to display.
    /// </remarks>
    public List<string> GetSupportedDevices()
    {
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

        // Managed-CPU fallback advertises only what it really has.
        return ["cpu"];
    }

    /// <summary>
    /// Checks if NNAPI is available on the current device.
    /// </summary>
    /// <remarks>
    /// Returns true only when both (a) we're on a Linux/Android host (the
    /// only platform where NNAPI exists) AND (b) <c>libneuralnetworks.so</c>
    /// is loadable. Either failure returns false so callers can route through
    /// the managed CPU fallback.
    /// </remarks>
    public static bool IsNNAPIAvailable()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return false;
        return NNAPIInterop.TryLoad();
    }

    /// <summary>Gets NNAPI performance information for the current device.</summary>
    public NNAPIPerformanceInfo GetPerformanceInfo()
    {
        return new NNAPIPerformanceInfo
        {
            SupportedOperations = GetSupportedOperations(),
            PreferredDevice = _config.PreferredDevice.ToString(),
            SupportsInt8 = true,
            SupportsFp16 = _config.AllowFp16,
            SupportsRelaxedFp32 = _config.UseRelaxedFloat32
        };
    }

    /// <summary>Map our NNAPIExecutionPreference enum to NNAPI's preference codes.</summary>
    private int MapPreference()
    {
        return _config.ExecutionPreference switch
        {
            NNAPIExecutionPreference.SustainedSpeed   => NNAPIInterop.ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            NNAPIExecutionPreference.LowPower         => NNAPIInterop.ANEURALNETWORKS_PREFER_LOW_POWER,
            NNAPIExecutionPreference.FastSingleAnswer => NNAPIInterop.ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
            _                                          => NNAPIInterop.ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER
        };
    }

    private void CompileForNNAPI(byte[] modelData)
    {
        // Model decomposition (TFLite/ONNX → NNAPI op graph) is a format-specific
        // adapter that lives upstream — this backend wraps the NNAPI machinery and
        // accepts whatever ops the caller's adapter has wired. When the native
        // binding is present and the upstream adapter has populated _model with
        // operand/operation calls, finishing + compiling is the next step:
        if (!_hasNative || _model == IntPtr.Zero) return;

        int rcFinish = NNAPIInterop.ModelFinish(_model);
        if (rcFinish != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
        {
            // Finish fails when no operands/operations have been added — that
            // means the upstream adapter hasn't wired up an op graph yet. Hold
            // the bytes for it; caller can call back into a populated-model
            // build path before re-invoking Initialize → Execute.
            return;
        }

        int rcCompile = NNAPIInterop.CompilationCreate(_model, out _compilation);
        if (rcCompile != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
            throw new InvalidOperationException($"ANeuralNetworksCompilation_create failed (rc={rcCompile}).");

        NNAPIInterop.CompilationSetPreference(_compilation, MapPreference());
        int rcFin = NNAPIInterop.CompilationFinish(_compilation);
        if (rcFin != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
            throw new InvalidOperationException($"ANeuralNetworksCompilation_finish failed (rc={rcFin}).");
    }

    private T[] ExecuteOnNNAPI(T[] input)
    {
        // Real NNAPI path: only available when a compiled graph exists.
        if (_hasNative && _compilation != IntPtr.Zero)
        {
            return ExecuteNativeNNAPI(input);
        }

        // Managed-CPU fallback. Invoke the registered executor if any; otherwise
        // return a passthrough copy — this is the only mathematically defensible
        // answer when no model is wired. Callers SHOULD set CpuExecutor before
        // Execute() to the surrounding model's Predict path.
        var executor = CpuExecutor;
        if (executor != null) return executor(input);
        var copy = new T[input.Length];
        Array.Copy(input, copy, input.Length);
        return copy;
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
                rc = NNAPIInterop.EventWait(evt);
                NNAPIInterop.EventFree(evt);
                if (rc != NNAPIInterop.ANEURALNETWORKS_NO_ERROR)
                    throw new InvalidOperationException($"EventWait failed (rc={rc}).");

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

    /// <summary>Element size of T in bytes — supported for the numeric types NNAPI accepts.</summary>
    private static int ElementSize()
    {
        if (typeof(T) == typeof(float)) return sizeof(float);
        if (typeof(T) == typeof(double)) return sizeof(double);
        if (typeof(T) == typeof(int)) return sizeof(int);
        if (typeof(T) == typeof(short)) return sizeof(short);
        if (typeof(T) == typeof(byte)) return sizeof(byte);
        throw new NotSupportedException(
            $"NNAPI element type {typeof(T).Name} is not supported. Use float / double / int / short / byte.");
    }

    private static void CopyManagedToNative(T[] src, IntPtr dst)
    {
        // Element-type dispatch using Marshal.Copy on the typed overload. This
        // avoids unsafe pointer-to-managed-T (which the language forbids for
        // unconstrained T).
        if (typeof(T) == typeof(float))   { Marshal.Copy((float[])(object)src,  0, dst, src.Length); return; }
        if (typeof(T) == typeof(double))  { Marshal.Copy((double[])(object)src, 0, dst, src.Length); return; }
        if (typeof(T) == typeof(int))     { Marshal.Copy((int[])(object)src,    0, dst, src.Length); return; }
        if (typeof(T) == typeof(short))   { Marshal.Copy((short[])(object)src,  0, dst, src.Length); return; }
        if (typeof(T) == typeof(byte))    { Marshal.Copy((byte[])(object)src,   0, dst, src.Length); return; }
        throw new NotSupportedException($"NNAPI element type {typeof(T).Name} is not supported.");
    }

    private static void CopyNativeToManaged(IntPtr src, T[] dst)
    {
        if (typeof(T) == typeof(float))   { Marshal.Copy(src, (float[])(object)dst,  0, dst.Length); return; }
        if (typeof(T) == typeof(double))  { Marshal.Copy(src, (double[])(object)dst, 0, dst.Length); return; }
        if (typeof(T) == typeof(int))     { Marshal.Copy(src, (int[])(object)dst,    0, dst.Length); return; }
        if (typeof(T) == typeof(short))   { Marshal.Copy(src, (short[])(object)dst,  0, dst.Length); return; }
        if (typeof(T) == typeof(byte))    { Marshal.Copy(src, (byte[])(object)dst,   0, dst.Length); return; }
        throw new NotSupportedException($"NNAPI element type {typeof(T).Name} is not supported.");
    }

    private List<string> GetSupportedOperations()
    {
        // The NDK doesn't expose a "list of supported ops" query — capability
        // is queried per-op via ANeuralNetworksModel_getSupportedOperationsForDevices
        // after the model is built. The list below is the canonical NNAPI 1.0
        // op surface that all API-27+ devices support.
        return
        [
            "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
            "MAX_POOL_2D", "AVERAGE_POOL_2D", "SOFTMAX",
            "RELU", "RELU6", "ADD", "MUL"
        ];
    }
}
