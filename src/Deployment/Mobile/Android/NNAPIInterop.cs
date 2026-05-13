using System.Runtime.InteropServices;

namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// P/Invoke surface for the Android NNAPI C ABI (<c>libneuralnetworks.so</c>).
/// </summary>
/// <remarks>
/// <para>
/// Binds the subset of the NNAPI C API needed to compile and execute an inference
/// graph. Function signatures mirror the official <c>NeuralNetworks.h</c> header
/// shipped with the NDK (API levels 27+). DllImport uses the standard SO name
/// (<c>libneuralnetworks.so</c>); when this is a managed-only / non-Android host
/// (Windows, macOS, desktop Linux), every P/Invoke will throw DllNotFoundException
/// on first call. Callers detect availability via <see cref="NNAPIBackend{T}.IsNNAPIAvailable"/>
/// before invoking and fall back to CPU otherwise.
/// </para>
/// <para>
/// Constants reproduce the relevant entries from
/// <c>NeuralNetworksTypes.h</c> — only those used in the backend implementation
/// are declared, to keep the binding surface small.
/// </para>
/// </remarks>
internal static class NNAPIInterop
{
    private const string LibName = "libneuralnetworks";

    // ANeuralNetworksResultCode — relevant subset.
    public const int ANEURALNETWORKS_NO_ERROR = 0;
    public const int ANEURALNETWORKS_OUT_OF_MEMORY = 1;
    public const int ANEURALNETWORKS_INCOMPLETE = 2;
    public const int ANEURALNETWORKS_UNEXPECTED_NULL = 3;
    public const int ANEURALNETWORKS_BAD_DATA = 4;
    public const int ANEURALNETWORKS_OP_FAILED = 5;
    public const int ANEURALNETWORKS_UNMAPPABLE = 7;
    public const int ANEURALNETWORKS_BAD_STATE = 6;
    public const int ANEURALNETWORKS_UNAVAILABLE_DEVICE = 9;

    // ANeuralNetworksOperandCode — relevant subset.
    public const int ANEURALNETWORKS_FLOAT32 = 0;
    public const int ANEURALNETWORKS_INT32 = 1;
    public const int ANEURALNETWORKS_UINT32 = 2;
    public const int ANEURALNETWORKS_TENSOR_FLOAT32 = 3;
    public const int ANEURALNETWORKS_TENSOR_INT32 = 4;
    public const int ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5;
    public const int ANEURALNETWORKS_TENSOR_FLOAT16 = 11;

    // ANeuralNetworksPreference.
    public const int ANEURALNETWORKS_PREFER_LOW_POWER = 0;
    public const int ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1;
    public const int ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2;

    /// <summary>
    /// Probes the loader for libneuralnetworks on the current platform.
    /// Returns true if the library is present and loadable, false otherwise —
    /// the C# P/Invoke layer does not throw on lazy bind, only on first call,
    /// so we proactively probe with <see cref="NativeLibrary.TryLoad(string, out IntPtr)"/>.
    /// Captures and immediately frees the probe handle so repeated
    /// IsNNAPIAvailable() calls don't leak a native reference per
    /// invocation. Uses <see cref="LibName"/> directly (matching the
    /// DllImport name) so the load path is consistent between the probe
    /// and the actual P/Invoke calls.
    /// </summary>
    public static bool TryLoad()
    {
#if NET5_0_OR_GREATER
        try
        {
            if (!NativeLibrary.TryLoad(LibName, out IntPtr handle))
            {
                return false;
            }
            NativeLibrary.Free(handle);
            return true;
        }
        catch
        {
            return false;
        }
#else
        // System.Runtime.InteropServices.NativeLibrary is a .NET 5+ API
        // and isn't available on net471. NNAPI itself is Android-only
        // and the calling Initialize() path also routes through
        // RuntimeInformation.IsOSPlatform(OSPlatform.Linux) — on net471
        // hosts (Windows / desktop only) the probe is unreachable in
        // practice, so returning false here is the correct "not loadable"
        // answer.
        return false;
#endif
    }

    // ─── Device introspection ───────────────────────────────────────────────

    [DllImport(LibName, EntryPoint = "ANeuralNetworks_getDeviceCount")]
    public static extern int GetDeviceCount(out uint deviceCount);

    [DllImport(LibName, EntryPoint = "ANeuralNetworks_getDevice")]
    public static extern int GetDevice(uint index, out IntPtr device);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksDevice_getName")]
    public static extern int GetDeviceName(IntPtr device, out IntPtr name);

    // ─── Model construction ─────────────────────────────────────────────────

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_create")]
    public static extern int ModelCreate(out IntPtr model);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_free")]
    public static extern void ModelFree(IntPtr model);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_addOperand")]
    public static extern int ModelAddOperand(IntPtr model, ref ANeuralNetworksOperandType type);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_setOperandValue")]
    public static extern int ModelSetOperandValue(IntPtr model, int index, IntPtr buffer, UIntPtr length);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_addOperation")]
    public static extern int ModelAddOperation(
        IntPtr model, int type,
        uint inputCount, uint[] inputs,
        uint outputCount, uint[] outputs);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_identifyInputsAndOutputs")]
    public static extern int ModelIdentifyInputsAndOutputs(
        IntPtr model,
        uint inputCount, uint[] inputs,
        uint outputCount, uint[] outputs);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksModel_finish")]
    public static extern int ModelFinish(IntPtr model);

    // ─── Compilation ────────────────────────────────────────────────────────

    [DllImport(LibName, EntryPoint = "ANeuralNetworksCompilation_create")]
    public static extern int CompilationCreate(IntPtr model, out IntPtr compilation);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksCompilation_createForDevices")]
    public static extern int CompilationCreateForDevices(
        IntPtr model, IntPtr[] devices, uint numDevices, out IntPtr compilation);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksCompilation_free")]
    public static extern void CompilationFree(IntPtr compilation);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksCompilation_setPreference")]
    public static extern int CompilationSetPreference(IntPtr compilation, int preference);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksCompilation_finish")]
    public static extern int CompilationFinish(IntPtr compilation);

    // ─── Execution ──────────────────────────────────────────────────────────

    [DllImport(LibName, EntryPoint = "ANeuralNetworksExecution_create")]
    public static extern int ExecutionCreate(IntPtr compilation, out IntPtr execution);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksExecution_free")]
    public static extern void ExecutionFree(IntPtr execution);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksExecution_setInput")]
    public static extern int ExecutionSetInput(
        IntPtr execution, int index, IntPtr typeOrNull, IntPtr buffer, UIntPtr length);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksExecution_setOutput")]
    public static extern int ExecutionSetOutput(
        IntPtr execution, int index, IntPtr typeOrNull, IntPtr buffer, UIntPtr length);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksExecution_startCompute")]
    public static extern int ExecutionStartCompute(IntPtr execution, out IntPtr eventHandle);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksEvent_wait")]
    public static extern int EventWait(IntPtr eventHandle);

    [DllImport(LibName, EntryPoint = "ANeuralNetworksEvent_free")]
    public static extern void EventFree(IntPtr eventHandle);

    // ─── Operand type struct ────────────────────────────────────────────────

    /// <summary>
    /// Mirrors <c>ANeuralNetworksOperandType</c> from <c>NeuralNetworksTypes.h</c>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ANeuralNetworksOperandType
    {
        public int type;
        public uint dimensionCount;
        public IntPtr dimensions; // const uint32_t*
        public float scale;
        public int zeroPoint;
    }
}
