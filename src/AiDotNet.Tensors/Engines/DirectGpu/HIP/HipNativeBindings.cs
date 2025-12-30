// Copyright (c) AiDotNet. All rights reserved.
// HIP Runtime P/Invoke bindings for AMD GPU access.
// Provides direct access to HIP for MFMA kernel execution.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP error codes returned by HIP runtime functions.
/// </summary>
public enum HipError
{
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorMemoryAllocation = 2,
    ErrorNotInitialized = 3,
    ErrorInitializationError = 3,
    ErrorDeinitialized = 4,
    ErrorProfilerDisabled = 5,
    ErrorProfilerNotInitialized = 6,
    ErrorProfilerAlreadyStarted = 7,
    ErrorProfilerAlreadyStopped = 8,
    ErrorInvalidConfiguration = 9,
    ErrorInvalidPitchValue = 12,
    ErrorInvalidSymbol = 13,
    ErrorInvalidDevicePointer = 17,
    ErrorInvalidMemcpyDirection = 21,
    ErrorInsufficientDriver = 35,
    ErrorMissingConfiguration = 52,
    ErrorPriorLaunchFailure = 53,
    ErrorInvalidDeviceFunction = 98,
    ErrorNoDevice = 100,
    ErrorInvalidDevice = 101,
    ErrorInvalidImage = 200,
    ErrorInvalidContext = 201,
    ErrorContextAlreadyCurrent = 202,
    ErrorMapFailed = 205,
    ErrorUnmapFailed = 206,
    ErrorArrayIsMapped = 207,
    ErrorAlreadyMapped = 208,
    ErrorNoBinaryForGpu = 209,
    ErrorAlreadyAcquired = 210,
    ErrorNotMapped = 211,
    ErrorNotMappedAsArray = 212,
    ErrorNotMappedAsPointer = 213,
    ErrorECCNotCorrectable = 214,
    ErrorUnsupportedLimit = 215,
    ErrorContextAlreadyInUse = 216,
    ErrorPeerAccessUnsupported = 217,
    ErrorInvalidKernelFile = 218,
    ErrorInvalidGraphicsContext = 219,
    ErrorInvalidSource = 300,
    ErrorFileNotFound = 301,
    ErrorSharedObjectSymbolNotFound = 302,
    ErrorSharedObjectInitFailed = 303,
    ErrorOperatingSystem = 304,
    ErrorInvalidHandle = 400,
    ErrorIllegalState = 401,
    ErrorNotFound = 500,
    ErrorNotReady = 600,
    ErrorIllegalAddress = 700,
    ErrorLaunchOutOfResources = 701,
    ErrorLaunchTimeOut = 702,
    ErrorPeerAccessAlreadyEnabled = 704,
    ErrorPeerAccessNotEnabled = 705,
    ErrorSetOnActiveProcess = 708,
    ErrorContextIsDestroyed = 709,
    ErrorAssert = 710,
    ErrorHostMemoryAlreadyRegistered = 712,
    ErrorHostMemoryNotRegistered = 713,
    ErrorLaunchFailure = 719,
    ErrorCooperativeLaunchTooLarge = 720,
    ErrorNotSupported = 801,
    ErrorStreamCaptureUnsupported = 900,
    ErrorStreamCaptureInvalidated = 901,
    ErrorStreamCaptureMerge = 902,
    ErrorStreamCaptureUnmatched = 903,
    ErrorStreamCaptureUnjoined = 904,
    ErrorStreamCaptureIsolation = 905,
    ErrorStreamCaptureImplicit = 906,
    ErrorCapturedEvent = 907,
    ErrorStreamCaptureWrongThread = 908,
    ErrorUnknown = 999
}

/// <summary>
/// HIP memory copy direction.
/// </summary>
public enum HipMemcpyKind
{
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4
}

/// <summary>
/// HIP device attribute for querying device properties.
/// </summary>
public enum HipDeviceAttribute
{
    MaxThreadsPerBlock = 0,
    MaxBlockDimX = 1,
    MaxBlockDimY = 2,
    MaxBlockDimZ = 3,
    MaxGridDimX = 4,
    MaxGridDimY = 5,
    MaxGridDimZ = 6,
    MaxSharedMemoryPerBlock = 7,
    TotalConstantMemory = 8,
    WarpSize = 9,
    MaxRegistersPerBlock = 10,
    ClockRate = 11,
    MemoryClockRate = 12,
    MemoryBusWidth = 13,
    MultiprocessorCount = 14,
    ComputeMode = 15,
    L2CacheSize = 16,
    MaxThreadsPerMultiProcessor = 17,
    ComputeCapabilityMajor = 18,
    ComputeCapabilityMinor = 19,
    GcnArchName = 100,
    MaxSharedMemoryPerMultiprocessor = 101,
    CooperativeLaunch = 102,
    CooperativeMultiDeviceLaunch = 103,
    MaxTexture1DWidth = 104,
    PageableMemoryAccess = 105,
    ConcurrentManagedAccess = 106
}

/// <summary>
/// HIP module load options.
/// </summary>
[Flags]
public enum HipJitOption
{
    MaxRegisters = 0,
    ThreadsPerBlock = 1,
    WallTime = 2,
    InfoLogBuffer = 3,
    InfoLogBufferSizeBytes = 4,
    ErrorLogBuffer = 5,
    ErrorLogBufferSizeBytes = 6,
    OptimizationLevel = 7,
    TargetFromContext = 8,
    Target = 9,
    FallbackStrategy = 10,
    GenerateDebugInfo = 11,
    LogVerbose = 12,
    GenerateLineInfo = 13,
    CacheMode = 14
}

/// <summary>
/// P/Invoke bindings for HIP runtime library.
/// Supports both Windows (amdhip64.dll) and Linux (libamdhip64.so).
/// </summary>
internal static class HipNativeBindings
{
    // Library name varies by platform and ROCm version
    // ROCm 6.4 on Windows installs versioned DLLs:
    //   C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll
    //   C:\Program Files\AMD\ROCm\6.4\bin\hiprtc0604.dll
#if WINDOWS
    private const string HipLibrary = "amdhip64_6";
    private const string HipRtcLibrary = "hiprtc0604";
#else
    private const string HipLibrary = "libamdhip64";
    private const string HipRtcLibrary = "libhiprtc";
#endif

    private static bool _isAvailable;
    private static bool _availabilityChecked;

    /// <summary>
    /// Gets whether HIP runtime is available on this system.
    /// </summary>
    public static bool IsAvailable
    {
        get
        {
            if (!_availabilityChecked)
            {
                _availabilityChecked = true;
                try
                {
                    int count = 0;
                    var result = hipGetDeviceCount(ref count);
                    _isAvailable = result == HipError.Success && count > 0;
                    Console.WriteLine($"[HIP Diagnostics] hipGetDeviceCount returned: {result}, device count: {count}, available: {_isAvailable}");
                }
                catch (DllNotFoundException ex)
                {
                    Console.WriteLine($"[HIP Diagnostics] DllNotFoundException: {ex.Message}");
                    Console.WriteLine($"[HIP Diagnostics] amdhip64.dll not found. AMD HIP SDK is not installed.");
                    Console.WriteLine($"[HIP Diagnostics] To use HIP, install AMD ROCm/HIP SDK from https://rocm.docs.amd.com/");
                    _isAvailable = false;
                }
                catch (EntryPointNotFoundException ex)
                {
                    Console.WriteLine($"[HIP Diagnostics] EntryPointNotFoundException: {ex.Message}");
                    Console.WriteLine($"[HIP Diagnostics] HIP DLL found but missing expected function. SDK version mismatch?");
                    _isAvailable = false;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[HIP Diagnostics] Exception during availability check: {ex.GetType().Name}: {ex.Message}");
                    _isAvailable = false;
                }
            }
            return _isAvailable;
        }
    }

    #region Device Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetDeviceCount(ref int count);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipSetDevice(int deviceId);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetDevice(ref int deviceId);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipDeviceSynchronize();

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipDeviceReset();

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipDeviceGetAttribute(
        ref int value,
        HipDeviceAttribute attribute,
        int deviceId);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetDeviceProperties(
        ref HipDeviceProperties props,
        int deviceId);

    #endregion

    #region Memory Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMalloc(ref IntPtr ptr, UIntPtr size);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipFree(IntPtr ptr);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemcpy(
        IntPtr dst,
        IntPtr src,
        UIntPtr sizeBytes,
        HipMemcpyKind kind);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemcpyAsync(
        IntPtr dst,
        IntPtr src,
        UIntPtr sizeBytes,
        HipMemcpyKind kind,
        IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemset(
        IntPtr ptr,
        int value,
        UIntPtr sizeBytes);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemGetInfo(
        ref UIntPtr free,
        ref UIntPtr total);

    #endregion

    #region Stream Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamCreate(ref IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamDestroy(IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamSynchronize(IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamWaitEvent(
        IntPtr stream,
        IntPtr hipEvent,
        uint flags);

    #endregion

    #region Module and Kernel Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLoad(
        ref IntPtr module,
        [MarshalAs(UnmanagedType.LPStr)] string fname);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLoadData(
        ref IntPtr module,
        IntPtr image);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLoadDataEx(
        ref IntPtr module,
        IntPtr image,
        uint numOptions,
        IntPtr options,
        IntPtr optionValues);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleUnload(IntPtr module);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleGetFunction(
        ref IntPtr function,
        IntPtr module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLaunchKernel(
        IntPtr function,
        uint gridDimX,
        uint gridDimY,
        uint gridDimZ,
        uint blockDimX,
        uint blockDimY,
        uint blockDimZ,
        uint sharedMemBytes,
        IntPtr stream,
        IntPtr kernelParams,
        IntPtr extra);

    #endregion

    #region RTC (Runtime Compilation)

    // Note: HIP RTC library name varies by platform
    // Windows: hiprtc.dll (without version suffix)
    // Linux: libhiprtc.so
    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcCreateProgram")]
    public static extern HipRtcResult hiprtcCreateProgram(
        ref IntPtr prog,
        [MarshalAs(UnmanagedType.LPStr)] string src,
        [MarshalAs(UnmanagedType.LPStr)] string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcCompileProgram")]
    public static extern HipRtcResult hiprtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcGetCodeSize")]
    public static extern HipRtcResult hiprtcGetCodeSize(
        IntPtr prog,
        ref UIntPtr codeSize);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcGetCode")]
    public static extern HipRtcResult hiprtcGetCode(
        IntPtr prog,
        IntPtr code);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcGetProgramLogSize")]
    public static extern HipRtcResult hiprtcGetProgramLogSize(
        IntPtr prog,
        ref UIntPtr logSize);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcGetProgramLog")]
    public static extern HipRtcResult hiprtcGetProgramLog(
        IntPtr prog,
        IntPtr log);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl, EntryPoint = "hiprtcDestroyProgram")]
    public static extern HipRtcResult hiprtcDestroyProgram(ref IntPtr prog);

    #endregion

    #region Event Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventCreate(ref IntPtr hipEvent);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventDestroy(IntPtr hipEvent);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventRecord(IntPtr hipEvent, IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventSynchronize(IntPtr hipEvent);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventElapsedTime(
        ref float ms,
        IntPtr start,
        IntPtr stop);

    #endregion

    #region Error Handling

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr hipGetErrorString(HipError error);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr hipGetErrorName(HipError error);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetLastError();

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipPeekAtLastError();

    #endregion

    /// <summary>
    /// Checks a HIP result and throws if it indicates an error.
    /// </summary>
    public static void CheckError(HipError error, string operation = "HIP operation")
    {
        if (error != HipError.Success)
        {
            string errorName = Marshal.PtrToStringAnsi(hipGetErrorName(error)) ?? "Unknown";
            string errorString = Marshal.PtrToStringAnsi(hipGetErrorString(error)) ?? "Unknown error";
            throw new HipException($"{operation} failed: {errorName} - {errorString}", error);
        }
    }
}

/// <summary>
/// HIP RTC (Runtime Compilation) result codes.
/// </summary>
public enum HipRtcResult
{
    Success = 0,
    ErrorOutOfMemory = 1,
    ErrorProgramCreationFailure = 2,
    ErrorInvalidInput = 3,
    ErrorInvalidProgram = 4,
    ErrorInvalidOption = 5,
    ErrorCompilation = 6,
    ErrorBuiltinOperationFailure = 7,
    ErrorNoNameExpressionsAfterCompilation = 8,
    ErrorNoLoweredNamesBeforeCompilation = 9,
    ErrorNameExpressionNotValid = 10,
    ErrorInternalError = 11
}

/// <summary>
/// HIP device properties structure.
/// </summary>
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public struct HipDeviceProperties
{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
    public string Name;

    public UIntPtr TotalGlobalMem;
    public UIntPtr SharedMemPerBlock;
    public int RegsPerBlock;
    public int WarpSize;
    public int MaxThreadsPerBlock;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public int[] MaxThreadsDim;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    public int[] MaxGridSize;

    public int ClockRate;
    public int MemoryClockRate;
    public int MemoryBusWidth;
    public UIntPtr TotalConstMem;
    public int Major;
    public int Minor;
    public int MultiProcessorCount;
    public int L2CacheSize;
    public int MaxThreadsPerMultiProcessor;
    public int ComputeMode;
    public int ClockInstructionRate;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
    public string GcnArchName;

    public int GcnArch;
    public int CooperativeLaunch;
    public int CooperativeMultiDeviceLaunch;

    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public int[] Reserved;
}

/// <summary>
/// Exception thrown when a HIP operation fails.
/// </summary>
public class HipException : Exception
{
    public HipError ErrorCode { get; }

    public HipException(string message, HipError errorCode)
        : base(message)
    {
        ErrorCode = errorCode;
    }
}
