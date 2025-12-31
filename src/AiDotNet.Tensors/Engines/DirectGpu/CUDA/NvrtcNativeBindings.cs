// Copyright (c) AiDotNet. All rights reserved.
// NVRTC bindings for runtime CUDA kernel compilation.
using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// NVRTC result codes.
/// </summary>
public enum NvrtcResult
{
    Success = 0,
    ErrorOutOfMemory = 2,
    ErrorProgramCreationFailure = 3,
    ErrorInvalidInput = 4,
    ErrorInvalidProgram = 5,
    ErrorInvalidOption = 6,
    ErrorCompilation = 7,
    ErrorBuiltinOperationFailure = 8,
    ErrorNoNameExpressionsAfterCompilation = 9,
    ErrorNoLoweredNamesBeforeCompilation = 10,
    ErrorNameExpressionNotValid = 11,
    ErrorInternalError = 12
}

internal static class NvrtcNativeBindings
{
#if WINDOWS
    private const string NvrtcLibrary = "nvrtc64_120_0";
#else
    private const string NvrtcLibrary = "libnvrtc";
#endif

    private static bool _isAvailable;
    private static bool _availabilityChecked;

    public static bool IsAvailable
    {
        get
        {
            if (!_availabilityChecked)
            {
                _availabilityChecked = true;
                try
                {
                    var result = nvrtcVersion(out _, out _);
                    _isAvailable = result == NvrtcResult.Success;
                }
                catch (DllNotFoundException)
                {
                    _isAvailable = false;
                }
                catch (EntryPointNotFoundException)
                {
                    _isAvailable = false;
                }
                catch
                {
                    _isAvailable = false;
                }
            }
            return _isAvailable;
        }
    }

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcVersion")]
    public static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    public static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    public static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetPTXSize")]
    public static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetPTX")]
    public static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetProgramLogSize")]
    public static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetProgramLog")]
    public static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcDestroyProgram")]
    public static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public static string GetErrorString(NvrtcResult result)
    {
        var ptr = nvrtcGetErrorString(result);
        return ptr == IntPtr.Zero ? "Unknown NVRTC error" : Marshal.PtrToStringAnsi(ptr) ?? "Unknown NVRTC error";
    }
}
