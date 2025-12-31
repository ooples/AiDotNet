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
    private static bool _availabilityChecked;
    private static INvrtcApi? _api;

    public static bool IsAvailable
    {
        get
        {
            EnsureInitialized();
            return _api != null;
        }
    }

    private static void EnsureInitialized()
    {
        if (_availabilityChecked)
            return;

        _availabilityChecked = true;
        _api = ResolveApi();
    }

    private static INvrtcApi? ResolveApi()
    {
#if WINDOWS
        INvrtcApi[] candidates =
        {
            new NvrtcApi120(),
            new NvrtcApi12(),
            new NvrtcApi118(),
            new NvrtcApi112()
        };
#else
        INvrtcApi[] candidates =
        {
            new NvrtcApi12(),
            new NvrtcApi118(),
            new NvrtcApi11(),
            new NvrtcApiDefault()
        };
#endif

        foreach (var candidate in candidates)
        {
            if (TryProbe(candidate))
                return candidate;
        }

        return null;
    }

    private static bool TryProbe(INvrtcApi api)
    {
        try
        {
            return api.Version(out _, out _) == NvrtcResult.Success;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        catch (BadImageFormatException)
        {
            return false;
        }
        catch
        {
            return false;
        }
    }

    private static INvrtcApi Api
    {
        get
        {
            EnsureInitialized();
            return _api ?? throw new DllNotFoundException("NVRTC library not found.");
        }
    }

    public static NvrtcResult nvrtcVersion(out int major, out int minor)
    {
        return Api.Version(out major, out minor);
    }

    public static NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames)
    {
        return Api.CreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    }

    public static NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        string[] options)
    {
        return Api.CompileProgram(prog, numOptions, options);
    }

    public static NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize)
    {
        return Api.GetPtxSize(prog, out ptxSize);
    }

    public static NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx)
    {
        return Api.GetPtx(prog, ptx);
    }

    public static NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize)
    {
        return Api.GetProgramLogSize(prog, out logSize);
    }

    public static NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log)
    {
        return Api.GetProgramLog(prog, log);
    }

    public static NvrtcResult nvrtcDestroyProgram(ref IntPtr prog)
    {
        return Api.DestroyProgram(ref prog);
    }

    public static string GetErrorString(NvrtcResult result)
    {
        var ptr = Api.GetErrorString(result);
        return ptr == IntPtr.Zero ? "Unknown NVRTC error" : Marshal.PtrToStringAnsi(ptr) ?? "Unknown NVRTC error";
    }
}

internal interface INvrtcApi
{
    NvrtcResult Version(out int major, out int minor);
    NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames);
    NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options);
    NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize);
    NvrtcResult GetPtx(IntPtr prog, IntPtr ptx);
    NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize);
    NvrtcResult GetProgramLog(IntPtr prog, IntPtr log);
    NvrtcResult DestroyProgram(ref IntPtr prog);
    IntPtr GetErrorString(NvrtcResult result);
}

#if WINDOWS
internal sealed class NvrtcApi120 : INvrtcApi
{
    private const string Library = "nvrtc64_120_0";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}

internal sealed class NvrtcApi12 : INvrtcApi
{
    private const string Library = "nvrtc64_12";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}

internal sealed class NvrtcApi118 : INvrtcApi
{
    private const string Library = "nvrtc64_118_0";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}

internal sealed class NvrtcApi112 : INvrtcApi
{
    private const string Library = "nvrtc64_112_0";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}
#else
internal sealed class NvrtcApi12 : INvrtcApi
{
    private const string Library = "libnvrtc.so.12";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}

internal sealed class NvrtcApi118 : INvrtcApi
{
    private const string Library = "libnvrtc.so.11.8";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}

internal sealed class NvrtcApi11 : INvrtcApi
{
    private const string Library = "libnvrtc.so.11";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}

internal sealed class NvrtcApiDefault : INvrtcApi
{
    private const string Library = "libnvrtc.so";

    [DllImport(Library, EntryPoint = "nvrtcVersion")]
    private static extern NvrtcResult nvrtcVersion(out int major, out int minor);

    [DllImport(Library, EntryPoint = "nvrtcCreateProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCreateProgram(
        ref IntPtr prog,
        string src,
        string name,
        int numHeaders,
        IntPtr headers,
        IntPtr includeNames);

    [DllImport(Library, EntryPoint = "nvrtcCompileProgram", CharSet = CharSet.Ansi)]
    private static extern NvrtcResult nvrtcCompileProgram(
        IntPtr prog,
        int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [DllImport(Library, EntryPoint = "nvrtcGetPTXSize")]
    private static extern NvrtcResult nvrtcGetPTXSize(IntPtr prog, out UIntPtr ptxSize);

    [DllImport(Library, EntryPoint = "nvrtcGetPTX")]
    private static extern NvrtcResult nvrtcGetPTX(IntPtr prog, IntPtr ptx);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLogSize")]
    private static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out UIntPtr logSize);

    [DllImport(Library, EntryPoint = "nvrtcGetProgramLog")]
    private static extern NvrtcResult nvrtcGetProgramLog(IntPtr prog, IntPtr log);

    [DllImport(Library, EntryPoint = "nvrtcDestroyProgram")]
    private static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(Library, EntryPoint = "nvrtcGetErrorString")]
    private static extern IntPtr nvrtcGetErrorString(NvrtcResult result);

    public NvrtcResult Version(out int major, out int minor) => nvrtcVersion(out major, out minor);
    public NvrtcResult CreateProgram(ref IntPtr prog, string src, string name, int numHeaders, IntPtr headers, IntPtr includeNames)
        => nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames);
    public NvrtcResult CompileProgram(IntPtr prog, int numOptions, string[] options) => nvrtcCompileProgram(prog, numOptions, options);
    public NvrtcResult GetPtxSize(IntPtr prog, out UIntPtr ptxSize) => nvrtcGetPTXSize(prog, out ptxSize);
    public NvrtcResult GetPtx(IntPtr prog, IntPtr ptx) => nvrtcGetPTX(prog, ptx);
    public NvrtcResult GetProgramLogSize(IntPtr prog, out UIntPtr logSize) => nvrtcGetProgramLogSize(prog, out logSize);
    public NvrtcResult GetProgramLog(IntPtr prog, IntPtr log) => nvrtcGetProgramLog(prog, log);
    public NvrtcResult DestroyProgram(ref IntPtr prog) => nvrtcDestroyProgram(ref prog);
    public IntPtr GetErrorString(NvrtcResult result) => nvrtcGetErrorString(result);
}
#endif
