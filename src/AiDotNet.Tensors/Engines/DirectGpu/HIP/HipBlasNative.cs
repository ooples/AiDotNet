// Copyright (c) AiDotNet. All rights reserved.
// hipBLAS native bindings for HIP GEMM acceleration.
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

internal static class HipBlasNative
{
    private const string HipBlasLibrary = "hipblas";
    private static bool _isAvailable;
    private static bool _checkedAvailability;
    private static readonly object AvailabilityLock = new object();

    internal enum HipBlasStatus
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        InvalidValue = 3,
        ArchMismatch = 4,
        MappingError = 5,
        ExecutionFailed = 6,
        InternalError = 7,
        NotSupported = 8
    }

    internal enum HipBlasOperation
    {
        None = 111,
        Transpose = 112,
        ConjugateTranspose = 113
    }

    internal static bool IsAvailable
    {
        get
        {
            if (_checkedAvailability)
            {
                return _isAvailable;
            }

            lock (AvailabilityLock)
            {
                if (_checkedAvailability)
                {
                    return _isAvailable;
                }

                _isAvailable = TryLoadLibrary();
                _checkedAvailability = true;
                return _isAvailable;
            }
        }
    }

    // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.
    [DllImport(HipBlasLibrary, EntryPoint = "hipblasCreate")]
    internal static extern HipBlasStatus hipblasCreate(ref IntPtr handle);

    // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.
    [DllImport(HipBlasLibrary, EntryPoint = "hipblasDestroy")]
    internal static extern HipBlasStatus hipblasDestroy(IntPtr handle);

    // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.
    [DllImport(HipBlasLibrary, EntryPoint = "hipblasSetStream")]
    internal static extern HipBlasStatus hipblasSetStream(IntPtr handle, IntPtr stream);

    // lgtm[cs/unmanaged-code] HIP BLAS requires native bindings.
    [DllImport(HipBlasLibrary, EntryPoint = "hipblasSgemm")]
    internal static extern HipBlasStatus hipblasSgemm(
        IntPtr handle,
        HipBlasOperation transA,
        HipBlasOperation transB,
        int m,
        int n,
        int k,
        ref float alpha,
        IntPtr A,
        int lda,
        IntPtr B,
        int ldb,
        ref float beta,
        IntPtr C,
        int ldc);

    private static bool TryLoadLibrary()
    {
        string[] candidates =
        {
            HipBlasLibrary,
            "hipblas.dll",
            "libhipblas.so",
            "libhipblas.dylib"
        };

        return candidates.Where(CanLoadLibrary).Any();
    }

    private static bool CanLoadLibrary(string name)
    {
#if NETFRAMEWORK
        var handle = LoadLibrary(name);
        if (handle == IntPtr.Zero)
        {
            return false;
        }

        FreeLibrary(handle);
        return true;
#else
        if (!NativeLibrary.TryLoad(name, out var handle))
        {
            return false;
        }

        NativeLibrary.Free(handle);
        return true;
#endif
    }

#if NETFRAMEWORK
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
#endif
}
