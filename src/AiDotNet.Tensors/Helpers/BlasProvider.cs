using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

internal static class BlasProvider
{
    private const int CblasRowMajor = 101;
    private const int CblasNoTrans = 111;

    private static readonly object InitLock = new object();
    private static bool _initialized;
    private static bool _available;
    private static IntPtr _libraryHandle;
    private static CblasSgemm? _sgemm;
    private static CblasDgemm? _dgemm;

#if NETFRAMEWORK
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
#endif

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void CblasSgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        float alpha,
        float* a, int lda,
        float* b, int ldb,
        float beta,
        float* c, int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void CblasDgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        double alpha,
        double* a, int lda,
        double* b, int ldb,
        double beta,
        double* c, int ldc);

    internal static bool TryGemm(int m, int n, int k, float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb, float[] c, int cOffset, int ldc)
    {
        if (!EnsureInitialized() || _sgemm == null)
        {
            return false;
        }

        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        unsafe
        {
            fixed (float* aPtrBase = a)
            fixed (float* bPtrBase = b)
            fixed (float* cPtrBase = c)
            {
                float* aPtr = aPtrBase + aOffset;
                float* bPtr = bPtrBase + bOffset;
                float* cPtr = cPtrBase + cOffset;
                _sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, aPtr, lda, bPtr, ldb, 0.0f, cPtr, ldc);
            }
        }

        return true;
    }

    internal static bool TryGemm(int m, int n, int k, double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb, double[] c, int cOffset, int ldc)
    {
        if (!EnsureInitialized() || _dgemm == null)
        {
            return false;
        }

        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        unsafe
        {
            fixed (double* aPtrBase = a)
            fixed (double* bPtrBase = b)
            fixed (double* cPtrBase = c)
            {
                double* aPtr = aPtrBase + aOffset;
                double* bPtr = bPtrBase + bOffset;
                double* cPtr = cPtrBase + cOffset;
                _dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, aPtr, lda, bPtr, ldb, 0.0, cPtr, ldc);
            }
        }

        return true;
    }

    private static bool EnsureInitialized()
    {
        if (_initialized)
        {
            return _available;
        }

        lock (InitLock)
        {
            if (_initialized)
            {
                return _available;
            }

            _available = TryLoadLibrary();
            _initialized = true;
            return _available;
        }
    }

    private static bool TryLoadLibrary()
    {
        var disable = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_BLAS");
        if (!string.IsNullOrWhiteSpace(disable) &&
            (string.Equals(disable, "1", StringComparison.OrdinalIgnoreCase) ||
             string.Equals(disable, "true", StringComparison.OrdinalIgnoreCase)))
        {
            return false;
        }

        string? explicitPath = Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PATH");
        if (!string.IsNullOrWhiteSpace(explicitPath) && TryLoadNativeLibrary(explicitPath, out _libraryHandle))
        {
            return TryLoadSymbols();
        }

        foreach (var candidate in GetCandidateLibraryNames())
        {
            if (TryLoadNativeLibrary(candidate, out _libraryHandle))
            {
                if (TryLoadSymbols())
                {
                    return true;
                }

                FreeNativeLibrary(_libraryHandle);
                _libraryHandle = IntPtr.Zero;
            }
        }

        return false;
    }

    private static bool TryLoadSymbols()
    {
        _sgemm = TryLoadSymbol<CblasSgemm>("cblas_sgemm");
        _dgemm = TryLoadSymbol<CblasDgemm>("cblas_dgemm");
        return _sgemm != null || _dgemm != null;
    }

    private static T? TryLoadSymbol<T>(string name) where T : class
    {
        if (_libraryHandle == IntPtr.Zero)
        {
            return null;
        }

        if (!TryGetNativeExport(_libraryHandle, name, out var symbol) || symbol == IntPtr.Zero)
        {
            return null;
        }

        return Marshal.GetDelegateForFunctionPointer(symbol, typeof(T)) as T;
    }

    private static bool HasEnoughData(int length, int offset, int rows, int cols, int stride)
    {
        if (length <= 0 || rows <= 0 || cols <= 0)
        {
            return false;
        }

        if (offset < 0 || stride < cols)
        {
            return false;
        }

        long lastIndex = (long)offset + (long)(rows - 1) * stride + (cols - 1);
        return lastIndex < length;
    }

    private static string[] GetCandidateLibraryNames()
    {
        return new[]
        {
            "openblas",
            "libopenblas",
            "openblas.dll",
            "libopenblas.dll",
            "libopenblas.so",
            "libopenblas.so.0",
            "libopenblas.dylib",
            "mkl_rt",
            "mkl_rt.dll",
            "libmkl_rt.so",
            "libmkl_rt.dylib"
        };
    }

#if NETFRAMEWORK
    private static bool TryLoadNativeLibrary(string path, out IntPtr handle)
    {
        handle = LoadLibrary(path);
        return handle != IntPtr.Zero;
    }

    private static bool TryGetNativeExport(IntPtr handle, string name, out IntPtr symbol)
    {
        symbol = GetProcAddress(handle, name);
        return symbol != IntPtr.Zero;
    }

    private static void FreeNativeLibrary(IntPtr handle)
    {
        if (handle != IntPtr.Zero)
        {
            FreeLibrary(handle);
        }
    }
#else
    private static bool TryLoadNativeLibrary(string path, out IntPtr handle)
        => NativeLibrary.TryLoad(path, out handle);

    private static bool TryGetNativeExport(IntPtr handle, string name, out IntPtr symbol)
        => NativeLibrary.TryGetExport(handle, name, out symbol);

    private static void FreeNativeLibrary(IntPtr handle)
        => NativeLibrary.Free(handle);
#endif
}
