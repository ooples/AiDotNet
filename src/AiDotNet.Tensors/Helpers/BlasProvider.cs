using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    private static BlasSetNumThreads? _setNumThreads;
    private static MklSetDynamic? _setDynamic;
    private static readonly int? ThreadCountOverride = ReadEnvInt("AIDOTNET_BLAS_THREADS");
    private static readonly bool PreferMkl = ReadEnvBool("AIDOTNET_BLAS_PREFER_MKL");

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

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void BlasSetNumThreads(int threads);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void MklSetDynamic(int enabled);

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
            if (TryLoadSymbols())
            {
                return true;
            }

            FreeNativeLibrary(_libraryHandle);
            _libraryHandle = IntPtr.Zero;
            return false;
        }

        foreach (var name in GetCandidateLibraryNames())
        {
            if (!TryLoadNativeLibrary(name, out _libraryHandle))
            {
                continue;
            }

            if (TryLoadSymbols())
            {
                return true;
            }

            FreeNativeLibrary(_libraryHandle);
            _libraryHandle = IntPtr.Zero;
        }

        return false;
    }

    private static bool TryLoadSymbols()
    {
        _sgemm = TryLoadSymbol<CblasSgemm>("cblas_sgemm");
        _dgemm = TryLoadSymbol<CblasDgemm>("cblas_dgemm");
        TryLoadThreadControls();
        return _sgemm != null || _dgemm != null;
    }

    private static void TryLoadThreadControls()
    {
        _setNumThreads = TryLoadSymbol<BlasSetNumThreads>("openblas_set_num_threads")
            ?? TryLoadSymbol<BlasSetNumThreads>("mkl_set_num_threads");
        _setDynamic = TryLoadSymbol<MklSetDynamic>("mkl_set_dynamic");
        ApplyThreadSettings();
    }

    private static void ApplyThreadSettings()
    {
        if (_setNumThreads == null)
        {
            return;
        }

        if (!ThreadCountOverride.HasValue || ThreadCountOverride.Value <= 0)
        {
            return;
        }

        try
        {
            _setDynamic?.Invoke(0);
            _setNumThreads(ThreadCountOverride.Value);
        }
        catch (DllNotFoundException)
        {
            // Ignore failures to keep BLAS optional and non-fatal.
        }
        catch (EntryPointNotFoundException)
        {
            // Ignore failures to keep BLAS optional and non-fatal.
        }
        catch (BadImageFormatException)
        {
            // Ignore failures to keep BLAS optional and non-fatal.
        }
        catch (SEHException)
        {
            // Ignore failures to keep BLAS optional and non-fatal.
        }
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
        string[] openblas =
        {
            "openblas",
            "libopenblas",
            "openblas.dll",
            "libopenblas.dll",
            "libopenblas.so",
            "libopenblas.so.0",
            "libopenblas.dylib"
        };

        string[] mkl =
        {
            "mkl_rt",
            "mkl_rt.dll",
            "libmkl_rt.so",
            "libmkl_rt.dylib"
        };

        bool preferMkl = PreferMkl || RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        var names = new List<string>(openblas.Length + mkl.Length);
        if (preferMkl)
        {
            names.AddRange(mkl);
            names.AddRange(openblas);
        }
        else
        {
            names.AddRange(openblas);
            names.AddRange(mkl);
        }

        // Also try loading from various known directories
        var additionalPaths = new List<string>();
        var directories = new List<string?>
        {
            AppContext.BaseDirectory,
            Path.GetDirectoryName(typeof(BlasProvider).Assembly.Location),
            Environment.CurrentDirectory
        };

        foreach (var dir in directories.Where(d => !string.IsNullOrEmpty(d)))
        {
            foreach (var name in names.ToArray())
            {
                var fullPath = Path.Combine(dir!, name);
                if (!additionalPaths.Contains(fullPath))
                {
                    additionalPaths.Add(fullPath);
                }
            }
        }

        // Add directory-relative paths at the beginning for priority
        names.InsertRange(0, additionalPaths);

        return names.ToArray();
    }

    private static int? ReadEnvInt(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return int.TryParse(raw, out var value) && value > 0 ? value : null;
    }

    private static bool ReadEnvBool(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        return string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase);
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
