using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using MKLNET;

namespace AiDotNet.Tensors.Helpers;

internal static class BlasProvider
{
    private const int CblasRowMajor = 101;
    private const int CblasNoTrans = 111;

    private static readonly object InitLock = new object();
    private static bool _initialized;
    private static bool _available;
    private static bool _useMklNet;  // True if using MKL.NET managed bindings
    private static IntPtr _libraryHandle;
    private static CblasSgemm? _sgemm;
    private static CblasDgemm? _dgemm;
    private static BlasSetNumThreads? _setNumThreads;
    private static MklSetDynamic? _setDynamic;
    private static readonly int? ThreadCountOverride = ReadEnvInt("AIDOTNET_BLAS_THREADS");
    private static readonly bool PreferMkl = ReadEnvBool("AIDOTNET_BLAS_PREFER_MKL");
    private static readonly bool TraceEnabled = ReadEnvBool("AIDOTNET_BLAS_TRACE");

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

    /// <summary>
    /// Returns true if a BLAS library has been successfully loaded.
    /// Useful for diagnostics to verify native library is available.
    /// </summary>
    internal static bool IsAvailable
    {
        get
        {
            EnsureInitialized();
            return _available && (_useMklNet || _sgemm != null);
        }
    }

    /// <summary>
    /// Returns the name of the BLAS backend being used.
    /// </summary>
    internal static string BackendName
    {
        get
        {
            EnsureInitialized();
            if (_useMklNet) return "Intel MKL.NET";
            if (_sgemm != null) return "Native BLAS";
            return "None";
        }
    }

    internal static bool TryGemm(int m, int n, int k, float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb, float[] c, int cOffset, int ldc)
    {
        if (!EnsureInitialized())
        {
            return false;
        }

        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        // Use MKL.NET if available (preferred for performance)
        if (_useMklNet)
        {
            return TryMklNetSgemm(m, n, k, a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc);
        }

        // Fall back to native library
        if (_sgemm == null)
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

    /// <summary>
    /// Span-based GEMM for zero-copy operations.
    /// </summary>
    internal static bool TryGemm(int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (!EnsureInitialized())
        {
            return false;
        }

        // Use MKL.NET if available (preferred for performance)
        if (_useMklNet)
        {
            try
            {
                Blas.gemm(
                    Layout.RowMajor,
                    Trans.No,
                    Trans.No,
                    m, n, k,
                    1.0f,
                    a, lda,
                    b, ldb,
                    0.0f,
                    c, ldc);
                return true;
            }
            catch (Exception)
            {
                // MKL.NET call failed, disable and fall back to native
                _useMklNet = false;
                return false;
            }
        }

        // Fall back to native library with pinned pointers
        if (_sgemm == null)
        {
            return false;
        }

        unsafe
        {
            fixed (float* aPtr = a)
            fixed (float* bPtr = b)
            fixed (float* cPtr = c)
            {
                _sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, aPtr, lda, bPtr, ldb, 0.0f, cPtr, ldc);
            }
        }

        return true;
    }

    /// <summary>
    /// Span-based GEMM for double precision.
    /// </summary>
    internal static bool TryGemm(int m, int n, int k, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (!EnsureInitialized())
        {
            return false;
        }

        if (_useMklNet)
        {
            try
            {
                Blas.gemm(
                    Layout.RowMajor,
                    Trans.No,
                    Trans.No,
                    m, n, k,
                    1.0,
                    a, lda,
                    b, ldb,
                    0.0,
                    c, ldc);
                return true;
            }
            catch (Exception)
            {
                // MKL.NET call failed, disable and fall back to native
                _useMklNet = false;
                return false;
            }
        }

        if (_dgemm == null)
        {
            return false;
        }

        unsafe
        {
            fixed (double* aPtr = a)
            fixed (double* bPtr = b)
            fixed (double* cPtr = c)
            {
                _dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, aPtr, lda, bPtr, ldb, 0.0, cPtr, ldc);
            }
        }

        return true;
    }

    private static bool TryMklNetSgemm(int m, int n, int k, float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb, float[] c, int cOffset, int ldc)
    {
        try
        {
            // Use MKL.NET's managed bindings - it handles native library loading automatically
            // Create offset views for the arrays
            var aSpan = a.AsSpan(aOffset);
            var bSpan = b.AsSpan(bOffset);
            var cSpan = c.AsSpan(cOffset);

            Blas.gemm(
                Layout.RowMajor,
                Trans.No,
                Trans.No,
                m, n, k,
                1.0f,
                aSpan, lda,
                bSpan, ldb,
                0.0f,
                cSpan, ldc);

            return true;
        }
        catch (Exception)
        {
            // If MKL.NET fails, mark it as unavailable and return false
            _useMklNet = false;
            return false;
        }
    }

    internal static bool TryGemm(int m, int n, int k, double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb, double[] c, int cOffset, int ldc)
    {
        if (!EnsureInitialized())
        {
            return false;
        }

        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        // Use MKL.NET if available (preferred for performance)
        if (_useMklNet)
        {
            return TryMklNetDgemm(m, n, k, a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc);
        }

        // Fall back to native library
        if (_dgemm == null)
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

    private static bool TryMklNetDgemm(int m, int n, int k, double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb, double[] c, int cOffset, int ldc)
    {
        try
        {
            var aSpan = a.AsSpan(aOffset);
            var bSpan = b.AsSpan(bOffset);
            var cSpan = c.AsSpan(cOffset);

            Blas.gemm(
                Layout.RowMajor,
                Trans.No,
                Trans.No,
                m, n, k,
                1.0,
                aSpan, lda,
                bSpan, ldb,
                0.0,
                cSpan, ldc);

            return true;
        }
        catch (Exception)
        {
            // MKL.NET call failed, disable and fall back to native
            _useMklNet = false;
            return false;
        }
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
        Trace("[BLAS] TryLoadLibrary starting...");
        Trace($"[BLAS] AppContext.BaseDirectory: {AppContext.BaseDirectory}");

        var disable = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_BLAS");
        if (!string.IsNullOrWhiteSpace(disable) &&
            (string.Equals(disable, "1", StringComparison.OrdinalIgnoreCase) ||
             string.Equals(disable, "true", StringComparison.OrdinalIgnoreCase)))
        {
            Trace("[BLAS] Disabled via AIDOTNET_DISABLE_BLAS");
            return false;
        }

        // Try MKL.NET first (preferred for best performance)
        if (TryInitializeMklNet())
        {
            Trace("[BLAS] Initialized MKL.NET successfully");
            _useMklNet = true;
            return true;
        }
        Trace("[BLAS] MKL.NET not available, trying native libraries...");

        string? explicitPath = Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PATH");
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            Trace($"[BLAS] Explicit path from env: {explicitPath}");
            if (TryLoadNativeLibrary(explicitPath, out _libraryHandle))
            {
                Trace("[BLAS] Loaded from explicit path");
                if (TryLoadSymbols())
                {
                    Trace("[BLAS] Symbols loaded successfully from explicit path");
                    return true;
                }

                FreeNativeLibrary(_libraryHandle);
                _libraryHandle = IntPtr.Zero;
            }
            else
            {
                Trace("[BLAS] Failed to load from explicit path");
            }
            return false;
        }

        var candidates = GetCandidateLibraryNames();
        Trace($"[BLAS] Searching {candidates.Length} candidate paths...");

        foreach (var name in candidates)
        {
            if (!TryLoadNativeLibrary(name, out _libraryHandle))
            {
                continue;
            }

            Trace($"[BLAS] Loaded native library from: {name}");
            if (TryLoadSymbols())
            {
                Trace("[BLAS] BLAS symbols loaded successfully!");
                return true;
            }

            Trace($"[BLAS] Failed to load symbols from: {name}");
            FreeNativeLibrary(_libraryHandle);
            _libraryHandle = IntPtr.Zero;
        }

        Trace("[BLAS] No BLAS library found in any candidate path");
        return false;
    }

    private static bool TryInitializeMklNet()
    {
        try
        {
            // Verify MKL.NET is working by doing a tiny GEMM
            // This will trigger MKL.NET to load its native libraries
            float[] a = { 1.0f };
            float[] b = { 1.0f };
            float[] c = { 0.0f };

            Blas.gemm(
                Layout.RowMajor,
                Trans.No,
                Trans.No,
                1, 1, 1,
                1.0f,
                a.AsSpan(), 1,
                b.AsSpan(), 1,
                0.0f,
                c.AsSpan(), 1);

            // If we get here, MKL.NET is working
            Trace($"[BLAS] MKL.NET verification: 1x1 GEMM result = {c[0]} (expected 1.0)");
            return Math.Abs(c[0] - 1.0f) < 0.001f;
        }
        catch (Exception ex)
        {
            Trace($"[BLAS] MKL.NET initialization failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static void Trace(string message)
    {
        if (TraceEnabled)
        {
            Console.WriteLine(message);
        }
    }

    private static bool TryLoadSymbols()
    {
        _sgemm = TryLoadSymbol<CblasSgemm>("cblas_sgemm");
        _dgemm = TryLoadSymbol<CblasDgemm>("cblas_dgemm");
        Trace($"[BLAS] cblas_sgemm loaded: {_sgemm != null}");
        Trace($"[BLAS] cblas_dgemm loaded: {_dgemm != null}");
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

        try
        {
            // Use override if specified, otherwise use all available processors
            int threadCount = ThreadCountOverride ?? Environment.ProcessorCount;
            _setDynamic?.Invoke(0);
            _setNumThreads(threadCount);
            Trace($"[BLAS] Set thread count to {threadCount}");
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

        // Also search in parent directories (helps with BenchmarkDotNet artifact subdirectories)
        var baseDir = AppContext.BaseDirectory;
        for (int i = 0; i < 5 && !string.IsNullOrEmpty(baseDir); i++)
        {
            baseDir = Path.GetDirectoryName(baseDir);
            if (!string.IsNullOrEmpty(baseDir))
            {
                directories.Add(baseDir);
            }
        }

        // Search in runtimes/win-x64/native subdirectory (common NuGet native layout)
        foreach (var dir in directories.ToList())
        {
            if (!string.IsNullOrEmpty(dir))
            {
                var nativePath = Path.Combine(dir!, "runtimes", "win-x64", "native");
                if (Directory.Exists(nativePath))
                {
                    directories.Add(nativePath);
                }
            }
        }

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
