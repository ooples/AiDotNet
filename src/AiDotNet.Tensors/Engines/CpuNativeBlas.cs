using System;

#if NET5_0_OR_GREATER
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
#endif

namespace AiDotNet.Tensors.Engines;

#if NET5_0_OR_GREATER
internal static class CpuNativeBlas
{
    private const string EnvVar = "AIDOTNET_CPU_BLAS";
    private const string LogEnvVar = "AIDOTNET_CPU_BLAS_LOG";
    private const string ThreadEnvVar = "AIDOTNET_CPU_BLAS_THREADS";
    private static readonly object s_logLock = new();
    private static TextWriter? s_logWriter;
    private static bool s_logInitialized;
    private static int s_loggedUsage;

    private static readonly Lazy<BlasState?> s_state = new Lazy<BlasState?>(LoadState, true);

    internal static bool IsAvailable => s_state.Value != null;

    internal static bool TryGemm(float[] a, float[] b, float[] c, int m, int n, int k)
        => TryGemm(a, 0, b, 0, c, 0, m, n, k, k, n, n);

    internal static bool TryGemm(double[] a, double[] b, double[] c, int m, int n, int k)
        => TryGemm(a, 0, b, 0, c, 0, m, n, k, k, n, n);

    internal static bool TryGemm(
        float[] a,
        int aOffset,
        float[] b,
        int bOffset,
        float[] c,
        int cOffset,
        int m,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc)
    {
        var state = s_state.Value;
        if (state == null || state.Sgemm == null)
        {
            return false;
        }

        LogUsageOnce(isDouble: false, m, n, k);

        unsafe
        {
            fixed (float* aPtrBase = a)
            fixed (float* bPtrBase = b)
            fixed (float* cPtrBase = c)
            {
                state.Sgemm(
                    CblasOrder.RowMajor,
                    CblasTranspose.NoTrans,
                    CblasTranspose.NoTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    aPtrBase + aOffset,
                    lda,
                    bPtrBase + bOffset,
                    ldb,
                    0.0f,
                    cPtrBase + cOffset,
                    ldc);
            }
        }

        return true;
    }

    internal static bool TryGemm(
        double[] a,
        int aOffset,
        double[] b,
        int bOffset,
        double[] c,
        int cOffset,
        int m,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc)
    {
        var state = s_state.Value;
        if (state == null || state.Dgemm == null)
        {
            return false;
        }

        LogUsageOnce(isDouble: true, m, n, k);

        unsafe
        {
            fixed (double* aPtrBase = a)
            fixed (double* bPtrBase = b)
            fixed (double* cPtrBase = c)
            {
                state.Dgemm(
                    CblasOrder.RowMajor,
                    CblasTranspose.NoTrans,
                    CblasTranspose.NoTrans,
                    m,
                    n,
                    k,
                    1.0,
                    aPtrBase + aOffset,
                    lda,
                    bPtrBase + bOffset,
                    ldb,
                    0.0,
                    cPtrBase + cOffset,
                    ldc);
            }
        }

        return true;
    }

    private static BlasState? LoadState()
    {
        var provider = ReadProvider(out var rawValue);
        var rawDisplay = string.IsNullOrWhiteSpace(rawValue) ? "auto" : rawValue.Trim();
        Log($"Provider: {provider} ({EnvVar}={rawDisplay})");
        if (provider == CpuBlasProvider.Off)
        {
            Log("BLAS disabled.");
            return null;
        }

        foreach (var candidate in GetCandidates(provider))
        {
            if (!NativeLibrary.TryLoad(candidate, out var handle))
            {
                Log($"Failed to load {candidate}.");
                continue;
            }

            if (TryGetGemm(handle, out var sgemm, out var dgemm) &&
                sgemm is not null &&
                dgemm is not null)
            {
                Log($"Loaded BLAS from {candidate}.");
                TryConfigureThreads(handle, provider);
                return new BlasState(handle, sgemm, dgemm);
            }

            Log($"Missing cblas exports in {candidate}.");
            NativeLibrary.Free(handle);
        }

        Log("No compatible BLAS library found.");
        return null;
    }

    private static void TryConfigureThreads(IntPtr handle, CpuBlasProvider provider)
    {
        var threadCount = ReadThreadCount();
        if (!threadCount.HasValue)
        {
            return;
        }

        int threads = Math.Max(1, threadCount.Value);
        bool configured = false;

        if (provider == CpuBlasProvider.Mkl || provider == CpuBlasProvider.Auto)
        {
            configured = TrySetThreads(handle, threads, "MKL_Set_Num_Threads", "mkl_set_num_threads");
        }

        if (!configured && (provider == CpuBlasProvider.OpenBlas || provider == CpuBlasProvider.Auto))
        {
            configured = TrySetThreads(handle, threads, "openblas_set_num_threads", "goto_set_num_threads");
        }

        if (configured)
        {
            Log($"Set BLAS threads to {threads} ({ThreadEnvVar}).");
        }
        else
        {
            Log($"Requested BLAS threads ({threads}) but no supported thread setter was found.");
        }
    }

    private static bool TrySetThreads(IntPtr handle, int threads, params string[] symbols)
    {
        foreach (var symbol in symbols)
        {
            if (!NativeLibrary.TryGetExport(handle, symbol, out var ptr))
            {
                continue;
            }

            var setter = Marshal.GetDelegateForFunctionPointer<BlasSetNumThreads>(ptr);
            setter(threads);
            return true;
        }

        return false;
    }

    private static int? ReadThreadCount()
    {
        var value = Environment.GetEnvironmentVariable(ThreadEnvVar);
        if (string.IsNullOrWhiteSpace(value))
        {
            var fallback = ReadThreadCountFromEnv(
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS");
            if (fallback.HasValue)
            {
                return fallback.Value;
            }

            return Environment.ProcessorCount;
        }

        if (int.TryParse(value.Trim(), out var threads) && threads > 0)
        {
            return threads;
        }

        Log($"Invalid {ThreadEnvVar} value '{value}'.");
        return Environment.ProcessorCount;
    }

    private static int? ReadThreadCountFromEnv(params string[] names)
    {
        foreach (var name in names)
        {
            var value = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            if (int.TryParse(value.Trim(), out var threads) && threads > 0)
            {
                return threads;
            }

            Log($"Invalid {name} value '{value}'.");
        }

        return null;
    }

    private static CpuBlasProvider ReadProvider(out string? rawValue)
    {
        rawValue = Environment.GetEnvironmentVariable(EnvVar);
        if (string.IsNullOrWhiteSpace(rawValue))
        {
            return CpuBlasProvider.Auto;
        }

        switch (rawValue.Trim().ToLowerInvariant())
        {
            case "0":
            case "off":
            case "false":
                return CpuBlasProvider.Off;
            case "openblas":
            case "open":
                return CpuBlasProvider.OpenBlas;
            case "mkl":
            case "intel":
                return CpuBlasProvider.Mkl;
            case "auto":
                return CpuBlasProvider.Auto;
            default:
                return CpuBlasProvider.Auto;
        }
    }

    private static TextWriter? GetLogWriter()
    {
        if (s_logInitialized)
        {
            return s_logWriter;
        }

        lock (s_logLock)
        {
            if (s_logInitialized)
            {
                return s_logWriter;
            }

            s_logInitialized = true;
            var value = Environment.GetEnvironmentVariable(LogEnvVar);
            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            value = value.Trim();
            if (value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("console", StringComparison.OrdinalIgnoreCase))
            {
                s_logWriter = Console.Error;
                return s_logWriter;
            }

            try
            {
                s_logWriter = new StreamWriter(value, append: true) { AutoFlush = true };
                return s_logWriter;
            }
            catch (Exception ex)
            {
                s_logWriter = Console.Error;
                Console.Error.WriteLine($"[AiDotNet] CPU BLAS: Failed to open log file '{value}': {ex.Message}");
                return s_logWriter;
            }
        }
    }

    private static void Log(string message)
    {
        var writer = GetLogWriter();
        if (writer == null)
        {
            return;
        }

        writer.WriteLine($"[AiDotNet] CPU BLAS: {message}");
    }

    private static void LogUsageOnce(bool isDouble, int m, int n, int k)
    {
        if (Volatile.Read(ref s_loggedUsage) != 0)
        {
            return;
        }

        if (Interlocked.Exchange(ref s_loggedUsage, 1) != 0)
        {
            return;
        }

        string typeLabel = isDouble ? "dgemm" : "sgemm";
        Log($"Using BLAS {typeLabel} (m={m}, n={n}, k={k}).");
    }

    private static IEnumerable<string> GetCandidates(CpuBlasProvider provider)  
    {
        foreach (var candidate in GetExplicitCandidatePaths(provider))
        {
            yield return candidate;
        }

        foreach (var candidate in GetCandidateNames(provider))
        {
            yield return candidate;
        }
    }

    private static IEnumerable<string> GetExplicitCandidatePaths(CpuBlasProvider provider)
    {
        var names = GetCandidateNames(provider);
        if (names.Count == 0)
        {
            yield break;
        }

        var searchPaths = GetNativeSearchPaths();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var path in searchPaths)
        {
            foreach (var name in names)
            {
                var candidate = Path.Combine(path, name);
                if (!seen.Add(candidate))
                {
                    continue;
                }

                if (File.Exists(candidate))
                {
                    yield return candidate;
                }
            }
        }
    }

    private static List<string> GetCandidateNames(CpuBlasProvider provider)
    {
        var names = new List<string>();
        void AddRange(IEnumerable<string> items)
        {
            foreach (var item in items)
            {
                if (!names.Contains(item))
                {
                    names.Add(item);
                }
            }
        }

        if (provider == CpuBlasProvider.Mkl)
        {
            AddRange(GetMklCandidates());
            return names;
        }

        if (provider == CpuBlasProvider.OpenBlas)
        {
            AddRange(GetOpenBlasCandidates());
            return names;
        }

        AddRange(GetMklCandidates());
        AddRange(GetOpenBlasCandidates());
        return names;
    }

    private static IEnumerable<string> GetNativeSearchPaths()
    {
        var baseDir = AppContext.BaseDirectory;
        if (string.IsNullOrWhiteSpace(baseDir))
        {
            yield break;
        }

        yield return baseDir;

        var rid = RuntimeInformation.RuntimeIdentifier;
        if (!string.IsNullOrWhiteSpace(rid))
        {
            yield return Path.Combine(baseDir, "runtimes", rid, "native");
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            yield return Path.Combine(baseDir, "runtimes", "win-x64", "native");
            yield return Path.Combine(baseDir, "win-x64", "native");
        }
    }

    private static IEnumerable<string> GetMklCandidates()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            yield return "mkl_rt.dll";
            yield break;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            yield return "libmkl_rt.so";
            yield return "libmkl_rt.so.2";
            yield break;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            yield return "libmkl_rt.dylib";
            yield break;
        }
    }

    private static IEnumerable<string> GetOpenBlasCandidates()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            yield return "openblas.dll";
            yield return "libopenblas.dll";
            yield return "libopenblas64_.dll";
            yield break;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            yield return "libopenblas.so";
            yield return "libopenblas.so.0";
            yield return "libopenblas64_.so";
            yield return "libopenblas64_.so.0";
            yield break;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            yield return "libopenblas.dylib";
            yield break;
        }
    }

    private static bool TryGetGemm(IntPtr handle, out BlasSgemm? sgemm, out BlasDgemm? dgemm)
    {
        sgemm = null;
        dgemm = null;

        if (!NativeLibrary.TryGetExport(handle, "cblas_sgemm", out var sgemmPtr))
        {
            return false;
        }

        if (!NativeLibrary.TryGetExport(handle, "cblas_dgemm", out var dgemmPtr))
        {
            return false;
        }

        sgemm = Marshal.GetDelegateForFunctionPointer<BlasSgemm>(sgemmPtr);
        dgemm = Marshal.GetDelegateForFunctionPointer<BlasDgemm>(dgemmPtr);
        return true;
    }

    private sealed class BlasState
    {
        public BlasState(IntPtr handle, BlasSgemm sgemm, BlasDgemm dgemm)
        {
            Handle = handle;
            Sgemm = sgemm;
            Dgemm = dgemm;
        }

        public IntPtr Handle { get; }
        public BlasSgemm Sgemm { get; }
        public BlasDgemm Dgemm { get; }
    }

    private enum CpuBlasProvider
    {
        Auto,
        OpenBlas,
        Mkl,
        Off
    }

    private enum CblasOrder
    {
        RowMajor = 101,
        ColumnMajor = 102
    }

    private enum CblasTranspose
    {
        NoTrans = 111,
        Trans = 112,
        ConjTrans = 113
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void BlasSgemm(
        CblasOrder order,
        CblasTranspose transA,
        CblasTranspose transB,
        int m,
        int n,
        int k,
        float alpha,
        float* a,
        int lda,
        float* b,
        int ldb,
        float beta,
        float* c,
        int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void BlasDgemm(
        CblasOrder order,
        CblasTranspose transA,
        CblasTranspose transB,
        int m,
        int n,
        int k,
        double alpha,
        double* a,
        int lda,
        double* b,
        int ldb,
        double beta,
        double* c,
        int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void BlasSetNumThreads(int threads);
}
#else
internal static class CpuNativeBlas
{
    internal static bool IsAvailable => false;

    internal static bool TryGemm(float[] a, float[] b, float[] c, int m, int n, int k) => false;

    internal static bool TryGemm(double[] a, double[] b, double[] c, int m, int n, int k) => false;

    internal static bool TryGemm(
        float[] a,
        int aOffset,
        float[] b,
        int bOffset,
        float[] c,
        int cOffset,
        int m,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc) => false;

    internal static bool TryGemm(
        double[] a,
        int aOffset,
        double[] b,
        int bOffset,
        double[] c,
        int cOffset,
        int m,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc) => false;
}
#endif
