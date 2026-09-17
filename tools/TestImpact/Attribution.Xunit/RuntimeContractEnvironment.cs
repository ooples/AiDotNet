using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;

namespace AiDotNet.TestImpact.Xunit;

// Inputs used by the bounded CPU/startup/file contracts. This is NOT a complete
// model of arbitrary process state. Unknown runtime effects still stay open.
internal static class RuntimeContractEnvironment
{
    private static readonly string[] Names =
    [
        "AIDOTNET_TEST_CPU_MDOP", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "AIDOTNET_QUIET", "AIDOTNET_DISABLE_GPU", "AIDOTNET_VERBOSE_INIT",
        "AIDOTNET_CACHEDB_MAXM", "AIDOTNET_ONEDNN_GEMM", "AIDOTNET_JIT_GEMM", "AIDOTNET_LAYERNORM_FUSED_FS64",
        "AIDOTNET_LN_FUSED_MAXBATCH", "AIDOTNET_LN_PARALLEL_MINWORK", "AIDOTNET_LN_PARALLEL_MINROWS",
        "AIDOTNET_LSTM_PARALLEL_MINROWS", "AIDOTNET_LSTM_PARALLEL", "AIDOTNET_COOP_POOL",
        "AIDOTNET_LICENSE_KEY", "AIDOTNET_LICENSE_TOKEN", "AIDOTNET_LICENSE_SCOPE", "AIDOTNET_BUILD_KEY", "AIDOTNET_LICENSE_SERVER_URL",
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TMP", "TEMP", "TMPDIR",
        "DOTNET_STARTUP_HOOKS", "CORECLR_ENABLE_PROFILING", "CORECLR_PROFILER", "CORECLR_PROFILER_PATH",
        "CORECLR_PROFILER_PATH_32", "CORECLR_PROFILER_PATH_64", "COR_ENABLE_PROFILING", "COR_PROFILER",
        "DOTNET_ENABLE_PROFILING", "DOTNET_PROFILER", "DOTNET_PROFILER_PATH",
        "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"
    ];

    internal static RuntimeEnvironmentBinding Capture() => Capture(Environment.GetEnvironmentVariable, Debugger.IsAttached);

    internal static RuntimeEnvironmentBinding Capture(Func<string, string?> read, bool debuggerAttached)
    {
        ArgumentNullException.ThrowIfNull(read);
        var values = Names.ToDictionary(name => name, read, StringComparer.Ordinal);
        bool observers = debuggerAttached || values.Any(pair =>
            (pair.Key == "DOTNET_STARTUP_HOOKS" || pair.Key.Contains("PROFILER", StringComparison.Ordinal)) &&
                !string.IsNullOrEmpty(pair.Value) ||
            pair.Key.EndsWith("ENABLE_PROFILING", StringComparison.Ordinal) && !string.IsNullOrEmpty(pair.Value) && pair.Value != "0");
        // Never serialize the actual environment values into an artifact: these
        // include license credentials. Only the combined digest leaves here.
        string fingerprint = Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(new
        {
            Schema = 1, DebuggerAttached = debuggerAttached,
            Inputs = values.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray()
        })));
        return new(1, fingerprint, observers ? RuntimeObserverSignals.Present : RuntimeObserverSignals.NoneReported);
    }
}
