using Newtonsoft.Json;
using System.Reflection;

/// <summary>
/// Optional phase boundaries consumed by the external measurement runner. Writes occur outside timed regions;
/// allocation diagnostics belong to separate runs and are never used as latency scores.
/// </summary>
internal static class BenchmarkPhaseTelemetry
{
    private static readonly string? Output = Environment.GetEnvironmentVariable("AIDOTNET_BENCH_PHASE_FILE");
    private static readonly bool Diagnostics = Environment.GetEnvironmentVariable("AIDOTNET_BENCH_DIAGNOSTICS") == "1";

    private static object EngineAllocators()
    {
        var type = typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<float>).Assembly
            .GetType("AiDotNet.Tensors.Engines.DirectGpu.BufferAllocationDiagnostics");
        var method = type?.GetMethod("Snapshot", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
        if (method is null) return new { coverage = "unavailable", reason = "Engine buffer diagnostics are not present in this build." };
        try { return method.Invoke(null, null) ?? new { coverage = "unavailable", reason = "Snapshot returned null." }; }
        catch (TargetInvocationException exception)
        {
            return new { coverage = "unavailable", reason = exception.InnerException?.Message ?? exception.Message };
        }
    }

    internal static void Mark(string model, string phase)
    {
        if (string.IsNullOrEmpty(Output)) return;
        var record = new
        {
            schema = 1,
            pid = Environment.ProcessId,
            unix_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            model,
            phase,
            diagnostics = Diagnostics,
            // These bytes include runtime and harness allocations. They are a diagnostic, not ATen storage bytes.
            runtime_managed_allocated_bytes = Diagnostics ? GC.GetTotalAllocatedBytes(true) : (long?)null,
            // Raw workspace-inclusive buffer requests are not mislabeled as complete ATen tensor-storage coverage.
            allocator_diagnostics = Diagnostics ? EngineAllocators() : null,
        };
        File.AppendAllText(Output, JsonConvert.SerializeObject(record) + Environment.NewLine);
    }
}
