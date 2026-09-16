using AttributionRuntime;
using Xunit.Abstractions;
using Xunit.Sdk;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace AiDotNet.TestImpact.Xunit;

internal static class RunnerInvocation
{
    private static string? plannedBundle;
    private static string? plannedFingerprint;
    public static AttributionRunMode Mode
    {
        get
        {
            string? raw = Environment.GetEnvironmentVariable("ATTRIBUTION_MODE");
            if (string.IsNullOrEmpty(raw)) return AttributionRunMode.Collect;
            return Enum.TryParse(raw, out AttributionRunMode mode) && Enum.IsDefined(mode)
                ? mode : throw new InvalidDataException("Unknown attribution execution mode.");
        }
    }

    public static IXunitTestCase[] Resolve(IXunitTestCase[] inventory, ITestFrameworkExecutionOptions options)
    {
        string? planPath = Environment.GetEnvironmentVariable("ATTRIBUTION_PLAN");
        if (Mode == AttributionRunMode.Collect)
        {
            if (!string.IsNullOrEmpty(planPath)) throw new InvalidDataException("A plan cannot be silently ignored in collection mode.");
            return inventory;
        }
        if (inventory.Length == 0 || inventory[0].TestMethod.TestClass.Class is not IReflectionTypeInfo type)
            throw new InvalidDataException("Cannot discover/plan an empty or non-reflection workload.");
        string bundle = Path.GetDirectoryName(type.Type.Assembly.Location) ?? throw new InvalidDataException("Missing binary bundle.");
        RunnerBinding.RequireOutsideBundle(Required("ATTRIBUTION_OUTPUT"), bundle);
        var effectiveProfile = new { DeclaredProfile = Required("ATTRIBUTION_PROFILE_HASH"),
            DiscoveryPolicy = AttributionDiscoveryPolicy.DeferredTheories,
            Runtime = RuntimeInformation.FrameworkDescription, RuntimeVersion = Environment.Version.ToString(),
            OS = RuntimeInformation.OSDescription, Architecture = RuntimeInformation.ProcessArchitecture,
            Culture = CultureInfo.CurrentCulture.Name, UICulture = CultureInfo.CurrentUICulture.Name,
            CpuCount = Environment.ProcessorCount, Parallel = options.ParallelAlgorithmOrDefault(),
            DisableParallel = options.DisableParallelizationOrDefault(), MaxThreads = options.MaxParallelThreadsOrDefault(),
            StopOnFailure = options.StopOnTestFailOrDefault(), SyncMessages = options.SynchronousMessageReportingOrDefault(),
            Diagnostics = options.DiagnosticMessagesOrDefault(), LiveOutput = options.ShowLiveOutputOrDefault() };
        var context = new ExecutionContextIdentity(Required("ATTRIBUTION_SOURCE_TREE"), RunnerBinding.HashBundle(bundle),
            Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(effectiveProfile))));
        string workload = Required("ATTRIBUTION_WORKLOAD");
        TestCaseIdentity[] cases = inventory.Select(test => new TestCaseIdentity(test.UniqueID,
            AttributionTestFramework.Owner(test))).ToArray();
        if (Mode == AttributionRunMode.Discover)
        {
            if (!string.IsNullOrEmpty(planPath)) throw new InvalidDataException("Discovery must not contain an execution plan.");
            string path = Required("ATTRIBUTION_INVENTORY");
            RunnerBinding.RequireOutsideBundle(path, bundle);
            // Validate identities/context but do not turn discovery into a passing
            // execution. The empty runner emits no passing cases or receipt.
            _ = ExecutionEvidence.CreatePlan(workload, cases, [], ValidationScope.FullWorkload, context);
            RunnerBinding.WriteNew(path, new DiscoveryManifest(1, workload, context, cases));
            return [];
        }
        planPath = Required("ATTRIBUTION_PLAN");
        RunnerBinding.RequireOutsideBundle(planPath, bundle);
        ExecutionPlan requested = ExecutionEvidence.ReadPlan(File.ReadAllText(planPath));
        ExecutionPlan validated = ExecutionEvidence.ValidatePlan(requested, cases, workload, context);
        Tracker.ConfigurePlan(validated);
        plannedBundle = bundle;
        plannedFingerprint = context.BuildFingerprint;
        HashSet<string> selected = validated.RequiredCases.Select(test => test.CaseId).ToHashSet(StringComparer.Ordinal);
        return inventory.Where(test => selected.Contains(test.UniqueID)).ToArray();
    }

    public static void CheckBundleAfterExecution()
    {
        if (plannedBundle is not null && RunnerBinding.HashBundle(plannedBundle) != plannedFingerprint)
            Tracker.InvalidateExecutionContext();
    }

    private static string Required(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value
        ? value : throw new InvalidDataException($"Missing {name}.");
}
