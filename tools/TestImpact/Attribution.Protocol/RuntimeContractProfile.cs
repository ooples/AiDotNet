using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.TestImpact;

public enum RuntimeObserverSignals { NoneReported, Present }
public enum RuntimeInitializationStatus { Missing, Recorded, Conflicting }
public enum RuntimeCpuMode { Cpu, Other }
public enum RuntimeCpuEntryMode { Other, PlainCpu, Missing, DerivedCpu }
public enum RuntimeCpuLogging { MayInvokeCallbacks, Suppressed }
public enum RuntimeGpuStartupPolicy { Unknown, AutoDetectionPermitted, Disabled }
public enum RuntimeGpuDiagnosticsPolicy { Unknown, NoDumpRequested, DumpRequested }
public sealed record RuntimeCpuResetInput(RuntimeCpuEntryMode Mode, RuntimeCpuLogging Logging);
public sealed record RuntimeEnvironmentBinding(int Schema, string Fingerprint, RuntimeObserverSignals ObserverSignals,
    RuntimeGpuStartupPolicy GpuStartup = RuntimeGpuStartupPolicy.Unknown,
    RuntimeGpuDiagnosticsPolicy GpuDiagnostics = RuntimeGpuDiagnosticsPolicy.Unknown);
public sealed record RuntimeCpuCompletion(RuntimeCpuMode Mode, int MaxDegreeOfParallelism);
public sealed record RuntimeInitializationBinding(RuntimeInitializationStatus Status, RuntimeEnvironmentBinding? Inputs,
    RuntimeCpuCompletion? Completion = null, RuntimeCpuResetInput? ResetInput = null);
public sealed record RuntimeContractProfile(RuntimeEnvironmentBinding Effective, RuntimeInitializationBinding Initialization);

public static class RuntimeProfileEvidence
{
    // A preimage is needed to check contract preconditions, not just compare an
    // opaque profile label. It remains consistency evidence until the emitting
    // runner binary and workflow have been authenticated independently.
    public static RuntimeContractProfile? Read(DiscoveryManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        if (manifest.ProfileJson is null) return null;
        if (manifest.Context is null || manifest.ProfileJson.Length > 65536 || Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(manifest.ProfileJson)))
            != manifest.Context.ProfileFingerprint)
            throw new EvidenceException(EvidenceFailure.Context, "Runtime profile preimage differs from the execution identity.");
        var options = new JsonSerializerOptions { AllowDuplicateProperties = false, RespectRequiredConstructorParameters = true,
            UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, MaxDepth = 32 };
        options.Converters.Add(new JsonStringEnumConverter(allowIntegerValues: false));
        try
        {
            JsonElement profile = JsonSerializer.Deserialize<JsonElement>(manifest.ProfileJson, options);
            if (profile.ValueKind != JsonValueKind.Object || !profile.TryGetProperty("RuntimeContracts", out JsonElement contracts))
                throw new JsonException("Missing runtime contract profile.");
            RuntimeContractProfile result = contracts.Deserialize<RuntimeContractProfile>(options)
                ?? throw new JsonException("Null runtime contract profile.");
            if (result.Initialization is null || !Enum.IsDefined(result.Initialization.Status) || !Valid(result.Effective) ||
                (result.Initialization.Status == RuntimeInitializationStatus.Recorded && result.Initialization.Inputs is null) ||
                (result.Initialization.Status == RuntimeInitializationStatus.Missing &&
                    (result.Initialization.Inputs is not null || result.Initialization.Completion is not null || result.Initialization.ResetInput is not null)) ||
                (result.Initialization.Inputs is not null && !Valid(result.Initialization.Inputs)) ||
                (result.Initialization.Completion is RuntimeCpuCompletion completion && !Enum.IsDefined(completion.Mode)) ||
                (result.Initialization.ResetInput is RuntimeCpuResetInput reset && (!Enum.IsDefined(reset.Mode) || !Enum.IsDefined(reset.Logging))))
                throw new JsonException("Invalid runtime contract observation.");
            return result;
        }
        catch (JsonException error) { throw new EvidenceException(EvidenceFailure.Format, error.Message); }
    }

    public static bool HasObservedCpuStartup(RuntimeContractProfile? profile) => profile is
    {
        Effective.ObserverSignals: RuntimeObserverSignals.NoneReported,
        Initialization: { Status: RuntimeInitializationStatus.Recorded,
            Inputs.ObserverSignals: RuntimeObserverSignals.NoneReported,
            Completion: { Mode: RuntimeCpuMode.Cpu, MaxDegreeOfParallelism: > 0 } }
    } && Valid(profile.Effective) && Valid(profile.Initialization.Inputs);

    // Observed entry conditions, not an atomic engine/environment snapshot or
    // proof of the reset's effects. Static/lifetime contracts must separately
    // rule out concurrent mutation. Legacy/missing entries inherit no fact.
    public static bool HasObservedCpuResetPreconditions(RuntimeContractProfile? profile) => HasObservedCpuStartup(profile) &&
        profile?.Initialization is
        {
            Inputs: { GpuStartup: RuntimeGpuStartupPolicy.Disabled, GpuDiagnostics: RuntimeGpuDiagnosticsPolicy.NoDumpRequested },
            ResetInput: { Mode: RuntimeCpuEntryMode.PlainCpu, Logging: RuntimeCpuLogging.Suppressed }
        };

    private static bool Valid(RuntimeEnvironmentBinding? value) => value is not null && value.Schema == 1 &&
        Enum.IsDefined(value.ObserverSignals) && Enum.IsDefined(value.GpuStartup) && Enum.IsDefined(value.GpuDiagnostics) && value.Fingerprint is { Length: 64 } &&
        value.Fingerprint.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f');
}
