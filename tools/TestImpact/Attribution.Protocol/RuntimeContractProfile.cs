using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.TestImpact;

public enum RuntimeObserverSignals { NoneReported, Present }
public enum RuntimeInitializationStatus { Missing, Recorded, Conflicting }
public enum RuntimeCpuMode { Cpu, Other }
public sealed record RuntimeEnvironmentBinding(int Schema, string Fingerprint, RuntimeObserverSignals ObserverSignals);
public sealed record RuntimeCpuCompletion(RuntimeCpuMode Mode, int MaxDegreeOfParallelism);
public sealed record RuntimeInitializationBinding(RuntimeInitializationStatus Status, RuntimeEnvironmentBinding? Inputs,
    RuntimeCpuCompletion? Completion = null);
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
                    (result.Initialization.Inputs is not null || result.Initialization.Completion is not null)) ||
                (result.Initialization.Inputs is not null && !Valid(result.Initialization.Inputs)) ||
                (result.Initialization.Completion is RuntimeCpuCompletion completion && !Enum.IsDefined(completion.Mode)))
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

    private static bool Valid(RuntimeEnvironmentBinding? value) => value is not null && value.Schema == 1 &&
        Enum.IsDefined(value.ObserverSignals) && value.Fingerprint is { Length: 64 } &&
        value.Fingerprint.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f');
}
