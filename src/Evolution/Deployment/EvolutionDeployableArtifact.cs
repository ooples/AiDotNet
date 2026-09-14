using System.Text;
using AiDotNet.Evolution.Programs;

namespace AiDotNet.Evolution.Deployment;

/// <summary>Application-owned exact applicability for deploying a program or trained model.</summary>
public sealed class EvolutionDeploymentEnvelope
{
    /// <summary>Creates an envelope from versioned runtime, device, compiler, data, workload and validation-protocol manifests.</summary>
    public EvolutionDeploymentEnvelope(string runtimeHash, string deviceHash, string compilerHash,
        string datasetHash, string workloadHash, string validationProtocolHash)
    {
        foreach (string hash in new[] { runtimeHash, deviceHash, compilerHash, datasetHash, workloadHash, validationProtocolHash })
            DeploymentEncoding.RequireHash(hash);
        RuntimeHash = runtimeHash; DeviceHash = deviceHash; CompilerHash = compilerHash;
        DatasetHash = datasetHash; WorkloadHash = workloadHash; ValidationProtocolHash = validationProtocolHash;
        Key = EvolutionHash.Combine(new[] { "consumer-deployment-envelope-v1", runtimeHash, deviceHash,
            compilerHash, datasetHash, workloadHash, validationProtocolHash });
    }
    /// <summary>Gets the runtime manifest digest.</summary>
    public string RuntimeHash { get; }
    /// <summary>Gets the device/driver manifest digest.</summary>
    public string DeviceHash { get; }
    /// <summary>Gets the compiler/toolchain manifest digest.</summary>
    public string CompilerHash { get; }
    /// <summary>Gets the data/preprocessing manifest digest.</summary>
    public string DatasetHash { get; }
    /// <summary>Gets the workload/dispatch contract digest.</summary>
    public string WorkloadHash { get; }
    /// <summary>Gets the independent validation protocol digest.</summary>
    public string ValidationProtocolHash { get; }
    /// <summary>Gets the exact combined applicability key.</summary>
    public string Key { get; }
}

/// <summary>The payload retained by a deployment artifact.</summary>
public enum EvolutionDeploymentArtifactKind
{
    /// <summary>Exact program source, language and codec metadata.</summary>
    Program,
    /// <summary>Serialized trained model state, not an AutoML hyperparameter specification.</summary>
    TrainedModel
}

/// <summary>Immutable deployable bytes; neither construction nor registration authorizes activation.</summary>
/// <remarks>Model factories and serializers are trusted application code. Hashes are integrity checks, not authentication.</remarks>
public sealed class EvolutionDeployableArtifact
{
    /// <summary>The maximum retained payload size; oversize models are refused, never truncated.</summary>
    public const int MaximumPayloadBytes = 64 * 1024 * 1024;
    private static readonly Encoding Utf8 = new UTF8Encoding(false, true);
    private readonly byte[] _payload;

    internal EvolutionDeployableArtifact(EvolutionDeploymentArtifactKind kind, string format, string typeIdentity,
        EvolutionDeploymentEnvelope envelope, byte[] payload)
    {
        if (!Enum.IsDefined(typeof(EvolutionDeploymentArtifactKind), kind)) throw new ArgumentOutOfRangeException(nameof(kind));
        DeploymentEncoding.RequireLabel(format, 256);
        DeploymentEncoding.RequireLabel(typeIdentity, 2048);
        Envelope = envelope ?? throw new ArgumentNullException(nameof(envelope));
        if (payload is null || payload.Length == 0 || payload.Length > MaximumPayloadBytes)
            throw new ArgumentException("Deployment payload must be nonempty and within its byte bound.", nameof(payload));
        Kind = kind; Format = format; TypeIdentity = typeIdentity;
        _payload = (byte[])payload.Clone();
        PayloadHash = DeploymentEncoding.Hash(_payload);
        Id = EvolutionHash.Combine(new[] { "consumer-deployable-artifact-v1", kind.ToString(), format, typeIdentity, envelope.Key, PayloadHash });
    }
    /// <summary>Gets the exact artifact identity, including applicability and serialization contract.</summary>
    public string Id { get; }
    /// <summary>Gets the exact serialized payload digest.</summary>
    public string PayloadHash { get; }
    /// <summary>Gets the payload kind.</summary>
    public EvolutionDeploymentArtifactKind Kind { get; }
    /// <summary>Gets the application-pinned serialization version.</summary>
    public string Format { get; }
    /// <summary>Gets the expected program codec or model runtime type; it is never used for reflection-based activation.</summary>
    public string TypeIdentity { get; }
    /// <summary>Gets the exact applicability envelope.</summary>
    public EvolutionDeploymentEnvelope Envelope { get; }
    /// <summary>Gets the payload size without copying its content.</summary>
    public int PayloadLength => _payload.Length;
    /// <summary>Returns an owned copy of the exact retained bytes.</summary>
    public byte[] CopyPayload() => (byte[])_payload.Clone();

    /// <summary>Packages an immutable program without normalizing or executing its source.</summary>
    public static EvolutionDeployableArtifact FromProgram(ProgramGenome program, EvolutionDeploymentEnvelope envelope)
    {
        var codec = new ProgramGenomeCodec();
        return new(EvolutionDeploymentArtifactKind.Program, codec.VersionHash, codec.Id, envelope,
            Utf8.GetBytes(codec.Serialize(program ?? throw new ArgumentNullException(nameof(program)))));
    }

    /// <summary>Packages the trained model's serialized state; the application must prevent concurrent mutation while serializing.</summary>
    public static EvolutionDeployableArtifact FromModel(IModelSerializer model, string serializationVersion,
        EvolutionDeploymentEnvelope envelope)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        return new(EvolutionDeploymentArtifactKind.TrainedModel, serializationVersion,
            model.GetType().AssemblyQualifiedName ?? throw new InvalidOperationException("The model type has no stable runtime identity."),
            envelope, model.Serialize());
    }

    /// <summary>Decodes exact canonical program bytes, without executing them.</summary>
    public ProgramGenome ReadProgram()
    {
        var codec = new ProgramGenomeCodec();
        if (Kind != EvolutionDeploymentArtifactKind.Program || Format != codec.VersionHash || TypeIdentity != codec.Id)
            throw new InvalidDataException("Artifact is not compatible with the program codec.");
        string payload = Utf8.GetString(_payload);
        ProgramGenome program = codec.Deserialize(payload);
        if (!string.Equals(codec.Serialize(program), payload, StringComparison.Ordinal))
            throw new InvalidDataException("Program artifact is not canonical.");
        return program;
    }

    /// <summary>Restores exact trained state into an application-selected model type; the caller owns the returned instance.</summary>
    public TModel RestoreModel<TModel>(Func<TModel> factory, string serializationVersion) where TModel : class, IModelSerializer
    {
        if (factory is null) throw new ArgumentNullException(nameof(factory));
        if (Kind != EvolutionDeploymentArtifactKind.TrainedModel || Format != serializationVersion)
            throw new InvalidDataException("Artifact model serialization contract differs.");
        TModel model = factory() ?? throw new InvalidOperationException("Model factory returned no instance.");
        try
        {
            if (model.GetType().AssemblyQualifiedName != TypeIdentity) throw new InvalidDataException("Model factory returned a different runtime type.");
            model.Deserialize(CopyPayload());
            return model;
        }
        catch (Exception loadError)
        {
            try { (model as IDisposable)?.Dispose(); }
            catch (Exception disposeError) { throw new AggregateException(loadError, disposeError); }
            throw;
        }
    }
}
