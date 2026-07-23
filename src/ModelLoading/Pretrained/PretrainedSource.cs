namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// A declarative description of where a pretrained model comes from and how to decode it.
/// Hand one to <c>AiModelBuilder.ConfigureModel(...)</c> and the builder resolves it to a
/// ready-to-use model — no loader classes to wire up yourself.
/// </summary>
/// <remarks>
/// <para>
/// Create one with a factory method and (optionally) refine it fluently:
/// </para>
/// <code>
/// // Hub repo id — downloaded and cached automatically:
/// PretrainedSource.HuggingFace("meta-llama/Llama-3.1-8B-Instruct")
///     .Revision("main")
///     .Dtype(PretrainedDType.BFloat16);
///
/// // Local checkpoints:
/// PretrainedSource.Safetensors("/models/llama");   // dir with config.json + *.safetensors
/// PretrainedSource.Onnx("/models/model.onnx");     // run the ONNX graph directly
/// PretrainedSource.Gguf("/models/model.gguf");     // llama.cpp checkpoint
/// </code>
/// <para><b>For Beginners:</b> This is just a small "recipe card" that says which model to load
/// and in what precision. It does not do any loading by itself — the builder reads the card and
/// does the work.
/// </para>
/// </remarks>
public sealed class PretrainedSource
{
    private PretrainedSource(PretrainedModelKind kind, string locator)
    {
        if (string.IsNullOrWhiteSpace(locator))
            throw new ArgumentException("Locator must be a non-empty repo id or path.", nameof(locator));

        Kind = kind;
        Locator = locator;
    }

    /// <summary>The kind of source (Hugging Face hub id, local safetensors dir, ONNX, or GGUF).</summary>
    public PretrainedModelKind Kind { get; }

    /// <summary>
    /// The source locator: a Hugging Face repo id for <see cref="PretrainedModelKind.HuggingFace"/>,
    /// otherwise a filesystem path (directory for safetensors, file for ONNX/GGUF).
    /// </summary>
    public string Locator { get; }

    /// <summary>
    /// The hub revision (branch, tag, or commit) for <see cref="PretrainedModelKind.HuggingFace"/>.
    /// Defaults to <c>"main"</c>. Ignored for local sources.
    /// </summary>
    public string Revision { get; private set; } = "main";

    /// <summary>The precision weights are decoded into. Defaults to <see cref="PretrainedDType.Auto"/>
    /// (keep the checkpoint's on-disk precision).</summary>
    public PretrainedDType DType { get; private set; } = PretrainedDType.Auto;

    /// <summary>
    /// Directory to cache downloaded hub files in. When <c>null</c>, a default per-user cache is
    /// used. Ignored for local sources.
    /// </summary>
    public string? CacheDirectory { get; private set; }

    /// <summary>
    /// Explicit architecture override (e.g. <c>"LlamaForCausalLM"</c>). When <c>null</c>, the
    /// architecture is read from the checkpoint's <c>config.json</c>. Rarely needed.
    /// </summary>
    public string? ArchitectureOverride { get; private set; }

    /// <summary>Creates a source pointing at a Hugging Face Hub repository id.</summary>
    /// <param name="repoId">The repo id, e.g. <c>meta-llama/Llama-3.1-8B-Instruct</c>.</param>
    public static PretrainedSource HuggingFace(string repoId) =>
        new(PretrainedModelKind.HuggingFace, repoId);

    /// <summary>Creates a source pointing at a local directory of safetensors weights + <c>config.json</c>.</summary>
    /// <param name="directory">Path to the model directory.</param>
    public static PretrainedSource Safetensors(string directory) =>
        new(PretrainedModelKind.Safetensors, directory);

    /// <summary>Creates a source pointing at a local ONNX graph, run directly through the ONNX runtime.</summary>
    /// <param name="path">Path to the <c>.onnx</c> file.</param>
    public static PretrainedSource Onnx(string path) =>
        new(PretrainedModelKind.Onnx, path);

    /// <summary>Creates a source pointing at a local GGUF checkpoint.</summary>
    /// <param name="path">Path to the <c>.gguf</c> file.</param>
    public static PretrainedSource Gguf(string path) =>
        new(PretrainedModelKind.Gguf, path);

    /// <summary>Sets the hub revision (branch, tag, or commit). Ignored for local sources.</summary>
    public PretrainedSource WithRevision(string revision)
    {
        if (string.IsNullOrWhiteSpace(revision))
            throw new ArgumentException("Revision must be non-empty.", nameof(revision));

        Revision = revision;
        return this;
    }

    /// <summary>Sets the precision weights are decoded into.</summary>
    public PretrainedSource Dtype(PretrainedDType dtype)
    {
        DType = dtype;
        return this;
    }

    /// <summary>Sets the directory used to cache downloaded hub files. Ignored for local sources.</summary>
    public PretrainedSource WithCacheDirectory(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
            throw new ArgumentException("Cache directory must be non-empty.", nameof(directory));

        CacheDirectory = directory;
        return this;
    }

    /// <summary>
    /// Overrides the architecture family instead of reading it from <c>config.json</c>
    /// (e.g. when a checkpoint's config omits or misreports its <c>architectures</c> entry).
    /// </summary>
    public PretrainedSource WithArchitecture(string architecture)
    {
        if (string.IsNullOrWhiteSpace(architecture))
            throw new ArgumentException("Architecture must be non-empty.", nameof(architecture));

        ArchitectureOverride = architecture;
        return this;
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"{Kind}:{Locator}" +
        (Kind == PretrainedModelKind.HuggingFace ? $"@{Revision}" : string.Empty) +
        (DType == PretrainedDType.Auto ? string.Empty : $" ({DType})");
}
