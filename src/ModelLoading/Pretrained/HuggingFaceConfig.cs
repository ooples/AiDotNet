using Newtonsoft.Json.Linq;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// A strongly-typed view over a Hugging Face decoder <c>config.json</c> (the LLaMA / Mistral /
/// Qwen2 text-model family). Only the fields needed to reconstruct the architecture are read;
/// unknown fields are ignored so newer configs still parse.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Every Hugging Face model ships a small <c>config.json</c> that
/// describes its shape — how wide it is, how many layers, how many attention heads, and so on.
/// This class turns that JSON into typed properties AiDotNet can build a matching model from.
/// </para>
/// </remarks>
public sealed class HuggingFaceConfig
{
    /// <summary>The declared architecture classes, e.g. <c>["LlamaForCausalLM"]</c>. May be empty.</summary>
    public IReadOnlyList<string> Architectures { get; private init; } = Array.Empty<string>();

    /// <summary>The short model type, e.g. <c>"llama"</c>, <c>"mistral"</c>, <c>"qwen2"</c>.</summary>
    public string ModelType { get; private init; } = string.Empty;

    /// <summary>Model (residual stream) width — the embedding / hidden dimension.</summary>
    public int HiddenSize { get; private init; }

    /// <summary>Feed-forward inner dimension (the SwiGLU gate/up width).</summary>
    public int IntermediateSize { get; private init; }

    /// <summary>Number of stacked decoder blocks.</summary>
    public int NumHiddenLayers { get; private init; }

    /// <summary>Number of query attention heads.</summary>
    public int NumAttentionHeads { get; private init; }

    /// <summary>
    /// Number of key/value heads. Equals <see cref="NumAttentionHeads"/> for multi-head attention;
    /// smaller for grouped-query attention (GQA). Defaults to <see cref="NumAttentionHeads"/> when
    /// the config omits it.
    /// </summary>
    public int NumKeyValueHeads { get; private init; }

    /// <summary>Vocabulary size (embedding rows and LM-head columns).</summary>
    public int VocabSize { get; private init; }

    /// <summary>Per-head dimension. Uses the config's <c>head_dim</c> when present, else
    /// <see cref="HiddenSize"/> / <see cref="NumAttentionHeads"/>.</summary>
    public int HeadDim { get; private init; }

    /// <summary>RMSNorm epsilon.</summary>
    public double RmsNormEps { get; private init; }

    /// <summary>RoPE base frequency (<c>rope_theta</c>).</summary>
    public double RopeTheta { get; private init; }

    /// <summary>Maximum position (context length) the RoPE table was trained for.</summary>
    public int MaxPositionEmbeddings { get; private init; }

    /// <summary>Whether the LM head shares weights with the input embedding.</summary>
    public bool TieWordEmbeddings { get; private init; }

    /// <summary>Feed-forward activation name, e.g. <c>"silu"</c>.</summary>
    public string HiddenActivation { get; private init; } = "silu";

    /// <summary>On-disk weight precision, e.g. <c>"bfloat16"</c>. Empty when unspecified.</summary>
    public string TorchDtype { get; private init; } = string.Empty;

    /// <summary>Final-logit soft-cap magnitude (Gemma-2 <c>final_logit_softcapping</c>); null when absent.</summary>
    public double? FinalLogitSoftcapping { get; private init; }

    /// <summary>Logit multiplier (Cohere <c>logit_scale</c>); null when absent.</summary>
    public double? LogitScale { get; private init; }

    /// <summary>Number of MoE experts (<c>num_local_experts</c> or <c>num_experts</c>); 0 for a dense decoder.</summary>
    public int NumLocalExperts { get; private init; }

    /// <summary>Experts activated per token (<c>num_experts_per_tok</c>); 0 for a dense decoder.</summary>
    public int NumExpertsPerTok { get; private init; }

    /// <summary>Routed-expert inner dimension (<c>moe_intermediate_size</c>); 0 when not an MoE decoder that
    /// distinguishes it from <see cref="IntermediateSize"/>.</summary>
    public int MoeIntermediateSize { get; private init; }

    /// <summary>Shared-expert inner dimension (Qwen2-MoE <c>shared_expert_intermediate_size</c>); 0 when absent.</summary>
    public int SharedExpertIntermediateSize { get; private init; }

    /// <summary>Beginning-of-sequence token id, when declared.</summary>
    public int? BosTokenId { get; private init; }

    /// <summary>End-of-sequence token id, when declared.</summary>
    public int? EosTokenId { get; private init; }

    /// <summary>Parses a <c>config.json</c> from its JSON text.</summary>
    /// <param name="json">The raw JSON contents.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="json"/> is null/empty.</exception>
    /// <exception cref="InvalidDataException">Thrown when the JSON is malformed or a required
    /// dimension is missing or non-positive.</exception>
    public static HuggingFaceConfig Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            throw new ArgumentException("config.json content must be non-empty.", nameof(json));

        JObject root;
        try
        {
            root = JObject.Parse(json);
        }
        catch (Newtonsoft.Json.JsonException ex)
        {
            throw new InvalidDataException("config.json is not valid JSON: " + ex.Message);
        }

        var architectures = new List<string>();
        if (root["architectures"] is JArray archArray)
        {
            foreach (var token in archArray)
            {
                var name = (string?)token;
                if (name is not null && name.Trim().Length > 0)
                    architectures.Add(name);
            }
        }

        // DBRX uses a different config schema (d_model / n_heads / n_layers, with nested attn_config +
        // ffn_config); translate it to the common shape.
        string modelType = (string?)root["model_type"] ?? string.Empty;
        if (string.Equals(modelType, "dbrx", StringComparison.OrdinalIgnoreCase))
            return ParseDbrx(root, architectures);

        int hiddenSize = RequirePositive(root, "hidden_size");
        int numHeads = RequirePositive(root, "num_attention_heads");
        int numKvHeads = OptionalInt(root, "num_key_value_heads") ?? numHeads;
        int headDim = OptionalInt(root, "head_dim") ?? (hiddenSize / numHeads);

        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new InvalidDataException(
                $"num_key_value_heads ({numKvHeads}) must be a positive divisor of num_attention_heads ({numHeads}).");

        return new HuggingFaceConfig
        {
            Architectures = architectures,
            ModelType = (string?)root["model_type"] ?? string.Empty,
            HiddenSize = hiddenSize,
            IntermediateSize = RequirePositive(root, "intermediate_size"),
            NumHiddenLayers = RequirePositive(root, "num_hidden_layers"),
            NumAttentionHeads = numHeads,
            NumKeyValueHeads = numKvHeads,
            VocabSize = RequirePositive(root, "vocab_size"),
            HeadDim = headDim,
            RmsNormEps = OptionalDouble(root, "rms_norm_eps") ?? 1e-5,
            RopeTheta = OptionalDouble(root, "rope_theta") ?? 10000.0,
            MaxPositionEmbeddings = OptionalInt(root, "max_position_embeddings") ?? 2048,
            TieWordEmbeddings = (bool?)root["tie_word_embeddings"] ?? false,
            HiddenActivation = (string?)root["hidden_act"] ?? "silu",
            TorchDtype = (string?)root["torch_dtype"] ?? string.Empty,
            FinalLogitSoftcapping = OptionalDouble(root, "final_logit_softcapping"),
            LogitScale = OptionalDouble(root, "logit_scale"),
            NumLocalExperts = OptionalInt(root, "num_local_experts") ?? OptionalInt(root, "num_experts") ?? 0,
            NumExpertsPerTok = OptionalInt(root, "num_experts_per_tok") ?? 0,
            MoeIntermediateSize = OptionalInt(root, "moe_intermediate_size") ?? 0,
            SharedExpertIntermediateSize = OptionalInt(root, "shared_expert_intermediate_size") ?? 0,
            BosTokenId = OptionalInt(root, "bos_token_id"),
            EosTokenId = OptionalInt(root, "eos_token_id"),
        };
    }

    // Translates DBRX's schema (d_model / n_heads / n_layers + nested attn_config/ffn_config) to the common shape.
    private static HuggingFaceConfig ParseDbrx(JObject root, List<string> architectures)
    {
        int hidden = RequirePositive(root, "d_model");
        int numHeads = RequirePositive(root, "n_heads");
        var attn = root["attn_config"] as JObject;
        var ffn = root["ffn_config"] as JObject;
        int numKvHeads = (attn is not null ? OptionalInt(attn, "kv_n_heads") : null) ?? numHeads;
        int headDim = hidden / numHeads;
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new InvalidDataException(
                $"DBRX kv_n_heads ({numKvHeads}) must be a positive divisor of n_heads ({numHeads}).");

        int experts = (ffn is not null ? OptionalInt(ffn, "moe_num_experts") : null) ?? 0;
        int topK = (ffn is not null ? OptionalInt(ffn, "moe_top_k") : null) ?? 0;
        int ffnHidden = (ffn is not null ? OptionalInt(ffn, "ffn_hidden_size") : null)
            ?? throw new InvalidDataException("DBRX ffn_config.ffn_hidden_size is required.");

        return new HuggingFaceConfig
        {
            Architectures = architectures,
            ModelType = "dbrx",
            HiddenSize = hidden,
            IntermediateSize = ffnHidden,
            NumHiddenLayers = RequirePositive(root, "n_layers"),
            NumAttentionHeads = numHeads,
            NumKeyValueHeads = numKvHeads,
            VocabSize = RequirePositive(root, "vocab_size"),
            HeadDim = headDim,
            RmsNormEps = 1e-5, // DBRX uses LayerNorm; this slot carries the norm epsilon
            RopeTheta = (attn is not null ? OptionalDouble(attn, "rope_theta") : null) ?? 500000.0,
            MaxPositionEmbeddings = OptionalInt(root, "max_seq_len") ?? 2048,
            TieWordEmbeddings = (bool?)root["tie_word_embeddings"] ?? false,
            HiddenActivation = "silu",
            TorchDtype = (string?)root["torch_dtype"] ?? string.Empty,
            NumLocalExperts = experts,
            NumExpertsPerTok = topK,
            MoeIntermediateSize = ffnHidden,
            BosTokenId = OptionalInt(root, "bos_token_id"),
            EosTokenId = OptionalInt(root, "eos_token_id"),
        };
    }

    /// <summary>Reads and parses a <c>config.json</c> file from disk.</summary>
    /// <param name="path">Path to the <c>config.json</c> file.</param>
    public static HuggingFaceConfig FromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path must be non-empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"config.json not found: {path}", path);

        return Parse(File.ReadAllText(path));
    }

    private static int RequirePositive(JObject root, string key)
    {
        int? value = OptionalInt(root, key);
        if (value is null)
            throw new InvalidDataException($"config.json is missing required field '{key}'.");
        if (value.Value <= 0)
            throw new InvalidDataException($"config.json field '{key}' must be positive (was {value.Value}).");
        return value.Value;
    }

    private static int? OptionalInt(JObject root, string key)
    {
        var token = root[key];
        if (token is null || token.Type == JTokenType.Null)
            return null;
        return (int)token;
    }

    private static double? OptionalDouble(JObject root, string key)
    {
        var token = root[key];
        if (token is null || token.Type == JTokenType.Null)
            return null;
        return (double)token;
    }
}
