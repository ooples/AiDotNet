using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Agentic.Models.Local;
using Newtonsoft.Json.Linq;
// GgufFile also exists in AiDotNet.Tensors.NumericOperations (pulled in via a global using); this alias
// pins every reference in this file to the Agentic safetensors/GGUF reader's type.
using GgufFile = AiDotNet.Agentic.Models.Local.GgufFile;
using GgufReader = AiDotNet.Agentic.Models.Local.GgufReader;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Opens a GGUF (llama.cpp) checkpoint as a decoder-ready source: it reads the architecture from the GGUF
/// metadata into a <see cref="HuggingFaceConfig"/> and exposes the weights under the Hugging Face parameter
/// names <see cref="LlamaModelBuilder{T}"/> expects, translating each read to the GGUF tensor name
/// (<c>token_embd.weight</c>, <c>blk.N.attn_q.weight</c>, …).
/// </summary>
/// <remarks>
/// <para>
/// GGUF stores linear weights <c>[out, in]</c> row-major — identical to Hugging Face safetensors — so no
/// re-layout is needed; the builder's usual <c>[out,in]→[in,out]</c> transpose applies unchanged. Quantized
/// tensors (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) and F16/F32 are dequantized on read by <see cref="GgufFile"/>.
/// Dispose the source to close the file handle.
/// </para>
/// <para><b>For Beginners:</b> GGUF is the format llama.cpp uses. This class understands a GGUF file well
/// enough to rebuild the same model inside AiDotNet and serve it.
/// </para>
/// </remarks>
public sealed class GgufModelSource : INamedTensorSource, IDisposable
{
    private readonly FileStream _stream;
    private readonly GgufFile _file;
    private readonly Dictionary<string, string> _hfToGguf;

    private GgufModelSource(FileStream stream, GgufFile file, HuggingFaceConfig config, Dictionary<string, string> map)
    {
        _stream = stream;
        _file = file;
        Config = config;
        _hfToGguf = map;
    }

    /// <summary>The architecture read from the GGUF metadata.</summary>
    public HuggingFaceConfig Config { get; }

    /// <summary>Opens a GGUF file and prepares its config + Hugging Face name mapping.</summary>
    /// <param name="path">Path to the <c>.gguf</c> file.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null/empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when required architecture metadata is missing.</exception>
    public static GgufModelSource Open(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path must be non-empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"GGUF file not found: {path}", path);

        var stream = File.OpenRead(path);
        try
        {
            var file = GgufReader.Read(stream);
            var (config, map) = BuildConfigAndMap(file);
            return new GgufModelSource(stream, file, config, map);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames => _hfToGguf.Keys;

    /// <inheritdoc/>
    public double[] ReadAsDouble(string name)
    {
        if (!_hfToGguf.TryGetValue(name, out var ggufName))
            throw new ArgumentException($"Tensor '{name}' is not present in this GGUF model.", nameof(name));
        return _file.ReadAsDouble(ggufName);
    }

    /// <inheritdoc/>
    public void Dispose() => _stream.Dispose();

    private static (HuggingFaceConfig, Dictionary<string, string>) BuildConfigAndMap(GgufFile file)
    {
        string arch = RequireString(file, "general.architecture");
        int hidden = RequireInt(file, $"{arch}.embedding_length");
        int layers = RequireInt(file, $"{arch}.block_count");
        int heads = RequireInt(file, $"{arch}.attention.head_count");
        int kvHeads = OptionalInt(file, $"{arch}.attention.head_count_kv") ?? heads;
        int ffn = RequireInt(file, $"{arch}.feed_forward_length");
        double rmsEps = OptionalDouble(file, $"{arch}.attention.layer_norm_rms_epsilon") ?? 1e-5;
        double ropeTheta = OptionalDouble(file, $"{arch}.rope.freq_base") ?? 10000.0;
        int maxPos = OptionalInt(file, $"{arch}.context_length") ?? 2048;

        // Vocab: prefer explicit metadata, else the token-embedding table's second (n_vocab) dimension.
        int vocab = OptionalInt(file, $"{arch}.vocab_size")
            ?? VocabFromEmbedding(file, hidden)
            ?? throw new InvalidDataException("GGUF metadata does not specify a vocabulary size.");

        // llama.cpp ties the LM head when there is no separate output.weight tensor.
        bool hasOutput = TensorExists(file, "output.weight");

        var configJson = new JObject
        {
            ["architectures"] = new JArray(),
            ["model_type"] = arch,
            ["hidden_size"] = hidden,
            ["intermediate_size"] = ffn,
            ["num_hidden_layers"] = layers,
            ["num_attention_heads"] = heads,
            ["num_key_value_heads"] = kvHeads,
            ["vocab_size"] = vocab,
            ["rms_norm_eps"] = rmsEps,
            ["rope_theta"] = ropeTheta,
            ["max_position_embeddings"] = maxPos,
            ["tie_word_embeddings"] = !hasOutput,
            ["hidden_act"] = "silu",
        };
        var config = HuggingFaceConfig.Parse(configJson.ToString());

        // Map the Hugging Face parameter names the builder requests to GGUF tensor names, adding only entries
        // whose GGUF tensor is actually present (so the builder's tie/absence checks stay correct).
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        AddIfPresent(file, map, "model.embed_tokens.weight", "token_embd.weight");
        AddIfPresent(file, map, "model.norm.weight", "output_norm.weight");
        AddIfPresent(file, map, "lm_head.weight", "output.weight");
        for (int i = 0; i < layers; i++)
        {
            string hf = $"model.layers.{i}.";
            string blk = $"blk.{i}.";
            AddIfPresent(file, map, hf + "input_layernorm.weight", blk + "attn_norm.weight");
            AddIfPresent(file, map, hf + "post_attention_layernorm.weight", blk + "ffn_norm.weight");
            AddIfPresent(file, map, hf + "self_attn.q_proj.weight", blk + "attn_q.weight");
            AddIfPresent(file, map, hf + "self_attn.k_proj.weight", blk + "attn_k.weight");
            AddIfPresent(file, map, hf + "self_attn.v_proj.weight", blk + "attn_v.weight");
            AddIfPresent(file, map, hf + "self_attn.o_proj.weight", blk + "attn_output.weight");
            AddIfPresent(file, map, hf + "mlp.gate_proj.weight", blk + "ffn_gate.weight");
            AddIfPresent(file, map, hf + "mlp.up_proj.weight", blk + "ffn_up.weight");
            AddIfPresent(file, map, hf + "mlp.down_proj.weight", blk + "ffn_down.weight");
            // Fused projections (Phi-3): expose the fused HF names so FusedProjectionSource can split them.
            AddIfPresent(file, map, hf + "self_attn.qkv_proj.weight", blk + "attn_qkv.weight");
            AddIfPresent(file, map, hf + "mlp.gate_up_proj.weight", blk + "ffn_gate_up.weight");
        }

        return (config, map);
    }

    private static void AddIfPresent(GgufFile file, Dictionary<string, string> map, string hfName, string ggufName)
    {
        if (TensorExists(file, ggufName))
            map[hfName] = ggufName;
    }

    private static bool TensorExists(GgufFile file, string name)
    {
        foreach (var n in file.TensorNames)
            if (string.Equals(n, name, StringComparison.Ordinal))
                return true;
        return false;
    }

    private static int? VocabFromEmbedding(GgufFile file, int hidden)
    {
        var info = file.Get("token_embd.weight");
        if (info is null || info.Dimensions.Count < 2) return null;
        // ggml dims: ne[0]=embedding_length, ne[1]=n_vocab.
        long vocab = info.Dimensions[info.Dimensions.Count - 1];
        return vocab > 0 && vocab <= int.MaxValue ? (int)vocab : null;
    }

    private static string RequireString(GgufFile file, string key)
    {
        var v = file.GetMetadata(key);
        if (v is string s && s.Length > 0) return s;
        throw new InvalidDataException($"GGUF metadata is missing required string '{key}'.");
    }

    private static int RequireInt(GgufFile file, string key) =>
        OptionalInt(file, key) ?? throw new InvalidDataException($"GGUF metadata is missing required integer '{key}'.");

    // GGUF metadata numbers are boxed as various integer/float CLR types; normalize through Convert.
    private static int? OptionalInt(GgufFile file, string key)
    {
        var v = file.GetMetadata(key);
        if (v is null) return null;
        try { return Convert.ToInt32(v, CultureInfo.InvariantCulture); }
        catch { return null; }
    }

    private static double? OptionalDouble(GgufFile file, string key)
    {
        var v = file.GetMetadata(key);
        if (v is null) return null;
        try { return Convert.ToDouble(v, CultureInfo.InvariantCulture); }
        catch { return null; }
    }
}
