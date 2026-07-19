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
    private readonly Dictionary<string, ExpertSlice> _expertSlices;

    // One per-expert weight: the e-th equal slice of a stacked GGUF expert tensor (ffn_*_exps.weight).
    private readonly struct ExpertSlice
    {
        public ExpertSlice(string ggufName, int index, int count)
        {
            GgufName = ggufName;
            Index = index;
            Count = count;
        }

        public string GgufName { get; }
        public int Index { get; }
        public int Count { get; }
    }

    private GgufModelSource(FileStream stream, GgufFile file, HuggingFaceConfig config,
        Dictionary<string, string> map, Dictionary<string, ExpertSlice> expertSlices)
    {
        _stream = stream;
        _file = file;
        Config = config;
        _hfToGguf = map;
        _expertSlices = expertSlices;
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
            var (config, map, experts) = BuildConfigAndMap(file);
            return new GgufModelSource(stream, file, config, map, experts);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames
    {
        get
        {
            var names = new List<string>(_hfToGguf.Count + _expertSlices.Count);
            names.AddRange(_hfToGguf.Keys);
            names.AddRange(_expertSlices.Keys);
            return names;
        }
    }

    /// <inheritdoc/>
    public double[] ReadAsDouble(string name)
    {
        if (_hfToGguf.TryGetValue(name, out var ggufName))
            return _file.ReadAsDouble(ggufName);
        if (_expertSlices.TryGetValue(name, out var slice))
            return ReadExpertSlice(slice);
        throw new ArgumentException($"Tensor '{name}' is not present in this GGUF model.", nameof(name));
    }

    // Reads one expert's weight as the index-th of `count` equal slices of the stacked GGUF expert tensor.
    // GGUF lays the stack out expert-major (ne = [in, out, n_expert]), so slice e is a contiguous run whose
    // shape/layout matches a standalone GGUF expert weight — the builder applies its own [out,in] transpose.
    private double[] ReadExpertSlice(ExpertSlice slice)
    {
        double[] all = _file.ReadAsDouble(slice.GgufName);
        if (all.Length % slice.Count != 0)
            throw new InvalidDataException(
                $"Stacked expert tensor '{slice.GgufName}' length {all.Length} is not divisible by expert count {slice.Count}.");
        int sliceSize = all.Length / slice.Count;
        var result = new double[sliceSize];
        Array.Copy(all, slice.Index * sliceSize, result, 0, sliceSize);
        return result;
    }

    /// <inheritdoc/>
    public void Dispose() => _stream.Dispose();

    private static (HuggingFaceConfig, Dictionary<string, string>, Dictionary<string, ExpertSlice>) BuildConfigAndMap(GgufFile file)
    {
        string arch = RequireString(file, "general.architecture");
        int hidden = RequireInt(file, $"{arch}.embedding_length");
        int layers = RequireInt(file, $"{arch}.block_count");
        int heads = RequireInt(file, $"{arch}.attention.head_count");
        int kvHeads = OptionalInt(file, $"{arch}.attention.head_count_kv") ?? heads;
        int ffn = RequireInt(file, $"{arch}.feed_forward_length");
        // Sparse mixture-of-experts: llama.cpp stores Mixtral under the generic "llama" arch (its stacked
        // expert tensors distinguish it) and Qwen2-MoE as "qwen2moe"; expert_count > 0 marks a MoE model.
        int expertCount = OptionalInt(file, $"{arch}.expert_count") ?? 0;
        string modelType = NormalizeModelType(arch, expertCount);
        // Norm epsilon: Llama-family uses RMSNorm (layer_norm_rms_epsilon); Cohere/StarCoder2 use plain
        // LayerNorm (layer_norm_epsilon). Read whichever the file carries so the correct eps reaches the builder.
        double normEps = OptionalDouble(file, $"{arch}.attention.layer_norm_rms_epsilon")
            ?? OptionalDouble(file, $"{arch}.attention.layer_norm_epsilon") ?? 1e-5;
        double ropeTheta = OptionalDouble(file, $"{arch}.rope.freq_base") ?? 10000.0;
        int maxPos = OptionalInt(file, $"{arch}.context_length") ?? 2048;
        // Gemma-family checkpoints declare an explicit per-head dimension (key_length) that differs from
        // hidden/heads (e.g. Gemma head_dim = 256); absent for models where head_dim == hidden/heads.
        int? headDim = OptionalInt(file, $"{arch}.attention.key_length");

        // Vocab: prefer explicit metadata, else the token-embedding table's second (n_vocab) dimension.
        int vocab = OptionalInt(file, $"{arch}.vocab_size")
            ?? VocabFromEmbedding(file, hidden)
            ?? throw new InvalidDataException("GGUF metadata does not specify a vocabulary size.");

        // llama.cpp ties the LM head when there is no separate output.weight tensor.
        bool hasOutput = TensorExists(file, "output.weight");

        JObject configJson;
        if (string.Equals(modelType, "dbrx", StringComparison.OrdinalIgnoreCase))
        {
            // DBRX uses a distinct nested config schema (d_model / n_heads with nested attn_config /
            // ffn_config); HuggingFaceConfig.Parse routes it to its DBRX branch. The norm epsilon slot is
            // fixed at 1e-5 there, so it is not carried across.
            configJson = new JObject
            {
                ["architectures"] = new JArray(),
                ["model_type"] = "dbrx",
                ["d_model"] = hidden,
                ["n_heads"] = heads,
                ["n_layers"] = layers,
                ["vocab_size"] = vocab,
                ["max_seq_len"] = maxPos,
                ["tie_word_embeddings"] = !hasOutput,
                ["attn_config"] = new JObject
                {
                    ["kv_n_heads"] = kvHeads,
                    ["rope_theta"] = ropeTheta,
                },
                ["ffn_config"] = new JObject
                {
                    ["moe_num_experts"] = expertCount,
                    ["moe_top_k"] = OptionalInt(file, $"{arch}.expert_used_count") ?? 0,
                    ["ffn_hidden_size"] = ffn,
                },
            };
        }
        else
        {
            configJson = new JObject
            {
                ["architectures"] = new JArray(),
                ["model_type"] = modelType,
                ["hidden_size"] = hidden,
                ["intermediate_size"] = ffn,
                ["num_hidden_layers"] = layers,
                ["num_attention_heads"] = heads,
                ["num_key_value_heads"] = kvHeads,
                ["vocab_size"] = vocab,
                ["rms_norm_eps"] = normEps,
                ["rope_theta"] = ropeTheta,
                ["max_position_embeddings"] = maxPos,
                ["tie_word_embeddings"] = !hasOutput,
                // hidden_act is informational only — every decoder builder fixes its own gate activation via
                // DecoderOptions, so it is left at the config default rather than guessed from the GGUF arch.
            };
            // Family-specific fields the builders actually consume: an explicit head dim (Gemma), the Gemma-2
            // attention/final logit soft-caps, and the Cohere logit multiplier. Only emit them when present so
            // HuggingFaceConfig keeps its own defaults (e.g. head_dim = hidden/heads) for models that omit them.
            if (headDim is { } hd)
                configJson["head_dim"] = hd;
            if (OptionalDouble(file, $"{arch}.attn_logit_softcapping") is { } attnCap)
                configJson["attn_logit_softcapping"] = attnCap;
            if (OptionalDouble(file, $"{arch}.final_logit_softcapping") is { } finalCap)
                configJson["final_logit_softcapping"] = finalCap;
            if (OptionalDouble(file, $"{arch}.logit_scale") is { } logitScale)
                configJson["logit_scale"] = logitScale;
            if (expertCount > 0)
            {
                configJson["num_local_experts"] = expertCount;
                if (OptionalInt(file, $"{arch}.expert_used_count") is { } used)
                    configJson["num_experts_per_tok"] = used;
                if (OptionalInt(file, $"{arch}.expert_feed_forward_length") is { } expFfn)
                    configJson["moe_intermediate_size"] = expFfn;
                if (OptionalInt(file, $"{arch}.expert_shared_feed_forward_length") is { } sharedFfn)
                    configJson["shared_expert_intermediate_size"] = sharedFfn;
            }
        }
        var config = HuggingFaceConfig.Parse(configJson.ToString());

        var (map, experts) = BuildTensorMap(file, modelType, layers, expertCount);
        return (config, map, experts);
    }

    // GGUF architecture names do not always match the Hugging Face model_type our registry resolves: Mixtral
    // ships under the generic "llama" arch (distinguished only by its expert tensors) and Qwen2-MoE under
    // "qwen2moe". All other families already share the name.
    private static string NormalizeModelType(string arch, int expertCount)
    {
        if (string.Equals(arch, "llama", StringComparison.OrdinalIgnoreCase) && expertCount > 0)
            return "mixtral";
        if (string.Equals(arch, "qwen2moe", StringComparison.OrdinalIgnoreCase))
            return "qwen2_moe";
        return arch;
    }

    // Maps the Hugging Face parameter names each decoder builder requests to GGUF tensor names, adding only
    // entries whose GGUF tensor is actually present (so the builders' tie/absence/optional checks stay correct).
    // The normalization-layer and feed-forward names are family-specific: Gemma-2 has four sandwiched norms,
    // StarCoder2 carries LayerNorm biases and a non-gated c_fc/c_proj MLP, while the LLaMA family (and Cohere's
    // single parallel norm) share the common names below.
    private static (Dictionary<string, string> map, Dictionary<string, ExpertSlice> experts) BuildTensorMap(
        GgufFile file, string modelType, int layers, int numExperts)
    {
        bool gemma2 = string.Equals(modelType, "gemma2", StringComparison.OrdinalIgnoreCase);
        bool starcoder2 = string.Equals(modelType, "starcoder2", StringComparison.OrdinalIgnoreCase);
        bool mixtral = string.Equals(modelType, "mixtral", StringComparison.OrdinalIgnoreCase);
        bool qwen2Moe = string.Equals(modelType, "qwen2_moe", StringComparison.OrdinalIgnoreCase);
        bool dbrx = string.Equals(modelType, "dbrx", StringComparison.OrdinalIgnoreCase);
        bool moe = numExperts > 0 && (mixtral || qwen2Moe || dbrx);

        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        var experts = new Dictionary<string, ExpertSlice>(StringComparer.Ordinal);
        AddIfPresent(file, map, "model.embed_tokens.weight", "token_embd.weight");
        AddIfPresent(file, map, "model.norm.weight", "output_norm.weight");
        AddIfPresent(file, map, "model.norm.bias", "output_norm.bias"); // StarCoder2 final LayerNorm bias
        AddIfPresent(file, map, "lm_head.weight", "output.weight");
        for (int i = 0; i < layers; i++)
        {
            string hf = $"model.layers.{i}.";
            string blk = $"blk.{i}.";

            // ---- normalization layers (family-specific) ----
            if (gemma2)
            {
                // Gemma-2 sandwiches attention and the FFN between two RMSNorms each.
                AddIfPresent(file, map, hf + "input_layernorm.weight", blk + "attn_norm.weight");
                AddIfPresent(file, map, hf + "post_attention_layernorm.weight", blk + "attn_post_norm.weight");
                AddIfPresent(file, map, hf + "pre_feedforward_layernorm.weight", blk + "ffn_norm.weight");
                AddIfPresent(file, map, hf + "post_feedforward_layernorm.weight", blk + "ffn_post_norm.weight");
            }
            else
            {
                // LLaMA family: input norm from attn_norm, post-attention norm from ffn_norm. Cohere has only
                // attn_norm (parallel residual), so the post-attention entry simply stays absent.
                AddIfPresent(file, map, hf + "input_layernorm.weight", blk + "attn_norm.weight");
                AddIfPresent(file, map, hf + "post_attention_layernorm.weight", blk + "ffn_norm.weight");
                if (starcoder2)
                {
                    // StarCoder2 uses biased LayerNorm.
                    AddIfPresent(file, map, hf + "input_layernorm.bias", blk + "attn_norm.bias");
                    AddIfPresent(file, map, hf + "post_attention_layernorm.bias", blk + "ffn_norm.bias");
                }
            }

            // ---- attention projections (weights + optional biases, all families) ----
            AddIfPresent(file, map, hf + "self_attn.q_proj.weight", blk + "attn_q.weight");
            AddIfPresent(file, map, hf + "self_attn.k_proj.weight", blk + "attn_k.weight");
            AddIfPresent(file, map, hf + "self_attn.v_proj.weight", blk + "attn_v.weight");
            AddIfPresent(file, map, hf + "self_attn.o_proj.weight", blk + "attn_output.weight");
            AddIfPresent(file, map, hf + "self_attn.q_proj.bias", blk + "attn_q.bias");
            AddIfPresent(file, map, hf + "self_attn.k_proj.bias", blk + "attn_k.bias");
            AddIfPresent(file, map, hf + "self_attn.v_proj.bias", blk + "attn_v.bias");
            AddIfPresent(file, map, hf + "self_attn.o_proj.bias", blk + "attn_output.bias");
            // Fused attention projection (Phi-3): FusedProjectionSource splits the fused HF name on read.
            AddIfPresent(file, map, hf + "self_attn.qkv_proj.weight", blk + "attn_qkv.weight");

            // ---- feed-forward (family-specific) ----
            if (starcoder2)
            {
                // Non-gated GELU MLP with biases: c_fc (up) then c_proj (down).
                AddIfPresent(file, map, hf + "mlp.c_fc.weight", blk + "ffn_up.weight");
                AddIfPresent(file, map, hf + "mlp.c_fc.bias", blk + "ffn_up.bias");
                AddIfPresent(file, map, hf + "mlp.c_proj.weight", blk + "ffn_down.weight");
                AddIfPresent(file, map, hf + "mlp.c_proj.bias", blk + "ffn_down.bias");
            }
            else if (moe)
            {
                // Sparse MoE FFN. GGUF stacks all experts into one 3D tensor per projection (ffn_*_exps) and
                // stores the router as ffn_gate_inp; each per-expert HF weight is one slice of the stack. The
                // HF names differ per family (Mixtral w1/w3/w2 under block_sparse_moe; Qwen2-MoE gate/up/down
                // under mlp), so dispatch the names but share the slicing.
                string routerHf = mixtral ? hf + "block_sparse_moe.gate.weight" : hf + "mlp.gate.weight";
                AddIfPresent(file, map, routerHf, blk + "ffn_gate_inp.weight");
                string expertsHf = mixtral ? hf + "block_sparse_moe.experts." : hf + "mlp.experts.";
                string gateName = mixtral ? ".w1.weight" : ".gate_proj.weight";
                string upName = mixtral ? ".w3.weight" : ".up_proj.weight";
                string downName = mixtral ? ".w2.weight" : ".down_proj.weight";
                for (int e = 0; e < numExperts; e++)
                {
                    AddExpertSlice(file, experts, expertsHf + e + gateName, blk + "ffn_gate_exps.weight", e, numExperts);
                    AddExpertSlice(file, experts, expertsHf + e + upName, blk + "ffn_up_exps.weight", e, numExperts);
                    AddExpertSlice(file, experts, expertsHf + e + downName, blk + "ffn_down_exps.weight", e, numExperts);
                }
                if (qwen2Moe)
                {
                    // Qwen2-MoE also has an always-on shared expert plus its sigmoid gate.
                    AddIfPresent(file, map, hf + "mlp.shared_expert.gate_proj.weight", blk + "ffn_gate_shexp.weight");
                    AddIfPresent(file, map, hf + "mlp.shared_expert.up_proj.weight", blk + "ffn_up_shexp.weight");
                    AddIfPresent(file, map, hf + "mlp.shared_expert.down_proj.weight", blk + "ffn_down_shexp.weight");
                    AddIfPresent(file, map, hf + "mlp.shared_expert_gate.weight", blk + "ffn_gate_inp_shexp.weight");
                }
            }
            else
            {
                // Gated SwiGLU/GeGLU MLP (LLaMA/Mistral/Qwen2/Gemma/Cohere), plus Phi-3's fused gate_up.
                AddIfPresent(file, map, hf + "mlp.gate_proj.weight", blk + "ffn_gate.weight");
                AddIfPresent(file, map, hf + "mlp.up_proj.weight", blk + "ffn_up.weight");
                AddIfPresent(file, map, hf + "mlp.down_proj.weight", blk + "ffn_down.weight");
                AddIfPresent(file, map, hf + "mlp.gate_up_proj.weight", blk + "ffn_gate_up.weight");
            }
        }

        return (map, experts);
    }

    // Registers one per-expert HF weight as slice <paramref name="index"/> of the stacked GGUF expert tensor,
    // provided the stack is present (so absence-tolerant loading still works if a checkpoint omits it).
    private static void AddExpertSlice(GgufFile file, Dictionary<string, ExpertSlice> experts,
        string hfName, string ggufStackedName, int index, int count)
    {
        if (TensorExists(file, ggufStackedName))
            experts[hfName] = new ExpertSlice(ggufStackedName, index, count);
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
