using System.Collections.Generic;
using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Adapts a DBRX checkpoint's tensors to the Hugging Face parameter names the pretrained builders expect.
/// DBRX names blocks <c>transformer.blocks.{i}.*</c>, fuses q/k/v into a single <c>attn.Wqkv</c>, and stacks
/// all experts' gate/up/down into single tensors (<c>ffn.experts.mlp.w1/v1/w2</c>); this source renames them,
/// slices each expert out, and transposes the down projection so the standard loaders can consume them.
/// </summary>
/// <remarks>
/// The stacked expert tensors are <c>[num_experts · ffn, hidden]</c> row-major, so expert <c>e</c> occupies
/// the element range <c>[e·ffn·hidden, (e+1)·ffn·hidden)</c>. <c>w1</c>/<c>v1</c> are <c>[out=ffn, in=hidden]</c>
/// (exposed as-is for the usual <c>[out,in]→[in,out]</c> load transpose); <c>w2</c> is stored <c>[ffn, hidden]</c>
/// (i.e. already <c>[in,out]</c>), so it is transposed to <c>[hidden, ffn]</c> here to match the down-projection
/// convention. The fused Wqkv is exposed as <c>self_attn.qkv_proj.weight</c> for a wrapping
/// <see cref="FusedProjectionSource"/> to split.
/// </remarks>
public sealed class DbrxTensorSource : INamedTensorSource
{
    // hfName -> (sourceTensor, expertSlice: -1 = whole tensor, transposeFfnHidden)
    private readonly Dictionary<string, (string Source, int Expert, bool Transpose)> _map;
    private readonly INamedTensorSource _inner;
    private readonly int _hidden;
    private readonly int _ffn;

    /// <summary>Wraps a raw DBRX tensor source, deriving the Hugging Face name mapping from <paramref name="config"/>.</summary>
    public DbrxTensorSource(INamedTensorSource inner, HuggingFaceConfig config)
    {
        Guard.NotNull(inner);
        Guard.NotNull(config);
        _inner = inner;
        _hidden = config.HiddenSize;
        _ffn = config.MoeIntermediateSize > 0 ? config.MoeIntermediateSize : config.IntermediateSize;
        int layers = config.NumHiddenLayers;
        int experts = config.NumLocalExperts;

        _map = new Dictionary<string, (string, int, bool)>(StringComparer.Ordinal)
        {
            ["model.embed_tokens.weight"] = ("transformer.wte.weight", -1, false),
            ["model.norm.weight"] = ("transformer.norm_f.weight", -1, false),
        };
        if (Has(inner, "lm_head.weight")) _map["lm_head.weight"] = ("lm_head.weight", -1, false);

        for (int i = 0; i < layers; i++)
        {
            string hf = $"model.layers.{i}.";
            string blk = $"transformer.blocks.{i}.";
            _map[hf + "input_layernorm.weight"] = (blk + "norm_attn_norm.norm_1.weight", -1, false);
            _map[hf + "post_attention_layernorm.weight"] = (blk + "norm_attn_norm.norm_2.weight", -1, false);
            _map[hf + "self_attn.qkv_proj.weight"] = (blk + "norm_attn_norm.attn.Wqkv.weight", -1, false);
            _map[hf + "self_attn.o_proj.weight"] = (blk + "norm_attn_norm.attn.out_proj.weight", -1, false);
            _map[hf + "mlp.gate.weight"] = (blk + "ffn.router.layer.weight", -1, false);
            for (int e = 0; e < experts; e++)
            {
                string ep = hf + $"mlp.experts.{e}.";
                _map[ep + "gate_proj.weight"] = (blk + "ffn.experts.mlp.w1", e, false);
                _map[ep + "up_proj.weight"] = (blk + "ffn.experts.mlp.v1", e, false);
                _map[ep + "down_proj.weight"] = (blk + "ffn.experts.mlp.w2", e, true); // [ffn,hidden] -> [hidden,ffn]
            }
        }
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames => _map.Keys;

    /// <inheritdoc/>
    public double[] ReadAsDouble(string name)
    {
        if (!_map.TryGetValue(name, out var m))
            throw new ArgumentException($"Tensor '{name}' is not present in this DBRX model.", nameof(name));

        var full = _inner.ReadAsDouble(m.Source);
        if (m.Expert < 0)
            return full;

        // Slice expert e's [ffn, hidden] block out of the stacked [num_experts*ffn, hidden] tensor.
        int block = _ffn * _hidden;
        int start = m.Expert * block;
        if (start + block > full.Length)
            throw new InvalidDataException($"stacked expert tensor '{m.Source}' too small for expert {m.Expert}.");

        if (!m.Transpose)
        {
            var slice = new double[block];
            Array.Copy(full, start, slice, 0, block);
            return slice;
        }

        // Transpose the sliced [ffn, hidden] block to [hidden, ffn] (down projection convention).
        var t = new double[block];
        for (int f = 0; f < _ffn; f++)
            for (int h = 0; h < _hidden; h++)
                t[h * _ffn + f] = full[start + f * _hidden + h];
        return t;
    }

    private static bool Has(INamedTensorSource source, string name)
    {
        foreach (var n in source.TensorNames)
            if (string.Equals(n, name, StringComparison.Ordinal))
                return true;
        return false;
    }
}
