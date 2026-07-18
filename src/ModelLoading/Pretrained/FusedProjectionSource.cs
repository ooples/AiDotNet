using System.Collections.Generic;
using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Wraps a tensor source that stores <em>fused</em> projections (Phi-3: a single
/// <c>self_attn.qkv_proj</c> and <c>mlp.gate_up_proj</c> per layer) and exposes the individual
/// <c>q_proj</c>/<c>k_proj</c>/<c>v_proj</c> and <c>gate_proj</c>/<c>up_proj</c> tensors that
/// <see cref="LlamaModelBuilder{T}"/> expects — slicing the fused matrix's output rows on read.
/// Any tensor already present under its split name (a non-fused checkpoint) passes straight through.
/// </summary>
/// <remarks>
/// Fused matrices are <c>[out, in]</c> row-major, so an output-row range maps to a contiguous element
/// range: <c>rows [r0, r1) → elements [r0*in, r1*in)</c>. QKV rows are stacked
/// <c>[Q(numHeads·headDim); K(numKVHeads·headDim); V(numKVHeads·headDim)]</c>; gate/up are
/// <c>[gate(intermediate); up(intermediate)]</c>.
/// </remarks>
public sealed class FusedProjectionSource : INamedTensorSource
{
    private const string QkvSuffix = "self_attn.qkv_proj.weight";
    private const string GateUpSuffix = "mlp.gate_up_proj.weight";

    private readonly INamedTensorSource _inner;
    private readonly int _hidden;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _numKVHeads;
    private readonly int _intermediate;
    private readonly HashSet<string> _underlying;
    private readonly HashSet<string> _names;

    /// <summary>Wraps <paramref name="inner"/>, deriving split-projection names from <paramref name="config"/>.</summary>
    public FusedProjectionSource(INamedTensorSource inner, HuggingFaceConfig config)
    {
        Guard.NotNull(inner);
        Guard.NotNull(config);
        _inner = inner;
        _hidden = config.HiddenSize;
        _headDim = config.HeadDim;
        _numHeads = config.NumAttentionHeads;
        _numKVHeads = config.NumKeyValueHeads;
        _intermediate = config.IntermediateSize;

        _underlying = new HashSet<string>(inner.TensorNames, StringComparer.Ordinal);
        _names = new HashSet<string>(_underlying, StringComparer.Ordinal);
        foreach (var n in _underlying)
        {
            if (n.EndsWith(QkvSuffix, StringComparison.Ordinal))
            {
                var pfx = n.Substring(0, n.Length - "qkv_proj.weight".Length);
                _names.Add(pfx + "q_proj.weight");
                _names.Add(pfx + "k_proj.weight");
                _names.Add(pfx + "v_proj.weight");
            }
            else if (n.EndsWith(GateUpSuffix, StringComparison.Ordinal))
            {
                var pfx = n.Substring(0, n.Length - "gate_up_proj.weight".Length);
                _names.Add(pfx + "gate_proj.weight");
                _names.Add(pfx + "up_proj.weight");
            }
        }
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames => _names;

    /// <inheritdoc/>
    public double[] ReadAsDouble(string name)
    {
        // A split name that the underlying source already provides directly (non-fused checkpoint) — passthrough.
        if (_underlying.Contains(name))
            return _inner.ReadAsDouble(name);

        if (TrySlice(name, out string fused, out int start, out int count))
        {
            var full = _inner.ReadAsDouble(fused);
            if (start + count > full.Length)
                throw new InvalidDataException(
                    $"fused tensor '{fused}' has {full.Length} elements; cannot take slice [{start},{start + count}).");
            var result = new double[count];
            Array.Copy(full, start, result, 0, count);
            return result;
        }

        throw new ArgumentException($"Tensor '{name}' is not present (directly or as a fused-projection slice).", nameof(name));
    }

    private bool TrySlice(string name, out string fused, out int start, out int count)
    {
        fused = string.Empty; start = 0; count = 0;
        int qRows = _numHeads * _headDim, kvRows = _numKVHeads * _headDim;

        if (TryReplaceSuffix(name, "self_attn.q_proj.weight", "self_attn.qkv_proj.weight", out fused))
        { start = 0; count = qRows * _hidden; return true; }
        if (TryReplaceSuffix(name, "self_attn.k_proj.weight", "self_attn.qkv_proj.weight", out fused))
        { start = qRows * _hidden; count = kvRows * _hidden; return true; }
        if (TryReplaceSuffix(name, "self_attn.v_proj.weight", "self_attn.qkv_proj.weight", out fused))
        { start = (qRows + kvRows) * _hidden; count = kvRows * _hidden; return true; }
        if (TryReplaceSuffix(name, "mlp.gate_proj.weight", "mlp.gate_up_proj.weight", out fused))
        { start = 0; count = _intermediate * _hidden; return true; }
        if (TryReplaceSuffix(name, "mlp.up_proj.weight", "mlp.gate_up_proj.weight", out fused))
        { start = _intermediate * _hidden; count = _intermediate * _hidden; return true; }

        return false;
    }

    private static bool TryReplaceSuffix(string name, string suffix, string replacement, out string result)
    {
        if (name.EndsWith(suffix, StringComparison.Ordinal))
        {
            result = name.Substring(0, name.Length - suffix.Length) + replacement;
            return true;
        }
        result = string.Empty;
        return false;
    }
}
