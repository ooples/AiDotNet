using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// The HRE transformer-replacement block — a drop-in replacement for a standard
/// transformer encoder/decoder block that replaces self-attention with IMD attention
/// and the FFN with a spectral Hebbian filter pair. Preserves the shape contract
/// <c>[B, S, E] → [B, S, E]</c> so it can be stacked like a transformer block.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A standard transformer block is four pieces:
/// LayerNorm → MultiHeadAttention → residual → LayerNorm → FFN → residual.
/// This HRE block has exactly the same structure, but swaps the two "learnable"
/// pieces for spectral equivalents:
/// </para>
/// <para>
/// <c>
/// x → LayerNorm → SequenceAxisIMDAttention → x + attn_out  <br/>
///   → LayerNorm → SpectralGatingFFN        → x + ffn_out
/// </c>
/// </para>
/// <para>
/// The attention pattern emerges from wave interference of the input (no Q/K/V
/// projection matrices, no learnable attention weights), and the FFN uses spectral
/// Hebbian filters (no learnable dense matrices). The only learnable parameters are
/// the two LayerNorm scale/shift pairs and the two Hebbian spectral filters inside
/// the FFN.
/// </para>
/// </remarks>
public class HREBlock<T> : LayerBase<T>
{
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly SequenceAxisIMDAttention<T> _attention;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly SpectralGatingFFN<T> _ffn;

    private readonly int _seqLen;
    private readonly int _embedDim;

    /// <inheritdoc/>
    public override string LayerName => $"HREBlock_{_seqLen}s_{_embedDim}e";

    /// <inheritdoc/>
    public override int ParameterCount =>
        _norm1.ParameterCount + _attention.ParameterCount + _norm2.ParameterCount + _ffn.ParameterCount;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the spectral gating FFN (for external Hebbian updates during training).
    /// </summary>
    public SpectralGatingFFN<T> FFN => _ffn;

    /// <summary>
    /// Gets the sequence-axis IMD attention layer (for visualization / debugging).
    /// </summary>
    public SequenceAxisIMDAttention<T> Attention => _attention;

    /// <summary>
    /// Creates a new HRE block.
    /// </summary>
    /// <param name="seqLen">Sequence length S. Determines the carrier count for sequence-axis attention.</param>
    /// <param name="embedDim">Embedding dimension E. Must be a power of 2 for the FFN's spectral filters.</param>
    /// <param name="fftSize">FFT size for the sequence-axis IMD attention. Typical choice: ~4·S² to
    /// accommodate Sidon-set carrier allocation over S positions without IMD collisions.</param>
    /// <param name="hebbianLearningRate">Learning rate for the Hebbian filters inside the FFN.</param>
    /// <param name="antiHebbianAlpha">Anti-Hebbian decorrelation strength for the Hebbian filters.</param>
    public HREBlock(
        int seqLen,
        int embedDim,
        int fftSize,
        double hebbianLearningRate = 0.01,
        double antiHebbianAlpha = 0.5)
        : base([seqLen, embedDim], [seqLen, embedDim])
    {
        _seqLen = seqLen;
        _embedDim = embedDim;

        _norm1 = new LayerNormalizationLayer<T>(embedDim);
        _attention = new SequenceAxisIMDAttention<T>(seqLen, embedDim, fftSize);
        _norm2 = new LayerNormalizationLayer<T>(embedDim);
        _ffn = new SpectralGatingFFN<T>(embedDim, hebbianLearningRate, antiHebbianAlpha);
    }

    /// <summary>
    /// Forward pass: LayerNorm → IMD Attention → residual → LayerNorm → FFN → residual.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Pre-norm transformer variant (used by GPT-2+): norm before each sublayer
        var normed1 = _norm1.Forward(input);
        var attnOut = _attention.Forward(normed1);
        var afterAttn = Engine.TensorAdd(input, attnOut);

        var normed2 = _norm2.Forward(afterAttn);
        var ffnOut = _ffn.Forward(normed2);
        var afterFfn = Engine.TensorAdd(afterAttn, ffnOut);

        return afterFfn;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parts = new List<Vector<T>>
        {
            _norm1.GetParameters(),
            _attention.GetParameters(),
            _norm2.GetParameters(),
            _ffn.GetParameters(),
        };

        int total = 0;
        foreach (var p in parts) total += p.Length;

        var combined = new Vector<T>(total);
        int idx = 0;
        foreach (var p in parts)
        {
            for (int i = 0; i < p.Length; i++) combined[idx + i] = p[i];
            idx += p.Length;
        }
        return combined;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        // HRE block is Hebbian-trained; no backward-pass gradients.
        return new Vector<T>(ParameterCount);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _ffn.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        int norm1Count = _norm1.ParameterCount;
        var p1 = new Vector<T>(norm1Count);
        for (int i = 0; i < norm1Count; i++) p1[i] = parameters[idx + i];
        _norm1.SetParameters(p1);
        idx += norm1Count;

        // Attention has 0 parameters, skip
        idx += _attention.ParameterCount;

        int norm2Count = _norm2.ParameterCount;
        var p2 = new Vector<T>(norm2Count);
        for (int i = 0; i < norm2Count; i++) p2[i] = parameters[idx + i];
        _norm2.SetParameters(p2);
        idx += norm2Count;

        int ffnCount = _ffn.ParameterCount;
        var pFfn = new Vector<T>(ffnCount);
        for (int i = 0; i < ffnCount; i++) pFfn[i] = parameters[idx + i];
        _ffn.SetParameters(pFfn);
    }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_seqLen);
        writer.Write(_embedDim);
        _norm1.Serialize(writer);
        _attention.Serialize(writer);
        _norm2.Serialize(writer);
        _ffn.Serialize(writer);
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // seqLen
        _ = reader.ReadInt32(); // embedDim
        _norm1.Deserialize(reader);
        _attention.Deserialize(reader);
        _norm2.Deserialize(reader);
        _ffn.Deserialize(reader);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _norm1.ResetState();
        _attention.ResetState();
        _norm2.ResetState();
        _ffn.ResetState();
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _norm1.SetTrainingMode(isTraining);
        _attention.SetTrainingMode(isTraining);
        _norm2.SetTrainingMode(isTraining);
        _ffn.SetTrainingMode(isTraining);
    }
}
