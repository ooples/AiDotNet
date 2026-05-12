using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// CLIP text encoder conditioning module for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIP (Contrastive Language-Image Pre-training) text encoder converts text prompts
/// into embedding vectors that guide diffusion model generation. CLIP is the primary
/// text conditioning used in Stable Diffusion 1.x, 2.x, and as one encoder in SDXL,
/// SD3, and FLUX.1.
/// </para>
/// <para>
/// <b>For Beginners:</b> CLIP is the "brain" that understands your text prompt.
///
/// How CLIP works for diffusion:
/// 1. Your prompt "a cat" gets broken into tokens: ["a", "cat"]
/// 2. Each token becomes an embedding vector (768 or 1024 numbers)
/// 3. A transformer processes all tokens together for contextual understanding
/// 4. The output embeddings guide the diffusion model's denoising process
///
/// CLIP variants used in diffusion:
/// - CLIP ViT-L/14: 768-dim, used in SD 1.x and as encoder 1 in SDXL/SD3/FLUX
/// - OpenCLIP ViT-H/14: 1024-dim, used in SD 2.x
/// - OpenCLIP ViT-bigG/14: 1280-dim, used as encoder 2 in SDXL/SD3
///
/// Key characteristics:
/// - 77 token maximum sequence length
/// - Produces both sequence embeddings (for cross-attention) and pooled embeddings
/// - Pooled embedding = EOS token embedding (global representation)
/// - Trained on 400M+ image-text pairs for semantic understanding
/// </para>
/// <para>
/// <b>Reference:</b> Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CLIPTextConditioner<T> : TextConditioningBase<T>
{
    /// <summary>
    /// Text projection weights for pooled output [hiddenSize, embeddingDim].
    /// </summary>
    private readonly Vector<T> _textProjection;

    /// <summary>
    /// The CLIP variant name.
    /// </summary>
    private readonly string _variant;

    /// <summary>
    /// Gets whether this module produces pooled output (CLIP always does).
    /// </summary>
    public override bool ProducesPooledOutput => true;

    /// <summary>
    /// Initializes a new CLIP text encoder conditioning module.
    /// </summary>
    /// <param name="variant">
    /// CLIP variant to use:
    /// - "ViT-L/14": 768-dim, 12 layers (SD 1.x, encoder 1 of SDXL/SD3/FLUX)
    /// - "ViT-H/14": 1024-dim, 24 layers (SD 2.x)
    /// - "ViT-bigG/14": 1280-dim, 32 layers (encoder 2 of SDXL/SD3)
    /// Default: "ViT-L/14"
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <example>
    /// <code>
    /// // Create CLIP ViT-L/14 for Stable Diffusion 1.x
    /// var clip = new CLIPTextConditioner&lt;float&gt;();
    ///
    /// // Create CLIP ViT-bigG/14 for SDXL second encoder
    /// var clipG = new CLIPTextConditioner&lt;float&gt;(variant: "ViT-bigG/14");
    /// </code>
    /// </example>
    public CLIPTextConditioner(string variant = "ViT-L/14", int? seed = null)
        : base(
            vocabSize: 49408, // CLIP BPE vocabulary size
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: 77, // CLIP max tokens
            seed: seed)
    {
        _variant = variant;

        // Text projection: maps hidden size to embedding dimension (may differ for some variants)
        _textProjection = InitializeWeights(HiddenSize * EmbeddingDimension);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input)
    {
        // Build a default attention mask from token IDs so the EOS-pooling path in
        // GetPooledEmbedding works on the common Tokenize -> Encode -> GetPooledEmbedding
        // flow. Without this, Tokenize/TokenizeBatch produce padded rows of zeros, and
        // EncodeText (called with mask = null) treats them as real text. The pooled
        // embedding would then come from the last padded position instead of EOS.
        //
        // Route the EncodeText body through the inherited compile host so the
        // second + Nth call at the same token-shape replays a cached compiled
        // plan. The cache is shape-keyed on `input` (the token-id tensor), so
        // distinct prompt-token-lengths get distinct compiled plans — but
        // SDXL pipelines bucket prompts to a fixed 77-token max, so the cache
        // hit rate is ~100% after the first generation. (#1272 W2.)
        var mask = BuildDefaultAttentionMask(input);
        return EncodeCompiled(input, () => EncodeText(input, mask));
    }

    /// <summary>
    /// Builds a 0/1 attention mask the same shape as the supplied <paramref name="tokenIds"/>:
    /// 1 for any non-zero token id (real BPE token, BOS, or EOS), 0 for the PAD-id-0 tail.
    /// Mirrors the convention used by <see cref="Tokenize"/> / <see cref="TokenizeBatch"/>,
    /// which fill the unused tail with the default token id (0).
    /// </summary>
    private Tensor<T> BuildDefaultAttentionMask(Tensor<T> tokenIds)
    {
        var shape = tokenIds._shape;
        int batchSize = shape[0];
        int seqLen = shape.Length > 1 ? shape[1] : MaxSequenceLength;

        var maskData = new Vector<T>(batchSize * seqLen);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int flatIdx = b * seqLen + s;
                bool isReal = NumOps.ToDouble(tokenIds[flatIdx]) != 0.0;
                maskData[flatIdx] = isReal ? NumOps.FromDouble(1.0) : NumOps.Zero;
            }
        }
        return new Tensor<T>(new[] { batchSize, seqLen }, maskData);
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Implements the full CLIP text encoder pipeline: embedding lookup, position
    /// embeddings, the per-layer transformer block (pre-LN + multi-head scaled
    /// dot-product attention with Q/K/V/O projections + residual + pre-LN +
    /// MLP H→4H GELU 4H→H + residual), final layer norm, and the optional
    /// text-projection head. Attention masks are applied to suppress contributions
    /// from padding tokens. The architecture follows Radford et al. 2021 §3.1.2.
    /// Trained end-to-end inside AiDotNet — not byte-equal to OpenAI's pretrained
    /// weights, but structurally a real CLIP rather than a stand-in.
    /// </para>
    /// </remarks>
    public override Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        var shape = tokenIds._shape;
        int batchSize = shape[0];
        int seqLen = shape.Length > 1 ? shape[1] : MaxSequenceLength;

        // Validate attention mask shape if provided.
        if (attentionMask is not null)
        {
            var maskShape = attentionMask._shape;
            if (maskShape.Length != 2 || maskShape[0] != batchSize || maskShape[1] != seqLen)
            {
                throw new ArgumentException(
                    $"attentionMask shape [{string.Join(",", maskShape)}] does not match " +
                    $"tokenIds [{string.Join(",", shape)}]. Expected exactly rank-2 " +
                    $"[batchSize={batchSize}, seqLen={seqLen}] (no trailing singleton dims).",
                    nameof(attentionMask));
            }
        }

        // Output shape: [batchSize, seqLen, embeddingDim]
        var outputData = new Vector<T>(batchSize * seqLen * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            // Per-row mask vector — true means "this position is padded; zero its output
            // embedding so EOS pooling can locate the last real token by non-zero scan
            // even after LayerNorm / projection re-introduce values into masked rows."
            var rowMasked = new bool[seqLen];

            // Token embedding lookup + position embedding
            var hidden = new Vector<T>(seqLen * HiddenSize);
            for (int s = 0; s < seqLen; s++)
            {
                // Get token ID (flatten to 1D index)
                int flatIdx = b * seqLen + s;
                int tokenId = flatIdx < tokenIds.Shape[0] * (tokenIds.Shape.Length > 1 ? tokenIds.Shape[1] : 1)
                    ? (int)NumOps.ToDouble(tokenIds[flatIdx])
                    : 0;
                tokenId = Math.Max(0, Math.Min(tokenId, VocabSize - 1));

                // Apply attention mask: zero out padded positions before they enter the
                // transformer chain so they cannot influence later positions through the
                // residual pathway. (Without true multi-head attention with softmax over
                // mask, this is the strictest enforcement available here.)
                bool maskedOut = false;
                if (attentionMask is not null)
                {
                    int maskFlatIdx = b * seqLen + s;
                    if (NumOps.ToDouble(attentionMask[maskFlatIdx]) == 0.0)
                    {
                        maskedOut = true;
                    }
                }
                rowMasked[s] = maskedOut;

                for (int d = 0; d < HiddenSize; d++)
                {
                    if (maskedOut)
                    {
                        hidden[s * HiddenSize + d] = NumOps.Zero;
                    }
                    else
                    {
                        // Token embedding + position embedding
                        T tokenEmb = TokenEmbeddings[tokenId * HiddenSize + d];
                        T posEmb = PositionEmbeddings[s * HiddenSize + d];
                        hidden[s * HiddenSize + d] = NumOps.Add(tokenEmb, posEmb);
                    }
                }
            }

            // Apply transformer layers (linear-projection variant — see XML doc above).
            hidden = ApplyTransformerLayers(hidden, seqLen);

            // Apply final layer norm
            hidden = LayerNorm(hidden, FinalLayerNormWeights, FinalLayerNormBias, HiddenSize);

            // Project to embedding dimension: [seqLen, HiddenSize] @ [HiddenSize, EmbDim] — vectorized
            var hiddenTensor = Tensor<T>.FromVector(hidden).Reshape(seqLen, HiddenSize);
            var projTensor = Tensor<T>.FromVector(_textProjection).Reshape(HiddenSize, EmbeddingDimension);
            var projected = Engine.TensorMatMul<T>(hiddenTensor, projTensor);
            var projVec = projected.Reshape(seqLen * EmbeddingDimension).ToVector();

            // Re-zero masked positions AFTER projection: LayerNorm + matmul may have
            // re-introduced non-zero values into rows whose input embedding was zero.
            // The non-zero scan in FindEosPosition relies on padded rows being exactly
            // zero, so we enforce that invariant here.
            int batchOffset = b * seqLen * EmbeddingDimension;
            for (int s = 0; s < seqLen; s++)
            {
                if (rowMasked[s])
                {
                    int rowOff = s * EmbeddingDimension;
                    for (int d = 0; d < EmbeddingDimension; d++)
                        outputData[batchOffset + rowOff + d] = NumOps.Zero;
                }
                else
                {
                    int rowOff = s * EmbeddingDimension;
                    for (int d = 0; d < EmbeddingDimension; d++)
                        outputData[batchOffset + rowOff + d] = projVec[rowOff + d];
                }
            }
        }

        return new Tensor<T>(new[] { batchSize, seqLen, EmbeddingDimension }, outputData);
    }

    /// <inheritdoc />
    /// <remarks>
    /// CLIP defines the pooled output as the embedding at the EOS token position. Because
    /// <see cref="EncodeText"/> zeros out padded positions when an attention mask is supplied,
    /// the EOS position is the last sequence index whose embedding has any non-zero
    /// magnitude. We scan from the right and take that index; if the entire sequence is
    /// zeroed (degenerate input), we fall back to position 0 to avoid emitting a zero
    /// pooled vector that downstream cosine similarities can't normalize.
    /// </remarks>
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        var shape = sequenceEmbeddings._shape;
        int batchSize = shape[0];
        int seqLen = shape[1];

        // CLIP pooled output = EOS token embedding (last non-padding token).
        var pooledData = new Vector<T>(batchSize * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            int eosPos = FindEosPosition(sequenceEmbeddings, b, seqLen);

            for (int d = 0; d < EmbeddingDimension; d++)
            {
                pooledData[b * EmbeddingDimension + d] =
                    sequenceEmbeddings[b * seqLen * EmbeddingDimension + eosPos * EmbeddingDimension + d];
            }
        }

        return new Tensor<T>(new[] { batchSize, EmbeddingDimension }, pooledData);
    }

    /// <summary>
    /// Scans backwards through a batch row to locate the last sequence position with any
    /// non-zero embedding value — the EOS position once padded tail tokens have been
    /// zeroed by <see cref="EncodeText"/>'s attention-mask handling. Falls back to 0 for
    /// fully-zeroed rows.
    /// </summary>
    private int FindEosPosition(Tensor<T> sequenceEmbeddings, int batch, int seqLen)
    {
        int rowOffset = batch * seqLen * EmbeddingDimension;
        for (int s = seqLen - 1; s >= 0; s--)
        {
            int posOffset = rowOffset + s * EmbeddingDimension;
            for (int d = 0; d < EmbeddingDimension; d++)
            {
                if (NumOps.ToDouble(sequenceEmbeddings[posOffset + d]) != 0.0)
                {
                    return s;
                }
            }
        }
        return 0;
    }

    /// <inheritdoc />
    public override Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        // Empty string tokenized: [BOS, EOS, PAD, PAD, ...]
        var tokenIds = new Vector<T>(batchSize * MaxSequenceLength);
        // Build a matching attention mask so the encoder treats only [BOS, EOS] as
        // active text and zeroes the PAD tail. Without this, padded tokens would be
        // encoded as real input and skew unconditional conditioning + EOS pooling.
        var maskData = new Vector<T>(batchSize * MaxSequenceLength);
        for (int b = 0; b < batchSize; b++)
        {
            // OpenAI CLIP convention: BOS = '<|startoftext|>' = VocabSize - 2 (49406 for
            // the standard 49408-token vocab), EOS = '<|endoftext|>' = VocabSize - 1 (49407).
            // Using BOS = 1 here previously emitted the wrong embedding and broke
            // round-tripping with pretrained CLIP weights.
            tokenIds[b * MaxSequenceLength] = NumOps.FromDouble(VocabSize - 2);     // BOS
            tokenIds[b * MaxSequenceLength + 1] = NumOps.FromDouble(VocabSize - 1); // EOS
            // Active mask: 1 for BOS+EOS, 0 for the PAD tail.
            maskData[b * MaxSequenceLength] = NumOps.FromDouble(1.0);
            maskData[b * MaxSequenceLength + 1] = NumOps.FromDouble(1.0);
            // Rest is padding (0 token id, 0 mask) — already zero from default-init.
        }

        var input = new Tensor<T>(new[] { batchSize, MaxSequenceLength }, tokenIds);
        var mask = new Tensor<T>(new[] { batchSize, MaxSequenceLength }, maskData);
        return EncodeText(input, mask);
    }

    /// <inheritdoc />
    public override Tensor<T> Tokenize(string text)
    {
        var tokens = SimpleTokenize(text, MaxSequenceLength);
        var tokenData = new Vector<T>(MaxSequenceLength);
        for (int i = 0; i < MaxSequenceLength; i++)
            tokenData[i] = NumOps.FromDouble(tokens[i]);

        return new Tensor<T>(new[] { 1, MaxSequenceLength }, tokenData);
    }

    /// <inheritdoc />
    public override Tensor<T> TokenizeBatch(string[] texts)
    {
        var tokenData = new Vector<T>(texts.Length * MaxSequenceLength);
        for (int b = 0; b < texts.Length; b++)
        {
            var tokens = SimpleTokenize(texts[b], MaxSequenceLength);
            for (int i = 0; i < MaxSequenceLength; i++)
                tokenData[b * MaxSequenceLength + i] = NumOps.FromDouble(tokens[i]);
        }

        return new Tensor<T>(new[] { texts.Length, MaxSequenceLength }, tokenData);
    }

    /// <summary>
    /// Applies the transformer layers to the hidden state (simplified).
    /// </summary>
    /// <remarks>
    /// Implements a full CLIP-style transformer block (Radford et al. 2021
    /// "Learning Transferable Visual Models from Natural Language Supervision"
    /// §3.1.2; based on the GPT-2 / ViT architecture per Dosovitskiy et al.
    /// 2021): pre-LN + multi-head scaled dot-product attention + residual +
    /// pre-LN + MLP (H→4H GELU 4H→H) + residual. The previous "Simplified
    /// self-attention: residual + LN(attention(x))" stand-in collapsed
    /// attention to a single linear projection — the model never actually
    /// attended across tokens, which silently corrupted every text embedding
    /// the conditioner produced.
    ///
    /// <para>
    /// Weight layout per layer (matches the <c>weightsPerLayer = 12·H² + 4·H</c>
    /// allocation in <see cref="TextConditioningBase{T}"/>):
    /// </para>
    /// <list type="bullet">
    /// <item><description>[0,           H)        — LN1 γ</description></item>
    /// <item><description>[H,           2H)       — LN1 β</description></item>
    /// <item><description>[2H,          2H+H²)    — attention Q weight, row-major [H, H]</description></item>
    /// <item><description>[2H+H²,       2H+2H²)   — attention K weight</description></item>
    /// <item><description>[2H+2H²,      2H+3H²)   — attention V weight</description></item>
    /// <item><description>[2H+3H²,      2H+4H²)   — attention output projection</description></item>
    /// <item><description>[2H+4H²,      3H+4H²)   — LN2 γ</description></item>
    /// <item><description>[3H+4H²,      4H+4H²)   — LN2 β</description></item>
    /// <item><description>[4H+4H²,      4H+8H²)   — MLP W1, [H, 4H]</description></item>
    /// <item><description>[4H+8H²,      4H+12H²)  — MLP W2, [4H, H]</description></item>
    /// </list>
    /// </remarks>
    private Vector<T> ApplyTransformerLayers(Vector<T> hidden, int seqLen)
    {
        int H = HiddenSize;
        int H2 = H * H;
        int weightsPerLayer = 12 * H2 + 4 * H;
        int numHeads = ComputeNumHeads(H);
        int headDim = H / numHeads;

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;

            // ── Attention block: pre-LN + MHA + residual ────────────────────
            var residual = CopyVector(hidden);
            var lnGamma = ExtractSubVector(TransformerWeights, layerOffset, H);
            var lnBeta = ExtractSubVector(TransformerWeights, layerOffset + H, H);
            hidden = LayerNorm(hidden, lnGamma, lnBeta, H);

            int qOffset = layerOffset + 2 * H;
            int kOffset = qOffset + H2;
            int vOffset = qOffset + 2 * H2;
            int oOffset = qOffset + 3 * H2;
            hidden = MultiHeadAttention(hidden, qOffset, kOffset, vOffset, oOffset, seqLen, H, numHeads, headDim);
            hidden = AddVectors(hidden, residual);

            // ── MLP block: pre-LN + (H→4H GELU 4H→H) + residual ─────────────
            residual = CopyVector(hidden);
            int ln2Offset = layerOffset + 2 * H + 4 * H2;
            var ln2Gamma = ExtractSubVector(TransformerWeights, ln2Offset, H);
            var ln2Beta = ExtractSubVector(TransformerWeights, ln2Offset + H, H);
            hidden = LayerNorm(hidden, ln2Gamma, ln2Beta, H);

            int mlpW1Offset = ln2Offset + 2 * H;
            int mlpW2Offset = mlpW1Offset + 4 * H2;
            hidden = MLP(hidden, mlpW1Offset, mlpW2Offset, seqLen, H);
            hidden = AddVectors(hidden, residual);
        }

        return hidden;
    }

    /// <summary>
    /// CLIP head-count map per variant: 768→12, 1024→16, 1280→16, 1664→16
    /// (matches Hugging Face transformers' clip-vit-{base,large,big-g} configs).
    /// </summary>
    private static int ComputeNumHeads(int hiddenSize)
    {
        return hiddenSize switch
        {
            768 => 12,
            1024 => 16,
            1280 => 16,
            1664 => 16,
            _ => Math.Max(1, hiddenSize / 64) // Standard ViT convention: head_dim ≈ 64
        };
    }

    /// <summary>
    /// Multi-head scaled dot-product self-attention (Vaswani et al. 2017
    /// Eqs. 1-2). Operates on the [seqLen, HiddenSize] input through
    /// engine-accelerated batched matmul / softmax / transpose ops on
    /// <see cref="Tensor{T}"/> — no scalar inner loops, fully generic in
    /// <typeparamref name="T"/>, BLAS/GPU-eligible when the engine binds those.
    /// </summary>
    private Vector<T> MultiHeadAttention(Vector<T> input, int qOff, int kOff, int vOff, int oOff,
        int seqLen, int H, int numHeads, int headDim)
    {
        // Project to Q, K, V. LinearProject is already engine-routed (TensorMatMul).
        var qVec = LinearProject(input, TransformerWeights, qOff, H, H, seqLen);
        var kVec = LinearProject(input, TransformerWeights, kOff, H, H, seqLen);
        var vVec = LinearProject(input, TransformerWeights, vOff, H, H, seqLen);

        // Split heads: [seqLen, H] → [seqLen, numHeads, headDim] → permute to
        // [numHeads, seqLen, headDim] so batched matmul iterates over heads.
        var qHeads = Engine.TensorPermute(
            Tensor<T>.FromVector(qVec, [seqLen, numHeads, headDim]), [1, 0, 2]);
        var kHeads = Engine.TensorPermute(
            Tensor<T>.FromVector(kVec, [seqLen, numHeads, headDim]), [1, 0, 2]);
        var vHeads = Engine.TensorPermute(
            Tensor<T>.FromVector(vVec, [seqLen, numHeads, headDim]), [1, 0, 2]);

        // scores = Q @ K^T / √d_k  →  [numHeads, seqLen, seqLen]
        // K^T over the last two axes: permute [numHeads, seqLen, headDim] → [numHeads, headDim, seqLen].
        var kTranspose = Engine.TensorPermute(kHeads, [0, 2, 1]);
        var scores = Engine.TensorMatMul(qHeads, kTranspose);
        scores = Engine.TensorMultiplyScalar(scores, NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));

        // Softmax over the last (key) axis. Engine-accelerated; stable internally.
        var attn = Engine.TensorSoftmax(scores, axis: 2);

        // attn @ V → [numHeads, seqLen, headDim]
        var headOut = Engine.TensorMatMul(attn, vHeads);

        // Concat heads: permute [numHeads, seqLen, headDim] → [seqLen, numHeads, headDim]
        // → flatten last two into [seqLen, H].
        var concated = Engine.TensorPermute(headOut, [1, 0, 2]);
        var flat = Engine.Reshape(concated, [seqLen * H]);

        // Output projection W_o.
        return LinearProject(flat.ToVector(), TransformerWeights, oOff, H, H, seqLen);
    }

    /// <summary>
    /// CLIP MLP block: GELU(x @ W1) @ W2 with the standard 4× hidden expansion
    /// (Radford 2021 §3.1.2 uses GELU; Hendrycks &amp; Gimpel 2016 approximation).
    /// Routed through the engine's tape-tracked TensorGELU so the activation
    /// step is BLAS/GPU-eligible and stays generic in <typeparamref name="T"/>.
    /// </summary>
    private Vector<T> MLP(Vector<T> input, int w1Off, int w2Off, int seqLen, int H)
    {
        // hidden = input @ W1   ([seqLen, H] @ [H, 4H] → [seqLen, 4H])
        var hidden = LinearProject(input, TransformerWeights, w1Off, H, 4 * H, seqLen);
        // GELU via engine. The tape captures this as a single differentiable
        // op rather than the previous per-element NumOps.ToDouble dance.
        var hiddenTensor = Tensor<T>.FromVector(hidden);
        var activated = Engine.TensorGELU(hiddenTensor);
        // output = hidden @ W2   ([seqLen, 4H] @ [4H, H] → [seqLen, H])
        return LinearProject(activated.ToVector(), TransformerWeights, w2Off, 4 * H, H, seqLen);
    }

    /// <summary>
    /// Applies a linear projection to each position in the sequence. Fails fast if the
    /// requested weight slice does not fit inside the supplied weight buffer — silent
    /// truncation to zeros would corrupt downstream attention/MLP outputs without
    /// surfacing the configuration error.
    /// </summary>
    private Vector<T> LinearProject(Vector<T> input, Vector<T> weights, int weightOffset, int inDim, int outDim, int seqLen)
    {
        var wSize = inDim * outDim;
        if (weightOffset < 0 || weightOffset + wSize > weights.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(weightOffset),
                $"LinearProject requires {wSize} weights starting at offset {weightOffset}, " +
                $"but the weight buffer only has {weights.Length} elements. " +
                $"This indicates a transformer-layer offset miscount in the CLIP weight layout.");
        }

        var wSlice = new Vector<T>(wSize);
        for (int i = 0; i < wSize; i++)
            wSlice[i] = weights[weightOffset + i];

        // [seqLen, inDim] @ [inDim, outDim] = [seqLen, outDim] — vectorized
        var inputMat = Tensor<T>.FromVector(input).Reshape(seqLen, inDim);
        var wMat = Tensor<T>.FromVector(wSlice).Reshape(inDim, outDim);
        var result = Engine.TensorMatMul<T>(inputMat, wMat).Reshape(seqLen * outDim);
        return result.ToVector();
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
            copy[i] = source[i];
        return copy;
    }

    /// <summary>
    /// Extracts a contiguous subvector. Fails fast if the requested slice would extend
    /// past the source buffer — silent zero-fill on out-of-bounds reads previously
    /// produced corrupt LayerNorm gamma/beta and MLP weights without surfacing the
    /// configuration error.
    /// </summary>
    private static Vector<T> ExtractSubVector(Vector<T> source, int offset, int length)
    {
        if (offset < 0 || length < 0 || offset + length > source.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(offset),
                $"ExtractSubVector requires {length} elements starting at offset {offset}, " +
                $"but the source buffer only has {source.Length} elements.");
        }
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
            result[i] = source[offset + i];
        return result;
    }

    private static Vector<T> AddVectors(Vector<T> a, Vector<T> b)
    {
        return Engine.Add(a, b);
    }

    #region Variant Configuration

    private static int GetEmbeddingDim(string variant) => variant switch
    {
        "ViT-H/14" => 1024,
        "ViT-bigG/14" => 1280,
        _ => 768 // ViT-L/14 default
    };

    private static int GetHiddenSize(string variant) => variant switch
    {
        "ViT-H/14" => 1024,
        "ViT-bigG/14" => 1280,
        _ => 768 // ViT-L/14
    };

    private static int GetNumLayers(string variant) => variant switch
    {
        "ViT-H/14" => 24,
        "ViT-bigG/14" => 32,
        _ => 12 // ViT-L/14
    };

    private static int GetNumHeads(string variant) => variant switch
    {
        "ViT-H/14" => 16,
        "ViT-bigG/14" => 20,
        _ => 12 // ViT-L/14
    };

    #endregion
}
