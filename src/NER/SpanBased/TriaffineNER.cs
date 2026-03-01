using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NER.SpanBased;

/// <summary>
/// Triaffine-NER: Three-way interaction model for nested Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Triaffine-NER (Yuan et al., ACL 2022 - "Fusing Heterogeneous Factors with Triaffine
/// Mechanism for Nested Named Entity Recognition") extends biaffine scoring with a third
/// factor that captures span content information, enabling richer span representations
/// for nested NER.
///
/// <b>Key Innovation - Triaffine Mechanism:</b>
/// While Biaffine-NER scores spans using only start and end boundary tokens:
///   score_biaffine(i,j) = h_start_i^T * W * h_end_j
///
/// Triaffine-NER adds a third factor that represents the span content:
///   score_triaffine(i,j) = h_start_i^T * W(h_content_{i:j}) * h_end_j
///
/// where W(h_content) is a weight matrix conditioned on the span content representation.
/// This creates a three-way interaction between start boundary, end boundary, and content,
/// allowing the model to differentiate spans with similar boundaries but different content.
///
/// <b>Heterogeneous Factors:</b>
/// The three factors capture different aspects of entity spans:
/// 1. <b>Start boundary (h_start):</b> Left context and entity beginning patterns
/// 2. <b>End boundary (h_end):</b> Right context and entity ending patterns
/// 3. <b>Content (h_content):</b> Internal span semantics (pooled over span tokens)
///
/// <b>Architecture:</b>
/// 1. Pre-trained transformer encoder produces token representations
/// 2. Three separate MLPs transform tokens into start, end, and content representations
/// 3. Triaffine scoring: For each (start, end, content) triple, compute entity type scores
/// 4. Greedy or optimal decoding to extract non-conflicting entity spans
///
/// <b>Performance (Nested NER):</b>
/// - ACE 2004: ~87.8% F1 (state-of-the-art)
/// - ACE 2005: ~86.5% F1
/// - GENIA: ~80.4% F1
///
/// <b>Advantage over Biaffine:</b>
/// Consider two overlapping spans with the same boundaries but different inner content:
/// "Bank of America" (ORG) vs "Bank of the River" (LOC). Biaffine only sees "Bank" and
/// the last token, while Triaffine additionally considers the middle tokens to make the
/// correct distinction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of Triaffine-NER as an improved version of Biaffine-NER.
/// While Biaffine only looks at the first and last word of a potential entity, Triaffine
/// also considers what's in the middle. This helps distinguish entities that start and end
/// the same way but have different content, like "Bank of America" vs "Bank of the River."
/// </para>
/// </remarks>
public class TriaffineNER<T> : SpanBasedNERBase<T>
{
    /// <summary>
    /// Creates a Triaffine-NER model in ONNX inference mode.
    /// </summary>
    public TriaffineNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions? options = null)
        : base(architecture, modelPath, options ?? new SpanBasedNEROptions(),
            "Triaffine-NER", "Yuan et al., ACL 2022")
    {
    }

    /// <summary>
    /// Creates a Triaffine-NER model in native training mode.
    /// </summary>
    public TriaffineNER(
        NeuralNetworkArchitecture<T> architecture,
        SpanBasedNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SpanBasedNEROptions(),
            "Triaffine-NER", "Yuan et al., ACL 2022", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IEnumerable<ILayer<T>> CreateDefaultLayers()
    {
        return LayerHelper<T>.CreateDefaultSpanBasedNERLayers(
            hiddenDimension: NEROptions.HiddenDimension,
            numAttentionHeads: NEROptions.NumAttentionHeads,
            numTransformerLayers: NEROptions.NumTransformerLayers,
            intermediateDimension: NEROptions.IntermediateDimension,
            spanEmbeddingDimension: NEROptions.SpanEmbeddingDimension,
            numLabels: NEROptions.NumLabels,
            dropoutRate: NEROptions.DropoutRate);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new SpanBasedNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new TriaffineNER<T>(Architecture, p, optionsCopy);
        return new TriaffineNER<T>(Architecture, optionsCopy);
    }
}
