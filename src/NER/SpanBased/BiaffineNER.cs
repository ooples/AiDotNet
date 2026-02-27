using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NER.SpanBased;

/// <summary>
/// Biaffine-NER: Named Entity Recognition as dependency parsing using biaffine classifiers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Biaffine-NER (Yu et al., ACL 2020 - "Named Entity Recognition as Dependency Parsing")
/// reformulates NER as identifying start and end boundaries of entity spans using biaffine
/// attention, an approach borrowed from dependency parsing.
///
/// <b>Key Innovation - Biaffine Attention for NER:</b>
/// Instead of BIO sequence labeling, Biaffine-NER constructs a span scoring matrix where
/// entry (i, j, k) represents the score that tokens i through j form an entity of type k.
/// The biaffine scoring function is:
///
/// score(i, j, k) = h_start_i^T * W_k * h_end_j + b_start_i^T * h_start_i + b_end_j^T * h_end_j + bias_k
///
/// where:
/// - h_start_i = MLP_start(encoder(x_i)) transforms the start token representation
/// - h_end_j = MLP_end(encoder(x_j)) transforms the end token representation
/// - W_k is a biaffine weight matrix for entity type k
/// - The biaffine term captures the interaction between start and end representations
///
/// <b>Architecture:</b>
/// 1. <b>Encoder:</b> BERT/BiLSTM produces contextual token representations
/// 2. <b>Start/End MLPs:</b> Separate feedforward networks for start and end boundary representations
/// 3. <b>Biaffine Classifier:</b> Scores all (start, end, entity-type) triples simultaneously
/// 4. <b>Decoding:</b> Select spans with score above threshold, resolve conflicts via greedy/optimal
///
/// <b>Advantages over BIO Tagging:</b>
/// - Naturally handles nested entities (overlapping spans get independent scores)
/// - No label transition constraints needed (no B-I-O consistency issues)
/// - Efficient: O(n^2 * k) scoring where n = seq length, k = entity types
/// - Joint boundary detection: start and end predictions are coupled via biaffine interaction
///
/// <b>Performance:</b>
/// - CoNLL-2003: ~93.5% F1 (flat NER)
/// - ACE 2004: ~87.3% F1 (nested NER)
/// - ACE 2005: ~86.7% F1 (nested NER)
/// - GENIA: ~79.2% F1 (nested biomedical NER)
/// </para>
/// <para>
/// <b>For Beginners:</b> Biaffine-NER treats entity recognition like finding matching
/// brackets: for each possible pair of words (start, end), it computes how likely they are
/// to be the boundaries of an entity. This is more flexible than labeling each word individually
/// because it can naturally handle overlapping entities (like "New York" being both a city
/// and part of "New York University").
/// </para>
/// </remarks>
public class BiaffineNER<T> : SpanBasedNERBase<T>
{
    /// <summary>
    /// Creates a Biaffine-NER model in ONNX inference mode.
    /// </summary>
    public BiaffineNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions? options = null)
        : base(architecture, modelPath, options ?? new SpanBasedNEROptions(),
            "Biaffine-NER", "Yu et al., ACL 2020")
    {
    }

    /// <summary>
    /// Creates a Biaffine-NER model in native training mode.
    /// </summary>
    public BiaffineNER(
        NeuralNetworkArchitecture<T> architecture,
        SpanBasedNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SpanBasedNEROptions(),
            "Biaffine-NER", "Yu et al., ACL 2020", optimizer)
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
            return new BiaffineNER<T>(Architecture, p, optionsCopy);
        return new BiaffineNER<T>(Architecture, optionsCopy);
    }
}
