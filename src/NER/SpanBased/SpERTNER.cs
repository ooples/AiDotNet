using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NER.SpanBased;

/// <summary>
/// SpERT: Span-based Entity and Relation Transformer for joint entity and relation extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SpERT (Eberts and Ulges, ECAI 2020 - "Span-based Joint Entity and Relation Extraction with
/// Transformer Pre-training") performs joint entity recognition and relation extraction using
/// a span-based approach built on top of a pre-trained transformer encoder.
///
/// <b>Architecture Overview:</b>
/// 1. <b>Transformer Encoder:</b> Pre-trained BERT encodes the input sentence into contextual
///    token representations
/// 2. <b>Span Representation:</b> For each candidate span (i, j), the representation is:
///    span(i,j) = [h_i; h_j; maxpool(h_i:h_j); width_embedding]
///    where h_i, h_j are boundary tokens, maxpool is over the span content, and
///    width_embedding encodes the span length
/// 3. <b>Entity Classifier:</b> A feedforward network classifies each span representation
///    into entity types or non-entity
/// 4. <b>Relation Classifier:</b> For each pair of predicted entity spans, a relation
///    classifier predicts the relation type using concatenated span representations
///    plus context between the entities
///
/// <b>Negative Sampling:</b>
/// Since most spans are non-entities and most entity pairs have no relation, SpERT uses
/// careful negative sampling during training. The ratio of negative to positive samples
/// is a key hyperparameter (typically 100:1 for entities).
///
/// <b>Performance:</b>
/// - CoNLL-2004: 86.3% entity F1, 72.9% relation F1
/// - ADE dataset: 89.3% entity F1, 79.2% relation F1
/// - SciERC: 70.3% entity F1, 48.4% relation F1
///
/// <b>Key Insight:</b>
/// By operating on spans rather than individual tokens, SpERT avoids BIO label constraint
/// issues and naturally handles multi-token entities. The joint entity-relation extraction
/// ensures that entity and relation decisions are mutually informed.
/// </para>
/// <para>
/// <b>For Beginners:</b> SpERT looks at all possible groups of consecutive words (spans) in a
/// sentence and classifies each as an entity type or non-entity. Unlike BiLSTM-CRF which labels
/// each word one at a time, SpERT considers entire phrases at once. It can also find relationships
/// between entities (e.g., "born in" between a person and a location).
/// </para>
/// </remarks>
public class SpERTNER<T> : SpanBasedNERBase<T>
{
    /// <summary>
    /// Creates a SpERT model in ONNX inference mode.
    /// </summary>
    public SpERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions? options = null)
        : base(architecture, modelPath, options ?? new SpanBasedNEROptions(),
            "SpERT", "Eberts and Ulges, ECAI 2020")
    {
    }

    /// <summary>
    /// Creates a SpERT model in native training mode.
    /// </summary>
    public SpERTNER(
        NeuralNetworkArchitecture<T> architecture,
        SpanBasedNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SpanBasedNEROptions(),
            "SpERT", "Eberts and Ulges, ECAI 2020", optimizer)
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
            return new SpERTNER<T>(Architecture, p, optionsCopy);
        return new SpERTNER<T>(Architecture, optionsCopy);
    }
}
