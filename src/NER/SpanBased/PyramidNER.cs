using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NER.SpanBased;

/// <summary>
/// Pyramid-NER: Hierarchical pyramid network for nested Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Pyramid-NER (Jue et al., ACL 2020 - "Pyramid: A Layered Model for Nested Named Entity
/// Recognition") introduces a novel layered architecture where each pyramid layer identifies
/// entities at a specific nesting level, and inner entities become features for outer entities.
///
/// <b>Key Innovation - Layered Pyramid Architecture:</b>
/// Instead of treating nested NER as a single flat classification problem, Pyramid-NER
/// builds a pyramid of L layers, where:
/// - Layer 1: Identifies the innermost (shortest) entities
/// - Layer 2: Identifies entities that may contain Layer 1 entities
/// - Layer L: Identifies the outermost (longest) entities
///
/// Each layer uses a BiLSTM or transformer encoder, and the identified entities from
/// lower layers are fed as additional features to higher layers through "inverse pyramid"
/// connections.
///
/// <b>Architecture:</b>
/// <code>
///   Input tokens
///       |
///   [Layer 1: BiLSTM] --> innermost entities (e.g., "York")
///       |     |
///   [Layer 2: BiLSTM] --> entities containing Layer 1 (e.g., "New York")
///       |     |     |
///   [Layer 3: BiLSTM] --> outermost entities (e.g., "New York University")
/// </code>
///
/// <b>Inverse Pyramid Connections:</b>
/// When Layer k identifies an entity span (i, j), a "flag" embedding is generated and
/// concatenated with the token representations at layer k+1. This tells higher layers
/// "there is an entity of type X spanning positions i to j", enabling the model to
/// learn that outer entities often contain inner entities of specific types.
///
/// <b>Performance (Nested NER):</b>
/// - ACE 2004: ~86.1% F1
/// - ACE 2005: ~84.9% F1
/// - GENIA: ~78.5% F1
/// - NNE: ~93.7% F1
///
/// <b>Advantages:</b>
/// - Naturally handles arbitrarily deep nesting levels
/// - Lower layers provide useful features for higher layers
/// - Simple, interpretable architecture
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine reading a sentence and first finding the smallest named
/// entities, then using those to help find bigger entities that contain them. For example,
/// first find "York" (location), then use that to find "New York" (location), then find
/// "New York University" (organization). Each step helps the next, building up like a pyramid.
/// </para>
/// </remarks>
public class PyramidNER<T> : SpanBasedNERBase<T>
{
    /// <summary>
    /// Creates a Pyramid-NER model in ONNX inference mode.
    /// </summary>
    public PyramidNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions? options = null)
        : base(architecture, modelPath, options ?? new SpanBasedNEROptions(),
            "Pyramid-NER", "Jue et al., ACL 2020")
    {
    }

    /// <summary>
    /// Creates a Pyramid-NER model in native training mode.
    /// </summary>
    public PyramidNER(
        NeuralNetworkArchitecture<T> architecture,
        SpanBasedNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SpanBasedNEROptions(),
            "Pyramid-NER", "Jue et al., ACL 2020", optimizer)
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
            return new PyramidNER<T>(Architecture, p, optionsCopy);
        return new PyramidNER<T>(Architecture, optionsCopy);
    }
}
