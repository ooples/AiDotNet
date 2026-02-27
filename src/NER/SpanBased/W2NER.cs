using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NER.SpanBased;

/// <summary>
/// W2NER: Word-Word Relation Classification for unified flat and nested NER.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// W2NER (Li et al., AAAI 2022 - "Unified Named Entity Recognition as Word-Word Relation
/// Classification") reformulates NER as a word-word relation classification problem, where
/// the relation between every pair of words indicates whether they are part of the same entity
/// and what type of entity they form.
///
/// <b>Key Innovation - Word-Word Relations:</b>
/// Instead of labeling individual tokens (BIO) or classifying spans, W2NER builds an n x n
/// table (where n is sentence length) and classifies each word-word pair into one of:
/// - <b>None:</b> Words are not related by any entity
/// - <b>Next-Neighboring-Word (NNW):</b> Words are consecutive within the same entity
/// - <b>Tail-Head-Word-* (THW-TYPE):</b> The pair represents the (tail, head) boundary
///   of an entity of the given TYPE
///
/// <b>Example:</b>
/// Sentence: "Barack Obama visited New York City"
/// - (Barack, Obama) = NNW (consecutive in same entity)
/// - (Obama, Barack) = THW-PER (tail-head pair of PER entity "Barack Obama")
/// - (New, York) = NNW, (York, City) = NNW
/// - (City, New) = THW-LOC (tail-head pair of LOC entity "New York City")
///
/// <b>Architecture:</b>
/// 1. <b>BERT Encoder:</b> Produces contextual token representations
/// 2. <b>Convolution Layer:</b> A convolutional layer over the word-pair grid captures
///    local interactions between neighboring word pairs
/// 3. <b>Co-Predictor:</b> Combines token-level and grid-level features using:
///    - CLN (Conditional Layer Normalization): h_ij conditioned on h_i and h_j
///    - Distance embeddings: relative position between word pairs
///    - Biaffine scoring for the final word-word relation classification
///
/// <b>Advantages:</b>
/// - Unified framework: handles flat, nested, and discontinuous entities
/// - No span enumeration: avoids the O(n^2) span enumeration cost
/// - Grid structure captures inter-entity dependencies naturally
///
/// <b>Performance:</b>
/// - CoNLL-2003 (flat): ~93.4% F1
/// - ACE 2004 (nested): ~87.3% F1
/// - ACE 2005 (nested): ~86.6% F1
/// - GENIA (nested): ~79.8% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> W2NER looks at every pair of words in a sentence and asks: "Are these
/// two words part of the same entity?" Instead of labeling each word separately, it builds a
/// grid showing relationships between all word pairs. This is powerful because it can handle
/// normal entities, nested entities, and even discontinuous entities (where an entity has
/// gaps) using the same unified approach.
/// </para>
/// </remarks>
public class W2NER<T> : SpanBasedNERBase<T>
{
    /// <summary>
    /// Creates a W2NER model in ONNX inference mode.
    /// </summary>
    public W2NER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions? options = null)
        : base(architecture, modelPath, options ?? new SpanBasedNEROptions(),
            "W2NER", "Li et al., AAAI 2022")
    {
    }

    /// <summary>
    /// Creates a W2NER model in native training mode.
    /// </summary>
    public W2NER(
        NeuralNetworkArchitecture<T> architecture,
        SpanBasedNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SpanBasedNEROptions(),
            "W2NER", "Li et al., AAAI 2022", optimizer)
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
            return new W2NER<T>(Architecture, p, optionsCopy);
        return new W2NER<T>(Architecture, optionsCopy);
    }
}
