using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

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
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 512, numClasses: 9);
/// var w2NER = new W2NER&lt;float&gt;(architecture);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unified Named Entity Recognition as Word-Word Relation Classification",
    "https://arxiv.org/abs/2112.10070",
    Year = 2022,
    Authors = "Jingye Li, Hao Fei, Jiang Liu, Shengqiong Wu, Meishan Zhang, Chong Teng, Donghong Ji, Fei Li")]
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
        // The encoder, then the paper's BiLSTM (directions summed here) and the word-pair grid classifier with the CoNLL-2003
        // configuration of the authors' code (conv_hid_size 96, ffnn_hid_size 128, biaffine_size 768); the relations are
        // None, NNW and one THW per entity type.
        foreach (var layer in CreateTokenEncoderLayers()) yield return layer;
        yield return new BidirectionalLayer<T>(new LSTMLayer<T>(NEROptions.HiddenDimension), mergeMode: true,
            activationFunction: (IActivationFunction<T>)new IdentityActivation<T>());
        yield return new WordPairGridLayer<T>(NEROptions.HiddenDimension, channels: 96, hiddenDim: 128, biaffineDim: 768,
            numRelations: NEROptions.NumLabels + 1);
    }

    /// <inheritdoc />
    /// <remarks>Each word pair's most likely relation (None, NNW or a THW type); a THW pair from tail i to head j (j at or
    /// before i) is an entity of that type over tokens j to i. Discontinuous entities, which the paper recovers by
    /// following NNW links from the head (Algorithm 1), are returned as their contiguous extent.</remarks>
    protected override IReadOnlyList<NamedEntitySpan> DecodeSpans(Tensor<T> output, int seqLen)
    {
        int relations = output.Shape[^1];
        var grid = Engine.Reshape(output, new[] { seqLen * seqLen, relations });
        var spans = new List<NamedEntitySpan>();
        for (int tail = 0; tail < seqLen; tail++)
            for (int head = 0; head <= tail; head++)
            {
                var (relation, probability) = MostLikely(grid, tail * seqLen + head, relations);
                if (relation >= 2) spans.Add(new NamedEntitySpan(head, tail, relation - 1, probability));
            }
        return spans;
    }

    /// <summary>
    /// Word-pair relation targets (section 2.1): each run of tokens sharing a non-zero label is an entity of that type; NNW
    /// (relation 1) links each word of an entity to the next, THW of the type (relation 1 + type) links its tail to its head,
    /// and every other pair is None (relation 0).
    /// </summary>
    protected override Tensor<T> BuildTrainingTargets(Tensor<T> labels, int seqLen)
    {
        var tokenLabels = PreprocessLabels(labels, seqLen);
        int batch = tokenLabels.Rank >= 2 ? tokenLabels.Shape[0] : 1;
        var targets = batch > 1 ? new Tensor<T>(new[] { batch, seqLen * seqLen }) : new Tensor<T>(new[] { seqLen * seqLen });
        for (int b = 0; b < batch; b++)
        {
            int Label(int t) => (int)Math.Round(NumOps.ToDouble(tokenLabels.Rank == 1 ? tokenLabels[t] : tokenLabels[b, t]));
            void Set(int i, int j, int relation)
            {
                if (batch > 1) targets[b, i * seqLen + j] = NumOps.FromDouble(relation);
                else targets[i * seqLen + j] = NumOps.FromDouble(relation);
            }
            for (int start = 0; start < seqLen;)
            {
                int type = Label(start), end = start;
                while (end + 1 < seqLen && type > 0 && Label(end + 1) == type) end++;
                if (type > 0)
                {
                    for (int t = start; t < end; t++) Set(t, t + 1, 1);
                    Set(end, start, 1 + type);
                }
                start = end + 1;
            }
        }
        return targets;
    }
}
