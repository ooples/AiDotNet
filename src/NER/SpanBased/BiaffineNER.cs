using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

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
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputSize: 512,
///     outputSize: 9,
///     hiddenLayers: new[] { 256, 128 },
///     networkType: NetworkType.Classification);
/// var biaffineNER = new BiaffineNER&lt;float&gt;(architecture);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Named Entity Recognition as Dependency Parsing",
    "https://arxiv.org/abs/2005.07150",
    Year = 2020,
    Authors = "Juntao Yu, Bernd Bohnet, Massimo Poesio")]
public class BiaffineNER<T> : SpanBasedNERBase<T>
{
    /// <summary>
    /// Creates a Biaffine-NER model in ONNX inference mode.
    /// </summary>
    public BiaffineNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions? options = null)
        : base(architecture, modelPath, options ?? new BiaffineNEROptions(),
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
        : base(architecture, options ?? new BiaffineNEROptions(),
            "Biaffine-NER", "Yu et al., ACL 2020", optimizer)
    {
    }

    /// <summary>
    /// The Biaffine-NER specific options, when supplied. Falls back to the paper's published
    /// values for the BiLSTM and embedding-dropout settings that only this model defines.
    /// </summary>
    private BiaffineNEROptions BiaffineOptions =>
        NEROptions as BiaffineNEROptions ?? _defaultBiaffineOptions;

    private static readonly BiaffineNEROptions _defaultBiaffineOptions = new();

    /// <inheritdoc />
    protected override IEnumerable<ILayer<T>> CreateDefaultLayers()
    {
        return LayerHelper<T>.CreateDefaultBiaffineNERLayers(
            hiddenDimension: NEROptions.HiddenDimension,
            numAttentionHeads: NEROptions.NumAttentionHeads,
            numTransformerLayers: NEROptions.NumTransformerLayers,
            intermediateDimension: NEROptions.IntermediateDimension,
            spanEmbeddingDimension: NEROptions.SpanEmbeddingDimension,
            numLabels: NEROptions.NumLabels,
            dropoutRate: NEROptions.DropoutRate,
            biLstmHiddenSize: BiaffineOptions.BiLstmHiddenSize,
            biLstmLayers: BiaffineOptions.BiLstmLayers,
            biLstmDropout: BiaffineOptions.BiLstmDropout,
            embeddingsDropout: BiaffineOptions.EmbeddingsDropout);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new SpanBasedNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new BiaffineNER<T>(Architecture, p, optionsCopy);
        return new BiaffineNER<T>(Architecture, optionsCopy);
    }

    /// <summary>
    /// Builds span-level supervision: one label per candidate (start, end) pair.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Yu, Bohnet and Poesio (ACL 2020) score every span into an l x l x c tensor, where c is
    /// the entity category count PLUS ONE for a dedicated non-entity class, and optimise softmax
    /// cross-entropy over it. The scorer already emits exactly that tensor, flattened to
    /// [l * l, c].
    /// </para>
    /// <para>
    /// The supervision did not match it. The base pairs the model's output with per-TOKEN
    /// labels, [l], so a 16-token sentence produced 256 span rows against 16 labels. Nothing
    /// related the two, the gradient was identically zero, and the memorization loss sat at
    /// 14.278996 for all 15 steps while Adam still moved 1.69M parameters by its normalized
    /// step -- motion that looks like training but carries no signal.
    /// </para>
    /// <para>
    /// Token labels annotate single-token entities, so span (i, j) takes token i's category when
    /// i == j and the non-entity class otherwise. Class 0 is the non-entity class, matching the
    /// paper's "+1" slot.
    /// </para>
    /// <para>
    /// Only spans the paper actually treats as candidates are supervised: those with
    /// <c>s &lt;= e</c> and no longer than <c>MaxSpanLength</c>. Everything else is marked with the
    /// -1 ignore sentinel, whose one-hot row is all zeros and therefore contributes no gradient
    /// (the same convention as PyTorch's <c>ignore_index</c>).
    /// </para>
    /// <para>
    /// Labelling those slots non-entity instead, as this first did, is not merely redundant — it
    /// swamps the objective. For a 12-token sequence it supervises 144 spans of which 12 can ever
    /// be an entity, so predicting "no entity" everywhere is close to optimal and the model
    /// collapses to a constant function that ignores its input. Restricting supervision to the
    /// paper's candidate set leaves 33 spans at MaxSpanLength 3, of which those same 12 carry
    /// signal.
    /// </para>
    /// </remarks>
    protected override Tensor<T> BuildTrainingTargets(Tensor<T> labels, int seqLen)
    {
        var tokenLabels = PreprocessLabels(labels, seqLen);

        // The scorer emits [batch, l*l, c] for a batched input and [l*l, c] unbatched, so the
        // target has to carry the same batch axis. Building a flat [l*l] regardless meant a
        // two-example batch produced a [26] target against a [2, 64, 9] prediction, which cannot
        // broadcast -- the failure DifferentInputs_AfterTraining reported, since that test is the
        // one that feeds two distinct inputs at once.
        int batch = tokenLabels.Rank >= 2 ? tokenLabels.Shape[0] : 1;

        var spanTargets = batch > 1
            ? new Tensor<T>([batch, seqLen * seqLen])
            : new Tensor<T>([seqLen * seqLen]);

        var ignore = NumOps.FromDouble(-1.0);
        int maxSpan = NEROptions.MaxSpanLength > 0 ? NEROptions.MaxSpanLength : seqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int start = 0; start < seqLen; start++)
            {
                var category = tokenLabels.Rank == 1 ? tokenLabels[start] : tokenLabels[b, start];

                for (int end = 0; end < seqLen; end++)
                {
                    bool isCandidate = end >= start && (end - start + 1) <= maxSpan;

                    // A single-token entity occupies the diagonal; other candidate spans are
                    // non-entities; non-candidates are ignored rather than taught as negatives.
                    var value = !isCandidate ? ignore
                        : end == start ? category
                        : NumOps.Zero;

                    if (batch > 1)
                    {
                        spanTargets[b, (start * seqLen) + end] = value;
                    }
                    else
                    {
                        spanTargets[(start * seqLen) + end] = value;
                    }
                }
            }
        }

        return spanTargets;
    }

}
