using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.SequenceLabeling;

/// <summary>
/// Base class for sequence labeling NER models that assign a BIO label to each token in a sequence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sequence labeling is the most common approach to Named Entity Recognition, where each token
/// in a sentence is assigned a label from a predefined set using the BIO (Begin, Inside, Outside)
/// tagging scheme. This base class provides the task-specific functionality shared by all
/// sequence labeling NER models, analogous to how <c>VideoSuperResolutionBase&lt;T&gt;</c>
/// provides shared functionality for all video super-resolution models.
///
/// The BIO scheme works as follows:
/// - <b>B-TYPE:</b> Beginning of an entity of the given type (e.g., B-PER for the first token of a person name)
/// - <b>I-TYPE:</b> Inside (continuation of) an entity (e.g., I-PER for subsequent tokens of a person name)
/// - <b>O:</b> Outside any entity (regular words like verbs, prepositions, articles)
///
/// For example: "Barack Obama was born in Honolulu"
/// - Barack -> B-PER (beginning of a person entity)
/// - Obama  -> I-PER (inside the same person entity)
/// - was    -> O     (not an entity)
/// - born   -> O     (not an entity)
/// - in     -> O     (not an entity)
/// - Honolulu -> B-LOC (beginning of a location entity)
///
/// This base class provides:
/// - Label sequence prediction (abstract, implemented by concrete models)
/// - Emission score computation (the per-token, per-label scores before CRF decoding)
/// - Argmax decoding fallback for models without CRF
/// - Label index to label name conversion utilities
/// - CRF toggle for enabling/disabling structured prediction
///
/// Derived classes implement specific architectures like BiLSTM-CRF, BERT-NER, SpanNER, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sequence labeling NER processes text one word at a time and assigns
/// a label to each word. The labels use a special coding scheme called BIO:
/// - <b>B</b> = "Begin" - marks the first word of an entity
/// - <b>I</b> = "Inside" - marks continuation words of an entity
/// - <b>O</b> = "Outside" - marks words that aren't part of any entity
///
/// For example, in "John Smith works at Google":
/// - "John" -> B-PER (Beginning of a Person name)
/// - "Smith" -> I-PER (Inside the same Person name)
/// - "works" -> O (not an entity)
/// - "at" -> O (not an entity)
/// - "Google" -> B-ORG (Beginning of an Organization name)
///
/// The key advantage of this approach is that it naturally handles multi-word entities
/// like "New York City" (B-LOC, I-LOC, I-LOC) and nested entities.
/// </para>
/// </remarks>
public abstract class SequenceLabelingNERBase<T> : NERNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets whether to use CRF (Conditional Random Field) decoding for label sequence prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true (default), the model uses a CRF layer that models label transition dependencies
    /// and uses the Viterbi algorithm to find the globally optimal label sequence. This enforces
    /// structural constraints like:
    /// - I-PER can only follow B-PER or I-PER (not B-ORG or I-LOC)
    /// - A sequence cannot start with an I- tag
    /// - The label sequence must form valid BIO spans
    ///
    /// When false, labels are predicted independently for each token using argmax on the emission
    /// scores. This is faster but can produce invalid label sequences (e.g., I-PER following B-ORG).
    ///
    /// Research consistently shows that CRF decoding improves F1 score by 1-2% on standard NER
    /// benchmarks compared to independent classification (Lample et al., 2016; Ma and Hovy, 2016).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The CRF layer is like a spell-checker for entity labels. Without it,
    /// the model might make mistakes like labeling "Smith" as part of an organization even though
    /// "John" before it was labeled as a person. The CRF learns the rules about which labels can
    /// follow which other labels and ensures the entire sequence makes sense together.
    ///
    /// You should almost always keep this enabled (true) for best accuracy. Only disable it if
    /// you need maximum speed and can tolerate slightly lower accuracy.
    /// </para>
    /// </remarks>
    public bool UseCRF { get; protected set; } = true;

    /// <summary>
    /// Gets the label names corresponding to each label index in the BIO tagging scheme.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default label set follows the CoNLL-2003 shared task (Tjong Kim Sang and De Meulder, 2003),
    /// which is the most widely-used NER benchmark. The 9 labels represent 4 entity types in BIO scheme:
    ///
    /// - Index 0: <b>O</b> - Outside any entity (the most common label, ~83% of tokens in CoNLL-2003)
    /// - Index 1: <b>B-PER</b> - Beginning of a person name (e.g., "Albert" in "Albert Einstein")
    /// - Index 2: <b>I-PER</b> - Inside a person name (e.g., "Einstein" in "Albert Einstein")
    /// - Index 3: <b>B-ORG</b> - Beginning of an organization (e.g., "Princeton" in "Princeton University")
    /// - Index 4: <b>I-ORG</b> - Inside an organization (e.g., "University" in "Princeton University")
    /// - Index 5: <b>B-LOC</b> - Beginning of a location (e.g., "New" in "New Jersey")
    /// - Index 6: <b>I-LOC</b> - Inside a location (e.g., "Jersey" in "New Jersey")
    /// - Index 7: <b>B-MISC</b> - Beginning of a miscellaneous entity (e.g., "Nobel" in "Nobel Prize")
    /// - Index 8: <b>I-MISC</b> - Inside a miscellaneous entity (e.g., "Prize" in "Nobel Prize")
    ///
    /// Custom label sets can be configured via the options class for different annotation schemes
    /// (e.g., OntoNotes with 18 entity types, or domain-specific schemes for medical/legal NER).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These are the names of the labels the model uses. Each word in the
    /// text gets assigned one of these labels. The "B-" prefix means "beginning of" and "I-" means
    /// "inside" an entity. "O" means the word is not part of any entity.
    /// </para>
    /// </remarks>
    public string[] LabelNames { get; protected set; } =
    [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ];

    /// <summary>
    /// Initializes a new instance of the SequenceLabelingNERBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="lossFunction">The loss function for training. If null, cross-entropy loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the basic structure that all sequence labeling
    /// NER models share. Concrete models like BiLSTM-CRF call this constructor and then add their
    /// specific layers (LSTM, CRF, etc.) on top.
    /// </para>
    /// </remarks>
    protected SequenceLabelingNERBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Predicts the optimal BIO label sequence for input token embeddings.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings with shape [sequenceLength, embeddingDim]
    /// for a single sentence or [batch, sequenceLength, embeddingDim] for multiple sentences.</param>
    /// <returns>Predicted label indices with shape [sequenceLength] or [batch, sequenceLength].
    /// Each value is an integer index into <see cref="LabelNames"/>.</returns>
    /// <remarks>
    /// <para>
    /// This is the primary inference method for sequence labeling NER. When CRF decoding is enabled,
    /// this method uses the Viterbi algorithm to find the globally optimal label sequence that
    /// maximizes the sum of emission scores (from the BiLSTM) and transition scores (from the CRF).
    /// When CRF is disabled, it performs independent argmax decoding at each token position.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Give this method the numerical representations of your words, and it
    /// returns a label for each word. The labels are numbers that correspond to the names in
    /// <see cref="LabelNames"/>. For example, a return value of [1, 2, 0, 0, 3] means the first
    /// word is B-PER, the second is I-PER, the third and fourth are O, and the fifth is B-ORG.
    /// Use <see cref="DecodeLabels"/> to convert these numbers to readable label names.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> PredictLabels(Tensor<T> tokenEmbeddings);

    /// <summary>
    /// Computes emission scores from token embeddings without CRF decoding.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings with shape [sequenceLength, embeddingDim].</param>
    /// <returns>Emission score matrix with shape [sequenceLength, numLabels]. Each row contains
    /// the raw scores for all possible labels at that token position.</returns>
    /// <remarks>
    /// <para>
    /// Emission scores represent the model's confidence that each token should receive each label,
    /// based purely on the token's contextual features (from the BiLSTM). Higher scores indicate
    /// higher confidence. These scores are the input to the CRF layer, which combines them with
    /// learned transition scores to find the optimal label sequence.
    ///
    /// In the CRF framework, the score for a label sequence y given input x is:
    /// score(x, y) = sum of emission[t, y_t] + sum of transition[y_{t-1}, y_t]
    ///
    /// The emission scores capture "what label does this token look like?" while the transition
    /// scores capture "what label typically follows the previous label?"
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Emission scores tell the model how likely each label is for each word,
    /// based on the word's context alone. For example, the word "Google" might have high emission
    /// scores for B-ORG (organization) and B-LOC (location, since Google is also a place name),
    /// but low scores for B-PER (person). The CRF layer then uses these scores along with its
    /// knowledge of label patterns to make the final decision.
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> ComputeEmissionScores(Tensor<T> tokenEmbeddings);

    /// <summary>
    /// Performs independent argmax decoding on emission scores to get label predictions.
    /// </summary>
    /// <param name="emissionScores">Emission score matrix with shape [sequenceLength, numLabels].</param>
    /// <returns>Label indices with shape [sequenceLength], where each value is the index of the
    /// highest-scoring label at that position.</returns>
    /// <remarks>
    /// <para>
    /// Argmax decoding selects the highest-scoring label independently at each token position.
    /// This is used as a fallback when CRF decoding is disabled. While simpler and faster than
    /// Viterbi decoding, it can produce invalid label sequences because it doesn't consider
    /// label transition dependencies.
    ///
    /// For example, argmax might produce: B-PER, I-ORG, O - which is invalid because I-ORG
    /// cannot follow B-PER in the BIO scheme. CRF decoding would correct this.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> For each word, this simply picks the label with the highest score.
    /// It's like taking a multiple-choice test and always picking the answer you're most confident
    /// about, without considering how your answers relate to each other. This is fast but can
    /// sometimes produce answers that don't make sense together.
    /// </para>
    /// </remarks>
    protected Tensor<T> ArgmaxDecode(Tensor<T> emissionScores)
    {
        int seqLen = emissionScores.Shape[0];
        int numLabels = emissionScores.Shape[1];
        var labels = new Tensor<T>([seqLen]);

        for (int s = 0; s < seqLen; s++)
        {
            int bestLabel = 0;
            double bestScore = double.NegativeInfinity;

            for (int l = 0; l < numLabels; l++)
            {
                double score = NumOps.ToDouble(emissionScores.Data.Span[s * numLabels + l]);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestLabel = l;
                }
            }

            labels.Data.Span[s] = NumOps.FromDouble(bestLabel);
        }

        return labels;
    }

    /// <summary>
    /// Converts predicted label indices to human-readable BIO label name strings.
    /// </summary>
    /// <param name="labelIndices">Label index tensor with shape [sequenceLength]. Each value is
    /// an integer index into <see cref="LabelNames"/>.</param>
    /// <returns>Array of label name strings (e.g., ["B-PER", "I-PER", "O", "O", "B-ORG"]).</returns>
    /// <remarks>
    /// <para>
    /// This utility method converts the numerical predictions from <see cref="PredictLabels"/>
    /// into readable label strings. Out-of-range indices are mapped to "O" (Outside) as a
    /// safe fallback.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After calling PredictLabels, you get numbers like [1, 2, 0, 0, 3].
    /// This method converts them to readable names like ["B-PER", "I-PER", "O", "O", "B-ORG"]
    /// so you can easily see which words are people, organizations, locations, etc.
    /// </para>
    /// </remarks>
    public string[] DecodeLabels(Tensor<T> labelIndices)
    {
        int seqLen = labelIndices.Shape[0];
        var result = new string[seqLen];

        for (int i = 0; i < seqLen; i++)
        {
            int idx = (int)NumOps.ToDouble(labelIndices.Data.Span[i]);
            result[i] = idx >= 0 && idx < LabelNames.Length ? LabelNames[idx] : "O";
        }

        return result;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return PredictLabels(input);
    }
}
