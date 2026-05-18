using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

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
/// like "New York City" (B-LOC, I-LOC, I-LOC).
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
    /// Supports both single-sequence (2D) and batched (3D) inputs.
    /// </summary>
    /// <param name="emissionScores">Emission score tensor with shape [sequenceLength, numLabels]
    /// for a single sequence, or [batch, sequenceLength, numLabels] for batched input.</param>
    /// <returns>Label indices with shape [sequenceLength] for single-sequence input, or
    /// [batch, sequenceLength] for batched input. Each value is the index of the
    /// highest-scoring label at that position.</returns>
    /// <remarks>
    /// <para>
    /// Argmax decoding selects the highest-scoring label independently at each token position.
    /// This is used as a fallback when CRF decoding is disabled and also for converting the
    /// CRF layer's one-hot output to integer label indices.
    ///
    /// For rank-3 (batched) input, each batch element is decoded independently. The output
    /// has one fewer dimension than the input (the label dimension is collapsed to indices).
    ///
    /// While simpler and faster than Viterbi decoding, independent argmax can produce invalid
    /// label sequences because it doesn't consider label transition dependencies.
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
        // Handle batched 3D input [batch, seqLen, numLabels]
        if (emissionScores.Rank == 3)
        {
            int batchSize = emissionScores.Shape[0];
            int seqLen = emissionScores.Shape[1];
            int numLabels = emissionScores.Shape[2];
            var batchLabels = new Tensor<T>([batchSize, seqLen]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int bestLabel = 0;
                    double bestScore = double.NegativeInfinity;
                    int baseIdx = (b * seqLen + s) * numLabels;

                    for (int l = 0; l < numLabels; l++)
                    {
                        double score = NumOps.ToDouble(emissionScores.Data.Span[baseIdx + l]);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestLabel = l;
                        }
                    }

                    batchLabels.Data.Span[b * seqLen + s] = NumOps.FromDouble(bestLabel);
                }
            }

            return batchLabels;
        }

        // Handle single-sequence 2D input [seqLen, numLabels]
        int seq = emissionScores.Shape[0];
        int labels2d = emissionScores.Shape[1];
        var result = new Tensor<T>([seq]);

        for (int s = 0; s < seq; s++)
        {
            int bestLabel = 0;
            double bestScore = double.NegativeInfinity;

            for (int l = 0; l < labels2d; l++)
            {
                double score = NumOps.ToDouble(emissionScores.Data.Span[s * labels2d + l]);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestLabel = l;
                }
            }

            result.Data.Span[s] = NumOps.FromDouble(bestLabel);
        }

        return result;
    }

    /// <summary>
    /// Converts predicted label indices to human-readable BIO label name strings.
    /// Supports single-sequence (1D) input only. For batched output, decode each
    /// sequence individually.
    /// </summary>
    /// <param name="labelIndices">Label index tensor with shape [sequenceLength]. Each value is
    /// an integer index into <see cref="LabelNames"/>. For batched output (2D), extract
    /// individual sequences first and decode each one separately.</param>
    /// <returns>Array of label name strings (e.g., ["B-PER", "I-PER", "O", "O", "B-ORG"]).</returns>
    /// <remarks>
    /// <para>
    /// This utility method converts the numerical predictions from <see cref="PredictLabels"/>
    /// into readable label strings. Out-of-range indices are mapped to "O" (Outside) as a
    /// safe fallback.
    ///
    /// For batched predictions (rank-2 tensor [batch, seqLen]), use <see cref="DecodeLabelsBatch"/>
    /// which returns a list of string arrays, one per batch element.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After calling PredictLabels, you get numbers like [1, 2, 0, 0, 3].
    /// This method converts them to readable names like ["B-PER", "I-PER", "O", "O", "B-ORG"]
    /// so you can easily see which words are people, organizations, locations, etc.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when labelIndices is not rank-1.</exception>
    public string[] DecodeLabels(Tensor<T> labelIndices)
    {
        if (labelIndices.Rank != 1)
            throw new ArgumentException(
                $"DecodeLabels expects a 1D tensor [sequenceLength]. Got rank {labelIndices.Rank}. " +
                "For batched predictions, use DecodeLabelsBatch instead.");

        int seqLen = labelIndices.Shape[0];
        var result = new string[seqLen];

        for (int i = 0; i < seqLen; i++)
        {
            int idx = (int)NumOps.ToDouble(labelIndices.Data.Span[i]);
            result[i] = idx >= 0 && idx < LabelNames.Length ? LabelNames[idx] : "O";
        }

        return result;
    }

    /// <summary>
    /// Converts batched predicted label indices to human-readable BIO label name strings.
    /// </summary>
    /// <param name="batchLabelIndices">Label index tensor with shape [batch, sequenceLength].
    /// Each value is an integer index into <see cref="LabelNames"/>.</param>
    /// <returns>A list of string arrays, one per batch element, each containing the label names
    /// for that sequence.</returns>
    /// <remarks>
    /// <para>
    /// This is the batched version of <see cref="DecodeLabels"/>. For each sequence in the batch,
    /// it converts numerical label indices to readable label strings. Out-of-range indices are
    /// mapped to "O" (Outside) as a safe fallback.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If you predicted labels for multiple sentences at once (batch mode),
    /// use this method instead of DecodeLabels. It returns a list where each element is the labels
    /// for one sentence.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when batchLabelIndices is not rank-2.</exception>
    public IReadOnlyList<string[]> DecodeLabelsBatch(Tensor<T> batchLabelIndices)
    {
        if (batchLabelIndices.Rank != 2)
            throw new ArgumentException(
                $"DecodeLabelsBatch expects a 2D tensor [batch, sequenceLength]. Got rank {batchLabelIndices.Rank}. " +
                "For single sequences, use DecodeLabels instead.");

        int batchSize = batchLabelIndices.Shape[0];
        int seqLen = batchLabelIndices.Shape[1];
        var results = new List<string[]>(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            var labels = new string[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                int idx = (int)NumOps.ToDouble(batchLabelIndices.Data.Span[b * seqLen + i]);
                labels[i] = idx >= 0 && idx < LabelNames.Length ? LabelNames[idx] : "O";
            }
            results.Add(labels);
        }

        return results;
    }

    /// <summary>
    /// Returns the CRF layer in the model's Layers list, or null if absent
    /// (e.g. <c>UseCRF == false</c> in options). The CRF is canonically the
    /// LAST layer in the default LSTM-CRF / CNN-BiLSTM-CRF / Transformer-CRF
    /// stack, so the search runs in reverse for the common case.
    /// </summary>
    /// <remarks>
    /// Lives on the base class so every sequence-labeling NER subclass
    /// (BiLSTMCRF, CNNBiLSTMCRF, future TransformerBiLSTMCRF, ...) shares
    /// the same lookup contract instead of each copy-pasting the same loop.
    /// The search is model-agnostic; it only depends on the layer-list
    /// shape exposed by <see cref="NeuralNetworkBase{T}.Layers"/>.
    /// </remarks>
    protected ConditionalRandomFieldLayer<T>? FindCrfLayer()
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (Layers[i] is ConditionalRandomFieldLayer<T> crf)
                return crf;
        }
        return null;
    }

    /// <summary>
    /// Pads or truncates a labels tensor along the sequence axis so its
    /// length matches the CRF's locked sequence length (<paramref name="targetSeqLen"/>).
    /// </summary>
    /// <param name="labels">Gold-label tensor. Rank-1 <c>[seqLen]</c> for a
    /// single example; rank-2 <c>[batch, seqLen]</c> for a batch. Each
    /// element is an integer class index in <c>[0, NumLabels)</c>, NOT a
    /// one-hot or multi-label encoding — the CRF NLL path interprets
    /// values as scalar label indices.</param>
    /// <param name="targetSeqLen">The sequence length the CRF layer expects
    /// (typically <c>MaxSequenceLength</c>). Sequences shorter than this
    /// are right-padded with zeros (label 0 = O in the default CoNLL BIO
    /// scheme); sequences longer are truncated.</param>
    /// <remarks>
    /// Lives on the base class so every sequence-labeling NER subclass
    /// shares the same labels-padding contract. The rank-2 path
    /// deliberately treats axis 0 as <b>batch</b> and axis 1 as
    /// <b>sequence</b> (matching <c>ComputeNegativeLogLikelihood</c>'s
    /// <c>[batch, seqLen]</c> contract). The prior duplicated copies in
    /// each subclass mis-interpreted rank-2 as <c>[seqLen, numLabels]</c>
    /// one-hot, which silently mangled batched labels before they hit
    /// the CRF NLL path — see PR #1356 review comment from CodeRabbit
    /// for the failure mode.
    /// </remarks>
    protected virtual Tensor<T> PreprocessLabels(Tensor<T> labels, int targetSeqLen)
    {
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (labels.Rank == 0) return labels;

        if (labels.Rank == 1)
        {
            int labelLen = labels.Shape[0];
            if (labelLen == targetSeqLen) return labels;
            var padded = new Tensor<T>([targetSeqLen]);
            int copyLen = Math.Min(labelLen, targetSeqLen);
            for (int i = 0; i < copyLen; i++)
                padded[i] = labels[i];
            return padded;
        }

        if (labels.Rank == 2)
        {
            // Rank-2 labels are [batch, seqLen] (matches the CRF NLL
            // contract). Pad the second axis to targetSeqLen.
            int batch = labels.Shape[0];
            int labelLen = labels.Shape[1];
            if (labelLen == targetSeqLen) return labels;
            var padded2 = new Tensor<T>([batch, targetSeqLen]);
            int copyLen2 = Math.Min(labelLen, targetSeqLen);
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < copyLen2; t++)
                    padded2[b, t] = labels[b, t];
            return padded2;
        }

        throw new ArgumentException(
            $"PreprocessLabels expects rank-1 [seqLen] or rank-2 [batch, seqLen]; got rank {labels.Rank}.",
            nameof(labels));
    }

    /// <summary>
    /// Runs one CRF-aware (when <paramref name="useCrf"/> is true and a CRF
    /// layer is present) or cross-entropy training step. Shared between the
    /// synchronous <c>Train(...)</c> entry point and the asynchronous
    /// <c>INERModel{T}.TrainAsync(...)</c> loop so both code paths use the
    /// same loss objective. Returns the scalar loss value as a double.
    /// </summary>
    /// <param name="input">Token embeddings (raw, NOT preprocessed).</param>
    /// <param name="expected">Gold labels (raw, NOT preprocessed).</param>
    /// <param name="useCrf">Whether to route through CRF NLL when a CRF
    /// layer is present in <c>Layers</c>. False falls through to standard
    /// tape-based cross-entropy training.</param>
    /// <param name="optimizer">Optimizer to step. Required.</param>
    /// <remarks>
    /// Sets training mode to true for the duration of the step and restores
    /// it to false in a finally block, mirroring the contract Train and
    /// TrainAsync both want. Preprocesses input + labels via the virtual
    /// <see cref="PreprocessLabels"/> + the subclass's <c>PreprocessTokens</c>
    /// override.
    /// </remarks>
    protected double RunCrfAwareTrainStep(
        Tensor<T> input,
        Tensor<T> expected,
        bool useCrf,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));

        SetTrainingMode(true);
        try
        {
            var preprocessedInput = PreprocessTokens(input);
            int targetSeqLen = preprocessedInput.Rank == 3
                ? preprocessedInput.Shape[1]
                : preprocessedInput.Shape[0];
            var preprocessedExpected = PreprocessLabels(expected, targetSeqLen);

            if (useCrf)
            {
                var crfLayer = FindCrfLayer();
                if (crfLayer is not null)
                {
                    // CRF-aware: log-partition − gold-score on the
                    // emissions tensor + gold labels. Backprop flows
                    // into upstream layers AND the CRF's own
                    // transition / start / end scores.
                    T crfLoss = TrainWithCustomLoss(
                        preprocessedInput,
                        emissions => crfLayer.ComputeNegativeLogLikelihood(emissions, preprocessedExpected),
                        optimizer);
                    return NumOps.ToDouble(crfLoss);
                }
                // No CRF in the layer stack despite useCrf=true — fall
                // through to standard cross-entropy. Defensive; shouldn't
                // happen with the default LayerHelper factories.
            }

            TrainWithTape(preprocessedInput, preprocessedExpected);
            // LastLoss is nullable on the base; coerce to double via
            // NumOps with a zero fallback when the tape path didn't
            // record one (rare — typically only when the model has no
            // trainable parameters at all). Pattern-matched so we
            // don't need the null-forgiving operator.
            return LastLoss is { } loss ? NumOps.ToDouble(loss) : 0.0;
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Mirror NeuralNetworkBase.Predict's contract:
        //   1. NoGradScope so inference doesn't grow the autodiff tape and
        //      so tape-tracked ops in stateful layers (CRF, etc.) take their
        //      inference-only code paths.
        //   2. SetTrainingMode(false) so stateful layers (Dropout / BatchNorm /
        //      GaussianNoise / etc.) behave deterministically. Without this,
        //      a freshly-constructed model (IsTrainingMode defaults to true on
        //      every LayerBase) emits a fresh random Dropout mask on every
        //      Predict call — making PredictLabels non-deterministic across
        //      successive calls with identical inputs. We override Predict to
        //      route through PredictLabels (so the BiLSTMCRF preprocess/CRF
        //      pipeline runs end-to-end), which bypasses the base Predict's
        //      mode-flip; restore the prior mode in finally so calling
        //      Predict mid-training-loop doesn't permanently flip the
        //      network out of training mode.
        using var _ = new NoGradScope<T>();
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return PredictLabels(input);
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }
}
