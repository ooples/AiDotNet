using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.SequenceLabeling;

/// <summary>
/// Base class for sequence labeling NER models that assign a label to each token in a sequence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sequence labeling is the most common approach to NER, where each token in a sentence is
/// assigned a label from a predefined set (e.g., BIO scheme). This base class provides:
///
/// - Token-level label prediction
/// - BIO constraint enforcement via CRF decoding
/// - Emission score computation
/// - Viterbi decoding for optimal label sequences
///
/// Derived classes implement specific architectures like BiLSTM-CRF, BERT-NER, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sequence labeling NER processes text one token at a time and assigns
/// a label to each token. For example, in "John works at Google":
/// - "John" -> B-PER (Beginning of Person)
/// - "works" -> O (Outside any entity)
/// - "at" -> O
/// - "Google" -> B-ORG (Beginning of Organization)
///
/// The BIO scheme uses B- (Begin), I- (Inside), and O (Outside) prefixes to handle
/// multi-token entities like "New York" -> B-LOC, I-LOC.
/// </para>
/// </remarks>
public abstract class SequenceLabelingNERBase<T> : NERNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets whether to use CRF decoding for label sequence prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses a Conditional Random Field layer to enforce valid
    /// label transition constraints (e.g., I-PER cannot follow B-ORG).
    /// When false, labels are predicted independently per token using argmax.
    /// </para>
    /// </remarks>
    public bool UseCRF { get; protected set; } = true;

    /// <summary>
    /// Gets the label names corresponding to each label index.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default CoNLL-2003 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    /// </para>
    /// </remarks>
    public string[] LabelNames { get; protected set; } =
    [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ];

    /// <summary>
    /// Initializes a new instance of the SequenceLabelingNERBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, cross-entropy loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected SequenceLabelingNERBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Predicts the best label sequence for input token embeddings.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings [sequenceLength, embeddingDim] or [batch, sequenceLength, embeddingDim].</param>
    /// <returns>Predicted label indices [sequenceLength] or [batch, sequenceLength].</returns>
    public abstract Tensor<T> PredictLabels(Tensor<T> tokenEmbeddings);

    /// <summary>
    /// Computes emission scores from token embeddings.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings [sequenceLength, embeddingDim].</param>
    /// <returns>Emission scores [sequenceLength, numLabels].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Emission scores tell the model how likely each label is for
    /// each token, based on the token's features alone (before considering label transitions).
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> ComputeEmissionScores(Tensor<T> tokenEmbeddings);

    /// <summary>
    /// Performs argmax decoding on emission scores to get label predictions.
    /// Used when CRF decoding is disabled.
    /// </summary>
    /// <param name="emissionScores">Emission scores [sequenceLength, numLabels].</param>
    /// <returns>Label indices [sequenceLength].</returns>
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
    /// Converts predicted label indices to label name strings.
    /// </summary>
    /// <param name="labelIndices">Label index tensor [sequenceLength].</param>
    /// <returns>Array of label name strings.</returns>
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
