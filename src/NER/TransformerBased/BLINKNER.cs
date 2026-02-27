using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// BLINK: BERT-based bi-encoder for entity linking and Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLINK (Wu et al., EMNLP 2020 - "Scalable Zero-shot Entity Linking with Dense Entity Retrieval")
/// is a bi-encoder architecture for entity linking that uses BERT to encode both entity mentions
/// and entity descriptions, then links mentions to knowledge base entries via dense retrieval.
///
/// <b>Key Innovation - Bi-Encoder Architecture:</b>
/// BLINK uses two independent BERT encoders:
/// 1. <b>Mention Encoder:</b> Encodes the entity mention with its surrounding context
///    Input: "[CLS] left context [Ms] mention [Me] right context [SEP]"
///    Output: 768-dimensional mention embedding
/// 2. <b>Entity Encoder:</b> Encodes each knowledge base entity using its title and description
///    Input: "[CLS] entity_title [ENT] entity_description [SEP]"
///    Output: 768-dimensional entity embedding
///
/// <b>Two-Stage Linking:</b>
/// - <b>Stage 1 - Candidate Retrieval:</b> Use FAISS (approximate nearest neighbor search) to
///   find the top-K entity candidates whose embeddings are closest to the mention embedding.
///   This is extremely fast (milliseconds for millions of entities) because entity embeddings
///   are pre-computed and indexed.
/// - <b>Stage 2 - Cross-Encoder Re-ranking:</b> A cross-encoder (BERT that sees both mention
///   and entity description together) re-ranks the top-K candidates for final prediction.
///
/// <b>Zero-Shot Entity Linking:</b>
/// BLINK can link to entities never seen during training because it compares mention
/// representations with entity description representations in a shared embedding space.
/// If a new entity is added to the knowledge base with a description, BLINK can link to
/// it without retraining.
///
/// <b>Performance:</b>
/// - AIDA-CoNLL (in-domain): ~87.5% accuracy
/// - Zero-shot (unseen entities): ~82.3% accuracy
/// - Cross-lingual entity linking: ~78-85% accuracy
///
/// <b>NER + Entity Linking Pipeline:</b>
/// In the combined pipeline, a separate NER model (e.g., BERT-NER) first identifies entity
/// mentions, then BLINK links each mention to its corresponding knowledge base entry.
/// </para>
/// <para>
/// <b>For Beginners:</b> BLINK goes beyond just finding entity mentions - it links them to
/// specific entries in a knowledge base (like Wikipedia). For example, given "Obama visited
/// Paris", it not only finds "Obama" (person) and "Paris" (location) but links "Obama" to
/// the Wikipedia article about Barack Obama and "Paris" to the article about Paris, France
/// (not Paris, Texas). This disambiguation is done by comparing embeddings of the mention
/// context with embeddings of entity descriptions.
/// </para>
/// </remarks>
public class BLINKNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a BLINK model in ONNX inference mode.
    /// </summary>
    public BLINKNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "BLINK", "Wu et al., EMNLP 2020")
    {
    }

    /// <summary>
    /// Creates a BLINK model in native training mode.
    /// </summary>
    public BLINKNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "BLINK", "Wu et al., EMNLP 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new BLINKNER<T>(Architecture, p, optionsCopy);
        return new BLINKNER<T>(Architecture, optionsCopy);
    }
}
