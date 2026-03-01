using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// REL: Radboud Entity Linker - end-to-end entity linking combining NER with entity disambiguation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// REL (van Hulst et al., SIGIR 2020 - "REL: An Entity Linker Standing on the Shoulders of Giants")
/// is a modular end-to-end entity linking system that combines mention detection (NER),
/// candidate generation, and entity disambiguation into a single pipeline.
///
/// <b>Pipeline Architecture:</b>
///
/// <b>Stage 1 - Mention Detection (NER):</b>
/// - Uses Flair NER (BiLSTM-CRF with contextual string embeddings) for entity mention detection
/// - Identifies potential entity spans and their types (PER, ORG, LOC, MISC)
/// - Also supports n-gram-based mention detection for higher recall
///
/// <b>Stage 2 - Candidate Generation:</b>
/// - For each detected mention, generates candidate entities from a knowledge base (Wikipedia)
/// - Uses multiple signals: exact match, fuzzy match, prior probability P(entity|mention)
/// - Prior probabilities are computed from Wikipedia anchor text statistics
/// - Typically generates 30-100 candidates per mention
///
/// <b>Stage 3 - Entity Disambiguation (ED):</b>
/// - Uses a neural ED model to select the correct entity from candidates
/// - Features include:
///   - <b>Entity embeddings:</b> Pre-trained Wikipedia2Vec entity representations
///   - <b>Context similarity:</b> TF-IDF or neural similarity between mention context and
///     entity description
///   - <b>Coherence:</b> How well the candidate entity fits with other entities in the document
///     (collective entity linking)
///   - <b>Prior probability:</b> How likely this mention refers to this entity in general
///
/// <b>Collective Entity Linking:</b>
/// REL performs collective entity linking, where the disambiguation of one mention can
/// influence the disambiguation of others. For example, in "Jordan played basketball
/// for the Bulls", disambiguating "Bulls" as "Chicago Bulls" (sports team) helps
/// disambiguate "Jordan" as "Michael Jordan" (basketball player) rather than "Jordan"
/// (country).
///
/// <b>Performance:</b>
/// - AIDA-CoNLL: ~84.3% F1 (end-to-end, including NER)
/// - MSNBC: ~73.1% F1
/// - AQUAINT: ~82.4% F1
/// - Processing speed: ~40 documents/second on GPU
///
/// <b>API and Deployment:</b>
/// REL provides a REST API for easy deployment, making it practical for production use.
/// It supports both local and remote knowledge bases and can be extended with custom
/// entity catalogs.
/// </para>
/// <para>
/// <b>For Beginners:</b> REL is a complete system that first finds entity mentions in text
/// (NER step), then figures out which specific real-world entity each mention refers to
/// (entity linking step). For example, "Apple" could refer to Apple Inc. (the tech company)
/// or apple (the fruit). REL uses context clues and knowledge from Wikipedia to make the
/// right choice. It's like a three-step process: find mentions, generate candidates, pick
/// the best match.
/// </para>
/// </remarks>
public class RELNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a REL model in ONNX inference mode.
    /// </summary>
    public RELNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "REL", "van Hulst et al., SIGIR 2020")
    {
    }

    /// <summary>
    /// Creates a REL model in native training mode.
    /// </summary>
    public RELNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "REL", "van Hulst et al., SIGIR 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new RELNER<T>(Architecture, p, optionsCopy);
        return new RELNER<T>(Architecture, optionsCopy);
    }
}
