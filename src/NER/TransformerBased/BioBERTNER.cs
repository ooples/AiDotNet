using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// BioBERT-NER: Biomedical domain-specific BERT for Named Entity Recognition in biomedical text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BioBERT-NER (Lee et al., Bioinformatics 2020 - "BioBERT: a pre-trained biomedical language
/// representation model for biomedical text mining") is BERT pre-trained on large-scale biomedical
/// corpora for domain-specific NER tasks like gene, protein, disease, drug, and species recognition.
///
/// <b>Pre-training Data:</b>
/// - PubMed abstracts: ~4.5B words from 30M+ biomedical abstracts
/// - PMC full-text articles: ~13.5B words from open-access articles
/// - Initialized from BERT-base weights, then continued pre-training on biomedical text
///
/// <b>Biomedical NER Tasks:</b>
/// - <b>Gene/Protein NER:</b> BC2GM (87.5% F1), JNLPBA (79.3% F1)
/// - <b>Disease NER:</b> NCBI Disease (89.4% F1), BC5CDR-Disease (87.2% F1)
/// - <b>Drug/Chemical NER:</b> BC5CDR-Chemical (93.4% F1), BC4CHEMD (92.4% F1)
/// - <b>Species NER:</b> Species-800 (74.4% F1), LINNAEUS (88.2% F1)
///
/// <b>Why domain-specific pre-training matters:</b>
/// General BERT understands "IL-6" as just tokens, but BioBERT understands it as interleukin-6,
/// a cytokine. This domain knowledge comes from reading millions of biomedical papers during
/// pre-training, enabling BioBERT to recognize biomedical entities that general BERT misses.
/// </para>
/// <para>
/// <b>For Beginners:</b> BioBERT is BERT that has been additionally trained on millions of
/// biomedical research papers. It understands medical/scientific terminology better than general
/// BERT. Use BioBERT-NER when you need to extract entities from biomedical text (diseases,
/// drugs, genes, proteins, chemicals). Also consider PubMedBERT for even better biomedical NER.
/// </para>
/// </remarks>
public class BioBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a BioBERT-NER model in ONNX inference mode.
    /// </summary>
    public BioBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "BioBERT-NER", "Lee et al., Bioinformatics 2020")
    {
    }

    /// <summary>
    /// Creates a BioBERT-NER model in native training mode.
    /// </summary>
    public BioBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "BioBERT-NER", "Lee et al., Bioinformatics 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new BioBERTNER<T>(Architecture, p, optionsCopy);
        return new BioBERTNER<T>(Architecture, optionsCopy);
    }
}
