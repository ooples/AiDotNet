using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// SciBERT-NER: Scientific domain BERT for Named Entity Recognition in scientific literature.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SciBERT-NER (Beltagy et al., EMNLP 2019 - "SciBERT: A Pretrained Language Model for
/// Scientific Text") is BERT pre-trained from scratch on 1.14M scientific papers from
/// Semantic Scholar, covering computer science (18%) and biomedical (82%) domains.
///
/// <b>Key Differences from BioBERT:</b>
/// - <b>Pre-trained from scratch:</b> SciBERT uses its own vocabulary (SciVocab) built from
///   scientific text, while BioBERT initializes from BERT's vocabulary
/// - <b>Domain-specific vocabulary:</b> SciVocab (31K tokens) contains scientific terms that
///   BERT's WordPiece vocabulary would split into many subwords
/// - <b>Broader scientific coverage:</b> Includes computer science papers, not just biomedical
///
/// <b>Scientific NER Tasks:</b>
/// - <b>SciERC:</b> Scientific entities (Task, Method, Metric, Material, Generic, OtherScientificTerm)
/// - <b>JNLPBA:</b> Biomedical entities (Protein, DNA, RNA, Cell Line, Cell Type) - 77.3% F1
/// - <b>BC5CDR:</b> Chemical and disease entities - 90.0% F1
/// - <b>NCBI Disease:</b> Disease mention recognition - 88.6% F1
///
/// <b>Architecture:</b>
/// Same as BERT-base (12 layers, 768 hidden, 12 heads, 110M params) but with SciVocab
/// tokenization. The key insight is that domain-specific vocabulary matters: "immunoglobulin"
/// is one token in SciVocab but ["im", "##mun", "##og", "##lo", "##bul", "##in"] in BERT.
/// </para>
/// <para>
/// <b>For Beginners:</b> SciBERT is BERT pre-trained entirely on scientific papers. It has a
/// specialized vocabulary designed for scientific terms. Use SciBERT-NER for extracting entities
/// from scientific literature, especially when working across both biomedical and computer
/// science domains. For purely biomedical NER, BioBERT or PubMedBERT may perform slightly better.
/// </para>
/// </remarks>
public class SciBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a SciBERT-NER model in ONNX inference mode.
    /// </summary>
    public SciBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "SciBERT-NER", "Beltagy et al., EMNLP 2019")
    {
    }

    /// <summary>
    /// Creates a SciBERT-NER model in native training mode.
    /// </summary>
    public SciBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "SciBERT-NER", "Beltagy et al., EMNLP 2019", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new SciBERTNER<T>(Architecture, p, optionsCopy);
        return new SciBERTNER<T>(Architecture, optionsCopy);
    }
}
