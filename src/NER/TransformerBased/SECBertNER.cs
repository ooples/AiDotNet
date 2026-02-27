using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// SEC-BERT-NER: Securities and Exchange Commission domain BERT for NER in regulatory filings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SEC-BERT-NER (Loukas et al., EMNLP 2022 - "FiNER: Financial Numeric Entity Recognition for
/// XBRL Tagging") and related work uses BERT pre-trained specifically on SEC EDGAR filings
/// for financial regulatory NER.
///
/// <b>Pre-training Data:</b>
/// - SEC EDGAR filings: 10-K, 10-Q, 8-K, proxy statements, prospectuses
/// - ~250K filing documents spanning 1993-2023
/// - Financial regulatory language with strict formatting conventions
///
/// <b>SEC-Specific NER Entity Types:</b>
/// - <b>XBRL Tags:</b> us-gaap:Revenue, us-gaap:NetIncomeLoss, dei:EntityRegistrantName
/// - <b>Filing Entity:</b> Registrant names, CIK numbers
/// - <b>Monetary Values:</b> Amounts in SEC-mandated formats ($X,XXX)
/// - <b>Date References:</b> Filing dates, fiscal periods, reporting dates
/// - <b>Regulatory References:</b> Item numbers, exhibit references, rule citations
/// - <b>Financial Statements:</b> Balance sheet items, income statement items
///
/// <b>Why SEC-specific NER matters:</b>
/// SEC filings follow strict formatting rules and use specialized terminology. A filing might
/// reference "Item 7" (MD&amp;A section) or "Exhibit 31.1" (SOX certification). SEC-BERT
/// understands these domain-specific patterns that general financial models miss.
///
/// <b>Performance:</b>
/// - XBRL entity recognition: ~85-88% F1 (vs general BERT ~75-78%)
/// - Filing entity extraction: ~91-93% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> SEC-BERT is a specialized version of BERT trained on SEC regulatory
/// filings (10-K, 10-Q reports that public companies must file). It excels at extracting
/// entities from these highly structured financial documents. Use SEC-BERT-NER when processing
/// SEC filings, XBRL documents, or regulatory financial text.
/// </para>
/// </remarks>
public class SECBertNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a SEC-BERT-NER model in ONNX inference mode.
    /// </summary>
    public SECBertNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "SEC-BERT-NER", "Loukas et al., EMNLP 2022")
    {
    }

    /// <summary>
    /// Creates a SEC-BERT-NER model in native training mode.
    /// </summary>
    public SECBertNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "SEC-BERT-NER", "Loukas et al., EMNLP 2022", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new SECBertNER<T>(Architecture, p, optionsCopy);
        return new SECBertNER<T>(Architecture, optionsCopy);
    }
}
