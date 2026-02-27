using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// Legal-BERT-NER: Legal domain BERT for Named Entity Recognition in legal documents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Legal-BERT-NER (Chalkidis et al., EMNLP 2020 Findings - "LEGAL-BERT: The Muppets straight
/// out of Law School") is BERT pre-trained on 12GB of diverse English legal text for
/// domain-specific legal NLP tasks including NER.
///
/// <b>Pre-training Data:</b>
/// - EU legislation (EU Acquis, treaties, regulations, directives)
/// - US court opinions (case law from all federal courts)
/// - US contracts (EDGAR-sourced commercial contracts)
/// - Legal academic papers (selected law review articles)
/// - ~12GB total, ~2.5B tokens of legal text
///
/// <b>Legal NER Entity Types:</b>
/// - <b>Court:</b> Supreme Court, District Court, European Court of Justice
/// - <b>Judge:</b> Justice Roberts, Judge Smith
/// - <b>Legislation:</b> Article 5(1), Section 230, Title VII
/// - <b>Citation:</b> Brown v. Board of Education, 347 U.S. 483 (1954)
/// - <b>Party:</b> Plaintiff, defendant, appellant names
/// - <b>Legal Concept:</b> Due process, habeas corpus, res judicata
/// - <b>Jurisdiction:</b> State of New York, European Union, federal
///
/// <b>Why Legal NER Needs Domain Models:</b>
/// Legal text has unique structure: nested citations ("see also 42 U.S.C. 1983"), Latin terms
/// (habeas corpus, stare decisis), and specific entity patterns (case citations like
/// "548 F.3d 290 (2d Cir. 2008)"). Legal-BERT understands these patterns.
///
/// <b>Performance:</b>
/// - Legal NER: ~88-91% F1 (vs general BERT ~82-85% on legal text)
/// - Contract entity extraction: ~86-89% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> Legal-BERT is BERT trained on court opinions, contracts, and legislation.
/// It understands legal jargon, citation formats, and legal entity types that general BERT
/// struggles with. Use Legal-BERT-NER for processing contracts, court filings, regulations,
/// or any legal documents where entity extraction is needed.
/// </para>
/// </remarks>
public class LegalBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a Legal-BERT-NER model in ONNX inference mode.
    /// </summary>
    public LegalBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "Legal-BERT-NER", "Chalkidis et al., EMNLP 2020 Findings")
    {
    }

    /// <summary>
    /// Creates a Legal-BERT-NER model in native training mode.
    /// </summary>
    public LegalBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "Legal-BERT-NER", "Chalkidis et al., EMNLP 2020 Findings", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new LegalBERTNER<T>(Architecture, p, optionsCopy);
        return new LegalBERTNER<T>(Architecture, optionsCopy);
    }
}
