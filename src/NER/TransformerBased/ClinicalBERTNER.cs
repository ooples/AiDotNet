using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// ClinicalBERT-NER: Clinical domain BERT for Named Entity Recognition in clinical notes and EHRs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ClinicalBERT-NER (Alsentzer et al., NAACL 2019 Clinical NLP Workshop - "Publicly Available
/// Clinical BERT Embeddings"; Huang et al., 2019 - "ClinicalBERT: Modeling Clinical Notes and
/// Predicting Hospital Readmission") is BERT further pre-trained on clinical text from
/// electronic health records (EHRs) for clinical NLP tasks.
///
/// <b>Pre-training Data:</b>
/// - MIMIC-III clinical notes (~880M words from 2M+ clinical notes)
/// - Discharge summaries, radiology reports, nursing notes
/// - Physician progress notes, operative reports
/// - Initialized from BioBERT weights (which were initialized from BERT)
///
/// <b>Clinical NER Entity Types:</b>
/// - <b>Problem/Diagnosis:</b> Type 2 diabetes mellitus, acute myocardial infarction
/// - <b>Treatment:</b> Metformin 500mg, coronary artery bypass graft
/// - <b>Test:</b> Complete blood count, chest X-ray, echocardiogram
/// - <b>Anatomy:</b> Left ventricle, right lower lobe, anterior cruciate ligament
/// - <b>Dosage:</b> 500mg BID, 10mg/kg/day
/// - <b>Duration:</b> 7-day course, q6h for 48 hours
/// - <b>Temporal:</b> Post-operative day 3, on admission, at discharge
///
/// <b>Why Clinical NER is Different:</b>
/// Clinical text is uniquely challenging: heavy abbreviations (SOB = shortness of breath, not
/// an expletive), misspellings, fragmented sentences, negation (patient denies chest pain),
/// and domain-specific jargon (PRN, NPO, PERRLA). ClinicalBERT handles these patterns.
///
/// <b>Performance:</b>
/// - i2b2 2010 Clinical NER: ~88.5% F1 (vs BERT ~84.3%, BioBERT ~86.1%)
/// - i2b2 2012 Temporal NER: ~85.2% F1
/// - n2c2 2018 Medication NER: ~91.3% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> ClinicalBERT is BERT trained on clinical notes from hospitals. It
/// understands medical abbreviations, drug names, diagnoses, and the unique writing style of
/// doctors and nurses. Use ClinicalBERT-NER for extracting medical entities from clinical notes,
/// discharge summaries, or electronic health records. Note: HIPAA compliance and data privacy
/// are critical when processing real clinical data.
/// </para>
/// </remarks>
public class ClinicalBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a ClinicalBERT-NER model in ONNX inference mode.
    /// </summary>
    public ClinicalBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "ClinicalBERT-NER", "Alsentzer et al., NAACL 2019 Clinical NLP Workshop")
    {
    }

    /// <summary>
    /// Creates a ClinicalBERT-NER model in native training mode.
    /// </summary>
    public ClinicalBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "ClinicalBERT-NER", "Alsentzer et al., NAACL 2019 Clinical NLP Workshop", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new ClinicalBERTNER<T>(Architecture, p, optionsCopy);
        return new ClinicalBERTNER<T>(Architecture, optionsCopy);
    }
}
