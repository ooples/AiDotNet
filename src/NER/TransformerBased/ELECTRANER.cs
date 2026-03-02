using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// ELECTRA-NER: Efficiently Learning an Encoder that Classifies Token Replacements Accurately for NER.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ELECTRA-NER (Clark et al., ICLR 2020 - "ELECTRA: Pre-training Text Encoders as Discriminators
/// Rather Than Generators") uses a novel replaced token detection (RTD) pre-training objective
/// instead of masked language modeling. ELECTRA is particularly efficient because:
///
/// <b>Key Innovation - Replaced Token Detection:</b>
/// Instead of masking 15% of tokens and predicting them (like BERT), ELECTRA:
/// 1. A small "generator" network (like a tiny BERT) predicts masked tokens
/// 2. Some predicted tokens are plausible but wrong replacements
/// 3. The main "discriminator" network learns to detect which tokens were replaced
/// 4. Every token position provides a training signal (vs only 15% for BERT)
///
/// This means ELECTRA learns from ALL tokens in every sentence, making it 4x more efficient
/// than BERT at the same compute budget. An ELECTRA-small model matches BERT-base performance
/// using only 1/4 of the compute.
///
/// <b>Performance (CoNLL-2003):</b>
/// - ELECTRA-small: ~91.5% F1 (matching BERT-base with 1/4 compute)
/// - ELECTRA-base: ~92.6% F1
/// - ELECTRA-large: ~93.3% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> ELECTRA is a smart, efficient transformer that learns faster than BERT.
/// While BERT learns by filling in blank words (like a fill-in-the-blank test), ELECTRA learns
/// by detecting fake words (like a fact-checker). This is more efficient because ELECTRA learns
/// from every word in every sentence, not just the few blanked-out words.
///
/// Use ELECTRA-NER when:
/// - You want good accuracy with limited compute budget
/// - You need a smaller model that still performs well
/// - Training efficiency is important
/// </para>
/// </remarks>
public class ELECTRANER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates an ELECTRA-NER model in ONNX inference mode.
    /// </summary>
    public ELECTRANER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "ELECTRA-NER", "Clark et al., ICLR 2020")
    {
    }

    /// <summary>
    /// Creates an ELECTRA-NER model in native training mode.
    /// </summary>
    public ELECTRANER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "ELECTRA-NER", "Clark et al., ICLR 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new ELECTRANER<T>(Architecture, p, optionsCopy);
        return new ELECTRANER<T>(Architecture, optionsCopy);
    }
}
