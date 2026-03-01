using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// FinBERT-NER: Financial domain BERT for Named Entity Recognition in financial text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinBERT-NER (Araci, 2019 - "FinBERT: Financial Sentiment Analysis with Pre-trained Language
/// Models"; Yang et al., IJCAI 2020 - "FinBERT: A Pretrained Language Model for Financial
/// Communications") is BERT further pre-trained on large-scale financial corpora for
/// domain-specific NLP tasks including financial NER.
///
/// <b>Pre-training Data:</b>
/// - Financial news articles (Reuters, Bloomberg)
/// - SEC filings (10-K, 10-Q, 8-K reports)
/// - Financial analyst reports
/// - Earnings call transcripts
/// - ~4.9B tokens of financial text
///
/// <b>Financial NER Entity Types:</b>
/// - <b>Company:</b> Apple Inc., Goldman Sachs, Tesla
/// - <b>Person:</b> CEOs, CFOs, board members, analysts
/// - <b>Financial Metric:</b> Revenue, EBITDA, P/E ratio, market cap
/// - <b>Currency/Amount:</b> $1.5 billion, EUR 200 million
/// - <b>Date/Period:</b> Q3 2023, fiscal year 2024, YoY
/// - <b>Regulation:</b> Dodd-Frank, Basel III, SOX
/// - <b>Index/Ticker:</b> S&amp;P 500, NASDAQ, AAPL
///
/// <b>Why Financial NER Needs Domain Models:</b>
/// Financial text has unique challenges: "Apple" is a company (not fruit), "bearish" describes
/// market sentiment (not an animal), and "spread" refers to yield difference (not butter).
/// FinBERT's financial pre-training resolves these ambiguities that confuse general BERT.
///
/// <b>Performance:</b>
/// - Financial NER: ~90-92% F1 (vs general BERT ~85-87% on financial text)
/// - Financial sentiment analysis: ~88% accuracy (state-of-the-art)
/// </para>
/// <para>
/// <b>For Beginners:</b> FinBERT is BERT trained on financial documents like SEC filings,
/// earnings reports, and financial news. It understands financial jargon, company names,
/// and financial metrics better than general BERT. Use FinBERT-NER for extracting entities
/// from financial text: company names, ticker symbols, financial figures, regulatory terms.
/// </para>
/// </remarks>
public class FinBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a FinBERT-NER model in ONNX inference mode.
    /// </summary>
    public FinBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "FinBERT-NER", "Yang et al., IJCAI 2020")
    {
    }

    /// <summary>
    /// Creates a FinBERT-NER model in native training mode.
    /// </summary>
    public FinBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "FinBERT-NER", "Yang et al., IJCAI 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new FinBERTNER<T>(Architecture, p, optionsCopy);
        return new FinBERTNER<T>(Architecture, optionsCopy);
    }
}
