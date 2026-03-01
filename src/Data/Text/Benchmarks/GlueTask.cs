namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// GLUE benchmark sub-tasks.
/// </summary>
public enum GlueTask
{
    /// <summary>Corpus of Linguistic Acceptability (single sentence, binary).</summary>
    CoLA,
    /// <summary>Stanford Sentiment Treebank (single sentence, binary).</summary>
    SST2,
    /// <summary>Microsoft Research Paraphrase Corpus (sentence pair, binary).</summary>
    MRPC,
    /// <summary>Quora Question Pairs (sentence pair, binary).</summary>
    QQP,
    /// <summary>Semantic Textual Similarity Benchmark (sentence pair, regression 0-5).</summary>
    STSB,
    /// <summary>Multi-Genre Natural Language Inference (sentence pair, 3-class).</summary>
    MNLI,
    /// <summary>Question NLI (sentence pair, binary).</summary>
    QNLI,
    /// <summary>Recognizing Textual Entailment (sentence pair, binary).</summary>
    RTE,
    /// <summary>Winograd NLI (sentence pair, binary).</summary>
    WNLI
}
