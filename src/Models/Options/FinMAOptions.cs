using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinMA (Financial Multi-Agent) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinMA is a financial multi-agent LLM system designed to handle complex financial
/// tasks through specialized agent coordination.
/// </para>
/// <para><b>For Beginners:</b> FinMA uses multi-agent approach for financial NLP:
///
/// <b>The Key Insight:</b>
/// Complex financial tasks often require multiple capabilities (sentiment analysis,
/// entity extraction, numerical reasoning). FinMA coordinates specialized agents
/// to handle different aspects of a financial analysis task.
///
/// <b>What Problems Does FinMA Solve?</b>
/// - Multi-faceted financial document analysis
/// - Complex financial question answering
/// - Coordinated analysis of earnings reports
/// - Integrated sentiment and entity extraction
/// - Financial reasoning with multiple information sources
///
/// <b>Key Benefits:</b>
/// - Specialized agents for different tasks
/// - Better handling of complex queries
/// - Modular and extensible architecture
/// - Can combine multiple analysis types
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "FinMA: A Multi-Agent Financial LLM System", 2024.
/// </para>
/// </remarks>
public class FinMAOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default FinMA configuration.
    /// </summary>
    public FinMAOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinMAOptions(FinMAOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumClasses = other.NumClasses;
        DropoutRate = other.DropoutRate;
        NumAgents = other.NumAgents;
    }

    /// <summary>Maximum sequence length in tokens (default: 2048).</summary>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 32000 for LLaMA-style).</summary>
    public int VocabularySize { get; set; } = 32000;

    /// <summary>Hidden dimension (default: 768).</summary>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Number of attention heads (default: 12).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Intermediate feed-forward dimension (default: 3072).</summary>
    public int IntermediateDimension { get; set; } = 3072;

    /// <summary>Number of transformer layers (default: 12).</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Number of output classes (default: 3).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Number of specialized agents (default: 4).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> FinMA uses multiple specialized agents:
    /// - Sentiment Agent: Analyzes sentiment
    /// - Entity Agent: Extracts financial entities
    /// - Reasoning Agent: Performs numerical reasoning
    /// - Summary Agent: Generates summaries
    /// </para>
    /// </remarks>
    public int NumAgents { get; set; } = 4;
}
