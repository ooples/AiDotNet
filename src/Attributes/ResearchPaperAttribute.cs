namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the academic paper(s) that introduced or describe a model, component, or algorithm.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to any class to reference the research paper
/// that describes how it works. This gives users a way to understand the theory and verify
/// correctness. You can apply it multiple times for classes based on multiple papers.
/// </para>
/// <para>
/// This attribute works for all three metadata tiers:
/// - Tier 1 (Models): "Attention Is All You Need" on Transformer
/// - Tier 2 (Components): "ColBERT: Efficient and Effective Passage Search" on ColBERTRetriever
/// - Tier 3 (Infrastructure): "Billion-scale similarity search with GPUs" on FaissIndex
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ResearchPaper("Attention Is All You Need", "https://arxiv.org/abs/1706.03762", Year = 2017)]
/// public class Transformer&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ResearchPaperAttribute : Attribute
{
    /// <summary>
    /// Gets the title of the paper.
    /// </summary>
    public string Title { get; }

    /// <summary>
    /// Gets the URL where the paper can be accessed (typically an arXiv or DOI link).
    /// </summary>
    public string Url { get; }

    /// <summary>
    /// Gets or sets the year the paper was published. Optional.
    /// </summary>
    public int Year { get; set; }

    /// <summary>
    /// Gets or sets the authors of the paper. Optional.
    /// </summary>
    public string Authors { get; set; } = string.Empty;

    /// <summary>
    /// Initializes a new instance of the <see cref="ResearchPaperAttribute"/> class.
    /// </summary>
    /// <param name="title">The title of the academic paper.</param>
    /// <param name="url">The URL where the paper can be accessed.</param>
    public ResearchPaperAttribute(string title, string url)
    {
        if (string.IsNullOrWhiteSpace(title))
            throw new ArgumentException("Paper title must not be empty.", nameof(title));
        if (string.IsNullOrWhiteSpace(url) || !url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Paper URL must be a valid https:// link.", nameof(url));
        Title = title;
        Url = url;
    }
}
