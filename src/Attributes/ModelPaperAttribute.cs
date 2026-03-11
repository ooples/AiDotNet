namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the academic paper(s) that introduced or describe a model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to your model class to reference the research
/// paper that describes how the model works. This gives users a way to understand the
/// theory behind the model and verify its correctness. You can apply it multiple times
/// if the model is based on multiple papers.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelPaper("Attention Is All You Need", "https://arxiv.org/abs/1706.03762")]
/// public class Transformer&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ModelPaperAttribute : Attribute
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
    /// Initializes a new instance of the <see cref="ModelPaperAttribute"/> class.
    /// </summary>
    /// <param name="title">The title of the academic paper.</param>
    /// <param name="url">The URL where the paper can be accessed.</param>
    public ModelPaperAttribute(string title, string url)
    {
        Title = title;
        Url = url;
    }
}
