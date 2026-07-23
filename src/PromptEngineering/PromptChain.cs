using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering;

/// <summary>
/// Composes multiple prompt templates into a single formatted prompt.
/// </summary>
[AiDotNet.Configuration.YamlConfigurable("PromptChain")]
public class PromptChain
{
    private readonly List<IPromptTemplate> _templates = new();
    private readonly string _separator;

    /// <summary>
    /// Initializes a new prompt chain.
    /// </summary>
    /// <param name="separator">Separator inserted between formatted template outputs.</param>
    public PromptChain(string separator = "\n")
    {
        _separator = separator;
    }

    /// <summary>
    /// Adds a prompt template to the chain.
    /// </summary>
    public PromptChain Add(IPromptTemplate template)
    {
        if (template is null)
        {
            throw new ArgumentNullException(nameof(template));
        }

        _templates.Add(template);
        return this;
    }

    /// <summary>
    /// Formats all templates in insertion order and joins the outputs.
    /// </summary>
    public string Format(Dictionary<string, string> variables)
    {
        if (variables is null)
        {
            throw new ArgumentNullException(nameof(variables));
        }

        return string.Join(_separator, _templates.Select(template => template.Format(variables)));
    }

    /// <summary>
    /// Gets the templates in this chain.
    /// </summary>
    public IReadOnlyList<IPromptTemplate> Templates => _templates;
}
