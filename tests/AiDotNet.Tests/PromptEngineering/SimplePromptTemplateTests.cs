using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class SimplePromptTemplateTests
{
    [Fact]
    public void Constructor_WithValidTemplate_Succeeds()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");

        Assert.NotNull(template);
        Assert.Equal("Translate {text} to {language}", template.Template);
        Assert.Equal(2, template.InputVariables.Count);
        Assert.Contains("text", template.InputVariables);
        Assert.Contains("language", template.InputVariables);
    }

    [Fact]
    public void Constructor_WithNullTemplate_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new SimplePromptTemplate(null!));
    }

    [Fact]
    public void Constructor_WithEmptyTemplate_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new SimplePromptTemplate(""));
        Assert.Throws<ArgumentException>(() => new SimplePromptTemplate("   "));
    }

    [Fact]
    public void Format_WithAllVariables_ReturnsFormattedString()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello world",
            ["language"] = "Spanish"
        };

        var result = template.Format(variables);

        Assert.Equal("Translate Hello world to Spanish", result);
    }

    [Fact]
    public void Format_WithMissingVariable_ThrowsArgumentException()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello world"
            // Missing "language"
        };

        Assert.Throws<ArgumentException>(() => template.Format(variables));
    }

    [Fact]
    public void Format_WithNullVariables_ThrowsArgumentNullException()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");

        Assert.Throws<ArgumentNullException>(() => template.Format(null!));
    }

    [Fact]
    public void Validate_WithAllVariables_ReturnsTrue()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello world",
            ["language"] = "Spanish"
        };

        var result = template.Validate(variables);

        Assert.True(result);
    }

    [Fact]
    public void Validate_WithMissingVariable_ReturnsFalse()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello world"
        };

        var result = template.Validate(variables);

        Assert.False(result);
    }

    [Fact]
    public void Validate_WithNullVariables_ReturnsFalse()
    {
        var template = new SimplePromptTemplate("Translate {text} to {language}");

        var result = template.Validate(null!);

        Assert.False(result);
    }

    [Fact]
    public void FromTemplate_CreatesValidInstance()
    {
        var template = SimplePromptTemplate.FromTemplate("Summarize {document}");

        Assert.NotNull(template);
        Assert.Equal("Summarize {document}", template.Template);
        Assert.Single(template.InputVariables);
        Assert.Contains("document", template.InputVariables);
    }

    [Fact]
    public void Format_WithMultipleOccurrencesOfSameVariable_ReplacesAll()
    {
        var template = new SimplePromptTemplate("Compare {item} with {item} carefully");
        var variables = new Dictionary<string, string>
        {
            ["item"] = "apple"
        };

        var result = template.Format(variables);

        Assert.Equal("Compare apple with apple carefully", result);
    }
}
