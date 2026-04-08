#nullable disable
using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class ChainOfThoughtTemplateTests
{
    [Fact]
    public void Constructor_WithQuestion_CreatesTemplate()
    {
        // Use question constructor by specifying context parameter
        var template = new ChainOfThoughtTemplate("What is 2 + 2?", context: null);

        Assert.NotNull(template);
        Assert.Contains("What is 2 + 2?", template.Template);
        Assert.Contains("step by step", template.Template.ToLower());
    }

    [Fact]
    public void Constructor_WithQuestionAndContext_IncludesContext()
    {
        var template = new ChainOfThoughtTemplate("What is 2 + 2?", "This is a math problem.");

        Assert.NotNull(template);
        Assert.Contains("What is 2 + 2?", template.Template);
        Assert.Contains("This is a math problem.", template.Template);
    }

    [Fact]
    public void Constructor_WithExamples_IncludesExamples()
    {
        var examples = new[]
        {
            new ChainOfThoughtExample("What is 1 + 1?", "1 + 1 equals 2", "2"),
            new ChainOfThoughtExample("What is 3 + 3?", "3 + 3 equals 6", "6")
        };

        var template = new ChainOfThoughtTemplate("What is 5 + 5?", examples);

        Assert.NotNull(template);
        Assert.Contains("What is 1 + 1?", template.Template);
        Assert.Contains("What is 3 + 3?", template.Template);
        Assert.Contains("What is 5 + 5?", template.Template);
    }

    [Fact]
    public void Constructor_WithCustomTemplate_UsesCustomTemplate()
    {
        var customTemplate = "Custom reasoning: {question}";
        var template = new ChainOfThoughtTemplate(customTemplate);

        Assert.Equal(customTemplate, template.Template);
    }

    [Fact]
    public void Format_ReturnsFormattedTemplate()
    {
        var template = new ChainOfThoughtTemplate("What is {x} + {y}?");
        var variables = new Dictionary<string, string>
        {
            ["x"] = "5",
            ["y"] = "3"
        };

        var result = template.Format(variables);

        Assert.Contains("5", result);
        Assert.Contains("3", result);
    }

    [Fact]
    public void ChainOfThoughtExample_PropertiesWork()
    {
        var example = new ChainOfThoughtExample
        {
            Question = "Test question",
            Reasoning = "Test reasoning",
            Answer = "Test answer"
        };

        Assert.Equal("Test question", example.Question);
        Assert.Equal("Test reasoning", example.Reasoning);
        Assert.Equal("Test answer", example.Answer);
    }

    [Fact]
    public void ChainOfThoughtExample_ConstructorWithValues()
    {
        var example = new ChainOfThoughtExample("Q", "R", "A");

        Assert.Equal("Q", example.Question);
        Assert.Equal("R", example.Reasoning);
        Assert.Equal("A", example.Answer);
    }

    [Fact]
    public void Builder_CreatesTemplateWithQuestion()
    {
        var template = ChainOfThoughtTemplate.Builder()
            .WithQuestion("What is the capital of France?")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("What is the capital of France?", template.Template);
    }

    [Fact]
    public void Builder_CreatesTemplateWithContext()
    {
        var template = ChainOfThoughtTemplate.Builder()
            .WithQuestion("What is the capital?")
            .WithContext("We are discussing European countries.")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("European countries", template.Template);
    }

    [Fact]
    public void Builder_AddExample_IncludesExample()
    {
        var template = ChainOfThoughtTemplate.Builder()
            .WithQuestion("What is 10 / 2?")
            .AddExample("What is 4 / 2?", "4 divided by 2 is 2", "2")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("What is 4 / 2?", template.Template);
        Assert.Contains("What is 10 / 2?", template.Template);
    }

    [Fact]
    public void Builder_AddExampleObject_IncludesExample()
    {
        var example = new ChainOfThoughtExample("Q1", "R1", "A1");
        var template = ChainOfThoughtTemplate.Builder()
            .WithQuestion("Q2")
            .AddExample(example)
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Q1", template.Template);
    }

    [Fact]
    public void Builder_AddNullExample_DoesNotThrow()
    {
        var template = ChainOfThoughtTemplate.Builder()
            .WithQuestion("Test")
            .AddExample(null!)
            .Build();

        Assert.NotNull(template);
    }

    [Fact]
    public void Constructor_WithNullExamples_HandlesGracefully()
    {
        var template = new ChainOfThoughtTemplate("Test question", (IEnumerable<ChainOfThoughtExample>?)null, null);

        Assert.NotNull(template);
    }

    [Fact]
    public void Constructor_WithEmptyExamples_CreatesTemplate()
    {
        var template = new ChainOfThoughtTemplate("Test question", Array.Empty<ChainOfThoughtExample>());

        Assert.NotNull(template);
        Assert.Contains("Test question", template.Template);
    }

    [Fact]
    public void Template_ContainsStepByStepInstructions()
    {
        // Use question constructor by specifying context parameter
        var template = new ChainOfThoughtTemplate("Solve this problem", context: null);

        Assert.Contains("step", template.Template.ToLower());
    }
}
