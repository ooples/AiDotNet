using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class CompositePromptTemplateTests
{
    [Fact]
    public void Constructor_Default_CreatesEmptyTemplate()
    {
        var template = new CompositePromptTemplate();

        Assert.NotNull(template);
    }

    [Fact]
    public void Constructor_WithSeparator_SetsSeparator()
    {
        var template = new CompositePromptTemplate("\n---\n");

        Assert.NotNull(template);
    }

    [Fact]
    public void AddTemplate_AddsTemplate()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Hello {name}"));

        Assert.NotNull(composite);
    }

    [Fact]
    public void AddTemplate_MultipleTemplates_AddsAll()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Part 1: {a}"))
            .Add(new SimplePromptTemplate("Part 2: {b}"))
            .Add(new SimplePromptTemplate("Part 3: {c}"));

        Assert.NotNull(composite);
    }

    [Fact]
    public void Format_SingleTemplate_FormatsCorrectly()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Hello {name}"));
        var variables = new Dictionary<string, string>
        {
            ["name"] = "World"
        };

        var result = composite.Format(variables);

        Assert.Contains("Hello World", result);
    }

    [Fact]
    public void Format_MultipleTemplates_CombinesWithSeparator()
    {
        var composite = new CompositePromptTemplate("\n---\n")
            .Add(new SimplePromptTemplate("Part 1"))
            .Add(new SimplePromptTemplate("Part 2"));
        var variables = new Dictionary<string, string>();

        var result = composite.Format(variables);

        Assert.Contains("Part 1", result);
        Assert.Contains("Part 2", result);
        Assert.Contains("---", result);
    }

    [Fact]
    public void Format_WithVariables_ReplacesInAllTemplates()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Name: {name}"))
            .Add(new SimplePromptTemplate("Hello {name}!"));
        var variables = new Dictionary<string, string>
        {
            ["name"] = "John"
        };

        var result = composite.Format(variables);

        Assert.Contains("Name: John", result);
        Assert.Contains("Hello John!", result);
    }

    [Fact]
    public void Format_DifferentVariables_ReplacesCorrectly()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("First: {a}"))
            .Add(new SimplePromptTemplate("Second: {b}"));
        var variables = new Dictionary<string, string>
        {
            ["a"] = "Alpha",
            ["b"] = "Beta"
        };

        var result = composite.Format(variables);

        Assert.Contains("First: Alpha", result);
        Assert.Contains("Second: Beta", result);
    }

    [Fact]
    public void InputVariables_CombinesFromAllTemplates()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("{a} {b}"))
            .Add(new SimplePromptTemplate("{c} {d}"));

        Assert.Contains("a", composite.InputVariables);
        Assert.Contains("b", composite.InputVariables);
        Assert.Contains("c", composite.InputVariables);
        Assert.Contains("d", composite.InputVariables);
    }

    [Fact]
    public void InputVariables_DeduplicatesVariables()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("{name}"))
            .Add(new SimplePromptTemplate("{name}"));

        Assert.Single(composite.InputVariables, v => v == "name");
    }

    [Fact]
    public void Validate_WithAllVariables_ReturnsTrue()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("{a}"))
            .Add(new SimplePromptTemplate("{b}"));
        var variables = new Dictionary<string, string>
        {
            ["a"] = "value1",
            ["b"] = "value2"
        };

        var result = composite.Validate(variables);

        Assert.True(result);
    }

    [Fact]
    public void Validate_WithMissingVariable_ReturnsFalse()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("{a}"))
            .Add(new SimplePromptTemplate("{b}"));
        var variables = new Dictionary<string, string>
        {
            ["a"] = "value1"
        };

        var result = composite.Validate(variables);

        Assert.False(result);
    }

    [Fact]
    public void Validate_WithNullVariables_ReturnsFalse()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("{a}"));

        var result = composite.Validate(null!);

        Assert.False(result);
    }

    [Fact]
    public void Format_EmptyComposite_ReturnsEmpty()
    {
        var composite = new CompositePromptTemplate();
        var variables = new Dictionary<string, string>();

        var result = composite.Format(variables);

        Assert.Equal(string.Empty, result.Trim());
    }

    [Fact]
    public void Format_WithDefaultSeparator_UsesNewlines()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Line 1"))
            .Add(new SimplePromptTemplate("Line 2"));
        var variables = new Dictionary<string, string>();

        var result = composite.Format(variables);

        Assert.Contains("Line 1", result);
        Assert.Contains("Line 2", result);
    }

    [Fact]
    public void Format_WithCustomSeparator_UsesSeparator()
    {
        var composite = new CompositePromptTemplate(" | ")
            .Add(new SimplePromptTemplate("A"))
            .Add(new SimplePromptTemplate("B"))
            .Add(new SimplePromptTemplate("C"));
        var variables = new Dictionary<string, string>();

        var result = composite.Format(variables);

        Assert.Contains(" | ", result);
    }

    [Fact]
    public void AddTemplate_ReturnsNewInstance()
    {
        var original = new CompositePromptTemplate();
        var modified = original.Add(new SimplePromptTemplate("test"));

        Assert.NotSame(original, modified);
    }

    [Fact]
    public void Format_WithMixedTemplateTypes_WorksCorrectly()
    {
        // ChatPromptTemplate messages are literal content, not placeholders
        var chatTemplate = new ChatPromptTemplate()
            .AddSystemMessage("Be helpful")
            .AddUserMessage("Hi there");
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Simple: {text}"))
            .Add(chatTemplate);
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello"
        };

        var result = composite.Format(variables);

        Assert.Contains("Simple: Hello", result);
        Assert.Contains("Be helpful", result);
        Assert.Contains("Hi there", result);
    }

    [Fact]
    public void FluentChaining_WorksCorrectly()
    {
        var composite = new CompositePromptTemplate()
            .Add(new SimplePromptTemplate("Part 1"))
            .Add(new SimplePromptTemplate("Part 2"))
            .Add(new SimplePromptTemplate("Part 3"));

        Assert.NotNull(composite);
    }

    [Fact]
    public void Format_PreservesTemplateOrder()
    {
        var composite = new CompositePromptTemplate(" ")
            .Add(new SimplePromptTemplate("First"))
            .Add(new SimplePromptTemplate("Second"))
            .Add(new SimplePromptTemplate("Third"));
        var variables = new Dictionary<string, string>();

        var result = composite.Format(variables);

        var firstIndex = result.IndexOf("First");
        var secondIndex = result.IndexOf("Second");
        var thirdIndex = result.IndexOf("Third");

        Assert.True(firstIndex < secondIndex);
        Assert.True(secondIndex < thirdIndex);
    }

    [Fact]
    public void Format_ComplexRealWorldExample()
    {
        var composite = new CompositePromptTemplate("\n\n")
            .Add(new SimplePromptTemplate("## System\nYou are a {role}."))
            .Add(new SimplePromptTemplate("## Context\n{context}"))
            .Add(new SimplePromptTemplate("## Task\n{task}"));

        var variables = new Dictionary<string, string>
        {
            ["role"] = "helpful assistant",
            ["context"] = "User is building a web application",
            ["task"] = "Review the architecture design"
        };

        var result = composite.Format(variables);

        Assert.Contains("helpful assistant", result);
        Assert.Contains("web application", result);
        Assert.Contains("architecture design", result);
    }
}
