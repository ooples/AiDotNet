using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class ConditionalPromptTemplateTests
{
    [Fact]
    public void Constructor_WithTemplate_CreatesTemplate()
    {
        var template = new ConditionalPromptTemplate("Hello {name}");

        Assert.NotNull(template);
        Assert.Contains("name", template.InputVariables);
    }

    [Fact]
    public void Format_WithIfCondition_WhenTrue_IncludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "Hello {{#if name}}{name}{{/if}}");
        var variables = new Dictionary<string, string>
        {
            ["name"] = "John"
        };

        var result = template.Format(variables);

        Assert.Contains("John", result);
    }

    [Fact]
    public void Format_WithIfCondition_WhenFalse_ExcludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "Hello {{#if name}}{name}{{/if}}World");
        var variables = new Dictionary<string, string>();

        var result = template.Format(variables);

        Assert.DoesNotContain("{name}", result);
        Assert.Contains("Hello", result);
        Assert.Contains("World", result);
    }

    [Fact]
    public void Format_WithIfCondition_WhenEmpty_ExcludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "Hello {{#if name}}{name}{{/if}}World");
        var variables = new Dictionary<string, string>
        {
            ["name"] = ""
        };

        var result = template.Format(variables);

        Assert.DoesNotContain("{name}", result);
    }

    [Fact]
    public void Format_WithIfCondition_WhenWhitespace_ExcludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "Hello {{#if name}}{name}{{/if}}World");
        var variables = new Dictionary<string, string>
        {
            ["name"] = "   "
        };

        var result = template.Format(variables);

        Assert.DoesNotContain("{name}", result);
    }

    [Fact]
    public void Format_WithUnlessCondition_WhenFalse_IncludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "{{#unless premium}}Free tier{{/unless}}");
        var variables = new Dictionary<string, string>();

        var result = template.Format(variables);

        Assert.Contains("Free tier", result);
    }

    [Fact]
    public void Format_WithUnlessCondition_WhenTrue_ExcludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "{{#unless premium}}Free tier{{/unless}}");
        var variables = new Dictionary<string, string>
        {
            ["premium"] = "yes"
        };

        var result = template.Format(variables);

        Assert.DoesNotContain("Free tier", result);
    }

    [Fact]
    public void Format_WithEqualsCondition_WhenMatches_IncludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "{{#equals status \"active\"}}User is active{{/equals}}");
        var variables = new Dictionary<string, string>
        {
            ["status"] = "active"
        };

        var result = template.Format(variables);

        Assert.Contains("User is active", result);
    }

    [Fact]
    public void Format_WithEqualsCondition_WhenNotMatches_ExcludesContent()
    {
        var template = new ConditionalPromptTemplate(
            "{{#equals status \"active\"}}User is active{{/equals}}");
        var variables = new Dictionary<string, string>
        {
            ["status"] = "inactive"
        };

        var result = template.Format(variables);

        Assert.DoesNotContain("User is active", result);
    }

    [Fact]
    public void Format_WithEqualsCondition_CaseInsensitive()
    {
        var template = new ConditionalPromptTemplate(
            "{{#equals status \"ACTIVE\"}}User is active{{/equals}}");
        var variables = new Dictionary<string, string>
        {
            ["status"] = "active"
        };

        var result = template.Format(variables);

        Assert.Contains("User is active", result);
    }

    [Fact]
    public void Format_WithMultipleConditions_ProcessesAll()
    {
        var template = new ConditionalPromptTemplate(
            "{{#if name}}Name: {name}{{/if}} {{#if email}}Email: {email}{{/if}}");
        var variables = new Dictionary<string, string>
        {
            ["name"] = "John",
            ["email"] = "john@example.com"
        };

        var result = template.Format(variables);

        Assert.Contains("Name: John", result);
        Assert.Contains("Email: john@example.com", result);
    }

    [Fact]
    public void Format_WithNestedVariables_ReplacesCorrectly()
    {
        var template = new ConditionalPromptTemplate(
            "Text: {text} {{#if context}}Context: {context}{{/if}}");
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello",
            ["context"] = "Greeting"
        };

        var result = template.Format(variables);

        Assert.Contains("Text: Hello", result);
        Assert.Contains("Context: Greeting", result);
    }

    [Fact]
    public void Validate_WithRequiredVariable_ReturnsTrue()
    {
        var template = new ConditionalPromptTemplate("Hello {name}");
        var variables = new Dictionary<string, string>
        {
            ["name"] = "John"
        };

        var result = template.Validate(variables);

        Assert.True(result);
    }

    [Fact]
    public void Validate_WithMissingRequiredVariable_ReturnsFalse()
    {
        var template = new ConditionalPromptTemplate("Hello {name}");
        var variables = new Dictionary<string, string>();

        var result = template.Validate(variables);

        Assert.False(result);
    }

    [Fact]
    public void Validate_WithConditionalVariable_NotRequired()
    {
        var template = new ConditionalPromptTemplate(
            "Hello {{#if name}}{name}{{/if}}");
        var variables = new Dictionary<string, string>();

        var result = template.Validate(variables);

        Assert.True(result);
    }

    [Fact]
    public void Validate_WithNullVariables_ReturnsFalse()
    {
        var template = new ConditionalPromptTemplate("Hello {name}");

        var result = template.Validate(null!);

        Assert.False(result);
    }

    [Fact]
    public void InputVariables_IncludesConditionalVariables()
    {
        var template = new ConditionalPromptTemplate(
            "{{#if name}}{name}{{/if}} {{#unless premium}}free{{/unless}}");

        Assert.Contains("name", template.InputVariables);
        Assert.Contains("premium", template.InputVariables);
    }

    [Fact]
    public void InputVariables_IncludesEqualsVariables()
    {
        var template = new ConditionalPromptTemplate(
            "{{#equals status \"active\"}}active{{/equals}}");

        Assert.Contains("status", template.InputVariables);
    }

    [Fact]
    public void Format_CleansUpExtraWhitespace()
    {
        var template = new ConditionalPromptTemplate(
            "Hello\n\n{{#if missing}}content{{/if}}\n\nWorld");
        var variables = new Dictionary<string, string>();

        var result = template.Format(variables);

        Assert.DoesNotContain("\n\n\n", result);
    }

    [Fact]
    public void Format_MultilineConditional_WorksCorrectly()
    {
        var template = new ConditionalPromptTemplate(
            "Start {{#if details}}\nLine 1\nLine 2\n{{/if}} End");
        var variables = new Dictionary<string, string>
        {
            ["details"] = "yes"
        };

        var result = template.Format(variables);

        Assert.Contains("Line 1", result);
        Assert.Contains("Line 2", result);
    }

    [Fact]
    public void Format_WithMixedConditions_ProcessesInOrder()
    {
        var template = new ConditionalPromptTemplate(
            "{{#equals type \"admin\"}}Admin{{/equals}}{{#if name}}{name}{{/if}}{{#unless guest}}Member{{/unless}}");
        var variables = new Dictionary<string, string>
        {
            ["type"] = "admin",
            ["name"] = "John"
        };

        var result = template.Format(variables);

        Assert.Contains("Admin", result);
        Assert.Contains("John", result);
        Assert.Contains("Member", result);
    }

    [Fact]
    public void Format_ComplexRealWorldExample()
    {
        var template = new ConditionalPromptTemplate(@"
            Analyze this text: {text}
            {{#if context}}Additional context: {context}{{/if}}
            {{#if style}}Writing style: {style}{{/if}}
            {{#equals format ""json""}}Output as JSON{{/equals}}
        ");

        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello world",
            ["context"] = "Programming tutorial",
            ["format"] = "json"
        };

        var result = template.Format(variables);

        Assert.Contains("Hello world", result);
        Assert.Contains("Programming tutorial", result);
        Assert.Contains("Output as JSON", result);
        Assert.DoesNotContain("Writing style", result);
    }
}
