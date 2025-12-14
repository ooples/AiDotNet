using AiDotNet.PromptEngineering;
using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

/// <summary>
/// Edge case tests for prompt engineering components.
/// Tests null handling, empty strings, special characters, and boundary conditions.
/// </summary>
public class EdgeCaseTests
{
    #region Null and Empty String Handling

    [Fact]
    public void SimplePromptTemplate_NullTemplate_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new SimplePromptTemplate(null!));
    }

    [Fact]
    public void SimplePromptTemplate_EmptyTemplate_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new SimplePromptTemplate(""));
    }

    [Fact]
    public void SimplePromptTemplate_WhitespaceTemplate_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new SimplePromptTemplate("   "));
    }

    [Fact]
    public void SimplePromptTemplate_Format_NullVariables_ThrowsArgumentNullException()
    {
        var template = new SimplePromptTemplate("Hello {name}");
        Assert.Throws<ArgumentNullException>(() => template.Format(null!));
    }

    [Fact]
    public void ChainOfThoughtTemplate_NullTask_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new ChainOfThoughtTemplate(null!));
    }

    [Fact]
    public void ChainOfThoughtTemplate_EmptyTask_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new ChainOfThoughtTemplate(""));
    }

    [Fact]
    public void StructuredOutputTemplate_NullSchema_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new StructuredOutputTemplate(null!, "json"));
    }

    [Fact]
    public void StructuredOutputTemplate_NullFormat_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new StructuredOutputTemplate("schema", null!));
    }

    [Fact]
    public void RolePlayingTemplate_NullRole_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new RolePlayingTemplate(null!, "task"));
    }

    [Fact]
    public void RolePlayingTemplate_NullTask_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new RolePlayingTemplate("role", null!));
    }

    [Fact]
    public void InstructionFollowingTemplate_NullInstructions_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new InstructionFollowingTemplate(null!));
    }

    [Fact]
    public void ConditionalPromptTemplate_NullTemplate_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new ConditionalPromptTemplate(null!));
    }

    [Fact]
    public void CompositePromptTemplate_NullTemplates_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new CompositePromptTemplate(null!));
    }

    [Fact]
    public void CompositePromptTemplate_EmptyTemplates_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new CompositePromptTemplate(new List<string>()));
    }

    #endregion

    #region Special Character Handling

    [Fact]
    public void SimplePromptTemplate_SpecialCharactersInTemplate_HandledCorrectly()
    {
        var template = new SimplePromptTemplate("Test with special chars: @#$%^&*()");
        var result = template.Format(new Dictionary<string, string>());

        Assert.Contains("@#$%^&*()", result);
    }

    [Fact]
    public void SimplePromptTemplate_UnicodeInTemplate_HandledCorrectly()
    {
        var template = new SimplePromptTemplate("Unicode test: \u00e9\u00e8\u00ea \u4e2d\u6587");
        var result = template.Format(new Dictionary<string, string>());

        Assert.Contains("\u00e9\u00e8\u00ea", result);
        Assert.Contains("\u4e2d\u6587", result);
    }

    [Fact]
    public void SimplePromptTemplate_NewlinesInTemplate_PreservedCorrectly()
    {
        var template = new SimplePromptTemplate("Line 1\nLine 2\r\nLine 3");
        var result = template.Format(new Dictionary<string, string>());

        Assert.Contains("Line 1", result);
        Assert.Contains("Line 2", result);
        Assert.Contains("Line 3", result);
    }

    [Fact]
    public void SimplePromptTemplate_TabsInTemplate_PreservedCorrectly()
    {
        var template = new SimplePromptTemplate("Column1\tColumn2\tColumn3");
        var result = template.Format(new Dictionary<string, string>());

        Assert.Contains("\t", result);
    }

    [Fact]
    public void SimplePromptTemplate_BracesInValue_NotInterpretedAsVariable()
    {
        var template = new SimplePromptTemplate("Code: {code}");
        var variables = new Dictionary<string, string>
        {
            { "code", "function() { return true; }" }
        };
        var result = template.Format(variables);

        Assert.Contains("function() { return true; }", result);
    }

    [Fact]
    public void ConditionalPromptTemplate_NestedBraces_HandledCorrectly()
    {
        var template = new ConditionalPromptTemplate("Data: {{#if show}}{value}{{/if}}");
        var variables = new Dictionary<string, string>
        {
            { "show", "true" },
            { "value", "test" }
        };
        var result = template.Format(variables);

        Assert.Contains("test", result);
    }

    #endregion

    #region Boundary Conditions

    [Fact]
    public void ContextWindowManager_ZeroMaxTokens_AllowsCreation()
    {
        var manager = new ContextWindowManager(0);
        Assert.Equal(0, manager.MaxTokens);
    }

    [Fact]
    public void ContextWindowManager_VeryLargeMaxTokens_AllowsCreation()
    {
        var manager = new ContextWindowManager(int.MaxValue);
        Assert.Equal(int.MaxValue, manager.MaxTokens);
    }

    [Fact]
    public void ContextWindowManager_NegativeReserved_TreatedAsZero()
    {
        var manager = new ContextWindowManager(1000);
        var result = manager.FitsInWindow("test", -100);

        // Should not throw, negative reserved should be handled
        Assert.True(result);
    }

    [Fact]
    public void ContextWindowManager_VeryLongText_Truncates()
    {
        var manager = new ContextWindowManager(100, text => text.Length);
        var longText = new string('x', 10000);

        var truncated = manager.TruncateToFit(longText);

        Assert.True(truncated.Length <= 100);
    }

    [Fact]
    public void ContextWindowManager_SplitIntoChunks_SingleCharPerChunk()
    {
        var manager = new ContextWindowManager(1, text => text.Length);
        var text = "abc";

        var chunks = manager.SplitIntoChunks(text);

        Assert.Equal(3, chunks.Count);
    }

    [Fact]
    public void SimplePromptTemplate_VeryLongVariableValue_HandledCorrectly()
    {
        var template = new SimplePromptTemplate("Data: {data}");
        var longValue = new string('x', 100000);
        var variables = new Dictionary<string, string> { { "data", longValue } };

        var result = template.Format(variables);

        Assert.Contains(longValue, result);
    }

    [Fact]
    public void SimplePromptTemplate_ManyVariables_AllSubstituted()
    {
        var templateParts = new List<string>();
        var variables = new Dictionary<string, string>();

        for (int i = 0; i < 100; i++)
        {
            templateParts.Add($"{{var{i}}}");
            variables[$"var{i}"] = $"value{i}";
        }

        var template = new SimplePromptTemplate(string.Join(" ", templateParts));
        var result = template.Format(variables);

        for (int i = 0; i < 100; i++)
        {
            Assert.Contains($"value{i}", result);
        }
    }

    #endregion

    #region Missing Variable Handling

    [Fact]
    public void SimplePromptTemplate_MissingVariable_ThrowsArgumentException()
    {
        var template = new SimplePromptTemplate("Hello {name}, welcome to {place}");
        var variables = new Dictionary<string, string>
        {
            { "name", "John" }
            // "place" is missing
        };

        Assert.Throws<ArgumentException>(() => template.Format(variables));
    }

    [Fact]
    public void SimplePromptTemplate_ExtraVariables_Ignored()
    {
        var template = new SimplePromptTemplate("Hello {name}");
        var variables = new Dictionary<string, string>
        {
            { "name", "John" },
            { "extra", "ignored" }
        };

        var result = template.Format(variables);

        Assert.Contains("Hello John", result);
        Assert.DoesNotContain("ignored", result);
    }

    [Fact]
    public void ConditionalPromptTemplate_MissingConditionalVariable_SectionRemoved()
    {
        var template = new ConditionalPromptTemplate("Base {{#if optional}}Optional: {optional}{{/if}}");
        var variables = new Dictionary<string, string>();

        var result = template.Format(variables);

        Assert.Contains("Base", result);
        Assert.DoesNotContain("Optional:", result);
    }

    [Fact]
    public void ConditionalPromptTemplate_EmptyConditionalVariable_SectionRemoved()
    {
        var template = new ConditionalPromptTemplate("Base {{#if optional}}Optional: {optional}{{/if}}");
        var variables = new Dictionary<string, string>
        {
            { "optional", "" }
        };

        var result = template.Format(variables);

        Assert.Contains("Base", result);
        Assert.DoesNotContain("Optional:", result);
    }

    [Fact]
    public void ConditionalPromptTemplate_WhitespaceConditionalVariable_SectionRemoved()
    {
        var template = new ConditionalPromptTemplate("Base {{#if optional}}Optional: {optional}{{/if}}");
        var variables = new Dictionary<string, string>
        {
            { "optional", "   " }
        };

        var result = template.Format(variables);

        Assert.Contains("Base", result);
        Assert.DoesNotContain("Optional:", result);
    }

    #endregion

    #region Validation Edge Cases

    [Fact]
    public void SimplePromptTemplate_Validate_NullVariables_ReturnsFalse()
    {
        var template = new SimplePromptTemplate("Hello {name}");

        var result = template.Validate(null!);

        Assert.False(result);
    }

    [Fact]
    public void SimplePromptTemplate_Validate_EmptyVariables_WithNoPlaceholders_ReturnsTrue()
    {
        var template = new SimplePromptTemplate("Hello World");

        var result = template.Validate(new Dictionary<string, string>());

        Assert.True(result);
    }

    [Fact]
    public void SimplePromptTemplate_Validate_NullVariableValue_ReturnsFalse()
    {
        var template = new SimplePromptTemplate("Hello {name}");
        var variables = new Dictionary<string, string>
        {
            { "name", null! }
        };

        var result = template.Validate(variables);

        Assert.False(result);
    }

    #endregion

    #region Regex Timeout Edge Cases

    [Fact]
    public void ConditionalPromptTemplate_NestedConditions_DoesNotHang()
    {
        // Test that complex nested patterns don't cause regex catastrophic backtracking
        var nested = "{{#if a}}{{#if b}}{{#if c}}{value}{{/if}}{{/if}}{{/if}}";
        var template = new ConditionalPromptTemplate(nested);
        var variables = new Dictionary<string, string>
        {
            { "a", "true" },
            { "b", "true" },
            { "c", "true" },
            { "value", "test" }
        };

        // Should complete quickly due to regex timeout
        var result = template.Format(variables);

        Assert.Contains("test", result);
    }

    [Fact]
    public void ConditionalPromptTemplate_ManyConditions_CompletesInReasonableTime()
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < 50; i++)
        {
            sb.Append($"{{{{#if var{i}}}}}Value{i}{{{{/if}}}}");
        }

        var template = new ConditionalPromptTemplate(sb.ToString());
        var variables = new Dictionary<string, string>();
        for (int i = 0; i < 50; i++)
        {
            variables[$"var{i}"] = "true";
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = template.Format(variables);
        sw.Stop();

        // Should complete in under 5 seconds (generous limit)
        Assert.True(sw.ElapsedMilliseconds < 5000, $"Took {sw.ElapsedMilliseconds}ms");

        for (int i = 0; i < 50; i++)
        {
            Assert.Contains($"Value{i}", result);
        }
    }

    #endregion
}
