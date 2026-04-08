using AiDotNet.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Tools;

/// <summary>
/// Unit tests for the SearchTool class.
/// </summary>
public class SearchToolTests
{
    [Fact]
    public void Name_ReturnsSearch()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var name = searchTool.Name;

        // Assert
        Assert.Equal("Search", name);
    }

    [Fact]
    public void Description_ReturnsNonEmptyString()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var description = searchTool.Description;

        // Assert
        Assert.False(string.IsNullOrWhiteSpace(description));
        Assert.Contains("search", description, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("capital of France", "Paris")]
    [InlineData("capital of Japan", "Tokyo")]
    [InlineData("capital of USA", "Washington")]
    public void Execute_KnownQuery_ReturnsExpectedResult(string query, string expectedSubstring)
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute(query);

        // Assert
        Assert.Contains(expectedSubstring, result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Execute_EmptyInput_ReturnsError()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute("");

        // Assert
        Assert.Contains("Error", result);
        Assert.Contains("empty", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Execute_WhitespaceInput_ReturnsError()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute("   ");

        // Assert
        Assert.Contains("Error", result);
    }

    [Fact]
    public void Execute_UnknownQuery_ReturnsGenericResponse()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute("this is a completely unknown query that doesn't match anything");

        // Assert
        Assert.Contains("Mock search results", result);
        Assert.Contains("no specific information available", result);
    }

    [Fact]
    public void Execute_CaseInsensitiveMatch_ReturnsResult()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result1 = searchTool.Execute("CAPITAL OF FRANCE");
        var result2 = searchTool.Execute("capital of france");
        var result3 = searchTool.Execute("Capital Of France");

        // Assert
        Assert.Contains("Paris", result1, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Paris", result2, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Paris", result3, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Execute_PartialMatch_ReturnsRelevantResult()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute("What is the capital of France?");

        // Assert
        Assert.Contains("Paris", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Constructor_WithCustomMockResults_UsesProvidedData()
    {
        // Arrange
        var customResults = new Dictionary<string, string>
        {
            ["test query"] = "test result",
            ["another query"] = "another result"
        };
        var searchTool = new SearchTool(customResults);

        // Act
        var result1 = searchTool.Execute("test query");
        var result2 = searchTool.Execute("another query");

        // Assert
        Assert.Equal("test result", result1);
        Assert.Equal("another result", result2);
    }

    [Fact]
    public void AddMockResult_NewQuery_AddsSuccessfully()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        searchTool.AddMockResult("custom query", "custom result");
        var result = searchTool.Execute("custom query");

        // Assert
        Assert.Equal("custom result", result);
    }

    [Fact]
    public void AddMockResult_ExistingQuery_UpdatesResult()
    {
        // Arrange
        var searchTool = new SearchTool();
        searchTool.AddMockResult("capital of France", "Updated answer: Paris is the capital");

        // Act
        var result = searchTool.Execute("capital of France");

        // Assert
        Assert.Equal("Updated answer: Paris is the capital", result);
    }

    [Fact]
    public void RemoveMockResult_ExistingQuery_RemovesAndReturnsTrue()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var removed = searchTool.RemoveMockResult("capital of France");
        var result = searchTool.Execute("capital of France");

        // Assert
        Assert.True(removed);
        Assert.Contains("Mock search results", result);
        Assert.Contains("no specific information available", result);
    }

    [Fact]
    public void RemoveMockResult_NonExistingQuery_ReturnsFalse()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var removed = searchTool.RemoveMockResult("this query does not exist");

        // Assert
        Assert.False(removed);
    }

    [Fact]
    public void ClearMockResults_RemovesAllResults()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        searchTool.ClearMockResults();
        var result1 = searchTool.Execute("capital of France");
        var result2 = searchTool.Execute("capital of Japan");

        // Assert
        Assert.Contains("Mock search results", result1);
        Assert.Contains("no specific information available", result1);
        Assert.Contains("Mock search results", result2);
        Assert.Contains("no specific information available", result2);
    }

    [Fact]
    public void Execute_QueryWithExtraWhitespace_HandlesCorrectly()
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute("  capital of France  ");

        // Assert
        Assert.Contains("Paris", result, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("speed of light")]
    [InlineData("largest ocean")]
    [InlineData("highest mountain")]
    public void Execute_VariousKnownQueries_ReturnsNonEmptyResults(string query)
    {
        // Arrange
        var searchTool = new SearchTool();

        // Act
        var result = searchTool.Execute(query);

        // Assert
        Assert.False(string.IsNullOrWhiteSpace(result));
        Assert.DoesNotContain("Error", result);
        Assert.DoesNotContain("no specific information available", result);
    }
}
