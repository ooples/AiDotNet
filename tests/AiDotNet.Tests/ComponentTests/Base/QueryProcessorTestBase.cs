using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.ComponentTests.Base;

/// <summary>
/// Base test class for IQueryProcessor implementations.
/// Tests query processing invariants: valid input returns non-null,
/// empty input is handled gracefully, and the return type is a string.
/// </summary>
public abstract class QueryProcessorTestBase
{
    /// <summary>
    /// Creates the query processor under test.
    /// </summary>
    protected abstract IQueryProcessor CreateProcessor();

    // =====================================================
    // INVARIANT: Valid input should return non-null
    // Processing a valid query must produce a non-null result.
    // =====================================================

    [Fact]
    public void ProcessQuery_WithValidInput_ReturnsNonNull()
    {
        var processor = CreateProcessor();

        var result = processor.ProcessQuery("How do I reset my password?");

        Assert.NotNull(result);
    }

    // =====================================================
    // INVARIANT: Empty input should not crash
    // An empty query must not throw; the processor should
    // handle it gracefully (returning empty or transformed).
    // =====================================================

    [Fact]
    public void ProcessQuery_WithEmptyInput_HandlesGracefully()
    {
        var processor = CreateProcessor();

        var exception = Record.Exception(() =>
        {
            var result = processor.ProcessQuery(string.Empty);
        });

        Assert.Null(exception);
    }

    // =====================================================
    // INVARIANT: Return value must be a string
    // The ProcessQuery method returns string per the interface.
    // This test ensures it is not null and is indeed a string.
    // =====================================================

    [Fact]
    public void ProcessQuery_ReturnType_IsString()
    {
        var processor = CreateProcessor();

        var result = processor.ProcessQuery("test query about machine learning");

        Assert.IsType<string>(result);
    }
}
