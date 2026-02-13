#nullable disable
// Nullable disabled because these tests deliberately pass null to verify runtime behavior.

using AiDotNet.Validation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Validation;

/// <summary>
/// Unit tests for the <see cref="Guard"/> utility class.
/// </summary>
public class GuardTests
{
    // ───────────────── NotNull ─────────────────

    [Fact]
    public void NotNull_WithNonNullValue_DoesNotThrow()
    {
        var obj = new object();
        var exception = Record.Exception(() => Guard.NotNull(obj));
        Assert.Null(exception);
    }

    [Fact]
    public void NotNull_WithNull_ThrowsArgumentNullException()
    {
        string value = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNull(value));
    }

    [Fact]
    public void NotNull_WithExplicitName_IncludesParameterNameInException()
    {
        object value = null;
        var ex = Assert.Throws<ArgumentNullException>(() => Guard.NotNull(value, "myParam"));
        Assert.Equal("myParam", ex.ParamName);
    }

    // ───────────────── NotNullOrEmpty ─────────────────

    [Fact]
    public void NotNullOrEmpty_WithValidString_DoesNotThrow()
    {
        var exception = Record.Exception(() => Guard.NotNullOrEmpty("hello"));
        Assert.Null(exception);
    }

    [Fact]
    public void NotNullOrEmpty_WithNull_ThrowsArgumentNullException()
    {
        string value = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNullOrEmpty(value));
    }

    [Fact]
    public void NotNullOrEmpty_WithEmptyString_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrEmpty(""));
    }

    [Fact]
    public void NotNullOrEmpty_WithWhitespace_DoesNotThrow()
    {
        // Whitespace is allowed by NotNullOrEmpty (use NotNullOrWhiteSpace for stricter check)
        var exception = Record.Exception(() => Guard.NotNullOrEmpty("   "));
        Assert.Null(exception);
    }

    // ───────────────── NotNullOrWhiteSpace ─────────────────

    [Fact]
    public void NotNullOrWhiteSpace_WithValidString_DoesNotThrow()
    {
        var exception = Record.Exception(() => Guard.NotNullOrWhiteSpace("hello"));
        Assert.Null(exception);
    }

    [Fact]
    public void NotNullOrWhiteSpace_WithNull_ThrowsArgumentNullException()
    {
        string value = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNullOrWhiteSpace(value));
    }

    [Fact]
    public void NotNullOrWhiteSpace_WithEmptyString_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace(""));
    }

    [Fact]
    public void NotNullOrWhiteSpace_WithWhitespace_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace("   "));
    }

    [Fact]
    public void NotNullOrWhiteSpace_WithTabs_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace("\t\t"));
    }

    // ───────────────── Positive (int) ─────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(42)]
    [InlineData(int.MaxValue)]
    public void Positive_Int_WithPositiveValue_DoesNotThrow(int value)
    {
        var exception = Record.Exception(() => Guard.Positive(value));
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(int.MinValue)]
    public void Positive_Int_WithNonPositiveValue_ThrowsArgumentOutOfRangeException(int value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(value));
    }

    // ───────────────── Positive (double) ─────────────────

    [Theory]
    [InlineData(0.001)]
    [InlineData(1.0)]
    [InlineData(double.MaxValue)]
    public void Positive_Double_WithPositiveValue_DoesNotThrow(double value)
    {
        var exception = Record.Exception(() => Guard.Positive(value));
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.001)]
    [InlineData(double.NegativeInfinity)]
    public void Positive_Double_WithNonPositiveValue_ThrowsArgumentOutOfRangeException(double value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(value));
    }

    [Fact]
    public void Positive_Double_WithNaN_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.NaN));
    }

    [Fact]
    public void Positive_Double_WithPositiveInfinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.PositiveInfinity));
    }

    // ───────────────── NonNegative (int) ─────────────────

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(int.MaxValue)]
    public void NonNegative_Int_WithNonNegativeValue_DoesNotThrow(int value)
    {
        var exception = Record.Exception(() => Guard.NonNegative(value));
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(int.MinValue)]
    public void NonNegative_Int_WithNegativeValue_ThrowsArgumentOutOfRangeException(int value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(value));
    }

    // ───────────────── NonNegative (double) ─────────────────

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.001)]
    [InlineData(double.MaxValue)]
    public void NonNegative_Double_WithNonNegativeValue_DoesNotThrow(double value)
    {
        var exception = Record.Exception(() => Guard.NonNegative(value));
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(-0.001)]
    [InlineData(double.NegativeInfinity)]
    public void NonNegative_Double_WithNegativeValue_ThrowsArgumentOutOfRangeException(double value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(value));
    }

    [Fact]
    public void NonNegative_Double_WithNaN_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.NaN));
    }

    [Fact]
    public void NonNegative_Double_WithPositiveInfinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.PositiveInfinity));
    }

    // ───────────────── InRange (int) ─────────────────

    [Theory]
    [InlineData(5, 1, 10)]
    [InlineData(1, 1, 10)]   // min boundary
    [InlineData(10, 1, 10)]  // max boundary
    [InlineData(0, 0, 0)]    // single-value range
    public void InRange_Int_WithValueInRange_DoesNotThrow(int value, int min, int max)
    {
        var exception = Record.Exception(() => Guard.InRange(value, min, max));
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(0, 1, 10)]   // below min
    [InlineData(11, 1, 10)]  // above max
    [InlineData(-1, 0, 100)] // negative below zero-based range
    public void InRange_Int_WithValueOutOfRange_ThrowsArgumentOutOfRangeException(int value, int min, int max)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(value, min, max));
    }

    // ───────────────── InRange (double) ─────────────────

    [Theory]
    [InlineData(0.5, 0.0, 1.0)]
    [InlineData(0.0, 0.0, 1.0)]  // min boundary
    [InlineData(1.0, 0.0, 1.0)]  // max boundary
    public void InRange_Double_WithValueInRange_DoesNotThrow(double value, double min, double max)
    {
        var exception = Record.Exception(() => Guard.InRange(value, min, max));
        Assert.Null(exception);
    }

    [Theory]
    [InlineData(-0.001, 0.0, 1.0)]  // below min
    [InlineData(1.001, 0.0, 1.0)]   // above max
    public void InRange_Double_WithValueOutOfRange_ThrowsArgumentOutOfRangeException(double value, double min, double max)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(value, min, max));
    }

    [Fact]
    public void InRange_Double_WithNaN_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(double.NaN, 0.0, 1.0));
    }

    [Fact]
    public void InRange_Double_WithPositiveInfinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(double.PositiveInfinity, 0.0, 1.0));
    }

    [Fact]
    public void InRange_Double_WithNegativeInfinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(double.NegativeInfinity, 0.0, 1.0));
    }

    // ───────────────── InRange bound validation ─────────────────

    [Fact]
    public void InRange_Int_WithMinGreaterThanMax_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() => Guard.InRange(5, 10, 1));
        Assert.Equal("min", ex.ParamName);
    }

    [Fact]
    public void InRange_Double_WithMinGreaterThanMax_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() => Guard.InRange(0.5, 1.0, 0.0));
        Assert.Equal("min", ex.ParamName);
    }

    [Fact]
    public void InRange_Double_WithNaNMin_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0.5, double.NaN, 1.0));
    }

    [Fact]
    public void InRange_Double_WithInfinityMax_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0.5, 0.0, double.PositiveInfinity));
    }

    // ───────────────── Parameter name propagation ─────────────────

    [Fact]
    public void Positive_Int_IncludesParameterNameInException()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(-1, "batchSize"));
        Assert.Equal("batchSize", ex.ParamName);
    }

    [Fact]
    public void InRange_Int_IncludesParameterNameInException()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(99, 0, 10, "epoch"));
        Assert.Equal("epoch", ex.ParamName);
    }
}
