using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.AdversarialRobustness;

/// <summary>
/// Comprehensive tests for SafetyFilter implementation.
/// </summary>
public class SafetyFilterTests
{
    private const double Tolerance = 1e-6;

    #region Constructor Tests

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithValidOptions_CreatesInstance()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };

        // Act
        var filter = new SafetyFilter<double>(options);

        // Assert
        Assert.NotNull(filter);
        Assert.NotNull(filter.GetOptions());
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SafetyFilter<double>(null!));
    }

    #endregion

    #region ValidateInput Tests

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => filter.ValidateInput(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithNaN_MarksInvalid()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, double.NaN, 1.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "InvalidValue");
        Assert.True(result.SafetyScore < 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithInfinity_MarksInvalid()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, double.PositiveInfinity, 1.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "InvalidValue");
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithNegativeInfinity_MarksInvalid()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, double.NegativeInfinity, 1.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.False(result.IsValid);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithExcessLength_MarksInvalid()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 2
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "LengthExceeded");
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithValidInput_ReturnsValidResult()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, 0.5, 1.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.True(result.IsValid);
        Assert.Equal(1.0, result.SafetyScore, 3);
        Assert.Empty(result.Issues);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithDisabledValidation_AlwaysReturnsValid()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = false,
            MaxInputLength = 2
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { double.NaN, double.PositiveInfinity, 0.0, 1.0, 2.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert - validation is disabled, so it should pass
        Assert.True(result.IsValid);
    }

    #endregion

    #region FilterOutput Tests

    [Fact(Timeout = 60000)]
    public async Task FilterOutput_WithNullOutput_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableOutputFiltering = true
        };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => filter.FilterOutput(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task FilterOutput_WithSafeOutput_ReturnsUnmodified()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableOutputFiltering = true,
            SafetyThreshold = 0.5
        };
        var filter = new SafetyFilter<double>(options);
        var output = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

        // Act
        var result = filter.FilterOutput(output);

        // Assert
        Assert.True(result.IsSafe);
        Assert.False(result.WasModified);
        Assert.Equal(output.Length, result.FilteredOutput.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task FilterOutput_WithDisabledFiltering_ReturnsUnmodified()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableOutputFiltering = false
        };
        var filter = new SafetyFilter<double>(options);
        var output = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

        // Act
        var result = filter.FilterOutput(output);

        // Assert
        Assert.True(result.IsSafe);
        Assert.False(result.WasModified);
    }

    #endregion

    #region DetectJailbreak Tests

    [Fact(Timeout = 60000)]
    public async Task DetectJailbreak_WithCleanInput_DetectsNoJailbreak()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 100
        };
        var filter = new SafetyFilter<double>(options);
        // Normal input values
        var input = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });

        // Act
        var result = filter.DetectJailbreak(input);

        // Assert
        Assert.False(result.JailbreakDetected);
        Assert.Equal(0.0, result.ConfidenceScore);
    }

    [Fact(Timeout = 60000)]
    public async Task DetectJailbreak_WithEmptyInput_DetectsNoJailbreak()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(0);

        // Act
        var result = filter.DetectJailbreak(input);

        // Assert
        Assert.False(result.JailbreakDetected);
    }

    #endregion

    #region IdentifyHarmfulContent Tests

    [Fact(Timeout = 60000)]
    public async Task IdentifyHarmfulContent_WithCleanInput_DetectsNoHarm()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            HarmfulContentCategories = new[] { "violence", "hatespeech" }
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

        // Act
        var result = filter.IdentifyHarmfulContent(input);

        // Assert
        Assert.False(result.HarmfulContentDetected);
        Assert.Equal(0.0, result.HarmScore);
    }

    [Fact(Timeout = 60000)]
    public async Task IdentifyHarmfulContent_WithEmptyInput_DetectsNoHarm()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            HarmfulContentCategories = new[] { "violence" }
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(0);

        // Act
        var result = filter.IdentifyHarmfulContent(input);

        // Assert
        Assert.False(result.HarmfulContentDetected);
    }

    [Fact(Timeout = 60000)]
    public async Task IdentifyHarmfulContent_WithNoCategories_DetectsNoHarm()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            HarmfulContentCategories = Array.Empty<string>()
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

        // Act
        var result = filter.IdentifyHarmfulContent(input);

        // Assert
        Assert.False(result.HarmfulContentDetected);
    }

    #endregion

    #region ComputeSafetyScore Tests

    [Fact(Timeout = 60000)]
    public async Task ComputeSafetyScore_ReturnsBoundedScore()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            EnableOutputFiltering = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

        // Act
        var score = filter.ComputeSafetyScore(input);

        // Assert
        Assert.True(score >= 0.0);
        Assert.True(score <= 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task ComputeSafetyScore_WithCleanInput_ReturnsHighScore()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            EnableOutputFiltering = true,
            MaxInputLength = 100
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

        // Act
        var score = filter.ComputeSafetyScore(input);

        // Assert
        Assert.True(score >= 0.8, $"Expected high safety score for clean input, got {score}");
    }

    [Fact(Timeout = 60000)]
    public async Task ComputeSafetyScore_WithInvalidInput_ReturnsLowerScore()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 2
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 }); // Exceeds max length

        // Act
        var score = filter.ComputeSafetyScore(input);

        // Assert
        Assert.True(score < 1.0, "Safety score should be reduced for invalid input");
    }

    #endregion

    #region GetOptions and Reset Tests

    [Fact(Timeout = 60000)]
    public async Task GetOptions_ReturnsConfiguredOptions()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            EnableOutputFiltering = false,
            MaxInputLength = 50,
            SafetyThreshold = 0.7
        };
        var filter = new SafetyFilter<double>(options);

        // Act
        var returnedOptions = filter.GetOptions();

        // Assert
        Assert.True(returnedOptions.EnableInputValidation);
        Assert.False(returnedOptions.EnableOutputFiltering);
        Assert.Equal(50, returnedOptions.MaxInputLength);
        Assert.Equal(0.7, returnedOptions.SafetyThreshold, 3);
    }

    [Fact(Timeout = 60000)]
    public async Task Reset_DoesNotThrow()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true
        };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert - Reset should not throw
        var exception = Record.Exception(() => filter.Reset());
        Assert.Null(exception);
    }

    #endregion

    #region Serialization Tests

    [Fact(Timeout = 60000)]
    public async Task Serialize_ReturnsNonEmptyByteArray()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 100
        };
        var filter = new SafetyFilter<double>(options);

        // Act
        var bytes = filter.Serialize();

        // Assert
        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 0, "Serialized data should not be empty");
    }

    [Fact(Timeout = 60000)]
    public async Task Deserialize_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => filter.Deserialize(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task SerializeDeserialize_PreservesOptions()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            EnableOutputFiltering = false,
            MaxInputLength = 75,
            SafetyThreshold = 0.65
        };
        var filter = new SafetyFilter<double>(options);

        // Act
        var bytes = filter.Serialize();
        filter.Deserialize(bytes);
        var restoredOptions = filter.GetOptions();

        // Assert
        Assert.True(restoredOptions.EnableInputValidation);
        Assert.False(restoredOptions.EnableOutputFiltering);
        Assert.Equal(75, restoredOptions.MaxInputLength);
        Assert.Equal(0.65, restoredOptions.SafetyThreshold, 3);
    }

    #endregion

    #region SaveModel/LoadModel Tests

    [Fact(Timeout = 60000)]
    public async Task SaveModel_WithNullFilePath_ThrowsArgumentException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => filter.SaveModel(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task SaveModel_WithEmptyFilePath_ThrowsArgumentException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => filter.SaveModel(""));
    }

    [Fact(Timeout = 60000)]
    public async Task SaveModel_WithWhitespaceFilePath_ThrowsArgumentException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => filter.SaveModel("   "));
    }

    [Fact(Timeout = 60000)]
    public async Task LoadModel_WithNullFilePath_ThrowsArgumentException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => filter.LoadModel(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task LoadModel_WithEmptyFilePath_ThrowsArgumentException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => filter.LoadModel(""));
    }

    [Fact(Timeout = 60000)]
    public async Task LoadModel_WithNonexistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true };
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => filter.LoadModel("nonexistent_file_12345.model"));
    }

    [Fact(Timeout = 60000)]
    public async Task SaveAndLoadModel_RoundTrip_PreservesOptions()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        try
        {
            var options = new SafetyFilterOptions<double>
            {
                EnableInputValidation = true,
                EnableOutputFiltering = true,
                MaxInputLength = 42,
                SafetyThreshold = 0.55
            };
            var filter = new SafetyFilter<double>(options);

            // Act
            filter.SaveModel(tempFile);

            var filter2 = new SafetyFilter<double>(new SafetyFilterOptions<double>());
            filter2.LoadModel(tempFile);
            var loadedOptions = filter2.GetOptions();

            // Assert
            Assert.True(loadedOptions.EnableInputValidation);
            Assert.True(loadedOptions.EnableOutputFiltering);
            Assert.Equal(42, loadedOptions.MaxInputLength);
            Assert.Equal(0.55, loadedOptions.SafetyThreshold, 3);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    #endregion

    #region Edge Case Tests

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_WithMultipleIssues_ReportsAllIssues()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 2
        };
        var filter = new SafetyFilter<double>(options);
        // Input exceeds length AND contains NaN
        var input = new Vector<double>(new[] { 0.0, double.NaN, 2.0 });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.False(result.IsValid);
        Assert.True(result.Issues.Count >= 2, "Should report multiple issues");
    }

    [Fact(Timeout = 60000)]
    public async Task ComputeSafetyScore_WithEmptyInput_ReturnsValidScore()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(0);

        // Act
        var score = filter.ComputeSafetyScore(input);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0);
    }

    #endregion

    #region Float Type Tests

    [Fact(Timeout = 60000)]
    public async Task ValidateInput_FloatType_WorksCorrectly()
    {
        // Arrange
        var options = new SafetyFilterOptions<float>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<float>(options);
        var input = new Vector<float>(new[] { 0.0f, 0.5f, 1.0f });

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact(Timeout = 60000)]
    public async Task ComputeSafetyScore_FloatType_ReturnsBoundedScore()
    {
        // Arrange
        var options = new SafetyFilterOptions<float>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<float>(options);
        var input = new Vector<float>(new[] { 0.0f, 0.5f, 1.0f });

        // Act
        var score = filter.ComputeSafetyScore(input);

        // Assert
        Assert.True(score >= 0.0f);
        Assert.True(score <= 1.0f);
    }

    #endregion
}
