using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.AdversarialRobustness;

/// <summary>
/// Tests for the ML-based content classification system.
/// </summary>
public class ContentClassifierTests
{
    #region Constructor Tests

    [Fact(Timeout = 60000)]
    public async Task RuleBasedContentClassifier_DefaultConstructor_InitializesCorrectly()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.True(classifier.IsReady());
        Assert.NotEmpty(classifier.GetSupportedCategories());
    }

    [Fact(Timeout = 60000)]
    public async Task RuleBasedContentClassifier_WithThreshold_SetsThreshold()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.7);

        Assert.True(classifier.IsReady());
    }

    [Fact(Timeout = 60000)]
    public async Task RuleBasedContentClassifier_WithCustomPatterns_UsesPatterns()
    {
        var patterns = new Dictionary<string, List<string>>
        {
            ["Custom"] = new List<string> { @"\bcustom\b" }
        };

        var classifier = new RuleBasedContentClassifier<double>(patterns, threshold: 0.5);

        var categories = classifier.GetSupportedCategories();
        Assert.Contains("Custom", categories);
    }

    [Fact(Timeout = 60000)]
    public async Task RuleBasedContentClassifier_WithNullPatterns_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new RuleBasedContentClassifier<double>(null!, threshold: 0.5));
    }

    #endregion

    #region ClassifyText Tests

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_NullOrEmpty_ReturnsSafeResult()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        var resultNull = classifier.ClassifyText(null!);
        var resultEmpty = classifier.ClassifyText(string.Empty);

        Assert.False(resultNull.IsHarmful);
        Assert.Equal("Allow", resultNull.RecommendedAction);
        Assert.False(resultEmpty.IsHarmful);
        Assert.Equal("Allow", resultEmpty.RecommendedAction);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_SafeContent_ReturnsNotHarmful()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        var result = classifier.ClassifyText("This is a friendly greeting. Hello world!");

        Assert.False(result.IsHarmful);
        Assert.Equal("Allow", result.RecommendedAction);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_ToxicContent_DetectsToxicity()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var result = classifier.ClassifyText("You are such a stupid idiot moron");

        Assert.True(result.IsHarmful);
        Assert.Contains("Toxic", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_ViolentContent_DetectsViolence()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.2);

        var result = classifier.ClassifyText("I will attack and destroy everything with a weapon");

        Assert.True(result.IsHarmful);
        Assert.Contains("Violence", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_HateSpeech_DetectsHateSpeech()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var result = classifier.ClassifyText("This is racist and discriminatory content");

        Assert.True(result.IsHarmful);
        Assert.Contains("HateSpeech", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_PrivateInformation_DetectsPrivateInfo()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var result = classifier.ClassifyText("My SSN is 123-45-6789");

        Assert.True(result.IsHarmful);
        Assert.Contains("PrivateInformation", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_CreditCardPattern_DetectsPrivateInfo()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var result = classifier.ClassifyText("Card number: 1234 5678 9012 3456");

        Assert.True(result.IsHarmful);
        Assert.Contains("PrivateInformation", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_SelfHarmContent_DetectsSelfHarm()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var result = classifier.ClassifyText("I'm feeling suicidal and want to self-harm");

        Assert.True(result.IsHarmful);
        Assert.Contains("SelfHarm", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_HarassmentContent_DetectsHarassment()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var result = classifier.ClassifyText("I will stalk and bully you and threaten you");

        Assert.True(result.IsHarmful);
        Assert.Contains("Harassment", result.DetectedCategories);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_MultipleCategories_DetectsAll()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.2);

        var result = classifier.ClassifyText("You stupid idiot, I will kill you with a weapon");

        Assert.True(result.IsHarmful);
        Assert.True(result.DetectedCategories.Length >= 2);
    }

    #endregion

    #region RecommendedAction Tests

    [Theory]
    [InlineData("Hello world", "Allow")]
    public void ClassifyText_ReturnsAppropriateAction(string text, string expectedAction)
    {
        var classifier = new RuleBasedContentClassifier<double>();

        var result = classifier.ClassifyText(text);

        Assert.Equal(expectedAction, result.RecommendedAction);
    }

    #endregion

    #region Pattern Management Tests

    [Fact(Timeout = 60000)]
    public async Task AddPattern_AddsNewCategory()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.5);

        classifier.AddPattern("CustomCategory", @"\bcustom\b");

        var categories = classifier.GetSupportedCategories();
        Assert.Contains("CustomCategory", categories);
    }

    [Fact(Timeout = 60000)]
    public async Task AddPattern_NullCategory_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentException>(() => classifier.AddPattern(null!, @"\btest\b"));
    }

    [Fact(Timeout = 60000)]
    public async Task AddPattern_EmptyPattern_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentException>(() => classifier.AddPattern("Test", string.Empty));
    }

    [Fact(Timeout = 60000)]
    public async Task ClearCategory_RemovesPatternsButKeepsCategory()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        // First verify toxic detection works
        var result1 = classifier.ClassifyText("You are stupid");
        Assert.Contains("Toxic", result1.DetectedCategories);

        // Clear the category
        classifier.ClearCategory("Toxic");

        // Now toxic content should not be detected (no patterns)
        var result2 = classifier.ClassifyText("You are stupid");
        Assert.DoesNotContain("Toxic", result2.DetectedCategories);
    }

    #endregion

    #region Serialization Tests

    [Fact(Timeout = 60000)]
    public async Task Serialize_ReturnsNonEmptyBytes()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        var bytes = classifier.Serialize();

        Assert.NotNull(bytes);
        Assert.NotEmpty(bytes);
    }

    [Fact(Timeout = 60000)]
    public async Task Deserialize_RestoresState()
    {
        var original = new RuleBasedContentClassifier<double>(threshold: 0.7);
        original.AddPattern("CustomCategory", @"\bcustom\b");

        var bytes = original.Serialize();

        var restored = new RuleBasedContentClassifier<double>();
        restored.Deserialize(bytes);

        Assert.True(restored.IsReady());
        Assert.Contains("CustomCategory", restored.GetSupportedCategories());
    }

    [Fact(Timeout = 60000)]
    public async Task Deserialize_NullData_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentNullException>(() => classifier.Deserialize(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task SerializeDeserialize_PreservesClassification()
    {
        var original = new RuleBasedContentClassifier<double>(threshold: 0.3);
        var testText = "You stupid idiot";

        var originalResult = original.ClassifyText(testText);
        var bytes = original.Serialize();

        var restored = new RuleBasedContentClassifier<double>();
        restored.Deserialize(bytes);
        var restoredResult = restored.ClassifyText(testText);

        Assert.Equal(originalResult.IsHarmful, restoredResult.IsHarmful);
        Assert.Equal(originalResult.DetectedCategories.Length, restoredResult.DetectedCategories.Length);
    }

    #endregion

    #region SaveModel/LoadModel Tests

    [Fact(Timeout = 60000)]
    public async Task SaveModel_NullPath_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentException>(() => classifier.SaveModel(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task SaveModel_EmptyPath_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentException>(() => classifier.SaveModel(string.Empty));
    }

    [Fact(Timeout = 60000)]
    public async Task LoadModel_NullPath_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentException>(() => classifier.LoadModel(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task LoadModel_NonExistentFile_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<FileNotFoundException>(() => classifier.LoadModel("nonexistent_file.json"));
    }

    [Fact(Timeout = 60000)]
    public async Task SaveAndLoadModel_PreservesState()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"classifier_test_{Guid.NewGuid()}.json");
        try
        {
            var original = new RuleBasedContentClassifier<double>(threshold: 0.6);
            original.AddPattern("TestCategory", @"\btest\b");
            original.SaveModel(tempPath);

            var loaded = new RuleBasedContentClassifier<double>();
            loaded.LoadModel(tempPath);

            Assert.True(loaded.IsReady());
            Assert.Contains("TestCategory", loaded.GetSupportedCategories());
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    #endregion

    #region Classify Vector Tests

    [Fact(Timeout = 60000)]
    public async Task Classify_NullVector_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentNullException>(() => classifier.Classify(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task Classify_EmptyVector_ReturnsResult()
    {
        var classifier = new RuleBasedContentClassifier<double>();
        var vector = new Vector<double>(10);

        var result = classifier.Classify(vector);

        Assert.NotNull(result);
    }

    #endregion

    #region ClassifyBatch Tests

    [Fact(Timeout = 60000)]
    public async Task ClassifyBatch_NullMatrix_ThrowsException()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        Assert.Throws<ArgumentNullException>(() => classifier.ClassifyBatch(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyBatch_ValidMatrix_ReturnsResults()
    {
        var classifier = new RuleBasedContentClassifier<double>();
        var matrix = new Matrix<double>(3, 10);

        var results = classifier.ClassifyBatch(matrix);

        Assert.NotNull(results);
        Assert.Equal(3, results.Length);
    }

    #endregion

    #region ContentClassificationResult Tests

    [Fact(Timeout = 60000)]
    public async Task ContentClassificationResult_DefaultValues()
    {
        var result = new ContentClassificationResult<double>();

        Assert.False(result.IsHarmful);
        Assert.Equal(string.Empty, result.PrimaryCategory);
        Assert.Equal("Allow", result.RecommendedAction);
        Assert.Equal(string.Empty, result.Explanation);
        Assert.NotNull(result.CategoryScores);
        Assert.NotNull(result.DetectedCategories);
    }

    #endregion

    #region Edge Cases and Regex Timeout Tests

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_CaseInsensitive_DetectsPatterns()
    {
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);

        var resultLower = classifier.ClassifyText("you are stupid");
        var resultUpper = classifier.ClassifyText("YOU ARE STUPID");
        var resultMixed = classifier.ClassifyText("You Are STUPID");

        Assert.Equal(resultLower.IsHarmful, resultUpper.IsHarmful);
        Assert.Equal(resultLower.IsHarmful, resultMixed.IsHarmful);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_VeryLongText_DoesNotTimeout()
    {
        var classifier = new RuleBasedContentClassifier<double>();
        var longText = string.Join(" ", Enumerable.Repeat("This is safe text.", 1000));

        var result = classifier.ClassifyText(longText);

        Assert.NotNull(result);
        Assert.False(result.IsHarmful);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassifyText_SpecialCharacters_HandlesGracefully()
    {
        var classifier = new RuleBasedContentClassifier<double>();
        var textWithSpecials = "Hello @#$%^&*() world! 123-456-7890";

        var result = classifier.ClassifyText(textWithSpecials);

        Assert.NotNull(result);
    }

    #endregion

    #region Default Categories Tests

    [Fact(Timeout = 60000)]
    public async Task GetSupportedCategories_ReturnsDefaultCategories()
    {
        var classifier = new RuleBasedContentClassifier<double>();

        var categories = classifier.GetSupportedCategories();

        Assert.Contains("Toxic", categories);
        Assert.Contains("Violence", categories);
        Assert.Contains("HateSpeech", categories);
        Assert.Contains("AdultContent", categories);
        Assert.Contains("Harassment", categories);
        Assert.Contains("SelfHarm", categories);
        Assert.Contains("PrivateInformation", categories);
    }

    #endregion
}
