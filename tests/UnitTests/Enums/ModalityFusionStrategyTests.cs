using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Enums;

namespace AiDotNetTests.UnitTests.Enums;

/// <summary>
/// Comprehensive unit tests for the ModalityFusionStrategy enum.
/// Tests enum values, uniqueness, conversion, and documentation coverage.
/// </summary>
[TestClass]
public class ModalityFusionStrategyTests
{
    [TestMethod]
    public void ModalityFusionStrategy_ShouldHaveAllExpectedValues()
    {
        // Arrange
        var expectedValues = new[]
        {
            ModalityFusionStrategy.EarlyFusion,
            ModalityFusionStrategy.LateFusion,
            ModalityFusionStrategy.CrossAttention,
            ModalityFusionStrategy.Hierarchical,
            ModalityFusionStrategy.Transformer,
            ModalityFusionStrategy.Gated,
            ModalityFusionStrategy.TensorFusion,
            ModalityFusionStrategy.BilinearPooling,
            ModalityFusionStrategy.AttentionWeighted,
            ModalityFusionStrategy.Concatenation
        };

        // Act
        var actualValues = Enum.GetValues(typeof(ModalityFusionStrategy))
            .Cast<ModalityFusionStrategy>()
            .ToArray();

        // Assert
        Assert.AreEqual(10, actualValues.Length, "Expected exactly 10 fusion strategy values");
        CollectionAssert.AreEquivalent(expectedValues, actualValues, "All expected values should be present");
    }

    [TestMethod]
    public void ModalityFusionStrategy_ShouldHaveUniqueValues()
    {
        // Arrange & Act
        var values = Enum.GetValues(typeof(ModalityFusionStrategy))
            .Cast<ModalityFusionStrategy>()
            .ToArray();

        var uniqueValues = values.Distinct().ToArray();

        // Assert
        Assert.AreEqual(values.Length, uniqueValues.Length, "All enum values should be unique");
    }

    [TestMethod]
    public void ModalityFusionStrategy_NoDuplicateHierarchical()
    {
        // Arrange & Act
        var hierarchicalValues = Enum.GetValues(typeof(ModalityFusionStrategy))
            .Cast<ModalityFusionStrategy>()
            .Where(v => v.ToString() == "Hierarchical")
            .ToArray();

        // Assert
        Assert.AreEqual(1, hierarchicalValues.Length, "Hierarchical should appear exactly once");
    }

    [TestMethod]
    public void EarlyFusion_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.EarlyFusion;

        // Act & Assert
        Assert.AreEqual("EarlyFusion", strategy.ToString());
        Assert.AreEqual(0, (int)strategy, "EarlyFusion should have value 0 as first enum member");
    }

    [TestMethod]
    public void LateFusion_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.LateFusion;

        // Act & Assert
        Assert.AreEqual("LateFusion", strategy.ToString());
        Assert.AreEqual(1, (int)strategy);
    }

    [TestMethod]
    public void CrossAttention_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.CrossAttention;

        // Act & Assert
        Assert.AreEqual("CrossAttention", strategy.ToString());
        Assert.AreEqual(2, (int)strategy);
    }

    [TestMethod]
    public void Hierarchical_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.Hierarchical;

        // Act & Assert
        Assert.AreEqual("Hierarchical", strategy.ToString());
        Assert.AreEqual(3, (int)strategy);
    }

    [TestMethod]
    public void Transformer_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.Transformer;

        // Act & Assert
        Assert.AreEqual("Transformer", strategy.ToString());
        Assert.AreEqual(4, (int)strategy);
    }

    [TestMethod]
    public void Gated_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.Gated;

        // Act & Assert
        Assert.AreEqual("Gated", strategy.ToString());
        Assert.AreEqual(5, (int)strategy);
    }

    [TestMethod]
    public void TensorFusion_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.TensorFusion;

        // Act & Assert
        Assert.AreEqual("TensorFusion", strategy.ToString());
        Assert.AreEqual(6, (int)strategy);
    }

    [TestMethod]
    public void BilinearPooling_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.BilinearPooling;

        // Act & Assert
        Assert.AreEqual("BilinearPooling", strategy.ToString());
        Assert.AreEqual(7, (int)strategy);
    }

    [TestMethod]
    public void AttentionWeighted_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.AttentionWeighted;

        // Act & Assert
        Assert.AreEqual("AttentionWeighted", strategy.ToString());
        Assert.AreEqual(8, (int)strategy);
    }

    [TestMethod]
    public void Concatenation_ShouldHaveCorrectValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.Concatenation;

        // Act & Assert
        Assert.AreEqual("Concatenation", strategy.ToString());
        Assert.AreEqual(9, (int)strategy);
    }

    [TestMethod]
    public void ParseString_EarlyFusion_ShouldSucceed()
    {
        // Act
        var parsed = Enum.Parse<ModalityFusionStrategy>("EarlyFusion");

        // Assert
        Assert.AreEqual(ModalityFusionStrategy.EarlyFusion, parsed);
    }

    [TestMethod]
    public void ParseString_LateFusion_ShouldSucceed()
    {
        // Act
        var parsed = Enum.Parse<ModalityFusionStrategy>("LateFusion");

        // Assert
        Assert.AreEqual(ModalityFusionStrategy.LateFusion, parsed);
    }

    [TestMethod]
    public void ParseString_CrossAttention_ShouldSucceed()
    {
        // Act
        var parsed = Enum.Parse<ModalityFusionStrategy>("CrossAttention");

        // Assert
        Assert.AreEqual(ModalityFusionStrategy.CrossAttention, parsed);
    }

    [TestMethod]
    public void ParseString_Hierarchical_ShouldSucceed()
    {
        // Act
        var parsed = Enum.Parse<ModalityFusionStrategy>("Hierarchical");

        // Assert
        Assert.AreEqual(ModalityFusionStrategy.Hierarchical, parsed);
    }

    [TestMethod]
    public void ParseString_CaseInsensitive_ShouldSucceed()
    {
        // Act
        var parsed = Enum.Parse<ModalityFusionStrategy>("earlyfusion", ignoreCase: true);

        // Assert
        Assert.AreEqual(ModalityFusionStrategy.EarlyFusion, parsed);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void ParseString_InvalidValue_ShouldThrowException()
    {
        // Act
        Enum.Parse<ModalityFusionStrategy>("InvalidStrategy");

        // Assert - Expects ArgumentException
    }

    [TestMethod]
    public void TryParse_ValidValue_ShouldReturnTrue()
    {
        // Act
        var success = Enum.TryParse<ModalityFusionStrategy>("Transformer", out var result);

        // Assert
        Assert.IsTrue(success);
        Assert.AreEqual(ModalityFusionStrategy.Transformer, result);
    }

    [TestMethod]
    public void TryParse_InvalidValue_ShouldReturnFalse()
    {
        // Act
        var success = Enum.TryParse<ModalityFusionStrategy>("NonExistent", out var result);

        // Assert
        Assert.IsFalse(success);
        Assert.AreEqual(default(ModalityFusionStrategy), result);
    }

    [TestMethod]
    public void GetNames_ShouldReturnAllStrategyNames()
    {
        // Act
        var names = Enum.GetNames(typeof(ModalityFusionStrategy));

        // Assert
        Assert.AreEqual(10, names.Length);
        Assert.IsTrue(names.Contains("EarlyFusion"));
        Assert.IsTrue(names.Contains("LateFusion"));
        Assert.IsTrue(names.Contains("CrossAttention"));
        Assert.IsTrue(names.Contains("Hierarchical"));
        Assert.IsTrue(names.Contains("Transformer"));
        Assert.IsTrue(names.Contains("Gated"));
        Assert.IsTrue(names.Contains("TensorFusion"));
        Assert.IsTrue(names.Contains("BilinearPooling"));
        Assert.IsTrue(names.Contains("AttentionWeighted"));
        Assert.IsTrue(names.Contains("Concatenation"));
    }

    [TestMethod]
    public void IsDefined_ValidValue_ShouldReturnTrue()
    {
        // Act & Assert
        Assert.IsTrue(Enum.IsDefined(typeof(ModalityFusionStrategy), ModalityFusionStrategy.EarlyFusion));
        Assert.IsTrue(Enum.IsDefined(typeof(ModalityFusionStrategy), ModalityFusionStrategy.Hierarchical));
        Assert.IsTrue(Enum.IsDefined(typeof(ModalityFusionStrategy), ModalityFusionStrategy.Concatenation));
    }

    [TestMethod]
    public void IsDefined_InvalidNumericValue_ShouldReturnFalse()
    {
        // Act & Assert
        Assert.IsFalse(Enum.IsDefined(typeof(ModalityFusionStrategy), 999));
    }

    [TestMethod]
    public void ConvertToInt_ShouldPreserveValue()
    {
        // Arrange
        var strategy = ModalityFusionStrategy.Gated;

        // Act
        var intValue = (int)strategy;
        var backToEnum = (ModalityFusionStrategy)intValue;

        // Assert
        Assert.AreEqual(5, intValue);
        Assert.AreEqual(strategy, backToEnum);
    }

    [TestMethod]
    public void EqualityComparison_ShouldWork()
    {
        // Arrange
        var strategy1 = ModalityFusionStrategy.TensorFusion;
        var strategy2 = ModalityFusionStrategy.TensorFusion;
        var strategy3 = ModalityFusionStrategy.BilinearPooling;

        // Act & Assert
        Assert.AreEqual(strategy1, strategy2);
        Assert.AreNotEqual(strategy1, strategy3);
    }

    [TestMethod]
    public void GetHashCode_SameValues_ShouldMatch()
    {
        // Arrange
        var strategy1 = ModalityFusionStrategy.AttentionWeighted;
        var strategy2 = ModalityFusionStrategy.AttentionWeighted;

        // Act & Assert
        Assert.AreEqual(strategy1.GetHashCode(), strategy2.GetHashCode());
    }

    [TestMethod]
    public void AllValues_ShouldBeSequential()
    {
        // Arrange
        var values = Enum.GetValues(typeof(ModalityFusionStrategy))
            .Cast<int>()
            .OrderBy(v => v)
            .ToArray();

        // Act & Assert - values should be 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        for (int i = 0; i < values.Length; i++)
        {
            Assert.AreEqual(i, values[i], $"Enum value at index {i} should be {i}");
        }
    }

    [TestMethod]
    public void Coverage_AllStrategiesAreTested()
    {
        // This test ensures we have explicit tests for all enum values
        // If a new strategy is added, this test will fail, reminding us to add tests

        // Arrange
        var allStrategies = Enum.GetValues(typeof(ModalityFusionStrategy))
            .Cast<ModalityFusionStrategy>()
            .ToArray();

        // Act - Count how many strategies we explicitly test above
        var testedStrategies = new[]
        {
            ModalityFusionStrategy.EarlyFusion,
            ModalityFusionStrategy.LateFusion,
            ModalityFusionStrategy.CrossAttention,
            ModalityFusionStrategy.Hierarchical,
            ModalityFusionStrategy.Transformer,
            ModalityFusionStrategy.Gated,
            ModalityFusionStrategy.TensorFusion,
            ModalityFusionStrategy.BilinearPooling,
            ModalityFusionStrategy.AttentionWeighted,
            ModalityFusionStrategy.Concatenation
        };

        // Assert
        Assert.AreEqual(allStrategies.Length, testedStrategies.Length,
            "All enum values should have individual tests. Update this test if you add new strategies.");

        foreach (var strategy in allStrategies)
        {
            Assert.IsTrue(testedStrategies.Contains(strategy),
                $"Strategy {strategy} should be in the tested strategies list");
        }
    }
}
