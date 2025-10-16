using AiDotNet.Enums;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AiDotNetTests.UnitTests.Enums;

/// <summary>
/// Unit tests for the ModalityFusionStrategy enum to ensure all enum members are unique and properly defined.
/// </summary>
[TestClass]
public class ModalityFusionStrategyTests
{
    /// <summary>
    /// Tests that all enum values are unique (no duplicates).
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_AllValuesAreUnique()
    {
        // Arrange
        var allValues = Enum.GetValues(typeof(ModalityFusionStrategy)).Cast<ModalityFusionStrategy>();
        var distinctValues = allValues.Distinct();

        // Act & Assert
        Assert.AreEqual(allValues.Count(), distinctValues.Count(),
            "ModalityFusionStrategy enum should not contain duplicate values");
    }

    /// <summary>
    /// Tests that the enum contains the expected number of members.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_HasExpectedMemberCount()
    {
        // Arrange
        var expectedCount = 10; // EarlyFusion, LateFusion, CrossAttention, Hierarchical, Transformer, Gated, TensorFusion, BilinearPooling, AttentionWeighted, Concatenation

        // Act
        var actualCount = Enum.GetValues(typeof(ModalityFusionStrategy)).Length;

        // Assert
        Assert.AreEqual(expectedCount, actualCount,
            $"ModalityFusionStrategy should have exactly {expectedCount} members");
    }

    /// <summary>
    /// Tests that each expected enum member exists and is accessible.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_AllExpectedMembersExist()
    {
        // Arrange & Act & Assert
        Assert.AreEqual(ModalityFusionStrategy.EarlyFusion, ModalityFusionStrategy.EarlyFusion);
        Assert.AreEqual(ModalityFusionStrategy.LateFusion, ModalityFusionStrategy.LateFusion);
        Assert.AreEqual(ModalityFusionStrategy.CrossAttention, ModalityFusionStrategy.CrossAttention);
        Assert.AreEqual(ModalityFusionStrategy.Hierarchical, ModalityFusionStrategy.Hierarchical);
        Assert.AreEqual(ModalityFusionStrategy.Transformer, ModalityFusionStrategy.Transformer);
        Assert.AreEqual(ModalityFusionStrategy.Gated, ModalityFusionStrategy.Gated);
        Assert.AreEqual(ModalityFusionStrategy.TensorFusion, ModalityFusionStrategy.TensorFusion);
        Assert.AreEqual(ModalityFusionStrategy.BilinearPooling, ModalityFusionStrategy.BilinearPooling);
        Assert.AreEqual(ModalityFusionStrategy.AttentionWeighted, ModalityFusionStrategy.AttentionWeighted);
        Assert.AreEqual(ModalityFusionStrategy.Concatenation, ModalityFusionStrategy.Concatenation);
    }

    /// <summary>
    /// Tests that the Hierarchical enum member has the expected value.
    /// This specifically validates that the duplicate was properly removed.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_HierarchicalHasCorrectValue()
    {
        // Arrange
        var expectedValue = 3; // Fourth member (0-indexed)

        // Act
        var actualValue = (int)ModalityFusionStrategy.Hierarchical;

        // Assert
        Assert.AreEqual(expectedValue, actualValue,
            "Hierarchical should be the 4th enum member with value 3");
    }

    /// <summary>
    /// Tests that enum can be parsed from string correctly.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_CanParseFromString()
    {
        // Arrange
        var strategyName = "Hierarchical";

        // Act
        var parsed = Enum.Parse(typeof(ModalityFusionStrategy), strategyName);

        // Assert
        Assert.AreEqual(ModalityFusionStrategy.Hierarchical, parsed);
    }

    /// <summary>
    /// Tests that enum names don't have duplicates by checking GetNames.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_NamesAreUnique()
    {
        // Arrange
        var allNames = Enum.GetNames(typeof(ModalityFusionStrategy));
        var distinctNames = allNames.Distinct();

        // Act & Assert
        Assert.AreEqual(allNames.Length, distinctNames.Count(),
            "ModalityFusionStrategy enum should not contain duplicate names");
    }

    /// <summary>
    /// Tests that the enum ToString() method works correctly for all members.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_ToStringWorksCorrectly()
    {
        // Arrange & Act & Assert
        Assert.AreEqual("EarlyFusion", ModalityFusionStrategy.EarlyFusion.ToString());
        Assert.AreEqual("LateFusion", ModalityFusionStrategy.LateFusion.ToString());
        Assert.AreEqual("CrossAttention", ModalityFusionStrategy.CrossAttention.ToString());
        Assert.AreEqual("Hierarchical", ModalityFusionStrategy.Hierarchical.ToString());
        Assert.AreEqual("Transformer", ModalityFusionStrategy.Transformer.ToString());
        Assert.AreEqual("Gated", ModalityFusionStrategy.Gated.ToString());
        Assert.AreEqual("TensorFusion", ModalityFusionStrategy.TensorFusion.ToString());
        Assert.AreEqual("BilinearPooling", ModalityFusionStrategy.BilinearPooling.ToString());
        Assert.AreEqual("AttentionWeighted", ModalityFusionStrategy.AttentionWeighted.ToString());
        Assert.AreEqual("Concatenation", ModalityFusionStrategy.Concatenation.ToString());
    }

    /// <summary>
    /// Tests that default enum value is the first member.
    /// </summary>
    [TestMethod]
    public void ModalityFusionStrategy_DefaultValueIsEarlyFusion()
    {
        // Arrange
        ModalityFusionStrategy defaultValue = default;

        // Act & Assert
        Assert.AreEqual(ModalityFusionStrategy.EarlyFusion, defaultValue);
        Assert.AreEqual(0, (int)defaultValue);
    }
}
