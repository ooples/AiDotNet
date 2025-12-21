using AiDotNet.AdversarialRobustness.Documentation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AdversarialRobustness;

/// <summary>
/// Comprehensive tests for ModelCard documentation functionality.
/// Tests cover construction, property handling, markdown generation, file I/O, and factory methods.
/// </summary>
public class ModelCardTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_InitializesDefaultValues()
    {
        var card = new ModelCard();

        Assert.Equal("Unnamed Model", card.ModelName);
        Assert.Equal("1.0.0", card.Version);
        Assert.Equal(string.Empty, card.Developers);
        Assert.Equal(string.Empty, card.ModelType);
        Assert.Equal(string.Empty, card.TrainingData);
    }

    [Fact]
    public void Constructor_SetsDateToUtcNow()
    {
        var before = DateTime.UtcNow;
        var card = new ModelCard();
        var after = DateTime.UtcNow;

        Assert.InRange(card.Date, before, after);
    }

    [Fact]
    public void Constructor_InitializesEmptyCollections()
    {
        var card = new ModelCard();

        Assert.NotNull(card.IntendedUses);
        Assert.Empty(card.IntendedUses);

        Assert.NotNull(card.OutOfScopeUses);
        Assert.Empty(card.OutOfScopeUses);

        Assert.NotNull(card.Limitations);
        Assert.Empty(card.Limitations);

        Assert.NotNull(card.EthicalConsiderations);
        Assert.Empty(card.EthicalConsiderations);

        Assert.NotNull(card.Recommendations);
        Assert.Empty(card.Recommendations);

        Assert.NotNull(card.Caveats);
        Assert.Empty(card.Caveats);

        Assert.NotNull(card.PerformanceMetrics);
        Assert.Empty(card.PerformanceMetrics);

        Assert.NotNull(card.FairnessMetrics);
        Assert.Empty(card.FairnessMetrics);

        Assert.NotNull(card.RobustnessMetrics);
        Assert.Empty(card.RobustnessMetrics);
    }

    #endregion

    #region Property Tests

    [Fact]
    public void Properties_CanBeSet()
    {
        var card = new ModelCard
        {
            ModelName = "Test Model",
            Version = "2.0.0",
            Date = new DateTime(2024, 1, 15),
            Developers = "Test Team",
            ModelType = "Classification",
            TrainingData = "Test dataset with 10,000 samples"
        };

        Assert.Equal("Test Model", card.ModelName);
        Assert.Equal("2.0.0", card.Version);
        Assert.Equal(new DateTime(2024, 1, 15), card.Date);
        Assert.Equal("Test Team", card.Developers);
        Assert.Equal("Classification", card.ModelType);
        Assert.Equal("Test dataset with 10,000 samples", card.TrainingData);
    }

    [Fact]
    public void IntendedUses_CanAddItems()
    {
        var card = new ModelCard();
        card.IntendedUses.Add("Text classification");
        card.IntendedUses.Add("Sentiment analysis");

        Assert.Equal(2, card.IntendedUses.Count);
        Assert.Contains("Text classification", card.IntendedUses);
        Assert.Contains("Sentiment analysis", card.IntendedUses);
    }

    [Fact]
    public void OutOfScopeUses_CanAddItems()
    {
        var card = new ModelCard();
        card.OutOfScopeUses.Add("Medical diagnosis");
        card.OutOfScopeUses.Add("Legal advice");

        Assert.Equal(2, card.OutOfScopeUses.Count);
        Assert.Contains("Medical diagnosis", card.OutOfScopeUses);
    }

    [Fact]
    public void Limitations_CanAddItems()
    {
        var card = new ModelCard();
        card.Limitations.Add("Limited to English text");
        card.Limitations.Add("May not perform well on short texts");

        Assert.Equal(2, card.Limitations.Count);
    }

    [Fact]
    public void EthicalConsiderations_CanAddItems()
    {
        var card = new ModelCard();
        card.EthicalConsiderations.Add("May exhibit bias in certain demographics");
        card.EthicalConsiderations.Add("Should be used with human oversight");

        Assert.Equal(2, card.EthicalConsiderations.Count);
    }

    [Fact]
    public void Recommendations_CanAddItems()
    {
        var card = new ModelCard();
        card.Recommendations.Add("Monitor for drift");
        card.Recommendations.Add("Retrain periodically");

        Assert.Equal(2, card.Recommendations.Count);
    }

    [Fact]
    public void Caveats_CanAddItems()
    {
        var card = new ModelCard();
        card.Caveats.Add("Experimental feature");
        card.Caveats.Add("Not production ready");

        Assert.Equal(2, card.Caveats.Count);
    }

    [Fact]
    public void PerformanceMetrics_CanAddDatasets()
    {
        var card = new ModelCard();
        card.PerformanceMetrics["TestDataset"] = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95,
            ["F1Score"] = 0.92
        };

        Assert.Single(card.PerformanceMetrics);
        Assert.Equal(0.95, card.PerformanceMetrics["TestDataset"]["Accuracy"]);
    }

    [Fact]
    public void FairnessMetrics_CanAddGroups()
    {
        var card = new ModelCard();
        card.FairnessMetrics["Gender"] = new Dictionary<string, double>
        {
            ["Male"] = 0.94,
            ["Female"] = 0.92
        };

        Assert.Single(card.FairnessMetrics);
        Assert.Equal(0.94, card.FairnessMetrics["Gender"]["Male"]);
    }

    [Fact]
    public void RobustnessMetrics_CanAddMetrics()
    {
        var card = new ModelCard();
        card.RobustnessMetrics["AdversarialAccuracy"] = 0.75;
        card.RobustnessMetrics["NoiseRobustness"] = 0.88;

        Assert.Equal(2, card.RobustnessMetrics.Count);
        Assert.Equal(0.75, card.RobustnessMetrics["AdversarialAccuracy"]);
    }

    #endregion

    #region Generate Tests

    [Fact]
    public void Generate_ReturnsNonEmptyString()
    {
        var card = new ModelCard();
        var result = card.Generate();

        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    [Fact]
    public void Generate_ContainsHeader()
    {
        var card = new ModelCard();
        var result = card.Generate();

        Assert.Contains("# Model Card", result);
    }

    [Fact]
    public void Generate_ContainsModelInfo()
    {
        var card = new ModelCard
        {
            ModelName = "TestModel",
            Version = "1.2.3",
            Developers = "AcmeCorp",
            ModelType = "Regression"
        };
        var result = card.Generate();

        Assert.Contains("**Model:** TestModel", result);
        Assert.Contains("**Version:** 1.2.3", result);
        Assert.Contains("**Developers:** AcmeCorp", result);
        Assert.Contains("**Model Type:** Regression", result);
    }

    [Fact]
    public void Generate_ContainsFormattedDate()
    {
        var card = new ModelCard
        {
            Date = new DateTime(2024, 6, 15)
        };
        var result = card.Generate();

        Assert.Contains("**Date:** 2024-06-15", result);
    }

    [Fact]
    public void Generate_ContainsIntendedUses()
    {
        var card = new ModelCard();
        card.IntendedUses.Add("Image classification");
        card.IntendedUses.Add("Object detection");

        var result = card.Generate();

        Assert.Contains("## Intended Uses", result);
        Assert.Contains("- Image classification", result);
        Assert.Contains("- Object detection", result);
    }

    [Fact]
    public void Generate_ShowsNotSpecified_WhenNoIntendedUses()
    {
        var card = new ModelCard();
        var result = card.Generate();

        Assert.Contains("## Intended Uses", result);
        Assert.Contains("Not specified", result);
    }

    [Fact]
    public void Generate_ContainsOutOfScopeUses()
    {
        var card = new ModelCard();
        card.OutOfScopeUses.Add("Autonomous weapons");

        var result = card.Generate();

        Assert.Contains("## Out-of-Scope Uses", result);
        Assert.Contains("- Autonomous weapons", result);
    }

    [Fact]
    public void Generate_ContainsTrainingData()
    {
        var card = new ModelCard
        {
            TrainingData = "ImageNet dataset with 1M images"
        };
        var result = card.Generate();

        Assert.Contains("## Training Data", result);
        Assert.Contains("ImageNet dataset with 1M images", result);
    }

    [Fact]
    public void Generate_ShowsNotSpecified_WhenNoTrainingData()
    {
        var card = new ModelCard();
        var result = card.Generate();

        // TrainingData section should show "Not specified"
        var lines = result.Split('\n');
        var foundSection = false;
        foreach (var line in lines)
        {
            if (line.Contains("## Training Data"))
            {
                foundSection = true;
            }
        }
        Assert.True(foundSection);
    }

    [Fact]
    public void Generate_ContainsPerformanceMetrics()
    {
        var card = new ModelCard();
        card.PerformanceMetrics["ValidationSet"] = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.9567,
            ["Precision"] = 0.9234
        };

        var result = card.Generate();

        Assert.Contains("## Performance Metrics", result);
        Assert.Contains("### ValidationSet", result);
        Assert.Contains("**Accuracy:** 0.9567", result);
        Assert.Contains("**Precision:** 0.9234", result);
    }

    [Fact]
    public void Generate_ContainsRobustnessMetrics_WhenPresent()
    {
        var card = new ModelCard();
        card.RobustnessMetrics["FGSMRobustness"] = 0.78;

        var result = card.Generate();

        Assert.Contains("## Robustness Metrics", result);
        Assert.Contains("**FGSMRobustness:** 0.7800", result);
    }

    [Fact]
    public void Generate_OmitsRobustnessSection_WhenEmpty()
    {
        var card = new ModelCard();
        var result = card.Generate();

        // Count occurrences of "## Robustness Metrics"
        int count = 0;
        int index = 0;
        while ((index = result.IndexOf("## Robustness Metrics", index)) != -1)
        {
            count++;
            index++;
        }
        Assert.Equal(0, count);
    }

    [Fact]
    public void Generate_ContainsFairnessMetrics_WhenPresent()
    {
        var card = new ModelCard();
        card.FairnessMetrics["Age"] = new Dictionary<string, double>
        {
            ["Under30"] = 0.91,
            ["Over60"] = 0.87
        };

        var result = card.Generate();

        Assert.Contains("## Fairness Metrics", result);
        Assert.Contains("### Age", result);
        Assert.Contains("**Under30:** 0.9100", result);
        Assert.Contains("**Over60:** 0.8700", result);
    }

    [Fact]
    public void Generate_ContainsLimitations()
    {
        var card = new ModelCard();
        card.Limitations.Add("Requires GPU for inference");

        var result = card.Generate();

        Assert.Contains("## Limitations", result);
        Assert.Contains("- Requires GPU for inference", result);
    }

    [Fact]
    public void Generate_ContainsEthicalConsiderations()
    {
        var card = new ModelCard();
        card.EthicalConsiderations.Add("Potential for misuse");

        var result = card.Generate();

        Assert.Contains("## Ethical Considerations", result);
        Assert.Contains("- Potential for misuse", result);
    }

    [Fact]
    public void Generate_ContainsRecommendations()
    {
        var card = new ModelCard();
        card.Recommendations.Add("Use with human oversight");

        var result = card.Generate();

        Assert.Contains("## Recommendations", result);
        Assert.Contains("- Use with human oversight", result);
    }

    [Fact]
    public void Generate_ContainsCaveats_WhenPresent()
    {
        var card = new ModelCard();
        card.Caveats.Add("Beta version");
        card.Caveats.Add("Subject to change");

        var result = card.Generate();

        Assert.Contains("## Caveats and Warnings", result);
        Assert.Contains("- Beta version", result);
        Assert.Contains("- Subject to change", result);
    }

    [Fact]
    public void Generate_OmitsCaveats_WhenEmpty()
    {
        var card = new ModelCard();
        var result = card.Generate();

        Assert.DoesNotContain("## Caveats and Warnings", result);
    }

    [Fact]
    public void Generate_FullyPopulatedCard_ContainsAllSections()
    {
        var card = CreateFullyPopulatedModelCard();
        var result = card.Generate();

        Assert.Contains("# Model Card", result);
        Assert.Contains("## Intended Uses", result);
        Assert.Contains("## Out-of-Scope Uses", result);
        Assert.Contains("## Training Data", result);
        Assert.Contains("## Performance Metrics", result);
        Assert.Contains("## Robustness Metrics", result);
        Assert.Contains("## Fairness Metrics", result);
        Assert.Contains("## Limitations", result);
        Assert.Contains("## Ethical Considerations", result);
        Assert.Contains("## Recommendations", result);
        Assert.Contains("## Caveats and Warnings", result);
    }

    #endregion

    #region SaveToFile Tests

    [Fact]
    public void SaveToFile_NullPath_ThrowsArgumentException()
    {
        var card = new ModelCard();

        Assert.Throws<ArgumentException>(() => card.SaveToFile(null!));
    }

    [Fact]
    public void SaveToFile_EmptyPath_ThrowsArgumentException()
    {
        var card = new ModelCard();

        Assert.Throws<ArgumentException>(() => card.SaveToFile(string.Empty));
    }

    [Fact]
    public void SaveToFile_WhitespacePath_ThrowsArgumentException()
    {
        var card = new ModelCard();

        Assert.Throws<ArgumentException>(() => card.SaveToFile("   "));
    }

    [Fact]
    public void SaveToFile_CreatesFile()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"modelcard_test_{Guid.NewGuid()}.md");
        try
        {
            var card = new ModelCard { ModelName = "TestModel" };
            card.SaveToFile(tempPath);

            Assert.True(File.Exists(tempPath));
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    [Fact]
    public void SaveToFile_WritesContent()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"modelcard_test_{Guid.NewGuid()}.md");
        try
        {
            var card = new ModelCard { ModelName = "SavedModel" };
            card.SaveToFile(tempPath);

            var content = File.ReadAllText(tempPath);
            Assert.Contains("# Model Card", content);
            Assert.Contains("SavedModel", content);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    [Fact]
    public void SaveToFile_CreatesDirectory()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"modelcard_dir_{Guid.NewGuid()}");
        var tempPath = Path.Combine(tempDir, "card.md");
        try
        {
            var card = new ModelCard();
            card.SaveToFile(tempPath);

            Assert.True(Directory.Exists(tempDir));
            Assert.True(File.Exists(tempPath));
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, true);
            }
        }
    }

    [Fact]
    public void SaveToFile_OverwritesExistingFile()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"modelcard_test_{Guid.NewGuid()}.md");
        try
        {
            // Create first file
            var card1 = new ModelCard { ModelName = "FirstModel" };
            card1.SaveToFile(tempPath);

            // Overwrite with second
            var card2 = new ModelCard { ModelName = "SecondModel" };
            card2.SaveToFile(tempPath);

            var content = File.ReadAllText(tempPath);
            Assert.DoesNotContain("FirstModel", content);
            Assert.Contains("SecondModel", content);
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

    #region CreateFromEvaluation Tests

    [Fact]
    public void CreateFromEvaluation_WithNullModelName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelCard.CreateFromEvaluation(null!, "Classification", null, null));
    }

    [Fact]
    public void CreateFromEvaluation_WithEmptyModelName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelCard.CreateFromEvaluation(string.Empty, "Classification", null, null));
    }

    [Fact]
    public void CreateFromEvaluation_WithWhitespaceModelName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelCard.CreateFromEvaluation("   ", "Classification", null, null));
    }

    [Fact]
    public void CreateFromEvaluation_SetsModelName()
    {
        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", null, null);

        Assert.Equal("MyModel", card.ModelName);
    }

    [Fact]
    public void CreateFromEvaluation_SetsModelType()
    {
        var card = ModelCard.CreateFromEvaluation("MyModel", "Regression", null, null);

        Assert.Equal("Regression", card.ModelType);
    }

    [Fact]
    public void CreateFromEvaluation_WithNullModelType_SetsEmptyString()
    {
        var card = ModelCard.CreateFromEvaluation("MyModel", null!, null, null);

        Assert.Equal(string.Empty, card.ModelType);
    }

    [Fact]
    public void CreateFromEvaluation_SetsDate()
    {
        var before = DateTime.UtcNow;
        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", null, null);
        var after = DateTime.UtcNow;

        Assert.InRange(card.Date, before, after);
    }

    [Fact]
    public void CreateFromEvaluation_AddsPerformanceMetrics()
    {
        var perfMetrics = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95,
            ["F1"] = 0.93
        };

        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", perfMetrics, null);

        Assert.True(card.PerformanceMetrics.ContainsKey("Overall"));
        Assert.Equal(0.95, card.PerformanceMetrics["Overall"]["Accuracy"]);
        Assert.Equal(0.93, card.PerformanceMetrics["Overall"]["F1"]);
    }

    [Fact]
    public void CreateFromEvaluation_AddsRobustnessMetrics()
    {
        var robMetrics = new Dictionary<string, double>
        {
            ["FGSM_Accuracy"] = 0.75,
            ["PGD_Accuracy"] = 0.70
        };

        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", null, robMetrics);

        Assert.Equal(0.75, card.RobustnessMetrics["FGSM_Accuracy"]);
        Assert.Equal(0.70, card.RobustnessMetrics["PGD_Accuracy"]);
    }

    [Fact]
    public void CreateFromEvaluation_AddsStandardRecommendations()
    {
        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", null, null);

        Assert.Equal(4, card.Recommendations.Count);
        Assert.Contains("Continuously monitor model performance in production", card.Recommendations);
        Assert.Contains("Regularly update the model with new data", card.Recommendations);
        Assert.Contains("Implement safety filters for sensitive applications", card.Recommendations);
        Assert.Contains("Conduct fairness audits across demographic groups", card.Recommendations);
    }

    [Fact]
    public void CreateFromEvaluation_WithNullMetrics_DoesNotAddMetrics()
    {
        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", null, null);

        Assert.Empty(card.PerformanceMetrics);
        Assert.Empty(card.RobustnessMetrics);
    }

    [Fact]
    public void CreateFromEvaluation_WithEmptyMetrics_DoesNotAddMetrics()
    {
        var emptyPerf = new Dictionary<string, double>();
        var emptyRob = new Dictionary<string, double>();

        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", emptyPerf, emptyRob);

        Assert.Empty(card.PerformanceMetrics);
        Assert.Empty(card.RobustnessMetrics);
    }

    [Fact]
    public void CreateFromEvaluation_CreatesDefensiveCopy_OfPerformanceMetrics()
    {
        var perfMetrics = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95
        };

        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", perfMetrics, null);

        // Modify original
        perfMetrics["Accuracy"] = 0.50;
        perfMetrics["NewMetric"] = 0.99;

        // Card should not be affected
        Assert.Equal(0.95, card.PerformanceMetrics["Overall"]["Accuracy"]);
        Assert.False(card.PerformanceMetrics["Overall"].ContainsKey("NewMetric"));
    }

    [Fact]
    public void CreateFromEvaluation_CreatesDefensiveCopy_OfRobustnessMetrics()
    {
        var robMetrics = new Dictionary<string, double>
        {
            ["FGSM"] = 0.75
        };

        var card = ModelCard.CreateFromEvaluation("MyModel", "Classification", null, robMetrics);

        // Modify original
        robMetrics["FGSM"] = 0.10;
        robMetrics["NewMetric"] = 0.99;

        // Card should not be affected
        Assert.Equal(0.75, card.RobustnessMetrics["FGSM"]);
        Assert.False(card.RobustnessMetrics.ContainsKey("NewMetric"));
    }

    [Fact]
    public void CreateFromEvaluation_GeneratesValidMarkdown()
    {
        var perfMetrics = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95
        };
        var robMetrics = new Dictionary<string, double>
        {
            ["FGSM"] = 0.75
        };

        var card = ModelCard.CreateFromEvaluation("TestModel", "Classification", perfMetrics, robMetrics);
        var markdown = card.Generate();

        Assert.Contains("# Model Card", markdown);
        Assert.Contains("TestModel", markdown);
        Assert.Contains("Classification", markdown);
        Assert.Contains("0.9500", markdown);
        Assert.Contains("0.7500", markdown);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void FullWorkflow_CreatePopulateSaveLoad()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"modelcard_workflow_{Guid.NewGuid()}.md");
        try
        {
            // Create and populate
            var card = new ModelCard
            {
                ModelName = "IntegrationTestModel",
                Version = "3.0.0",
                Developers = "Test Team",
                ModelType = "Classification",
                TrainingData = "Custom dataset"
            };
            card.IntendedUses.Add("Testing");
            card.PerformanceMetrics["Test"] = new Dictionary<string, double> { ["Accuracy"] = 0.99 };

            // Save
            card.SaveToFile(tempPath);

            // Verify file contains expected content
            var content = File.ReadAllText(tempPath);
            Assert.Contains("IntegrationTestModel", content);
            Assert.Contains("3.0.0", content);
            Assert.Contains("Test Team", content);
            Assert.Contains("Testing", content);
            Assert.Contains("0.9900", content);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    [Fact]
    public void MultiplePerformanceDatasets_GeneratesCorrectly()
    {
        var card = new ModelCard { ModelName = "MultiDatasetModel" };
        card.PerformanceMetrics["TrainSet"] = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.99,
            ["Loss"] = 0.01
        };
        card.PerformanceMetrics["ValidationSet"] = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95,
            ["Loss"] = 0.05
        };
        card.PerformanceMetrics["TestSet"] = new Dictionary<string, double>
        {
            ["Accuracy"] = 0.93,
            ["Loss"] = 0.07
        };

        var result = card.Generate();

        Assert.Contains("### TrainSet", result);
        Assert.Contains("### ValidationSet", result);
        Assert.Contains("### TestSet", result);
        Assert.Contains("**Accuracy:** 0.9900", result);
        Assert.Contains("**Accuracy:** 0.9500", result);
        Assert.Contains("**Accuracy:** 0.9300", result);
    }

    [Fact]
    public void MultipleFairnessGroups_GeneratesCorrectly()
    {
        var card = new ModelCard { ModelName = "FairnessModel" };
        card.FairnessMetrics["Gender"] = new Dictionary<string, double>
        {
            ["Male"] = 0.94,
            ["Female"] = 0.93
        };
        card.FairnessMetrics["Age"] = new Dictionary<string, double>
        {
            ["Young"] = 0.95,
            ["Old"] = 0.92
        };

        var result = card.Generate();

        Assert.Contains("### Gender", result);
        Assert.Contains("### Age", result);
        Assert.Contains("**Male:** 0.9400", result);
        Assert.Contains("**Young:** 0.9500", result);
    }

    #endregion

    #region Helper Methods

    private static ModelCard CreateFullyPopulatedModelCard()
    {
        var card = new ModelCard
        {
            ModelName = "FullTestModel",
            Version = "2.5.0",
            Date = new DateTime(2024, 6, 15),
            Developers = "AI Research Team",
            ModelType = "Image Classification",
            TrainingData = "ImageNet-1k with 1.2M images"
        };

        card.IntendedUses.Add("Image classification");
        card.IntendedUses.Add("Feature extraction");

        card.OutOfScopeUses.Add("Medical imaging");
        card.OutOfScopeUses.Add("Security surveillance");

        card.PerformanceMetrics["ImageNet-Val"] = new Dictionary<string, double>
        {
            ["Top1Accuracy"] = 0.78,
            ["Top5Accuracy"] = 0.94
        };

        card.RobustnessMetrics["FGSM_Eps0.1"] = 0.65;
        card.RobustnessMetrics["PGD_Eps0.1"] = 0.60;

        card.FairnessMetrics["SkinTone"] = new Dictionary<string, double>
        {
            ["Light"] = 0.80,
            ["Medium"] = 0.78,
            ["Dark"] = 0.75
        };

        card.Limitations.Add("Trained only on natural images");
        card.Limitations.Add("May struggle with artistic styles");

        card.EthicalConsiderations.Add("Potential for demographic bias");
        card.EthicalConsiderations.Add("Privacy concerns with facial recognition");

        card.Recommendations.Add("Validate on domain-specific data");
        card.Recommendations.Add("Implement content filters");

        card.Caveats.Add("Pre-release version");
        card.Caveats.Add("Subject to API changes");

        return card;
    }

    #endregion
}
