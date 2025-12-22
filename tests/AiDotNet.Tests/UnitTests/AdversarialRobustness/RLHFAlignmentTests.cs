using AiDotNet.AdversarialRobustness.Alignment;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AdversarialRobustness;

/// <summary>
/// Comprehensive tests for RLHF (Reinforcement Learning from Human Feedback) alignment.
/// Tests cover model alignment, evaluation, constitutional AI, red teaming, and serialization.
/// </summary>
public class RLHFAlignmentTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidOptions_CreatesInstance()
    {
        var options = new AlignmentMethodOptions<double>();

        var alignment = new RLHFAlignment<double>(options);

        Assert.NotNull(alignment);
        Assert.NotNull(alignment.GetOptions());
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new RLHFAlignment<double>(null!));
    }

    [Fact]
    public void Constructor_WithCustomOptions_PreservesOptions()
    {
        var options = new AlignmentMethodOptions<double>
        {
            LearningRate = 0.001,
            TrainingIterations = 500,
            Gamma = 0.95,
            KLCoefficient = 0.2,
            UseConstitutionalAI = false,
            CritiqueIterations = 5,
            EnableRedTeaming = false,
            RedTeamingAttempts = 50,
            RewardModelArchitecture = "MLP"
        };

        var alignment = new RLHFAlignment<double>(options);
        var retrievedOptions = alignment.GetOptions();

        Assert.Equal(0.001, retrievedOptions.LearningRate);
        Assert.Equal(500, retrievedOptions.TrainingIterations);
        Assert.Equal(0.95, retrievedOptions.Gamma);
        Assert.Equal(0.2, retrievedOptions.KLCoefficient);
        Assert.False(retrievedOptions.UseConstitutionalAI);
        Assert.Equal(5, retrievedOptions.CritiqueIterations);
        Assert.False(retrievedOptions.EnableRedTeaming);
        Assert.Equal(50, retrievedOptions.RedTeamingAttempts);
        Assert.Equal("MLP", retrievedOptions.RewardModelArchitecture);
    }

    #endregion

    #region AlignModel Tests

    [Fact]
    public void AlignModel_WithValidData_ReturnsAlignedModel()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();

        var alignedModel = alignment.AlignModel(baseModel, feedbackData);

        Assert.NotNull(alignedModel);
    }

    [Fact]
    public void AlignModel_SetsRewardModelTrained()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();

        Assert.False(alignment.IsRewardModelTrained);

        alignment.AlignModel(baseModel, feedbackData);

        Assert.True(alignment.IsRewardModelTrained);
    }

    [Fact]
    public void AlignModel_AlignedModelProducesPredictions()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();

        var alignedModel = alignment.AlignModel(baseModel, feedbackData);
        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var output = alignedModel.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void AlignModel_AlignedModelOutputsClampedTo01()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();

        var alignedModel = alignment.AlignModel(baseModel, feedbackData);
        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var output = alignedModel.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.InRange(output[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void AlignModel_WithEmptyFeedback_StillWorks()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = new AlignmentFeedbackData<double>();

        var alignedModel = alignment.AlignModel(baseModel, feedbackData);

        Assert.NotNull(alignedModel);
    }

    #endregion

    #region EvaluateAlignment Tests

    [Fact]
    public void EvaluateAlignment_WithNullModel_ThrowsArgumentNullException()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var evalData = CreateSampleEvaluationData();

        Assert.Throws<ArgumentNullException>(() => alignment.EvaluateAlignment(null!, evalData));
    }

    [Fact]
    public void EvaluateAlignment_WithNullEvaluationData_ThrowsArgumentNullException()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();

        Assert.Throws<ArgumentNullException>(() => alignment.EvaluateAlignment(model, null!));
    }

    [Fact]
    public void EvaluateAlignment_ReturnsValidMetrics()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();
        var evalData = CreateSampleEvaluationData();

        // Train the reward model first
        alignment.AlignModel(baseModel, feedbackData);

        var metrics = alignment.EvaluateAlignment(baseModel, evalData);

        Assert.NotNull(metrics);
        Assert.InRange(metrics.HelpfulnessScore, 0.0, 1.0);
        Assert.InRange(metrics.HarmlessnessScore, 0.0, 1.0);
        Assert.InRange(metrics.HonestyScore, 0.0, 1.0);
        Assert.InRange(metrics.OverallAlignmentScore, 0.0, 1.0);
    }

    [Fact]
    public void EvaluateAlignment_HonestyScoreIsOne()
    {
        // The IsHonest method always returns true (placeholder)
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var evalData = CreateSampleEvaluationData();

        var metrics = alignment.EvaluateAlignment(model, evalData);

        Assert.Equal(1.0, metrics.HonestyScore);
    }

    [Fact]
    public void EvaluateAlignment_OverallScoreIsAverageOfThreeScores()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var evalData = CreateSampleEvaluationData();

        var metrics = alignment.EvaluateAlignment(model, evalData);

        var expectedOverall = (metrics.HelpfulnessScore + metrics.HarmlessnessScore + metrics.HonestyScore) / 3.0;
        Assert.Equal(expectedOverall, metrics.OverallAlignmentScore, precision: 10);
    }

    [Fact]
    public void EvaluateAlignment_WithReferenceScores_ComputesPreferenceMatch()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();

        // Train reward model
        alignment.AlignModel(baseModel, feedbackData);

        var evalData = new AlignmentEvaluationData<double>
        {
            TestInputs = new Matrix<double>(new double[,] { { 0.5, 0.5, 0.5 } }),
            ExpectedOutputs = new Matrix<double>(new double[,] { { 0.5, 0.5, 0.5 } }),
            ReferenceScores = new double[] { 0.8 }
        };

        var metrics = alignment.EvaluateAlignment(baseModel, evalData);

        Assert.True(metrics.PreferenceMatchRate >= 0.0 && metrics.PreferenceMatchRate <= 1.0);
    }

    [Fact]
    public void EvaluateAlignment_WithoutRewardModel_UsesDefaultScore()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var evalData = new AlignmentEvaluationData<double>
        {
            TestInputs = new Matrix<double>(new double[,] { { 0.5, 0.5, 0.5 } }),
            ExpectedOutputs = new Matrix<double>(new double[,] { { 0.5, 0.5, 0.5 } }),
            ReferenceScores = new double[] { 0.5 }
        };

        // No AlignModel call, so reward model is not trained
        Assert.False(alignment.IsRewardModelTrained);

        var metrics = alignment.EvaluateAlignment(model, evalData);

        // Should still compute metrics without exception
        Assert.NotNull(metrics);
    }

    #endregion

    #region ApplyConstitutionalPrinciples Tests

    [Fact]
    public void ApplyConstitutionalPrinciples_WithNullModel_ThrowsArgumentNullException()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var principles = new[] { "Be helpful", "Be honest" };

        Assert.Throws<ArgumentNullException>(() => alignment.ApplyConstitutionalPrinciples(null!, principles));
    }

    [Fact]
    public void ApplyConstitutionalPrinciples_WithNullPrinciples_ThrowsArgumentNullException()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();

        Assert.Throws<ArgumentNullException>(() => alignment.ApplyConstitutionalPrinciples(model, null!));
    }

    [Fact]
    public void ApplyConstitutionalPrinciples_ReturnsWrappedModel()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var principles = new[] { "Be helpful", "Be honest", "Be harmless" };

        var constitutionalModel = alignment.ApplyConstitutionalPrinciples(model, principles);

        Assert.NotNull(constitutionalModel);
        Assert.NotSame(model, constitutionalModel);
    }

    [Fact]
    public void ApplyConstitutionalPrinciples_WrappedModelProducesPredictions()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var principles = new[] { "Be helpful", "Be honest" };

        var constitutionalModel = alignment.ApplyConstitutionalPrinciples(model, principles);
        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var output = constitutionalModel.Predict(input);

        Assert.NotNull(output);
        Assert.Equal(3, output.Length);
    }

    [Fact]
    public void ApplyConstitutionalPrinciples_WithEmptyPrinciples_Works()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var principles = Array.Empty<string>();

        var constitutionalModel = alignment.ApplyConstitutionalPrinciples(model, principles);

        Assert.NotNull(constitutionalModel);
    }

    [Fact]
    public void ApplyConstitutionalPrinciples_ModelMetadataPreserved()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var principles = new[] { "Be helpful" };

        var constitutionalModel = alignment.ApplyConstitutionalPrinciples(model, principles);
        var metadata = constitutionalModel.GetModelMetadata();

        Assert.NotNull(metadata);
    }

    #endregion

    #region PerformRedTeaming Tests

    [Fact]
    public void PerformRedTeaming_WithNullModel_ThrowsArgumentNullException()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var prompts = new Matrix<double>(2, 3);

        Assert.Throws<ArgumentNullException>(() => alignment.PerformRedTeaming(null!, prompts));
    }

    [Fact]
    public void PerformRedTeaming_WithNullPrompts_ThrowsArgumentNullException()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();

        Assert.Throws<ArgumentNullException>(() => alignment.PerformRedTeaming(model, null!));
    }

    [Fact]
    public void PerformRedTeaming_WithEmptyPrompts_ReturnsEmptyResults()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var prompts = Matrix<double>.Empty();

        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.NotNull(results);
        Assert.Equal(0.0, results.SuccessRate);
        Assert.Equal(0.0, results.AverageSeverity);
        Assert.Empty(results.Vulnerabilities);
    }

    [Fact]
    public void PerformRedTeaming_ReturnsValidResults()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MockPredictiveModel();
        var prompts = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5, 0.5 },
            { 0.3, 0.3, 0.3 },
            { 0.7, 0.7, 0.7 }
        });

        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.NotNull(results);
        Assert.Equal(3, results.SuccessfulAttacks.Length);
        Assert.Equal(3, results.SeverityScores.Length);
        Assert.Equal(3, results.VulnerabilityTypes.Length);
        Assert.InRange(results.SuccessRate, 0.0, 1.0);
    }

    [Fact]
    public void PerformRedTeaming_DetectsHighVarianceVulnerability()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new HighVarianceModel(); // Model that produces high variance output
        var prompts = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5, 0.5 }
        });

        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.True(results.SuccessfulAttacks[0]);
        Assert.Equal("HighVariance", results.VulnerabilityTypes[0]);
    }

    [Fact]
    public void PerformRedTeaming_DetectsExtremeBiasVulnerability()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new ExtremeBiasModel(); // Model that produces extreme bias
        var prompts = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5, 0.5 }
        });

        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.True(results.SuccessfulAttacks[0]);
        Assert.Equal("ExtremeBias", results.VulnerabilityTypes[0]);
    }

    [Fact]
    public void PerformRedTeaming_NoVulnerabilityForNormalOutput()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new NormalOutputModel(); // Model that produces normal output
        var prompts = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5, 0.5 }
        });

        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.False(results.SuccessfulAttacks[0]);
        Assert.Equal("None", results.VulnerabilityTypes[0]);
    }

    [Fact]
    public void PerformRedTeaming_CreatesVulnerabilityReports()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new HighVarianceModel();
        var prompts = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5, 0.5 }
        });

        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.Single(results.Vulnerabilities);
        var report = results.Vulnerabilities[0];
        Assert.Equal("HighVariance", report.Type);
        Assert.True(report.Severity > 0.0);
        Assert.NotEmpty(report.Description);
        Assert.NotEmpty(report.Recommendations);
    }

    [Fact]
    public void PerformRedTeaming_CalculatesCorrectSuccessRate()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var model = new MixedVulnerabilityModel();
        var prompts = new Matrix<double>(new double[,]
        {
            { 0.1, 0.1, 0.1 }, // Extreme bias (vulnerable)
            { 0.5, 0.5, 0.5 }, // Normal (not vulnerable)
            { 0.9, 0.9, 0.9 }  // Extreme bias (vulnerable)
        });

        var results = alignment.PerformRedTeaming(model, prompts);

        // 2 out of 3 should be vulnerable
        int vulnerableCount = results.SuccessfulAttacks.Count(x => x);
        Assert.Equal((double)vulnerableCount / 3, results.SuccessRate, precision: 10);
    }

    #endregion

    #region GetOptions Tests

    [Fact]
    public void GetOptions_ReturnsSameOptions()
    {
        var options = new AlignmentMethodOptions<double> { LearningRate = 0.002 };
        var alignment = new RLHFAlignment<double>(options);

        var retrieved = alignment.GetOptions();

        Assert.Same(options, retrieved);
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_CompletesWithoutError()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);

        // Should not throw
        alignment.Reset();
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_ReturnsNonEmptyBytes()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);

        var bytes = alignment.Serialize();

        Assert.NotNull(bytes);
        Assert.NotEmpty(bytes);
    }

    [Fact]
    public void Deserialize_WithNullData_ThrowsArgumentNullException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<ArgumentNullException>(() => alignment.Deserialize(null!));
    }

    [Fact]
    public void Deserialize_RestoresOptions()
    {
        var options = new AlignmentMethodOptions<double>
        {
            LearningRate = 0.003,
            TrainingIterations = 2000,
            KLCoefficient = 0.15
        };
        var alignment = new RLHFAlignment<double>(options);

        var bytes = alignment.Serialize();

        var newAlignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());
        newAlignment.Deserialize(bytes);

        var restored = newAlignment.GetOptions();
        Assert.Equal(0.003, restored.LearningRate);
        Assert.Equal(2000, restored.TrainingIterations);
        Assert.Equal(0.15, restored.KLCoefficient);
    }

    [Fact]
    public void Deserialize_ResetsRewardModel()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new RLHFAlignment<double>(options);
        var baseModel = new MockPredictiveModel();
        var feedbackData = CreateSampleFeedbackData();

        // Train reward model
        alignment.AlignModel(baseModel, feedbackData);
        Assert.True(alignment.IsRewardModelTrained);

        // Serialize and deserialize
        var bytes = alignment.Serialize();
        alignment.Deserialize(bytes);

        // Reward model should be null after deserialization
        Assert.False(alignment.IsRewardModelTrained);
    }

    [Fact]
    public void SerializeDeserialize_RoundTrip()
    {
        var options = new AlignmentMethodOptions<double>
        {
            LearningRate = 0.005,
            Gamma = 0.97,
            UseConstitutionalAI = false,
            CritiqueIterations = 7,
            EnableRedTeaming = false,
            RedTeamingAttempts = 200,
            RewardModelArchitecture = "CNN"
        };
        var alignment = new RLHFAlignment<double>(options);

        var bytes = alignment.Serialize();
        var newAlignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());
        newAlignment.Deserialize(bytes);

        var restored = newAlignment.GetOptions();
        Assert.Equal(0.005, restored.LearningRate);
        Assert.Equal(0.97, restored.Gamma);
        Assert.False(restored.UseConstitutionalAI);
        Assert.Equal(7, restored.CritiqueIterations);
        Assert.False(restored.EnableRedTeaming);
        Assert.Equal(200, restored.RedTeamingAttempts);
        Assert.Equal("CNN", restored.RewardModelArchitecture);
    }

    #endregion

    #region SaveModel/LoadModel Tests

    [Fact]
    public void SaveModel_NullPath_ThrowsException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<ArgumentException>(() => alignment.SaveModel(null!));
    }

    [Fact]
    public void SaveModel_EmptyPath_ThrowsException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<ArgumentException>(() => alignment.SaveModel(string.Empty));
    }

    [Fact]
    public void SaveModel_WhitespacePath_ThrowsException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<ArgumentException>(() => alignment.SaveModel("   "));
    }

    [Fact]
    public void LoadModel_NullPath_ThrowsException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<ArgumentException>(() => alignment.LoadModel(null!));
    }

    [Fact]
    public void LoadModel_EmptyPath_ThrowsException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<ArgumentException>(() => alignment.LoadModel(string.Empty));
    }

    [Fact]
    public void LoadModel_NonExistentFile_ThrowsFileNotFoundException()
    {
        var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());

        Assert.Throws<FileNotFoundException>(() => alignment.LoadModel("nonexistent_file.json"));
    }

    [Fact]
    public void SaveAndLoadModel_PreservesState()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"rlhf_test_{Guid.NewGuid()}.json");
        try
        {
            var options = new AlignmentMethodOptions<double>
            {
                LearningRate = 0.007,
                TrainingIterations = 1500,
                KLCoefficient = 0.25
            };
            var alignment = new RLHFAlignment<double>(options);
            alignment.SaveModel(tempPath);

            var loadedAlignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());
            loadedAlignment.LoadModel(tempPath);

            var loaded = loadedAlignment.GetOptions();
            Assert.Equal(0.007, loaded.LearningRate);
            Assert.Equal(1500, loaded.TrainingIterations);
            Assert.Equal(0.25, loaded.KLCoefficient);
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
    public void SaveModel_CreatesDirectory()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"rlhf_test_dir_{Guid.NewGuid()}");
        var tempPath = Path.Combine(tempDir, "model.json");
        try
        {
            var alignment = new RLHFAlignment<double>(new AlignmentMethodOptions<double>());
            alignment.SaveModel(tempPath);

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

    #endregion

    #region AlignmentFeedbackData Tests

    [Fact]
    public void AlignmentFeedbackData_ValidatePreferences_EmptyPreferences_ReturnsTrue()
    {
        var data = new AlignmentFeedbackData<double>();

        Assert.True(data.ValidatePreferences());
    }

    [Fact]
    public void AlignmentFeedbackData_ValidatePreferences_ValidPreferences_ReturnsTrue()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = new Matrix<double>(3, 2),
            Preferences = new[] { (0, 1), (1, 2), (0, 2) }
        };

        Assert.True(data.ValidatePreferences());
    }

    [Fact]
    public void AlignmentFeedbackData_ValidatePreferences_OutOfBoundsPreferred_ReturnsFalse()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = new Matrix<double>(3, 2),
            Preferences = new[] { (5, 1) } // 5 is out of bounds
        };

        Assert.False(data.ValidatePreferences());
    }

    [Fact]
    public void AlignmentFeedbackData_ValidatePreferences_NegativeIndex_ReturnsFalse()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = new Matrix<double>(3, 2),
            Preferences = new[] { (-1, 1) }
        };

        Assert.False(data.ValidatePreferences());
    }

    [Fact]
    public void AlignmentFeedbackData_ValidatePreferences_SameIndex_ReturnsFalse()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = new Matrix<double>(3, 2),
            Preferences = new[] { (1, 1) } // Same index
        };

        Assert.False(data.ValidatePreferences());
    }

    [Fact]
    public void AlignmentFeedbackData_EnsurePreferencesValid_ValidPreferences_NoException()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = new Matrix<double>(3, 2),
            Preferences = new[] { (0, 1) }
        };

        // Should not throw
        data.EnsurePreferencesValid();
    }

    [Fact]
    public void AlignmentFeedbackData_EnsurePreferencesValid_InvalidPreferences_ThrowsException()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = new Matrix<double>(3, 2),
            Preferences = new[] { (5, 1) }
        };

        Assert.Throws<InvalidOperationException>(() => data.EnsurePreferencesValid());
    }

    [Fact]
    public void AlignmentFeedbackData_EnsurePreferencesValid_PreferencesWithEmptyOutputs_ThrowsException()
    {
        var data = new AlignmentFeedbackData<double>
        {
            Outputs = Matrix<double>.Empty(),
            Preferences = new[] { (0, 1) }
        };

        Assert.Throws<InvalidOperationException>(() => data.EnsurePreferencesValid());
    }

    #endregion

    #region AlignmentMetrics Tests

    [Fact]
    public void AlignmentMetrics_DefaultValues()
    {
        var metrics = new AlignmentMetrics<double>();

        Assert.Equal(0.0, metrics.HelpfulnessScore);
        Assert.Equal(0.0, metrics.HarmlessnessScore);
        Assert.Equal(0.0, metrics.HonestyScore);
        Assert.Equal(0.0, metrics.OverallAlignmentScore);
        Assert.Equal(0.0, metrics.PreferenceMatchRate);
        Assert.Equal(0.0, metrics.ConstitutionalComplianceScore);
        Assert.NotNull(metrics.AdditionalMetrics);
    }

    [Fact]
    public void AlignmentMetrics_AdditionalMetrics_CanAdd()
    {
        var metrics = new AlignmentMetrics<double>();
        metrics.AdditionalMetrics["CustomMetric"] = 0.75;

        Assert.Equal(0.75, metrics.AdditionalMetrics["CustomMetric"]);
    }

    #endregion

    #region RedTeamingResults Tests

    [Fact]
    public void RedTeamingResults_DefaultValues()
    {
        var results = new RedTeamingResults<double>();

        Assert.NotNull(results.AdversarialPrompts);
        Assert.NotNull(results.ModelResponses);
        Assert.NotNull(results.SuccessfulAttacks);
        Assert.NotNull(results.SeverityScores);
        Assert.NotNull(results.VulnerabilityTypes);
        Assert.NotNull(results.Vulnerabilities);
        Assert.Equal(0.0, results.SuccessRate);
        Assert.Equal(0.0, results.AverageSeverity);
    }

    #endregion

    #region VulnerabilityReport Tests

    [Fact]
    public void VulnerabilityReport_DefaultValues()
    {
        var report = new VulnerabilityReport();

        Assert.Equal(string.Empty, report.Type);
        Assert.Equal(0.0, report.Severity);
        Assert.Equal(string.Empty, report.Description);
        Assert.Equal(string.Empty, report.ExamplePrompt);
        Assert.Equal(string.Empty, report.ProblematicResponse);
        Assert.NotNull(report.Recommendations);
    }

    [Fact]
    public void VulnerabilityReport_CanSetProperties()
    {
        var report = new VulnerabilityReport
        {
            Type = "TestVulnerability",
            Severity = 0.8,
            Description = "Test description",
            ExamplePrompt = "Test prompt",
            ProblematicResponse = "Test response",
            Recommendations = new[] { "Recommendation 1", "Recommendation 2" }
        };

        Assert.Equal("TestVulnerability", report.Type);
        Assert.Equal(0.8, report.Severity);
        Assert.Equal("Test description", report.Description);
        Assert.Equal("Test prompt", report.ExamplePrompt);
        Assert.Equal("Test response", report.ProblematicResponse);
        Assert.Equal(2, report.Recommendations.Length);
    }

    #endregion

    #region AlignmentMethodOptions Tests

    [Fact]
    public void AlignmentMethodOptions_DefaultValues()
    {
        var options = new AlignmentMethodOptions<double>();

        Assert.Equal(1e-5, options.LearningRate);
        Assert.Equal(1000, options.TrainingIterations);
        Assert.Equal(0.99, options.Gamma);
        Assert.Equal(0.1, options.KLCoefficient);
        Assert.True(options.UseConstitutionalAI);
        Assert.Equal(3, options.CritiqueIterations);
        Assert.True(options.EnableRedTeaming);
        Assert.Equal(100, options.RedTeamingAttempts);
        Assert.Equal("Transformer", options.RewardModelArchitecture);
    }

    #endregion

    #region Helper Methods

    private static AlignmentFeedbackData<double> CreateSampleFeedbackData()
    {
        return new AlignmentFeedbackData<double>
        {
            Inputs = new Matrix<double>(new double[,]
            {
                { 0.5, 0.5, 0.5 },
                { 0.3, 0.3, 0.3 }
            }),
            Outputs = new Matrix<double>(new double[,]
            {
                { 0.6, 0.6, 0.6 },
                { 0.4, 0.4, 0.4 }
            }),
            Preferences = new[] { (0, 1) },
            Ratings = new[] { 0.8, 0.6 },
            Rewards = new[] { 1.0, 0.5 }
        };
    }

    private static AlignmentEvaluationData<double> CreateSampleEvaluationData()
    {
        return new AlignmentEvaluationData<double>
        {
            TestInputs = new Matrix<double>(new double[,]
            {
                { 0.5, 0.5, 0.5 },
                { 0.3, 0.3, 0.3 }
            }),
            ExpectedOutputs = new Matrix<double>(new double[,]
            {
                { 0.5, 0.5, 0.5 },
                { 0.3, 0.3, 0.3 }
            }),
            EvaluationCriteria = new[] { "helpfulness", "harmlessness" },
            Labels = new[] { 0, 1 },
            ReferenceScores = new[] { 0.8, 0.7 }
        };
    }

    #endregion

    #region Mock Models

    /// <summary>
    /// Mock predictive model for testing RLHF alignment.
    /// </summary>
    private class MockPredictiveModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * 0.9 + 0.05;
            }
            return output;
        }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                Name = "MockPredictiveModel",
                ModelType = ModelType.NeuralNetwork
            };
        }

        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    /// <summary>
    /// Model that produces high variance output (triggers HighVariance vulnerability).
    /// </summary>
    private class HighVarianceModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        public Vector<double> Predict(Vector<double> input)
        {
            // Produce output with variance > 0.3
            // Mean = 0.6, Variance = (0.36 + 0.36 + 1.44) / 3 = 0.72 > 0.3
            return new Vector<double>(new double[] { 0.0, 0.0, 1.8 });
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    /// <summary>
    /// Model that produces extreme bias output (triggers ExtremeBias vulnerability).
    /// </summary>
    private class ExtremeBiasModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        public Vector<double> Predict(Vector<double> input)
        {
            // Mean < 0.2 triggers extreme bias
            return new Vector<double>(new double[] { 0.1, 0.1, 0.1 });
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    /// <summary>
    /// Model that produces normal output (no vulnerability).
    /// </summary>
    private class NormalOutputModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        public Vector<double> Predict(Vector<double> input)
        {
            // Mean in [0.2, 0.8] and variance < 0.3
            return new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    /// <summary>
    /// Model that produces different outputs based on input to test mixed vulnerabilities.
    /// </summary>
    private class MixedVulnerabilityModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        public Vector<double> Predict(Vector<double> input)
        {
            double sum = 0;
            for (int i = 0; i < input.Length; i++) sum += input[i];
            double mean = sum / input.Length;

            if (mean < 0.3)
            {
                // Extreme bias (mean < 0.2)
                return new Vector<double>(new double[] { 0.1, 0.1, 0.1 });
            }
            else if (mean > 0.7)
            {
                // Extreme bias (mean > 0.8)
                return new Vector<double>(new double[] { 0.9, 0.9, 0.9 });
            }
            else
            {
                // Normal output
                return new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
            }
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    #endregion
}
