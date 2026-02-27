using AiDotNet.AutoML;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AutoML;

/// <summary>
/// Deep integration tests for AutoML:
/// Architecture (operations, serialization, description),
/// TrialResult (defaults, clone, clone-redacted),
/// ParameterRange (defaults, clone),
/// SearchConstraint (defaults, clone),
/// CompressionTrial (defaults),
/// ConstraintType enum.
/// </summary>
public class AutoMLDeepMathIntegrationTests
{
    // ============================
    // Architecture: Construction
    // ============================

    [Fact]
    public void Architecture_Defaults_EmptyOperations()
    {
        var arch = new Architecture<double>();
        Assert.Empty(arch.Operations);
        Assert.Equal(0, arch.NodeCount);
        Assert.Empty(arch.NodeChannels);
    }

    [Fact]
    public void Architecture_AddOperation_UpdatesNodeCount()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        Assert.Single(arch.Operations);
        Assert.Equal(2, arch.NodeCount); // max(1, 0) + 1 = 2
    }

    [Fact]
    public void Architecture_AddMultipleOperations_NodeCountTracksMax()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 0, "skip");
        arch.AddOperation(3, 1, "relu");

        Assert.Equal(3, arch.Operations.Count);
        Assert.Equal(4, arch.NodeCount); // max(3, 1) + 1 = 4
    }

    [Fact]
    public void Architecture_GetDescription_ContainsNodeInfo()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 1, "batch_norm");

        string desc = arch.GetDescription();
        Assert.Contains("2", desc); // Node count is in description
        Assert.Contains("conv3x3", desc);
        Assert.Contains("batch_norm", desc);
    }

    [Fact]
    public void Architecture_GetDescription_EmptyArchitecture()
    {
        var arch = new Architecture<double>();
        string desc = arch.GetDescription();
        Assert.Contains("0", desc); // 0 nodes
    }

    // ============================
    // Architecture: NodeChannels
    // ============================

    [Fact]
    public void Architecture_NodeChannels_SetAndGet()
    {
        var arch = new Architecture<double>();
        arch.NodeChannels[0] = 16;
        arch.NodeChannels[1] = 32;
        arch.NodeChannels[2] = 64;

        Assert.Equal(16, arch.NodeChannels[0]);
        Assert.Equal(32, arch.NodeChannels[1]);
        Assert.Equal(64, arch.NodeChannels[2]);
    }

    // ============================
    // Architecture: JSON Serialization
    // ============================

    [Fact]
    public void Architecture_ToJson_RoundTrip()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 1, "relu");
        arch.NodeChannels[0] = 3;
        arch.NodeChannels[1] = 16;

        string json = arch.ToJson();
        var restored = Architecture<double>.FromJson(json);

        Assert.Equal(arch.NodeCount, restored.NodeCount);
        Assert.Equal(arch.Operations.Count, restored.Operations.Count);
        Assert.Equal("conv3x3", restored.Operations[0].Operation);
        Assert.Equal("relu", restored.Operations[1].Operation);
    }

    [Fact]
    public void Architecture_ToJson_IndentedByDefault()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");

        string json = arch.ToJson();
        Assert.Contains("\n", json); // Indented has newlines
    }

    [Fact]
    public void Architecture_ToJson_CompactWhenNotIndented()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");

        string json = arch.ToJson(indented: false);
        Assert.DoesNotContain("\n", json);
    }

    [Fact]
    public void Architecture_FromJson_NullThrows()
    {
        Assert.Throws<ArgumentNullException>(() => Architecture<double>.FromJson(null!));
    }

    [Fact]
    public void Architecture_FromJson_EmptyThrows()
    {
        Assert.Throws<ArgumentNullException>(() => Architecture<double>.FromJson(string.Empty));
    }

    // ============================
    // Architecture: Binary Serialization
    // ============================

    [Fact]
    public void Architecture_ToBytes_RoundTrip()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 1, "max_pool");
        arch.NodeChannels[0] = 3;
        arch.NodeChannels[1] = 16;

        byte[] bytes = arch.ToBytes();
        var restored = Architecture<double>.FromBytes(bytes);

        Assert.Equal(arch.NodeCount, restored.NodeCount);
        Assert.Equal(arch.Operations.Count, restored.Operations.Count);
        Assert.Equal("conv3x3", restored.Operations[0].Operation);
        Assert.Equal(16, restored.NodeChannels[1]);
    }

    [Fact]
    public void Architecture_FromBytes_NullThrows()
    {
        Assert.Throws<ArgumentNullException>(() => Architecture<double>.FromBytes(null!));
    }

    [Fact]
    public void Architecture_FromBytes_EmptyThrows()
    {
        Assert.Throws<ArgumentNullException>(() => Architecture<double>.FromBytes(Array.Empty<byte>()));
    }

    [Fact]
    public void Architecture_ToBytes_CompactThanJson()
    {
        var arch = new Architecture<double>();
        for (int i = 0; i < 10; i++)
        {
            arch.AddOperation(i + 1, i, $"op_{i}");
        }

        byte[] bytes = arch.ToBytes();
        string json = arch.ToJson(indented: false);

        // Binary should typically be more compact than JSON
        Assert.True(bytes.Length < json.Length,
            $"Binary ({bytes.Length} bytes) should be more compact than JSON ({json.Length} chars)");
    }

    // ============================
    // TrialResult: Defaults
    // ============================

    [Fact]
    public void TrialResult_Defaults()
    {
        var result = new TrialResult();
        Assert.Equal(0, result.TrialId);
        Assert.Null(result.CandidateModelType);
        Assert.NotNull(result.Parameters);
        Assert.Empty(result.Parameters);
        Assert.Equal(0.0, result.Score);
        Assert.Equal(TimeSpan.Zero, result.Duration);
        Assert.Null(result.Metadata);
        Assert.True(result.Success);
        Assert.Null(result.ErrorMessage);
    }

    [Fact]
    public void TrialResult_SetProperties()
    {
        var result = new TrialResult
        {
            TrialId = 42,
            CandidateModelType = ModelType.SimpleRegression,
            Score = 0.95,
            Duration = TimeSpan.FromSeconds(30),
            Timestamp = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            Success = true
        };
        result.Parameters["learning_rate"] = 0.01;

        Assert.Equal(42, result.TrialId);
        Assert.Equal(0.95, result.Score);
        Assert.Single(result.Parameters);
        Assert.Equal(0.01, result.Parameters["learning_rate"]);
    }

    // ============================
    // TrialResult: Clone
    // ============================

    [Fact]
    public void TrialResult_Clone_DeepCopiesParameters()
    {
        var original = new TrialResult
        {
            TrialId = 1,
            Score = 0.85,
            Parameters = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["lr"] = 0.01,
                ["epochs"] = 100
            }
        };

        var clone = original.Clone();

        Assert.Equal(original.TrialId, clone.TrialId);
        Assert.Equal(original.Score, clone.Score);
        Assert.Equal(2, clone.Parameters.Count);

        // Verify deep copy: modifying clone doesn't affect original
        clone.Parameters["lr"] = 0.1;
        Assert.Equal(0.01, original.Parameters["lr"]);
    }

    [Fact]
    public void TrialResult_Clone_PreservesAllFields()
    {
        var original = new TrialResult
        {
            TrialId = 5,
            CandidateModelType = ModelType.AutoML,
            Score = 0.92,
            Duration = TimeSpan.FromMinutes(2),
            Timestamp = DateTime.UtcNow,
            Success = false,
            ErrorMessage = "timeout"
        };

        var clone = original.Clone();
        Assert.Equal(original.TrialId, clone.TrialId);
        Assert.Equal(original.CandidateModelType, clone.CandidateModelType);
        Assert.Equal(original.Score, clone.Score);
        Assert.Equal(original.Duration, clone.Duration);
        Assert.Equal(original.Success, clone.Success);
        Assert.Equal(original.ErrorMessage, clone.ErrorMessage);
    }

    // ============================
    // TrialResult: CloneRedacted
    // ============================

    [Fact]
    public void TrialResult_CloneRedacted_ClearsParameters()
    {
        var original = new TrialResult
        {
            TrialId = 1,
            Score = 0.85,
            Parameters = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["lr"] = 0.01,
                ["epochs"] = 100
            },
            Metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["internal_data"] = "secret"
            }
        };

        var redacted = original.CloneRedacted();

        Assert.Equal(original.TrialId, redacted.TrialId);
        Assert.Equal(original.Score, redacted.Score);
        Assert.Empty(redacted.Parameters); // Parameters redacted
        Assert.Null(redacted.Metadata); // Metadata nulled
    }

    [Fact]
    public void TrialResult_CloneRedacted_PreservesPublicFields()
    {
        var original = new TrialResult
        {
            TrialId = 10,
            CandidateModelType = ModelType.SimpleRegression,
            Score = 0.99,
            Duration = TimeSpan.FromHours(1),
            Success = true,
            ErrorMessage = null
        };

        var redacted = original.CloneRedacted();
        Assert.Equal(original.CandidateModelType, redacted.CandidateModelType);
        Assert.Equal(original.Duration, redacted.Duration);
        Assert.True(redacted.Success);
    }

    // ============================
    // ParameterRange: Defaults
    // ============================

    [Fact]
    public void ParameterRange_Defaults()
    {
        var range = new ParameterRange();
        Assert.Equal(ParameterType.Integer, range.Type); // First enum value
        Assert.Null(range.MinValue);
        Assert.Null(range.MaxValue);
        Assert.Null(range.Step);
        Assert.Null(range.CategoricalValues);
        Assert.False(range.UseLogScale);
        Assert.Null(range.DefaultValue);
    }

    [Fact]
    public void ParameterRange_SetProperties()
    {
        var range = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = 0.001,
            MaxValue = 1.0,
            Step = 0.001,
            UseLogScale = true,
            DefaultValue = 0.01
        };

        Assert.Equal(ParameterType.Float, range.Type);
        Assert.Equal(0.001, range.MinValue);
        Assert.Equal(1.0, range.MaxValue);
        Assert.Equal(0.001, range.Step);
        Assert.True(range.UseLogScale);
        Assert.Equal(0.01, range.DefaultValue);
    }

    // ============================
    // ParameterRange: Clone
    // ============================

    [Fact]
    public void ParameterRange_Clone_DeepCopy()
    {
        var original = new ParameterRange
        {
            Type = ParameterType.Categorical,
            CategoricalValues = new List<object> { "relu", "tanh", "sigmoid" },
            DefaultValue = "relu"
        };

        var clone = (ParameterRange)original.Clone();

        Assert.Equal(original.Type, clone.Type);
        Assert.Equal(3, clone.CategoricalValues?.Count);

        // Verify deep copy
        clone.CategoricalValues?.Add("elu");
        Assert.Equal(3, original.CategoricalValues.Count);
    }

    // ============================
    // ParameterRange: Log Scale Math
    // ============================

    [Theory]
    [InlineData(0.001, 1.0, 10)]
    [InlineData(0.0001, 0.1, 10)]
    public void ParameterRange_LogScale_SamplingRange(double min, double max, int numSamples)
    {
        var range = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = min,
            MaxValue = max,
            UseLogScale = true
        };

        // Log-uniform sampling: sample = exp(uniform(log(min), log(max)))
        double logMin = Math.Log((double)range.MinValue);
        double logMax = Math.Log((double)range.MaxValue);

        Assert.True(logMin < logMax);
        Assert.True(logMax - logMin > 0);

        // Verify log spacing creates geometrically distributed samples
        for (int i = 0; i < numSamples; i++)
        {
            double t = (double)i / (numSamples - 1);
            double logSample = logMin + t * (logMax - logMin);
            double sample = Math.Exp(logSample);

            Assert.True(sample >= (double)range.MinValue - 1e-15);
            Assert.True(sample <= (double)range.MaxValue + 1e-15);
        }
    }

    // ============================
    // SearchConstraint: Defaults
    // ============================

    [Fact]
    public void SearchConstraint_Defaults()
    {
        var constraint = new SearchConstraint();
        Assert.Equal(string.Empty, constraint.Name);
        Assert.Equal(ConstraintType.Range, constraint.Type);
        Assert.NotNull(constraint.ParameterNames);
        Assert.Empty(constraint.ParameterNames);
        Assert.Equal(string.Empty, constraint.Expression);
        Assert.Null(constraint.MinValue);
        Assert.Null(constraint.MaxValue);
        Assert.True(constraint.IsHardConstraint);
        Assert.NotNull(constraint.Metadata);
        Assert.Empty(constraint.Metadata);
    }

    [Fact]
    public void SearchConstraint_SetProperties()
    {
        var constraint = new SearchConstraint
        {
            Name = "memory_limit",
            Type = ConstraintType.Resource,
            ParameterNames = new List<string> { "hidden_size", "num_layers" },
            MaxValue = 8192.0,
            IsHardConstraint = true
        };

        Assert.Equal("memory_limit", constraint.Name);
        Assert.Equal(ConstraintType.Resource, constraint.Type);
        Assert.Equal(2, constraint.ParameterNames.Count);
        Assert.Equal(8192.0, constraint.MaxValue);
    }

    // ============================
    // SearchConstraint: Clone
    // ============================

    [Fact]
    public void SearchConstraint_Clone_DeepCopy()
    {
        var original = new SearchConstraint
        {
            Name = "test",
            ParameterNames = new List<string> { "p1", "p2" },
            Metadata = new Dictionary<string, object> { ["key"] = "value" }
        };

        var clone = (SearchConstraint)original.Clone();

        Assert.Equal(original.Name, clone.Name);
        Assert.Equal(2, clone.ParameterNames.Count);

        // Verify deep copy of lists
        clone.ParameterNames.Add("p3");
        Assert.Equal(2, original.ParameterNames.Count);

        // Verify deep copy of dictionary
        clone.Metadata["key2"] = "value2";
        Assert.Single(original.Metadata);
    }

    // ============================
    // ConstraintType Enum
    // ============================

    [Fact]
    public void ConstraintType_HasFiveValues()
    {
        var values = (((ConstraintType[])Enum.GetValues(typeof(ConstraintType))));
        Assert.Equal(5, values.Length);
    }

    [Theory]
    [InlineData(ConstraintType.Range)]
    [InlineData(ConstraintType.Dependency)]
    [InlineData(ConstraintType.Exclusion)]
    [InlineData(ConstraintType.Resource)]
    [InlineData(ConstraintType.Custom)]
    public void ConstraintType_AllValuesValid(ConstraintType type)
    {
        Assert.True(Enum.IsDefined(typeof(ConstraintType), type));
    }

    // ============================
    // CompressionTrial: Defaults
    // ============================

    [Fact]
    public void CompressionTrial_Defaults()
    {
        var trial = new CompressionTrial<double>();
        Assert.NotNull(trial.Hyperparameters);
        Assert.Empty(trial.Hyperparameters);
        Assert.Null(trial.Metrics);
        Assert.False(trial.Success);
        Assert.Null(trial.ErrorMessage);
    }

    [Fact]
    public void CompressionTrial_SetProperties()
    {
        var trial = new CompressionTrial<double>
        {
            Technique = CompressionType.SparsePruning,
            FitnessScore = 0.95,
            Success = true
        };
        trial.Hyperparameters["sparsity"] = 0.5;

        Assert.Equal(CompressionType.SparsePruning, trial.Technique);
        Assert.Equal(0.95, trial.FitnessScore);
        Assert.True(trial.Success);
        Assert.Single(trial.Hyperparameters);
    }
}
