using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AutoML;

/// <summary>
/// Integration tests for AutoML core components: ParameterSampler, ParameterRange,
/// TrialResult, Architecture, SearchSpace, and BudgetDefaults.
/// </summary>
public class AutoMLCoreComponentsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region AutoMLParameterSampler Tests

    [Fact]
    public void ParameterSampler_IntegerRange_StaysInBounds()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["epochs"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 10,
                MaxValue = 100
            }
        };

        for (int trial = 0; trial < 100; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            int value = (int)sample["epochs"];

            Assert.True(value >= 10 && value <= 100,
                $"Trial {trial}: epochs={value} outside [10, 100]");
        }
    }

    [Fact]
    public void ParameterSampler_FloatRange_StaysInBounds()
    {
        var random = RandomHelper.CreateSeededRandom(137);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["learningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.001,
                MaxValue = 0.1
            }
        };

        for (int trial = 0; trial < 100; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            double value = (double)sample["learningRate"];

            Assert.True(value >= 0.001 - Tolerance && value <= 0.1 + Tolerance,
                $"Trial {trial}: learningRate={value} outside [0.001, 0.1]");
        }
    }

    [Fact]
    public void ParameterSampler_LogScale_ConcentratesInLowerRange()
    {
        var random = RandomHelper.CreateSeededRandom(173);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["lr"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 1e-5,
                MaxValue = 1.0,
                UseLogScale = true
            }
        };

        int belowMedian = 0;
        double linearMedian = (1e-5 + 1.0) / 2;

        for (int trial = 0; trial < 1000; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            double value = (double)sample["lr"];

            Assert.True(value >= 1e-5 - Tolerance && value <= 1.0 + Tolerance);

            if (value < linearMedian)
            {
                belowMedian++;
            }
        }

        // With log scale, most samples should be below the linear median
        Assert.True(belowMedian > 800, $"Log scale should concentrate values below median, but only {belowMedian}/1000 were");
    }

    [Fact]
    public void ParameterSampler_BooleanType_ReturnsTrueOrFalse()
    {
        var random = RandomHelper.CreateSeededRandom(43);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["useBatchNorm"] = new ParameterRange { Type = ParameterType.Boolean }
        };

        bool sawTrue = false;
        bool sawFalse = false;

        for (int trial = 0; trial < 100; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            bool value = (bool)sample["useBatchNorm"];

            if (value) sawTrue = true;
            else sawFalse = true;
        }

        Assert.True(sawTrue, "Boolean sampler should produce true at least once in 100 trials");
        Assert.True(sawFalse, "Boolean sampler should produce false at least once in 100 trials");
    }

    [Fact]
    public void ParameterSampler_Categorical_OnlyFromAllowedValues()
    {
        var random = RandomHelper.CreateSeededRandom(59);
        var allowed = new List<object> { "relu", "gelu", "silu", "tanh" };
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["activation"] = new ParameterRange
            {
                Type = ParameterType.Categorical,
                CategoricalValues = allowed
            }
        };

        var seen = new HashSet<string>();

        for (int trial = 0; trial < 100; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            string value = (string)sample["activation"];

            Assert.Contains(value, allowed.Cast<string>());
            seen.Add(value);
        }

        // Should have seen at least 3 of 4 categories in 100 trials
        Assert.True(seen.Count >= 3, $"Only saw {seen.Count} categories in 100 trials");
    }

    [Fact]
    public void ParameterSampler_CategoricalWithNullOrEmptyList_FallsBackToDefault()
    {
        var random = RandomHelper.CreateSeededRandom(67);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["optimizer"] = new ParameterRange
            {
                Type = ParameterType.Categorical,
                CategoricalValues = null,
                DefaultValue = "adam"
            }
        };

        var sample = AutoMLParameterSampler.Sample(random, searchSpace);
        Assert.Equal("adam", sample["optimizer"]);
    }

    [Fact]
    public void ParameterSampler_IntegerWithStep_QuantizedCorrectly()
    {
        var random = RandomHelper.CreateSeededRandom(71);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["batchSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 16,
                MaxValue = 128,
                Step = 16
            }
        };

        for (int trial = 0; trial < 100; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            int value = (int)sample["batchSize"];

            Assert.True(value >= 16 && value <= 128);
            Assert.Equal(0, (value - 16) % 16);  // Must be quantized to step
        }
    }

    [Fact]
    public void ParameterSampler_FloatWithStep_QuantizedCorrectly()
    {
        var random = RandomHelper.CreateSeededRandom(73);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["dropout"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 0.5,
                Step = 0.1
            }
        };

        for (int trial = 0; trial < 100; trial++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            double value = (double)sample["dropout"];

            Assert.True(value >= -Tolerance && value <= 0.5 + Tolerance);

            // Value should be approximately on a 0.1 grid
            double remainder = (value - 0.0) / 0.1;
            Assert.True(Math.Abs(remainder - Math.Round(remainder)) < 0.01,
                $"dropout={value} not on 0.1 step grid");
        }
    }

    [Fact]
    public void ParameterSampler_MultipleParameters_AllPresent()
    {
        var random = RandomHelper.CreateSeededRandom(79);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["lr"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.001, MaxValue = 0.1 },
            ["epochs"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 10, MaxValue = 100 },
            ["useBN"] = new ParameterRange { Type = ParameterType.Boolean },
            ["act"] = new ParameterRange
            {
                Type = ParameterType.Categorical,
                CategoricalValues = new List<object> { "relu", "gelu" }
            }
        };

        var sample = AutoMLParameterSampler.Sample(random, searchSpace);

        Assert.True(sample.ContainsKey("lr"));
        Assert.True(sample.ContainsKey("epochs"));
        Assert.True(sample.ContainsKey("useBN"));
        Assert.True(sample.ContainsKey("act"));
        Assert.IsType<double>(sample["lr"]);
        Assert.IsType<int>(sample["epochs"]);
        Assert.IsType<bool>(sample["useBN"]);
        Assert.IsType<string>(sample["act"]);
    }

    [Fact]
    public void ParameterSampler_Deterministic_SameSeedSameResult()
    {
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["lr"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.001, MaxValue = 0.1 },
            ["epochs"] = new ParameterRange { Type = ParameterType.Integer, MinValue = 10, MaxValue = 100 },
        };

        var r1 = RandomHelper.CreateSeededRandom(42);
        var r2 = RandomHelper.CreateSeededRandom(42);

        var s1 = AutoMLParameterSampler.Sample(r1, searchSpace);
        var s2 = AutoMLParameterSampler.Sample(r2, searchSpace);

        Assert.Equal((double)s1["lr"], (double)s2["lr"], Tolerance);
        Assert.Equal((int)s1["epochs"], (int)s2["epochs"]);
    }

    [Fact]
    public void ParameterSampler_NullRandom_Throws()
    {
        var searchSpace = new Dictionary<string, ParameterRange>();

        Assert.Throws<ArgumentNullException>(() =>
            AutoMLParameterSampler.Sample(null!, searchSpace));
    }

    [Fact]
    public void ParameterSampler_NullSearchSpace_Throws()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        Assert.Throws<ArgumentNullException>(() =>
            AutoMLParameterSampler.Sample(random, null!));
    }

    [Fact]
    public void ParameterSampler_EqualMinMax_ReturnsSingleValue()
    {
        var random = RandomHelper.CreateSeededRandom(83);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["fixed"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 42,
                MaxValue = 42
            }
        };

        var sample = AutoMLParameterSampler.Sample(random, searchSpace);
        Assert.Equal(42, (int)sample["fixed"]);
    }

    [Fact]
    public void ParameterSampler_SwappedMinMax_StillWorks()
    {
        var random = RandomHelper.CreateSeededRandom(89);
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["value"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 100,  // min > max
                MaxValue = 10
            }
        };

        // Should swap internally and not throw
        var sample = AutoMLParameterSampler.Sample(random, searchSpace);
        int value = (int)sample["value"];
        Assert.True(value >= 10 && value <= 100);
    }

    #endregion

    #region ParameterRange Tests

    [Fact]
    public void ParameterRange_Clone_DeepCopy()
    {
        var original = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = 0.001,
            MaxValue = 0.1,
            Step = 0.01,
            UseLogScale = true,
            DefaultValue = 0.01,
            CategoricalValues = new List<object> { "a", "b", "c" }
        };

        var clone = (ParameterRange)original.Clone();

        // Values should match
        Assert.Equal(ParameterType.Float, clone.Type);
        Assert.Equal(0.001, clone.MinValue);
        Assert.Equal(0.1, clone.MaxValue);
        Assert.Equal(0.01, clone.Step);
        Assert.True(clone.UseLogScale);
        Assert.Equal(0.01, clone.DefaultValue);
        Assert.Equal(3, clone.CategoricalValues?.Count);

        // Modifying clone should not affect original
        clone.CategoricalValues?.Add("d");
        Assert.Equal(3, original.CategoricalValues?.Count);
    }

    [Fact]
    public void ParameterRange_Clone_NullCategoricalValues_HandledGracefully()
    {
        var original = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1,
            MaxValue = 10,
            CategoricalValues = null
        };

        var clone = (ParameterRange)original.Clone();

        Assert.Null(clone.CategoricalValues);
        Assert.Equal(ParameterType.Integer, clone.Type);
    }

    #endregion

    #region TrialResult Tests

    [Fact]
    public void TrialResult_DefaultValues()
    {
        var result = new TrialResult();

        Assert.Equal(0, result.TrialId);
        Assert.Equal(0.0, result.Score);
        Assert.True(result.Success);
        Assert.Null(result.ErrorMessage);
        Assert.Null(result.CandidateModelType);
        Assert.NotNull(result.Parameters);
        Assert.Empty(result.Parameters);
    }

    [Fact]
    public void TrialResult_Clone_DeepCopy()
    {
        var original = new TrialResult
        {
            TrialId = 5,
            CandidateModelType = ModelType.NeuralNetwork,
            Score = 0.95,
            Duration = TimeSpan.FromSeconds(30),
            Timestamp = new DateTime(2024, 6, 1, 12, 0, 0, DateTimeKind.Utc),
            Success = true,
            ErrorMessage = null,
            Parameters = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                { "lr", 0.01 },
                { "epochs", 100 },
                { "layers", new int[] { 64, 32, 16 } }
            },
            Metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                { "gpu", "A100" }
            }
        };

        var clone = original.Clone();

        Assert.Equal(5, clone.TrialId);
        Assert.Equal(ModelType.NeuralNetwork, clone.CandidateModelType);
        Assert.Equal(0.95, clone.Score);
        Assert.Equal(TimeSpan.FromSeconds(30), clone.Duration);
        Assert.True(clone.Success);
        Assert.Equal(0.01, clone.Parameters["lr"]);
        Assert.Equal(100, clone.Parameters["epochs"]);
        Assert.NotNull(clone.Metadata);
        Assert.Equal("A100", clone.Metadata["gpu"]);

        // Deep copy: modifying clone should not affect original
        clone.Parameters["lr"] = 0.1;
        Assert.Equal(0.01, original.Parameters["lr"]);
    }

    [Fact]
    public void TrialResult_CloneRedacted_HidesParameters()
    {
        var original = new TrialResult
        {
            TrialId = 3,
            Score = 0.88,
            Parameters = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                { "lr", 0.01 },
                { "batchSize", 32 }
            },
            Metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                { "internal_config", "secret" }
            }
        };

        var redacted = original.CloneRedacted();

        Assert.Equal(3, redacted.TrialId);
        Assert.Equal(0.88, redacted.Score);

        // Parameters should be empty (redacted)
        Assert.Empty(redacted.Parameters);

        // Metadata should be null (redacted)
        Assert.Null(redacted.Metadata);
    }

    [Fact]
    public void TrialResult_Clone_ArrayParameter_DeepCopied()
    {
        var original = new TrialResult
        {
            Parameters = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                { "hiddenSizes", new int[] { 128, 64, 32 } }
            }
        };

        var clone = original.Clone();

        var originalArray = (int[])original.Parameters["hiddenSizes"];
        var cloneArray = (int[])clone.Parameters["hiddenSizes"];

        Assert.Equal(originalArray.Length, cloneArray.Length);
        for (int i = 0; i < originalArray.Length; i++)
        {
            Assert.Equal(originalArray[i], cloneArray[i]);
        }

        // Modifying clone should not affect original
        cloneArray[0] = 256;
        Assert.Equal(128, originalArray[0]);
    }

    [Fact]
    public void TrialResult_FailedTrial_PropertiesCorrect()
    {
        var result = new TrialResult
        {
            TrialId = 99,
            Score = 0.0,
            Success = false,
            ErrorMessage = "Out of memory during training"
        };

        Assert.False(result.Success);
        Assert.Equal("Out of memory during training", result.ErrorMessage);
        Assert.Equal(0.0, result.Score);
    }

    #endregion

    #region Architecture Tests

    [Fact]
    public void Architecture_AddOperation_UpdatesNodeCount()
    {
        var arch = new Architecture<double>();

        arch.AddOperation(1, 0, "conv3x3");
        Assert.Equal(2, arch.NodeCount);

        arch.AddOperation(3, 1, "maxpool3x3");
        Assert.Equal(4, arch.NodeCount);

        arch.AddOperation(2, 0, "identity");
        Assert.Equal(4, arch.NodeCount); // no change, already 4
    }

    [Fact]
    public void Architecture_Operations_TrackCorrectly()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 1, "conv5x5");
        arch.AddOperation(3, 2, "identity");

        Assert.Equal(3, arch.Operations.Count);
        Assert.Equal("conv3x3", arch.Operations[0].Operation);
        Assert.Equal(0, arch.Operations[0].FromNode);
        Assert.Equal(1, arch.Operations[0].ToNode);
    }

    [Fact]
    public void Architecture_GetDescription_FormatsCorrectly()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 1, "avgpool3x3");

        string desc = arch.GetDescription();

        Assert.Contains("3 nodes", desc); // nodes 0, 1, 2
        Assert.Contains("conv3x3", desc);
        Assert.Contains("avgpool3x3", desc);
    }

    [Fact]
    public void Architecture_NodeChannels_CanBeSet()
    {
        var arch = new Architecture<double>();
        arch.AddOperation(1, 0, "conv3x3");
        arch.NodeChannels[0] = 3;
        arch.NodeChannels[1] = 16;

        Assert.Equal(3, arch.NodeChannels[0]);
        Assert.Equal(16, arch.NodeChannels[1]);
    }

    [Fact]
    public void Architecture_EmptyOperations_HasZeroNodes()
    {
        var arch = new Architecture<double>();

        Assert.Equal(0, arch.NodeCount);
        Assert.Empty(arch.Operations);
    }

    #endregion

    #region SearchSpaceBase Tests

    [Fact]
    public void SearchSpaceBase_DefaultOperations_HasStandardSet()
    {
        var space = new SearchSpaceBase<double>();

        Assert.Contains("identity", space.Operations);
        Assert.Contains("conv3x3", space.Operations);
        Assert.Contains("conv5x5", space.Operations);
        Assert.Contains("maxpool3x3", space.Operations);
        Assert.Contains("avgpool3x3", space.Operations);
        Assert.Equal(5, space.Operations.Count);
    }

    [Fact]
    public void SearchSpaceBase_DefaultProperties()
    {
        var space = new SearchSpaceBase<double>();

        Assert.Equal(8, space.MaxNodes);
        Assert.Equal(1, space.InputChannels);
        Assert.Equal(1, space.OutputChannels);
    }

    [Fact]
    public void SearchSpaceBase_CustomOperations_Settable()
    {
        var space = new SearchSpaceBase<double>
        {
            Operations = new List<string> { "depthwise_conv", "dilated_conv", "skip" },
            MaxNodes = 16,
            InputChannels = 3,
            OutputChannels = 64
        };

        Assert.Equal(3, space.Operations.Count);
        Assert.Contains("depthwise_conv", space.Operations);
        Assert.Equal(16, space.MaxNodes);
        Assert.Equal(3, space.InputChannels);
        Assert.Equal(64, space.OutputChannels);
    }

    #endregion

    #region AutoMLBudgetDefaults Tests

    [Fact]
    public void BudgetDefaults_CI_Preset()
    {
        var (timeLimit, trialLimit) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.CI);

        Assert.Equal(TimeSpan.FromMinutes(5), timeLimit);
        Assert.Equal(10, trialLimit);
    }

    [Fact]
    public void BudgetDefaults_Fast_Preset()
    {
        var (timeLimit, trialLimit) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.Fast);

        Assert.Equal(TimeSpan.FromMinutes(15), timeLimit);
        Assert.Equal(30, trialLimit);
    }

    [Fact]
    public void BudgetDefaults_Standard_Preset()
    {
        var (timeLimit, trialLimit) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.Standard);

        Assert.Equal(TimeSpan.FromMinutes(30), timeLimit);
        Assert.Equal(100, trialLimit);
    }

    [Fact]
    public void BudgetDefaults_Thorough_Preset()
    {
        var (timeLimit, trialLimit) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.Thorough);

        Assert.Equal(TimeSpan.FromHours(2), timeLimit);
        Assert.Equal(300, trialLimit);
    }

    [Fact]
    public void BudgetDefaults_PresetsHaveIncreasingLimits()
    {
        var (ciTime, ciTrials) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.CI);
        var (fastTime, fastTrials) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.Fast);
        var (stdTime, stdTrials) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.Standard);
        var (thorTime, thorTrials) = AutoMLBudgetDefaults.Resolve(AutoMLBudgetPreset.Thorough);

        Assert.True(ciTime < fastTime);
        Assert.True(fastTime < stdTime);
        Assert.True(stdTime < thorTime);

        Assert.True(ciTrials < fastTrials);
        Assert.True(fastTrials < stdTrials);
        Assert.True(stdTrials < thorTrials);
    }

    #endregion

    #region CompressionOptimizer Tests

    [Fact]
    public void CompressionOptimizer_ConstructsWithDefaults()
    {
        var optimizer = new CompressionOptimizer<double>();

        Assert.NotNull(optimizer.TrialHistory);
        Assert.Empty(optimizer.TrialHistory);
        Assert.Null(optimizer.BestTrial);
    }

    [Fact]
    public void CompressionOptimizer_ConstructsWithOptions()
    {
        var options = new CompressionOptimizerOptions
        {
            MaxTrials = 5,
            RandomSeed = 42
        };

        var optimizer = new CompressionOptimizer<double>(options);

        Assert.NotNull(optimizer);
        Assert.Empty(optimizer.TrialHistory);
    }

    [Fact]
    public void CompressionOptimizer_Optimize_RunsTrials()
    {
        var options = new CompressionOptimizerOptions
        {
            MaxTrials = 3,
            RandomSeed = 42,
        };

        var optimizer = new CompressionOptimizer<double>(options);
        var weights = new Vector<double>(Enumerable.Range(0, 100).Select(i => (double)i).ToArray());

        // Evaluator that returns a fixed accuracy
        Func<Vector<double>, double> evaluator = w => 0.9;

        var best = optimizer.Optimize(weights, evaluator);

        Assert.NotNull(best);
        Assert.True(optimizer.TrialHistory.Count > 0);
        Assert.NotNull(optimizer.BestTrial);
    }

    [Fact]
    public void CompressionOptimizer_Deterministic_SameSeedSameResult()
    {
        var weights = new Vector<double>(Enumerable.Range(0, 50).Select(i => (double)i).ToArray());
        Func<Vector<double>, double> evaluator = w => 0.85;

        var opt1 = new CompressionOptimizer<double>(new CompressionOptimizerOptions { MaxTrials = 3, RandomSeed = 42 });
        var opt2 = new CompressionOptimizer<double>(new CompressionOptimizerOptions { MaxTrials = 3, RandomSeed = 42 });

        var best1 = opt1.Optimize(weights, evaluator);
        var best2 = opt2.Optimize(weights, evaluator);

        Assert.Equal(opt1.TrialHistory.Count, opt2.TrialHistory.Count);
    }

    #endregion

    #region End-to-End AutoML Search Space Tests

    [Fact]
    public void EndToEnd_DefineSearchSpace_SampleAndRecord()
    {
        // Define a realistic search space for a neural network
        var searchSpace = new Dictionary<string, ParameterRange>
        {
            ["learningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 1e-5,
                MaxValue = 1e-1,
                UseLogScale = true
            },
            ["numLayers"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 8
            },
            ["hiddenSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 32,
                MaxValue = 512,
                Step = 32
            },
            ["activation"] = new ParameterRange
            {
                Type = ParameterType.Categorical,
                CategoricalValues = new List<object> { "relu", "gelu", "swish" }
            },
            ["dropout"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 0.5,
                Step = 0.05
            },
            ["useBatchNorm"] = new ParameterRange
            {
                Type = ParameterType.Boolean
            }
        };

        var random = RandomHelper.CreateSeededRandom(42);
        var trials = new List<TrialResult>();

        for (int i = 0; i < 10; i++)
        {
            var sample = AutoMLParameterSampler.Sample(random, searchSpace);
            double fakeScore = random.NextDouble(); // simulate model evaluation

            trials.Add(new TrialResult
            {
                TrialId = i,
                Parameters = sample,
                Score = fakeScore,
                Duration = TimeSpan.FromSeconds(random.Next(1, 60)),
                Timestamp = DateTime.UtcNow,
                Success = true
            });
        }

        Assert.Equal(10, trials.Count);
        Assert.All(trials, t => Assert.True(t.Success));

        // Find best trial
        var best = trials.OrderByDescending(t => t.Score).First();
        Assert.True(best.Score > 0);
        Assert.True(best.Parameters.ContainsKey("learningRate"));
        Assert.True(best.Parameters.ContainsKey("activation"));

        // Redacted clone should hide parameters
        var redacted = best.CloneRedacted();
        Assert.Equal(best.TrialId, redacted.TrialId);
        Assert.Equal(best.Score, redacted.Score);
        Assert.Empty(redacted.Parameters);
    }

    [Fact]
    public void Architecture_BuildAndDescribe_FullPipeline()
    {
        var arch = new Architecture<double>();

        // Build a simple 4-node architecture
        arch.AddOperation(1, 0, "conv3x3");
        arch.AddOperation(2, 0, "identity"); // skip connection
        arch.AddOperation(3, 1, "conv5x5");
        arch.AddOperation(3, 2, "avgpool3x3");

        arch.NodeChannels[0] = 3;   // input RGB
        arch.NodeChannels[1] = 16;
        arch.NodeChannels[2] = 3;   // skip retains channels
        arch.NodeChannels[3] = 32;

        Assert.Equal(4, arch.NodeCount);
        Assert.Equal(4, arch.Operations.Count);

        string description = arch.GetDescription();
        Assert.Contains("4 nodes", description);
        Assert.Contains("conv3x3", description);
        Assert.Contains("identity", description);
    }

    #endregion
}
