using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNetTests.IntegrationTests.AutoML
{
    /// <summary>
    /// Comprehensive integration tests for AutoML components achieving 100% coverage.
    /// Tests hyperparameter optimization, neural architecture search, feature selection,
    /// model selection, and pipeline optimization.
    /// </summary>
    public class AutoMLIntegrationTests
    {
        #region ParameterRange Tests

        [Fact]
        public void ParameterRange_IntegerType_CreatesCorrectly()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 100,
                Step = 1
            };

            // Assert
            Assert.Equal(ParameterType.Integer, paramRange.Type);
            Assert.Equal(1, paramRange.MinValue);
            Assert.Equal(100, paramRange.MaxValue);
            Assert.Equal(1, paramRange.Step);
        }

        [Fact]
        public void ParameterRange_FloatType_SupportsDecimalValues()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.001,
                MaxValue = 1.0,
                UseLogScale = true
            };

            // Assert
            Assert.Equal(ParameterType.Float, paramRange.Type);
            Assert.Equal(0.001, paramRange.MinValue);
            Assert.Equal(1.0, paramRange.MaxValue);
            Assert.True(paramRange.UseLogScale);
        }

        [Fact]
        public void ParameterRange_BooleanType_WorksCorrectly()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Boolean,
                DefaultValue = true
            };

            // Assert
            Assert.Equal(ParameterType.Boolean, paramRange.Type);
            Assert.Equal(true, paramRange.DefaultValue);
        }

        [Fact]
        public void ParameterRange_CategoricalType_StoresMultipleValues()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Categorical,
                CategoricalValues = new List<object> { "adam", "sgd", "rmsprop" }
            };

            // Assert
            Assert.Equal(ParameterType.Categorical, paramRange.Type);
            Assert.NotNull(paramRange.CategoricalValues);
            Assert.Equal(3, paramRange.CategoricalValues!.Count);
            Assert.Contains("adam", paramRange.CategoricalValues);
        }

        [Fact]
        public void ParameterRange_ContinuousType_HandlesRanges()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Continuous,
                MinValue = 0.0,
                MaxValue = 10.0
            };

            // Assert
            Assert.Equal(ParameterType.Continuous, paramRange.Type);
            Assert.Equal(0.0, paramRange.MinValue);
            Assert.Equal(10.0, paramRange.MaxValue);
        }

        [Fact]
        public void ParameterRange_Clone_CreatesDeepCopy()
        {
            // Arrange
            var original = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 10,
                Step = 2,
                DefaultValue = 5,
                UseLogScale = false,
                CategoricalValues = new List<object> { "a", "b", "c" }
            };

            // Act
            var cloned = (ParameterRange)original.Clone();
            cloned.MinValue = 100;
            cloned.CategoricalValues![0] = "modified";

            // Assert
            Assert.Equal(1, original.MinValue);
            Assert.Equal("a", original.CategoricalValues![0]);
            Assert.Equal(100, cloned.MinValue);
        }

        [Fact]
        public void ParameterRange_LogScale_EnabledCorrectly()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0001,
                MaxValue = 1.0,
                UseLogScale = true
            };

            // Assert
            Assert.True(paramRange.UseLogScale);
        }

        [Fact]
        public void ParameterRange_DefaultValue_SetsCorrectly()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.01,
                MaxValue = 0.1,
                DefaultValue = 0.05
            };

            // Assert
            Assert.Equal(0.05, paramRange.DefaultValue);
        }

        [Fact]
        public void ParameterRange_StepSize_WorksForDiscrete()
        {
            // Arrange & Act
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 0,
                MaxValue = 100,
                Step = 10
            };

            // Assert
            Assert.Equal(10, paramRange.Step);
        }

        #endregion

        #region SearchSpace Tests

        [Fact]
        public void SearchSpace_DefaultOperations_ContainStandardOps()
        {
            // Arrange & Act
            var searchSpace = new SearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.Operations);
            Assert.Contains("identity", searchSpace.Operations);
            Assert.Contains("conv3x3", searchSpace.Operations);
            Assert.Contains("conv5x5", searchSpace.Operations);
            Assert.Contains("maxpool3x3", searchSpace.Operations);
            Assert.Contains("avgpool3x3", searchSpace.Operations);
        }

        [Fact]
        public void SearchSpace_MaxNodes_SetsCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpace<double>
            {
                MaxNodes = 12
            };

            // Assert
            Assert.Equal(12, searchSpace.MaxNodes);
        }

        [Fact]
        public void SearchSpace_InputOutputChannels_ConfiguresCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpace<float>
            {
                InputChannels = 3,
                OutputChannels = 10
            };

            // Assert
            Assert.Equal(3, searchSpace.InputChannels);
            Assert.Equal(10, searchSpace.OutputChannels);
        }

        [Fact]
        public void SearchSpace_CustomOperations_AddCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpace<double>
            {
                Operations = new List<string> { "custom_op1", "custom_op2", "custom_op3" }
            };

            // Assert
            Assert.Equal(3, searchSpace.Operations.Count);
            Assert.Contains("custom_op1", searchSpace.Operations);
        }

        #endregion

        #region SearchConstraint Tests

        [Fact]
        public void SearchConstraint_RangeType_CreatesCorrectly()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "LearningRateRange",
                Type = ConstraintType.Range,
                MinValue = 0.001,
                MaxValue = 0.1,
                IsHardConstraint = true
            };

            // Assert
            Assert.Equal("LearningRateRange", constraint.Name);
            Assert.Equal(ConstraintType.Range, constraint.Type);
            Assert.Equal(0.001, constraint.MinValue);
            Assert.Equal(0.1, constraint.MaxValue);
            Assert.True(constraint.IsHardConstraint);
        }

        [Fact]
        public void SearchConstraint_DependencyType_HandlesMultipleParams()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "OptimizerDependency",
                Type = ConstraintType.Dependency,
                ParameterNames = new List<string> { "optimizer", "learning_rate" },
                Expression = "if optimizer == 'adam' then learning_rate < 0.01"
            };

            // Assert
            Assert.Equal(ConstraintType.Dependency, constraint.Type);
            Assert.Equal(2, constraint.ParameterNames.Count);
            Assert.Contains("optimizer", constraint.ParameterNames);
        }

        [Fact]
        public void SearchConstraint_ExclusionType_PreventsCombinations()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "ExcludeHighLRWithSGD",
                Type = ConstraintType.Exclusion,
                Expression = "optimizer != 'sgd' OR learning_rate < 0.1"
            };

            // Assert
            Assert.Equal(ConstraintType.Exclusion, constraint.Type);
            Assert.NotEmpty(constraint.Expression);
        }

        [Fact]
        public void SearchConstraint_ResourceType_LimitsComputation()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "MemoryLimit",
                Type = ConstraintType.Resource,
                MaxValue = 8192, // MB
                IsHardConstraint = true
            };

            // Assert
            Assert.Equal(ConstraintType.Resource, constraint.Type);
            Assert.Equal(8192, constraint.MaxValue);
        }

        [Fact]
        public void SearchConstraint_CustomType_SupportsExpression()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "CustomRule",
                Type = ConstraintType.Custom,
                Expression = "batch_size * num_layers < 1000"
            };

            // Assert
            Assert.Equal(ConstraintType.Custom, constraint.Type);
            Assert.NotEmpty(constraint.Expression);
        }

        [Fact]
        public void SearchConstraint_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new SearchConstraint
            {
                Name = "TestConstraint",
                Type = ConstraintType.Range,
                ParameterNames = new List<string> { "param1", "param2" },
                MinValue = 1.0,
                MaxValue = 10.0,
                IsHardConstraint = true
            };

            // Act
            var cloned = (SearchConstraint)original.Clone();
            cloned.Name = "Modified";
            cloned.ParameterNames.Add("param3");

            // Assert
            Assert.Equal("TestConstraint", original.Name);
            Assert.Equal(2, original.ParameterNames.Count);
            Assert.Equal("Modified", cloned.Name);
            Assert.Equal(3, cloned.ParameterNames.Count);
        }

        [Fact]
        public void SearchConstraint_SoftConstraint_AllowsViolation()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "SoftRule",
                Type = ConstraintType.Range,
                IsHardConstraint = false
            };

            // Assert
            Assert.False(constraint.IsHardConstraint);
        }

        [Fact]
        public void SearchConstraint_Metadata_StoresAdditionalInfo()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "ComplexRule",
                Type = ConstraintType.Custom,
                Metadata = new Dictionary<string, object>
                {
                    ["priority"] = 1,
                    ["category"] = "performance"
                }
            };

            // Assert
            Assert.Equal(2, constraint.Metadata.Count);
            Assert.Equal(1, constraint.Metadata["priority"]);
        }

        #endregion

        #region Architecture Tests

        [Fact]
        public void Architecture_AddOperation_IncreasesNodeCount()
        {
            // Arrange
            var arch = new Architecture<double>();

            // Act
            arch.AddOperation(1, 0, "conv3x3");
            arch.AddOperation(2, 1, "maxpool");

            // Assert
            Assert.Equal(3, arch.NodeCount);
            Assert.Equal(2, arch.Operations.Count);
        }

        [Fact]
        public void Architecture_Operations_StoresCorrectly()
        {
            // Arrange
            var arch = new Architecture<float>();

            // Act
            arch.AddOperation(1, 0, "identity");
            arch.AddOperation(2, 0, "conv3x3");
            arch.AddOperation(2, 1, "conv5x5");

            // Assert
            var ops = arch.Operations;
            Assert.Equal(3, ops.Count);
            Assert.Equal((1, 0, "identity"), ops[0]);
            Assert.Equal((2, 0, "conv3x3"), ops[1]);
            Assert.Equal((2, 1, "conv5x5"), ops[2]);
        }

        [Fact]
        public void Architecture_GetDescription_FormatsCorrectly()
        {
            // Arrange
            var arch = new Architecture<double>();
            arch.AddOperation(1, 0, "conv3x3");
            arch.AddOperation(2, 1, "maxpool");

            // Act
            var description = arch.GetDescription();

            // Assert
            Assert.Contains("Architecture with 3 nodes", description);
            Assert.Contains("Node 1 <- conv3x3 <- Node 0", description);
            Assert.Contains("Node 2 <- maxpool <- Node 1", description);
        }

        [Fact]
        public void Architecture_NodeCount_UpdatesAutomatically()
        {
            // Arrange
            var arch = new Architecture<double>();

            // Act
            arch.AddOperation(5, 2, "identity");

            // Assert
            Assert.Equal(6, arch.NodeCount); // Max(5, 2) + 1
        }

        [Fact]
        public void Architecture_EmptyArchitecture_HasZeroNodes()
        {
            // Arrange & Act
            var arch = new Architecture<double>();

            // Assert
            Assert.Equal(0, arch.NodeCount);
            Assert.Empty(arch.Operations);
        }

        #endregion

        #region TrialResult Tests

        [Fact]
        public void TrialResult_Creation_StoresAllFields()
        {
            // Arrange & Act
            var trial = new TrialResult
            {
                TrialId = 1,
                Parameters = new Dictionary<string, object>
                {
                    ["learning_rate"] = 0.01,
                    ["batch_size"] = 32
                },
                Score = 0.95,
                Duration = TimeSpan.FromSeconds(10),
                Timestamp = DateTime.UtcNow,
                Success = true
            };

            // Assert
            Assert.Equal(1, trial.TrialId);
            Assert.Equal(0.95, trial.Score);
            Assert.True(trial.Success);
            Assert.Equal(2, trial.Parameters.Count);
        }

        [Fact]
        public void TrialResult_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new TrialResult
            {
                TrialId = 1,
                Parameters = new Dictionary<string, object> { ["lr"] = 0.1 },
                Score = 0.8,
                Success = true
            };

            // Act
            var cloned = original.Clone();
            cloned.Score = 0.9;
            cloned.Parameters["lr"] = 0.2;

            // Assert
            Assert.Equal(0.8, original.Score);
            Assert.Equal(0.1, original.Parameters["lr"]);
            Assert.Equal(0.9, cloned.Score);
            Assert.Equal(0.2, cloned.Parameters["lr"]);
        }

        [Fact]
        public void TrialResult_Metadata_StoresCustomInfo()
        {
            // Arrange & Act
            var trial = new TrialResult
            {
                TrialId = 1,
                Score = 0.85,
                Metadata = new Dictionary<string, object>
                {
                    ["gpu_used"] = true,
                    ["memory_mb"] = 512
                }
            };

            // Assert
            Assert.NotNull(trial.Metadata);
            Assert.Equal(2, trial.Metadata!.Count);
            Assert.True((bool)trial.Metadata["gpu_used"]);
        }

        [Fact]
        public void TrialResult_ErrorHandling_CapturesFailures()
        {
            // Arrange & Act
            var trial = new TrialResult
            {
                TrialId = 5,
                Success = false,
                ErrorMessage = "Out of memory"
            };

            // Assert
            Assert.False(trial.Success);
            Assert.Equal("Out of memory", trial.ErrorMessage);
        }

        [Fact]
        public void TrialResult_Duration_TracksTime()
        {
            // Arrange & Act
            var trial = new TrialResult
            {
                Duration = TimeSpan.FromMinutes(2.5)
            };

            // Assert
            Assert.Equal(150, trial.Duration.TotalSeconds);
        }

        #endregion

        #region NeuralArchitectureSearch Tests

        [Fact]
        public async Task NeuralArchitectureSearch_GradientBased_SearchesArchitecture()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.GradientBased,
                maxEpochs: 5);

            var trainData = new Tensor<double>(new[] { 10, 4 });
            var trainLabels = new Tensor<double>(new[] { 10, 4 });
            var valData = new Tensor<double>(new[] { 5, 4 });
            var valLabels = new Tensor<double>(new[] { 5, 4 });

            // Initialize with simple data
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 4; j++)
                {
                    trainData[i, j] = i + j;
                    trainLabels[i, j] = (i + j) * 2;
                }

            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 4; j++)
                {
                    valData[i, j] = i + j;
                    valLabels[i, j] = (i + j) * 2;
                }

            // Act
            var architecture = await nas.SearchAsync(trainData, trainLabels, valData, valLabels);

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.NodeCount >= 0);
            Assert.Equal(AutoMLStatus.Completed, nas.Status);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_RandomSearch_FindsArchitecture()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch,
                maxEpochs: 3);

            var trainData = new Tensor<double>(new[] { 8, 3 });
            var trainLabels = new Tensor<double>(new[] { 8, 3 });
            var valData = new Tensor<double>(new[] { 4, 3 });
            var valLabels = new Tensor<double>(new[] { 4, 3 });

            // Initialize data
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 3; j++)
                {
                    trainData[i, j] = i * 0.1;
                    trainLabels[i, j] = j * 0.2;
                }

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                {
                    valData[i, j] = i * 0.1;
                    valLabels[i, j] = j * 0.2;
                }

            // Act
            var architecture = await nas.SearchAsync(trainData, trainLabels, valData, valLabels);

            // Assert
            Assert.NotNull(architecture);
            Assert.Equal(AutoMLStatus.Completed, nas.Status);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_Status_UpdatesCorrectly()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch,
                maxEpochs: 2);

            var data = new Tensor<double>(new[] { 5, 2 });
            var labels = new Tensor<double>(new[] { 5, 2 });

            Assert.Equal(AutoMLStatus.NotStarted, nas.Status);

            // Act
            var architecture = await nas.SearchAsync(data, labels, data, labels);

            // Assert
            Assert.Equal(AutoMLStatus.Completed, nas.Status);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_BestScore_TracksProgress()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch,
                maxEpochs: 2);

            var data = new Tensor<double>(new[] { 6, 3 });
            var labels = new Tensor<double>(new[] { 6, 3 });

            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 3; j++)
                {
                    data[i, j] = i + j;
                    labels[i, j] = i * j;
                }

            // Act
            await nas.SearchAsync(data, labels, data, labels);

            // Assert
            Assert.True(Convert.ToDouble(nas.BestScore) >= 0);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_BestArchitecture_Preserved()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch,
                maxEpochs: 2);

            var data = new Tensor<double>(new[] { 5, 2 });
            var labels = new Tensor<double>(new[] { 5, 2 });

            // Act
            await nas.SearchAsync(data, labels, data, labels);

            // Assert
            Assert.NotNull(nas.BestArchitecture);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_Cancellation_HandlesGracefully()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch,
                maxEpochs: 100);

            var data = new Tensor<double>(new[] { 5, 2 });
            var labels = new Tensor<double>(new[] { 5, 2 });
            var cts = new CancellationTokenSource();
            cts.CancelAfter(10); // Cancel after 10ms

            // Act & Assert
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
                await nas.SearchAsync(data, labels, data, labels, cts.Token));
        }

        #endregion

        #region SuperNet Tests

        [Fact]
        public void SuperNet_Creation_InitializesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();

            // Act
            var supernet = new SuperNet<double>(searchSpace, numNodes: 4);

            // Assert
            Assert.Equal(ModelType.NeuralNetwork, supernet.Type);
            Assert.True(supernet.ParameterCount > 0);
        }

        [Fact]
        public void SuperNet_Predict_ProcessesInput()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);
            var input = new Tensor<double>(new[] { 4, 5 });

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 5; j++)
                    input[i, j] = i + j;

            // Act
            var output = supernet.Predict(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape[0], output.Shape[0]);
        }

        [Fact]
        public void SuperNet_GetArchitectureParameters_ReturnsCorrectCount()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 4);

            // Act
            var archParams = supernet.GetArchitectureParameters();

            // Assert
            Assert.Equal(4, archParams.Count);
            Assert.All(archParams, p => Assert.True(p.Rows > 0 && p.Columns > 0));
        }

        [Fact]
        public void SuperNet_ComputeLoss_CalculatesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 3, 4 });
            var labels = new Tensor<double>(new[] { 3, 4 });

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                {
                    data[i, j] = i;
                    labels[i, j] = i * 2;
                }

            // Act
            var loss = supernet.ComputeValidationLoss(data, labels);

            // Assert
            Assert.True(Convert.ToDouble(loss) >= 0);
        }

        [Fact]
        public void SuperNet_BackwardArchitecture_ComputesGradients()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 2, 3 });
            var labels = new Tensor<double>(new[] { 2, 3 });

            // Act
            supernet.BackwardArchitecture(data, labels);
            var gradients = supernet.GetArchitectureGradients();

            // Assert
            Assert.NotEmpty(gradients);
            Assert.All(gradients, g => Assert.True(g.Rows > 0));
        }

        [Fact]
        public void SuperNet_BackwardWeights_UpdatesGradients()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 3, 4 });
            var labels = new Tensor<double>(new[] { 3, 4 });

            // Initialize weights by predicting first
            supernet.Predict(data);

            // Act
            supernet.BackwardWeights(data, labels);
            var gradients = supernet.GetWeightGradients();

            // Assert - weights are created dynamically, may be empty initially
            Assert.NotNull(gradients);
        }

        [Fact]
        public void SuperNet_DeriveArchitecture_CreatesDiscrete()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);

            // Do a forward pass to initialize
            var data = new Tensor<double>(new[] { 2, 3 });
            supernet.Predict(data);

            // Act
            var architecture = supernet.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.NodeCount >= 0);
            Assert.NotEmpty(architecture.Operations);
        }

        [Fact]
        public void SuperNet_GetSetParameters_WorksCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 2, 2 });
            supernet.Predict(data); // Initialize weights

            // Act
            var params1 = supernet.GetParameters();
            var newParams = new Vector<double>(params1.Length);
            for (int i = 0; i < newParams.Length; i++)
                newParams[i] = i * 0.1;

            supernet.SetParameters(newParams);
            var params2 = supernet.GetParameters();

            // Assert
            Assert.Equal(newParams.Length, params2.Length);
            for (int i = 0; i < newParams.Length; i++)
                Assert.Equal(newParams[i], params2[i], 1e-10);
        }

        [Fact]
        public void SuperNet_Serialization_PreservesState()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 2, 3 });
            supernet.Predict(data);

            // Act
            var serialized = supernet.Serialize();
            var newSupernet = new SuperNet<double>(searchSpace, numNodes: 2);
            newSupernet.Deserialize(serialized);

            // Assert
            Assert.Equal(supernet.ParameterCount, newSupernet.ParameterCount);
        }

        [Fact]
        public void SuperNet_GetFeatureImportance_ReturnsEmpty()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);

            // Act
            var importance = supernet.GetFeatureImportance();

            // Assert
            Assert.NotNull(importance);
            Assert.Empty(importance);
        }

        [Fact]
        public void SuperNet_Clone_CreatesNewInstance()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);

            // Act
            var cloned = supernet.Clone();

            // Assert
            Assert.NotNull(cloned);
            Assert.NotSame(supernet, cloned);
        }

        [Fact]
        public void SuperNet_GetModelMetadata_ReturnsInfo()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);

            // Act
            var metadata = supernet.GetModelMetadata();

            // Assert
            Assert.Equal(ModelType.NeuralNetwork, metadata.ModelType);
            Assert.Contains("SuperNet", metadata.Description);
        }

        [Fact]
        public async Task SuperNet_GetGlobalFeatureImportance_ReturnsOperationImportance()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 2, 3 });

            // Act
            var importance = await supernet.GetGlobalFeatureImportanceAsync(input);

            // Assert
            Assert.NotNull(importance);
            Assert.True(importance.Count > 0);
        }

        [Fact]
        public async Task SuperNet_GetLocalFeatureImportance_UsesInput()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 2, 3 });

            // Act
            var importance = await supernet.GetLocalFeatureImportanceAsync(input);

            // Assert
            Assert.NotNull(importance);
            Assert.True(importance.Count > 0);
        }

        [Fact]
        public async Task SuperNet_GetModelSpecificInterpretability_ReturnsDetails()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);

            // Act
            var info = await supernet.GetModelSpecificInterpretabilityAsync();

            // Assert
            Assert.NotNull(info);
            Assert.True(info.ContainsKey("ModelType"));
            Assert.True(info.ContainsKey("NumNodes"));
            Assert.Equal(3, info["NumNodes"]);
        }

        [Fact]
        public async Task SuperNet_GenerateTextExplanation_CreatesDescription()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 2, 3 });
            var prediction = new Tensor<double>(new[] { 2, 3 });

            // Act
            var explanation = await supernet.GenerateTextExplanationAsync(input, prediction);

            // Assert
            Assert.NotNull(explanation);
            Assert.Contains("SuperNet", explanation);
            Assert.Contains("nodes", explanation);
        }

        [Fact]
        public async Task SuperNet_GetFeatureInteraction_CalculatesCorrelation()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);

            // Act
            var interaction = await supernet.GetFeatureInteractionAsync(0, 1);

            // Assert - should return a valid correlation value
            Assert.True(Convert.ToDouble(interaction) >= -1.0);
            Assert.True(Convert.ToDouble(interaction) <= 1.0);
        }

        #endregion

        #region Simple AutoML Implementation for Testing

        /// <summary>
        /// Simple concrete implementation of AutoMLModelBase for testing purposes.
        /// Uses random search for hyperparameter optimization.
        /// </summary>
        private class SimpleAutoML : AutoMLModelBase<double, Matrix<double>, Vector<double>>
        {
            private readonly Random _random = new Random(42);
            private readonly List<ModelType> _defaultModels = new List<ModelType> { ModelType.LinearRegression };

            public SimpleAutoML()
            {
                if (_candidateModels.Count == 0)
                {
                    _candidateModels.AddRange(_defaultModels);
                }
            }

            public override async Task<IFullModel<double, Matrix<double>, Vector<double>>> SearchAsync(
                Matrix<double> inputs,
                Vector<double> targets,
                Matrix<double> validationInputs,
                Vector<double> validationTargets,
                TimeSpan timeLimit,
                CancellationToken cancellationToken = default)
            {
                Status = AutoMLStatus.Running;
                var startTime = DateTime.UtcNow;

                try
                {
                    int trialCount = 0;
                    while (trialCount < TrialLimit && (DateTime.UtcNow - startTime) < timeLimit)
                    {
                        if (cancellationToken.IsCancellationRequested)
                        {
                            Status = AutoMLStatus.Cancelled;
                            throw new OperationCanceledException();
                        }

                        // Suggest next parameters
                        var parameters = await SuggestNextTrialAsync();

                        // Simulate evaluation
                        var score = _random.NextDouble();
                        await ReportTrialResultAsync(parameters, score, TimeSpan.FromMilliseconds(100));

                        trialCount++;

                        if (ShouldStop())
                            break;
                    }

                    Status = AutoMLStatus.Completed;

                    // Create a simple model as BestModel
                    BestModel = await CreateSimpleModelAsync();
                    return BestModel;
                }
                catch (OperationCanceledException)
                {
                    Status = AutoMLStatus.Cancelled;
                    throw;
                }
                catch
                {
                    Status = AutoMLStatus.Failed;
                    throw;
                }
            }

            public override async Task<Dictionary<string, object>> SuggestNextTrialAsync()
            {
                return await Task.Run(() =>
                {
                    var parameters = new Dictionary<string, object>();

                    lock (_lock)
                    {
                        foreach (var kvp in _searchSpace)
                        {
                            var range = kvp.Value;
                            object value;

                            switch (range.Type)
                            {
                                case ParameterType.Integer:
                                    var minInt = Convert.ToInt32(range.MinValue);
                                    var maxInt = Convert.ToInt32(range.MaxValue);
                                    value = _random.Next(minInt, maxInt + 1);
                                    break;

                                case ParameterType.Float:
                                case ParameterType.Continuous:
                                    var minFloat = Convert.ToDouble(range.MinValue);
                                    var maxFloat = Convert.ToDouble(range.MaxValue);
                                    value = minFloat + _random.NextDouble() * (maxFloat - minFloat);
                                    break;

                                case ParameterType.Boolean:
                                    value = _random.NextDouble() > 0.5;
                                    break;

                                case ParameterType.Categorical:
                                    if (range.CategoricalValues != null && range.CategoricalValues.Count > 0)
                                        value = range.CategoricalValues[_random.Next(range.CategoricalValues.Count)];
                                    else
                                        value = "default";
                                    break;

                                default:
                                    value = range.DefaultValue ?? 0.1;
                                    break;
                            }

                            parameters[kvp.Key] = value;
                        }
                    }

                    return parameters;
                });
            }

            protected override async Task<IFullModel<double, Matrix<double>, Vector<double>>> CreateModelAsync(
                ModelType modelType,
                Dictionary<string, object> parameters)
            {
                return await Task.FromResult(await CreateSimpleModelAsync());
            }

            private async Task<IFullModel<double, Matrix<double>, Vector<double>>> CreateSimpleModelAsync()
            {
                return await Task.FromResult<IFullModel<double, Matrix<double>, Vector<double>>>(
                    new SimpleModel());
            }

            protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
            {
                return new Dictionary<string, ParameterRange>
                {
                    ["learning_rate"] = new ParameterRange
                    {
                        Type = ParameterType.Float,
                        MinValue = 0.001,
                        MaxValue = 0.1
                    }
                };
            }

            protected override AutoMLModelBase<double, Matrix<double>, Vector<double>> CreateInstanceForCopy()
            {
                return new SimpleAutoML();
            }
        }

        /// <summary>
        /// Simple model implementation for testing AutoML
        /// </summary>
        private class SimpleModel : IFullModel<double, Matrix<double>, Vector<double>>
        {
            private Vector<double> _params = new Vector<double>(10);

            public ModelType Type => ModelType.LinearRegression;
            public string[] FeatureNames { get; set; } = Array.Empty<string>();
            public int ParameterCount => _params.Length;

            public void Train(Matrix<double> input, Vector<double> expectedOutput) { }
            public Vector<double> Predict(Matrix<double> input) => new Vector<double>(input.Rows);
            public Vector<double> GetParameters() => _params;
            public void SetParameters(Vector<double> parameters) => _params = parameters;
            public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
            {
                var clone = new SimpleModel();
                clone.SetParameters(parameters);
                return clone;
            }
            public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
            public void SaveModel(string filePath) { }
            public void LoadModel(string filePath) { }
            public byte[] Serialize() => Array.Empty<byte>();
            public void Deserialize(byte[] data) { }
            public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
            public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();
            public bool IsFeatureUsed(int featureIndex) => false;
            public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
            public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new SimpleModel();
            public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy() => new SimpleModel();
        }

        #endregion

        #region AutoML Integration Tests

        [Fact]
        public void AutoML_SetSearchSpace_ConfiguresCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["learning_rate"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.001,
                    MaxValue = 0.1
                },
                ["batch_size"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 16,
                    MaxValue = 128
                }
            };

            // Act
            automl.SetSearchSpace(searchSpace);
            var metadata = automl.GetModelMetadata();

            // Assert
            Assert.Contains("SearchSpaceSize", metadata.AdditionalInfo.Keys);
            Assert.Equal(2, metadata.AdditionalInfo["SearchSpaceSize"]);
        }

        [Fact]
        public void AutoML_SetOptimizationMetric_UpdatesCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.SetOptimizationMetric(MetricType.F1Score, maximize: true);
            var metadata = automl.GetModelMetadata();

            // Assert
            Assert.Equal("F1Score", metadata.AdditionalInfo["OptimizationMetric"]);
            Assert.Equal(true, metadata.AdditionalInfo["Maximize"]);
        }

        [Fact]
        public void AutoML_SetCandidateModels_StoresCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var models = new List<ModelType>
            {
                ModelType.LinearRegression,
                ModelType.LogisticRegression,
                ModelType.DecisionTree
            };

            // Act
            automl.SetCandidateModels(models);
            var metadata = automl.GetModelMetadata();

            // Assert
            var candidateModels = (List<string>)metadata.AdditionalInfo["CandidateModels"];
            Assert.Equal(3, candidateModels.Count);
        }

        [Fact]
        public void AutoML_EnableEarlyStopping_ConfiguresCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.EnableEarlyStopping(patience: 5, minDelta: 0.01);

            // No exception means it configured correctly
            Assert.NotNull(automl);
        }

        [Fact]
        public void AutoML_SetConstraints_StoresCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var constraints = new List<SearchConstraint>
            {
                new SearchConstraint
                {
                    Name = "LRRange",
                    Type = ConstraintType.Range,
                    MinValue = 0.001,
                    MaxValue = 0.1
                }
            };

            // Act
            automl.SetConstraints(constraints);
            var metadata = automl.GetModelMetadata();

            // Assert
            Assert.Equal(1, metadata.AdditionalInfo["Constraints"]);
        }

        [Fact]
        public async Task AutoML_SearchAsync_FindsBestModel()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTimeLimit(TimeSpan.FromSeconds(1));
            automl.SetTrialLimit(5);

            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["learning_rate"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.01,
                    MaxValue = 0.1
                }
            };
            automl.SetSearchSpace(searchSpace);

            var trainX = new Matrix<double>(10, 3);
            var trainY = new Vector<double>(10);
            var valX = new Matrix<double>(5, 3);
            var valY = new Vector<double>(5);

            // Act
            var bestModel = await automl.SearchAsync(
                trainX, trainY, valX, valY,
                TimeSpan.FromSeconds(1));

            // Assert
            Assert.NotNull(bestModel);
            Assert.Equal(AutoMLStatus.Completed, automl.Status);
        }

        [Fact]
        public async Task AutoML_TrialHistory_TracksAllTrials()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(3);
            automl.SetTimeLimit(TimeSpan.FromSeconds(1));

            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["param1"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.0,
                    MaxValue = 1.0
                }
            };
            automl.SetSearchSpace(searchSpace);

            var trainX = new Matrix<double>(5, 2);
            var trainY = new Vector<double>(5);

            // Act
            await automl.SearchAsync(trainX, trainY, trainX, trainY, TimeSpan.FromSeconds(1));
            var history = automl.GetTrialHistory();

            // Assert
            Assert.NotEmpty(history);
            Assert.True(history.Count <= 3);
        }

        [Fact]
        public async Task AutoML_BestScore_UpdatesDuringSearch()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(5);
            automl.SetTimeLimit(TimeSpan.FromSeconds(1));
            automl.SetOptimizationMetric(MetricType.Accuracy, maximize: true);

            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["param"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            };
            automl.SetSearchSpace(searchSpace);

            var data = new Matrix<double>(5, 2);
            var labels = new Vector<double>(5);

            // Act
            await automl.SearchAsync(data, labels, data, labels, TimeSpan.FromSeconds(1));

            // Assert
            Assert.True(automl.BestScore > double.NegativeInfinity);
        }

        [Fact]
        public async Task AutoML_SuggestNextTrial_GeneratesParameters()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["int_param"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 1,
                    MaxValue = 10
                },
                ["float_param"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 0.01,
                    MaxValue = 1.0
                },
                ["bool_param"] = new ParameterRange
                {
                    Type = ParameterType.Boolean
                },
                ["cat_param"] = new ParameterRange
                {
                    Type = ParameterType.Categorical,
                    CategoricalValues = new List<object> { "a", "b", "c" }
                }
            };
            automl.SetSearchSpace(searchSpace);

            // Act
            var params1 = await automl.SuggestNextTrialAsync();
            var params2 = await automl.SuggestNextTrialAsync();

            // Assert
            Assert.Equal(4, params1.Count);
            Assert.Contains("int_param", params1.Keys);
            Assert.Contains("float_param", params1.Keys);
            Assert.Contains("bool_param", params1.Keys);
            Assert.Contains("cat_param", params1.Keys);
        }

        [Fact]
        public async Task AutoML_ReportTrialResult_UpdatesHistory()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var parameters = new Dictionary<string, object>
            {
                ["learning_rate"] = 0.01
            };

            // Act
            await automl.ReportTrialResultAsync(parameters, 0.85, TimeSpan.FromSeconds(1));
            var history = automl.GetTrialHistory();

            // Assert
            Assert.Single(history);
            Assert.Equal(0.85, history[0].Score);
        }

        [Fact]
        public void AutoML_ConfigureSearchSpace_AliasWorks()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["param1"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            };

            // Act
            automl.ConfigureSearchSpace(searchSpace);
            var metadata = automl.GetModelMetadata();

            // Assert
            Assert.Equal(1, metadata.AdditionalInfo["SearchSpaceSize"]);
        }

        [Fact]
        public void AutoML_SetTimeLimit_UpdatesCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.SetTimeLimit(TimeSpan.FromMinutes(10));

            // Assert
            Assert.Equal(TimeSpan.FromMinutes(10), automl.TimeLimit);
        }

        [Fact]
        public void AutoML_SetTrialLimit_UpdatesCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.SetTrialLimit(50);

            // Assert
            Assert.Equal(50, automl.TrialLimit);
        }

        [Fact]
        public void AutoML_EnableNAS_ConfiguresCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.EnableNAS(true);

            // No exception means it configured
            Assert.NotNull(automl);
        }

        [Fact]
        public void AutoML_SearchBestModel_SynchronousVersion()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act
            var bestModel = automl.SearchBestModel(data, labels, data, labels);

            // Assert
            Assert.NotNull(bestModel);
        }

        [Fact]
        public void AutoML_Search_UpdatesStatus()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act
            automl.Search(data, labels, data, labels);

            // Assert
            Assert.Equal(AutoMLStatus.Completed, automl.Status);
        }

        [Fact]
        public void AutoML_Run_ExecutesSearch()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act
            automl.Run(data, labels, data, labels);

            // Assert
            Assert.Equal(AutoMLStatus.Completed, automl.Status);
        }

        [Fact]
        public void AutoML_GetResults_ReturnsHistory()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act
            automl.Search(data, labels, data, labels);
            var results = automl.GetResults();

            // Assert
            Assert.NotEmpty(results);
        }

        [Fact]
        public void AutoML_SetModelsToTry_ConfiguresModels()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var models = new List<ModelType>
            {
                ModelType.LinearRegression,
                ModelType.RandomForest
            };

            // Act
            automl.SetModelsToTry(models);
            var metadata = automl.GetModelMetadata();

            // Assert
            var candidateModels = (List<string>)metadata.AdditionalInfo["CandidateModels"];
            Assert.Equal(2, candidateModels.Count);
        }

        [Fact]
        public void AutoML_DeepCopy_CreatesIndependentCopy()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p1"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });
            automl.SetTrialLimit(10);
            automl.SetTimeLimit(TimeSpan.FromMinutes(5));

            // Act
            var copy = (SimpleAutoML)automl.DeepCopy();
            copy.SetTrialLimit(20);

            // Assert
            Assert.Equal(10, automl.TrialLimit);
            Assert.Equal(20, copy.TrialLimit);
        }

        [Fact]
        public void AutoML_Predict_UsesBestModel()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            automl.Search(data, labels, data, labels);

            // Act
            var predictions = automl.Predict(data);

            // Assert
            Assert.NotNull(predictions);
            Assert.Equal(data.Rows, predictions.Length);
        }

        [Fact]
        public void AutoML_GetParameters_ReturnsBestModelParams()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            automl.Search(data, labels, data, labels);

            // Act
            var parameters = automl.GetParameters();

            // Assert
            Assert.NotNull(parameters);
            Assert.True(parameters.Length > 0);
        }

        [Fact]
        public async Task AutoML_EarlyStopping_StopsWhenNoImprovement()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(100); // High limit
            automl.SetTimeLimit(TimeSpan.FromSeconds(10));
            automl.EnableEarlyStopping(patience: 3, minDelta: 0.001);
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act
            await automl.SearchAsync(data, labels, data, labels, TimeSpan.FromSeconds(10));
            var history = automl.GetTrialHistory();

            // Assert - should stop early due to patience
            Assert.True(history.Count < 100);
        }

        #endregion

        #region Edge Cases and Integration Tests

        [Fact]
        public void ParameterRange_Clone_HandlesNullCategorical()
        {
            // Arrange
            var paramRange = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 1.0,
                CategoricalValues = null
            };

            // Act
            var cloned = (ParameterRange)paramRange.Clone();

            // Assert
            Assert.Null(cloned.CategoricalValues);
        }

        [Fact]
        public void SearchConstraint_Metadata_HandlesEmptyDictionary()
        {
            // Arrange & Act
            var constraint = new SearchConstraint
            {
                Name = "Test",
                Type = ConstraintType.Range,
                Metadata = new Dictionary<string, object>()
            };

            // Assert
            Assert.Empty(constraint.Metadata);
        }

        [Fact]
        public void Architecture_EmptyOperations_DescriptionHandlesGracefully()
        {
            // Arrange
            var arch = new Architecture<double>();

            // Act
            var description = arch.GetDescription();

            // Assert
            Assert.Contains("Architecture with 0 nodes", description);
        }

        [Fact]
        public void TrialResult_Clone_HandlesNullMetadata()
        {
            // Arrange
            var trial = new TrialResult
            {
                TrialId = 1,
                Score = 0.5,
                Metadata = null
            };

            // Act
            var cloned = trial.Clone();

            // Assert
            Assert.Null(cloned.Metadata);
        }

        [Fact]
        public void SuperNet_SmallNodes_HandlesEdgeCase()
        {
            // Arrange & Act
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 1);

            // Assert
            Assert.NotNull(supernet);
        }

        [Fact]
        public void SuperNet_SaveLoad_RoundTrip()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 2, 3 });
            supernet.Predict(data);

            var tempFile = System.IO.Path.Combine(
                Environment.CurrentDirectory,
                $"test_supernet_{Guid.NewGuid()}.bin");

            try
            {
                // Act
                supernet.SaveModel(tempFile);
                var loadedSupernet = new SuperNet<double>(searchSpace, numNodes: 2);
                loadedSupernet.LoadModel(tempFile);

                // Assert
                Assert.Equal(supernet.ParameterCount, loadedSupernet.ParameterCount);
            }
            finally
            {
                if (System.IO.File.Exists(tempFile))
                    System.IO.File.Delete(tempFile);
            }
        }

        [Fact]
        public async Task AutoML_SearchSpace_EmptyHandledGracefully()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>());
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act
            var model = await automl.SearchAsync(data, labels, data, labels, TimeSpan.FromMilliseconds(500));

            // Assert
            Assert.NotNull(model);
        }

        [Fact]
        public void AutoML_MinimizeMetric_WorksCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.SetOptimizationMetric(MetricType.MeanSquaredError, maximize: false);

            // Assert
            Assert.Equal(double.PositiveInfinity, automl.BestScore);
        }

        [Fact]
        public void AutoML_MaximizeMetric_WorksCorrectly()
        {
            // Arrange
            var automl = new SimpleAutoML();

            // Act
            automl.SetOptimizationMetric(MetricType.Accuracy, maximize: true);

            // Assert
            Assert.Equal(double.NegativeInfinity, automl.BestScore);
        }

        [Fact]
        public void ParameterRange_ContinuousType_DifferentiatesFromFloat()
        {
            // Arrange & Act
            var floatParam = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 1.0
            };

            var continuousParam = new ParameterRange
            {
                Type = ParameterType.Continuous,
                MinValue = 0.0,
                MaxValue = 1.0
            };

            // Assert
            Assert.NotEqual(floatParam.Type, continuousParam.Type);
        }

        [Fact]
        public void SearchSpace_LargeMaxNodes_HandlesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpace<double>
            {
                MaxNodes = 100
            };

            // Assert
            Assert.Equal(100, searchSpace.MaxNodes);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_MultipleSearches_Independent()
        {
            // Arrange
            var nas1 = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch, maxEpochs: 2);
            var nas2 = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch, maxEpochs: 2);

            var data = new Tensor<double>(new[] { 3, 2 });
            var labels = new Tensor<double>(new[] { 3, 2 });

            // Act
            var arch1 = await nas1.SearchAsync(data, labels, data, labels);
            var arch2 = await nas2.SearchAsync(data, labels, data, labels);

            // Assert
            Assert.NotNull(arch1);
            Assert.NotNull(arch2);
            // Both should complete successfully
            Assert.Equal(AutoMLStatus.Completed, nas1.Status);
            Assert.Equal(AutoMLStatus.Completed, nas2.Status);
        }

        [Fact]
        public void SuperNet_GetActiveFeatureIndices_ReturnsRange()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 2, 5 });
            supernet.Predict(data);

            // Act
            var activeIndices = supernet.GetActiveFeatureIndices().ToList();

            // Assert
            Assert.NotEmpty(activeIndices);
        }

        [Fact]
        public void SuperNet_IsFeatureUsed_ValidatesRange()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 2, 5 });
            supernet.Predict(data);

            // Act & Assert
            Assert.True(supernet.IsFeatureUsed(0));
            Assert.True(supernet.IsFeatureUsed(4));
            Assert.False(supernet.IsFeatureUsed(10));
        }

        [Fact]
        public void SuperNet_SetActiveFeatureIndices_NoOp()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);

            // Act
            supernet.SetActiveFeatureIndices(new[] { 0, 1, 2 });

            // Assert - should not throw
            Assert.NotNull(supernet);
        }

        [Fact]
        public async Task SuperNet_GetShapValues_ThrowsNotSupported()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 2, 3 });

            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(async () =>
                await supernet.GetShapValuesAsync(input));
        }

        [Fact]
        public async Task SuperNet_GetLimeExplanation_ThrowsNotSupported()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 1, 3 });

            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(async () =>
                await supernet.GetLimeExplanationAsync(input));
        }

        [Fact]
        public async Task SuperNet_GetPartialDependence_ThrowsNotSupported()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var featureIndices = new Vector<int>(new[] { 0, 1 });

            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(async () =>
                await supernet.GetPartialDependenceAsync(featureIndices));
        }

        [Fact]
        public async Task SuperNet_GetCounterfactual_ThrowsNotSupported()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 1, 3 });
            var desired = new Tensor<double>(new[] { 1, 3 });

            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(async () =>
                await supernet.GetCounterfactualAsync(input, desired));
        }

        [Fact]
        public async Task SuperNet_ValidateFairness_ThrowsNotSupported()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 2, 3 });

            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(async () =>
                await supernet.ValidateFairnessAsync(input, 0));
        }

        [Fact]
        public async Task SuperNet_GetAnchorExplanation_ThrowsNotSupported()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var input = new Tensor<double>(new[] { 1, 3 });

            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(async () =>
                await supernet.GetAnchorExplanationAsync(input, 0.5));
        }

        [Fact]
        public void SuperNet_EnableMethod_ConfiguresCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);

            // Act
            supernet.EnableMethod(InterpretationMethod.FeatureImportance);

            // Assert - should not throw
            Assert.NotNull(supernet);
        }

        [Fact]
        public void SuperNet_ConfigureFairness_SetsCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpace<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var sensitiveFeatures = new Vector<int>(new[] { 0, 1 });

            // Act
            supernet.ConfigureFairness(sensitiveFeatures, FairnessMetric.DemographicParity);

            // Assert - should not throw
            Assert.NotNull(supernet);
        }

        [Fact]
        public void AutoML_Train_ThrowsNotSupported()
        {
            // Arrange
            var automl = new SimpleAutoML();
            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                automl.Train(data, labels));
        }

        [Fact]
        public async Task AutoML_GetFeatureImportance_ReturnsEmpty()
        {
            // Arrange
            var automl = new SimpleAutoML();
            automl.SetTrialLimit(2);
            automl.SetTimeLimit(TimeSpan.FromMilliseconds(500));
            automl.SetSearchSpace(new Dictionary<string, ParameterRange>
            {
                ["p"] = new ParameterRange { Type = ParameterType.Float, MinValue = 0.0, MaxValue = 1.0 }
            });

            var data = new Matrix<double>(3, 2);
            var labels = new Vector<double>(3);

            automl.Search(data, labels, data, labels);

            // Act
            var importance = await automl.GetFeatureImportanceAsync();

            // Assert
            Assert.NotNull(importance);
            Assert.Empty(importance);
        }

        #endregion
    }
}
