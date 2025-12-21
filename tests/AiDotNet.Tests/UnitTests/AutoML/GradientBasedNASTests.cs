using System;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML
{
    public class GradientBasedNASTests
    {
        [Fact]
        public void SuperNet_Constructor_Initializes_Correctly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 4);

            // Assert
            Assert.NotNull(supernet);
            Assert.Equal(ModelType.NeuralNetwork, supernet.Type);
            Assert.True(supernet.ParameterCount > 0);
        }

        [Fact]
        public void SuperNet_Implements_IFullModel_Interface()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace);

            // Act
            var parameters = supernet.GetParameters();
            var metadata = supernet.GetModelMetadata();
            var clone = supernet.Clone();

            // Assert
            Assert.NotNull(parameters);
            Assert.True(parameters.Length > 0);
            Assert.NotNull(metadata);
            Assert.Equal("Differentiable Architecture Search SuperNet", metadata.Description);
            Assert.NotNull(clone);
            Assert.NotSame(supernet, clone);
        }

        [Fact]
        public void SuperNet_Predict_Returns_Valid_Output()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);
            var input = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < input.Shape[1]; i++)
                input[0, i] = 1.0;

            // Act
            var output = supernet.Predict(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
        }

        [Fact]
        public void SuperNet_GetArchitectureParameters_Returns_Valid_Alphas()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 4);

            // Act
            var alphas = supernet.GetArchitectureParameters();

            // Assert
            Assert.NotNull(alphas);
            Assert.Equal(4, alphas.Count); // One alpha matrix per node
            Assert.True(alphas[0].Rows == 1); // First node has 1 previous node (input)
            Assert.True(alphas[1].Rows == 2); // Second node has 2 previous nodes
            Assert.True(alphas[2].Rows == 3); // Third node has 3 previous nodes
            Assert.True(alphas[3].Rows == 4); // Fourth node has 4 previous nodes
        }

        [Fact]
        public void SuperNet_DeriveArchitecture_Returns_Valid_Architecture()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);

            // Act
            var architecture = supernet.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.Operations.Count > 0);
            Assert.True(architecture.NodeCount >= 3);
            var description = architecture.GetDescription();
            Assert.Contains("Architecture with", description);
        }

        [Fact]
        public void SuperNet_ComputeValidationLoss_Returns_Numeric_Value()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 1, 10 });
            var labels = new Tensor<double>(new[] { 1, 10 });
            for (int i = 0; i < 10; i++)
            {
                data[0, i] = 1.0;
                labels[0, i] = 1.0;
            }

            // Act
            var loss = supernet.ComputeValidationLoss(data, labels);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void SuperNet_BackwardArchitecture_Updates_Gradients()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 1, 5 });
            var labels = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                data[0, i] = 1.0;
                labels[0, i] = 1.0;
            }

            // Act
            supernet.BackwardArchitecture(data, labels);
            var gradients = supernet.GetArchitectureGradients();

            // Assert
            Assert.NotNull(gradients);
            Assert.Equal(2, gradients.Count);
        }

        [Fact]
        public void SuperNet_BackwardWeights_Updates_Weight_Gradients()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var data = new Tensor<double>(new[] { 1, 5 });
            var labels = new Tensor<double>(new[] { 1, 5 });
            for (int i = 0; i < 5; i++)
            {
                data[0, i] = 1.0;
                labels[0, i] = 1.0;
            }

            // Force weight initialization by calling predict first
            supernet.Predict(data);

            // Act
            supernet.BackwardWeights(data, labels, supernet.DefaultLossFunction);
            var weightGrads = supernet.GetWeightGradients();

            // Assert
            Assert.NotNull(weightGrads);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_GradientBased_Completes_Successfully()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.GradientBased,
                maxEpochs: 5); // Small number for testing

            var trainData = new Tensor<double>(new[] { 10, 10 });
            var trainLabels = new Tensor<double>(new[] { 10, 10 });
            var valData = new Tensor<double>(new[] { 5, 10 });
            var valLabels = new Tensor<double>(new[] { 5, 10 });

            // Initialize with random data
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    trainData[i, j] = random.NextDouble();
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    trainLabels[i, j] = random.NextDouble();
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 10; j++)
                    valData[i, j] = random.NextDouble();
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 10; j++)
                    valLabels[i, j] = random.NextDouble();

            // Act
            var architecture = await nas.SearchAsync(trainData, trainLabels, valData, valLabels);

            // Assert
            Assert.NotNull(architecture);
            Assert.Equal(AutoMLStatus.Completed, nas.Status);
            Assert.NotNull(nas.BestArchitecture);
            Assert.True(nas.BestScore >= 0.0);
        }

        [Fact]
        public async Task NeuralArchitectureSearch_RandomSearch_Completes_Successfully()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.RandomSearch,
                maxEpochs: 5);

            var trainData = new Tensor<double>(new[] { 10, 10 });
            var trainLabels = new Tensor<double>(new[] { 10, 10 });
            var valData = new Tensor<double>(new[] { 5, 10 });
            var valLabels = new Tensor<double>(new[] { 5, 10 });

            // Initialize with random data
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    trainData[i, j] = random.NextDouble();
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    trainLabels[i, j] = random.NextDouble();
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 10; j++)
                    valData[i, j] = random.NextDouble();
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 10; j++)
                    valLabels[i, j] = random.NextDouble();

            // Act
            var architecture = await nas.SearchAsync(trainData, trainLabels, valData, valLabels);

            // Assert
            Assert.NotNull(architecture);
            Assert.Equal(AutoMLStatus.Completed, nas.Status);
        }

        [Fact]
        public void SearchSpace_Has_Valid_Default_Operations()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();

            // Assert
            Assert.NotNull(searchSpace.Operations);
            Assert.True(searchSpace.Operations.Count > 0);
            Assert.Contains("identity", searchSpace.Operations);
            Assert.True(searchSpace.MaxNodes > 0);
        }

        [Fact]
        public void Architecture_AddOperation_Increases_NodeCount()
        {
            // Arrange
            var arch = new Architecture<double>();

            // Act
            arch.AddOperation(1, 0, "conv3x3");
            arch.AddOperation(2, 0, "maxpool");
            arch.AddOperation(2, 1, "identity");

            // Assert
            Assert.Equal(3, arch.Operations.Count);
            Assert.Equal(3, arch.NodeCount);
        }

        [Fact]
        public void Architecture_GetDescription_Returns_Valid_String()
        {
            // Arrange
            var arch = new Architecture<double>();
            arch.AddOperation(1, 0, "conv3x3");
            arch.AddOperation(2, 1, "maxpool");

            // Act
            var description = arch.GetDescription();

            // Assert
            Assert.NotNull(description);
            Assert.Contains("Architecture with", description);
            Assert.Contains("conv3x3", description);
            Assert.Contains("maxpool", description);
        }

        [Fact]
        public void SuperNet_SetParameters_And_GetParameters_Roundtrip()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var originalParams = supernet.GetParameters();

            // Act
            var newParams = new Vector<double>(originalParams.Length);
            for (int i = 0; i < newParams.Length; i++)
                newParams[i] = i * 0.1;

            supernet.SetParameters(newParams);
            var retrievedParams = supernet.GetParameters();

            // Assert
            Assert.Equal(newParams.Length, retrievedParams.Length);
            for (int i = 0; i < newParams.Length; i++)
            {
                Assert.Equal(newParams[i], retrievedParams[i], precision: 6);
            }
        }

        [Fact]
        public void SuperNet_WithParameters_Creates_New_Instance()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);
            var newParams = new Vector<double>(supernet.GetParameters().Length);

            // Act
            var newSupernet = supernet.WithParameters(newParams);

            // Assert
            Assert.NotNull(newSupernet);
            Assert.NotSame(supernet, newSupernet);
        }

        [Fact]
        public void SuperNet_Clone_Creates_Independent_Copy()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var supernet = new SuperNet<double>(searchSpace, numNodes: 2);

            // Act
            var clone = supernet.Clone();
            var cloneParams = clone.GetParameters();
            var originalParams = supernet.GetParameters();

            // Modify clone
            cloneParams[0] = 999.0;
            ((SuperNet<double>)clone).SetParameters(cloneParams);

            // Assert
            Assert.NotEqual(originalParams[0], clone.GetParameters()[0]);
        }
    }
}
