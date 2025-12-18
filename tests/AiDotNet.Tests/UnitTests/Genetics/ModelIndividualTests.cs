using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Genetics;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Genetics
{
    /// <summary>
    /// Unit tests for ModelIndividual class to verify IFullModel, IFeatureAware, ICloneable, and IParameterizable implementations.
    /// </summary>
    public class ModelIndividualTests
    {
        private class MockModel : IFullModel<double, double[], double[]>
        {
            private Vector<double> _parameters;
            private readonly int _parameterCount;

            public MockModel(int parameterCount = 5)
            {
                _parameterCount = parameterCount;
                _parameters = new Vector<double>(parameterCount);
                for (int i = 0; i < parameterCount; i++)
                {
                    _parameters[i] = i * 0.1;
                }
            }

            public double[] Predict(double[] input)
            {
                return new double[] { input.Sum() * 2.0 };
            }

            public void Train(double[] input, double[] expectedOutput)
            {
                // Mock training - just update first parameter
                if (_parameters.Length > 0)
                {
                    _parameters[0] = _parameters[0] + 0.01;
                }
            }

            public ModelMetadata<double> GetModelMetadata()
            {
                var metadata = new ModelMetadata<double>
                {
                    Name = "MockModel",
                    ModelType = Enums.ModelType.None,
                    FeatureCount = 3,
                    Complexity = 1
                };
                metadata.AdditionalInfo["loss"] = 0.5;
                return metadata;
            }

            public Vector<double> GetParameters()
            {
                return _parameters;
            }

            public void SetParameters(Vector<double> parameters)
            {
                if (parameters == null)
                    throw new ArgumentNullException(nameof(parameters));
                _parameters = new Vector<double>(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    _parameters[i] = parameters[i];
                }
            }

            public int ParameterCount
            {
                get { return _parameters.Length; }
            }

            public IFullModel<double, double[], double[]> WithParameters(Vector<double> parameters)
            {
                var newModel = new MockModel(_parameterCount);
                newModel._parameters = new Vector<double>(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    newModel._parameters[i] = parameters[i];
                }
                return newModel;
            }

            public byte[] Serialize()
            {
                var data = new byte[_parameters.Length * sizeof(double)];
                Buffer.BlockCopy(_parameters.ToArray(), 0, data, 0, data.Length);
                return data;
            }

            public void Deserialize(byte[] data)
            {
                var values = new double[data.Length / sizeof(double)];
                Buffer.BlockCopy(data, 0, values, 0, data.Length);
                _parameters = new Vector<double>(values);
            }

            // ICheckpointableModel implementation
            public void SaveState(Stream stream) { }
            public void LoadState(Stream stream) { }

            public IEnumerable<int> GetActiveFeatureIndices()
            {
                return new[] { 0, 1, 2 };
            }

            public Dictionary<string, double> GetFeatureImportance()
            {
                return new Dictionary<string, double>
                {
                    { "feature_0", 0.5 },
                    { "feature_1", 0.3 },
                    { "feature_2", 0.2 }
                };
            }

            public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
            {
                // Mock implementation - no-op
            }

            public bool IsFeatureUsed(int featureIndex)
            {
                return featureIndex >= 0 && featureIndex < 3;
            }

            public IFullModel<double, double[], double[]> DeepCopy()
            {
                var copy = new MockModel(_parameterCount);
                copy._parameters = new Vector<double>(_parameters.Length);
                for (int i = 0; i < _parameters.Length; i++)
                {
                    copy._parameters[i] = _parameters[i];
                }
                return copy;
            }

            IFullModel<double, double[], double[]> ICloneable<IFullModel<double, double[], double[]>>.Clone()
            {
                return DeepCopy();
            }

            public void SaveModel(string filePath)
            {
                File.WriteAllBytes(filePath, Serialize());
            }

            public void LoadModel(string filePath)
            {
                Deserialize(File.ReadAllBytes(filePath));
            }

            // IGradientComputable implementation
            public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

            public Vector<double> ComputeGradients(double[] input, double[] target, ILossFunction<double>? lossFunction = null)
            {
                return new Vector<double>(ParameterCount);
            }

            public void ApplyGradients(Vector<double> gradients, double learningRate)
            {
                // Mock implementation - simple parameter update
                for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
                {
                    _parameters[i] -= learningRate * gradients[i];
                }
            }

            // IJitCompilable implementation
            public bool SupportsJitCompilation => true;

            public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
            {
                // Create a simple computation graph for the mock model
                var inputShape = new int[] { 1, _parameterCount };
                var inputTensor = new Tensor<double>(inputShape);
                var inputNode = TensorOperations<double>.Variable(inputTensor, "input");
                inputNodes.Add(inputNode);

                // Create parameter node
                var paramTensor = new Tensor<double>(new int[] { _parameterCount }, _parameters);
                var paramNode = TensorOperations<double>.Variable(paramTensor, "parameters");
                inputNodes.Add(paramNode);

                // Compute element-wise multiply and sum
                var mulNode = TensorOperations<double>.ElementwiseMultiply(inputNode, paramNode);
                var outputNode = TensorOperations<double>.Sum(mulNode);
                return outputNode;
            }
        }

        private class ModelParameterGene : ICloneable
        {
            public double Value { get; set; }

            public ModelParameterGene(double value)
            {
                Value = value;
            }

            public object Clone()
            {
                return new ModelParameterGene(Value);
            }
        }

        private IFullModel<double, double[], double[]> CreateMockModel(ICollection<ModelParameterGene> genes)
        {
            var paramCount = genes.Count > 0 ? genes.Count : 5;
            return new MockModel(paramCount);
        }

        [Fact]
        public void Constructor_WithGenes_InitializesCorrectly()
        {
            // Arrange
            var genes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2),
                new ModelParameterGene(0.3)
            };

            // Act
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(genes, CreateMockModel);

            // Assert
            Assert.NotNull(individual);
            Assert.Equal(3, individual.GetGenes().Count);
        }

        [Fact]
        public void Constructor_WithModel_InitializesCorrectly()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2)
            };

            // Act
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Assert
            Assert.NotNull(individual);
            Assert.Equal(2, individual.GetGenes().Count);
        }

        [Fact]
        public void Train_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var input = new double[] { 1.0, 2.0, 3.0 };
            var expectedOutput = new double[] { 6.0 };
            var paramsBefore = individual.GetParameters()[0];

            // Act
            individual.Train(input, expectedOutput);

            // Assert
            var paramsAfter = individual.GetParameters()[0];
            Assert.NotEqual(paramsBefore, paramsAfter); // Parameters should change after training
        }

        [Fact]
        public void GetModelMetadata_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var metadata = individual.GetModelMetadata();

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal("MockModel", metadata.Name);
            Assert.Equal(ModelType.None, metadata.ModelType);
            Assert.Equal(3, metadata.FeatureCount);
            Assert.Equal(1, metadata.Complexity);
        }

        [Fact]
        public void GetActiveFeatureIndices_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var indices = individual.GetActiveFeatureIndices();

            // Assert
            Assert.NotNull(indices);
            Assert.Equal(new[] { 0, 1, 2 }, indices);
        }

        [Fact]
        public void GetFeatureImportance_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var importance = individual.GetFeatureImportance();

            // Assert
            Assert.NotNull(importance);
            Assert.Equal(3, importance.Count);
            Assert.Equal(0.5, importance["feature_0"]);
            Assert.Equal(0.3, importance["feature_1"]);
            Assert.Equal(0.2, importance["feature_2"]);
        }

        [Fact]
        public void SetActiveFeatureIndices_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var newIndices = new[] { 0, 2 };

            // Act & Assert - Should not throw
            individual.SetActiveFeatureIndices(newIndices);
        }

        [Fact]
        public void IsFeatureUsed_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var isUsed0 = individual.IsFeatureUsed(0);
            var isUsed1 = individual.IsFeatureUsed(1);
            var isUsed5 = individual.IsFeatureUsed(5);

            // Assert
            Assert.True(isUsed0);
            Assert.True(isUsed1);
            Assert.False(isUsed5);
        }

        [Fact]
        public void DeepCopy_CreatesIndependentCopy()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2)
            };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            individual.SetFitness(0.95);

            // Act
            var copy = individual.DeepCopy();

            // Assert
            Assert.NotNull(copy);
            Assert.NotSame(individual, copy);

            // Modify original
            var originalParams = individual.GetParameters();
            var newParams = new Vector<double>(originalParams.Length);
            for (int i = 0; i < originalParams.Length; i++)
            {
                newParams[i] = originalParams[i] + 100.0;
            }
            individual.SetParameters(newParams);

            // Copy should remain unchanged
            var copyParams = copy.GetParameters();
            Assert.NotEqual(originalParams[0] + 100.0, copyParams[0], 5);
        }

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2)
            };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var clone = ((ICloneable<IFullModel<double, double[], double[]>>)individual).Clone();

            // Assert
            Assert.NotNull(clone);
            Assert.NotSame(individual, clone);

            // Verify parameters are copied
            var originalParams = individual.GetParameters();
            var cloneParams = clone.GetParameters();
            Assert.Equal(originalParams.Length, cloneParams.Length);
            for (int i = 0; i < originalParams.Length; i++)
            {
                Assert.Equal(originalParams[i], cloneParams[i]);
            }
        }

        [Fact]
        public void SetParameters_UpdatesModelParameters()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var newParams = new Vector<double>(5);
            for (int i = 0; i < 5; i++)
            {
                newParams[i] = i * 2.0;
            }

            // Act
            individual.SetParameters(newParams);

            // Assert
            var updatedParams = individual.GetParameters();
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(i * 2.0, updatedParams[i]);
            }
        }

        [Fact]
        public void ParameterCount_ReturnsCorrectCount()
        {
            // Arrange
            var model = new MockModel(5);
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var count = individual.ParameterCount;

            // Assert
            Assert.Equal(5, count);
        }

        [Fact]
        public void ParameterCount_IsCached()
        {
            // Arrange
            var model = new MockModel(5);
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act
            var count1 = individual.ParameterCount;
            var count2 = individual.ParameterCount;

            // Assert
            Assert.Equal(count1, count2);
            Assert.Equal(5, count1);
        }

        [Fact]
        public void SetParameters_InvalidatesParameterCountCache()
        {
            // Arrange
            var model = new MockModel(5);
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var initialCount = individual.ParameterCount;

            // Act
            var newParams = new Vector<double>(3);
            for (int i = 0; i < 3; i++)
            {
                newParams[i] = i * 1.5;
            }
            individual.SetParameters(newParams);
            var newCount = individual.ParameterCount;

            // Assert
            Assert.Equal(5, initialCount);
            Assert.Equal(3, newCount);
        }

        [Fact]
        public void Predict_DelegatesToInnerModel()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var input = new double[] { 1.0, 2.0, 3.0 };

            // Act
            var output = individual.Predict(input);

            // Assert
            Assert.NotNull(output);
            Assert.Single(output);
            Assert.Equal(12.0, output[0]); // Sum is 6.0, multiplied by 2.0
        }

        [Fact]
        public void SaveAndLoadModel_WorksCorrectly()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var tempFile = Path.Combine(Path.GetTempPath(), $"test_model_{Guid.NewGuid()}.bin");

            try
            {
                // Act
                individual.SaveModel(tempFile);
                var loadedIndividual = new ModelIndividual<double, double[], double[], ModelParameterGene>(new MockModel(), genes, CreateMockModel);
                loadedIndividual.LoadModel(tempFile);

                // Assert
                var originalParams = individual.GetParameters();
                var loadedParams = loadedIndividual.GetParameters();
                Assert.Equal(originalParams.Length, loadedParams.Length);
                for (int i = 0; i < originalParams.Length; i++)
                {
                    Assert.Equal(originalParams[i], loadedParams[i]);
                }
            }
            finally
            {
                if (File.Exists(tempFile))
                {
                    File.Delete(tempFile);
                }
            }
        }

        [Fact]
        public void SaveModel_ThrowsOnNullPath()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => individual.SaveModel(null));
            Assert.Throws<ArgumentException>(() => individual.SaveModel(string.Empty));
            Assert.Throws<ArgumentException>(() => individual.SaveModel("   "));
        }

        [Fact]
        public void LoadModel_ThrowsOnNullPath()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => individual.LoadModel(null));
            Assert.Throws<ArgumentException>(() => individual.LoadModel(string.Empty));
            Assert.Throws<ArgumentException>(() => individual.LoadModel("   "));
        }

        [Fact]
        public void LoadModel_ThrowsOnNonExistentFile()
        {
            // Arrange
            var model = new MockModel();
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(model, genes, CreateMockModel);
            var nonExistentFile = Path.Combine(Path.GetTempPath(), $"nonexistent_{Guid.NewGuid()}.bin");

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => individual.LoadModel(nonExistentFile));
        }

        [Fact]
        public void GetGenes_ReturnsGeneCollection()
        {
            // Arrange
            var genes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2),
                new ModelParameterGene(0.3)
            };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(genes, CreateMockModel);

            // Act
            var retrievedGenes = individual.GetGenes();

            // Assert
            Assert.NotNull(retrievedGenes);
            Assert.Equal(3, retrievedGenes.Count);
        }

        [Fact]
        public void SetGenes_UpdatesGenesAndModel()
        {
            // Arrange
            var initialGenes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2)
            };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(initialGenes, CreateMockModel);

            var newGenes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.5),
                new ModelParameterGene(0.6),
                new ModelParameterGene(0.7)
            };

            // Act
            individual.SetGenes(newGenes);

            // Assert
            var retrievedGenes = individual.GetGenes();
            Assert.Equal(3, retrievedGenes.Count);
        }

        [Fact]
        public void Fitness_CanBeSetAndRetrieved()
        {
            // Arrange
            var genes = new List<ModelParameterGene> { new ModelParameterGene(0.1) };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(genes, CreateMockModel);

            // Act
            individual.SetFitness(0.85);
            var fitness = individual.GetFitness();

            // Assert
            Assert.Equal(0.85, fitness);
        }

        [Fact]
        public void EvolvableClone_CreatesIndependentClone()
        {
            // Arrange
            var genes = new List<ModelParameterGene>
            {
                new ModelParameterGene(0.1),
                new ModelParameterGene(0.2)
            };
            var individual = new ModelIndividual<double, double[], double[], ModelParameterGene>(genes, CreateMockModel);
            individual.SetFitness(0.75);

            // Act
            var clone = individual.Clone();

            // Assert
            Assert.NotNull(clone);
            Assert.NotSame(individual, clone);
            Assert.Equal(0.75, clone.GetFitness());
            Assert.Equal(individual.GetGenes().Count, clone.GetGenes().Count);
        }
    }
}
