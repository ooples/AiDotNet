using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Deployment.Techniques;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AiDotNetTests.UnitTests.Deployment.Techniques
{
    /// <summary>
    /// Unit tests for ModelQuantizer class.
    /// Tests the duplicate removal fixes for BUG-005.
    /// </summary>
    [TestClass]
    public class ModelQuantizerTests
    {
        private ModelQuantizer<double, double[], double[]> _quantizer = default!;
        private MockFullModel _mockModel = default!;

        [TestInitialize]
        public void Setup()
        {
            var config = new QuantizationConfig
            {
                DefaultStrategy = "int8",
                ValidateAccuracy = false, // Disable validation for faster tests
                CalibrationBatches = 5
            };
            _quantizer = new ModelQuantizer<double, double[], double[]>(config);
            _mockModel = new MockFullModel();
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public void Constructor_WithNullConfig_UsesDefaultConfig()
        {
            // Arrange & Act
            var quantizer = new ModelQuantizer<double, double[], double[]>(null);

            // Assert
            Assert.IsNotNull(quantizer);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public void Constructor_WithConfig_InitializesStrategies()
        {
            // Arrange
            var config = new QuantizationConfig { DefaultStrategy = "int16" };

            // Act
            var quantizer = new ModelQuantizer<double, double[], double[]>(config);

            // Assert
            Assert.IsNotNull(quantizer);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public void Quantize_SynchronousMethod_ReturnsQuantizedModel()
        {
            // Arrange
            var model = _mockModel;

            // Act
            var result = _quantizer.Quantize(model, "int8");

            // Assert
            Assert.IsNotNull(result);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task QuantizeModelAsync_WithValidStrategy_ReturnsQuantizedModel()
        {
            // Arrange
            var model = _mockModel;

            // Act
            var result = await _quantizer.QuantizeModelAsync(model, "int8");

            // Assert
            Assert.IsNotNull(result);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        [ExpectedException(typeof(ArgumentException))]
        public async Task QuantizeModelAsync_WithInvalidStrategy_ThrowsArgumentException()
        {
            // Arrange
            var model = _mockModel;

            // Act
            await _quantizer.QuantizeModelAsync(model, "invalid_strategy");

            // Assert - expects exception
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public void AnalyzeModel_ReturnsAnalysisWithRecommendations()
        {
            // Arrange
            var model = _mockModel;

            // Act
            var analysis = _quantizer.AnalyzeModel(model);

            // Assert
            Assert.IsNotNull(analysis);
            Assert.IsNotNull(analysis.SupportedStrategies);
            Assert.IsTrue(analysis.SupportedStrategies.Count > 0);
            Assert.IsNotNull(analysis.RecommendedStrategy);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public void AnalyzeModel_OrdersStrategiesByCompressionRatio()
        {
            // Arrange
            var model = _mockModel;

            // Act
            var analysis = _quantizer.AnalyzeModel(model);

            // Assert
            Assert.IsNotNull(analysis.SupportedStrategies);
            for (int i = 0; i < analysis.SupportedStrategies.Count - 1; i++)
            {
                Assert.IsTrue(
                    analysis.SupportedStrategies[i].ExpectedCompressionRatio >=
                    analysis.SupportedStrategies[i + 1].ExpectedCompressionRatio);
            }
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task LayerWiseQuantizeAsync_WithNonNeuralNetworkModel_ThrowsArgumentException()
        {
            // Arrange
            var model = _mockModel;
            var layerStrategies = new Dictionary<string, string> { { "layer1", "int8" } };

            // Act & Assert
            await Assert.ThrowsExceptionAsync<ArgumentException>(
                async () => await _quantizer.LayerWiseQuantizeAsync(model, layerStrategies));
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task PostTrainingOptimizationAsync_ReturnsOptimizedModel()
        {
            // Arrange
            var model = _mockModel;
            var options = new OptimizationOptions
            {
                Strategy = "int8",
                EnableFineTuning = false,
                EnableGraphOptimization = false,
                TargetHardware = null
            };

            // Act
            var result = await _quantizer.PostTrainingOptimizationAsync(model, options);

            // Assert
            Assert.IsNotNull(result);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task PostTrainingOptimizationAsync_WithFineTuning_ReturnsOptimizedModel()
        {
            // Arrange
            var model = _mockModel;
            var options = new OptimizationOptions
            {
                Strategy = "int8",
                EnableFineTuning = true,
                EnableGraphOptimization = false,
                TargetHardware = null
            };

            // Act
            var result = await _quantizer.PostTrainingOptimizationAsync(model, options);

            // Assert
            Assert.IsNotNull(result);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task PostTrainingOptimizationAsync_WithHardwareOptimization_ReturnsOptimizedModel()
        {
            // Arrange
            var model = _mockModel;
            var options = new OptimizationOptions
            {
                Strategy = "int8",
                EnableFineTuning = false,
                EnableGraphOptimization = false,
                TargetHardware = "CPU"
            };

            // Act
            var result = await _quantizer.PostTrainingOptimizationAsync(model, options);

            // Assert
            Assert.IsNotNull(result);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task PostTrainingOptimizationAsync_WithGraphOptimization_ReturnsOptimizedModel()
        {
            // Arrange
            var model = _mockModel;
            var options = new OptimizationOptions
            {
                Strategy = "int8",
                EnableFineTuning = false,
                EnableGraphOptimization = true,
                TargetHardware = null
            };

            // Act
            var result = await _quantizer.PostTrainingOptimizationAsync(model, options);

            // Assert
            Assert.IsNotNull(result);
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("DuplicateRemoval")]
        public async Task QuantizeModelAsync_MultipleStrategies_AllSucceed()
        {
            // Arrange
            var model = _mockModel;
            var strategies = new[] { "int8", "int16", "dynamic", "qat", "mixed", "binary", "ternary" };

            // Act & Assert
            foreach (var strategy in strategies)
            {
                var result = await _quantizer.QuantizeModelAsync(model, strategy);
                Assert.IsNotNull(result, $"Strategy {strategy} failed");
            }
        }

        [TestMethod]
        [TestCategory("ModelQuantizer")]
        [TestCategory("CalibrationData")]
        public async Task CollectCalibrationData_NoDuplicateVariables_ExecutesSuccessfully()
        {
            // This test verifies that the duplicate variable declarations have been removed
            // by ensuring the quantization process completes without compilation errors

            // Arrange
            var model = _mockModel;

            // Act
            var result = await _quantizer.QuantizeModelAsync(model, "int8");

            // Assert
            Assert.IsNotNull(result);
        }
    }

    #region Mock Classes

    /// <summary>
    /// Mock implementation of IFullModel for testing purposes.
    /// </summary>
    internal class MockFullModel : IFullModel<double, double[], double[]>
    {
        public double[] Predict(double[] input)
        {
            return new double[] { 0.5, 0.5 };
        }

        public void Train(IEnumerable<(double[] Input, double[] Output)> trainingData)
        {
            // Mock implementation
        }

        public double Evaluate(IEnumerable<(double[] Input, double[] Output)> testData)
        {
            return 0.95; // Mock 95% accuracy
        }

        public void Save(string filePath)
        {
            // Mock implementation
        }

        public void Load(string filePath)
        {
            // Mock implementation
        }

        public IModelMetrics GetMetrics()
        {
            return new MockModelMetrics();
        }

        public void SetHyperparameters(Dictionary<string, object> hyperparameters)
        {
            // Mock implementation
        }

        public Dictionary<string, object> GetHyperparameters()
        {
            return new Dictionary<string, object>();
        }
    }

    /// <summary>
    /// Mock implementation of IModelMetrics for testing purposes.
    /// </summary>
    internal class MockModelMetrics : IModelMetrics
    {
        public double Accuracy => 0.95;
        public double Loss => 0.05;
        public Dictionary<string, double> AdditionalMetrics => new Dictionary<string, double>();
    }

    #endregion
}
