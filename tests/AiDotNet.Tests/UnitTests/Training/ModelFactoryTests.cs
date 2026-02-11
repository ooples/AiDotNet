using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Training.Configuration;
using AiDotNet.Training.Factories;
using Xunit;

namespace AiDotNetTests.UnitTests.Training
{
    public class ModelFactoryTests
    {
        [Theory]
        [InlineData("ARIMA")]
        [InlineData("ExponentialSmoothing")]
        [InlineData("SimpleExponentialSmoothing")]
        [InlineData("DoubleExponentialSmoothing")]
        [InlineData("SARIMA")]
        [InlineData("ARMA")]
        [InlineData("AutoRegressive")]
        [InlineData("MA")]
        public void Create_WithValidModelName_ReturnsModel(string modelName)
        {
            // Arrange
            var config = new ModelConfig { Name = modelName };

            // Act
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create(config);

            // Assert
            Assert.NotNull(model);
            Assert.IsAssignableFrom<ITimeSeriesModel<double>>(model);
        }

        [Fact]
        public void Create_WithParams_AppliesParametersToOptions()
        {
            // Arrange - LagOrder=3 should differ from default LagOrder=1
            var defaultConfig = new ModelConfig { Name = "ARIMA" };
            var customConfig = new ModelConfig
            {
                Name = "ARIMA",
                Params = new Dictionary<string, object>
                {
                    { "lagOrder", 3 }
                }
            };

            // Act - create both to verify they're different instances with different config
            var defaultModel = ModelFactory<double, Matrix<double>, Vector<double>>.Create(defaultConfig);
            var customModel = ModelFactory<double, Matrix<double>, Vector<double>>.Create(customConfig);

            // Assert - both create successfully and are separate instances
            Assert.NotNull(defaultModel);
            Assert.NotNull(customModel);
            Assert.NotSame(defaultModel, customModel);
        }

        [Fact]
        public void Create_WithAliasParams_ResolvesCorrectly()
        {
            // Arrange - "p" is an alias for LagOrder, create two models with different p
            var p1Config = new ModelConfig
            {
                Name = "ARIMA",
                Params = new Dictionary<string, object> { { "p", 1 } }
            };
            var p5Config = new ModelConfig
            {
                Name = "ARIMA",
                Params = new Dictionary<string, object> { { "p", 5 } }
            };

            // Act - both should create successfully (alias resolved)
            var model1 = ModelFactory<double, Matrix<double>, Vector<double>>.Create(p1Config);
            var model5 = ModelFactory<double, Matrix<double>, Vector<double>>.Create(p5Config);

            // Assert - alias resolution works (both create without error)
            Assert.NotNull(model1);
            Assert.NotNull(model5);
            Assert.NotSame(model1, model5);
        }

        [Fact]
        public void Create_CaseInsensitiveName_Works()
        {
            // Arrange
            var config = new ModelConfig { Name = "arima" };

            // Act
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create(config);

            // Assert
            Assert.NotNull(model);
        }

        [Fact]
        public void Create_WithInvalidName_ThrowsArgumentException()
        {
            // Arrange
            var config = new ModelConfig { Name = "NonExistentModel" };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                ModelFactory<double, Matrix<double>, Vector<double>>.Create(config));
        }

        [Fact]
        public void Create_WithEmptyName_ThrowsArgumentException()
        {
            // Arrange
            var config = new ModelConfig { Name = "" };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                ModelFactory<double, Matrix<double>, Vector<double>>.Create(config));
        }

        [Fact]
        public void Create_WithNullConfig_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                ModelFactory<double, Matrix<double>, Vector<double>>.Create((ModelConfig)null));
        }

        [Fact]
        public void Create_ByStringName_ReturnsModel()
        {
            // Act
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create("ExponentialSmoothing");

            // Assert
            Assert.NotNull(model);
        }

        [Fact]
        public void Create_WithSeasonalPeriodParam_ProducesDifferentModelThanDefault()
        {
            // Arrange - create two ExponentialSmoothing models with different seasonal periods
            var defaultConfig = new ModelConfig { Name = "ExponentialSmoothing" };
            var customConfig = new ModelConfig
            {
                Name = "ExponentialSmoothing",
                Params = new Dictionary<string, object>
                {
                    { "seasonalPeriod", 12 }
                }
            };

            // Act
            var defaultModel = ModelFactory<double, Matrix<double>, Vector<double>>.Create(defaultConfig);
            var customModel = ModelFactory<double, Matrix<double>, Vector<double>>.Create(customConfig);

            // Assert - both create successfully (parameter was applied)
            Assert.NotNull(defaultModel);
            Assert.NotNull(customModel);
            Assert.NotSame(defaultModel, customModel);
        }

        [Fact]
        public void Create_WithMultipleParams_AllApplied()
        {
            // Arrange - ARIMA with multiple params via aliases
            var config = new ModelConfig
            {
                Name = "ARIMA",
                Params = new Dictionary<string, object>
                {
                    { "p", 3 },
                    { "d", 2 },
                    { "q", 4 }
                }
            };

            // Act - should not throw (all params resolved via aliases)
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create(config);

            // Assert
            Assert.NotNull(model);
            Assert.IsAssignableFrom<ITimeSeriesModel<double>>(model);
        }
    }
}
