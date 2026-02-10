using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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
            // Arrange
            var config = new ModelConfig
            {
                Name = "ARIMA",
                Params = new Dictionary<string, object>
                {
                    { "lagOrder", 3 }
                }
            };

            // Act
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create(config);

            // Assert
            Assert.NotNull(model);
        }

        [Fact]
        public void Create_WithAliasParams_ResolvesCorrectly()
        {
            // Arrange - "p" is an alias for LagOrder
            var config = new ModelConfig
            {
                Name = "ARIMA",
                Params = new Dictionary<string, object>
                {
                    { "p", 2 }
                }
            };

            // Act
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create(config);

            // Assert
            Assert.NotNull(model);
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
        public void Create_WithSeasonalPeriodParam_Works()
        {
            // Arrange
            var config = new ModelConfig
            {
                Name = "ExponentialSmoothing",
                Params = new Dictionary<string, object>
                {
                    { "seasonalPeriod", 12 }
                }
            };

            // Act
            var model = ModelFactory<double, Matrix<double>, Vector<double>>.Create(config);

            // Assert
            Assert.NotNull(model);
        }
    }
}
