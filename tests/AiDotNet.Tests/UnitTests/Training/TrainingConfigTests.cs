using System;
using AiDotNet.Configuration;
using AiDotNet.Training.Configuration;
using Xunit;

namespace AiDotNetTests.UnitTests.Training
{
    public class TrainingConfigTests
    {
        [Fact]
        public void LoadFromString_FullRecipe_DeserializesAllSections()
        {
            // Arrange
            var yaml = @"
model:
  name: ""ARIMA""
  params:
    lagOrder: 3

dataset:
  name: ""test-data""
  path: ""data/test.csv""
  batchSize: 64
  hasHeader: true
  labelColumn: -1

optimizer:
  name: ""Adam""
  learningRate: 0.01

lossFunction:
  name: ""Huber""
  params:
    delta: 1.5

trainer:
  epochs: 100
  enableLogging: false
  seed: 42
";

            // Act
            var config = YamlConfigLoader.LoadFromString<TrainingRecipeConfig>(yaml);

            // Assert - Model section
            Assert.NotNull(config.Model);
            Assert.Equal("ARIMA", config.Model.Name);
            Assert.True(config.Model.Params.ContainsKey("lagOrder"));

            // Assert - Dataset section
            Assert.NotNull(config.Dataset);
            Assert.Equal("test-data", config.Dataset.Name);
            Assert.Equal("data/test.csv", config.Dataset.Path);
            Assert.Equal(64, config.Dataset.BatchSize);
            Assert.True(config.Dataset.HasHeader);
            Assert.Equal(-1, config.Dataset.LabelColumn);

            // Assert - Optimizer section
            Assert.NotNull(config.Optimizer);
            Assert.Equal("Adam", config.Optimizer.Name);
            Assert.Equal(0.01, config.Optimizer.LearningRate, 10);

            // Assert - Loss function section
            Assert.NotNull(config.LossFunction);
            Assert.Equal("Huber", config.LossFunction.Name);
            Assert.True(config.LossFunction.Params.ContainsKey("delta"));

            // Assert - Trainer section
            Assert.NotNull(config.Trainer);
            Assert.Equal(100, config.Trainer.Epochs);
            Assert.False(config.Trainer.EnableLogging);
            Assert.Equal(42, config.Trainer.Seed);
        }

        [Fact]
        public void LoadFromString_MinimalRecipe_UsesDefaults()
        {
            // Arrange
            var yaml = @"
model:
  name: ""ExponentialSmoothing""
";

            // Act
            var config = YamlConfigLoader.LoadFromString<TrainingRecipeConfig>(yaml);

            // Assert
            Assert.NotNull(config.Model);
            Assert.Equal("ExponentialSmoothing", config.Model.Name);
            Assert.Null(config.Dataset);
            Assert.Null(config.Optimizer);
            Assert.Null(config.LossFunction);
            Assert.Null(config.Trainer);
        }

        [Fact]
        public void LoadFromString_EmptyYaml_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                YamlConfigLoader.LoadFromString<TrainingRecipeConfig>(""));
        }

        [Fact]
        public void TrainingRecipeConfig_DefaultValues_AreCorrect()
        {
            // Act
            var config = new TrainingRecipeConfig();

            // Assert
            Assert.Null(config.Model);
            Assert.Null(config.Dataset);
            Assert.Null(config.Optimizer);
            Assert.Null(config.LossFunction);
            Assert.Null(config.Trainer);
        }

        [Fact]
        public void DatasetConfig_DefaultValues_AreCorrect()
        {
            // Act
            var config = new DatasetConfig();

            // Assert
            Assert.Equal(string.Empty, config.Name);
            Assert.Equal(string.Empty, config.Path);
            Assert.Equal(32, config.BatchSize);
            Assert.True(config.HasHeader);
            Assert.Equal(-1, config.LabelColumn);
        }

        [Fact]
        public void TrainerSettings_DefaultValues_AreCorrect()
        {
            // Act
            var config = new TrainerSettings();

            // Assert
            Assert.Equal(10, config.Epochs);
            Assert.True(config.EnableLogging);
            Assert.Null(config.Seed);
        }

        [Fact]
        public void OptimizerConfig_DefaultValues_AreCorrect()
        {
            // Act
            var config = new OptimizerConfig();

            // Assert
            Assert.Equal(string.Empty, config.Name);
            Assert.Equal(0.001, config.LearningRate, 10);
        }

        [Fact]
        public void LossFunctionConfig_DefaultValues_AreCorrect()
        {
            // Act
            var config = new LossFunctionConfig();

            // Assert
            Assert.Equal(string.Empty, config.Name);
            Assert.NotNull(config.Params);
            Assert.Empty(config.Params);
        }

        [Fact]
        public void ModelConfig_DefaultValues_AreCorrect()
        {
            // Act
            var config = new ModelConfig();

            // Assert
            Assert.Equal(string.Empty, config.Name);
            Assert.NotNull(config.Params);
            Assert.Empty(config.Params);
        }

        [Fact]
        public void LoadFromString_WithOnlyTrainerSection_DeserializesCorrectly()
        {
            // Arrange
            var yaml = @"
model:
  name: ""ARIMA""
trainer:
  epochs: 25
  seed: 123
";

            // Act
            var config = YamlConfigLoader.LoadFromString<TrainingRecipeConfig>(yaml);

            // Assert
            Assert.NotNull(config.Trainer);
            Assert.Equal(25, config.Trainer.Epochs);
            Assert.Equal(123, config.Trainer.Seed);
            Assert.True(config.Trainer.EnableLogging); // default
        }

        [Fact]
        public void LoadFromString_WithLossFunctionParams_DeserializesParamsDictionary()
        {
            // Arrange
            var yaml = @"
model:
  name: ""ARIMA""
lossFunction:
  name: ""Focal""
  params:
    gamma: 3.0
    alpha: 0.5
";

            // Act
            var config = YamlConfigLoader.LoadFromString<TrainingRecipeConfig>(yaml);

            // Assert
            Assert.NotNull(config.LossFunction);
            Assert.Equal("Focal", config.LossFunction.Name);
            Assert.True(config.LossFunction.Params.ContainsKey("gamma"));
            Assert.True(config.LossFunction.Params.ContainsKey("alpha"));
        }
    }
}
