using System;
using System.IO;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Training.Configuration;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Training
{
    public class TrainerTests
    {
        [Fact]
        public void Trainer_WithInMemoryData_CompletesTraining()
        {
            // Arrange - create a simple time series dataset
            var config = new TrainingRecipeConfig
            {
                Model = new ModelConfig { Name = "ExponentialSmoothing" },
                LossFunction = new LossFunctionConfig { Name = "MeanSquaredError" },
                Trainer = new TrainerSettings
                {
                    Epochs = 3,
                    EnableLogging = false
                }
            };

            var trainer = new Trainer<double>(config);

            // Create simple synthetic data (10 samples, 2 features)
            var features = new Matrix<double>(10, 2);
            var labels = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                features[i, 0] = i * 1.0;
                features[i, 1] = i * 2.0;
                labels[i] = i * 3.0;
            }

            trainer.SetData(features, labels);

            // Act
            var result = trainer.Run();

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Completed);
            Assert.Equal(3, result.TotalEpochs);
            Assert.Equal(3, result.EpochLosses.Count);
            Assert.NotNull(result.TrainedModel);
            Assert.True(result.TrainingDuration.TotalMilliseconds > 0);
        }

        [Fact]
        public void Trainer_WithCsvData_CompletesTraining()
        {
            // Arrange - create a temporary CSV file
            var tempFile = Path.GetTempFileName();
            try
            {
                var csvContent = "feature1,feature2,target\n";
                for (int i = 0; i < 20; i++)
                {
                    csvContent += $"{i * 1.0},{i * 2.0},{i * 3.0}\n";
                }
                File.WriteAllText(tempFile, csvContent);

                var config = new TrainingRecipeConfig
                {
                    Model = new ModelConfig { Name = "ExponentialSmoothing" },
                    Dataset = new DatasetConfig
                    {
                        Path = tempFile,
                        HasHeader = true,
                        LabelColumn = -1,
                        BatchSize = 32
                    },
                    LossFunction = new LossFunctionConfig { Name = "MeanSquaredError" },
                    Trainer = new TrainerSettings
                    {
                        Epochs = 2,
                        EnableLogging = false
                    }
                };

                var trainer = new Trainer<double>(config);

                // Act
                var result = trainer.Run();

                // Assert
                Assert.NotNull(result);
                Assert.True(result.Completed);
                Assert.Equal(2, result.TotalEpochs);
                Assert.NotNull(result.TrainedModel);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        [Fact]
        public void Trainer_FromYamlString_CreatesAndRunsCorrectly()
        {
            // Arrange - create temp CSV and YAML files
            var csvFile = Path.GetTempFileName();
            var yamlFile = Path.GetTempFileName();
            try
            {
                var csvContent = "x,y,target\n";
                for (int i = 0; i < 15; i++)
                {
                    csvContent += $"{i},{i * 2},{i * 3}\n";
                }
                File.WriteAllText(csvFile, csvContent);

                var yamlContent = $@"
model:
  name: ""ExponentialSmoothing""

dataset:
  path: ""{csvFile.Replace("\\", "/")}""
  hasHeader: true
  labelColumn: -1

lossFunction:
  name: ""MeanSquaredError""

trainer:
  epochs: 2
  enableLogging: false
";
                File.WriteAllText(yamlFile, yamlContent);

                // Act
                var trainer = new Trainer<double>(yamlFile);
                var result = trainer.Run();

                // Assert
                Assert.NotNull(result);
                Assert.True(result.Completed);
                Assert.Equal(2, result.TotalEpochs);
                Assert.NotNull(result.TrainedModel);
            }
            finally
            {
                File.Delete(csvFile);
                File.Delete(yamlFile);
            }
        }

        [Fact]
        public void Trainer_WithNoData_ThrowsInvalidOperationException()
        {
            // Arrange
            var config = new TrainingRecipeConfig
            {
                Model = new ModelConfig { Name = "ExponentialSmoothing" },
                Trainer = new TrainerSettings { Epochs = 1, EnableLogging = false }
            };

            var trainer = new Trainer<double>(config);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => trainer.Run());
        }

        [Fact]
        public void Trainer_WithNoModel_ThrowsArgumentException()
        {
            // Arrange
            var config = new TrainingRecipeConfig();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Trainer<double>(config));
        }

        [Fact]
        public void Trainer_WithNullConfig_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new Trainer<double>((TrainingRecipeConfig)null));
        }

        [Fact]
        public void Trainer_EpochLosses_AreRecorded()
        {
            // Arrange
            var config = new TrainingRecipeConfig
            {
                Model = new ModelConfig { Name = "ExponentialSmoothing" },
                LossFunction = new LossFunctionConfig { Name = "MeanSquaredError" },
                Trainer = new TrainerSettings
                {
                    Epochs = 5,
                    EnableLogging = false
                }
            };

            var trainer = new Trainer<double>(config);

            var features = new Matrix<double>(10, 2);
            var labels = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                features[i, 0] = i;
                features[i, 1] = i * 0.5;
                labels[i] = i * 2.0;
            }
            trainer.SetData(features, labels);

            // Act
            var result = trainer.Run();

            // Assert
            Assert.Equal(5, result.EpochLosses.Count);
            foreach (var loss in result.EpochLosses)
            {
                // Loss values should be non-negative for MSE
                Assert.True(loss >= 0.0);
            }
        }

        [Fact]
        public void Trainer_ConfigProperty_ReturnsOriginalConfig()
        {
            // Arrange
            var config = new TrainingRecipeConfig
            {
                Model = new ModelConfig { Name = "ExponentialSmoothing" },
                Trainer = new TrainerSettings { Epochs = 10, EnableLogging = false }
            };

            // Act
            var trainer = new Trainer<double>(config);

            // Assert
            Assert.Same(config, trainer.Config);
            Assert.Equal("ExponentialSmoothing", trainer.Config.Model?.Name);
        }
    }
}
