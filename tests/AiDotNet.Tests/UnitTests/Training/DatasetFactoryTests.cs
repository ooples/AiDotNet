using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.Training.Configuration;
using AiDotNet.Training.Factories;
using Xunit;

namespace AiDotNetTests.UnitTests.Training
{
    public class DatasetFactoryTests
    {
        [Fact]
        public void Create_WithNullConfig_ReturnsNull()
        {
            // Act
            var loader = DatasetFactory<double>.Create(null);

            // Assert
            Assert.Null(loader);
        }

        [Fact]
        public void Create_WithEmptyPath_ReturnsNull()
        {
            // Arrange
            var config = new DatasetConfig { Path = "" };

            // Act
            var loader = DatasetFactory<double>.Create(config);

            // Assert
            Assert.Null(loader);
        }

        [Fact]
        public void Create_WithValidPath_ReturnsCsvDataLoader()
        {
            // Arrange
            var config = new DatasetConfig
            {
                Path = "test.csv",
                HasHeader = true,
                LabelColumn = -1,
                BatchSize = 64
            };

            // Act
            var loader = DatasetFactory<double>.Create(config);

            // Assert - verify the loader was created with the correct type and config values applied
            Assert.NotNull(loader);
            var csvLoader = Assert.IsType<CsvDataLoader<double>>(loader);
            Assert.Equal(64, csvLoader.BatchSize);
            Assert.Equal("CsvDataLoader", csvLoader.Name);
        }

        [Fact]
        public async Task CsvDataLoader_LoadsValidCsv_CorrectDimensions()
        {
            // Arrange - create a temporary CSV file
            var tempFile = Path.GetTempFileName();
            try
            {
                File.WriteAllText(tempFile, "a,b,c,label\n1.0,2.0,3.0,10.0\n4.0,5.0,6.0,20.0\n7.0,8.0,9.0,30.0\n");

                var loader = new CsvDataLoader<double>(tempFile, hasHeader: true, labelColumn: -1, batchSize: 32);

                // Act
                await loader.LoadAsync();

                // Assert
                Assert.Equal(3, loader.TotalCount);   // 3 data rows
                Assert.Equal(3, loader.FeatureCount);  // 3 feature columns (a, b, c)
                Assert.Equal(1, loader.OutputDimension);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        [Fact]
        public async Task CsvDataLoader_LoadsCorrectValues()
        {
            // Arrange
            var tempFile = Path.GetTempFileName();
            try
            {
                File.WriteAllText(tempFile, "x,y\n1.0,10.0\n2.0,20.0\n");

                var loader = new CsvDataLoader<double>(tempFile, hasHeader: true, labelColumn: -1, batchSize: 32);

                // Act
                await loader.LoadAsync();

                // Assert
                Assert.Equal(1.0, loader.Features[0, 0], 10);
                Assert.Equal(2.0, loader.Features[1, 0], 10);
                Assert.Equal(10.0, loader.Labels[0], 10);
                Assert.Equal(20.0, loader.Labels[1], 10);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        [Fact]
        public async Task CsvDataLoader_WithNoHeader_LoadsCorrectly()
        {
            // Arrange
            var tempFile = Path.GetTempFileName();
            try
            {
                File.WriteAllText(tempFile, "1.0,2.0,10.0\n3.0,4.0,20.0\n");

                var loader = new CsvDataLoader<double>(tempFile, hasHeader: false, labelColumn: -1, batchSize: 32);

                // Act
                await loader.LoadAsync();

                // Assert
                Assert.Equal(2, loader.TotalCount);
                Assert.Equal(2, loader.FeatureCount);
                Assert.Equal(10.0, loader.Labels[0], 10);
                Assert.Equal(20.0, loader.Labels[1], 10);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        [Fact]
        public async Task CsvDataLoader_WithFirstColumnLabel_LoadsCorrectly()
        {
            // Arrange
            var tempFile = Path.GetTempFileName();
            try
            {
                File.WriteAllText(tempFile, "label,a,b\n10.0,1.0,2.0\n20.0,3.0,4.0\n");

                var loader = new CsvDataLoader<double>(tempFile, hasHeader: true, labelColumn: 0, batchSize: 32);

                // Act
                await loader.LoadAsync();

                // Assert
                Assert.Equal(2, loader.TotalCount);
                Assert.Equal(2, loader.FeatureCount);
                Assert.Equal(10.0, loader.Labels[0], 10);
                Assert.Equal(1.0, loader.Features[0, 0], 10);
                Assert.Equal(2.0, loader.Features[0, 1], 10);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        [Fact]
        public async Task CsvDataLoader_WithMissingFile_ThrowsFileNotFoundException()
        {
            // Arrange
            var loader = new CsvDataLoader<double>("nonexistent_file.csv", hasHeader: true);

            // Act & Assert
            await Assert.ThrowsAsync<FileNotFoundException>(() => loader.LoadAsync());
        }

        [Fact]
        public void CsvDataLoader_WithEmptyPath_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new CsvDataLoader<double>(""));
        }
    }
}
