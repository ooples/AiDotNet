using System;
using System.IO;
using AiDotNet.Onnx;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.Onnx
{
    public class OnnxModelTests
    {
        [Fact(Timeout = 60000)]
        public async Task OnnxModelOptions_DefaultValues()
        {
            // Arrange & Act
            var options = new OnnxModelOptions();

            // Assert
            Assert.Equal(OnnxExecutionProvider.Auto, options.ExecutionProvider);
            Assert.True(options.EnableMemoryArena);
            Assert.True(options.EnableMemoryPattern);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelOptions_SetExecutionProvider()
        {
            // Arrange & Act
            var options = new OnnxModelOptions
            {
                ExecutionProvider = OnnxExecutionProvider.Cuda
            };

            // Assert
            Assert.Equal(OnnxExecutionProvider.Cuda, options.ExecutionProvider);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxExecutionProvider_HasExpectedValues()
        {
            // Assert - verify the enum values exist
            Assert.True(Enum.IsDefined(typeof(OnnxExecutionProvider), OnnxExecutionProvider.Cpu));
            Assert.True(Enum.IsDefined(typeof(OnnxExecutionProvider), OnnxExecutionProvider.Cuda));
            Assert.True(Enum.IsDefined(typeof(OnnxExecutionProvider), OnnxExecutionProvider.TensorRT));
            Assert.True(Enum.IsDefined(typeof(OnnxExecutionProvider), OnnxExecutionProvider.DirectML));
            Assert.True(Enum.IsDefined(typeof(OnnxExecutionProvider), OnnxExecutionProvider.CoreML));
            Assert.True(Enum.IsDefined(typeof(OnnxExecutionProvider), OnnxExecutionProvider.Auto));
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_DefaultCacheDirectory_IsSet()
        {
            // Arrange & Act
            var downloader = new OnnxModelDownloader();

            // Assert - verify it doesn't throw
            Assert.NotNull(downloader);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_CustomCacheDirectory()
        {
            // Arrange
            var tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-test-cache-" + Guid.NewGuid());

            try
            {
                // Act
                var downloader = new OnnxModelDownloader(tempDir);

                // Assert
                Assert.NotNull(downloader);
                Assert.True(Directory.Exists(tempDir), "Cache directory should be created");
            }
            finally
            {
                // Cleanup
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, true);
                }
            }
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_GetCachedPath_ReturnsNullForNonExistent()
        {
            // Arrange
            var downloader = new OnnxModelDownloader();

            // Act
            var path = downloader.GetCachedPath("nonexistent/model", "model.onnx");

            // Assert
            Assert.Null(path);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_GetCacheSize_ReturnsZeroForEmptyCache()
        {
            // Arrange
            var tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-empty-cache-" + Guid.NewGuid());
            var downloader = new OnnxModelDownloader(tempDir);

            try
            {
                // Act
                long size = downloader.GetCacheSize();

                // Assert
                Assert.Equal(0, size);
            }
            finally
            {
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, true);
                }
            }
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_ListCachedModels_ReturnsEmptyForNewCache()
        {
            // Arrange
            var tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-new-cache-" + Guid.NewGuid());
            var downloader = new OnnxModelDownloader(tempDir);

            try
            {
                // Act
                var models = downloader.ListCachedModels();

                // Assert
                Assert.NotNull(models);
                Assert.Empty(models);
            }
            finally
            {
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, true);
                }
            }
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_ClearCache_DoesNotThrow()
        {
            // Arrange
            var tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-clear-cache-" + Guid.NewGuid());
            var downloader = new OnnxModelDownloader(tempDir);

            try
            {
                // Act & Assert - should not throw
                downloader.ClearCache();
            }
            finally
            {
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, true);
                }
            }
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelRepositories_Whisper_HasCorrectValues()
        {
            // Assert
            Assert.Equal("openai/whisper-tiny", OnnxModelRepositories.Whisper.Tiny);
            Assert.Equal("openai/whisper-base", OnnxModelRepositories.Whisper.Base);
            Assert.Equal("openai/whisper-small", OnnxModelRepositories.Whisper.Small);
            Assert.Equal("openai/whisper-medium", OnnxModelRepositories.Whisper.Medium);
            Assert.Equal("openai/whisper-large-v3", OnnxModelRepositories.Whisper.Large);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelRepositories_AudioGen_HasCorrectValues()
        {
            // Assert
            Assert.Equal("facebook/audiogen-medium", OnnxModelRepositories.AudioGen.Small);
            Assert.Equal("facebook/musicgen-small", OnnxModelRepositories.AudioGen.MusicGenSmall);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelDownloader_HuggingFaceBaseUrl_IsCorrect()
        {
            // Assert
            Assert.Equal("https://huggingface.co", OnnxModelDownloader.HuggingFaceBaseUrl);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelOptions_ForCpu_SetsCorrectProvider()
        {
            // Act
            var options = OnnxModelOptions.ForCpu();

            // Assert
            Assert.Equal(OnnxExecutionProvider.Cpu, options.ExecutionProvider);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelOptions_ForCuda_SetsCorrectProvider()
        {
            // Act
            var options = OnnxModelOptions.ForCuda();

            // Assert
            Assert.Equal(OnnxExecutionProvider.Cuda, options.ExecutionProvider);
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxModelOptions_ForDirectML_SetsCorrectProvider()
        {
            // Act
            var options = OnnxModelOptions.ForDirectML();

            // Assert
            Assert.Equal(OnnxExecutionProvider.DirectML, options.ExecutionProvider);
        }

        [Fact(Timeout = 60000)]
        public async Task GraphOptimizationLevel_HasExpectedValues()
        {
            // Assert
            Assert.True(Enum.IsDefined(typeof(GraphOptimizationLevel), GraphOptimizationLevel.None));
            Assert.True(Enum.IsDefined(typeof(GraphOptimizationLevel), GraphOptimizationLevel.Basic));
            Assert.True(Enum.IsDefined(typeof(GraphOptimizationLevel), GraphOptimizationLevel.Extended));
            Assert.True(Enum.IsDefined(typeof(GraphOptimizationLevel), GraphOptimizationLevel.All));
        }

        [Fact(Timeout = 60000)]
        public async Task OnnxLogLevel_HasExpectedValues()
        {
            // Assert
            Assert.True(Enum.IsDefined(typeof(OnnxLogLevel), OnnxLogLevel.Verbose));
            Assert.True(Enum.IsDefined(typeof(OnnxLogLevel), OnnxLogLevel.Info));
            Assert.True(Enum.IsDefined(typeof(OnnxLogLevel), OnnxLogLevel.Warning));
            Assert.True(Enum.IsDefined(typeof(OnnxLogLevel), OnnxLogLevel.Error));
            Assert.True(Enum.IsDefined(typeof(OnnxLogLevel), OnnxLogLevel.Fatal));
        }
    }
}
