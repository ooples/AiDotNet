using System.Text;
using AiDotNet.Serving.Services;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class AesGcmModelArtifactProtectorTests
{
    [Theory]
    [InlineData(null, "in.bin", "out", "modelName")]
    [InlineData(" ", "in.bin", "out", "modelName")]
    [InlineData("model", null, "out", "sourcePath")]
    [InlineData("model", " ", "out", "sourcePath")]
    [InlineData("model", "in.bin", null, "outputDirectory")]
    [InlineData("model", "in.bin", " ", "outputDirectory")]
    public void ProtectToFile_Throws_WhenRequiredArgumentsMissing(
        string? modelName,
        string? sourcePath,
        string? outputDirectory,
        string expectedParamName)
    {
        var protector = new AesGcmModelArtifactProtector();

        var ex = Assert.Throws<ArgumentException>(() =>
            protector.ProtectToFile(modelName!, sourcePath!, outputDirectory!));

        Assert.Equal(expectedParamName, ex.ParamName);
    }

    [Fact]
    public void ProtectToFile_WritesHeaderAndReturnsArtifact()
    {
        var protector = new AesGcmModelArtifactProtector();

        var workDir = Path.Combine(Path.GetTempPath(), "AiDotNet.Serving.Tests", Guid.NewGuid().ToString("N"));
        var inputPath = Path.Combine(workDir, "model.bin");
        var outputDir = Path.Combine(workDir, "out");

        try
        {
            Directory.CreateDirectory(workDir);
            File.WriteAllBytes(inputPath, Encoding.UTF8.GetBytes("hello world"));

            var artifact = protector.ProtectToFile("my:model", inputPath, outputDir);

            Assert.Equal("my:model", artifact.ModelName);
            Assert.EndsWith(".aidn.enc", artifact.EncryptedPath, StringComparison.OrdinalIgnoreCase);
            Assert.Equal("AES-256-GCM", artifact.Algorithm);
            Assert.False(string.IsNullOrWhiteSpace(artifact.KeyId));
            Assert.True(File.Exists(artifact.EncryptedPath));
            Assert.Equal("my_model.aidn.enc", Path.GetFileName(artifact.EncryptedPath));

            var bytes = File.ReadAllBytes(artifact.EncryptedPath);
            Assert.True(bytes.Length > 4);

            Assert.Equal((byte)'A', bytes[0]);
            Assert.Equal((byte)'I', bytes[1]);
            Assert.Equal((byte)'D', bytes[2]);
            Assert.Equal((byte)'N', bytes[3]);
            Assert.Equal(1, bytes[4]); // version
        }
        finally
        {
            if (Directory.Exists(workDir))
            {
                Directory.Delete(workDir, recursive: true);
            }
        }
    }
}
