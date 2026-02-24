using AiDotNet.Onnx;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Onnx;

/// <summary>
/// Integration tests for ONNX utility classes:
/// OnnxModelOptions, OnnxModelMetadata, OnnxTensorInfo, OnnxTensorConverter,
/// OnnxModelDownloader, OnnxExecutionProvider, OnnxModelBuilder, OnnxModelRepositories.
/// Tests that don't require an actual ONNX model file.
/// </summary>
public class OnnxIntegrationTests
{
    #region OnnxModelOptions - Defaults

    [Fact]
    public void OnnxModelOptions_DefaultValues()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(OnnxExecutionProvider.Auto, options.ExecutionProvider);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.True(options.EnableMemoryPattern);
        Assert.True(options.EnableMemoryArena);
        Assert.Equal(0, options.IntraOpNumThreads);
        Assert.Equal(0, options.InterOpNumThreads);
        Assert.Equal(GraphOptimizationLevel.All, options.OptimizationLevel);
        Assert.False(options.EnableProfiling);
        Assert.Null(options.ProfileOutputPath);
        Assert.False(options.AutoWarmUp);
        Assert.Equal(0L, options.CudaMemoryLimit);
        Assert.True(options.CudaUseArena);
        Assert.Equal(OnnxLogLevel.Warning, options.LogLevel);
    }

    [Fact]
    public void OnnxModelOptions_FallbackProviders_HasDefaults()
    {
        var options = new OnnxModelOptions();
        Assert.NotNull(options.FallbackProviders);
        Assert.Contains(OnnxExecutionProvider.Cuda, options.FallbackProviders);
        Assert.Contains(OnnxExecutionProvider.DirectML, options.FallbackProviders);
        Assert.Contains(OnnxExecutionProvider.Cpu, options.FallbackProviders);
    }

    [Fact]
    public void OnnxModelOptions_CustomOptions_Empty()
    {
        var options = new OnnxModelOptions();
        Assert.NotNull(options.CustomOptions);
        Assert.Empty(options.CustomOptions);
    }

    #endregion

    #region OnnxModelOptions - Factory Methods

    [Fact]
    public void OnnxModelOptions_ForCpu_SetsProvider()
    {
        var options = OnnxModelOptions.ForCpu();
        Assert.Equal(OnnxExecutionProvider.Cpu, options.ExecutionProvider);
        Assert.Empty(options.FallbackProviders);
        Assert.True(options.IntraOpNumThreads > 0);
    }

    [Fact]
    public void OnnxModelOptions_ForCpu_CustomThreads()
    {
        var options = OnnxModelOptions.ForCpu(threads: 4);
        Assert.Equal(4, options.IntraOpNumThreads);
    }

    [Fact]
    public void OnnxModelOptions_ForCuda_SetsProvider()
    {
        var options = OnnxModelOptions.ForCuda();
        Assert.Equal(OnnxExecutionProvider.Cuda, options.ExecutionProvider);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.Contains(OnnxExecutionProvider.Cpu, options.FallbackProviders);
    }

    [Fact]
    public void OnnxModelOptions_ForCuda_CustomDevice()
    {
        var options = OnnxModelOptions.ForCuda(deviceId: 1);
        Assert.Equal(1, options.GpuDeviceId);
    }

    [Fact]
    public void OnnxModelOptions_ForDirectML_SetsProvider()
    {
        var options = OnnxModelOptions.ForDirectML();
        Assert.Equal(OnnxExecutionProvider.DirectML, options.ExecutionProvider);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.Contains(OnnxExecutionProvider.Cpu, options.FallbackProviders);
    }

    [Fact]
    public void OnnxModelOptions_ForTensorRT_SetsProvider()
    {
        var options = OnnxModelOptions.ForTensorRT();
        Assert.Equal(OnnxExecutionProvider.TensorRT, options.ExecutionProvider);
        Assert.Contains(OnnxExecutionProvider.Cuda, options.FallbackProviders);
        Assert.Contains(OnnxExecutionProvider.Cpu, options.FallbackProviders);
    }

    #endregion

    #region OnnxModelOptions - Mutable Properties

    [Fact]
    public void OnnxModelOptions_SetProperties()
    {
        var options = new OnnxModelOptions
        {
            ExecutionProvider = OnnxExecutionProvider.Cuda,
            GpuDeviceId = 2,
            EnableMemoryPattern = false,
            EnableMemoryArena = false,
            IntraOpNumThreads = 8,
            InterOpNumThreads = 4,
            OptimizationLevel = GraphOptimizationLevel.Basic,
            EnableProfiling = true,
            ProfileOutputPath = "/tmp/profile",
            AutoWarmUp = true,
            CudaMemoryLimit = 1024 * 1024 * 1024L,
            CudaUseArena = false,
            LogLevel = OnnxLogLevel.Error
        };

        Assert.Equal(OnnxExecutionProvider.Cuda, options.ExecutionProvider);
        Assert.Equal(2, options.GpuDeviceId);
        Assert.False(options.EnableMemoryPattern);
        Assert.Equal(8, options.IntraOpNumThreads);
        Assert.True(options.EnableProfiling);
        Assert.True(options.AutoWarmUp);
        Assert.Equal(OnnxLogLevel.Error, options.LogLevel);
    }

    #endregion

    #region OnnxModelMetadata

    [Fact]
    public void OnnxModelMetadata_DefaultValues()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Equal(string.Empty, metadata.ModelName);
        Assert.Null(metadata.Description);
        Assert.Null(metadata.ProducerName);
        Assert.Null(metadata.ProducerVersion);
        Assert.Equal(0L, metadata.OpsetVersion);
        Assert.Empty(metadata.Inputs);
        Assert.Empty(metadata.Outputs);
        Assert.Null(metadata.Domain);
        Assert.Null(metadata.GraphName);
        Assert.Empty(metadata.CustomMetadata);
    }

    [Fact]
    public void OnnxModelMetadata_InitProperties()
    {
        var metadata = new OnnxModelMetadata
        {
            ModelName = "TestModel",
            Description = "A test model",
            ProducerName = "AiDotNet",
            ProducerVersion = "1.0",
            OpsetVersion = 17,
            Domain = "ai.onnx",
            GraphName = "main_graph"
        };

        Assert.Equal("TestModel", metadata.ModelName);
        Assert.Equal("A test model", metadata.Description);
        Assert.Equal("AiDotNet", metadata.ProducerName);
        Assert.Equal("1.0", metadata.ProducerVersion);
        Assert.Equal(17L, metadata.OpsetVersion);
        Assert.Equal("ai.onnx", metadata.Domain);
        Assert.Equal("main_graph", metadata.GraphName);
    }

    [Fact]
    public void OnnxModelMetadata_WithInputsAndOutputs()
    {
        var inputs = new List<AiDotNet.Interfaces.IOnnxTensorInfo>
        {
            new OnnxTensorInfo { Name = "input", Shape = new[] { 1, 3, 224, 224 }, ElementType = "float" }
        };
        var outputs = new List<AiDotNet.Interfaces.IOnnxTensorInfo>
        {
            new OnnxTensorInfo { Name = "output", Shape = new[] { 1, 1000 }, ElementType = "float" }
        };

        var metadata = new OnnxModelMetadata
        {
            Inputs = inputs,
            Outputs = outputs
        };

        Assert.Single(metadata.Inputs);
        Assert.Single(metadata.Outputs);
        Assert.Equal("input", metadata.Inputs[0].Name);
        Assert.Equal("output", metadata.Outputs[0].Name);
    }

    [Fact]
    public void OnnxModelMetadata_CustomMetadata()
    {
        var custom = new Dictionary<string, string>
        {
            ["author"] = "test",
            ["framework"] = "AiDotNet"
        };

        var metadata = new OnnxModelMetadata { CustomMetadata = custom };
        Assert.Equal(2, metadata.CustomMetadata.Count);
        Assert.Equal("test", metadata.CustomMetadata["author"]);
    }

    #endregion

    #region OnnxTensorInfo

    [Fact]
    public void OnnxTensorInfo_DefaultValues()
    {
        var info = new OnnxTensorInfo();
        Assert.Equal(string.Empty, info.Name);
        Assert.Empty(info.Shape);
        Assert.Equal("float", info.ElementType);
    }

    [Fact]
    public void OnnxTensorInfo_SetProperties()
    {
        var info = new OnnxTensorInfo
        {
            Name = "input_ids",
            Shape = new[] { -1, 512 },
            ElementType = "int64"
        };

        Assert.Equal("input_ids", info.Name);
        Assert.Equal(new[] { -1, 512 }, info.Shape);
        Assert.Equal("int64", info.ElementType);
    }

    [Fact]
    public void OnnxTensorInfo_HasDynamicShape_True()
    {
        var info = new OnnxTensorInfo { Shape = new[] { -1, 3, 224, 224 } };
        Assert.True(info.HasDynamicShape);
    }

    [Fact]
    public void OnnxTensorInfo_HasDynamicShape_False()
    {
        var info = new OnnxTensorInfo { Shape = new[] { 1, 3, 224, 224 } };
        Assert.False(info.HasDynamicShape);
    }

    [Fact]
    public void OnnxTensorInfo_TotalElements_StaticShape()
    {
        var info = new OnnxTensorInfo { Shape = new[] { 1, 3, 224, 224 } };
        Assert.Equal(1L * 3 * 224 * 224, info.TotalElements);
    }

    [Fact]
    public void OnnxTensorInfo_TotalElements_DynamicShape_ReturnsMinusOne()
    {
        var info = new OnnxTensorInfo { Shape = new[] { -1, 3, 224, 224 } };
        Assert.Equal(-1L, info.TotalElements);
    }

    [Fact]
    public void OnnxTensorInfo_ToString_StaticShape()
    {
        var info = new OnnxTensorInfo { Name = "output", Shape = new[] { 1, 10 }, ElementType = "float" };
        var str = info.ToString();
        Assert.Contains("output", str);
        Assert.Contains("float", str);
        Assert.Contains("1", str);
        Assert.Contains("10", str);
    }

    [Fact]
    public void OnnxTensorInfo_ToString_DynamicShape_UsesQuestionMark()
    {
        var info = new OnnxTensorInfo { Name = "x", Shape = new[] { -1, 512 }, ElementType = "int64" };
        var str = info.ToString();
        Assert.Contains("?", str);
        Assert.Contains("512", str);
    }

    #endregion

    #region OnnxTensorConverter - GetOnnxTypeName

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_Float()
    {
        Assert.Equal("float", OnnxTensorConverter.GetOnnxTypeName(typeof(float)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_Double()
    {
        Assert.Equal("double", OnnxTensorConverter.GetOnnxTypeName(typeof(double)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_Int32()
    {
        Assert.Equal("int32", OnnxTensorConverter.GetOnnxTypeName(typeof(int)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_Int64()
    {
        Assert.Equal("int64", OnnxTensorConverter.GetOnnxTypeName(typeof(long)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_Byte()
    {
        Assert.Equal("uint8", OnnxTensorConverter.GetOnnxTypeName(typeof(byte)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_Bool()
    {
        Assert.Equal("bool", OnnxTensorConverter.GetOnnxTypeName(typeof(bool)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_String()
    {
        Assert.Equal("string", OnnxTensorConverter.GetOnnxTypeName(typeof(string)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_UnknownType()
    {
        Assert.Equal("unknown", OnnxTensorConverter.GetOnnxTypeName(typeof(decimal)));
    }

    [Fact]
    public void OnnxTensorConverter_GetOnnxTypeName_AllIntTypes()
    {
        Assert.Equal("int16", OnnxTensorConverter.GetOnnxTypeName(typeof(short)));
        Assert.Equal("int8", OnnxTensorConverter.GetOnnxTypeName(typeof(sbyte)));
        Assert.Equal("uint16", OnnxTensorConverter.GetOnnxTypeName(typeof(ushort)));
        Assert.Equal("uint32", OnnxTensorConverter.GetOnnxTypeName(typeof(uint)));
        Assert.Equal("uint64", OnnxTensorConverter.GetOnnxTypeName(typeof(ulong)));
    }

    #endregion

    #region OnnxExecutionProvider Enum

    [Fact]
    public void OnnxExecutionProvider_HasExpectedValues()
    {
        Assert.Equal(0, (int)OnnxExecutionProvider.Cpu);
        Assert.Equal(1, (int)OnnxExecutionProvider.Cuda);
        Assert.Equal(2, (int)OnnxExecutionProvider.TensorRT);
        Assert.Equal(3, (int)OnnxExecutionProvider.DirectML);
        Assert.Equal(4, (int)OnnxExecutionProvider.CoreML);
        Assert.Equal(5, (int)OnnxExecutionProvider.OpenVINO);
        Assert.Equal(6, (int)OnnxExecutionProvider.ROCm);
        Assert.Equal(7, (int)OnnxExecutionProvider.NNAPI);
        Assert.Equal(100, (int)OnnxExecutionProvider.Auto);
    }

    #endregion

    #region GraphOptimizationLevel Enum

    [Fact]
    public void GraphOptimizationLevel_HasExpectedValues()
    {
        Assert.Equal(0, (int)GraphOptimizationLevel.None);
        Assert.Equal(1, (int)GraphOptimizationLevel.Basic);
        Assert.Equal(2, (int)GraphOptimizationLevel.Extended);
        Assert.Equal(99, (int)GraphOptimizationLevel.All);
    }

    #endregion

    #region OnnxLogLevel Enum

    [Fact]
    public void OnnxLogLevel_HasExpectedValues()
    {
        Assert.Equal(0, (int)OnnxLogLevel.Verbose);
        Assert.Equal(1, (int)OnnxLogLevel.Info);
        Assert.Equal(2, (int)OnnxLogLevel.Warning);
        Assert.Equal(3, (int)OnnxLogLevel.Error);
        Assert.Equal(4, (int)OnnxLogLevel.Fatal);
    }

    #endregion

    #region OnnxModelRepositories

    [Fact]
    public void OnnxModelRepositories_Whisper_HasExpectedValues()
    {
        Assert.Equal("openai/whisper-tiny", OnnxModelRepositories.Whisper.Tiny);
        Assert.Equal("openai/whisper-base", OnnxModelRepositories.Whisper.Base);
        Assert.Equal("openai/whisper-small", OnnxModelRepositories.Whisper.Small);
        Assert.Equal("openai/whisper-medium", OnnxModelRepositories.Whisper.Medium);
        Assert.Equal("openai/whisper-large-v3", OnnxModelRepositories.Whisper.Large);
    }

    [Fact]
    public void OnnxModelRepositories_Tts_HasExpectedValues()
    {
        Assert.Equal("microsoft/speecht5_tts", OnnxModelRepositories.Tts.FastSpeech2);
        Assert.Equal("facebook/hifigan", OnnxModelRepositories.Tts.HiFiGan);
    }

    [Fact]
    public void OnnxModelRepositories_AudioGen_HasExpectedValues()
    {
        Assert.Equal("facebook/audiogen-medium", OnnxModelRepositories.AudioGen.Small);
        Assert.Equal("facebook/musicgen-small", OnnxModelRepositories.AudioGen.MusicGenSmall);
    }

    #endregion

    #region OnnxModelDownloader - Construction and Cache

    [Fact]
    public void OnnxModelDownloader_DefaultConstruction_DoesNotThrow()
    {
        using var downloader = new OnnxModelDownloader(Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N")));
        Assert.NotNull(downloader);
    }

    [Fact]
    public void OnnxModelDownloader_GetCachedPath_ReturnsNullForMissing()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N"));
        using var downloader = new OnnxModelDownloader(cacheDir);
        var result = downloader.GetCachedPath("nonexistent/model", "model.onnx");
        Assert.Null(result);
    }

    [Fact]
    public void OnnxModelDownloader_GetCacheSize_ReturnsZeroForEmpty()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N"));
        using var downloader = new OnnxModelDownloader(cacheDir);
        var size = downloader.GetCacheSize();
        Assert.Equal(0L, size);
    }

    [Fact]
    public void OnnxModelDownloader_ListCachedModels_EmptyForNewCache()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N"));
        using var downloader = new OnnxModelDownloader(cacheDir);
        var models = downloader.ListCachedModels();
        Assert.Empty(models);
    }

    [Fact]
    public void OnnxModelDownloader_ClearCache_DoesNotThrowOnEmpty()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N"));
        using var downloader = new OnnxModelDownloader(cacheDir);
        downloader.ClearCache();
    }

    [Fact]
    public void OnnxModelDownloader_Dispose_CanCallMultipleTimes()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N"));
        var downloader = new OnnxModelDownloader(cacheDir);
        downloader.Dispose();
        downloader.Dispose(); // Should not throw
    }

    [Fact]
    public async Task OnnxModelDownloader_DownloadAsync_NullModelId_Throws()
    {
        var cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet_test_cache_" + Guid.NewGuid().ToString("N"));
        using var downloader = new OnnxModelDownloader(cacheDir);
        await Assert.ThrowsAsync<ArgumentNullException>(() => downloader.DownloadAsync(null!));
    }

    [Fact]
    public void OnnxModelDownloader_HuggingFaceBaseUrl_IsSet()
    {
        Assert.Equal("https://huggingface.co", OnnxModelDownloader.HuggingFaceBaseUrl);
    }

    #endregion

    #region OnnxModel - Validation

    [Fact]
    public void OnnxModel_NullPath_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxModel<float>(modelPath: null!));
    }

    [Fact]
    public void OnnxModel_EmptyPath_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxModel<float>(modelPath: ""));
    }

    [Fact]
    public void OnnxModel_NonexistentFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => new OnnxModel<float>(modelPath: "/nonexistent/model.onnx"));
    }

    [Fact]
    public void OnnxModel_NullBytes_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxModel<float>(modelBytes: null!));
    }

    #endregion
}
