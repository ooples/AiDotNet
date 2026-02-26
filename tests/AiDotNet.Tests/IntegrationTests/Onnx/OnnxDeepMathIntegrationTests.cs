using AiDotNet.Interfaces;
using AiDotNet.Onnx;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Onnx;

/// <summary>
/// Deep integration tests for Onnx:
/// OnnxModelMetadata/OnnxTensorInfo data models and computed properties,
/// OnnxModelOptions defaults and factory methods,
/// OnnxExecutionProvider/GraphOptimizationLevel/OnnxLogLevel enums.
/// </summary>
public class OnnxDeepMathIntegrationTests
{
    // ============================
    // OnnxModelMetadata: Defaults
    // ============================

    [Fact]
    public void OnnxModelMetadata_Defaults_ModelNameEmpty()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Equal(string.Empty, metadata.ModelName);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_DescriptionNull()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Null(metadata.Description);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_ProducerNameNull()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Null(metadata.ProducerName);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_ProducerVersionNull()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Null(metadata.ProducerVersion);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_OpsetVersionZero()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Equal(0, metadata.OpsetVersion);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_InputsEmpty()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Empty(metadata.Inputs);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_OutputsEmpty()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Empty(metadata.Outputs);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_DomainNull()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Null(metadata.Domain);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_GraphNameNull()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Null(metadata.GraphName);
    }

    [Fact]
    public void OnnxModelMetadata_Defaults_CustomMetadataEmpty()
    {
        var metadata = new OnnxModelMetadata();
        Assert.Empty(metadata.CustomMetadata);
    }

    [Fact]
    public void OnnxModelMetadata_SetProperties()
    {
        var inputs = new List<IOnnxTensorInfo>
        {
            new OnnxTensorInfo { Name = "input", Shape = new[] { 1, 3, 224, 224 }, ElementType = "float" }
        };
        var outputs = new List<IOnnxTensorInfo>
        {
            new OnnxTensorInfo { Name = "output", Shape = new[] { 1, 1000 }, ElementType = "float" }
        };
        var customMeta = new Dictionary<string, string> { { "author", "test" }, { "version", "1.0" } };

        var metadata = new OnnxModelMetadata
        {
            ModelName = "ResNet50",
            Description = "Image classification model",
            ProducerName = "PyTorch",
            ProducerVersion = "2.0",
            OpsetVersion = 17,
            Inputs = inputs,
            Outputs = outputs,
            Domain = "ai.onnx",
            GraphName = "main_graph",
            CustomMetadata = customMeta
        };

        Assert.Equal("ResNet50", metadata.ModelName);
        Assert.Equal("Image classification model", metadata.Description);
        Assert.Equal("PyTorch", metadata.ProducerName);
        Assert.Equal("2.0", metadata.ProducerVersion);
        Assert.Equal(17, metadata.OpsetVersion);
        Assert.Single(metadata.Inputs);
        Assert.Single(metadata.Outputs);
        Assert.Equal("ai.onnx", metadata.Domain);
        Assert.Equal("main_graph", metadata.GraphName);
        Assert.Equal(2, metadata.CustomMetadata.Count);
        Assert.Equal("test", metadata.CustomMetadata["author"]);
    }

    // ============================
    // OnnxTensorInfo: Defaults
    // ============================

    [Fact]
    public void OnnxTensorInfo_Defaults_NameEmpty()
    {
        var info = new OnnxTensorInfo();
        Assert.Equal(string.Empty, info.Name);
    }

    [Fact]
    public void OnnxTensorInfo_Defaults_ShapeEmpty()
    {
        var info = new OnnxTensorInfo();
        Assert.Empty(info.Shape);
    }

    [Fact]
    public void OnnxTensorInfo_Defaults_ElementTypeFloat()
    {
        var info = new OnnxTensorInfo();
        Assert.Equal("float", info.ElementType);
    }

    // ============================
    // OnnxTensorInfo: HasDynamicShape
    // ============================

    [Fact]
    public void OnnxTensorInfo_StaticShape_NotDynamic()
    {
        var info = new OnnxTensorInfo { Shape = new[] { 1, 3, 224, 224 } };
        Assert.False(info.HasDynamicShape);
    }

    [Fact]
    public void OnnxTensorInfo_DynamicBatchDim_IsDynamic()
    {
        var info = new OnnxTensorInfo { Shape = new[] { -1, 3, 224, 224 } };
        Assert.True(info.HasDynamicShape);
    }

    [Fact]
    public void OnnxTensorInfo_MultipleDynamic_IsDynamic()
    {
        var info = new OnnxTensorInfo { Shape = new[] { -1, -1, 224, 224 } };
        Assert.True(info.HasDynamicShape);
    }

    [Fact]
    public void OnnxTensorInfo_EmptyShape_NotDynamic()
    {
        var info = new OnnxTensorInfo { Shape = Array.Empty<int>() };
        Assert.False(info.HasDynamicShape);
    }

    // ============================
    // OnnxTensorInfo: TotalElements
    // ============================

    [Fact]
    public void OnnxTensorInfo_TotalElements_StaticShape()
    {
        var info = new OnnxTensorInfo { Shape = new[] { 1, 3, 224, 224 } };
        // 1 * 3 * 224 * 224 = 150528
        Assert.Equal(150528L, info.TotalElements);
    }

    [Fact]
    public void OnnxTensorInfo_TotalElements_DynamicShape_ReturnsNegativeOne()
    {
        var info = new OnnxTensorInfo { Shape = new[] { -1, 3, 224, 224 } };
        Assert.Equal(-1L, info.TotalElements);
    }

    [Fact]
    public void OnnxTensorInfo_TotalElements_SingleDim()
    {
        var info = new OnnxTensorInfo { Shape = new[] { 1000 } };
        Assert.Equal(1000L, info.TotalElements);
    }

    [Fact]
    public void OnnxTensorInfo_TotalElements_ScalarShape()
    {
        // Empty shape means scalar with 1 element (product of empty = 1)
        var info = new OnnxTensorInfo { Shape = Array.Empty<int>() };
        Assert.Equal(1L, info.TotalElements);
    }

    [Fact]
    public void OnnxTensorInfo_TotalElements_LargeShape()
    {
        var info = new OnnxTensorInfo { Shape = new[] { 8, 512, 512, 3 } };
        // 8 * 512 * 512 * 3 = 6291456
        Assert.Equal(6291456L, info.TotalElements);
    }

    // ============================
    // OnnxTensorInfo: ToString
    // ============================

    [Fact]
    public void OnnxTensorInfo_ToString_StaticShape()
    {
        var info = new OnnxTensorInfo { Name = "input", Shape = new[] { 1, 3, 224, 224 }, ElementType = "float" };
        var str = info.ToString();
        Assert.Equal("input: float[1, 3, 224, 224]", str);
    }

    [Fact]
    public void OnnxTensorInfo_ToString_DynamicShape()
    {
        var info = new OnnxTensorInfo { Name = "output", Shape = new[] { -1, 1000 }, ElementType = "float" };
        var str = info.ToString();
        Assert.Equal("output: float[?, 1000]", str);
    }

    [Fact]
    public void OnnxTensorInfo_ToString_DoublePrecision()
    {
        var info = new OnnxTensorInfo { Name = "data", Shape = new[] { 100 }, ElementType = "double" };
        var str = info.ToString();
        Assert.Equal("data: double[100]", str);
    }

    // ============================
    // OnnxModelOptions: Defaults
    // ============================

    [Fact]
    public void OnnxModelOptions_Defaults_ExecutionProviderAuto()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(OnnxExecutionProvider.Auto, options.ExecutionProvider);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_FallbackProviders()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(3, options.FallbackProviders.Count);
        Assert.Equal(OnnxExecutionProvider.Cuda, options.FallbackProviders[0]);
        Assert.Equal(OnnxExecutionProvider.DirectML, options.FallbackProviders[1]);
        Assert.Equal(OnnxExecutionProvider.Cpu, options.FallbackProviders[2]);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_GpuDeviceIdZero()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(0, options.GpuDeviceId);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_MemoryPatternEnabled()
    {
        var options = new OnnxModelOptions();
        Assert.True(options.EnableMemoryPattern);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_MemoryArenaEnabled()
    {
        var options = new OnnxModelOptions();
        Assert.True(options.EnableMemoryArena);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_ThreadsZero()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(0, options.IntraOpNumThreads);
        Assert.Equal(0, options.InterOpNumThreads);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_OptimizationLevelAll()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(GraphOptimizationLevel.All, options.OptimizationLevel);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_ProfilingDisabled()
    {
        var options = new OnnxModelOptions();
        Assert.False(options.EnableProfiling);
        Assert.Null(options.ProfileOutputPath);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_CustomOptionsEmpty()
    {
        var options = new OnnxModelOptions();
        Assert.Empty(options.CustomOptions);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_LogLevelWarning()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(OnnxLogLevel.Warning, options.LogLevel);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_AutoWarmUpDisabled()
    {
        var options = new OnnxModelOptions();
        Assert.False(options.AutoWarmUp);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_CudaMemoryLimitZero()
    {
        var options = new OnnxModelOptions();
        Assert.Equal(0, options.CudaMemoryLimit);
    }

    [Fact]
    public void OnnxModelOptions_Defaults_CudaArenaEnabled()
    {
        var options = new OnnxModelOptions();
        Assert.True(options.CudaUseArena);
    }

    // ============================
    // OnnxModelOptions: Factory Methods
    // ============================

    [Fact]
    public void OnnxModelOptions_ForCpu_SetsProviderAndEmptyFallbacks()
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
    public void OnnxModelOptions_ForCuda_SetsProviderAndFallback()
    {
        var options = OnnxModelOptions.ForCuda();
        Assert.Equal(OnnxExecutionProvider.Cuda, options.ExecutionProvider);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.Single(options.FallbackProviders);
        Assert.Equal(OnnxExecutionProvider.Cpu, options.FallbackProviders[0]);
    }

    [Fact]
    public void OnnxModelOptions_ForCuda_CustomDevice()
    {
        var options = OnnxModelOptions.ForCuda(deviceId: 2);
        Assert.Equal(2, options.GpuDeviceId);
    }

    [Fact]
    public void OnnxModelOptions_ForDirectML_SetsProviderAndFallback()
    {
        var options = OnnxModelOptions.ForDirectML();
        Assert.Equal(OnnxExecutionProvider.DirectML, options.ExecutionProvider);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.Single(options.FallbackProviders);
        Assert.Equal(OnnxExecutionProvider.Cpu, options.FallbackProviders[0]);
    }

    [Fact]
    public void OnnxModelOptions_ForTensorRT_SetsProviderAndFallbacks()
    {
        var options = OnnxModelOptions.ForTensorRT();
        Assert.Equal(OnnxExecutionProvider.TensorRT, options.ExecutionProvider);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.Equal(2, options.FallbackProviders.Count);
        Assert.Equal(OnnxExecutionProvider.Cuda, options.FallbackProviders[0]);
        Assert.Equal(OnnxExecutionProvider.Cpu, options.FallbackProviders[1]);
    }

    [Fact]
    public void OnnxModelOptions_ForTensorRT_CustomDevice()
    {
        var options = OnnxModelOptions.ForTensorRT(deviceId: 1);
        Assert.Equal(1, options.GpuDeviceId);
    }

    // ============================
    // OnnxModelOptions: Custom Properties
    // ============================

    [Fact]
    public void OnnxModelOptions_CustomValues_AllSet()
    {
        var options = new OnnxModelOptions
        {
            ExecutionProvider = OnnxExecutionProvider.Cuda,
            GpuDeviceId = 1,
            EnableMemoryPattern = false,
            EnableMemoryArena = false,
            IntraOpNumThreads = 8,
            InterOpNumThreads = 4,
            OptimizationLevel = GraphOptimizationLevel.Extended,
            EnableProfiling = true,
            ProfileOutputPath = "/tmp/profile",
            LogLevel = OnnxLogLevel.Verbose,
            AutoWarmUp = true,
            CudaMemoryLimit = 4L * 1024 * 1024 * 1024,
            CudaUseArena = false
        };

        Assert.Equal(OnnxExecutionProvider.Cuda, options.ExecutionProvider);
        Assert.Equal(1, options.GpuDeviceId);
        Assert.False(options.EnableMemoryPattern);
        Assert.False(options.EnableMemoryArena);
        Assert.Equal(8, options.IntraOpNumThreads);
        Assert.Equal(4, options.InterOpNumThreads);
        Assert.Equal(GraphOptimizationLevel.Extended, options.OptimizationLevel);
        Assert.True(options.EnableProfiling);
        Assert.Equal("/tmp/profile", options.ProfileOutputPath);
        Assert.Equal(OnnxLogLevel.Verbose, options.LogLevel);
        Assert.True(options.AutoWarmUp);
        Assert.Equal(4L * 1024 * 1024 * 1024, options.CudaMemoryLimit);
        Assert.False(options.CudaUseArena);
    }

    // ============================
    // OnnxExecutionProvider Enum
    // ============================

    [Fact]
    public void OnnxExecutionProvider_HasNineValues()
    {
        var values = Enum.GetValues<OnnxExecutionProvider>();
        Assert.Equal(9, values.Length);
    }

    [Theory]
    [InlineData(OnnxExecutionProvider.Cpu, 0)]
    [InlineData(OnnxExecutionProvider.Cuda, 1)]
    [InlineData(OnnxExecutionProvider.TensorRT, 2)]
    [InlineData(OnnxExecutionProvider.DirectML, 3)]
    [InlineData(OnnxExecutionProvider.CoreML, 4)]
    [InlineData(OnnxExecutionProvider.OpenVINO, 5)]
    [InlineData(OnnxExecutionProvider.ROCm, 6)]
    [InlineData(OnnxExecutionProvider.NNAPI, 7)]
    [InlineData(OnnxExecutionProvider.Auto, 100)]
    public void OnnxExecutionProvider_Values(OnnxExecutionProvider provider, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)provider);
    }

    // ============================
    // GraphOptimizationLevel Enum
    // ============================

    [Fact]
    public void GraphOptimizationLevel_HasFourValues()
    {
        var values = Enum.GetValues<GraphOptimizationLevel>();
        Assert.Equal(4, values.Length);
    }

    [Theory]
    [InlineData(GraphOptimizationLevel.None, 0)]
    [InlineData(GraphOptimizationLevel.Basic, 1)]
    [InlineData(GraphOptimizationLevel.Extended, 2)]
    [InlineData(GraphOptimizationLevel.All, 99)]
    public void GraphOptimizationLevel_Values(GraphOptimizationLevel level, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)level);
    }

    // ============================
    // OnnxLogLevel Enum
    // ============================

    [Fact]
    public void OnnxLogLevel_HasFiveValues()
    {
        var values = Enum.GetValues<OnnxLogLevel>();
        Assert.Equal(5, values.Length);
    }

    [Theory]
    [InlineData(OnnxLogLevel.Verbose, 0)]
    [InlineData(OnnxLogLevel.Info, 1)]
    [InlineData(OnnxLogLevel.Warning, 2)]
    [InlineData(OnnxLogLevel.Error, 3)]
    [InlineData(OnnxLogLevel.Fatal, 4)]
    public void OnnxLogLevel_Values(OnnxLogLevel level, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)level);
    }
}
