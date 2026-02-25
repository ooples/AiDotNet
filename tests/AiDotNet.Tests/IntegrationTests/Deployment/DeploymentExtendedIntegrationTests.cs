using AiDotNet.Deployment.Configuration;
using AiDotNet.Deployment.Edge;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Deployment.Mobile.Android;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Deployment.Optimization.Quantization.Calibration;
using AiDotNet.Deployment.Runtime;
using AiDotNet.Deployment.TensorRT;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Deployment;

/// <summary>
/// Extended integration tests for Deployment module covering untested classes:
/// DeploymentConfiguration, OnnxGraph/Node/Operation, PartitionedModel,
/// AdaptiveInferenceConfig, quantizer formats, calibration, TensorRT, and mobile.
/// </summary>
public class DeploymentExtendedIntegrationTests
{
    #region DeploymentConfiguration

    [Fact]
    public void DeploymentConfiguration_DefaultValues_AllNull()
    {
        var config = new DeploymentConfiguration();

        Assert.Null(config.Quantization);
        Assert.Null(config.Caching);
        Assert.Null(config.Versioning);
        Assert.Null(config.ABTesting);
        Assert.Null(config.Telemetry);
        Assert.Null(config.Export);
        Assert.Null(config.GpuAcceleration);
        Assert.Null(config.Compression);
        Assert.Null(config.Profiling);
    }

    [Fact]
    public void DeploymentConfiguration_Create_SetsAllProperties()
    {
        var quant = new QuantizationConfig();
        var cache = new CacheConfig();
        var versioning = new VersioningConfig();
        var abTesting = new ABTestingConfig();
        var telemetry = new TelemetryConfig();
        var export = new ExportConfig();
        var compression = new CompressionConfig();
        var profiling = new ProfilingConfig();

        var config = DeploymentConfiguration.Create(
            quantization: quant,
            caching: cache,
            versioning: versioning,
            abTesting: abTesting,
            telemetry: telemetry,
            export: export,
            gpuAcceleration: null,
            compression: compression,
            profiling: profiling);

        Assert.Same(quant, config.Quantization);
        Assert.Same(cache, config.Caching);
        Assert.Same(versioning, config.Versioning);
        Assert.Same(abTesting, config.ABTesting);
        Assert.Same(telemetry, config.Telemetry);
        Assert.Same(export, config.Export);
        Assert.Null(config.GpuAcceleration);
        Assert.Same(compression, config.Compression);
        Assert.Same(profiling, config.Profiling);
    }

    #endregion

    #region OnnxGraph

    [Fact]
    public void OnnxGraph_DefaultValues()
    {
        var graph = new OnnxGraph();

        Assert.Equal(string.Empty, graph.Name);
        Assert.Equal(13, graph.OpsetVersion);
        Assert.NotNull(graph.Inputs);
        Assert.Empty(graph.Inputs);
        Assert.NotNull(graph.Outputs);
        Assert.Empty(graph.Outputs);
        Assert.NotNull(graph.Operations);
        Assert.Empty(graph.Operations);
        Assert.NotNull(graph.Initializers);
        Assert.Empty(graph.Initializers);
    }

    [Fact]
    public void OnnxGraph_AddInputsOutputsOperations()
    {
        var graph = new OnnxGraph { Name = "TestGraph", OpsetVersion = 15 };

        graph.Inputs.Add(new OnnxNode
        {
            Name = "input",
            DataType = "float",
            Shape = new[] { 1, 3, 224, 224 }
        });
        graph.Outputs.Add(new OnnxNode
        {
            Name = "output",
            DataType = "float",
            Shape = new[] { 1, 1000 }
        });
        graph.Operations.Add(new OnnxOperation
        {
            Type = "Conv",
            Name = "conv1"
        });

        Assert.Equal("TestGraph", graph.Name);
        Assert.Equal(15, graph.OpsetVersion);
        Assert.Single(graph.Inputs);
        Assert.Single(graph.Outputs);
        Assert.Single(graph.Operations);
        Assert.Equal("input", graph.Inputs[0].Name);
    }

    #endregion

    #region OnnxNode

    [Fact]
    public void OnnxNode_DefaultValues()
    {
        var node = new OnnxNode();

        Assert.Equal(string.Empty, node.Name);
        Assert.Equal("float", node.DataType);
        Assert.Null(node.Shape);
        Assert.Null(node.DocString);
    }

    [Fact]
    public void OnnxNode_SetProperties()
    {
        var node = new OnnxNode
        {
            Name = "features",
            DataType = "double",
            Shape = new[] { 1, 512 },
            DocString = "Feature vector"
        };

        Assert.Equal("features", node.Name);
        Assert.Equal("double", node.DataType);
        Assert.Equal(new[] { 1, 512 }, node.Shape);
        Assert.Equal("Feature vector", node.DocString);
    }

    #endregion

    #region OnnxOperation

    [Fact]
    public void OnnxOperation_DefaultValues()
    {
        var op = new OnnxOperation();

        Assert.Equal(string.Empty, op.Type);
        Assert.NotNull(op.Inputs);
        Assert.Empty(op.Inputs);
        Assert.NotNull(op.Outputs);
        Assert.Empty(op.Outputs);
        Assert.NotNull(op.Attributes);
        Assert.Empty(op.Attributes);
        Assert.Null(op.Name);
        Assert.Equal("ai.onnx", op.Domain);
    }

    [Fact]
    public void OnnxOperation_SetProperties()
    {
        var op = new OnnxOperation
        {
            Type = "Conv",
            Name = "conv_0",
            Domain = "com.custom"
        };
        op.Inputs.Add("input");
        op.Inputs.Add("weight");
        op.Outputs.Add("output");
        op.Attributes["kernel_shape"] = new[] { 3, 3 };
        op.Attributes["strides"] = new[] { 1, 1 };

        Assert.Equal("Conv", op.Type);
        Assert.Equal("conv_0", op.Name);
        Assert.Equal("com.custom", op.Domain);
        Assert.Equal(2, op.Inputs.Count);
        Assert.Single(op.Outputs);
        Assert.Equal(2, op.Attributes.Count);
    }

    #endregion

    #region AdaptiveInferenceConfig

    [Fact]
    public void AdaptiveInferenceConfig_DefaultValues()
    {
        var config = new AdaptiveInferenceConfig();

        Assert.Equal(default(QualityLevel), config.QualityLevel);
        Assert.False(config.UseQuantization);
        Assert.Equal(0, config.QuantizationBits);
        Assert.NotNull(config.SkipLayers);
        Assert.Empty(config.SkipLayers);
    }

    [Fact]
    public void AdaptiveInferenceConfig_SetProperties()
    {
        var config = new AdaptiveInferenceConfig
        {
            QualityLevel = QualityLevel.High,
            UseQuantization = true,
            QuantizationBits = 8
        };
        config.SkipLayers.Add("attention_layer_12");
        config.SkipLayers.Add("attention_layer_11");

        Assert.Equal(QualityLevel.High, config.QualityLevel);
        Assert.True(config.UseQuantization);
        Assert.Equal(8, config.QuantizationBits);
        Assert.Equal(2, config.SkipLayers.Count);
    }

    #endregion

    #region LayerQuantizationParams

    [Fact]
    public void LayerQuantizationParams_DefaultValues()
    {
        var param = new LayerQuantizationParams();

        Assert.Equal(1.0, param.ScaleFactor);
        Assert.Equal(0, param.ZeroPoint);
        Assert.False(param.Skip);
        Assert.Null(param.BitWidth);
        Assert.Null(param.Mode);
    }

    [Fact]
    public void LayerQuantizationParams_SetProperties()
    {
        var param = new LayerQuantizationParams
        {
            ScaleFactor = 0.125,
            ZeroPoint = 128,
            Skip = true,
            BitWidth = 4,
            Mode = QuantizationMode.Int8
        };

        Assert.Equal(0.125, param.ScaleFactor);
        Assert.Equal(128, param.ZeroPoint);
        Assert.True(param.Skip);
        Assert.Equal(4, param.BitWidth);
        Assert.Equal(QuantizationMode.Int8, param.Mode);
    }

    #endregion

    #region CacheStatistics

    [Fact]
    public void CacheStatistics_DefaultValues()
    {
        var stats = new CacheStatistics();

        Assert.Equal(0, stats.TotalEntries);
        Assert.Equal(0L, stats.TotalAccessCount);
        Assert.Equal(0.0, stats.AverageAccessCount);
        Assert.Equal(TimeSpan.Zero, stats.OldestEntryAge);
        Assert.Equal(TimeSpan.Zero, stats.NewestEntryAge);
    }

    [Fact]
    public void CacheStatistics_SetProperties()
    {
        var stats = new CacheStatistics
        {
            TotalEntries = 50,
            TotalAccessCount = 1000,
            AverageAccessCount = 20.0,
            OldestEntryAge = TimeSpan.FromHours(2),
            NewestEntryAge = TimeSpan.FromMinutes(5)
        };

        Assert.Equal(50, stats.TotalEntries);
        Assert.Equal(1000L, stats.TotalAccessCount);
        Assert.Equal(20.0, stats.AverageAccessCount);
        Assert.Equal(TimeSpan.FromHours(2), stats.OldestEntryAge);
        Assert.Equal(TimeSpan.FromMinutes(5), stats.NewestEntryAge);
    }

    #endregion

    #region InferenceStatistics

    [Fact]
    public void InferenceStatistics_DefaultValues()
    {
        var stats = new InferenceStatistics();

        Assert.Equal(0, stats.NumStreams);
        Assert.Equal(0, stats.AvailableStreams);
        Assert.Equal(0, stats.ActiveStreams);
    }

    [Fact]
    public void InferenceStatistics_SetProperties()
    {
        var stats = new InferenceStatistics
        {
            NumStreams = 4,
            AvailableStreams = 2,
            ActiveStreams = 2
        };

        Assert.Equal(4, stats.NumStreams);
        Assert.Equal(2, stats.AvailableStreams);
        Assert.Equal(2, stats.ActiveStreams);
    }

    #endregion

    #region NNAPIPerformanceInfo

    [Fact]
    public void NNAPIPerformanceInfo_DefaultValues()
    {
        var info = new NNAPIPerformanceInfo();

        Assert.NotNull(info.SupportedOperations);
        Assert.Empty(info.SupportedOperations);
        Assert.Equal(string.Empty, info.PreferredDevice);
        Assert.False(info.SupportsInt8);
        Assert.False(info.SupportsFp16);
        Assert.False(info.SupportsRelaxedFp32);
    }

    [Fact]
    public void NNAPIPerformanceInfo_SetProperties()
    {
        var info = new NNAPIPerformanceInfo
        {
            PreferredDevice = "GPU",
            SupportsInt8 = true,
            SupportsFp16 = true,
            SupportsRelaxedFp32 = true
        };
        info.SupportedOperations.Add("CONV_2D");
        info.SupportedOperations.Add("FULLY_CONNECTED");
        info.SupportedOperations.Add("RELU");

        Assert.Equal("GPU", info.PreferredDevice);
        Assert.True(info.SupportsInt8);
        Assert.True(info.SupportsFp16);
        Assert.True(info.SupportsRelaxedFp32);
        Assert.Equal(3, info.SupportedOperations.Count);
    }

    #endregion

    #region ActivationStatistics

    [Fact]
    public void ActivationStatistics_DefaultValues()
    {
        var stats = new ActivationStatistics<double>();

        Assert.NotNull(stats.LayerStats);
        Assert.Empty(stats.LayerStats);
        Assert.Null(stats.GlobalActivationMagnitudes);
        Assert.Null(stats.GlobalMaxAbsActivations);
        Assert.Equal(0, stats.SampleCount);
        Assert.False(stats.IsFromRealForwardPasses);
        Assert.NotNull(stats.CalibrationWarnings);
        Assert.Empty(stats.CalibrationWarnings);
    }

    [Fact]
    public void ActivationStatistics_SetProperties()
    {
        var stats = new ActivationStatistics<double>
        {
            SampleCount = 100,
            IsFromRealForwardPasses = true,
            GlobalActivationMagnitudes = new double[] { 0.5, 0.8, 0.3 },
            GlobalMaxAbsActivations = new double[] { 2.5, 3.0, 1.8 }
        };
        stats.CalibrationWarnings.Add("Layer 3 has high variance");

        Assert.Equal(100, stats.SampleCount);
        Assert.True(stats.IsFromRealForwardPasses);
        Assert.Equal(3, stats.GlobalActivationMagnitudes.Length);
        Assert.Single(stats.CalibrationWarnings);
    }

    [Fact]
    public void LayerActivationStats_DefaultValues()
    {
        var stats = new LayerActivationStats<double>();

        Assert.Equal(string.Empty, stats.LayerName);
        Assert.Equal(double.MaxValue, stats.MinValue);
        Assert.Equal(double.MinValue, stats.MaxValue);
        Assert.Equal(0.0, stats.MaxAbsValue);
        Assert.Equal(0.0, stats.Mean);
        Assert.Equal(0.0, stats.Variance);
        Assert.Null(stats.PerChannelMaxAbs);
        Assert.Equal(0, stats.SampleCount);
        Assert.Equal(0.0, stats.StandardDeviation);
    }

    [Fact]
    public void LayerActivationStats_UpdateWithTensor()
    {
        var stats = new LayerActivationStats<double> { LayerName = "conv1" };
        var tensor = new Tensor<double>(new[] { 2, 3 });

        // Set some values in the tensor
        tensor[0, 0] = 1.0;
        tensor[0, 1] = -2.0;
        tensor[0, 2] = 3.0;
        tensor[1, 0] = -1.0;
        tensor[1, 1] = 0.5;
        tensor[1, 2] = 2.0;

        stats.Update(tensor);

        Assert.Equal("conv1", stats.LayerName);
        Assert.Equal(-2.0, stats.MinValue);
        Assert.Equal(3.0, stats.MaxValue);
        Assert.Equal(3.0, stats.MaxAbsValue);
        Assert.Equal(6, stats.SampleCount);
        Assert.True(stats.StandardDeviation > 0);
    }

    [Fact]
    public void LayerActivationStats_StandardDeviation_SingleSample()
    {
        var stats = new LayerActivationStats<double>();
        var tensor = new Tensor<double>(new[] { 1 });
        tensor[0] = 5.0;

        stats.Update(tensor);

        Assert.Equal(1, stats.SampleCount);
        Assert.Equal(0.0, stats.StandardDeviation);
    }

    [Fact]
    public void ActivationStatistics_AddLayerStats()
    {
        var stats = new ActivationStatistics<double>();
        var layerStats = new LayerActivationStats<double> { LayerName = "fc1" };

        stats.LayerStats["fc1"] = layerStats;

        Assert.Single(stats.LayerStats);
        Assert.Equal("fc1", stats.LayerStats["fc1"].LayerName);
    }

    #endregion

    #region TensorRT Enums

    [Fact]
    public void TensorRTPrecision_HasExpectedValues()
    {
        Assert.Equal(0, (int)TensorRTPrecision.FP32);
        Assert.Equal(1, (int)TensorRTPrecision.FP16);
        Assert.Equal(2, (int)TensorRTPrecision.INT8);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void Integration_BuildOnnxGraph_WithMultipleOperations()
    {
        var graph = new OnnxGraph { Name = "ResNet18", OpsetVersion = 15 };

        graph.Inputs.Add(new OnnxNode
        {
            Name = "data",
            DataType = "float",
            Shape = new[] { 1, 3, 224, 224 }
        });

        graph.Operations.Add(new OnnxOperation
        {
            Type = "Conv",
            Name = "conv1"
        });
        graph.Operations[0].Inputs.Add("data");
        graph.Operations[0].Inputs.Add("conv1_weight");
        graph.Operations[0].Outputs.Add("conv1_out");
        graph.Operations[0].Attributes["kernel_shape"] = new[] { 7, 7 };

        graph.Operations.Add(new OnnxOperation
        {
            Type = "BatchNormalization",
            Name = "bn1"
        });
        graph.Operations[1].Inputs.Add("conv1_out");
        graph.Operations[1].Outputs.Add("bn1_out");

        graph.Operations.Add(new OnnxOperation
        {
            Type = "Relu",
            Name = "relu1"
        });
        graph.Operations[2].Inputs.Add("bn1_out");
        graph.Operations[2].Outputs.Add("relu1_out");

        graph.Outputs.Add(new OnnxNode
        {
            Name = "output",
            DataType = "float",
            Shape = new[] { 1, 1000 }
        });

        Assert.Equal(3, graph.Operations.Count);
        Assert.Equal("Conv", graph.Operations[0].Type);
        Assert.Equal("BatchNormalization", graph.Operations[1].Type);
        Assert.Equal("Relu", graph.Operations[2].Type);
    }

    [Fact]
    public void Integration_DeploymentConfig_FullPipeline()
    {
        var config = DeploymentConfiguration.Create(
            quantization: new QuantizationConfig(),
            caching: new CacheConfig(),
            versioning: new VersioningConfig(),
            abTesting: new ABTestingConfig(),
            telemetry: new TelemetryConfig(),
            export: new ExportConfig(),
            gpuAcceleration: null,
            compression: new CompressionConfig(),
            profiling: new ProfilingConfig());

        Assert.NotNull(config.Quantization);
        Assert.NotNull(config.Caching);
        Assert.NotNull(config.Versioning);
        Assert.NotNull(config.ABTesting);
        Assert.NotNull(config.Telemetry);
        Assert.NotNull(config.Export);
        Assert.NotNull(config.Compression);
        Assert.NotNull(config.Profiling);
        Assert.Null(config.GpuAcceleration);
    }

    [Fact]
    public void Integration_PerLayerQuantization_Config()
    {
        var layers = new Dictionary<string, LayerQuantizationParams>
        {
            ["conv1"] = new LayerQuantizationParams
            {
                ScaleFactor = 0.05,
                ZeroPoint = 128,
                BitWidth = 8
            },
            ["fc1"] = new LayerQuantizationParams
            {
                ScaleFactor = 0.01,
                ZeroPoint = 0,
                BitWidth = 4,
                Mode = QuantizationMode.Float16
            },
            ["embedding"] = new LayerQuantizationParams
            {
                Skip = true
            }
        };

        Assert.Equal(3, layers.Count);
        Assert.Equal(8, layers["conv1"].BitWidth);
        Assert.True(layers["embedding"].Skip);
        Assert.Equal(QuantizationMode.Float16, layers["fc1"].Mode);
    }

    [Fact]
    public void Integration_ActivationStatistics_MultipleLayerUpdates()
    {
        var stats = new ActivationStatistics<double>
        {
            SampleCount = 50,
            IsFromRealForwardPasses = true
        };

        var layer1Stats = new LayerActivationStats<double> { LayerName = "conv1" };
        var tensor1 = new Tensor<double>(new[] { 2, 4 });
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                tensor1[r, c] = ((r * 4 + c) - 4.0) * 0.5;
            }
        }
        layer1Stats.Update(tensor1);

        var layer2Stats = new LayerActivationStats<double> { LayerName = "conv2" };
        var tensor2 = new Tensor<double>(new[] { 2, 4 });
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                tensor2[r, c] = ((r * 4 + c) - 3.0) * 0.25;
            }
        }
        layer2Stats.Update(tensor2);

        stats.LayerStats["conv1"] = layer1Stats;
        stats.LayerStats["conv2"] = layer2Stats;

        Assert.Equal(2, stats.LayerStats.Count);
        Assert.True(stats.LayerStats["conv1"].MaxAbsValue > 0);
        Assert.True(stats.LayerStats["conv2"].MaxAbsValue > 0);
        Assert.True(stats.LayerStats["conv1"].SampleCount > 0);
    }

    #endregion
}
