using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Controllers;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Serving.Tests.Controllers;

public sealed class InferenceControllerTests
{
    [Fact]
    public async Task Predict_ReturnsNotFound_WhenModelDoesNotExist()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(modelInfo: null),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = await controller.Predict("missing", new PredictionRequest
        {
            Features = [[1.0]]
        });

        Assert.IsType<NotFoundObjectResult>(result);
    }

    [Fact]
    public async Task Predict_ReturnsBadRequest_WhenFeaturesMissing()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(new ModelInfo { Name = "m", InputDimension = 1, OutputDimension = 1, NumericType = NumericType.Double }),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = await controller.Predict("m", new PredictionRequest { Features = [] });

        Assert.IsType<BadRequestObjectResult>(result);
    }

    [Fact]
    public async Task Predict_ReturnsOk_WithPredictions_ForDoubleModel()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(new ModelInfo { Name = "m", InputDimension = 2, OutputDimension = 2, NumericType = NumericType.Double }),
            requestBatcher: new FakeRequestBatcher(new Vector<double>(new[] { 10.0, 20.0 })));

        var result = await controller.Predict("m", new PredictionRequest
        {
            RequestId = "r1",
            Features = [[1.0, 2.0], [3.0, 4.0]]
        });

        var ok = Assert.IsType<OkObjectResult>(result);
        var response = Assert.IsType<PredictionResponse>(ok.Value);
        Assert.Equal("r1", response.RequestId);
        Assert.Equal(2, response.BatchSize);
        Assert.Equal(2, response.Predictions.Length);
        Assert.Equal(new[] { 10.0, 20.0 }, response.Predictions[0]);
        Assert.Equal(new[] { 10.0, 20.0 }, response.Predictions[1]);
    }

    [Fact]
    public void GetStatistics_ReturnsOk()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(modelInfo: null),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GetStatistics();

        var ok = Assert.IsType<OkObjectResult>(result.Result);
        Assert.IsType<Dictionary<string, object>>(ok.Value);
    }

    [Fact]
    public void GetPerformanceMetrics_ReturnsOk()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(modelInfo: null),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GetPerformanceMetrics();

        var ok = Assert.IsType<OkObjectResult>(result.Result);
        Assert.IsType<Dictionary<string, object>>(ok.Value);
    }

    [Fact]
    public void GenerateWithSpeculativeDecoding_ReturnsNotImplemented_WhenModelExists()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(new ModelInfo { Name = "m", InputDimension = 1, OutputDimension = 1, NumericType = NumericType.Double }),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GenerateWithSpeculativeDecoding("m", new SpeculativeDecodingRequest
        {
            InputTokens = [1, 2, 3],
            MaxNewTokens = 4
        });

        var objectResult = Assert.IsType<ObjectResult>(result);
        Assert.Equal(501, objectResult.StatusCode);
        Assert.IsType<SpeculativeDecodingResponse>(objectResult.Value);
    }

    private static InferenceController CreateController(IModelRepository modelRepository, IRequestBatcher requestBatcher)
    {
        return new InferenceController(modelRepository, requestBatcher, NullLogger<InferenceController>.Instance);
    }

    private sealed class FakeModelRepository : IModelRepository
    {
        private readonly ModelInfo? _modelInfo;

        public FakeModelRepository(ModelInfo? modelInfo)
        {
            _modelInfo = modelInfo;
        }

        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => throw new NotSupportedException();

        public IServableModel<T>? GetModel<T>(string name) => throw new NotSupportedException();

        public bool UnloadModel(string name) => throw new NotSupportedException();

        public List<ModelInfo> GetAllModelInfo() => _modelInfo is null ? [] : [_modelInfo];

        public ModelInfo? GetModelInfo(string name) => _modelInfo is not null && string.Equals(_modelInfo.Name, name, StringComparison.Ordinal) ? _modelInfo : null;

        public bool ModelExists(string name) => GetModelInfo(name) is not null;

        public bool LoadModelFromRegistry<T>(string name, IServableModel<T> model, int registryVersion, string registryStage, string? sourcePath = null)
            => throw new NotSupportedException();

        public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null)
            => throw new NotSupportedException();

        public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name) => null;
    }

    private sealed class FakeRequestBatcher : IRequestBatcher
    {
        private readonly object _resultVector;

        public FakeRequestBatcher(object resultVector)
        {
            _resultVector = resultVector;
        }

        public Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input, Serving.Scheduling.RequestPriority priority = Serving.Scheduling.RequestPriority.Normal)
        {
            return Task.FromResult((Vector<T>)_resultVector);
        }

        public Dictionary<string, object> GetStatistics() => new() { { "ok", true } };

        public Dictionary<string, object> GetPerformanceMetrics() => new() { { "ok", true } };
    }
}
