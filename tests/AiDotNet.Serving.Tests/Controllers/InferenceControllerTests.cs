using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Controllers;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Tests.Controllers;

public sealed class InferenceControllerTests
{
    [Fact(Timeout = 60000)]
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

    [Fact(Timeout = 60000)]
    public async Task Predict_ReturnsBadRequest_WhenFeaturesMissing()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(new ModelInfo { Name = "m", InputDimension = 1, OutputDimension = 1, NumericType = NumericType.Double }),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = await controller.Predict("m", new PredictionRequest { Features = [] });

        Assert.IsType<BadRequestObjectResult>(result);
    }

    [Fact(Timeout = 60000)]
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

    [Fact(Timeout = 60000)]
    public async Task GetStatistics_ReturnsOk()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(modelInfo: null),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GetStatistics();

        var ok = Assert.IsType<OkObjectResult>(result.Result);
        Assert.IsType<Dictionary<string, object>>(ok.Value);
    }

    [Fact(Timeout = 60000)]
    public async Task GetPerformanceMetrics_ReturnsOk()
    {
        var controller = CreateController(
            modelRepository: new FakeModelRepository(modelInfo: null),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GetPerformanceMetrics();

        var ok = Assert.IsType<OkObjectResult>(result.Result);
        Assert.IsType<Dictionary<string, object>>(ok.Value);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerateWithSpeculativeDecoding_ReturnsNotFound_WhenModelMissing()
    {
        await Task.Yield();
        var controller = CreateController(
            modelRepository: new FakeModelRepository(modelInfo: null),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GenerateWithSpeculativeDecoding("missing", new SpeculativeDecodingRequest
        {
            InputTokens = [1, 2, 3],
            MaxNewTokens = 4
        });

        Assert.IsType<NotFoundObjectResult>(result);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerateWithSpeculativeDecoding_ReturnsBadRequest_WhenModelNotGenerative()
    {
        await Task.Yield();
        // A regression/prediction model (no token-level forward) cannot generate text.
        var nonGenerative = new ServableModelWrapper<double>("m", inputDimension: 1, outputDimension: 1, predictFunc: v => v);
        var controller = CreateController(
            modelRepository: new FakeModelRepository(
                new ModelInfo { Name = "m", InputDimension = 1, OutputDimension = 1, NumericType = NumericType.Double },
                nonGenerative),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GenerateWithSpeculativeDecoding("m", new SpeculativeDecodingRequest
        {
            InputTokens = [1, 2, 3],
            MaxNewTokens = 4
        });

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var response = Assert.IsType<SpeculativeDecodingResponse>(badRequest.Value);
        Assert.Contains("does not support text generation", response.Error);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerateWithSpeculativeDecoding_ReturnsGeneratedTokens_ForGenerativeModel()
    {
        await Task.Yield();
        const int vocab = 10;
        const int peakToken = 7;

        // A token-to-logits model that always peaks at peakToken (deterministic generation).
        Func<Tensor<double>, Tensor<double>> forward = input =>
        {
            int seqLen = input.Shape[input.Shape.Length - 1];
            var logits = new Tensor<double>(new[] { 1, seqLen, vocab });
            for (int p = 0; p < seqLen; p++)
            {
                logits[new[] { 0, p, peakToken }] = 100.0;
            }
            return logits;
        };

        var generative = new ServableModelWrapper<double>(
            "lm", inputDimension: 1, outputDimension: vocab, predictFunc: v => v, generationForward: forward);

        var controller = CreateController(
            modelRepository: new FakeModelRepository(
                new ModelInfo { Name = "lm", InputDimension = 1, OutputDimension = vocab, NumericType = NumericType.Double },
                generative),
            requestBatcher: new FakeRequestBatcher(Vector<double>.CreateDefault(1, 0d)));

        var result = controller.GenerateWithSpeculativeDecoding("lm", new SpeculativeDecodingRequest
        {
            InputTokens = [1, 5],
            MaxNewTokens = 3,
            EosTokenId = 2,
            NumDraftTokens = 2
        });

        var ok = Assert.IsType<OkObjectResult>(result);
        var response = Assert.IsType<SpeculativeDecodingResponse>(ok.Value);
        Assert.Null(response.Error);
        Assert.Equal(3, response.NumGenerated);
        Assert.All(response.GeneratedTokens, t => Assert.Equal(peakToken, t));
        Assert.Equal(new[] { 1, 5, 7, 7, 7 }, response.AllTokens);
    }

    private static InferenceController CreateController(IModelRepository modelRepository, IRequestBatcher requestBatcher)
    {
        var textGenerationService = new TextGenerationService(modelRepository, NullLogger<TextGenerationService>.Instance);
        return new InferenceController(modelRepository, requestBatcher, textGenerationService, NullLogger<InferenceController>.Instance);
    }

    private sealed class FakeModelRepository : IModelRepository
    {
        private readonly ModelInfo? _modelInfo;
        private readonly object? _model;

        public FakeModelRepository(ModelInfo? modelInfo, object? model = null)
        {
            _modelInfo = modelInfo;
            _model = model;
        }

        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => throw new NotSupportedException();

        public IServableModel<T>? GetModel<T>(string name)
            => _modelInfo is not null && string.Equals(_modelInfo.Name, name, StringComparison.Ordinal) && _model is IServableModel<T> typed
                ? typed
                : null;

        public bool UnloadModel(string name) => throw new NotSupportedException();

        public List<ModelInfo> GetAllModelInfo() => _modelInfo is null ? [] : [_modelInfo];

        public ModelInfo? GetModelInfo(string name) => _modelInfo is not null && string.Equals(_modelInfo.Name, name, StringComparison.Ordinal) ? _modelInfo : null;

        public bool ModelExists(string name) => GetModelInfo(name) is not null;

        public bool LoadModelFromRegistry<T>(string name, IServableModel<T> model, int registryVersion, string registryStage, string? sourcePath = null)
            => throw new NotSupportedException();

        public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null)
            => throw new NotSupportedException();

        public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name)
            => throw new NotSupportedException();
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
