// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// Proves the unified <see cref="ContinuousBatchingRequestBatcher"/> performs genuine batched
/// inference: multiple queued requests are served by a single shared model forward pass
/// (<c>PredictBatch</c>), not one <c>Predict</c> per request. Before the unification this batcher
/// ran a separate per-request forward, so it never actually batched despite its name.
/// </summary>
public class ContinuousBatchingRequestBatcherTests
{
    private static IOptions<ServingOptions> BuildOptions()
        => Options.Create(new ServingOptions
        {
            MaxBatchSize = 32,
            // Large window => the idle loop sleeps long enough for all requests to be enqueued
            // before the next drain, so they are admitted as a single batch deterministically.
            BatchingWindowMs = 20000,
            AdaptiveBatchSize = false,
            MaxQueueSize = 1000,
            EnablePerformanceMetrics = true
        });

    [Fact(Timeout = 120000)]
    public async Task QueueRequest_SharesOneForwardPass_AcrossManyRequests()
    {
        int callCount = 0;
        int maxBatchRows = 0;
        var sync = new object();

        // Records each batched forward and returns row*2 so per-request scatter can be verified.
        Matrix<float> PredictBatch(Matrix<float> inputs)
        {
            lock (sync)
            {
                callCount++;
                if (inputs.Rows > maxBatchRows)
                {
                    maxBatchRows = inputs.Rows;
                }
            }

            var outputs = new Matrix<float>(inputs.Rows, inputs.Columns);
            for (int i = 0; i < inputs.Rows; i++)
            {
                for (int j = 0; j < inputs.Columns; j++)
                {
                    outputs[i, j] = inputs[i, j] * 2f;
                }
            }
            return outputs;
        }

        var model = new ServableModelWrapper<float>(
            "m",
            inputDimension: 2,
            outputDimension: 2,
            predictFunc: v => v,
            predictBatchFunc: PredictBatch);

        var repo = new SingleModelRepository("m", model);
        using var batcher = new ContinuousBatchingRequestBatcher(
            repo, NullLogger<ContinuousBatchingRequestBatcher>.Instance, BuildOptions());

        const int n = 8;
        var tasks = new Task<Vector<float>>[n];
        for (int i = 0; i < n; i++)
        {
            tasks[i] = batcher.QueueRequest("m", new Vector<float>(new[] { (float)i, (float)(i + 1) }));
        }

        var results = await Task.WhenAll(tasks);

        // Every request gets its own correct result (row*2), proving correct scatter.
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(2, results[i].Length);
            Assert.Equal(i * 2f, results[i][0]);
            Assert.Equal((i + 1) * 2f, results[i][1]);
        }

        // The forward pass was actually batched: at least one call carried multiple rows, and the
        // total number of forward passes is fewer than the number of requests.
        Assert.True(maxBatchRows > 1, $"expected a batched forward but max rows was {maxBatchRows}");
        Assert.True(callCount < n, $"expected fewer forward passes than requests but got {callCount}");
    }

    [Fact(Timeout = 120000)]
    public async Task QueueRequest_PropagatesError_WhenModelMissing()
    {
        var repo = new SingleModelRepository("m", model: null);
        using var batcher = new ContinuousBatchingRequestBatcher(
            repo, NullLogger<ContinuousBatchingRequestBatcher>.Instance, BuildOptions());

        await Assert.ThrowsAnyAsync<InvalidOperationException>(
            () => batcher.QueueRequest("missing", new Vector<float>(new[] { 1f, 2f })));
    }

    /// <summary>Repository that serves at most one model by name.</summary>
    private sealed class SingleModelRepository : IModelRepository
    {
        private readonly string _name;
        private readonly object? _model;

        public SingleModelRepository(string name, object? model)
        {
            _name = name;
            _model = model;
        }

        public IServableModel<T>? GetModel<T>(string name)
            => name == _name && _model is IServableModel<T> typed ? typed : null;

        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public bool UnloadModel(string name) => throw new NotSupportedException();
        public List<ModelInfo> GetAllModelInfo() => [];
        public ModelInfo? GetModelInfo(string name) => null;
        public bool ModelExists(string name) => name == _name && _model is not null;
        public bool LoadModelFromRegistry<T>(string name, IServableModel<T> model, int registryVersion, string registryStage, string? sourcePath = null) => throw new NotSupportedException();
        public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name) => throw new NotSupportedException();
    }
}
