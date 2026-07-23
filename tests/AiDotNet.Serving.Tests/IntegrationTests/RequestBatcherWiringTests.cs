// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// Proves the serving entry point (<see cref="Program.AddConfiguredRequestBatcher"/>) wires the
/// configured <see cref="BatchingStrategyType"/> to the correct <see cref="IRequestBatcher"/>
/// implementation. The vLLM-style <see cref="ContinuousBatchingRequestBatcher"/> was fully built
/// but unreachable before this wiring — only the static <see cref="RequestBatcher"/> was registered,
/// regardless of configuration. These tests exercise the real DI registration used by Main, not a
/// duplicate of the selection logic.
/// </summary>
public class RequestBatcherWiringTests
{
    private static ServiceProvider BuildProvider(BatchingStrategyType strategy)
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddSingleton<IModelRepository>(new EmptyModelRepository());
        services.AddSingleton<IOptions<ServingOptions>>(
            Options.Create(new ServingOptions { BatchingStrategy = strategy }));

        // The exact registration Main uses.
        Program.AddConfiguredRequestBatcher(services);

        return services.BuildServiceProvider();
    }

    [Fact(Timeout = 120000)]
    public async Task ContinuousStrategy_Resolves_ContinuousBatchingRequestBatcher()
    {
        await Task.Yield();
        using var provider = BuildProvider(BatchingStrategyType.Continuous);
        var batcher = provider.GetRequiredService<IRequestBatcher>();
        Assert.IsType<ContinuousBatchingRequestBatcher>(batcher);
    }

    [Theory]
    [InlineData(BatchingStrategyType.Adaptive)]
    [InlineData(BatchingStrategyType.Timeout)]
    [InlineData(BatchingStrategyType.Size)]
    [InlineData(BatchingStrategyType.Bucket)]
    public void NonContinuousStrategies_Resolve_StaticRequestBatcher(BatchingStrategyType strategy)
    {
        using var provider = BuildProvider(strategy);
        var batcher = provider.GetRequiredService<IRequestBatcher>();
        Assert.IsType<RequestBatcher>(batcher);
    }

    [Fact(Timeout = 120000)]
    public async Task Batcher_IsRegistered_AsSingleton()
    {
        await Task.Yield();
        using var provider = BuildProvider(BatchingStrategyType.Continuous);
        var first = provider.GetRequiredService<IRequestBatcher>();
        var second = provider.GetRequiredService<IRequestBatcher>();
        Assert.Same(first, second);
    }

    /// <summary>Minimal repository so the batchers can be constructed via ActivatorUtilities.</summary>
    private sealed class EmptyModelRepository : IModelRepository
    {
        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public IServableModel<T>? GetModel<T>(string name) => throw new NotSupportedException();
        public bool UnloadModel(string name) => throw new NotSupportedException();
        public List<ModelInfo> GetAllModelInfo() => [];
        public ModelInfo? GetModelInfo(string name) => null;
        public bool ModelExists(string name) => false;
        public bool LoadModelFromRegistry<T>(string name, IServableModel<T> model, int registryVersion, string registryStage, string? sourcePath = null) => throw new NotSupportedException();
        public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name) => throw new NotSupportedException();
    }
}
