using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 12 — Configure* methods for distributed / federated /
/// pipeline-parallel training. Each test verifies the configured
/// value reaches the matching consumer inside
/// BuildSupervisedInternalAsync.
/// </summary>
[Collection("ConfigureMethodCoverage")]
public class Bucket12_DistributedTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket12_DistributedTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureDistributedTraining — verifies the distributed branch at
    /// <c>AiModelBuilder.cs:2573</c> detects the configured backend and
    /// wraps the user's model with an <see cref="IShardedModel{T,TInput,TOutput}"/>
    /// (DDPModel for the DDP strategy). Stored-but-not-consumed would
    /// leave <c>result.Model</c> as the original unwrapped Transformer.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureDistributedTraining_DDP_WrapsModelAsShardedModel()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1);

        // Observable side-effect strategy: instrument the
        // InMemoryCommunicationBackend by subclassing it to record
        // whether it was wired into a DDP wrapper context. The
        // distributed dispatch switch at AiModelBuilder.cs:2595
        // constructs DDPModel(_model, shardingConfig) using the user's
        // backend; if we can prove the backend was consulted during
        // construction, the wrapping fired.
        //
        // The cleanest observable: the wrap switch invokes
        // shardingConfig.Backend which we can intercept via a
        // recording wrapper. If we can't intercept, we instead assert
        // that result.Model implements IShardedModel (when build
        // completes).
        AiModelResult<float, Tensor<float>, Tensor<float>>? result = null;
        System.Exception? buildException = null;
        try
        {
            result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureDistributedTraining(backend, DistributedStrategy.DDP)
                .BuildAsync();
        }
        catch (System.Exception ex)
        {
            buildException = ex;
        }

        // Whether build succeeded or failed downstream, the wrap-switch
        // runs synchronously BEFORE the optimizer. The strongest
        // routing-observable is result.Model being an IShardedModel —
        // a stored-but-not-consumed regression would leave it as the
        // raw Transformer. If build failed, fall back to checking the
        // exception originated FROM the distributed namespace by
        // walking the stack-trace frames for a known DistributedTraining /
        // ShardedModelBase frame (review #1368: substring-match on the
        // raw ex.ToString() is brittle to message renames and matches
        // frame text from unrelated places).
        if (result != null)
        {
            Assert.IsAssignableFrom<IShardedModel<float, Tensor<float>, Tensor<float>>>(result.Model);
        }
        else
        {
            Assert.NotNull(buildException);
            Assert.True(
                IsExceptionFromNamespace(buildException!, "AiDotNet.DistributedTraining"),
                $"ConfigureDistributedTraining build failed, but the failure did not originate inside " +
                $"the AiDotNet.DistributedTraining namespace. Stored-but-not-consumed regression likely. " +
                $"Top-frame: {buildException!.GetType().FullName} | message: {buildException.Message}");
        }
    }

    /// <summary>
    /// Walks the exception chain (this + InnerException + AggregateException's
    /// InnerExceptions) and returns true if any frame's declaring type
    /// is in <paramref name="targetNamespacePrefix"/>. Used by the
    /// distributed / federated routing assertions to replace brittle
    /// substring matching on raw exception ToString() (review #1368).
    /// </summary>
    private static bool IsExceptionFromNamespace(System.Exception ex, string targetNamespacePrefix)
    {
        var visit = new System.Collections.Generic.Stack<System.Exception>();
        visit.Push(ex);
        while (visit.Count > 0)
        {
            var current = visit.Pop();
            if (current.TargetSite?.DeclaringType?.FullName is string declType
                && declType.StartsWith(targetNamespacePrefix, System.StringComparison.Ordinal))
                return true;
            // Also walk the StackTrace for frames in the target namespace —
            // TargetSite is only the innermost throw, but a routing failure
            // might surface as an unrelated exception type thrown deep
            // inside our target code.
            if (current.StackTrace is string st)
            {
                // Frame format: "at AiDotNet.DistributedTraining.X.Method(...)".
                if (st.Contains("at " + targetNamespacePrefix + ".", System.StringComparison.Ordinal))
                    return true;
            }
            if (current.InnerException is not null) visit.Push(current.InnerException);
            if (current is System.AggregateException agg)
            {
                foreach (var inner in agg.InnerExceptions) visit.Push(inner);
            }
        }
        return false;
    }

    /// <summary>
    /// ConfigurePipelineParallelism — verifies the configured pipeline
    /// strategy reaches the distributed wrap switch when paired with
    /// <see cref="DistributedStrategy.PipelineParallel"/>.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePipelineParallelism_WithDistributedBackend_RoutesToPipelineParallelBranch()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1);

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureModel(model);
        builder.ConfigureDataLoader(loader);
        builder.ConfigurePipelineParallelism(microBatchCount: 1);
        builder.ConfigureDistributedTraining(backend, DistributedStrategy.PipelineParallel);

        AiModelResult<float, Tensor<float>, Tensor<float>>? result = null;
        System.Exception? buildException = null;
        try
        {
            result = await builder.BuildAsync();
        }
        catch (System.Exception ex)
        {
            buildException = ex;
        }

        // Observable side-effect: with PipelineParallel strategy + a
        // backend, the dispatch switch at AiModelBuilder.cs:2612
        // constructs a PipelineParallelModel wrapper synchronously
        // BEFORE the optimizer runs. Verify either the wrap survived
        // to result.Model, OR the build failure originated from inside
        // the pipeline-parallel code path (which still proves the
        // routing fired).
        if (result != null)
        {
            Assert.IsAssignableFrom<IShardedModel<float, Tensor<float>, Tensor<float>>>(result.Model);
        }
        else
        {
            Assert.NotNull(buildException);
            Assert.True(
                IsExceptionFromNamespace(buildException!, "AiDotNet.DistributedTraining"),
                $"ConfigurePipelineParallelism build failed, but the failure did not originate inside " +
                $"the AiDotNet.DistributedTraining namespace. Stored-but-not-consumed regression likely. " +
                $"Top-frame: {buildException!.GetType().FullName} | message: {buildException.Message}");
        }
    }

    /// <summary>
    /// ConfigureFederatedLearning — uses a federated client data loader
    /// to drive the federated branch at
    /// <c>AiModelBuilder.cs:3042</c>. Verifies the federated trainer
    /// gets constructed (it requires at least one client partition).
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureFederatedLearning_WithClientDataLoader_EntersFederatedBranch()
    {
        var (features, labels) = MakeMemorizationSet();
        var model = MakeCanaryModel();

        // Standard canary loader satisfies IInputOutputDataLoader but
        // NOT IFederatedClientDataLoader. The federated branch falls
        // back to client-range partitioning of the in-memory data when
        // the loader doesn't provide explicit client partitions
        // (AiModelBuilder.cs:3162's CreateFederatedClientPartitions
        // path).
        var loader = MakeCanaryLoader(features, labels);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 2,
            MaxRounds = 1,
            LocalEpochs = 1,
            ClientSelectionFraction = 1.0,
        };

        // Federated branch enters at AiModelBuilder.cs:3042 when
        // _federatedLearningOptions != null. The downstream
        // InMemoryFederatedTrainer requires more setup than the canary
        // provides (e.g. an aggregation strategy); a throw inside the
        // branch still proves the routing fired. We capture the
        // exception and assert it originated from the FederatedLearning
        // namespace (review #1368: bare ThrowsAnyAsync<Exception>
        // accepts unrelated NRE / OOM / a builder-side bug thrown
        // BEFORE the federated branch — narrow to provenance instead).
        System.Exception? buildException = null;
        try
        {
            await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureFederatedLearning(flOptions)
                .BuildAsync();
        }
        catch (System.Exception ex)
        {
            buildException = ex;
        }
        Assert.NotNull(buildException);
        Assert.True(
            IsExceptionFromNamespace(buildException!, "AiDotNet.FederatedLearning"),
            $"ConfigureFederatedLearning build failed, but the failure did not originate inside " +
            $"the AiDotNet.FederatedLearning namespace. Stored-but-not-consumed regression " +
            $"would skip the federated branch and fall through to the supervised path. " +
            $"Top-frame: {buildException!.GetType().FullName} | message: {buildException.Message}");
    }
}
