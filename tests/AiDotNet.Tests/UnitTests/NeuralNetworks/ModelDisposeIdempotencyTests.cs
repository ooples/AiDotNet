using System;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A second <c>Dispose</c> of a model must be a no-op.
///
/// <para>Callers routinely dispose the same model twice without knowing it: an <see cref="AiModelResult{T,TInput,TOutput}"/>
/// disposes the model it wraps, and the code that configured that model disposes it as well. Before the fix,
/// <see cref="NeuralNetworkBase{T}.Dispose()"/> had no disposed flag, so every repeated call re-ran the whole
/// teardown. That teardown is not local to the model: <see cref="TapeTrainingStep{T}.InvalidateCache"/> clears
/// the THREAD-GLOBAL parameter cache, so re-disposing a long-dead network silently evicted the cache of whatever
/// live model the thread had trained since. Derived <c>Dispose(bool)</c> overrides also ran their own teardown
/// again, releasing sub-models and sessions a second time.</para>
/// </summary>
public partial class ModelDisposeIdempotencyTests
{
    private static NeuralNetwork<double> Network()
        => new(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2));

    [Fact]
    public void Redisposing_a_dead_network_does_not_evict_a_live_models_training_cache()
    {
        var dead = Network();
        var live = Network();
        live.MaterializeParameters();

        dead.Dispose();

        // Premise: the live model's parameter walk is cached on this thread, so asking twice returns
        // the very same list.
        var cached = TapeTrainingStep<double>.CollectParameters(live.Layers);
        Assert.Same(cached, TapeTrainingStep<double>.CollectParameters(live.Layers));

        dead.Dispose();

        Assert.Same(cached, TapeTrainingStep<double>.CollectParameters(live.Layers));
    }

    [Fact]
    public void First_dispose_still_tears_down_the_training_cache_and_releases_every_layer()
    {
        var disposed = Network();
        var live = Network();
        live.MaterializeParameters();
        var cached = TapeTrainingStep<double>.CollectParameters(live.Layers);
        Assert.Same(cached, TapeTrainingStep<double>.CollectParameters(live.Layers));

        disposed.Dispose();

        // The first dispose keeps its existing, deliberately conservative behaviour: it invalidates the
        // thread's tape cache (a stale plan over returned buffers is worse than one recomputation) ...
        Assert.NotSame(cached, TapeTrainingStep<double>.CollectParameters(live.Layers));

        // ... and it cascades into every layer through the once-only guard, which therefore refuses them now.
        Assert.NotEmpty(disposed.Layers);
        foreach (var layer in disposed.Layers)
        {
            Assert.False(DisposeOnceGuard.TryDispose(Assert.IsAssignableFrom<IDisposable>(layer)));
        }

        Assert.Null(Record.Exception(() => disposed.Dispose()));
    }

    [Fact]
    public void A_derived_networks_teardown_runs_once_across_repeated_disposes()
    {
        var network = new CountingNetwork(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2));

        network.Dispose();
        network.Dispose();
        network.Dispose();

        Assert.Equal(1, network.DisposeCalls);
    }

    [Fact]
    public void Disposing_a_result_and_then_the_model_it_wraps_tears_the_model_down_once()
    {
        // The downstream pattern: dispose the AiModelResult (which disposes its Model), then dispose the model
        // that was configured into the builder -- the same instance.
        var model = new CountingNetwork(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2));
        var result = new AiModelResult<double, Tensor<double>, Tensor<double>> { Model = model };

        result.Dispose();
        result.Dispose();
        model.Dispose();

        Assert.Equal(1, model.DisposeCalls);
    }

    [Fact]
    public void A_derived_ModelBase_teardown_runs_once_across_repeated_disposes()
    {
        var model = new CountingVectorModel(new Vector<double>(new[] { 1.0, 2.0 }));

        model.Dispose();
        model.Dispose();

        Assert.Equal(1, model.DisposeCalls);
    }

    [Fact]
    public void A_derived_TimeSeriesModelBase_teardown_runs_once_across_repeated_disposes()
    {
        var model = new CountingArimaModel();

        model.Dispose();
        model.Dispose();

        Assert.Equal(1, model.DisposeCalls);
    }

    [Fact]
    public void A_derived_AiModelResult_teardown_runs_once_across_repeated_disposes()
    {
        var result = new CountingResult();

        result.Dispose();
        result.Dispose();

        Assert.Equal(1, result.DisposeCalls);
    }

    private sealed partial class CountingNetwork : NeuralNetwork<double>
    {
        public CountingNetwork(NeuralNetworkArchitecture<double> architecture) : base(architecture)
        {
        }

        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            // The common override shape: own teardown first, then base.
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }

    private sealed partial class CountingVectorModel : VectorModel<double>
    {
        public CountingVectorModel(Vector<double> coefficients) : base(coefficients)
        {
        }

        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }

    private sealed partial class CountingArimaModel : ARIMAModel<double>
    {
        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }

    private sealed partial class CountingResult : AiModelResult<double, Tensor<double>, Tensor<double>>
    {
        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }
}
