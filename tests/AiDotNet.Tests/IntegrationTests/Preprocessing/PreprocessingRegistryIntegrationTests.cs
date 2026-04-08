using AiDotNet.Data.Loaders;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Integration tests verifying that the static PreprocessingRegistry is no longer set
/// by AiModelBuilder, preventing race conditions in concurrent model building.
/// These tests mutate static global state and must not run in parallel with other tests
/// that touch the same registry specialization.
/// </summary>
[Collection("PreprocessingRegistryTests")]
public class PreprocessingRegistryIntegrationTests
{
    /// <summary>
    /// Verifies that building two models concurrently with different preprocessing
    /// does not cause cross-contamination via the static registry.
    /// Previously, AiModelBuilder.ConfigurePreprocessing() wrote to a static global
    /// PreprocessingRegistry, so concurrent builds would overwrite each other's pipeline.
    /// After the fix, the registry is never set by AiModelBuilder.
    /// </summary>
    [Fact]
    public async Task ConcurrentBuilds_WithDifferentPreprocessing_DoNotCrossContaminate()
    {
        // Arrange: clear any leftover state from other tests
#pragma warning disable CS0618 // Obsolete warning expected — testing the deprecated registry
        PreprocessingRegistry<double, Matrix<double>>.Clear();
#pragma warning restore CS0618

        // Each builder gets its own data loader to avoid thread-safety issues
        // (DataLoaderBase.LoadAsync() is not thread-safe for shared instances)
        static (Matrix<double> x, Vector<double> y) CreateData()
        {
            var x = new Matrix<double>(new double[,]
            {
                { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 },
                { 11, 12 }, { 13, 14 }, { 15, 16 }, { 17, 18 }, { 19, 20 }
            });
            var y = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            return (x, y);
        }

        // Act: build two models concurrently, each with different preprocessing
        var task1 = Task.Run(async () =>
        {
            var (x, y) = CreateData();
            var loader = DataLoaders.FromMatrixVector(x, y);
            var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new MultipleRegression<double>())
                .ConfigurePreprocessing(pipeline => pipeline
                    .Add(new StandardScaler<double>()));

            return await builder.BuildAsync();
        });

        var task2 = Task.Run(async () =>
        {
            var (x, y) = CreateData();
            var loader = DataLoaders.FromMatrixVector(x, y);
            var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new MultipleRegression<double>())
                .ConfigurePreprocessing(pipeline => pipeline
                    .Add(new MinMaxScaler<double>()));

            return await builder.BuildAsync();
        });

        var results = await Task.WhenAll(task1, task2);

        // Assert: both builds succeed with preprocessing applied
        Assert.NotNull(results[0]);
        Assert.NotNull(results[1]);

        // Assert: each result has its own fitted preprocessing
        Assert.NotNull(results[0].PreprocessingInfo);
        Assert.True(results[0].PreprocessingInfo?.IsFitted ?? false,
            "First build's preprocessing should be fitted");
        Assert.NotNull(results[1].PreprocessingInfo);
        Assert.True(results[1].PreprocessingInfo?.IsFitted ?? false,
            "Second build's preprocessing should be fitted");

        // Assert: the static registry was NOT set by either builder
#pragma warning disable CS0618
        Assert.False(PreprocessingRegistry<double, Matrix<double>>.IsConfigured,
            "PreprocessingRegistry should NOT be set by AiModelBuilder after the fix. " +
            "The pipeline should flow instance-locally via PreprocessingInfo instead.");
#pragma warning restore CS0618
    }

    /// <summary>
    /// Verifies that a model built without the static registry still succeeds.
    /// The preprocessing pipeline flows through the builder's instance field to AiModelResult.PreprocessingInfo,
    /// not through the global static PreprocessingRegistry.
    /// </summary>
    [Fact]
    public async Task BuildAsync_WithPreprocessing_Succeeds_WithoutRegistry()
    {
        // Arrange: clear any leftover state
#pragma warning disable CS0618
        PreprocessingRegistry<double, Matrix<double>>.Clear();
#pragma warning restore CS0618

        var x = new Matrix<double>(new double[,]
        {
            { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 },
            { 11, 12 }, { 13, 14 }, { 15, 16 }, { 17, 18 }, { 19, 20 }
        });
        var y = new Vector<double>(new double[] { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 });

        var loader = DataLoaders.FromMatrixVector(x, y);

        // Act: build with preprocessing
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .BuildAsync();

        // Assert: the model was built successfully
        Assert.NotNull(result);

        // Assert: preprocessing was actually applied via instance-level PreprocessingInfo
        Assert.NotNull(result.PreprocessingInfo);
        Assert.True(result.PreprocessingInfo?.IsFitted ?? false,
            "Preprocessing should be fitted via instance-level flow");

        // Assert: the static registry was NOT set (proves instance-level, not static)
#pragma warning disable CS0618
        Assert.False(PreprocessingRegistry<double, Matrix<double>>.IsConfigured,
            "PreprocessingRegistry should NOT be set by AiModelBuilder.");
#pragma warning restore CS0618
    }

    /// <summary>
    /// Verifies that the deprecated PreprocessingRegistry still compiles and functions
    /// for any external code that may reference it directly, but produces an obsolete warning.
    /// The [Obsolete] attribute is tested at compile time (CS0618 warning), and here we
    /// verify the runtime behavior still works for backward compatibility.
    /// </summary>
    [Fact]
    public void DeprecatedRegistry_StillFunctionsForBackwardCompatibility()
    {
#pragma warning disable CS0618 // Intentionally testing deprecated API
        try
        {
            // Arrange
            PreprocessingRegistry<double, Matrix<double>>.Clear();
            Assert.False(PreprocessingRegistry<double, Matrix<double>>.IsConfigured);

            // Act: set a pipeline manually (external code might still do this)
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new StandardScaler<double>());
            PreprocessingRegistry<double, Matrix<double>>.Current = pipeline;

            // Assert: the registry still works
            Assert.True(PreprocessingRegistry<double, Matrix<double>>.IsConfigured);
            Assert.NotNull(PreprocessingRegistry<double, Matrix<double>>.Current);
        }
        finally
        {
            // Always clean up static state even if assertions fail
            PreprocessingRegistry<double, Matrix<double>>.Clear();
        }
        Assert.False(PreprocessingRegistry<double, Matrix<double>>.IsConfigured);
#pragma warning restore CS0618
    }

    /// <summary>
    /// Verifies that multiple sequential builds do not leave stale state in the registry.
    /// </summary>
    [Fact]
    public async Task SequentialBuilds_DoNotLeaveStaleRegistryState()
    {
#pragma warning disable CS0618
        PreprocessingRegistry<double, Matrix<double>>.Clear();
#pragma warning restore CS0618

        var x = new Matrix<double>(new double[,]
        {
            { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 },
            { 11, 12 }, { 13, 14 }, { 15, 16 }, { 17, 18 }, { 19, 20 }
        });
        var y = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var loader = DataLoaders.FromMatrixVector(x, y);

        // Build first model
        var result1 = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigurePreprocessing(new StandardScaler<double>())
            .BuildAsync();

        // Build second model
        var result2 = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigurePreprocessing(new MinMaxScaler<double>())
            .BuildAsync();

        // Assert: both succeeded with preprocessing applied
        Assert.NotNull(result1);
        Assert.NotNull(result2);

        // Assert: each result has its own fitted preprocessing
        Assert.NotNull(result1.PreprocessingInfo);
        Assert.True(result1.PreprocessingInfo?.IsFitted ?? false,
            "First build's preprocessing should be fitted");
        Assert.NotNull(result2.PreprocessingInfo);
        Assert.True(result2.PreprocessingInfo?.IsFitted ?? false,
            "Second build's preprocessing should be fitted");

        // Assert: registry was never set
#pragma warning disable CS0618
        Assert.False(PreprocessingRegistry<double, Matrix<double>>.IsConfigured,
            "PreprocessingRegistry should remain empty after builds.");
#pragma warning restore CS0618
    }
}
