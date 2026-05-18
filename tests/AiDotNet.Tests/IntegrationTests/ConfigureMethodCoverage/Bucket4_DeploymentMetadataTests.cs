using AiDotNet.Configuration;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 4 — Configure* methods whose ONLY observable effect is that the
/// configured value lands on <see cref="AiModelResult{T,TInput,TOutput}.DeploymentConfiguration"/>
/// (or on a process-wide config flag).
/// </summary>
/// <remarks>
/// <para>
/// Each test sets a NON-DEFAULT value on its config (e.g.
/// <c>MaxCacheSize = 99</c>) and asserts that the SAME non-default value
/// is observable on the post-build result. This screens for the systemic
/// "stored on the builder but never consumed" pattern that PR #1357 / #1361
/// found across the Configure* surface.
/// </para>
/// <para>
/// <b>Process-wide state warning:</b> The ConfigureGpuDiagnostics
/// test mutates the process-wide static
/// <c>GpuDiagnosticsConfig.Level</c>. The collection fixture
/// (<see cref="ConfigureMethodTestCpuFixture"/>) serialises tests
/// inside the <c>ConfigureMethodCoverage</c> collection, but xUnit
/// runs OTHER test collections in parallel by default — concurrently-
/// running tests that read <c>GpuDiagnosticsConfig.Level</c> may
/// observe the sentinel <c>Verbose</c> setting briefly while the
/// test is mid-run. The test restores the previous value in a
/// <c>finally</c> block; any future test that reads
/// <c>GpuDiagnosticsConfig.Level</c> must either join the
/// <c>ConfigureMethodCoverage</c> collection or tolerate transient
/// observations of the sentinel.
/// </para>
/// <para>
/// Methods covered (5 of the methods not touched by the other open PRs
/// in flight — #1361 covers AdversarialRobustness, #1362 covers
/// MixedPrecision, #1367 covers ModelRegistry, #1351 covers Adam,
/// #1349/#1363 cover INT8):
/// </para>
/// <list type="bullet">
///   <item>ConfigureCaching</item>
///   <item>ConfigureVersioning</item>
///   <item>ConfigureABTesting</item>
///   <item>ConfigureExport</item>
///   <item>ConfigureGpuDiagnostics</item>
/// </list>
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket4_DeploymentMetadataTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket4_DeploymentMetadataTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureCaching — a non-default <c>MaxCacheSize</c> set via
    /// <c>ConfigureCaching</c> MUST land on <c>result.DeploymentConfiguration.Caching.MaxCacheSize</c>
    /// after BuildAsync. Verifies the field flows through
    /// <c>DeploymentConfiguration.Create</c> into the result instead of being
    /// silently dropped.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureCaching_NonDefaultValue_LandsOnDeploymentConfiguration()
    {
        const int sentinel = 99;
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var cacheCfg = new CacheConfig { MaxCacheSize = sentinel };
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureCaching(cacheCfg)
            .BuildAsync();

        Assert.NotNull(result.DeploymentConfiguration);
        Assert.NotNull(result.DeploymentConfiguration!.Caching);
        Assert.Equal(sentinel, result.DeploymentConfiguration.Caching!.MaxCacheSize);
    }

    /// <summary>
    /// ConfigureVersioning — a non-default <c>DefaultVersion</c> set via
    /// <c>ConfigureVersioning</c> MUST land on
    /// <c>result.DeploymentConfiguration.Versioning.DefaultVersion</c>.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureVersioning_NonDefaultValue_LandsOnDeploymentConfiguration()
    {
        const string sentinel = "v999-integration-test";
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var versCfg = new VersioningConfig { DefaultVersion = sentinel };
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureVersioning(versCfg)
            .BuildAsync();

        Assert.NotNull(result.DeploymentConfiguration);
        Assert.NotNull(result.DeploymentConfiguration!.Versioning);
        Assert.Equal(sentinel, result.DeploymentConfiguration.Versioning!.DefaultVersion);
    }

    /// <summary>
    /// ConfigureABTesting — a non-default <c>DefaultTrafficSplit</c> set via
    /// <c>ConfigureABTesting</c> MUST land on
    /// <c>result.DeploymentConfiguration.ABTesting.DefaultTrafficSplit</c>.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureABTesting_NonDefaultValue_LandsOnDeploymentConfiguration()
    {
        const double sentinel = 0.123;
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var abCfg = new ABTestingConfig { Enabled = true, DefaultTrafficSplit = sentinel };
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureABTesting(abCfg)
            .BuildAsync();

        Assert.NotNull(result.DeploymentConfiguration);
        Assert.NotNull(result.DeploymentConfiguration!.ABTesting);
        Assert.Equal(sentinel, result.DeploymentConfiguration.ABTesting!.DefaultTrafficSplit);
    }

    /// <summary>
    /// ConfigureExport — a non-default <c>TargetPlatform</c> set via
    /// <c>ConfigureExport</c> MUST land on
    /// <c>result.DeploymentConfiguration.Export.TargetPlatform</c>. Verifies
    /// the export config flows into the result so downstream
    /// <c>ExportToOnnx</c> / <c>ExportToCoreML</c> / etc. methods see it.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureExport_NonDefaultTarget_LandsOnDeploymentConfiguration()
    {
        const TargetPlatform sentinel = TargetPlatform.TFLite;
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var expCfg = new ExportConfig { TargetPlatform = sentinel };
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureExport(expCfg)
            .BuildAsync();

        Assert.NotNull(result.DeploymentConfiguration);
        Assert.NotNull(result.DeploymentConfiguration!.Export);
        Assert.Equal(sentinel, result.DeploymentConfiguration.Export!.TargetPlatform);
    }

    /// <summary>
    /// ConfigureGpuDiagnostics — sets the process-wide
    /// <c>GpuDiagnosticsConfig.Level</c>. Picks <c>Verbose</c> as the
    /// sentinel because the test fixture's CPU-mode default leaves
    /// <c>Level</c> at <c>Silent</c>; asserts the explicit Configure call
    /// flipped the global to the requested value.
    /// </summary>
    /// <remarks>
    /// Uses <c>GpuDiagnosticsConfig.PushLevel</c> for scoped restoration:
    /// the <c>using</c>-declaration guarantees the previous level is
    /// restored even if BuildAsync or the assertion throws (which the
    /// hand-rolled try/finally + Level = previous pattern would also
    /// catch but is easier to forget on future edits). The scoped
    /// override pattern was added in this PR to address review
    /// concerns about Bucket 4 mutating process-global GpuDiagnosticsConfig
    /// state. NOTE: the static slot is a single value, NOT a per-thread
    /// stack, so xUnit-parallel collections still need
    /// <c>[Collection("ConfigureMethodCoverage")]</c> serialization
    /// to prevent cross-test interference.
    /// </remarks>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureGpuDiagnostics_LevelOverride_AppliesToGlobalConfig()
    {
        const GpuDiagnosticLevel sentinel = GpuDiagnosticLevel.Verbose;
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        // Snapshot the level via PushLevel — the no-op "push current,
        // assign current" pair is intentional: PushLevel's primary value
        // here is the lifo-stack restoration on Dispose, not the
        // intermediate assignment. ConfigureGpuDiagnostics below installs
        // the sentinel; the scope's Dispose pops back to whatever the
        // level was at the start of the test (this PR's review C7mmp:
        // the apparent no-op middle is a deliberate save-point, not a bug).
        using var _scope = GpuDiagnosticsConfig.PushLevel(GpuDiagnosticsConfig.Level);

        var diag = new GpuDiagnosticsOptions { Level = sentinel };
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureGpuDiagnostics(diag)
            .BuildAsync();

        Assert.Equal(sentinel, GpuDiagnosticsConfig.Level);
    }
}
