using AiDotNet.DataVersionControl;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Xunit;
using Xunit.Abstractions;
using AiDotNet.Tests.Helpers;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 5 — Configure* methods that wire build-lifecycle concerns
/// (license validation, data-version tracking, safety-pipeline attachment).
/// Each test exercises an OBSERVABLE side-effect of the configure call
/// post-build, not just the setter.
/// </summary>
/// <remarks>
/// Methods covered (none overlap with the other open Configure*-related PRs):
/// <list type="bullet">
///   <item>ConfigureLicenseKey</item>
///   <item>ConfigureDataVersionControl</item>
///   <item>ConfigureSafety</item>
/// </list>
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket5_LifecycleTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket5_LifecycleTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureLicenseKey — verifies the configured key is applied to the
    /// process-wide <c>ModelPersistenceGuard</c> license scope inside
    /// BuildAsync (see <c>AiModelBuilder.cs:1414</c>). We can't observe the
    /// scope directly post-build (it's <c>using var</c> and goes out of
    /// scope before BuildAsync returns), so the test exercises the
    /// equivalent guarantee: BuildAsync completes without throwing on a
    /// valid offline-mode key. A "stored but never consumed" regression
    /// would also pass this trivially, so we additionally assert the
    /// configured key is reachable via the internal accessor PR #1361
    /// established for cross-Configure* wiring verification.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureLicenseKey_OfflineKey_ReachesLicenseScopeDuringBuild()
    {
        // Fail-closed validation requires a verifiable signature: inject a build key for this scope so
        // the signed offline key below validates (Active) and the build reaches the license-wiring check.
        using var _buildKey = LicenseTestSupport.WithBuildKey();
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        // Test-only placeholder key. The current ModelPersistenceGuard
        // .SetActiveLicenseKey contract accepts any key in offline mode
        // (empty ServerUrl); if upstream tightens validation, this test
        // becomes a canary that the license-key wiring contract changed —
        // the test's failure surface will point at the key-construction
        // line, not at the wiring assertion (this PR's review: documented
        // for future readers; not switching to a "real" test key because
        // any such key would need to be checked in and would either be
        // an actual offline-license credential or another placeholder
        // that has the same property).
        var key = new AiDotNetLicenseKey(LicenseTestSupport.SignedKey("test"))
        {
            ServerUrl = "", // offline-only mode
            Environment = "test",
            EnableTelemetry = false,
        };

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureLicenseKey(key);
        builder.ConfigureModel(model);
        builder.ConfigureDataLoader(loader);

        // Internal accessor (added by PR #1361 for this exact wiring check)
        // confirms the setter wrote the field that the BuildAsync
        // licenseScope at AiModelBuilder.cs:1414 will read.
        Assert.Same(key, ((AiDotNet.Configuration.IConfiguredView<float, Tensor<float>, Tensor<float>>)builder).ConfiguredLicenseKey);

        var result = await builder.BuildAsync();

        // BuildAsync's `using var licenseScope = ModelPersistenceGuard
        // .SetActiveLicenseKey(_licenseKey)` runs through the licensing
        // path. A key that fails validation throws here — so successful
        // BuildAsync completion proves the configured key was consumed.
        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureLicenseKey");
    }

    /// <summary>
    /// ConfigureDataVersionControl — verifies the DVC is consulted by
    /// BuildSupervisedInternalAsync (which calls
    /// <c>_dataVersionControl.LinkDatasetToRun</c> when an experiment
    /// tracker is also configured — see <c>AiModelBuilder.cs:2850</c>).
    /// </summary>
    /// <remarks>
    /// Stored-but-never-consumed regression would NOT call
    /// LinkDatasetToRun, so the test uses a recording DVC instance whose
    /// <c>LinkDatasetToRun</c> appends to an in-memory list, then
    /// asserts the list is non-empty post-build.
    /// </remarks>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureDataVersionControl_PairedWithExperimentTracker_LinksDatasetToRun()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        string trackerDir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "AiDotNetTrackerTest_" + System.Guid.NewGuid().ToString("N"));
        var recordingDvc = new RecordingDataVersionControl<float>();
        try
        {
            var tracker = new AiDotNet.ExperimentTracking.ExperimentTracker<float>(trackerDir);

            var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureDataVersionControl(recordingDvc)
                .ConfigureExperimentTracker(tracker)
                .BuildAsync();

            // BuildSupervisedInternalAsync calls _dataVersionControl
            // .LinkDatasetToRun(...) when both DVC and tracker are wired
            // and dataVersionHash is non-null. A stored-but-not-consumed bug
            // would never call this; the recording DVC catches the call.
            Assert.NotEmpty(recordingDvc.LinkedRuns);
        }
        finally
        {
            // Clean up both temp dirs the test created — matches Bucket3
            // ConfigureModelRegistry's cleanup pattern. Without this each
            // run leaks a tracker dir + a RecordingDataVersionControl dir
            // into %TEMP%, accumulating on CI. Both guards are defensive:
            // trackerDir is always non-null (the Path.Combine above can't
            // fail), but the directory may not exist if the ExperimentTracker
            // ctor threw before creating it. recordingDvc was initialized
            // OUTSIDE the try (L108) so it can't be null when the finally
            // runs, but a future refactor that moves it inside the try
            // would re-introduce a NRE here — keep the null-conditional
            // (this PR's review C6WPz).
            TryDeleteDir(trackerDir);
            TryDeleteDir(recordingDvc?.StorageDirectory);
        }
    }

    private static void TryDeleteDir(string? path)
    {
        if (string.IsNullOrEmpty(path)) return;
        try { if (System.IO.Directory.Exists(path)) System.IO.Directory.Delete(path, recursive: true); }
        catch (System.IO.IOException) { }
        catch (System.UnauthorizedAccessException) { }
    }

    /// <summary>
    /// ConfigureSafety — verifies the safety pipeline is constructed by
    /// <c>SafetyPipelineFactory</c> and attached to
    /// <c>result.SafetyPipeline</c> (a public property). A stored-but-
    /// never-consumed bug would leave the property null even after
    /// ConfigureSafety was called.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureSafety_DefaultConfig_AttachesSafetyPipelineToResult()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureSafety(safety => { /* defaults */ })
            .BuildAsync();

        // SafetyPipelineFactory.Create is called by AttachSafetyPipeline
        // (AiModelBuilder.cs:1623) and its result is assigned to
        // AiModelResult.SafetyPipeline. Asserting non-null proves the
        // configure call reached the factory and the factory's output
        // reached the result.
        Assert.NotNull(result.SafetyPipeline);
    }

    /// <summary>
    /// Recording DVC that subclasses the concrete <see cref="DataVersionControl{T}"/>
    /// and overrides <c>LinkDatasetToRun</c> to capture every call so the test
    /// can assert the configure → build path actually invoked it (vs the
    /// stored-but-not-consumed regression).
    /// </summary>
    private sealed class RecordingDataVersionControl<TNum> : DataVersionControl<TNum>
    {
        // ConcurrentBag (thread-safe lock-free append) instead of a raw List
        // so a concurrent BuildSupervisedInternalAsync path that fans
        // LinkDatasetToRun across multiple threads doesn't tear the list
        // (this PR's review). Order-of-arrival is not asserted on, so the
        // bag's unordered semantics are fine.
        public readonly System.Collections.Concurrent.ConcurrentBag<(string Dataset, string Version, string Run, string? Model)> LinkedRuns
            = new();

        public string StorageDirectory { get; }

        public RecordingDataVersionControl()
            : this(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "AiDotNetDVCRecorder_" + System.Guid.NewGuid().ToString("N")))
        {
        }

        private RecordingDataVersionControl(string storageDirectory)
            : base(storageDirectory)
        {
            StorageDirectory = storageDirectory;
        }

        public override void LinkDatasetToRun(string datasetName, string versionHash, string runId, string? modelId = null)
        {
            // Intentionally do NOT chain to base. Base requires the version
            // to exist in the underlying store (this recording stub never
            // creates it), and the test only cares about observing the
            // side effect — chaining would force test setup to materialise
            // a real DVC store and assert on more than the wiring claim
            // (this PR's review: justified because this is a recording test
            // double, not a production substitute. If the DVC base contract
            // ever starts requiring side effects that production callers
            // depend on, that contract change should be caught by a unit
            // test on DataVersionControl<T> itself, not by every consumer's
            // recording stub).
            LinkedRuns.Add((datasetName, versionHash, runId, modelId));
        }
    }
}
