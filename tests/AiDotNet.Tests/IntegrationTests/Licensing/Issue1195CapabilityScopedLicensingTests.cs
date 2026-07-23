using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Licensing;

/// <summary>
/// Regression tests for issue #1195 — capability-scoped license enforcement.
///
/// The issue ships in three coordinated stages: server adds <c>capabilities</c>
/// to its 200 response, upstream AiDotNet (this repo) tags requests with
/// <c>package=AiDotNet</c> and forwards the key into the tensor layer, and
/// AiDotNet.Tensors ships its capability-scoped guard last.
///
/// These tests cover the upstream half:
/// <list type="bullet">
///   <item>§2d — every validation request includes <c>package=AiDotNet</c>.</item>
///   <item>Response parser surfaces the new <c>capabilities</c> array
///     (and tolerates older servers that don't send the field).</item>
///   <item><see cref="LicenseValidationResult.HasCapability"/> matches
///     ordinally so a typo or case-fold can't grant capabilities.</item>
///   <item>Forward-compatible <see cref="TensorLicenseFlow"/> degrades to
///     no-op when the AiDotNet.Tensors licensing API is not present —
///     the rollout state during which this PR ships.</item>
/// </list>
/// The acceptance-criterion test "trial counter ticks once on the upstream
/// side, zero on the tensor side" is covered by
/// <see cref="TrialCounter_TicksOnceOnUpstreamLayer_ZeroOnTensorLayer"/>:
/// it verifies the upstream tick is observable today and the bridge would
/// route suppression into the tensor layer once the API ships, by checking
/// the reflection bridge's resolution state.
/// </summary>
[Collection("LicensingTests")]
public class Issue1195CapabilityScopedLicensingTests
{
    // ─── §2d: package tag in request body ───

    /// <summary>
    /// Every validate-license request sent from upstream AiDotNet must
    /// include <c>package=AiDotNet</c> so server-side analytics can
    /// attribute traffic by SDK origin. AiDotNet.Tensors sends
    /// <c>package=AiDotNet.Tensors</c> from its own validator.
    /// </summary>
    [Fact]
    public void BuildRequestBody_AlwaysIncludesPackageTag_AiDotNet()
    {
        var validator = new LicenseValidator(new AiDotNetLicenseKey("aidn.testkey1234.signature5678")
        {
            ServerUrl = string.Empty,
            EnableTelemetry = false
        });

        var body = validator.BuildRequestBody();

        Assert.True(body.ContainsKey("package"),
            "Issue #1195 §2d: request body must include the 'package' analytics tag.");
        Assert.Equal("AiDotNet", body["package"]);
    }

    /// <summary>
    /// The package tag must travel with the body even when telemetry is
    /// disabled — otherwise a privacy-conscious user disabling telemetry
    /// would also opt out of the analytics tag, which is not the intent
    /// (the tag carries no PII; it identifies the SDK package, not the
    /// user or machine).
    /// </summary>
    [Fact]
    public void BuildRequestBody_IncludesPackageEvenWhenTelemetryDisabled()
    {
        var validator = new LicenseValidator(new AiDotNetLicenseKey("aidn.testkey1234.signature5678")
        {
            ServerUrl = string.Empty,
            EnableTelemetry = false
        });

        var body = validator.BuildRequestBody();

        // Telemetry off ⇒ no hostname / os_description, but package still set.
        Assert.False(body.ContainsKey("hostname"));
        Assert.False(body.ContainsKey("os_description"));
        Assert.Equal("AiDotNet", body["package"]);
    }

    // ─── §1: capabilities response shape ───

    /// <summary>
    /// A 200 response that includes the new <c>capabilities</c> array
    /// (post issue #1195 server rollout) is parsed and surfaced via
    /// <see cref="LicenseValidationResult.Capabilities"/>.
    /// </summary>
    [Fact]
    public void ParseResponse_WithCapabilitiesArray_ExposesViaResult()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "professional",
            capabilities = new[] { "tensors:save", "tensors:load", "model:save", "model:load" },
            message = "License validated."
        });

        var result = LicenseValidator.ParseResponse(responseJson, 200);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal(4, result.Capabilities.Count);
        Assert.Contains("tensors:save", result.Capabilities);
        Assert.Contains("tensors:load", result.Capabilities);
        Assert.Contains("model:save", result.Capabilities);
        Assert.Contains("model:load", result.Capabilities);
    }

    /// <summary>
    /// A 200 response that omits <c>capabilities</c> (older server) must
    /// still parse cleanly and yield an empty capability list — the
    /// capability check then naturally denies, matching the issue's note:
    /// "real keys validate as Active with an empty capability set on the
    /// tensor side, and the tensor guard rejects them."
    /// </summary>
    [Fact]
    public void ParseResponse_WithoutCapabilities_ReturnsEmptyList()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "community",
            message = "Older server, no capabilities field."
        });

        var result = LicenseValidator.ParseResponse(responseJson, 200);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.NotNull(result.Capabilities);
        Assert.Empty(result.Capabilities);
    }

    /// <summary>
    /// <see cref="LicenseValidationResult.HasCapability"/> uses ordinal
    /// (case-sensitive) matching so a typo or case-fold can't accidentally
    /// grant access.
    /// </summary>
    [Fact]
    public void HasCapability_MatchesOrdinally()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "professional",
            capabilities = new[] { "tensors:save", "model:load" }
        });
        var result = LicenseValidator.ParseResponse(responseJson, 200);

        // Granted capabilities — exact match wins.
        Assert.True(result.HasCapability("tensors:save"));
        Assert.True(result.HasCapability("model:load"));

        // Not granted.
        Assert.False(result.HasCapability("tensors:load"));
        Assert.False(result.HasCapability("model:save"));

        // Case mismatch — ordinal comparison rejects.
        Assert.False(result.HasCapability("Tensors:Save"));
        Assert.False(result.HasCapability("TENSORS:SAVE"));

        // Empty / null inputs are safe.
        Assert.False(result.HasCapability(string.Empty));
    }

    /// <summary>
    /// <see cref="LicenseValidationResult.Capabilities"/> must reject
    /// in-place mutation by the caller — the cached result is shared
    /// across the validator's grace-period window, and a mutation would
    /// silently re-grant or revoke capabilities behind the guard's back.
    /// </summary>
    [Fact]
    public void Capabilities_ListIsReadOnly()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "professional",
            capabilities = new[] { "tensors:save" }
        });
        var result = LicenseValidator.ParseResponse(responseJson, 200);

        // The runtime type is ReadOnlyCollection<string> (from List.AsReadOnly()).
        // Cast to IList<string> unconditionally — if this cast fails the test
        // must fail too, not silently skip the mutability check.
        var asList = Assert.IsAssignableFrom<System.Collections.Generic.IList<string>>(result.Capabilities);
        Assert.True(asList.IsReadOnly);

        // Behavioural guard: the read-only flag is documentation; the
        // operations must actually throw. ReadOnlyCollection throws
        // NotSupportedException on Add/Remove/Clear/Insert/indexer-set.
        Assert.Throws<System.NotSupportedException>(() => asList.Add("model:save"));
        Assert.Throws<System.NotSupportedException>(() => asList.Remove("tensors:save"));
        Assert.Throws<System.NotSupportedException>(() => asList.Clear());
    }

    // ─── §2a-c: forward-compatible tensor-side bridge ───

    /// <summary>
    /// In the rollout state at the time this PR ships, the AiDotNet.Tensors
    /// NuGet package does NOT yet expose its
    /// <c>AiDotNet.Tensors.Licensing</c> namespace. <see cref="TensorLicenseFlow"/>
    /// must detect this and degrade to no-op rather than throwing — the
    /// upstream guard otherwise becomes a hard dependency on a
    /// not-yet-released Tensors version, which would block this entire PR
    /// chain.
    /// </summary>
    [Fact]
    public void TensorLicenseFlow_GracefullyHandlesAbsentTensorsApi()
    {
        // Both entry points must succeed and return a usable IDisposable
        // even when the tensor-side licensing API is not present.
        var key = new AiDotNetLicenseKey("aidn.testkey1234.signature5678");

        using var keyScope = TensorLicenseFlow.SetActiveLicenseKey(key);
        Assert.NotNull(keyScope);

        using var opScope = TensorLicenseFlow.InternalOperation();
        Assert.NotNull(opScope);

        // Disposing an extra time must be a safe no-op (we'll dispose
        // again at end of scope).
        keyScope.Dispose();
        opScope.Dispose();
    }

    /// <summary>
    /// Null key — the bridge must not blow up. AiModelBuilder constructs
    /// without a key are common (trial-only users), and the bridge gets
    /// called regardless. Also verify the no-op contract behaviourally:
    /// disposal must be safe AND idempotent (the upstream
    /// <see cref="ModelPersistenceGuard.SetActiveLicenseKey"/> nests this
    /// scope inside its own composite, which can dispose more than once
    /// during cleanup of nested BuildAsync failures), and repeat calls
    /// must each return their own usable scope.
    /// </summary>
    [Fact]
    public void TensorLicenseFlow_NullKey_ReturnsNoopScope()
    {
        var scope = TensorLicenseFlow.SetActiveLicenseKey(null);
        Assert.NotNull(scope);

        // Idempotent disposal — must not throw on the second Dispose.
        var disposalEx = Record.Exception(() =>
        {
            scope.Dispose();
            scope.Dispose();
        });
        Assert.Null(disposalEx);

        // Repeat calls return another usable no-op scope (not null, not
        // throwing on dispose). This guards against any future "single
        // shared instance with one-shot dispose" regression that would
        // make the second call unusable after the first scope tore down.
        var second = TensorLicenseFlow.SetActiveLicenseKey(null);
        Assert.NotNull(second);
        var secondDisposalEx = Record.Exception(() => second.Dispose());
        Assert.Null(secondDisposalEx);
    }

    /// <summary>
    /// Issue #1195 acceptance-criterion regression: "trial counter ticks
    /// once on the upstream side, zero on the tensor side." Observe both
    /// halves directly via the upstream <see cref="TrialStateManager"/>
    /// (whose file path the guard reads through
    /// <see cref="ModelPersistenceGuard.SetTestTrialFilePathOverride"/>):
    /// a single <c>EnforceBeforeSave</c> increments
    /// <c>OperationsUsed</c> from 0 → 1, and a follow-up
    /// <c>EnforceBeforeSerialize</c> nested inside an
    /// <see cref="ModelPersistenceGuard.InternalOperation"/> scope leaves
    /// the counter unchanged.
    ///
    /// The tensor-side leg of the assertion is structural until the
    /// AiDotNet.Tensors release with the capability-scoped guard ships:
    /// <see cref="TensorLicenseFlow.IsTensorsLicensingAvailable"/>
    /// reflects current rollout state, and
    /// <see cref="ModelPersistenceGuard.InternalOperation"/> routes
    /// suppression into the tensor layer automatically the moment the API
    /// becomes loadable.
    /// </summary>
    [Fact]
    public void TrialCounter_TicksOnceOnUpstreamLayer_ZeroOnTensorLayer()
    {
        // AIDOTNET_LICENSE_KEY is process-wide. Capture the prior
        // value and restore it in the finally block so this test
        // doesn't leak state into later licensing tests (which would
        // become order-dependent — e.g., a downstream test that
        // expects a real key would silently exercise the trial
        // fallback path instead).
        string? originalLicenseEnv = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", null);

        string tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-issue1195-counter-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        string trialPath = Path.Combine(tempDir, "trial.json");

        try
        {
            using (ModelPersistenceGuard.SetTestTrialFilePathOverride(trialPath))
            {
                var manager = new TrialStateManager(trialPath);

                // Baseline: fresh trial counter at 0.
                int baseline = manager.GetStatus().OperationsUsed;
                Assert.Equal(0, baseline);

                // One user-facing Save outside any InternalOperation scope.
                // The upstream guard MUST tick the counter exactly once.
                ModelPersistenceGuard.EnforceBeforeSave();

                int afterSave = manager.GetStatus().OperationsUsed;
                Assert.Equal(baseline + 1, afterSave);

                // The pairing contract: an upstream InternalOperation
                // scope wrapped around an EnforceBeforeSerialize must
                // suppress the upstream tick — that's the byte-level
                // helper the issue describes as the place double-counting
                // would otherwise happen. The composite scope from the
                // production code ALSO enters the tensor-layer
                // InternalOperation when the API is present, so the
                // tensor-side counter (private to that layer) likewise
                // doesn't tick. Verify the upstream half by counter, the
                // tensor half by the bridge's reflection-resolution flag.
                using (ModelPersistenceGuard.InternalOperation())
                {
                    ModelPersistenceGuard.EnforceBeforeSerialize();
                }

                int afterNestedSerialize = manager.GetStatus().OperationsUsed;
                Assert.Equal(afterSave, afterNestedSerialize);

                // Tensor-layer half. When the AiDotNet.Tensors release
                // with PersistenceGuard ships, IsTensorsLicensingAvailable
                // flips to true and the bridge automatically routes
                // suppression into the tensor layer. Until then, the
                // tensor side has no enforcement at all — verified by
                // the no-op behaviour of TensorLicenseFlow.InternalOperation
                // covered in TensorLicenseFlow_GracefullyHandlesAbsentTensorsApi.
                bool tensorsApiPresent = TensorLicenseFlow.IsTensorsLicensingAvailable;
                using (var bridgeScope = TensorLicenseFlow.InternalOperation())
                {
                    Assert.NotNull(bridgeScope);
                    // When the API is present, the runtime type is the
                    // tensor-layer scope (not NoopScope). When absent,
                    // it is NoopScope. Either way, dispose must succeed.
                    Assert.True(tensorsApiPresent
                        || bridgeScope.GetType().FullName!.EndsWith("NoopScope", StringComparison.Ordinal),
                        "When the Tensors licensing API is absent, the bridge must return its NoopScope sentinel.");
                }
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", originalLicenseEnv);
            try { if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true); }
            catch { /* best effort */ }
        }
    }
}
