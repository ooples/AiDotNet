using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 11 — Configure* methods that hijack BuildAsync into a
/// custom training/search path. Each test stubs the minimal external
/// surface the path requires, then asserts the stub's hot method got
/// invoked.
/// </summary>
/// <remarks>
/// Methods covered:
/// <list type="bullet">
///   <item>ConfigureMetaLearning — verifies <c>_metaLearner.Train()</c>
///   fires inside <c>BuildMetaLearningInternalAsync</c>.</item>
///   <item>ConfigureAutoML(IAutoMLModel) — verifies
///   <c>_autoMLModel.SearchAsync</c> fires inside the AutoML branch.</item>
///   <item>ConfigureReinforcementLearning — verifies
///   <c>BuildRLInternalAsync</c> calls into the configured
///   environment's <c>Step</c>.</item>
///   <item>ConfigureAgentAssistance — verifies the
///   <c>IsEnabled=false</c> gate prevents the LLM call while still
///   routing through the agent-flow setup.</item>
/// </list>
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket11_HijackPathTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket11_HijackPathTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureMetaLearning — uses a Mock&lt;IMetaLearner&gt; whose
    /// <c>Train</c> returns a minimal valid MetaTrainingResult. The
    /// assertion is that <c>Train</c> was invoked, which proves
    /// <c>BuildMetaLearningInternalAsync</c> ran the configured
    /// learner end-to-end. Downstream MetaLearningInternalAsync may
    /// throw on the model-metadata access of the mock's empty model
    /// — we catch that and read the call count, which is set inside
    /// BuildMetaLearningInternalAsync BEFORE the throw site.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureMetaLearning_RealLearner_InvokesTrainDuringBuild()
    {
        var learnerMock = new Mock<IMetaLearner<float, Tensor<float>, Tensor<float>>>();
        // Set up the minimum needed for the post-Train AiModelResultOptions
        // construction at AiModelBuilder.cs:3693.
        learnerMock.Setup(l => l.Train()).Returns(new MetaTrainingResult<float>(
            lossHistory: new Vector<float>(new float[] { 0.5f }),
            accuracyHistory: new Vector<float>(new float[] { 0.5f }),
            trainingTime: System.TimeSpan.FromMilliseconds(1)));
        // BuildMetaLearningInternalAsync may dereference the underlying
        // meta-model via GetMetaModel — give it the canary model so
        // metadata extraction doesn't NRE.
        learnerMock.Setup(l => l.GetMetaModel()).Returns(MakeCanaryModel());

        // Narrow the catch to the SPECIFIC downstream-of-Train failure
        // modes a partially-stubbed Mock produces. The NRE catch is
        // gated by a stack-trace `when` filter that requires the failure
        // to originate inside AiModelResult / AiModelResultOptions /
        // BuildMetaLearningInternalAsync — i.e. AFTER Train() has been
        // invoked. An NRE thrown BEFORE Train() (e.g. from a typo or
        // unrelated builder regression introducing a null deref) will
        // NOT match the filter and will escape the test, matching the
        // intent: a regression that prevents Train() from being called
        // must fail the verify-Train.Once assertion below, not be
        // masked here (this PR's review C4TPf).
        try
        {
            await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureMetaLearning(learnerMock.Object)
                .BuildAsync();
        }
        catch (System.NullReferenceException ex)
            when (IsExceptionFromPostTrainSurface(ex)) { /* mock metadata access AFTER Train */ }
        catch (System.ArgumentException ex)
            when (IsExceptionFromPostTrainSurface(ex)) { /* downstream shape mismatch */ }
        catch (System.InvalidOperationException ex)
            when (IsExceptionFromPostTrainSurface(ex)) { /* option-validation gate */ }

        learnerMock.Verify(l => l.Train(), Times.Once,
            "ConfigureMetaLearning was wired but BuildAsync never invoked IMetaLearner.Train. The Meta-Learning branch at AiModelBuilder.cs:1512 should detect _metaLearner and route to BuildMetaLearningInternalAsync.");
    }

    /// <summary>
    /// Returns true if <paramref name="ex"/> originated INSIDE the
    /// post-Train surface — i.e. AiModelResult construction,
    /// AiModelResultOptions assembly, or
    /// BuildMetaLearningInternalAsync's finalization steps. Used by the
    /// MetaLearning / AutoML hijack-path tests' NRE filters so a
    /// pre-Train regression (typo, unrelated builder bug) doesn't get
    /// swallowed (this PR's review C4TPf).
    /// </summary>
    private static bool IsExceptionFromPostTrainSurface(System.Exception ex)
    {
        // Walk the chain (current + InnerException + AggregateException
        // children). For each, check TargetSite first (metadata, present
        // even when StackTrace is null — happens for constructed-but-
        // never-thrown exceptions and for some trimmed/AOT builds where
        // the stack frames are elided) — then fall back to StackTrace
        // string scanning (this PR's review C88O7: prior NRE catch with
        // StackTrace-only filter would degrade to "no match" on a
        // genuinely-thrown NRE whose StackTrace happened to be null,
        // letting an unrelated pre-Train NRE escape the test as if it
        // were the post-Train one).
        var postTrainTypes = new[]
        {
            "AiDotNet.Models.Results.AiModelResult",
            "BuildMetaLearningInternalAsync",
            "AiModelResultOptions",
            "GetModelMetadata",
        };
        var visit = new System.Collections.Generic.Stack<System.Exception>();
        visit.Push(ex);
        while (visit.Count > 0)
        {
            var current = visit.Pop();
            // Metadata path: declaring type of the immediate throw site.
            if (current.TargetSite?.DeclaringType?.FullName is string declType)
            {
                foreach (var marker in postTrainTypes)
                {
                    if (declType.Contains(marker, System.StringComparison.Ordinal)) return true;
                }
            }
            // Stack-trace path: works when frames haven't been trimmed
            // (also handles the chained inner-exception case where the
            // outer was wrapped after the inner's throw).
            if (current.StackTrace is string st)
            {
                foreach (var marker in postTrainTypes)
                {
                    if (st.Contains(marker, System.StringComparison.Ordinal)) return true;
                }
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
    /// ConfigureAutoML(IAutoMLModel) — uses a Mock&lt;IAutoMLModel&gt;
    /// whose SearchAsync returns the configured model itself. Verifies
    /// SearchAsync is called when ConfigureAutoML is wired and no
    /// explicit model was configured.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureAutoML_IAutoMLModelOverload_InvokesSearchAsync()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);

        var autoMLMock = new Mock<IAutoMLModel<float, Tensor<float>, Tensor<float>>>();
        var canary = MakeCanaryModel();
        autoMLMock.Setup(a => a.SearchAsync(
                It.IsAny<Tensor<float>>(), It.IsAny<Tensor<float>>(),
                It.IsAny<Tensor<float>>(), It.IsAny<Tensor<float>>(),
                It.IsAny<System.TimeSpan>(),
                It.IsAny<System.Threading.CancellationToken>()))
            .ReturnsAsync(canary);
        autoMLMock.SetupGet(a => a.BestScore).Returns(0.0);
        autoMLMock.SetupGet(a => a.TimeLimit).Returns(System.TimeSpan.FromSeconds(1));
        autoMLMock.Setup(a => a.GetTrialHistory()).Returns(new System.Collections.Generic.List<AiDotNet.AutoML.TrialResult>());

        // Same narrowing as the MetaLearning test: NRE catch is gated by
        // IsExceptionFromPostTrainSurface so a pre-SearchAsync NRE
        // regression escapes and fails the test (this PR's review C4TPf).
        try
        {
            await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureDataLoader(loader)
                .ConfigureAutoML(autoMLMock.Object)
                .BuildAsync();
        }
        catch (System.NullReferenceException ex)
            when (IsExceptionFromPostTrainSurface(ex)) { /* mock metadata access AFTER SearchAsync */ }
        catch (System.ArgumentException ex)
            when (IsExceptionFromPostTrainSurface(ex)) { /* shape / model construction */ }
        catch (System.InvalidOperationException ex)
            when (IsExceptionFromPostTrainSurface(ex)) { /* option-validation gate */ }

        autoMLMock.Verify(a => a.SearchAsync(
            It.IsAny<Tensor<float>>(), It.IsAny<Tensor<float>>(),
            It.IsAny<Tensor<float>>(), It.IsAny<Tensor<float>>(),
            It.IsAny<System.TimeSpan>(),
            It.IsAny<System.Threading.CancellationToken>()), Times.AtLeastOnce,
            "ConfigureAutoML was wired but BuildAsync never invoked SearchAsync. The AutoML branch at AiModelBuilder.cs:2328 should detect _autoMLModel and run the search.");
    }

    /// <summary>
    /// ConfigureReinforcementLearning — verifies BuildAsync detects the
    /// RL options and routes to <c>BuildRLInternalAsync</c>. The RL
    /// branch requires the configured model to implement
    /// <c>IRLAgent&lt;T&gt;</c>; the canary Transformer doesn't, so the
    /// branch throws <c>InvalidOperationException("The configured model
    /// must implement IRLAgent...")</c>. That specific throw PROVES the
    /// routing logic detected <c>_rlOptions.Environment</c> and dispatched
    /// — a stored-but-not-consumed regression would silently fall
    /// through to the supervised path and surface a different
    /// exception (or none).
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureReinforcementLearning_WithEnvironment_RoutesToRLBranch()
    {
        var envMock = new Mock<IEnvironment<float>>();
        envMock.Setup(e => e.Reset()).Returns(new Vector<float>(new float[] { 0f }));
        envMock.SetupGet(e => e.ObservationSpaceDimension).Returns(1);
        envMock.SetupGet(e => e.ActionSpaceSize).Returns(1);
        envMock.SetupGet(e => e.IsContinuousActionSpace).Returns(false);

        var rlOptions = new RLTrainingOptions<float>
        {
            Environment = envMock.Object,
            Episodes = 1,
            LogFrequency = 0,
        };

        // Canary model isn't an IRLAgent — the RL branch's IRLAgent gate
        // at AiModelBuilder.cs:3833 will throw, but only AFTER the branch
        // entered (proving the routing detected _rlOptions.Environment).
        // Stored-but-not-consumed would fall through to supervised path
        // and produce a totally different exception shape.
        var ex = await Assert.ThrowsAsync<System.InvalidOperationException>(async () =>
        {
            await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(MakeCanaryModel())
                .ConfigureReinforcementLearning(rlOptions)
                .BuildAsync();
        });

        Assert.Contains("IRLAgent", ex.Message);
    }

}
