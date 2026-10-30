using System;
using System.Collections.Generic;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Known runtime paper-fidelity defects: the failures of the generated option-sensitivity and
/// learns-a-toy-task tests that existed when those tests were introduced. Listed so a NEW failure fails
/// the build while the known ones are worked down.
/// </summary>
/// <remarks>
/// <para>
/// This is the runtime half of the ratchet whose static half is <c>OptionsFidelityBaselines</c> and
/// <c>DefectBaselines</c> in AiDotNet.Generators. The static baselines are computed by the analyzers at
/// build time; these entries can only be learned by running the tests, which happens in CI, so they are
/// filled from the first CI run's failures rather than guessed.
/// </para>
/// <para>
/// Options that ADNOPT001 already proves are never read are NOT repeated here: the scaffold generator
/// reads <c>OptionsFidelityBaselines.UnreadOptions</c> and emits their generated cases as skipped, so
/// the static list is the single source of truth for them.
/// </para>
/// <para>
/// Each list may only shrink. A listed entry that now passes fails its test with an instruction to
/// remove the line, and a listed option whose outcome changes (say from <c>OptionsNotRouted</c> to
/// <c>Insensitive</c>) fails too, so an entry always states the defect that is actually present.
/// </para>
/// </remarks>
public static class RuntimeFidelityBaseline
{
    /// <summary>The recorded outcome of a toy-task test that cleared its bound.</summary>
    public const string ToyTaskLearned = "Learned";

    /// <summary>The recorded outcome of a toy-task test that did not clear its bound.</summary>
    public const string ToyTaskNotLearned = "NotLearned";

    /// <summary>
    /// Option-sensitivity failures, keyed <c>ModelGenericTypeDefinitionFullName::OptionName</c>
    /// (for example <c>AiDotNet.ReinforcementLearning.Agents.DQNAgent`1::WarmupSteps</c>), with the
    /// outcome that was observed.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, OptionSensitivityOutcome> OptionSensitivity =
        new Dictionary<string, OptionSensitivityOutcome>(StringComparer.Ordinal)
        {
        };

    /// <summary>
    /// Learns-a-toy-task failures, keyed <c>ModelGenericTypeDefinitionFullName::TestName</c>.
    /// </summary>
    public static readonly ISet<string> ToyTaskFailures =
        new HashSet<string>(StringComparer.Ordinal)
        {
        };

    /// <summary>
    /// Applies <see cref="ToyTaskFailures"/> to one toy-task outcome: pass, skip as a known defect, or fail.
    /// </summary>
    /// <param name="fixtureType">The concrete test class, for the machine-readable result record.</param>
    /// <param name="modelType">The runtime type of the model under test.</param>
    /// <param name="testName">The toy-task test's name.</param>
    /// <param name="learned">Whether the model cleared the improvement bound.</param>
    /// <param name="failureMessage">What was measured, for the failure or skip message.</param>
    public static void EnforceToyTask(Type fixtureType, Type modelType, string testName, bool learned, string failureMessage)
    {
        string modelKey = OptionSensitivityHarness.ModelKey(modelType);
        string key = modelKey + "::" + testName;
        PaperFidelityResultLog.Record(PaperFidelityCheck.ToyTask, fixtureType, modelKey, testName,
            learned ? ToyTaskLearned : ToyTaskNotLearned, learned ? null : failureMessage);
        bool baselined = ToyTaskFailures.Contains(key);
        if (learned)
        {
            Assert.False(baselined,
                $"'{key}' is listed in RuntimeFidelityBaseline.ToyTaskFailures but now learns the task. Remove its entry so "
                + "the baseline stays an exact inventory.");
            return;
        }

        Skip.If(baselined, "Known defect, listed in RuntimeFidelityBaseline.ToyTaskFailures. " + failureMessage);
        Assert.Fail(failureMessage);
    }
}
