using System.Reflection;
using AiDotNet.Finance.Base;
using AiDotNet.Finance.Forecasting.Foundation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards the two structural facts that <c>NeuralNetworkModelTestBase.PredictLeavesLossDomain</c>
/// is built from, so it cannot decay into a check that silently answers "no" for everything.
/// </summary>
/// <remarks>
/// <para>
/// <b>What the guarded condition does.</b> The shared model invariants read a model's
/// <c>DefaultLossFunction</c> to decide both what domain its target belongs in and how to score its
/// prediction. That is right for almost every model and wrong for a tokenized forecaster, whose
/// declared loss describes an <i>internal</i> objective: Chronos quantizes real-valued future
/// values onto the context's scale and supervises vocabulary logits with cross-entropy, while
/// <c>Predict</c> returns detokenized forecasts in the series' own units. Scoring
/// cross-entropy-with-logits on those forecasts reported a training loss of 384.375 for a model
/// whose head actually emits logits within [-2.6, 2.6].
/// </para>
/// <para>
/// <b>Why this test exists.</b> The condition is "declares its own <c>ForwardNativeForTraining</c>
/// <i>and</i> carries a logits loss", resolved by reflection on a method name. A rename or a
/// removed override does not break the build and does not fail any assertion — the reflection just
/// returns false, the model silently falls back to the wrong metric, and the regression returns
/// wearing the same 384.375 it arrived with. Both halves are therefore pinned here.
/// </para>
/// </remarks>
public sealed class LossDomainMismatchInventoryTests
{
    private const string TrainingForwardName = "ForwardNativeForTraining";

    private const BindingFlags DeclaredMembers =
        BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly;

    [Fact]
    public void FinancialModelBase_StillDeclaresTheTrainingForwardTheConditionWalksTo()
    {
        var declared = typeof(FinancialModelBase<double>).GetMethod(TrainingForwardName, DeclaredMembers);

        Assert.True(declared is not null,
            $"{nameof(FinancialModelBase<double>)} no longer declares {TrainingForwardName}. "
            + "PredictLeavesLossDomain walks up the hierarchy and stops at this declaration to tell "
            + "an override apart from the default. With the name gone the walk stops at nothing, "
            + "every model answers 'does not leave the loss domain', and Chronos silently goes back "
            + "to being scored with cross-entropy on detokenized forecasts. Rename here and in "
            + "NeuralNetworkModelTestBase.PredictLeavesLossDomain together.");

        Assert.True(declared!.IsVirtual,
            $"{TrainingForwardName} is no longer virtual, so no model can override it and the "
            + "condition can never be true for anyone.");
    }

    [Fact]
    public void Chronos_StillDeclaresItsOwnTrainingForward()
    {
        var declared = typeof(Chronos<double>).GetMethod(TrainingForwardName, DeclaredMembers);

        Assert.True(
            declared is not null,
            $"Chronos no longer declares its own {TrainingForwardName}. That override is what marks "
            + "its training forward (vocabulary logits) as a different domain from its inference "
            + "forward (detokenized forecasts). Without it PredictLeavesLossDomain returns false and "
            + "the generic invariants resume scoring cross-entropy-with-logits against real-valued "
            + "forecasts. If the override was removed deliberately, the condition needs rewriting "
            + "rather than deleting.");

        Assert.True(declared!.IsVirtual,
            $"Chronos declares {TrainingForwardName}, but it is not virtual and therefore cannot "
            + "replace the training path dispatched by FinancialModelBase.");
        Assert.Same(typeof(FinancialModelBase<double>), declared.GetBaseDefinition().DeclaringType);
    }

    /// <summary>
    /// The condition deliberately requires BOTH halves. This pins the fact that the override alone
    /// is common — so a future change that drops the loss half would sweep 30 unrelated forecasters
    /// into the mismatch branch rather than the one model that belongs there.
    /// </summary>
    [Fact]
    public void DeclaringTheTrainingForward_IsCommonEnoughThatTheLossHalfIsLoadBearing()
    {
        // Walk the base chain by GENERIC TYPE DEFINITION. Assembly.GetTypes() hands back open
        // definitions (Chronos`1, not Chronos<double>), and
        // typeof(FinancialModelBase<double>).IsAssignableFrom(Chronos`1) is false for every one of
        // them -- an earlier revision of this test used exactly that and counted 0 models, which is
        // what its own assertion caught. The production condition in PredictLeavesLossDomain is not
        // affected: it inspects a CLOSED runtime type off a live instance.
        var declaringModels = typeof(FinancialModelBase<double>).Assembly
            .GetTypes()
            .Where(DerivesFromFinancialModelBase)
            .Where(type => type.GetMethod(TrainingForwardName, DeclaredMembers) is not null)
            .ToArray();

        Assert.True(declaringModels.Length > 1,
            "Only " + declaringModels.Length + " model declares its own " + TrainingForwardName
            + ". The two-part condition exists because the override on its own is widespread and "
            + "does not imply a change of loss domain; if that is no longer true, the simpler "
            + "condition should replace it deliberately.");
    }

    private static bool DerivesFromFinancialModelBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType
                && t.GetGenericTypeDefinition() == typeof(FinancialModelBase<>))
            {
                return true;
            }
        }

        return false;
    }
}
