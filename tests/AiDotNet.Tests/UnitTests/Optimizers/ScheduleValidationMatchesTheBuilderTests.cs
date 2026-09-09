using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The schedule validator and the schedule builder must agree about what is supported (#1928).
/// </summary>
/// <remarks>
/// <para>
/// Validation runs before construction so that a malformed declaration throws instead of training
/// quietly at a constant rate. That only works while its allow-list covers everything the builder
/// can actually map. The two drifted apart the moment a schedule was added on one branch and the
/// validator written on another: NoamHoldAnnealing could be built, was not on the list, and
/// Squeezeformer — which declares it — would have thrown at construction.
/// </para>
/// <para>
/// A list that has to be kept in step with a switch by hand will drift again, so this asserts the
/// agreement directly rather than restating either side.
/// </para>
/// </remarks>
public class ScheduleValidationMatchesTheBuilderTests
{
    /// <summary>A recipe fully specified for the given schedule, so only support is being tested.</summary>
    private static PaperOptimizerAttribute FullySpecified(LearningRateSchedulerType schedule)
        => new(OptimizerKind.Adam)
        {
            Source = "fixture: schedule support probe",
            LearningRate = 0.1,
            MinLearningRate = 0.0,
            WarmupSteps = 4,
            StepSize = 3,
            DecayRate = 0.5,
            Milestones = [2, 4],
            HoldFraction = 0.4,
            Schedule = schedule,
        };

    [Fact]
    public void NoScheduleTheBuilderSupportsIsRejectedByValidation()
    {
        var inconsistent = new List<string>();

        foreach (LearningRateSchedulerType schedule in ((LearningRateSchedulerType[])Enum.GetValues(typeof(LearningRateSchedulerType))))
        {
            var recipe = FullySpecified(schedule);

            bool rejected = false;
            try
            {
                PaperOptimizerFactory.BuildScheduler(recipe, recipe.LearningRate);
            }
            catch (NotSupportedException)
            {
                rejected = true;
            }
            catch (ArgumentException)
            {
                // Under-specified for this schedule rather than unsupported; the fixture supplies
                // every parameter, so this would itself be a contract gap worth surfacing.
                inconsistent.Add($"{schedule} reports missing parameters despite a complete recipe");
                continue;
            }

            // A schedule the builder has an arm for must not be rejected outright. Degrading to a
            // report because the run has no horizon is fine and is not a rejection.
            if (rejected && HasBuilderArm(schedule))
                inconsistent.Add($"{schedule} can be built but validation rejects it");
        }

        Assert.True(inconsistent.Count == 0, string.Join("; ", inconsistent));
    }

    /// <summary>Schedules the recipe path can construct, as opposed to those it refuses.</summary>
    /// <remarks>
    /// Deliberately spelled out rather than derived: this is the independent statement of intent the
    /// test compares the implementation against, and deriving it from the implementation would make
    /// the test agree with any change, including a wrong one.
    /// </remarks>
    private static bool HasBuilderArm(LearningRateSchedulerType schedule)
        => schedule is LearningRateSchedulerType.Constant
            or LearningRateSchedulerType.LinearWarmup
            or LearningRateSchedulerType.TriStage
            or LearningRateSchedulerType.Noam
            or LearningRateSchedulerType.NoamHoldAnnealing
            or LearningRateSchedulerType.Exponential
            or LearningRateSchedulerType.Step
            or LearningRateSchedulerType.MultiStep
            or LearningRateSchedulerType.CosineAnnealing
            or LearningRateSchedulerType.Polynomial
            or LearningRateSchedulerType.OneCycle
            or LearningRateSchedulerType.Cyclic
            or LearningRateSchedulerType.ReduceOnPlateau;

    [Fact]
    public void AScheduleWithNoArmIsStillRefusedLoudly()
    {
        // The other half of the contract: silently accepting an unmappable schedule would let a
        // model declare one and train at a constant rate with nothing saying so.
        var unmappable = ((LearningRateSchedulerType[])Enum.GetValues(typeof(LearningRateSchedulerType)))
            .Where(schedule => !HasBuilderArm(schedule))
            .ToList();

        Assert.NotEmpty(unmappable);

        foreach (LearningRateSchedulerType schedule in unmappable)
        {
            Assert.Throws<NotSupportedException>(
                () => PaperOptimizerFactory.BuildScheduler(FullySpecified(schedule), 0.1));
        }
    }
}
