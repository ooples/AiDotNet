using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Guards the defect class introduced by the options-surface migration (issue #2090): a family
/// base whose <c>ValidateCore</c> requires a property that not every member of the family sets.
/// </summary>
/// <remarks>
/// <para>
/// Each concrete options class assigns its paper's values in its parameterless constructor, and
/// exposes a <c>Validate()</c> that calls its family's <c>ValidateCore</c>. If the base requires
/// more than every member actually assigns, the leaf that does not assign it throws
/// <c>ArgumentException</c> from its model's constructor — at the model's defaults, with the user
/// having configured nothing.
/// </para>
/// <para>
/// This happened four times during the migration (GanOptions, DocumentNeuralNetworkOptions,
/// VideoHyperparameterOptions, VisionLanguageModelOptions), and the fourth shipped in phase 3 and
/// was not noticed until phase 7, because the only thing that detected it was running the model
/// tests for that family — hours of them, sampled rather than exhaustive.
/// </para>
/// <para>
/// The defect is fully enumerable instead: construct every options class at its defaults and
/// validate it. That is exhaustive, takes about a second, and cannot regress silently. The rule
/// it encodes is the one written into each family base: <b>a base may only require what every
/// member has.</b>
/// </para>
/// </remarks>
public class OptionsDefaultsValidateTests
{
    /// <summary>
    /// Every concrete options class must validate successfully at the defaults its own
    /// parameterless constructor assigns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A failure here means one of two things, and the exception message names the property:
    /// either the family base requires something this leaf does not set (narrow the base, or
    /// move the requirement into the <c>Validate()</c> of the leaves that need it), or this
    /// leaf's constructor is missing its paper default (assign it).
    /// </para>
    /// </remarks>
    [Fact]
    public void EveryOptionsClassValidatesAtItsOwnDefaults()
    {
        var failures = new List<string>();
        int validated = 0;

        foreach (var type in GetConcreteOptionsTypes())
        {
            object instance;
            try
            {
                instance = Activator.CreateInstance(type)
                    ?? throw new InvalidOperationException("Activator returned null.");
            }
            catch (Exception ex)
            {
                failures.Add($"{type.Name}: constructing at defaults threw {Describe(ex)}");
                continue;
            }

            var validate = type.GetMethod(
                "Validate", BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
            if (validate == null) continue;

            validated++;
            try
            {
                validate.Invoke(instance, null);
            }
            catch (TargetInvocationException ex) when (ex.InnerException != null)
            {
                failures.Add($"{type.Name}: Validate() threw {Describe(ex.InnerException)}");
            }
            catch (Exception ex)
            {
                failures.Add($"{type.Name}: Validate() threw {Describe(ex)}");
            }
        }

        Assert.True(validated > 0, "No options class exposed a public parameterless Validate(); "
            + "the reflection query is wrong, not the code under test.");

        if (failures.Count > 0)
        {
            var message = new StringBuilder()
                .AppendLine($"{failures.Count} of {validated} options classes fail validation at their own defaults.")
                .AppendLine("A family base may only require what every member of the family assigns.")
                .AppendLine();
            foreach (var failure in failures.OrderBy(f => f, StringComparer.Ordinal))
            {
                message.AppendLine("  " + failure);
            }

            Assert.Fail(message.ToString());
        }
    }

    /// <summary>
    /// The number of options classes with no <c>Validate()</c> may fall but never rise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A class with no <c>Validate()</c> — its own or inherited — has its defaults checked by
    /// nothing, so a property its constructor forgot to assign reaches the model as a zero and
    /// produces a zero-width layer far from the cause.
    /// </para>
    /// <para>
    /// This is a ratchet rather than a requirement of zero, deliberately. Demanding that every
    /// class expose a <c>Validate()</c> would be satisfied by adding one that checks nothing,
    /// which buys no safety and reads as though it did — the same over-strictness that made a
    /// family base require what its members lacked, five times over. What is worth asserting is
    /// that the uncovered set shrinks: each entry below is a model whose published defaults
    /// nothing checks, and the two family bases here validate nothing today because neither
    /// family shares a universal knob.
    /// </para>
    /// </remarks>
    [Fact]
    public void OptionsClassesWithoutValidate_DoesNotRegress()
    {
        // Raised from 14 to 23 deliberately, and this is the one direction the number is not
        // supposed to move — so the reason belongs here rather than in a commit message.
        // Re-parenting PhysicsInformedOptions onto ModelHyperparameterOptions (#2090, PINN
        // phase) brought twelve physics-informed options classes into this scan for the first
        // time. Nine of them are property-less shells for models not yet migrated:
        // DeepOperatorNetwork, DeepRitzMethod, FourierNeuralOperator, GraphNeuralOperator,
        // LagrangianNeuralNetwork, MultiScalePINN, UniversalDifferentialEquations,
        // VariationalPINN, and PhysicsInformedOptions itself. They gained no defect; they
        // became visible. Giving each a Validate() that checks nothing would satisfy this
        // test while buying no safety, which is why it is a ratchet and not a demand for zero.
        // Each falls off the list when its model is migrated.
        //
        // Raised again, 23 -> 81, by the same mechanism one cluster later. The maxGradNorm
        // migration re-parented FinancialNeuralNetworkOptions onto ModelHyperparameterOptions so
        // that RiskModelOptions and everything beneath it would inherit MaxGradNorm; that brought
        // the financial, tabular and synthetic-data options classes into this scan for the first
        // time. They gained no defect either — CTGANOptions and TabNetOptions have always had no
        // Validate(); nothing could see them before.
        //
        // The precedent above governs: adding 58 Validate() methods that check nothing would turn
        // this test green and buy no safety at all. Each falls off the list when its model is
        // migrated and there are values worth requiring.
        const int UncoveredBaseline = 81;

        var all = GetConcreteOptionsTypes().ToList();
        var missing = all
            .Where(t => t.GetMethod(
                "Validate", BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null) == null)
            .Select(t => t.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();

        Assert.True(
            all.Count >= 100,
            $"Expected the migrated options surface to be well over 100 classes, found {all.Count}. "
                + "A collapsed count means the reflection query is wrong, not that the surface shrank.");

        Assert.True(
            missing.Count <= UncoveredBaseline,
            $"{missing.Count} options classes expose no Validate(), up from {UncoveredBaseline}. "
                + "Give the new one a Validate() that requires the values its model actually reads: "
                + string.Join(", ", missing));
    }

    private static string Describe(Exception ex)
    {
        string text = ex.Message;
        int newline = text.IndexOfAny(new[] { '\r', '\n' });
        if (newline >= 0) text = text.Substring(0, newline);
        return $"{ex.GetType().Name}: {text}";
    }

    /// <summary>
    /// Every concrete, constructible options class in the model assembly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Resolved by assignability to <see cref="ModelHyperparameterOptions"/> rather than by name
    /// or namespace, for the same reason the ratchet resolves models by assignability: options
    /// classes live in several namespaces and are not reliably suffixed.
    /// </para>
    /// <para>
    /// Generic options classes are closed over <see cref="double"/>, the numeric type the model
    /// tests use. Classes without a public parameterless constructor are skipped: the pattern
    /// under test is "defaults assigned in the parameterless constructor", so a class that has
    /// none is not making that claim.
    /// </para>
    /// </remarks>
    private static IEnumerable<Type> GetConcreteOptionsTypes()
    {
        Type[] types;
        try
        {
            types = typeof(ModelHyperparameterOptions).Assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException ex)
        {
            types = ex.Types.Where(t => t != null).ToArray()!;
        }

        foreach (var type in types)
        {
            if (type == null || type.IsAbstract || type.IsInterface || !type.IsClass) continue;
            if (!type.IsPublic && !type.IsNestedPublic) continue;
            if (!typeof(ModelHyperparameterOptions).IsAssignableFrom(type)) continue;

            Type candidate = type;
            if (type.IsGenericTypeDefinition)
            {
                if (type.GetGenericArguments().Length != 1) continue;
                Type closed;
                try
                {
                    closed = type.MakeGenericType(typeof(double));
                }
                catch (ArgumentException)
                {
                    // A constraint double does not satisfy; nothing to construct.
                    continue;
                }

                candidate = closed;
            }

            if (candidate.GetConstructor(Type.EmptyTypes) == null) continue;

            yield return candidate;
        }
    }
}
