using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Sweeps every model through DeepCopy to find which ones lose construction state.
/// </summary>
/// <remarks>
/// <para>
/// The point is the coverage number, not a pass, for the same reason the layer sweep exists: a
/// green build says nothing about whether a clone carries what it was built with.
/// </para>
/// <para>
/// DeepCopy rebuilds a model by calling its hand-written <c>CreateNewInstance</c> for the shape and
/// then copying parameters in, so <c>CreateNewInstance</c> is the model's factory and every one of
/// the 144 overrides is a hand-maintained argument list. A hyperparameter it forgets to pass is
/// silently defaulted in the copy — the same defect this work removes for options and layers. What
/// this measures is how many models survive that round trip with their parameter count and
/// architecture intact, and stay independent of the original afterwards.
/// </para>
/// <para>
/// Models that cannot be constructed from a standard architecture are reported separately from
/// models that fail to clone. Conflating a shortfall in this harness with a shortfall in the
/// feature is how a coverage number stops meaning anything.
/// </para>
/// </remarks>
public class AllModelsCloneTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes a new instance of the <see cref="AllModelsCloneTests"/> class.</summary>
    /// <param name="output">Sink for the coverage summary.</param>
    public AllModelsCloneTests(ITestOutputHelper output) => _output = output;

    /// <summary>Reports how many models survive a DeepCopy with their construction state.</summary>
    /// <returns>A task representing the test.</returns>
    [Fact(Timeout = 900000)]
    public async System.Threading.Tasks.Task EveryModel_ReportsWhetherItSurvivesADeepCopy()
    {
        await System.Threading.Tasks.Task.Yield();

        var candidates = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(DerivesFromNeuralNetworkBase)
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        var cloned = new List<string>();
        var failed = new List<string>();
        var notConstructed = new List<string>();

        foreach (var open in candidates)
        {
            Type closed;
            try
            {
                closed = open.MakeGenericType(typeof(double));
            }
            catch (Exception)
            {
                notConstructed.Add($"{open.Name}: constraints reject double");
                continue;
            }

            var model = TryConstruct(closed);
            if (model is null)
            {
                notConstructed.Add($"{open.Name}: no constructor takes a standard architecture");
                continue;
            }

            try
            {
                var before = model.ParameterCount;
                var copy = model.DeepCopy() as NeuralNetworkBase<double>;

                if (copy is null)
                {
                    failed.Add($"{open.Name}: DeepCopy returned null");
                    continue;
                }

                if (copy.GetType() != closed)
                {
                    failed.Add($"{open.Name}: copy is {copy.GetType().Name}");
                    continue;
                }

                // A copy with a different parameter count was rebuilt to a different shape, which
                // means CreateNewInstance did not carry every argument that determines size.
                if (copy.ParameterCount != before)
                {
                    failed.Add($"{open.Name}: {copy.ParameterCount} parameters against {before}");
                    continue;
                }

                if (!ReferenceEquals(copy, model) && IsIndependent(model, copy))
                {
                    cloned.Add(open.Name);
                }
                else
                {
                    failed.Add($"{open.Name}: copy is not independent of the original");
                }
            }
            catch (Exception ex)
            {
                var inner = ex.InnerException ?? ex;
                var message = inner.Message;
                failed.Add($"{open.Name}: {inner.GetType().Name}: "
                    + message.Substring(0, Math.Min(90, message.Length)));
            }
            finally
            {
                (model as IDisposable)?.Dispose();
            }
        }

        _output.WriteLine($"model types        : {candidates.Count}");
        _output.WriteLine($"cloned OK          : {cloned.Count}");
        _output.WriteLine($"clone FAILED       : {failed.Count}");
        _output.WriteLine($"not constructed    : {notConstructed.Count} (harness limit, not a clone result)");
        _output.WriteLine(string.Empty);

        foreach (var line in failed) _output.WriteLine($"FAIL  {line}");
        foreach (var line in notConstructed) _output.WriteLine($"skip  {line}");
    }

    /// <summary>Whether writing through one model leaves the other alone.</summary>
    private static bool IsIndependent(
        NeuralNetworkBase<double> original,
        NeuralNetworkBase<double> copy)
    {
        var parameters = original.GetParameters();
        if (parameters.Length == 0) return true;

        var mutated = new Vector<double>(parameters.Length);
        for (var i = 0; i < parameters.Length; i++) mutated[i] = parameters[i] + 1.0;

        copy.UpdateParameters(mutated);

        var after = original.GetParameters();
        for (var i = 0; i < after.Length; i++)
        {
            if (Math.Abs(after[i] - parameters[i]) > 1e-12) return false;
        }

        return true;
    }

    private static NeuralNetworkBase<double>? TryConstruct(Type closed)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);

        // The widest constructor whose every remaining argument is optional, so a model is built
        // through the one carrying the most configuration rather than the narrowest.
        foreach (var ctor in closed.GetConstructors()
                     .OrderByDescending(c => c.GetParameters().Length))
        {
            var formal = ctor.GetParameters();
            if (formal.Length == 0) continue;

            var args = new object?[formal.Length];
            var usable = true;

            for (var i = 0; i < formal.Length && usable; i++)
            {
                if (formal[i].ParameterType.IsInstanceOfType(architecture)) args[i] = architecture;
                else if (formal[i].HasDefaultValue) args[i] = formal[i].DefaultValue;
                else usable = false;
            }

            if (!usable || args.All(a => a is null)) continue;

            try
            {
                return ctor.Invoke(args) as NeuralNetworkBase<double>;
            }
            catch (Exception)
            {
                // A model that rejects the standard architecture is a harness limit, not a defect.
            }
        }

        return null;
    }

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.IsGenericType && b.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
            if (b.Name.StartsWith("NeuralNetworkBase", StringComparison.Ordinal)) return true;
        }

        return false;
    }
}
