using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Training;

/// <summary>Accumulates an additive objective on separate tapes before one optimizer update.</summary>
internal static class SummedTapeObjective<T>
{
    private sealed class Term
    {
        internal T Loss;
        internal IReadOnlyList<Tensor<T>> Parameters = Array.Empty<Tensor<T>>();
        internal Dictionary<Tensor<T>, Tensor<T>> Gradients = new(TensorReferenceComparer<Tensor<T>>.Instance);

        internal Term(T loss) => Loss = loss;
    }

    // Keep each graph out of the caller's stack frame before constructing the next one.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Term Differentiate(
        Func<Tensor<T>> objective,
        Func<IReadOnlyList<Tensor<T>>> parameters,
        TapeTrainingStepper<T>.EagerStepHooks? hooks,
        Func<Tensor<T>, Tensor<T>>? scaleLossForBackward)
    {
        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
        using var tape = new GradientTape<T>();
        var loss = objective();
        if (loss is null || loss.Length != 1)
            throw new InvalidOperationException("An additive training objective must return one scalar loss.");
        T unscaledLoss = loss[0];
        var lossForBackward = scaleLossForBackward is not null ? scaleLossForBackward(loss) : loss;
        var result = new Term(unscaledLoss) { Parameters = parameters() };
        var sources = hooks?.GradientSources?.Invoke(result.Parameters) ?? result.Parameters;
        foreach (var pair in tape.ComputeGradients(lossForBackward, sources))
            result.Gradients.Add(pair.Key, new Tensor<T>(pair.Value.ToArray(), pair.Value.Shape.ToArray()));
        return result;
    }

    [AiDotNet.Attributes.ScalarKernel("Sums scalar per-objective loss terms across additive objectives")]
    internal static (T TotalLoss, List<Tensor<T>> SelectedParameters, Dictionary<Tensor<T>, Tensor<T>> AllGradients) AccumulateTerms(
        IReadOnlyList<Func<Tensor<T>>> objectives,
        Func<IReadOnlyList<Tensor<T>>> parameters,
        TapeTrainingStepper<T>.EagerStepHooks? hooks = null,
        Func<Tensor<T>, Tensor<T>>? scaleLossForBackward = null,
        bool requireFiniteTotalLoss = true)
    {
        if (objectives.Count == 0)
            throw new ArgumentException("An additive objective needs at least one term.", nameof(objectives));

        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var all = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var selected = new List<Tensor<T>>();
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        T total = ops.Zero;
        foreach (var objective in objectives)
        {
            var term = Differentiate(objective, parameters, hooks, scaleLossForBackward);
            total = ops.Add(total, term.Loss);
            if (requireFiniteTotalLoss)
            {
                double totalValue = ops.ToDouble(total);
                if (double.IsNaN(totalValue) || double.IsInfinity(totalValue))
                    throw new InvalidOperationException("An additive training objective must have a finite total loss.");
            }

            foreach (var parameter in term.Parameters)
            {
                if (parameter is not null && seen.Add(parameter))
                    selected.Add(parameter);
            }

            foreach (var pair in term.Gradients)
            {
                if (all.TryGetValue(pair.Key, out var sum))
                    engine.TensorAddInPlace(sum, pair.Value);
                else
                    all.Add(pair.Key, pair.Value);
            }
        }

        return (total, selected, all);
    }

    internal static T Step(IReadOnlyList<Func<Tensor<T>>> objectives,
        Func<IReadOnlyList<Tensor<T>>> parameters, Action<TapeStepContext<T>> update,
        Tensor<T> input, Tensor<T> target,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeForward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeLoss,
        TapeTrainingStepper<T>.EagerStepHooks? hooks = null)
    {
        var (total, selected, all) = AccumulateTerms(objectives, parameters, hooks);
        hooks?.OnTapeGradients?.Invoke(all);
        var reached = selected.Where(all.ContainsKey).ToList();
        var gradients = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var parameter in reached) gradients.Add(parameter, all[parameter]);
        hooks?.OnGradients?.Invoke(gradients);
        hooks?.RegularizeAndClip(reached, gradients);
        hooks?.BeforeUpdate?.Invoke();
        update(new TapeStepContext<T>(reached, gradients, total, input, target,
            recomputeForward, recomputeLoss, parameterBuffer: null));
        return total;
    }
}
