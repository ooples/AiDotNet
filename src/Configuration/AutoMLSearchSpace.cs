using AiDotNet.AutoML;

namespace AiDotNet.Configuration;

/// <summary>
/// Expert control over the AutoML <b>search space</b> — which model types AutoML tries and
/// the hyperparameter ranges it explores for them.
/// </summary>
/// <remarks>
/// <para>
/// Every property is <b>optional</b>. Leave the whole object (and each property) null for the
/// beginner-friendly, task-appropriate defaults AutoML picks automatically. Set any property to
/// take expert control of exactly that dimension while the rest keep their smart defaults.
/// </para>
/// <para><b>For Beginners:</b> The "search space" is the set of models and settings AutoML is
/// allowed to try. By default AiDotNet chooses a sensible set for your task. If you know your
/// domain, you can narrow it (only try these models), widen it (add candidates), rule some out
/// (exclude these), or hand-tune the ranges it searches — without giving up defaults for the rest.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class AutoMLSearchSpace<T, TInput, TOutput>
{
    /// <summary>
    /// Model types AutoML should search over. When null or empty, AutoML uses the default
    /// candidate set for the resolved task family. Each type must implement
    /// <see cref="Interfaces.IFullModel{T, TInput, TOutput}"/>.
    /// </summary>
    public List<Type>? CandidateModels { get; set; }

    /// <summary>
    /// Model types to remove from the candidate set (whether that set came from
    /// <see cref="CandidateModels"/> or the task-family defaults). Applied after the include set,
    /// so you can start from the defaults and simply rule a few models out. Null = exclude nothing.
    /// </summary>
    public List<Type>? ExcludedModels { get; set; }

    /// <summary>
    /// Hyperparameter search ranges keyed by parameter name (e.g., "learning_rate" → [1e-4, 1e-1]).
    /// When null, each candidate model uses its own default ranges. Set this to hand-tune the ranges
    /// AutoML explores.
    /// </summary>
    public Dictionary<string, ParameterRange>? HyperparameterSpace { get; set; }

    /// <summary>
    /// Resolves the effective candidate list: <paramref name="defaults"/> (or
    /// <see cref="CandidateModels"/> if provided) minus <see cref="ExcludedModels"/>.
    /// </summary>
    public List<Type> ResolveCandidates(IReadOnlyList<Type> defaults)
    {
        var baseSet = (CandidateModels is { Count: > 0 }) ? CandidateModels : defaults.ToList();
        if (ExcludedModels is { Count: > 0 })
        {
            var excluded = new HashSet<Type>(ExcludedModels);
            baseSet = baseSet.Where(t => !excluded.Contains(t)).ToList();
        }
        return baseSet;
    }
}
