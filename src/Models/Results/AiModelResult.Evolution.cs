using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using Newtonsoft.Json;

namespace AiDotNet.Models.Results;

/// <summary>
/// Evolution results for AiModelResult.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the redacted summary of the evolution run, if evolution was configured.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Populated when the build went through <c>ConfigureEvolution</c> or <c>ConfigureProgramEvolution</c>. It holds
    /// identifiers, counters, island statuses, archive coordinates, and file paths — never a genome — so it is safe
    /// to log and it survives serialization with the rest of the result.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the report card for the search: why it stopped, how much work it did, the best
    /// score it reached, and the good-and-different candidates it collected. To get the candidates themselves, use
    /// <see cref="GetEvolutionRunResult{TGenome}"/> for a typed run or <see cref="ProgramEvolution"/> for a
    /// program-evolution run.
    /// </para>
    /// </remarks>
    public EvolutionRunSummary? EvolutionSummary { get; internal set; }

    /// <summary>
    /// Gets the program-evolution result, if source code was evolved.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Populated only by <c>ConfigureProgramEvolution</c>. Unlike <see cref="EvolutionSummary"/> this does carry
    /// program text: <c>BestProgram</c> is the winning source in full, and each entry of <c>Elites</c> carries a
    /// copy bounded by <c>ProgramEvolutionOptions.MaxEliteSourceChars</c>, with <c>IsSourceTruncated</c> saying
    /// whether it was cut. That text was produced by a language model and executed as untrusted input, so treat it
    /// as data to review rather than as code to run unread. It is excluded from serialization for the same reason —
    /// a saved model file should not silently carry generated source.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After evolving a program this is where the winning program is. Read
    /// <c>ProgramEvolution.BestProgram.Source</c> for the code and <c>ProgramEvolution.BestQuality</c> for its
    /// score.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public ProgramEvolutionResult? ProgramEvolution { get; internal set; }

    /// <summary>
    /// Gets the engine's own typed run result, boxed because the genome type is not part of this result's signature.
    /// </summary>
    /// <remarks>
    /// Use <see cref="GetEvolutionRunResult{TGenome}"/> rather than this property; it is exposed only so the value
    /// can be carried through the clone paths.
    /// </remarks>
    [JsonIgnore]
    internal object? EvolutionRunResultObject { get; set; }

    /// <summary>
    /// Gets the engine's own run result for the genome type the run used.
    /// </summary>
    /// <typeparam name="TGenome">The genome type passed to <c>ConfigureEvolution</c>.</typeparam>
    /// <returns>
    /// The typed run result, or <see langword="null"/> when no evolution ran or when the run used a different
    /// genome type.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This is the full result: the live archive snapshots, every elite with its genome, the global elite index, and
    /// the pending artifacts. It is not serialized with the model, because a genome can be arbitrarily large and its
    /// type is not known to the result; read it while the built result is still in memory, or persist what you need
    /// from it yourself.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Pass the same genome type you passed to <c>ConfigureEvolution</c>. Getting
    /// <see langword="null"/> back with a run that definitely happened almost always means the type argument does
    /// not match — for a program-evolution run, for instance, the genome type is
    /// <see cref="ProgramGenome"/>.
    /// </para>
    /// </remarks>
    public EvolutionRunResult<TGenome>? GetEvolutionRunResult<TGenome>() =>
        EvolutionRunResultObject as EvolutionRunResult<TGenome>;

    /// <summary>
    /// Gets whether this result came from an evolution run that did not materialize its winner as a model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Evolution normally returns genomes rather than fitting a model. A genome-only result has an empty
    /// <c>OptimizationResult</c> and a <see langword="null"/> <c>Model</c>, so the prediction and persistence
    /// surface cannot work: <c>Predict</c> throws <see cref="InvalidOperationException"/> explaining that no model
    /// was built, and <c>SaveModel</c> writes a payload with no model in it. Everything the run actually produced is
    /// on <see cref="EvolutionSummary"/>, <see cref="ProgramEvolution"/>, and
    /// <see cref="GetEvolutionRunResult{TGenome}"/>.
    /// </para>
    /// <para>
    /// A typed evolution configuration can instead supply a winner model factory; in that case this property is
    /// <see langword="false"/> and the ordinary prediction surface uses the materialized winner.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Check this before calling <c>Predict</c>. If it is <see langword="true"/>, there is no
    /// model to predict with — the answer to your search is the best candidate on the summary, not a fitted model.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public bool IsGenomeOnlyResult => EvolutionSummary is not null && Model is null;
}
