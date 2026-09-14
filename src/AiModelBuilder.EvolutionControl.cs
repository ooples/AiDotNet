using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    private EvolutionRunControl? _evolutionRunControl;
    private IEvolutionObserver<ProgramGenome>? _programEvolutionObserver;

    /// <summary>Connects a one-run graceful-stop handle before building a typed or program evolution run.</summary>
    /// <param name="control">The caller-owned handle; it may receive a stop request before execution begins.</param>
    /// <returns>This concrete builder. Set this before interface-returning Configure calls when using a fluent chain.</returns>
    /// <remarks>This does not enable checkpointing, bypass coordinated-ledger restrictions or implement in-flight recovery.</remarks>
    public AiModelBuilder<T, TInput, TOutput> WithEvolutionControl(EvolutionRunControl control)
    {
        _evolutionRunControl = control ?? throw new ArgumentNullException(nameof(control));
        return this;
    }

    /// <summary>Adds a caller-owned program observer alongside configured output, artifact and trace observers.</summary>
    /// <param name="observer">The observer, invoked under the engine's bounded event and exception policy.</param>
    /// <returns>This concrete builder.</returns>
    /// <remarks>Observers must not block awaiting a later engine event. No program source is redacted for this in-process observer.</remarks>
    public AiModelBuilder<T, TInput, TOutput> ObserveProgramEvolution(IEvolutionObserver<ProgramGenome> observer)
    {
        _programEvolutionObserver = observer ?? throw new ArgumentNullException(nameof(observer));
        return this;
    }
}
