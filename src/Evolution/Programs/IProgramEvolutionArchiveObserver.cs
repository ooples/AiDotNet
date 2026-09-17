namespace AiDotNet.Evolution.Programs;

/// <summary>A program observer that also inspects the run's archive views.</summary>
/// <remarks>
/// The builder registers each island before the engine starts or restores its checkpoint. Read the views only
/// during an event callback, when engine mutation is serialized; publish copied snapshots to other threads.
/// A view is not an immutable snapshot or a security boundary. Use a fresh observer for each run.
/// </remarks>
public interface IProgramEvolutionArchiveObserver : IEvolutionObserver<ProgramGenome>
{
    /// <summary>Registers one island's view before the run begins.</summary>
    /// <param name="archive">The live, caller-read-only view.</param>
    void AddArchive(IEvolutionArchiveView<ProgramGenome> archive);
}
