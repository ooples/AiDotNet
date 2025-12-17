namespace AiDotNet.Interfaces;

/// <summary>
/// Represents an individual that can evolve through genetic operations.
/// </summary>
/// <typeparam name="TGene">The type representing a gene in the genetic model.</typeparam>
/// <typeparam name="T">The numeric type used for fitness calculations.</typeparam>
public interface IEvolvable<TGene, T> where TGene : class
{
    /// <summary>
    /// Gets the genes of this individual.
    /// </summary>
    /// <returns>The collection of genes.</returns>
    ICollection<TGene> GetGenes();

    /// <summary>
    /// Sets the genes of this individual.
    /// </summary>
    /// <param name="genes">The genes to set.</param>
    void SetGenes(ICollection<TGene> genes);

    /// <summary>
    /// Gets the fitness of this individual.
    /// </summary>
    /// <returns>The fitness score.</returns>
    T GetFitness();

    /// <summary>
    /// Sets the fitness of this individual.
    /// </summary>
    /// <param name="fitness">The fitness score to set.</param>
    void SetFitness(T fitness);

    /// <summary>
    /// Creates a deep clone of this individual.
    /// </summary>
    /// <returns>A clone of this individual.</returns>
    IEvolvable<TGene, T> Clone();
}
