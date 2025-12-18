namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual encoded as a permutation, suitable for problems like TSP.
/// </summary>
public class PermutationIndividual : IEvolvable<PermutationGene, double>
{
    private List<PermutationGene> _genes = [];
    private double _fitness;

    /// <summary>
    /// Creates a new permutation individual with a random permutation of the specified size.
    /// </summary>
    /// <param name="size">The size of the permutation.</param>
    /// <param name="random">Random number generator for initialization.</param>
    public PermutationIndividual(int size, Random random)
    {
        // Create a permutation of 0...size-1
        var indices = Enumerable.Range(0, size).ToList();

        // Shuffle using Fisher-Yates algorithm
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Create genes
        foreach (int index in indices)
        {
            _genes.Add(new PermutationGene(index));
        }
    }

    /// <summary>
    /// Creates a permutation individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    public PermutationIndividual(ICollection<PermutationGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the permutation as an array of indices.
    /// </summary>
    /// <returns>An array of indices representing the permutation.</returns>
    public int[] GetPermutation()
    {
        return [.. _genes.Select(g => g.Index)];
    }

    /// <summary>
    /// Applies the Order Crossover (OX) operator.
    /// </summary>
    /// <param name="other">The other parent.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Two child permutations created via order crossover.</returns>
    public (PermutationIndividual, PermutationIndividual) OrderCrossover(
        PermutationIndividual other, Random random)
    {
        int size = _genes.Count;
        int[] parent1 = GetPermutation();
        int[] parent2 = other.GetPermutation();
        int[] child1 = new int[size];
        int[] child2 = new int[size];

        // Select substring boundaries
        int start = random.Next(size);
        int end = start + random.Next(size - start);

        /// Initialize children with -1 (empty)
        for (int i = 0; i < child1.Length; i++)
        {
            child1[i] = -1;
            child2[i] = -1;
        }

        // Copy substring from parents to children
        for (int i = start; i <= end; i++)
        {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }

        // Fill remaining positions
        FillFromOther(parent2, child1, start, end);
        FillFromOther(parent1, child2, start, end);

        // Create new individuals
        return (
            new PermutationIndividual([.. child1.Select(i => new PermutationGene(i))]),
            new PermutationIndividual([.. child2.Select(i => new PermutationGene(i))])
        );
    }

    /// <summary>
    /// Helper method for Order Crossover.
    /// </summary>
    private void FillFromOther(int[] parent, int[] child, int start, int end)
    {
        int size = parent.Length;
        int childPos = (end + 1) % size;

        for (int i = 0; i < size; i++)
        {
            int parentPos = (end + 1 + i) % size;
            int parentValue = parent[parentPos];

            if (Array.IndexOf(child, parentValue, 0, size) == -1)
            {
                child[childPos] = parentValue;
                childPos = (childPos + 1) % size;

                // Skip positions that are already filled
                while (childPos >= start && childPos <= end)
                {
                    childPos = (childPos + 1) % size;
                }
            }
        }
    }

    /// <summary>
    /// Applies a swap mutation by swapping two random positions.
    /// </summary>
    /// <param name="random">Random number generator.</param>
    public void SwapMutation(Random random)
    {
        int size = _genes.Count;
        int pos1 = random.Next(size);
        int pos2 = random.Next(size);

        // Swap genes
        (_genes[pos1], _genes[pos2]) = (_genes[pos2], _genes[pos1]);
    }

    /// <summary>
    /// Applies the inversion mutation by reversing a random subsequence.
    /// </summary>
    /// <param name="random">Random number generator.</param>
    public void InversionMutation(Random random)
    {
        int size = _genes.Count;
        int pos1 = random.Next(size);
        int pos2 = random.Next(size);

        // Ensure pos1 <= pos2
        if (pos1 > pos2)
            (pos1, pos2) = (pos2, pos1);

        // Reverse subsequence
        int steps = (pos2 - pos1 + 1) / 2;
        for (int i = 0; i < steps; i++)
        {
            (_genes[pos1 + i], _genes[pos2 - i]) = (_genes[pos2 - i], _genes[pos1 + i]);
        }
    }

    public ICollection<PermutationGene> GetGenes()
    {
        return _genes;
    }

    public void SetGenes(ICollection<PermutationGene> genes)
    {
        _genes = [.. genes];
    }

    public double GetFitness()
    {
        return _fitness;
    }

    public void SetFitness(double fitness)
    {
        _fitness = fitness;
    }

    public IEvolvable<PermutationGene, double> Clone()
    {
        var clone = new PermutationIndividual([]);
        foreach (var gene in _genes)
        {
            clone._genes.Add(gene.Clone());
        }
        clone._fitness = _fitness;

        return clone;
    }
}
