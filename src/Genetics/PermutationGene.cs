namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene in a permutation (the index of an element in a sequence).
/// </summary>
public class PermutationGene
{
    /// <summary>
    /// Gets or sets the index value.
    /// </summary>
    public int Index { get; set; }

    public PermutationGene(int index)
    {
        Index = index;
    }

    public PermutationGene Clone()
    {
        return new PermutationGene(Index);
    }

    public override bool Equals(object? obj)
    {
        return obj is PermutationGene gene && gene.Index == Index;
    }

    public override int GetHashCode()
    {
        return Index.GetHashCode();
    }
}
