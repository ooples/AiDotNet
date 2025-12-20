namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual encoded with binary genes, suitable for classic GA problems.
/// </summary>
public class BinaryIndividual : IEvolvable<BinaryGene, double>
{
    private List<BinaryGene> _genes = [];
    private double _fitness;

    /// <summary>
    /// Creates a new binary individual with the specified chromosome length.
    /// </summary>
    /// <param name="length">The number of genes (bits) in the chromosome.</param>
    /// <param name="random">Random number generator for initialization.</param>
    public BinaryIndividual(int length, Random random)
    {
        for (int i = 0; i < length; i++)
        {
            _genes.Add(new BinaryGene(random.Next(2))); // Randomly 0 or 1
        }
    }

    /// <summary>
    /// Creates a binary individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    public BinaryIndividual(ICollection<BinaryGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the binary value of this individual as an integer.
    /// </summary>
    /// <returns>The integer value represented by this binary string.</returns>
    public int GetValueAsInt()
    {
        int value = 0;
        for (int i = 0; i < _genes.Count; i++)
        {
            if (_genes[i].Value == 1)
            {
                value |= (1 << i);
            }
        }

        return value;
    }

    /// <summary>
    /// Gets the binary value of this individual as a double in the range [0,1].
    /// </summary>
    /// <returns>A double value between 0 and 1.</returns>
    public double GetValueAsNormalizedDouble()
    {
        double maxValue = Math.Pow(2, _genes.Count) - 1;
        return GetValueAsInt() / maxValue;
    }

    /// <summary>
    /// Maps the binary string to a double value within the specified range.
    /// </summary>
    /// <param name="min">The minimum value of the range.</param>
    /// <param name="max">The maximum value of the range.</param>
    /// <returns>A double value between min and max.</returns>
    public double GetValueMapped(double min, double max)
    {
        return min + (GetValueAsNormalizedDouble() * (max - min));
    }

    public ICollection<BinaryGene> GetGenes()
    {
        return _genes;
    }

    public void SetGenes(ICollection<BinaryGene> genes)
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

    public IEvolvable<BinaryGene, double> Clone()
    {
        var clone = new BinaryIndividual([]);
        foreach (var gene in _genes)
        {
            clone._genes.Add(gene.Clone());
        }
        clone._fitness = _fitness;

        return clone;
    }
}
