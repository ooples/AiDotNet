namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual encoded with real-valued genes, suitable for numerical optimization problems.
/// </summary>
public class RealValuedIndividual : IEvolvable<RealGene, double>
{
    private List<RealGene> _genes = new List<RealGene>();
    private double _fitness;

    /// <summary>
    /// Creates a new individual with random values within the specified range.
    /// </summary>
    /// <param name="dimensionCount">The number of dimensions (genes).</param>
    /// <param name="minValue">The minimum value for initialization.</param>
    /// <param name="maxValue">The maximum value for initialization.</param>
    /// <param name="random">Random number generator for initialization.</param>
    public RealValuedIndividual(int dimensionCount, double minValue, double maxValue, Random random)
    {
        for (int i = 0; i < dimensionCount; i++)
        {
            double value = minValue + (maxValue - minValue) * random.NextDouble();
            _genes.Add(new RealGene(value));
        }
    }

    /// <summary>
    /// Creates a real-valued individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    public RealValuedIndividual(ICollection<RealGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the values of all genes as an array.
    /// </summary>
    /// <returns>An array of double values.</returns>
    public double[] GetValuesAsArray()
    {
        return [.. _genes.Select(g => g.Value)];
    }

    /// <summary>
    /// Updates the step sizes according to Evolutionary Strategies 1/5 success rule.
    /// </summary>
    /// <param name="successRatio">The ratio of successful mutations.</param>
    public void UpdateStepSizes(double successRatio)
    {
        const double c = 0.817; // Constant derived from theoretical considerations

        // Increase step size if success ratio is high, decrease if low
        double adjustmentFactor = successRatio > 0.2 ? 1.0 / c : c;

        foreach (var gene in _genes)
        {
            gene.StepSize *= adjustmentFactor;
        }
    }

    public ICollection<RealGene> GetGenes()
    {
        return _genes;
    }

    public void SetGenes(ICollection<RealGene> genes)
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

    public IEvolvable<RealGene, double> Clone()
    {
        var clone = new RealValuedIndividual([]);
        foreach (var gene in _genes)
        {
            clone._genes.Add(gene.Clone());
        }
        clone._fitness = _fitness;

        return clone;
    }
}
