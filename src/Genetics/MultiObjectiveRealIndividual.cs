namespace AiDotNet.Genetics;

/// <summary>
/// A real-valued individual supporting multi-objective optimization.
/// </summary>
public class MultiObjectiveRealIndividual : RealValuedIndividual, IMultiObjectiveIndividual<double>
{
    private List<double> _objectiveValues = new List<double>();
    private int _rank;
    private double _crowdingDistance;

    public MultiObjectiveRealIndividual(int dimensionCount, double minValue, double maxValue, Random random)
        : base(dimensionCount, minValue, maxValue, random)
    {
    }

    public MultiObjectiveRealIndividual(ICollection<RealGene> genes)
        : base(genes)
    {
    }

    public ICollection<double> GetObjectiveValues()
    {
        return _objectiveValues;
    }

    public void SetObjectiveValues(ICollection<double> values)
    {
        _objectiveValues = [.. values];
    }

    public int GetRank()
    {
        return _rank;
    }

    public void SetRank(int rank)
    {
        _rank = rank;
    }

    public double GetCrowdingDistance()
    {
        return _crowdingDistance;
    }

    public void SetCrowdingDistance(double distance)
    {
        _crowdingDistance = distance;
    }

    /// <summary>
    /// Checks if this individual dominates another individual.
    /// </summary>
    /// <param name="other">The other individual to compare with.</param>
    /// <returns>True if this individual dominates the other, false otherwise.</returns>
    public bool Dominates(MultiObjectiveRealIndividual other)
    {
        bool atLeastOneBetter = false;

        for (int i = 0; i < _objectiveValues.Count; i++)
        {
            if (_objectiveValues[i] > other._objectiveValues[i])
            {
                return false; // This individual is worse in at least one objective
            }

            if (_objectiveValues[i] < other._objectiveValues[i])
            {
                atLeastOneBetter = true; // This individual is better in at least one objective
            }
        }

        return atLeastOneBetter;
    }

    /// <summary>
    /// Creates a deep clone of this multi-objective individual.
    /// </summary>
    /// <returns>A clone of this individual with all properties preserved.</returns>
    public new MultiObjectiveRealIndividual Clone()
    {
        // Deep clone genes to avoid shared mutable state between individuals
        var originalGenes = GetGenes();
        var clonedGenes = new List<RealGene>(originalGenes.Count);
        foreach (var gene in originalGenes)
        {
            clonedGenes.Add(gene.Clone());
        }

        var clone = new MultiObjectiveRealIndividual(clonedGenes);
        clone.SetFitness(GetFitness());
        clone._objectiveValues = [.. _objectiveValues];
        clone._rank = _rank;
        clone._crowdingDistance = _crowdingDistance;
        return clone;
    }
}
