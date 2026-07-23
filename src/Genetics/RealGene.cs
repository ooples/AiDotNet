namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene with a real (double) value.
/// </summary>
public class RealGene
{
    /// <summary>
    /// Gets or sets the real value.
    /// </summary>
    public double Value { get; set; }

    /// <summary>
    /// Gets or sets the mutation step size (used in evolutionary strategies).
    /// </summary>
    public double StepSize { get; set; }

    public RealGene(double value = 0.0, double stepSize = 0.1)
    {
        Value = value;
        StepSize = stepSize;
    }

    public RealGene Clone()
    {
        return new RealGene(Value, StepSize);
    }

    public override bool Equals(object? obj)
    {
        return obj is RealGene gene &&
               Math.Abs(gene.Value - Value) < 1e-10 &&
               Math.Abs(gene.StepSize - StepSize) < 1e-10;
    }

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + Value.GetHashCode();
            hash = hash * 23 + StepSize.GetHashCode();

            return hash;
        }
    }
}
