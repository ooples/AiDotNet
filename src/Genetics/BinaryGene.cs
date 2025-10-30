namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene that holds a binary value (0 or 1).
/// </summary>
public class BinaryGene
{
    /// <summary>
    /// Gets or sets the binary value (0 or 1).
    /// </summary>
    public int Value { get; set; }

    public BinaryGene(int value = 0)
    {
        Value = value > 0 ? 1 : 0; // Ensure binary value
    }

    public BinaryGene Clone()
    {
        return new BinaryGene(Value);
    }

    public override bool Equals(object? obj)
    {
        return obj is BinaryGene gene && gene.Value == Value;
    }

    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }
}
