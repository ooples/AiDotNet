namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene that corresponds to a parameter in a machine learning model.
/// </summary>
/// <typeparam name="T">The numeric type used for the value.</typeparam>
public class ModelParameterGene<T>
{
    /// <summary>
    /// Gets the index of the parameter in the model's parameter vector.
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the parameter value.
    /// </summary>
    public T Value { get; }

    /// <summary>
    /// Initializes a new instance of the ModelParameterGene class.
    /// </summary>
    /// <param name="index">The index of the parameter.</param>
    /// <param name="value">The value of the parameter.</param>
    public ModelParameterGene(int index, T value)
    {
        Index = index;
        Value = value;
    }

    /// <summary>
    /// Creates a clone of this gene.
    /// </summary>
    /// <returns>A new ModelParameterGene with the same index and value.</returns>
    public ModelParameterGene<T> Clone()
    {
        return new ModelParameterGene<T>(Index, Value);
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current gene.
    /// </summary>
    /// <param name="obj">The object to compare with the current gene.</param>
    /// <returns>true if the specified object is equal to the current gene; otherwise, false.</returns>
    public override bool Equals(object? obj)
    {
        return obj is ModelParameterGene<T> gene &&
               gene.Index == Index &&
               Equals(gene.Value, Value);
    }

    /// <summary>
    /// Returns a hash code for this gene.
    /// </summary>
    /// <returns>A hash code for the current gene.</returns>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + Index.GetHashCode();
            hash = hash * 23 + (Value?.GetHashCode() ?? 0);

            return hash;
        }
    }
}
