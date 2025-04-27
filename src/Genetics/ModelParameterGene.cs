namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene that corresponds to a parameter in a machine learning model.
/// </summary>
/// <typeparam name="T">The numeric type used for the value.</typeparam>
/// <remarks>
/// <para>
/// The ModelParameterGene class is a specialized gene type designed for evolving machine learning models.
/// Each gene directly represents a single parameter (like a weight or coefficient) in the model,
/// allowing genetic algorithms to optimize model parameters through evolution.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a specific trait in the DNA of a machine learning model.
/// 
/// Just as a gene in human DNA might control eye color or height:
/// - Each ModelParameterGene controls one specific number in your machine learning model
/// - The Index tells us which parameter this gene controls (like "parameter #5")
/// - The Value contains the actual number used by the model (like "1.43")
/// 
/// When genetic algorithms evolve these genes, they're essentially tweaking specific 
/// numbers in your model to find the combination that works best for your problem.
/// </para>
/// </remarks>
public class ModelParameterGene<T>
{
    /// <summary>
    /// Gets the index of the parameter in the model's parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the position of the parameter within the model's parameter collection.
    /// It serves as an identifier that links this gene to a specific parameter in the model.
    /// </para>
    /// <para><b>For Beginners:</b> This is like an address or location.
    /// 
    /// Think of your model as having many knobs that can be adjusted:
    /// - The Index tells you exactly which knob this gene controls
    /// - It's like saying "this gene controls the 5th knob on the machine"
    /// - This allows the genetic algorithm to know precisely which part of the model it's changing
    /// 
    /// Without this index, we wouldn't know which parameter in the model to update
    /// when a gene's value changes during evolution.
    /// </para>
    /// </remarks>
    public int Index { get; }

    /// <summary>
    /// Gets the parameter value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the actual value of the model parameter represented by this gene.
    /// This value will be used by the model during prediction and will be modified by
    /// genetic operations like mutation and crossover during evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual setting or measurement.
    /// 
    /// Using our knob analogy:
    /// - If Index tells you which knob to adjust, Value tells you what position to set it to
    /// - It might be a number like 0.8 or -2.5
    /// - This value directly affects how the model behaves and makes predictions
    /// - During evolution, these values get adjusted to find the best model performance
    /// 
    /// These values are what the genetic algorithm is actually optimizing to improve the model.
    /// </para>
    /// </remarks>
    public T Value { get; }

    /// <summary>
    /// Initializes a new instance of the ModelParameterGene class.
    /// </summary>
    /// <param name="index">The index of the parameter.</param>
    /// <param name="value">The value of the parameter.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new gene with the specified parameter index and value.
    /// These values are immutable after creation to ensure genetic stability during operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a new trait in the model's DNA.
    /// 
    /// When creating a new gene:
    /// - You specify which parameter it controls (the index)
    /// - You set its initial value
    /// - Once created, these properties can't be changed; to evolve the model, 
    ///   new genes with different values are created instead
    /// 
    /// This immutability helps prevent unintended changes during genetic operations
    /// and maintains the integrity of the evolutionary process.
    /// </para>
    /// </remarks>
    public ModelParameterGene(int index, T value)
    {
        Index = index;
        Value = value;
    }

    /// <summary>
    /// Creates a clone of this gene.
    /// </summary>
    /// <returns>A new ModelParameterGene with the same index and value.</returns>
    /// <remarks>
    /// <para>
    /// This method creates an exact copy of the gene with the same index and value.
    /// Cloning is an essential operation in genetic algorithms for creating offspring
    /// and preserving genetic information.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating an identical copy of a specific trait.
    /// 
    /// When cloning a gene:
    /// - You get a completely new gene object
    /// - It has the exact same index (controls the same parameter)
    /// - It has the exact same value (sets the parameter to the same number)
    /// - Changes to the clone won't affect the original
    /// 
    /// This operation is used when creating offspring in genetic algorithms,
    /// allowing traits to be passed down unchanged from parents to children.
    /// </para>
    /// </remarks>
    public ModelParameterGene<T> Clone()
    {
        return new ModelParameterGene<T>(Index, Value);
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current gene.
    /// </summary>
    /// <param name="obj">The object to compare with the current gene.</param>
    /// <returns>true if the specified object is equal to the current gene; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Equals method to provide value-based equality comparison.
    /// Two genes are considered equal if they have the same index and value.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the system decides if two genes are identical.
    /// 
    /// When comparing two genes:
    /// - It checks if they control the same parameter (same index)
    /// - It checks if they have the same value
    /// - Only if both conditions are true, the genes are considered equal
    /// 
    /// This is useful in genetic algorithms for detecting duplicate genes
    /// or for comparing genes before and after operations to measure changes.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method overrides the base GetHashCode method to ensure that the hash code
    /// is consistent with the Equals method. It combines the hash codes of the Index and Value
    /// properties to create a unique hash code for the gene.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique fingerprint for the gene.
    /// 
    /// The hash code:
    /// - Provides a number that can be used to quickly compare or identify genes
    /// - Is calculated based on both the index and value of the gene
    /// - Ensures that identical genes get the same number (consistent with Equals)
    /// - Helps when storing genes in collections like dictionaries or hash sets
    /// 
    /// While not directly used in genetic operations, this method is important for
    /// the proper functioning of many collection types and algorithms.
    /// </para>
    /// </remarks>
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