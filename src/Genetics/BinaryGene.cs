namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene that holds a binary value (0 or 1).
/// </summary>
/// <remarks>
/// <para>
/// The BinaryGene class is a fundamental building block in genetic algorithms that use binary encoding.
/// Each gene represents a single bit of information in a chromosome, which is the basic unit of genetic evolution.
/// </para>
/// <para><b>For Beginners:</b> Think of a BinaryGene as a simple on/off switch.
/// 
/// In the world of genetic algorithms, complex solutions are often built from many simple pieces:
/// - A BinaryGene is like a single switch that can be either on (1) or off (0)
/// - Many of these switches together form a "chromosome" (like a row of light switches)
/// - These chromosomes represent potential solutions to problems
/// - During evolution, these switches get flipped (mutation) or swapped (crossover)
/// 
/// This simple on/off structure is powerful because many complex problems can be encoded
/// using patterns of binary digits.
/// </para>
/// </remarks>
public class BinaryGene
{
    /// <summary>
    /// Gets or sets the binary value (0 or 1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property holds the actual binary value of the gene. It is restricted to only allow
    /// values of 0 or 1, representing the two possible states of a binary gene.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual state of the switch - either off (0) or on (1).
    /// When algorithms work with genes, they're essentially deciding whether to turn this switch on or off.
    /// </para>
    /// </remarks>
    public int Value { get; set; }

    /// <summary>
    /// Initializes a new instance of the BinaryGene class with an optional initial value.
    /// </summary>
    /// <param name="value">The initial value for the gene. Any positive number will be converted to 1, and 0 or negative will be 0.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new BinaryGene with the specified value. If the provided value
    /// is greater than 0, the gene will be set to 1; otherwise, it will be set to 0. This ensures
    /// that the gene always holds a valid binary value.
    /// </para>
    /// <para><b>For Beginners:</b> This is like installing a new switch in your system.
    /// 
    /// When you create a new gene:
    /// - You can optionally specify its initial state (on or off)
    /// - If you provide any positive number, it's treated as "on" (1)
    /// - If you provide 0 or any negative number, it's treated as "off" (0)
    /// - This ensures that every gene is always in a valid state
    /// </para>
    /// </remarks>
    public BinaryGene(int value = 0)
    {
        Value = value > 0 ? 1 : 0; // Ensure binary value
    }

    /// <summary>
    /// Creates an independent copy of this gene.
    /// </summary>
    /// <returns>A new BinaryGene with the same value as this one.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new BinaryGene instance that is a copy of the current gene.
    /// Cloning is an important operation in genetic algorithms since it allows genes to be
    /// duplicated without maintaining references to the original.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating an exact duplicate of a switch.
    /// 
    /// When you clone a gene:
    /// - You get a brand new switch that's completely separate from the original
    /// - The new switch starts in the same position as the original (on or off)
    /// - Changes to one switch won't affect the other
    /// 
    /// This is important during evolution when copying genetic information from one
    /// generation to the next.
    /// </para>
    /// </remarks>
    public BinaryGene Clone()
    {
        return new BinaryGene(Value);
    }

    /// <summary>
    /// Determines whether this gene is equal to another object.
    /// </summary>
    /// <param name="obj">The object to compare with the current gene.</param>
    /// <returns>True if the specified object is a BinaryGene with the same value; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Equals method to provide value-based equality comparison for BinaryGene objects.
    /// Two BinaryGene instances are considered equal if they have the same Value.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the system decides if two switches are identical.
    /// 
    /// When comparing two genes:
    /// - The method checks if the other object is actually a gene (not something else)
    /// - Then it checks if both switches are in the same position (both on or both off)
    /// - If both conditions are true, the genes are considered equal
    /// 
    /// This is used in genetic algorithms to identify duplicate genes or to compare
    /// genes during selection and crossover operations.
    /// </para>
    /// </remarks>
    public override bool Equals(object? obj)
    {
        return obj is BinaryGene gene && gene.Value == Value;
    }

    /// <summary>
    /// Returns a hash code for this gene.
    /// </summary>
    /// <returns>A hash code based on the gene's value.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base GetHashCode method to ensure that the hash code is consistent with the Equals method.
    /// It simply returns the hash code of the Value property, since that's the only field that determines equality.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique identifier for a switch based on its position.
    /// 
    /// The hash code:
    /// - Provides a number that can be used to quickly compare genes
    /// - Ensures that identical genes (switches in the same position) get the same number
    /// - Helps when storing genes in collections like dictionaries or hash sets
    /// 
    /// While not often directly used in genetic algorithm operations, this method
    /// is important for the proper functioning of many collections and data structures.
    /// </para>
    /// </remarks>
    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }
}
