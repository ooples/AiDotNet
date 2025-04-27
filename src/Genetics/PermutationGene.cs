namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene in a permutation (the index of an element in a sequence).
/// </summary>
/// <remarks>
/// <para>
/// The PermutationGene class represents a single element in a permutation-based encoding for genetic algorithms.
/// Each gene holds an index value that determines the position or order of an element in a sequence.
/// Permutation encoding is particularly useful for problems where the order of elements matters,
/// such as the Traveling Salesman Problem, scheduling problems, or routing problems.
/// </para>
/// <para><b>For Beginners:</b> Think of a PermutationGene like a card in a deck.
/// 
/// Imagine you have a deck of cards numbered 1 through 10:
/// - Each PermutationGene is like one of these cards
/// - The Index is the number on the card
/// - A complete solution (an individual) would be like a specific arrangement of these cards
/// - In problems like route planning, each card might represent a city to visit
/// - The order of the cards determines the order in which you visit each city
/// 
/// Unlike other types of genes that might represent values or rules, permutation genes
/// focus specifically on order and arrangement, which is crucial for many optimization problems.
/// </para>
/// </remarks>
public class PermutationGene
{
    /// <summary>
    /// Gets or sets the index value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the index value of the gene, which represents the position
    /// or identity of an element in a permutation. The meaning of this index depends on
    /// the specific problem being solved.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the number on a playing card.
    /// 
    /// For example:
    /// - In a route planning problem, the index might represent a specific city (City #3)
    /// - In a scheduling problem, it might represent a specific task (Task #7)
    /// - In a resource allocation problem, it might represent a specific resource (Machine #2)
    /// 
    /// The arrangement of these indices in a complete solution determines the order or
    /// sequence in which elements appear, which is what makes permutation-based problems unique.
    /// </para>
    /// </remarks>
    public int Index { get; set; }

    /// <summary>
    /// Initializes a new instance of the PermutationGene class with the specified index.
    /// </summary>
    /// <param name="index">The index value for this gene.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PermutationGene with the specified index value.
    /// The index typically corresponds to an element in the problem being solved,
    /// such as a city in a travel route or a task in a schedule.
    /// </para>
    /// <para><b>For Beginners:</b> This is like picking a specific card from the deck.
    /// 
    /// When creating a new gene:
    /// - You specify which index (or card number) it represents
    /// - This establishes the identity of this particular gene
    /// - For example, creating a gene with index 5 is like picking the card with number 5
    /// 
    /// This constructor is the standard way to create a new gene for use in
    /// permutation-based genetic algorithms.
    /// </para>
    /// </remarks>
    public PermutationGene(int index)
    {
        Index = index;
    }

    /// <summary>
    /// Creates a copy of this gene.
    /// </summary>
    /// <returns>A new PermutationGene with the same index value.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new PermutationGene that is an exact copy of the current one.
    /// Cloning is essential in genetic algorithms for creating offspring without modifying
    /// the original genes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a duplicate of a card.
    /// 
    /// When cloning a gene:
    /// - You get a completely new gene object
    /// - It has the same index (or card number) as the original
    /// - Changes to the clone won't affect the original
    /// 
    /// This operation is used during genetic operations like crossover and mutation,
    /// allowing genes to be copied between individuals without changing the originals.
    /// </para>
    /// </remarks>
    public PermutationGene Clone()
    {
        return new PermutationGene(Index);
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current gene.
    /// </summary>
    /// <param name="obj">The object to compare with the current gene.</param>
    /// <returns>true if the specified object is a PermutationGene with the same index; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Equals method to provide value-based equality comparison.
    /// Two PermutationGene objects are considered equal if they have the same Index value.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking if two cards have the same number.
    /// 
    /// When comparing two genes:
    /// - It checks if the other object is also a PermutationGene
    /// - If it is, it checks if both genes have the same index value
    /// - Only if both conditions are true, the genes are considered equal
    /// 
    /// This is useful in genetic algorithms for detecting duplicate genes
    /// or for comparing genes before and after operations.
    /// </para>
    /// </remarks>
    public override bool Equals(object? obj)
    {
        return obj is PermutationGene gene && gene.Index == Index;
    }

    /// <summary>
    /// Returns a hash code for this gene.
    /// </summary>
    /// <returns>A hash code for the current gene.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base GetHashCode method to ensure that the hash code
    /// is consistent with the Equals method. It returns the hash code of the Index property.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique identifier for the card.
    /// 
    /// The hash code:
    /// - Provides a number that can be used to quickly compare or identify genes
    /// - Is based on the index value of the gene
    /// - Ensures that genes with the same index get the same hash code
    /// - Helps when storing genes in collections like dictionaries or hash sets
    /// 
    /// While not directly used in genetic operations, this method is important for
    /// the proper functioning of many collections and algorithms.
    /// </para>
    /// </remarks>
    public override int GetHashCode()
    {
        return Index.GetHashCode();
    }
}