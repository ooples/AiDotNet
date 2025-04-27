namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual encoded as a permutation, suitable for problems like TSP.
/// </summary>
/// <remarks>
/// <para>
/// The PermutationIndividual class implements a permutation-based encoding for genetic algorithms.
/// It represents a solution as an ordered sequence of elements, where each element appears exactly once.
/// This encoding is particularly useful for problems where the order of elements matters, such as
/// the Traveling Salesman Problem (TSP), scheduling problems, or sequencing problems.
/// </para>
/// <para><b>For Beginners:</b> Think of a PermutationIndividual like a complete deck of cards arranged in a specific order.
/// 
/// Imagine you have a deck of playing cards:
/// - The PermutationIndividual represents one specific arrangement of these cards
/// - Each card (gene) appears exactly once in the arrangement
/// - The order of the cards is what matters - it represents a particular solution
/// - For example, in a traveling salesman problem, it might represent the order of cities to visit
/// - Different arrangements (individuals) represent different possible solutions
/// 
/// During evolution, these arrangements get shuffled, combined, and modified to find
/// the best possible sequence for solving the problem at hand.
/// </para>
/// </remarks>
public class PermutationIndividual : IEvolvable<PermutationGene, double>
{
    /// <summary>
    /// The collection of genes that form this individual's permutation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the ordered sequence of genes that define this individual's permutation.
    /// Each gene represents an element in the sequence, and the order of genes defines the permutation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual arrangement of cards in the deck.
    /// 
    /// The genes list:
    /// - Contains each card exactly once
    /// - The order of the cards is what makes each arrangement unique
    /// - Different arrangements will have the same cards but in different orders
    /// - The quality of an arrangement depends on how well it solves the specific problem
    /// 
    /// This ordered sequence is the core representation of the solution that will be
    /// manipulated by the genetic algorithm.
    /// </para>
    /// </remarks>
    private List<PermutationGene> _genes = [];

    /// <summary>
    /// The fitness score that indicates how well this individual solves the problem.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the individual's fitness score, which quantifies how well this permutation
    /// performs on the given problem. For minimization problems like TSP, lower values typically
    /// indicate better solutions (shorter routes).
    /// </para>
    /// <para><b>For Beginners:</b> This is like a score that rates how good this specific arrangement is.
    /// 
    /// For example:
    /// - In a traveling salesman problem, it might represent the total distance traveled
    /// - A lower score usually means a better solution (shorter overall distance)
    /// - This score helps determine which arrangements survive and reproduce
    /// - Arrangements with better scores have a higher chance of passing their patterns to the next generation
    /// 
    /// The fitness score is the key measure that drives the entire evolutionary process
    /// toward better solutions.
    /// </para>
    /// </remarks>
    private double _fitness;

    /// <summary>
    /// Creates a new permutation individual with a random permutation of the specified size.
    /// </summary>
    /// <param name="size">The size of the permutation.</param>
    /// <param name="random">Random number generator for initialization.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PermutationIndividual with a random permutation of indices
    /// from 0 to size-1. It uses the Fisher-Yates shuffle algorithm to ensure an unbiased,
    /// random permutation where each element appears exactly once.
    /// </para>
    /// <para><b>For Beginners:</b> This is like thoroughly shuffling a deck of cards.
    /// 
    /// When creating a new random arrangement:
    /// - You specify how many cards (elements) to include
    /// - The constructor creates a list with cards numbered 0 through size-1
    /// - It then shuffles these cards using a proven shuffling method (Fisher-Yates)
    /// - The result is a completely random arrangement where each card appears exactly once
    /// 
    /// This creates a starting point for the evolutionary algorithm to begin exploring
    /// possible solutions.
    /// </para>
    /// </remarks>
    public PermutationIndividual(int size, Random random)
    {
        // Create a permutation of 0...size-1
        var indices = Enumerable.Range(0, size).ToList();

        // Shuffle using Fisher-Yates algorithm
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Create genes
        foreach (int index in indices)
        {
            _genes.Add(new PermutationGene(index));
        }
    }

    /// <summary>
    /// Creates a permutation individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PermutationIndividual with the provided collection of genes.
    /// It's used when creating new individuals with predefined permutations, typically during
    /// genetic operations like crossover.
    /// </para>
    /// <para><b>For Beginners:</b> This is like arranging cards in a specific order that you already know.
    /// 
    /// When creating a specific arrangement:
    /// - You provide the exact sequence of cards you want
    /// - This is often used when creating offspring during evolution
    /// - For example, after combining features from two parent arrangements
    /// - Or when copying an existing arrangement that performed well
    /// 
    /// This constructor allows the algorithm to create new individuals with predetermined
    /// permutations, which is essential for genetic operations.
    /// </para>
    /// </remarks>
    public PermutationIndividual(ICollection<PermutationGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the permutation as an array of indices.
    /// </summary>
    /// <returns>An array of indices representing the permutation.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the raw permutation as an array of integer indices from the gene sequence.
    /// It's useful for evaluating the permutation or for applying problem-specific operations
    /// that work directly with indices rather than genes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like reading off the sequence of card numbers in order.
    /// 
    /// For example:
    /// - If your arrangement is [Card 3, Card 1, Card 4, Card 0, Card 2]
    /// - This method returns the array [3, 1, 4, 0, 2]
    /// - This makes it easier to work with the actual sequence of numbers
    /// - It's often used when calculating the fitness or applying specialized operations
    /// 
    /// This provides a simpler representation of the permutation for operations that
    /// don't need to work with the full gene objects.
    /// </para>
    /// </remarks>
    public int[] GetPermutation()
    {
        return [.. _genes.Select(g => g.Index)];
    }

    /// <summary>
    /// Applies the Order Crossover (OX) operator.
    /// </summary>
    /// <param name="other">The other parent.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Two child permutations created via order crossover.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Order Crossover (OX) operator, a specialized crossover technique for permutations.
    /// It creates two children by:
    /// 1. Selecting a random subsequence from each parent
    /// 2. Copying this subsequence to the corresponding positions in the child
    /// 3. Filling the remaining positions with elements from the other parent, maintaining their relative order
    /// This approach ensures that the resulting children are valid permutations where each element appears exactly once.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating two new card arrangements by combining patterns from two existing ones.
    /// 
    /// The process works like this:
    /// 1. Randomly select a section of cards from each parent arrangement
    /// 2. Create two new arrangements, initially empty
    /// 3. Copy the selected section from Parent A into the same positions in Child A
    /// 4. Fill the remaining positions in Child A with cards from Parent B, keeping their original order
    /// 5. Repeat steps 3-4 with parents reversed to create Child B
    /// 
    /// This creates two new arrangements that inherit patterns from both parents
    /// while ensuring each card still appears exactly once in each new arrangement.
    /// </para>
    /// </remarks>
    public (PermutationIndividual, PermutationIndividual) OrderCrossover(
        PermutationIndividual other, Random random)
    {
        int size = _genes.Count;
        int[] parent1 = GetPermutation();
        int[] parent2 = other.GetPermutation();
        int[] child1 = new int[size];
        int[] child2 = new int[size];

        // Select substring boundaries
        int start = random.Next(size);
        int end = start + random.Next(size - start);

        /// Initialize children with -1 (empty)
        for (int i = 0; i < child1.Length; i++)
        {
            child1[i] = -1;
            child2[i] = -1;
        }

        // Copy substring from parents to children
        for (int i = start; i <= end; i++)
        {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }

        // Fill remaining positions
        FillFromOther(parent2, child1, start, end);
        FillFromOther(parent1, child2, start, end);

        // Create new individuals
        return (
            new PermutationIndividual([.. child1.Select(i => new PermutationGene(i))]),
            new PermutationIndividual([.. child2.Select(i => new PermutationGene(i))])
        );
    }

    /// <summary>
    /// Helper method for Order Crossover.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This private method helps implement the Order Crossover by filling the remaining positions
    /// in a child permutation with elements from the other parent while maintaining their relative order.
    /// It ensures that no element is duplicated in the resulting permutation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like completing a partially filled card arrangement.
    /// 
    /// After copying a section from one parent:
    /// - We need to fill the rest of the positions with cards from the other parent
    /// - But we can't use any cards that are already in the arrangement (no duplicates)
    /// - So we take cards from the other parent in their original order
    /// - Skip any cards that are already in the arrangement
    /// - This ensures each card appears exactly once in the final arrangement
    /// 
    /// This process maintains patterns from both parents while creating a valid permutation.
    /// </para>
    /// </remarks>
    private void FillFromOther(int[] parent, int[] child, int start, int end)
    {
        int size = parent.Length;
        int childPos = (end + 1) % size;

        for (int i = 0; i < size; i++)
        {
            int parentPos = (end + 1 + i) % size;
            int parentValue = parent[parentPos];

            if (Array.IndexOf(child, parentValue, 0, size) == -1)
            {
                child[childPos] = parentValue;
                childPos = (childPos + 1) % size;

                // Skip positions that are already filled
                while (childPos >= start && childPos <= end)
                {
                    childPos = (childPos + 1) % size;
                }
            }
        }
    }

    /// <summary>
    /// Applies a swap mutation by swapping two random positions.
    /// </summary>
    /// <param name="random">Random number generator.</param>
    /// <remarks>
    /// <para>
    /// This method implements the swap mutation operator for permutations, which randomly
    /// selects two positions in the permutation and swaps their values. This simple mutation
    /// preserves the validity of the permutation while introducing variation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like swapping two cards in your arrangement.
    /// 
    /// The process is simple:
    /// 1. Randomly select two positions in the arrangement
    /// 2. Swap the cards at these positions
    /// 
    /// For example:
    /// - If you have [3, 1, 4, 0, 2] and positions 1 and 3 are chosen
    /// - The result would be [3, 0, 4, 1, 2]
    /// 
    /// This creates a small change that might improve the solution while ensuring
    /// each card still appears exactly once in the arrangement.
    /// </para>
    /// </remarks>
    public void SwapMutation(Random random)
    {
        int size = _genes.Count;
        int pos1 = random.Next(size);
        int pos2 = random.Next(size);

        // Swap genes
        (_genes[pos1], _genes[pos2]) = (_genes[pos2], _genes[pos1]);
    }

    /// <summary>
    /// Applies the inversion mutation by reversing a random subsequence.
    /// </summary>
    /// <param name="random">Random number generator.</param>
    /// <remarks>
    /// <para>
    /// This method implements the inversion mutation operator for permutations, which randomly
    /// selects a subsequence in the permutation and reverses its order. This mutation is particularly
    /// effective for problems like TSP, as it preserves good subpaths while potentially improving others.
    /// </para>
    /// <para><b>For Beginners:</b> This is like reversing the order of a section of cards in your arrangement.
    /// 
    /// The process works like this:
    /// 1. Randomly select a starting and ending position in the arrangement
    /// 2. Reverse the order of all cards between these positions (inclusive)
    /// 
    /// For example:
    /// - If you have [3, 1, 4, 0, 2] and positions 1 and 3 are chosen
    /// - The section [1, 4, 0] would be reversed to [0, 4, 1]
    /// - Resulting in [3, 0, 4, 1, 2]
    /// 
    /// This type of mutation is especially useful for route optimization problems,
    /// as it can eliminate route crossings that increase travel distance.
    /// </para>
    /// </remarks>
    public void InversionMutation(Random random)
    {
        int size = _genes.Count;
        int pos1 = random.Next(size);
        int pos2 = random.Next(size);

        // Ensure pos1 <= pos2
        if (pos1 > pos2)
            (pos1, pos2) = (pos2, pos1);

        // Reverse subsequence
        int steps = (pos2 - pos1 + 1) / 2;
        for (int i = 0; i < steps; i++)
        {
            (_genes[pos1 + i], _genes[pos2 - i]) = (_genes[pos2 - i], _genes[pos1 + i]);
        }
    }

    /// <summary>
    /// Gets the genes of this individual.
    /// </summary>
    /// <returns>The collection of genes that form this individual's permutation.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the individual's genetic information as required by the IEvolvable interface.
    /// It returns the complete ordered sequence of genes that define this permutation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like revealing the entire sequence of cards in your arrangement.
    /// 
    /// This method:
    /// - Returns the complete ordered sequence of cards
    /// - Allows other parts of the algorithm to examine the arrangement
    /// - Is used during operations like crossover and mutation
    /// 
    /// This is one of the core methods required by the genetic algorithm to work
    /// with permutation-based individuals.
    /// </para>
    /// </remarks>
    public ICollection<PermutationGene> GetGenes()
    {
        return _genes;
    }

    /// <summary>
    /// Sets the genes of this individual.
    /// </summary>
    /// <param name="genes">The collection of genes to set.</param>
    /// <remarks>
    /// <para>
    /// This method replaces the individual's genetic information as required by the IEvolvable interface.
    /// It updates the permutation with a new ordered sequence of genes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like rearranging all the cards in your sequence.
    /// 
    /// This method:
    /// - Replaces the entire card arrangement with a new one
    /// - Is used when creating new offspring or applying specialized operations
    /// - Updates how this individual represents a solution to the problem
    /// 
    /// This is one of the core methods required by the genetic algorithm to modify
    /// permutation-based individuals during evolution.
    /// </para>
    /// </remarks>
    public void SetGenes(ICollection<PermutationGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the fitness of this individual.
    /// </summary>
    /// <returns>The fitness score as a double value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the individual's fitness score as required by the IEvolvable interface.
    /// For permutation problems, this typically represents a measure of solution quality,
    /// such as the total distance in a TSP problem.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking the score of your card arrangement.
    /// 
    /// The fitness score:
    /// - Indicates how good this particular arrangement is at solving the problem
    /// - For many problems, lower scores are better (like shorter travel distances)
    /// - This score determines which arrangements survive and reproduce
    /// - It drives the entire evolutionary process toward better solutions
    /// 
    /// This is one of the core methods required by the genetic algorithm to evaluate
    /// and compare different solutions.
    /// </para>
    /// </remarks>
    public double GetFitness()
    {
        return _fitness;
    }

    /// <summary>
    /// Sets the fitness of this individual.
    /// </summary>
    /// <param name="fitness">The fitness score to set.</param>
    /// <remarks>
    /// <para>
    /// This method assigns the fitness score to the individual as required by the IEvolvable interface.
    /// It is typically called after evaluating the quality of the permutation for the specific problem.
    /// </para>
    /// <para><b>For Beginners:</b> This is like recording the score for your card arrangement.
    /// 
    /// After testing how well your arrangement performs:
    /// - The score is calculated based on problem-specific criteria
    /// - This method stores that score with the arrangement
    /// - Later, this score will be used to compare different arrangements
    /// - Better-scoring arrangements have a higher chance of influencing the next generation
    /// 
    /// This is one of the core methods required by the genetic algorithm to track
    /// the quality of different solutions during evolution.
    /// </para>
    /// </remarks>
    public void SetFitness(double fitness)
    {
        _fitness = fitness;
    }

    /// <summary>
    /// Creates a deep copy of this individual.
    /// </summary>
    /// <returns>A new PermutationIndividual that is a copy of this one.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the individual, including all its genes
    /// and its fitness score. It's essential for genetic operations where individuals need to be
    /// copied without modifying the originals.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating an exact duplicate of your card arrangement.
    /// 
    /// When cloning an individual:
    /// - A completely new arrangement is created
    /// - Each card is copied to the same position in the new arrangement
    /// - The fitness score is also copied
    /// - Changes to one arrangement won't affect the other
    /// 
    /// This operation is crucial in genetic algorithms for preserving good solutions
    /// while experimenting with modifications to create potentially better ones.
    /// </para>
    /// </remarks>
    public IEvolvable<PermutationGene, double> Clone()
    {
        var clone = new PermutationIndividual([]);
        foreach (var gene in _genes)
        {
            clone._genes.Add(gene.Clone());
        }
        clone._fitness = _fitness;

        return clone;
    }
}