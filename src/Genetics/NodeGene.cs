namespace AiDotNet.Genetics;

/// <summary>
/// Represents a node in a genetic programming tree.
/// </summary>
/// <remarks>
/// <para>
/// The NodeGene class is a fundamental building block of genetic programming trees, which are used to evolve
/// programs or mathematical expressions. Each node represents either a function (operation) or a terminal
/// (value or variable) in the expression tree.
/// </para>
/// <para><b>For Beginners:</b> Think of a NodeGene as a building block in a LEGO structure.
/// 
/// In genetic programming:
/// - Each NodeGene is like a LEGO piece in a larger structure
/// - Some pieces are operations (like add, subtract, multiply)
/// - Some pieces are values or variables (like numbers or x, y, z)
/// - These pieces connect together to form a tree-like structure
/// - The complete structure represents a mathematical formula or a computer program
/// 
/// During evolution, these structures get rearranged, pieces get swapped,
/// and new structures are created to find better solutions to problems.
/// </para>
/// </remarks>
public class NodeGene
{
    /// <summary>
    /// Gets or sets the type of node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property determines whether the node is a function (like an operation) or a terminal
    /// (like a constant value or variable). The type affects how the node behaves in the expression
    /// tree and what operations can be performed on it during evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the category of the LEGO piece.
    /// 
    /// Just as LEGO pieces come in different types:
    /// - Some NodeGenes are "function" types (like math operations: +, -, ×, ÷)
    /// - Others are "terminal" types (like numbers or variables: 5, x, y)
    /// - This categorization determines what role the piece plays in the structure
    /// - It also affects how the piece can be used during the evolutionary process
    /// 
    /// The type is essential because it determines how a node connects to others
    /// and what operations it can perform in the overall program.
    /// </para>
    /// </remarks>
    public GeneticNodeType Type { get; set; }

    /// <summary>
    /// Gets or sets the value or function name.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains the actual content of the node. For function nodes, it holds the operation
    /// name (like "add", "multiply", etc.). For terminal nodes, it holds the value or variable name
    /// (like "5", "x", etc.). This value defines what the node actually does in the expression tree.
    /// </para>
    /// <para><b>For Beginners:</b> This is the specific identity of the LEGO piece.
    /// 
    /// If Type tells you the category:
    /// - Value tells you exactly which piece it is within that category
    /// - For function nodes, it might be "add", "subtract", "if-then", etc.
    /// - For terminal nodes, it might be a number like "42" or a variable like "x"
    /// - This specific identity determines the node's behavior in the program
    /// 
    /// Think of it as the difference between knowing something is a "brick" (Type)
    /// versus knowing it's a "red 2×4 brick" (Value).
    /// </para>
    /// </remarks>
    public string Value { get; set; }

    /// <summary>
    /// Gets or sets the child nodes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains the list of child nodes connected to this node. For function nodes,
    /// the children represent the operands or arguments to the function. For terminal nodes,
    /// this list is typically empty. The structure of children forms the expression tree.
    /// </para>
    /// <para><b>For Beginners:</b> These are the LEGO pieces attached to this piece.
    /// 
    /// In our LEGO analogy:
    /// - Children are the pieces attached underneath this piece
    /// - Function nodes (like "add") need pieces attached to them (the values to add)
    /// - Terminal nodes (like "5") don't have any pieces attached to them
    /// - The arrangement of pieces creates the complete structure of the program
    /// 
    /// For example, if this node is "add", its children might be "5" and "x",
    /// representing the expression "5 + x".
    /// </para>
    /// </remarks>
    public List<NodeGene> Children { get; set; }

    /// <summary>
    /// Initializes a new instance of the NodeGene class with the specified type and value.
    /// </summary>
    /// <param name="type">The type of node (function or terminal).</param>
    /// <param name="value">The value or function name for this node.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new NodeGene with the specified type and value, and initializes
    /// an empty list of children. This is used to create nodes when building or modifying expression trees.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a new LEGO piece out of the box.
    /// 
    /// When creating a new node:
    /// - You specify what category it belongs to (function or terminal)
    /// - You specify exactly which piece it is (its value or operation)
    /// - It starts with no connections to other pieces (empty children list)
    /// - You can then attach other pieces to it as needed
    /// 
    /// This is the starting point for building any node in the genetic program tree.
    /// </para>
    /// </remarks>
    public NodeGene(GeneticNodeType type, string value)
    {
        Type = type;
        Value = value;
        Children = [];
    }

    /// <summary>
    /// Creates a deep copy of this node and all its children.
    /// </summary>
    /// <returns>A new NodeGene that is a copy of this node and its entire subtree.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a deep clone of the node and its entire subtree by recursively cloning
    /// all child nodes. This is essential for genetic operations like crossover and mutation,
    /// where subtrees need to be copied between different individuals.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating an exact duplicate of a LEGO structure.
    /// 
    /// When cloning a node:
    /// - You get a completely new node with the same type and value
    /// - For each piece attached to the original, a duplicate is attached to the clone
    /// - This process repeats for all pieces in the structure
    /// - The result is an identical copy that can be modified independently
    /// 
    /// This operation is crucial during evolution when parts of one solution need to be
    /// copied and incorporated into another solution without affecting the original.
    /// </para>
    /// </remarks>
    public NodeGene Clone()
    {
        var clone = new NodeGene(Type, Value);
        foreach (var child in Children)
        {
            clone.Children.Add(child.Clone());
        }

        return clone;
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current node.
    /// </summary>
    /// <param name="obj">The object to compare with the current node.</param>
    /// <returns>True if the specified object is equal to the current node; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Equals method to provide structural equality comparison for NodeGene objects.
    /// Two nodes are considered equal if they have the same type, value, and all their children are equal.
    /// This requires a recursive comparison of the entire subtree.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if two LEGO structures are identical in every way.
    /// 
    /// When comparing two nodes:
    /// - First, it checks if they're the same category (function or terminal)
    /// - Then it checks if they're the same specific piece (same value)
    /// - Then it checks if they have the same number of pieces attached
    /// - Finally, it checks if each attached piece is identical between the two
    /// 
    /// Only if all of these conditions are met are the nodes considered equal.
    /// This is like checking if two LEGO models are built exactly the same way.
    /// </para>
    /// </remarks>
    public override bool Equals(object? obj)
    {
        if (obj is not NodeGene other)
            return false;
        if (Type != other.Type || Value != other.Value || Children.Count != other.Children.Count)
            return false;
        for (int i = 0; i < Children.Count; i++)
        {
            if (!Children[i].Equals(other.Children[i]))
                return false;
        }
        return true;
    }

    /// <summary>
    /// Returns a hash code for this node.
    /// </summary>
    /// <returns>A hash code for the current node.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base GetHashCode method to ensure that the hash code is consistent with the Equals method.
    /// It computes a hash based on the node's type, value, and the hash codes of all its children.
    /// This ensures that equal nodes will have the same hash code, which is important for collections.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a unique fingerprint for the LEGO structure.
    /// 
    /// The hash code:
    /// - Provides a number that can be used to quickly identify or compare nodes
    /// - Is calculated based on the node's type, value, and all attached pieces
    /// - Ensures that identical structures get the same number
    /// - Helps when storing nodes in collections like dictionaries or hash sets
    /// 
    /// This is used internally by various data structures to efficiently organize
    /// and retrieve nodes, especially when there are many of them.
    /// </para>
    /// </remarks>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + Type.GetHashCode();
            hash = hash * 23 + (Value?.GetHashCode() ?? 0);
            if (Children != null)
            {
                foreach (var child in Children)
                {
                    hash = hash * 23 + (child?.GetHashCode() ?? 0);
                }
            }

            return hash;
        }
    }
}