namespace AiDotNet.Genetics;

/// <summary>
/// Represents a node in a genetic programming tree.
/// </summary>
public class NodeGene
{
    /// <summary>
    /// Gets or sets the type of node.
    /// </summary>
    public GeneticNodeType Type { get; set; }

    /// <summary>
    /// Gets or sets the value or function name.
    /// </summary>
    public string Value { get; set; }

    /// <summary>
    /// Gets or sets the child nodes.
    /// </summary>
    public List<NodeGene> Children { get; set; }

    public NodeGene(GeneticNodeType type, string value)
    {
        Type = type;
        Value = value;
        Children = [];
    }

    public NodeGene Clone()
    {
        var clone = new NodeGene(Type, Value);
        foreach (var child in Children)
        {
            clone.Children.Add(child.Clone());
        }

        return clone;
    }

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
