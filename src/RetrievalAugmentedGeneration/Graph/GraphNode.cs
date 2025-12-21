using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Represents a node in a knowledge graph, typically an entity extracted from text.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// A graph node stores an entity (person, place, concept, etc.) along with its properties and embeddings.
/// Nodes are connected by edges to form a knowledge graph that captures relationships between entities.
/// </para>
/// <para><b>For Beginners:</b> Think of a node as a person in a social network.
/// 
/// Just like a Facebook profile has:
/// - Name: "John Smith"
/// - Properties: age, location, interests
/// - Connections: friends, family, coworkers
/// 
/// A GraphNode has:
/// - Id: Unique identifier
/// - Label: Entity type (PERSON, ORGANIZATION, LOCATION)
/// - Properties: Additional metadata
/// - Embedding: Numeric representation for similarity search
/// 
/// For example:
/// - Id: "person_123"
/// - Label: "PERSON"
/// - Properties: { "name": "Albert Einstein", "occupation": "Physicist" }
/// - Embedding: [0.23, -0.45, 0.67, ...] (vector representation)
/// </para>
/// </remarks>
public class GraphNode<T>
{
    /// <summary>
    /// Unique identifier for this node.
    /// </summary>
    public string Id { get; set; }

    /// <summary>
    /// The entity label or type (e.g., PERSON, ORGANIZATION, LOCATION).
    /// </summary>
    public string Label { get; set; }

    /// <summary>
    /// Additional properties and metadata for this entity.
    /// </summary>
    public Dictionary<string, object> Properties { get; set; }

    /// <summary>
    /// Vector embedding for similarity search and clustering.
    /// </summary>
    public Vector<T>? Embedding { get; set; }

    /// <summary>
    /// Timestamp when this node was created.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Timestamp when this node was last updated.
    /// </summary>
    public DateTime UpdatedAt { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphNode{T}"/> class.
    /// </summary>
    /// <param name="id">Unique identifier for the node.</param>
    /// <param name="label">The entity label or type.</param>
    public GraphNode(string id, string label)
    {
        if (string.IsNullOrWhiteSpace(id))
            throw new ArgumentException("Node ID cannot be null or whitespace", nameof(id));
        if (string.IsNullOrWhiteSpace(label))
            throw new ArgumentException("Node label cannot be null or whitespace", nameof(label));

        Id = id;
        Label = label;
        Properties = new Dictionary<string, object>();
        CreatedAt = DateTime.UtcNow;
        UpdatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Adds or updates a property on this node.
    /// </summary>
    /// <param name="key">The property key.</param>
    /// <param name="value">The property value.</param>
    public void SetProperty(string key, object value)
    {
        Properties[key] = value;
        UpdatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets a property value from this node.
    /// </summary>
    /// <typeparam name="TValue">The expected type of the property value.</typeparam>
    /// <param name="key">The property key.</param>
    /// <returns>The property value, or default if not found or conversion fails.</returns>
    /// <remarks>
    /// <para>
    /// This method handles JSON deserialization quirks where numeric types may differ
    /// (e.g., int stored as long after JSON round-trip). It uses Convert.ChangeType
    /// for IConvertible types to handle such conversions gracefully.
    /// </para>
    /// <para>
    /// The method catches and handles the following exceptions during conversion:
    /// - InvalidCastException: When the types are incompatible
    /// - FormatException: When the string representation is invalid
    /// - OverflowException: When the value is outside the target type's range
    /// </para>
    /// </remarks>
    public TValue? GetProperty<TValue>(string key)
    {
        if (!Properties.TryGetValue(key, out var value) || value == null)
            return default;

        // Direct type match
        if (value is TValue typedValue)
            return typedValue;

        // Handle numeric type conversions (JSON deserializes integers as long)
        if (value is IConvertible && typeof(TValue).IsValueType)
        {
            try
            {
                return (TValue)Convert.ChangeType(value, typeof(TValue));
            }
            catch (InvalidCastException)
            {
                return default;
            }
            catch (FormatException)
            {
                return default;
            }
            catch (OverflowException)
            {
                return default;
            }
        }

        return default;
    }

    public override string ToString()
    {
        var name = GetProperty<string>("name") ?? Id;
        return $"{Label}:{name}";
    }

    public override bool Equals(object? obj)
    {
        return obj is GraphNode<T> other && Id == other.Id;
    }

    public override int GetHashCode()
    {
        return Id.GetHashCode();
    }
}
