using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Represents a directed edge (relationship) between two nodes in a knowledge graph.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// A graph edge represents a relationship between two entities, such as "works_for", "located_in", or "friend_of".
/// Edges are directed (from source to target) and can have properties to store additional relationship metadata.
/// </para>
/// <para><b>For Beginners:</b> Think of an edge as a relationship or connection between two people.
/// 
/// In a social network:
/// - "Alice WORKS_FOR Microsoft" (Alice is the source, Microsoft is the target)
/// - "Bob LIVES_IN Seattle" (Bob is the source, Seattle is the target)
/// - "Charlie KNOWS David" (Charlie knows David, but maybe David doesn't know Charlie - it's directional!)
/// 
/// In a knowledge graph:
/// - Source: The entity the relationship starts from
/// - Target: The entity the relationship points to
/// - Type: The kind of relationship (WORKS_FOR, LIVES_IN, KNOWS)
/// - Properties: Extra info (since: "2020", strength: 0.9)
/// - Weight: How important or strong this relationship is (0.0 to 1.0)
/// 
/// For example:
/// Source: "Albert Einstein" (PERSON)
/// Target: "Princeton University" (ORGANIZATION)
/// Type: "WORKED_AT"
/// Properties: { "from": "1933", "to": "1955" }
/// Weight: 0.95 (very strong relationship)
/// </para>
/// </remarks>
public class GraphEdge<T>
{
    /// <summary>
    /// Unique identifier for this edge.
    /// </summary>
    public string Id { get; set; }

    /// <summary>
    /// The source node ID (where the relationship starts).
    /// </summary>
    public string SourceId { get; set; }

    /// <summary>
    /// The target node ID (where the relationship points to).
    /// </summary>
    public string TargetId { get; set; }

    /// <summary>
    /// The relationship type (e.g., WORKS_FOR, LOCATED_IN, FRIEND_OF).
    /// </summary>
    public string RelationType { get; set; }

    /// <summary>
    /// Additional properties and metadata for this relationship.
    /// </summary>
    public Dictionary<string, object> Properties { get; set; }

    /// <summary>
    /// Weight or strength of this relationship (0.0 to 1.0).
    /// </summary>
    public double Weight { get; set; }

    /// <summary>
    /// Timestamp when this edge was created.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Start of the temporal validity window for this relationship.
    /// Null means the relationship has no defined start time (always valid from the past).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some facts are only true during a specific time period.
    /// For example, "Einstein WORKED_AT Princeton" was valid from 1933 to 1955.
    /// ValidFrom = 1933-01-01 means this relationship started in 1933.
    /// If null, the relationship is considered to have always been valid.
    /// </para>
    /// </remarks>
    public DateTime? ValidFrom { get; set; }

    /// <summary>
    /// End of the temporal validity window for this relationship.
    /// Null means the relationship has no defined end time (still valid).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ValidUntil defines when a fact stopped being true.
    /// For example, "Einstein WORKED_AT Princeton" had ValidUntil = 1955-04-18.
    /// If null, the relationship is considered to still be valid (ongoing).
    /// </para>
    /// </remarks>
    public DateTime? ValidUntil { get; set; }

    /// <summary>
    /// Checks whether this edge is valid at a specific point in time.
    /// Uses half-open interval [ValidFrom, ValidUntil) so consecutive edges
    /// (e.g., "president 2009-2017" and "president 2017-2021") don't overlap.
    /// </summary>
    /// <param name="timestamp">The point in time to check.</param>
    /// <returns>True if the edge is valid at the given time; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// An edge is considered valid at a timestamp if:
    /// - ValidFrom is null OR timestamp >= ValidFrom, AND
    /// - ValidUntil is null OR timestamp &lt; ValidUntil (exclusive upper bound)
    /// Edges with no temporal bounds (both null) are always valid.
    /// </para>
    /// </remarks>
    public bool IsValidAt(DateTime timestamp)
    {
        if (ValidFrom.HasValue && timestamp < ValidFrom.Value)
            return false;
        if (ValidUntil.HasValue && timestamp >= ValidUntil.Value)
            return false;
        return true;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphEdge{T}"/> class.
    /// </summary>
    /// <param name="sourceId">The source node ID.</param>
    /// <param name="targetId">The target node ID.</param>
    /// <param name="relationType">The relationship type.</param>
    /// <param name="weight">The relationship weight (default: 1.0).</param>
    public GraphEdge(string sourceId, string targetId, string relationType, double weight = 1.0)
    {
        if (string.IsNullOrWhiteSpace(sourceId))
            throw new ArgumentException("Source ID cannot be null or whitespace", nameof(sourceId));
        if (string.IsNullOrWhiteSpace(targetId))
            throw new ArgumentException("Target ID cannot be null or whitespace", nameof(targetId));
        if (string.IsNullOrWhiteSpace(relationType))
            throw new ArgumentException("Relation type cannot be null or whitespace", nameof(relationType));
        if (weight < 0.0 || weight > 1.0)
            throw new ArgumentOutOfRangeException(nameof(weight), "Weight must be between 0.0 and 1.0");

        Id = $"{sourceId}_{relationType}_{targetId}";
        SourceId = sourceId;
        TargetId = targetId;
        RelationType = relationType;
        Weight = weight;
        Properties = new Dictionary<string, object>();
        CreatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Sets the temporal validity window for this edge with validation.
    /// </summary>
    /// <param name="validFrom">Start of validity (inclusive). Null means always valid from the past.</param>
    /// <param name="validUntil">End of validity (exclusive). Null means still valid (ongoing).</param>
    /// <exception cref="ArgumentException">Thrown when validFrom >= validUntil.</exception>
    public void SetTemporalWindow(DateTime? validFrom, DateTime? validUntil)
    {
        if (validFrom.HasValue && validUntil.HasValue && validFrom.Value >= validUntil.Value)
        {
            throw new ArgumentException(
                $"ValidFrom ({validFrom.Value:O}) must be before ValidUntil ({validUntil.Value:O}).");
        }

        ValidFrom = validFrom;
        ValidUntil = validUntil;
    }

    /// <summary>
    /// Adds or updates a property on this edge.
    /// </summary>
    /// <param name="key">The property key.</param>
    /// <param name="value">The property value.</param>
    public void SetProperty(string key, object value)
    {
        Properties[key] = value;
    }

    /// <summary>
    /// Gets a property value from this edge.
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
        return $"{SourceId} -{RelationType}-> {TargetId} (weight: {Weight:F2})";
    }

    public override bool Equals(object? obj)
    {
        return obj is GraphEdge<T> other && Id == other.Id;
    }

    public override int GetHashCode()
    {
        return Id.GetHashCode();
    }
}
