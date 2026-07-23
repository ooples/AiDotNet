using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Pure-logic helpers for the Neo4j graph backend: Cypher identifier escaping, the parameterized
/// Cypher statements used by <see cref="Neo4jGraphStore{T}"/>, and lossless (de)serialization of the
/// property/temporal model. Kept free of any Neo4j.Driver types so the query/param construction can be
/// unit-tested without a live database.
/// </summary>
/// <remarks>
/// <para><b>Injection safety:</b> Cypher labels and relationship types cannot be supplied as query
/// parameters — they must appear as literal identifiers. Every such identifier used by this backend is a
/// value configured once at construction (the node base label and the relationship type), never
/// attacker-controlled per-request data, and is passed through <see cref="EscapeIdentifier"/> which
/// backtick-quotes it and doubles any embedded backticks (the standard, complete Cypher-identifier
/// escaping). Every per-operation value — node ids, the entity type, properties, embeddings, weights,
/// timestamps — is bound as a real Cypher <c>$parameter</c>, so no user data is ever concatenated into a
/// query string.</para>
/// </remarks>
public sealed class Neo4jCypher
{
    // Property keys written onto Neo4j nodes/relationships. Kept as constants so the mapper and the
    // Cypher builders agree on the exact schema.
    internal const string PropId = "id";
    internal const string PropLabel = "label";
    internal const string PropProps = "props";
    internal const string PropEmbedding = "embedding";
    internal const string PropRelationType = "relationType";
    internal const string PropWeight = "weight";
    internal const string PropCreatedAt = "createdAt";
    internal const string PropUpdatedAt = "updatedAt";
    internal const string PropValidFrom = "validFrom";
    internal const string PropValidUntil = "validUntil";

    /// <summary>The backtick-escaped node label literal (e.g. <c>`Entity`</c>).</summary>
    public string NodeLabel { get; }

    /// <summary>The backtick-escaped relationship-type literal (e.g. <c>`RELATED`</c>).</summary>
    public string RelationshipType { get; }

    /// <summary>
    /// Creates a Cypher builder for the given node base label and relationship type. Both are validated
    /// and escaped once here; the escaped forms are the only literal identifiers that ever reach a query.
    /// </summary>
    /// <param name="nodeLabel">The base label applied to every node (default <c>Entity</c>). Acts as the
    /// per-namespace partition when different stores use different labels.</param>
    /// <param name="relationshipType">The relationship type applied to every edge (default
    /// <c>RELATED</c>).</param>
    public Neo4jCypher(string nodeLabel = "Entity", string relationshipType = "RELATED")
    {
        NodeLabel = EscapeIdentifier(nodeLabel);
        RelationshipType = EscapeIdentifier(relationshipType);
    }

    /// <summary>
    /// Escapes a Cypher identifier (label or relationship type) by wrapping it in backticks and doubling
    /// any embedded backticks, which makes it impossible for the value to break out of the identifier
    /// token. Rejects null/empty/whitespace and values containing a NUL character.
    /// </summary>
    public static string EscapeIdentifier(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier))
        {
            throw new ArgumentException("Cypher identifier cannot be null, empty, or whitespace.", nameof(identifier));
        }

        if (identifier.IndexOf('\0') >= 0)
        {
            throw new ArgumentException("Cypher identifier cannot contain a NUL character.", nameof(identifier));
        }

        return "`" + identifier.Replace("`", "``") + "`";
    }

    // ---- Cypher statements. All per-operation data is bound via $parameters. ----

    /// <summary>Upserts a node by id, then sets its type/props/embedding/timestamps.</summary>
    public string UpsertNode() =>
        $"MERGE (n:{NodeLabel} {{{PropId}: $id}}) " +
        $"SET n.{PropLabel} = $label, n.{PropProps} = $props, n.{PropEmbedding} = $embedding, " +
        $"n.{PropCreatedAt} = $createdAt, n.{PropUpdatedAt} = $updatedAt";

    /// <summary>
    /// Matches both endpoints, upserts the relationship by id, and returns it so the caller can detect a
    /// missing endpoint (no row =&gt; a MATCH failed).
    /// </summary>
    public string UpsertEdge() =>
        $"MATCH (s:{NodeLabel} {{{PropId}: $sourceId}}) " +
        $"MATCH (t:{NodeLabel} {{{PropId}: $targetId}}) " +
        $"MERGE (s)-[r:{RelationshipType} {{{PropId}: $id}}]->(t) " +
        $"SET r.{PropRelationType} = $relationType, r.{PropWeight} = $weight, r.{PropProps} = $props, " +
        $"r.{PropCreatedAt} = $createdAt, r.{PropValidFrom} = $validFrom, r.{PropValidUntil} = $validUntil " +
        "RETURN r";

    public string GetNode() =>
        $"MATCH (n:{NodeLabel} {{{PropId}: $id}}) RETURN n LIMIT 1";

    public string GetEdge() =>
        $"MATCH (s:{NodeLabel})-[r:{RelationshipType} {{{PropId}: $id}}]->(t:{NodeLabel}) " +
        "RETURN s.id AS sourceId, t.id AS targetId, r LIMIT 1";

    public string RemoveNode() =>
        $"MATCH (n:{NodeLabel} {{{PropId}: $id}}) DETACH DELETE n RETURN count(*) AS deleted";

    public string RemoveEdge() =>
        $"MATCH (:{NodeLabel})-[r:{RelationshipType} {{{PropId}: $id}}]->(:{NodeLabel}) " +
        "DELETE r RETURN count(*) AS deleted";

    public string OutgoingEdges() =>
        $"MATCH (s:{NodeLabel} {{{PropId}: $id}})-[r:{RelationshipType}]->(t:{NodeLabel}) " +
        "RETURN s.id AS sourceId, t.id AS targetId, r";

    public string IncomingEdges() =>
        $"MATCH (s:{NodeLabel})-[r:{RelationshipType}]->(t:{NodeLabel} {{{PropId}: $id}}) " +
        "RETURN s.id AS sourceId, t.id AS targetId, r";

    public string NodesByLabel() =>
        $"MATCH (n:{NodeLabel} {{{PropLabel}: $label}}) RETURN n";

    public string AllNodes() =>
        $"MATCH (n:{NodeLabel}) RETURN n";

    public string AllEdges() =>
        $"MATCH (s:{NodeLabel})-[r:{RelationshipType}]->(t:{NodeLabel}) " +
        "RETURN s.id AS sourceId, t.id AS targetId, r";

    public string NodeCount() =>
        $"MATCH (n:{NodeLabel}) RETURN count(n) AS c";

    public string EdgeCount() =>
        $"MATCH (:{NodeLabel})-[r:{RelationshipType}]->(:{NodeLabel}) RETURN count(r) AS c";

    public string Clear() =>
        $"MATCH (n:{NodeLabel}) DETACH DELETE n";

    // ---- Lossless model (de)serialization (T-independent parts). ----

    /// <summary>Serializes a property bag to a JSON string (null/empty becomes an empty object).</summary>
    public static string SerializeProperties(IDictionary<string, object>? properties)
    {
        if (properties == null || properties.Count == 0)
        {
            return "{}";
        }

        return JsonConvert.SerializeObject(properties);
    }

    /// <summary>
    /// Deserializes a property bag from a JSON string. Mirrors the JSON round-trip semantics of the other
    /// stores (e.g. integers come back as <see cref="long"/>), which
    /// <c>GraphNode.GetProperty</c>/<c>GraphEdge.GetProperty</c> already normalizes via
    /// <see cref="Convert.ChangeType(object, Type)"/>.
    /// </summary>
    public static Dictionary<string, object> DeserializeProperties(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return new Dictionary<string, object>();
        }

        return JsonConvert.DeserializeObject<Dictionary<string, object>>(json!) ?? new Dictionary<string, object>();
    }

    /// <summary>Converts a UTC-normalized timestamp to Unix epoch milliseconds (stored as a Neo4j long).</summary>
    public static long ToEpochMillis(DateTime value)
    {
        var utc = value.Kind == DateTimeKind.Unspecified
            ? DateTime.SpecifyKind(value, DateTimeKind.Utc)
            : value.ToUniversalTime();
        return new DateTimeOffset(utc).ToUnixTimeMilliseconds();
    }

    /// <summary>Converts an optional timestamp to nullable epoch milliseconds.</summary>
    public static long? ToEpochMillis(DateTime? value) =>
        value.HasValue ? ToEpochMillis(value.Value) : (long?)null;

    /// <summary>Reconstructs a UTC <see cref="DateTime"/> from epoch milliseconds.</summary>
    public static DateTime FromEpochMillis(long millis) =>
        DateTimeOffset.FromUnixTimeMilliseconds(millis).UtcDateTime;

    /// <summary>Reconstructs an optional UTC <see cref="DateTime"/> from nullable epoch milliseconds.</summary>
    public static DateTime? FromEpochMillis(long? millis) =>
        millis.HasValue ? FromEpochMillis(millis.Value) : (DateTime?)null;
}
