using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;

/// <summary>
/// Constructs a knowledge graph from unstructured text using heuristic entity and relation extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for graph operations.</typeparam>
/// <remarks>
/// <para>
/// The construction pipeline operates in four stages:
/// 1. <b>Chunk Text:</b> Split text into overlapping chunks for processing
/// 2. <b>Extract Entities:</b> Identify named entities using regex patterns and capitalization heuristics
/// 3. <b>Extract Relations:</b> Detect relations via co-occurrence and proximity-based patterns
/// 4. <b>Entity Resolution:</b> Merge similar entity names to reduce duplicates
///
/// This implementation works without an external LLM (purely heuristic), providing a baseline
/// that can be extended or replaced with LLM-based extraction.
/// </para>
/// <para><b>For Beginners:</b> This class reads text and automatically builds a knowledge graph.
///
/// Given text: "Albert Einstein was born in Ulm, Germany. He worked at Princeton University."
/// It extracts:
/// - Entities: Albert Einstein (PERSON), Ulm (LOCATION), Germany (LOCATION), Princeton University (ORGANIZATION)
/// - Relations: Einstein BORN_IN Ulm, Einstein LOCATED_IN Germany, Einstein WORKED_AT Princeton University
///
/// The heuristic approach uses patterns like:
/// - Capitalized words = likely entity names
/// - Words like "born in", "works at", "located in" = relation indicators
/// - Entities appearing near each other = likely related
/// </para>
/// </remarks>
public class KGConstructor<T>
{
    // Matches capitalized phrases including:
    // - Multi-word names: "Albert Einstein", "New York City"
    // - Names with connectors: "University of Cambridge", "Ludwig van Beethoven"
    // - Hyphenated names: "Jean-Pierre", "Rolls-Royce"
    // - Titles followed by names: "Dr. Smith", "Prof. Einstein"
    // - Possessives are excluded via post-processing
    // Atomic groups (?>...) prevent catastrophic backtracking on nested quantifiers
    // Atomic groups (?>...) prevent catastrophic backtracking on nested quantifiers.
    // The 1-second timeout is a defense-in-depth safeguard for untrusted input that
    // may contain unexpected Unicode or patterns the atomic groups don't fully cover.
    private static readonly Regex CapitalizedPhraseRegex = new(
        @"\b(?:(?:Dr|Mr|Mrs|Ms|Prof|Rev|Gen|Sgt|Cpt|Sir|Dame)\.?\s+)?((?>[A-Z][a-z]+(?:-[A-Z][a-z]+)*)(?:\s+(?:(?:of|the|and|for|in|at|de|von|van|del|la|le|el)\s+)?(?>[A-Z][a-z]+(?:-[A-Z][a-z]+)*))*)(?:'s)?\b",
        RegexOptions.Compiled, TimeSpan.FromSeconds(1));

    // Matches abbreviations: U.S.A., U.N., NATO, IBM, etc.
    private static readonly Regex AbbreviationRegex = new(
        @"\b((?:[A-Z]\.){2,}[A-Z]?\.?|[A-Z]{2,})\b",
        RegexOptions.Compiled, TimeSpan.FromSeconds(1));

    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private static readonly (Regex regex, string relationType)[] RelationPatterns =
    [
        (new Regex(@"born\s+in", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "BORN_IN"),
        (new Regex(@"lives?\s+in", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "LIVES_IN"),
        (new Regex(@"located\s+in", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "LOCATED_IN"),
        (new Regex(@"work(?:s|ed)?\s+(?:at|for)", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "WORKS_AT"),
        (new Regex(@"found(?:ed)?\s+(?:by)?", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "FOUNDED"),
        (new Regex(@"created?\s+(?:by)?", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "CREATED"),
        (new Regex(@"part\s+of", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "PART_OF"),
        (new Regex(@"member\s+of", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "MEMBER_OF"),
        (new Regex(@"CEO\s+of", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "CEO_OF"),
        (new Regex(@"president\s+of", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "PRESIDENT_OF"),
        (new Regex(@"capital\s+of", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "CAPITAL_OF"),
        (new Regex(@"married?\s+to", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "MARRIED_TO"),
        (new Regex(@"studied\s+at", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "STUDIED_AT"),
        (new Regex(@"graduated\s+from", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "GRADUATED_FROM"),
        (new Regex(@"acquired\s+(?:by)?", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "ACQUIRED"),
        (new Regex(@"developed\s+(?:by)?", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "DEVELOPED"),
        (new Regex(@"owned\s+by", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "OWNED_BY"),
        (new Regex(@"is\s+a", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "IS_A"),
        (new Regex(@"known\s+(?:as|for)", RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), "KNOWN_FOR")
    ];

    private static readonly HashSet<string> OrganizationIndicators = new(StringComparer.OrdinalIgnoreCase)
    {
        "University", "Institute", "Corporation", "Inc", "Ltd", "Company",
        "Foundation", "Organization", "Association", "Department", "Agency",
        "Hospital", "School", "College", "Bank", "Group"
    };

    private static readonly HashSet<string> LocationIndicators = new(StringComparer.OrdinalIgnoreCase)
    {
        "City", "State", "Country", "Province", "Region", "Island",
        "Mountain", "River", "Lake", "Ocean", "Sea", "Street", "Avenue"
    };

    /// <summary>
    /// Constructs a knowledge graph from the given text.
    /// </summary>
    /// <param name="text">The input text to extract entities and relations from.</param>
    /// <param name="graph">The target knowledge graph to populate. If null, a new one is created.</param>
    /// <param name="options">Construction options.</param>
    /// <returns>The populated knowledge graph.</returns>
    public KnowledgeGraph<T> ConstructFromText(
        string text,
        KnowledgeGraph<T>? graph = null,
        KGConstructionOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        var opts = options ?? new KGConstructionOptions();
        opts.ValidateCrossFieldConstraints();
        graph ??= new KnowledgeGraph<T>();

        // Step 1: Chunk text
        var chunks = ChunkText(text, opts.GetEffectiveMaxChunkSize(), opts.GetEffectiveChunkOverlap());

        // Step 2 & 3: Extract entities and relations from each chunk
        var allEntities = new List<ExtractedEntity>();
        var allRelations = new List<ExtractedRelation>();

        foreach (var chunk in chunks)
        {
            var entities = ExtractEntities(chunk, opts.GetEffectiveEntityConfidenceThreshold());
            allEntities.AddRange(entities);

            var relations = ExtractRelations(chunk, entities, opts.GetEffectiveMaxEntitiesPerSentence());
            allRelations.AddRange(relations);
        }

        // Step 4: Entity resolution
        if (opts.GetEffectiveEnableEntityResolution())
        {
            var resolvedMap = ResolveEntities(allEntities, opts.GetEffectiveEntitySimilarityThreshold());
            allEntities = DeduplicateEntities(allEntities, resolvedMap);
            allRelations = RemapRelations(allRelations, resolvedMap);
        }

        // Step 5: Add to graph
        AddToGraph(graph, allEntities, allRelations);

        return graph;
    }

    /// <summary>
    /// Extracts entities from a text chunk using heuristic patterns.
    /// </summary>
    /// <param name="text">Text to extract entities from.</param>
    /// <param name="confidenceThreshold">Minimum confidence to include an entity.</param>
    /// <returns>List of extracted entities.</returns>
    public List<ExtractedEntity> ExtractEntities(string text, double confidenceThreshold = 0.5)
    {
        var entities = new List<ExtractedEntity>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Pattern 1: Capitalized phrases (proper nouns, titles, hyphenated names)
        foreach (Match match in CapitalizedPhraseRegex.Matches(text))
        {
            // Use capture group 1 which excludes titles
            var name = match.Groups[1].Value.Trim();
            if (name.Length < 2 || seen.Contains(name)) continue;

            // Strip trailing possessive 's if somehow captured
            if (name.EndsWith("'s", StringComparison.Ordinal))
                name = name.Substring(0, name.Length - 2).TrimEnd();

            // Skip common sentence starters and short articles
            if (IsCommonWord(name)) continue;

            seen.Add(name);
            string label = ClassifyEntity(name);
            double confidence = ComputeEntityConfidence(name, text);

            if (confidence >= confidenceThreshold)
            {
                entities.Add(new ExtractedEntity
                {
                    Name = name,
                    Label = label,
                    Confidence = confidence,
                    StartOffset = match.Index,
                    EndOffset = match.Index + match.Length
                });
            }
        }

        // Pattern 2: Abbreviations (U.S.A., NATO, IBM, etc.)
        foreach (Match match in AbbreviationRegex.Matches(text))
        {
            var name = match.Value.Trim();
            if (name.Length < 2 || seen.Contains(name)) continue;
            if (IsCommonWord(name)) continue;

            seen.Add(name);
            double confidence = ComputeEntityConfidence(name, text);

            if (confidence >= confidenceThreshold)
            {
                entities.Add(new ExtractedEntity
                {
                    Name = name,
                    Label = "ORGANIZATION", // Abbreviations are usually organizations
                    Confidence = confidence,
                    StartOffset = match.Index,
                    EndOffset = match.Index + match.Length
                });
            }
        }

        return entities;
    }

    /// <summary>
    /// Extracts relations between entities based on proximity and pattern matching.
    /// </summary>
    /// <param name="text">The text to analyze.</param>
    /// <param name="entities">Entities already extracted from this text.</param>
    /// <returns>List of extracted relations.</returns>
    public List<ExtractedRelation> ExtractRelations(string text, List<ExtractedEntity> entities, int maxEntitiesPerSentence = 20)
    {
        var relations = new List<ExtractedRelation>();
        if (entities.Count < 2) return relations;

        // Pattern-based relation extraction (pre-compiled static regexes)
        foreach (var (regex, relationType) in RelationPatterns)
        {
            foreach (Match match in regex.Matches(text))
            {
                // Find closest entity before and after the pattern
                var before = entities
                    .Where(e => e.EndOffset <= match.Index)
                    .OrderByDescending(e => e.EndOffset)
                    .FirstOrDefault();

                var after = entities
                    .Where(e => e.StartOffset >= match.Index + match.Length)
                    .OrderBy(e => e.StartOffset)
                    .FirstOrDefault();

                if (before != null && after != null)
                {
                    relations.Add(new ExtractedRelation
                    {
                        SourceEntity = before.Name,
                        TargetEntity = after.Name,
                        RelationType = relationType,
                        Confidence = Math.Min(before.Confidence, after.Confidence) * 0.9
                    });
                }
            }
        }

        // Co-occurrence based relations (entities in same sentence, no explicit pattern)
        var existingPairs = new HashSet<(string, string)>(
            relations.Select(r => (r.SourceEntity, r.TargetEntity)));

        var sentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        int offset = 0;
        foreach (var sentence in sentences)
        {
            var sentenceEntities = entities
                .Where(e => e.StartOffset >= offset && e.EndOffset <= offset + sentence.Length)
                .Take(maxEntitiesPerSentence)
                .ToList();

            // Generate co-occurrence relations between entities in the same sentence
            for (int i = 0; i < sentenceEntities.Count; i++)
            {
                for (int j = i + 1; j < sentenceEntities.Count; j++)
                {
                    var pair = (sentenceEntities[i].Name, sentenceEntities[j].Name);
                    if (!existingPairs.Contains(pair))
                    {
                        existingPairs.Add(pair);
                        relations.Add(new ExtractedRelation
                        {
                            SourceEntity = pair.Item1,
                            TargetEntity = pair.Item2,
                            RelationType = "RELATED_TO",
                            Confidence = 0.3 // Low confidence for co-occurrence
                        });
                    }
                }
            }

            offset += sentence.Length + 1; // +1 for the delimiter
        }

        return relations;
    }

    private static List<string> ChunkText(string text, int maxChunkSize, int overlap)
    {
        if (maxChunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxChunkSize), "MaxChunkSize must be > 0.");
        if (overlap < 0 || overlap >= maxChunkSize)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap must be >= 0 and < MaxChunkSize.");

        var chunks = new List<string>();
        if (text.Length <= maxChunkSize)
        {
            chunks.Add(text);
            return chunks;
        }

        int start = 0;
        while (start < text.Length)
        {
            int length = Math.Min(maxChunkSize, text.Length - start);

            // Try to break at sentence boundary, but only if it gives at least half a chunk of progress
            if (start + length < text.Length)
            {
                int searchStart = start + length - 1;
                int lastPeriod = text.LastIndexOf('.', searchStart, Math.Min(length, text.Length - start));
                if (lastPeriod > start + length / 2)
                    length = lastPeriod - start + 1;
            }

            chunks.Add(text.Substring(start, length));
            start += length - overlap;
            if (start + overlap >= text.Length) break;
        }

        return chunks;
    }

    private static Dictionary<string, string> ResolveEntities(
        List<ExtractedEntity> entities, double similarityThreshold)
    {
        var resolvedMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var uniqueNames = entities.Select(e => e.Name).Distinct(StringComparer.OrdinalIgnoreCase).ToList();

        for (int i = 0; i < uniqueNames.Count; i++)
        {
            if (resolvedMap.ContainsKey(uniqueNames[i])) continue;

            for (int j = i + 1; j < uniqueNames.Count; j++)
            {
                if (resolvedMap.ContainsKey(uniqueNames[j])) continue;

                double similarity = ComputeStringSimilarity(uniqueNames[i], uniqueNames[j]);
                if (similarity >= similarityThreshold)
                {
                    // Merge shorter name into longer name (longer is usually more complete)
                    if (uniqueNames[i].Length >= uniqueNames[j].Length)
                        resolvedMap[uniqueNames[j]] = uniqueNames[i];
                    else
                        resolvedMap[uniqueNames[i]] = uniqueNames[j];
                }
            }
        }

        return resolvedMap;
    }

    private static List<ExtractedEntity> DeduplicateEntities(
        List<ExtractedEntity> entities, Dictionary<string, string> resolvedMap)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var result = new List<ExtractedEntity>();

        foreach (var entity in entities)
        {
            var canonicalName = resolvedMap.TryGetValue(entity.Name, out var resolved)
                ? resolved
                : entity.Name;

            if (seen.Contains(canonicalName)) continue;
            seen.Add(canonicalName);

            result.Add(new ExtractedEntity
            {
                Name = canonicalName,
                Label = entity.Label,
                Confidence = entity.Confidence,
                StartOffset = entity.StartOffset,
                EndOffset = entity.EndOffset
            });
        }

        return result;
    }

    private static List<ExtractedRelation> RemapRelations(
        List<ExtractedRelation> relations, Dictionary<string, string> resolvedMap)
    {
        return relations.Select(r => new ExtractedRelation
        {
            SourceEntity = resolvedMap.TryGetValue(r.SourceEntity, out var s) ? s : r.SourceEntity,
            TargetEntity = resolvedMap.TryGetValue(r.TargetEntity, out var t) ? t : r.TargetEntity,
            RelationType = r.RelationType,
            Confidence = r.Confidence
        })
        .Where(r => r.SourceEntity != r.TargetEntity) // Remove self-loops from resolution
        .ToList();
    }

    private static void AddToGraph(
        KnowledgeGraph<T> graph,
        List<ExtractedEntity> entities,
        List<ExtractedRelation> relations)
    {
        var entityIdMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var entity in entities)
        {
            string nodeId = entity.Name.ToLowerInvariant().Replace(' ', '_');
            entityIdMap[entity.Name] = nodeId;

            var node = new GraphNode<T>(nodeId, entity.Label);
            node.SetProperty("name", entity.Name);
            node.SetProperty("confidence", entity.Confidence);
            graph.AddNode(node);
        }

        var failedEdges = new List<string>();

        foreach (var relation in relations)
        {
            if (!entityIdMap.TryGetValue(relation.SourceEntity, out var sourceId) ||
                !entityIdMap.TryGetValue(relation.TargetEntity, out var targetId))
                continue;

            try
            {
                var edge = new GraphEdge<T>(sourceId, targetId, relation.RelationType,
                    Math.Max(0.0, Math.Min(1.0, relation.Confidence)));
                graph.AddEdge(edge);
            }
            catch (InvalidOperationException ex)
            {
                string failure = $"{sourceId} -[{relation.RelationType}]-> {targetId}: {ex.Message}";
                failedEdges.Add(failure);
                System.Diagnostics.Trace.TraceWarning($"KGConstructor: Failed to add edge {failure}");
            }
        }

        if (failedEdges.Count > 0)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"KGConstructor: {failedEdges.Count} edge(s) failed during graph construction.");
        }
    }

    private static string ClassifyEntity(string name)
    {
        var words = name.Split(new[] { ' ', '-' }, StringSplitOptions.RemoveEmptyEntries);
        var lastWord = words.Length > 0 ? words[^1] : name;

        // All-caps abbreviation is usually an organization
        if (name.Length >= 2 && name.Replace(".", "").All(char.IsUpper))
            return "ORGANIZATION";

        if (OrganizationIndicators.Contains(lastWord))
            return "ORGANIZATION";
        if (LocationIndicators.Contains(lastWord))
            return "LOCATION";

        // Multi-word capitalized phrases with 2-3 words are likely person names
        if (words.Length >= 2 && words.Length <= 4 && words.All(w => w.Length > 0 && char.IsUpper(w[0])))
            return "PERSON";

        return "ENTITY";
    }

    private static double ComputeEntityConfidence(string name, string text)
    {
        double confidence = 0.5;

        // Longer names are more likely to be real entities
        if (name.Length > 10) confidence += 0.1;
        if (name.Contains(' ')) confidence += 0.15; // Multi-word = more specific

        // Frequent mentions increase confidence
        int count = 0;
        int index = 0;
        while ((index = text.IndexOf(name, index, StringComparison.OrdinalIgnoreCase)) >= 0)
        {
            count++;
            index += name.Length;
        }
        if (count > 1) confidence += 0.1;
        if (count > 3) confidence += 0.1;

        return Math.Min(1.0, confidence);
    }

    private static readonly HashSet<string> CommonWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "The", "This", "That", "These", "Those", "There", "Here",
        "What", "When", "Where", "Which", "Who", "How", "Why",
        "Some", "Many", "Most", "All", "Each", "Every",
        "However", "Therefore", "Furthermore", "Moreover", "Although",
        "It", "He", "She", "They", "We", "But", "And", "Or",
        "In", "On", "At", "To", "For", "Of", "By", "As",
        "If", "So", "No", "Not", "Yes", "An", "A",
        "After", "Before", "During", "While", "Since", "Until",
        "Also", "Then", "Next", "First", "Second", "Finally",
        "Just", "Only", "About", "Into", "From", "With"
    };

    private static bool IsCommonWord(string word) => CommonWords.Contains(word);

    private static double ComputeStringSimilarity(string a, string b)
    {
        if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b)) return 0.0;

        // Check if one contains the other
        if (a.IndexOf(b, StringComparison.OrdinalIgnoreCase) >= 0 ||
            b.IndexOf(a, StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return (double)Math.Min(a.Length, b.Length) / Math.Max(a.Length, b.Length);
        }

        // Levenshtein distance-based similarity
        int maxLen = Math.Max(a.Length, b.Length);
        if (maxLen == 0) return 1.0;

        int distance = LevenshteinDistance(a.ToLowerInvariant(), b.ToLowerInvariant());
        return 1.0 - (double)distance / maxLen;
    }

    private static int LevenshteinDistance(string s, string t)
    {
        int n = s.Length;
        int m = t.Length;

        // Guard against pathologically long strings (e.g., regex bugs producing sentence-length "entities")
        if (n > 1000 || m > 1000)
            throw new ArgumentException(
                $"Input strings exceed the maximum supported length of 1000 characters (got {n} and {m}). " +
                "This typically indicates malformed entity extraction output.");

        // Two-row optimization: O(min(n,m)) memory instead of O(n*m)
        var prev = new int[m + 1];
        var curr = new int[m + 1];

        for (int j = 0; j <= m; j++) prev[j] = j;

        for (int i = 1; i <= n; i++)
        {
            curr[0] = i;
            for (int j = 1; j <= m; j++)
            {
                int cost = s[i - 1] == t[j - 1] ? 0 : 1;
                curr[j] = Math.Min(
                    Math.Min(prev[j] + 1, curr[j - 1] + 1),
                    prev[j - 1] + cost);
            }
            (prev, curr) = (curr, prev);
        }

        return prev[m];
    }
}
