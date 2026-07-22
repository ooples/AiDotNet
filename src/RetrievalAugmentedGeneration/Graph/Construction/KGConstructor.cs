using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json.Linq;

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
/// When an <see cref="IGenerator{T}"/> LLM text generator is injected, construction instead prompts
/// the model to extract entities and relations as structured JSON (Microsoft GraphRAG parity),
/// parsing the result with Newtonsoft.Json. If no generator is injected, or if the LLM returns
/// malformed JSON, construction transparently falls back to the regex/heuristic path documented above
/// (an explicit offline fallback — never silent pretending).
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
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.DataIngestion)]
public class KGConstructor<T>
{
    /// <summary>
    /// Optional LLM text generator used for structured entity/relation extraction.
    /// When null, extraction uses the regex/heuristic fallback.
    /// </summary>
    private readonly IGenerator<T>? _generator;

    /// <summary>
    /// Initializes a new instance of the <see cref="KGConstructor{T}"/> class using only the
    /// regex/heuristic (offline, no-LLM) extraction path.
    /// </summary>
    public KGConstructor()
    {
        _generator = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="KGConstructor{T}"/> class with an optional
    /// LLM text generator. When <paramref name="generator"/> is non-null (and LLM extraction is
    /// enabled in the options), entities and relations are extracted by prompting the model for
    /// structured JSON; otherwise the regex/heuristic fallback is used.
    /// </summary>
    /// <param name="generator">
    /// The LLM text generator to drive extraction, or null to use the regex/heuristic fallback.
    /// </param>
    public KGConstructor(IGenerator<T>? generator)
    {
        _generator = generator;
    }

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
        var allClaims = new List<ExtractedClaim>();

        bool useLlm = _generator != null && opts.GetEffectiveUseLlmExtraction();
        bool extractClaims = useLlm && opts.GetEffectiveExtractClaims();

        foreach (var chunk in chunks)
        {
            List<ExtractedEntity> entities;
            List<ExtractedRelation> relations;

            // LLM-based extraction path (Microsoft GraphRAG parity), with a transparent
            // degrade-to-regex fallback whenever the model output cannot be parsed.
            if (useLlm &&
                TryExtractWithLlm(chunk, extractClaims, out var llmEntities, out var llmRelations, out var llmClaims))
            {
                entities = llmEntities;
                relations = llmRelations;
                allClaims.AddRange(llmClaims);
            }
            else
            {
                entities = ExtractEntities(chunk, opts.GetEffectiveEntityConfidenceThreshold());
                relations = ExtractRelations(chunk, entities, opts.GetEffectiveMaxEntitiesPerSentence());
            }

            allEntities.AddRange(entities);
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

        // Step 6: Attach extracted claims (covariates) to their subject nodes, if any.
        if (allClaims.Count > 0)
            AttachClaims(graph, allClaims);

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

    #region LLM-based extraction (Microsoft GraphRAG parity)

    /// <summary>
    /// Attempts LLM-driven structured extraction of entities and relations (and optionally claims)
    /// from a text chunk. Returns false — signaling the caller to use the regex/heuristic
    /// fallback — when the generator is unavailable, throws, returns empty text, or emits JSON that
    /// cannot be parsed into at least one entity.
    /// </summary>
    /// <param name="chunk">The text chunk to extract from.</param>
    /// <param name="includeClaims">Whether to request claim/covariate extraction.</param>
    /// <param name="entities">The parsed entities on success; empty otherwise.</param>
    /// <param name="relations">The parsed relations on success; empty otherwise.</param>
    /// <param name="claims">The parsed claims on success; empty otherwise.</param>
    /// <returns>True if extraction succeeded and yielded at least one entity; otherwise false.</returns>
    private bool TryExtractWithLlm(
        string chunk,
        bool includeClaims,
        out List<ExtractedEntity> entities,
        out List<ExtractedRelation> relations,
        out List<ExtractedClaim> claims)
    {
        entities = new List<ExtractedEntity>();
        relations = new List<ExtractedRelation>();
        claims = new List<ExtractedClaim>();

        if (_generator == null)
            return false;

        string response;
        try
        {
            response = _generator.Generate(BuildExtractionPrompt(chunk, includeClaims));
        }
        catch (Exception ex)
        {
            // Any generator failure degrades to the regex/heuristic fallback rather than throwing.
            System.Diagnostics.Trace.TraceWarning(
                $"KGConstructor: LLM extraction generator threw ({ex.GetType().Name}); using regex fallback.");
            return false;
        }

        if (string.IsNullOrWhiteSpace(response))
            return false;

        if (!TryParseExtractionJson(response, includeClaims, out entities, out relations, out claims))
        {
            System.Diagnostics.Trace.TraceWarning(
                "KGConstructor: LLM extraction returned malformed JSON; using regex fallback.");
            return false;
        }

        // A successful parse must yield at least one entity to be usable; otherwise fall back so
        // the graph is not silently left empty for a chunk that clearly contains content.
        if (entities.Count == 0)
            return false;

        // Ensure every relation endpoint exists as an entity node so no edges are silently dropped.
        EnsureRelationEndpointsExist(entities, relations);
        return true;
    }

    /// <summary>
    /// Builds the extraction prompt sent to the LLM. Requests strict JSON in a fixed schema.
    /// </summary>
    internal static string BuildExtractionPrompt(string chunk, bool includeClaims)
    {
        var sb = new StringBuilder();
        sb.AppendLine("You are an information-extraction system that builds knowledge graphs.");
        sb.AppendLine("Extract the named entities and the relationships between them from the TEXT below.");
        sb.AppendLine();
        sb.AppendLine("Respond with ONLY a single JSON object (no markdown, no code fences, no commentary)");
        sb.AppendLine("using EXACTLY this schema:");
        sb.AppendLine("{");
        sb.AppendLine("  \"entities\": [");
        sb.AppendLine("    { \"name\": \"<entity name>\", \"type\": \"<PERSON|ORGANIZATION|LOCATION|EVENT|CONCEPT|PRODUCT|OTHER>\", \"description\": \"<one sentence>\" }");
        sb.AppendLine("  ],");
        sb.AppendLine("  \"relations\": [");
        sb.AppendLine("    { \"source\": \"<entity name>\", \"relation\": \"<RELATION_TYPE>\", \"target\": \"<entity name>\", \"description\": \"<one sentence>\" }");
        if (includeClaims)
        {
            sb.AppendLine("  ],");
            sb.AppendLine("  \"claims\": [");
            sb.AppendLine("    { \"subject\": \"<entity name>\", \"object\": \"<entity name or NONE>\", \"type\": \"<claim type>\", \"description\": \"<one sentence>\", \"status\": \"<TRUE|FALSE|SUSPECTED>\" }");
            sb.AppendLine("  ]");
        }
        else
        {
            sb.AppendLine("  ]");
        }
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("Rules:");
        sb.AppendLine("- Use the entity's exact surface name. Relation source/target MUST match an entity name.");
        sb.AppendLine("- RELATION_TYPE should be a short UPPER_SNAKE_CASE verb phrase (e.g., WORKS_AT, BORN_IN, FOUNDED).");
        sb.AppendLine("- If there are no entities, return {\"entities\": [], \"relations\": []}.");
        sb.AppendLine();
        sb.AppendLine("TEXT:");
        sb.AppendLine("\"\"\"");
        sb.AppendLine(chunk);
        sb.AppendLine("\"\"\"");
        return sb.ToString();
    }

    /// <summary>
    /// Parses the LLM JSON response into entities, relations, and (optionally) claims.
    /// Tolerates surrounding prose or markdown code fences by isolating the outermost JSON object.
    /// Returns false on any parse failure so the caller can degrade to the regex fallback.
    /// </summary>
    internal static bool TryParseExtractionJson(
        string response,
        bool includeClaims,
        out List<ExtractedEntity> entities,
        out List<ExtractedRelation> relations,
        out List<ExtractedClaim> claims)
    {
        entities = new List<ExtractedEntity>();
        relations = new List<ExtractedRelation>();
        claims = new List<ExtractedClaim>();

        var json = ExtractJsonObject(response);
        if (json == null)
            return false;

        JObject root;
        try
        {
            root = JObject.Parse(json);
        }
        catch (Exception)
        {
            // Newtonsoft throws JsonReaderException/JsonException on malformed input; treat any
            // failure as malformed and degrade to the fallback.
            return false;
        }

        try
        {
            if (root["entities"] is JArray entityArray)
            {
                foreach (var item in entityArray)
                {
                    var name = GetString(item, "name");
                    if (string.IsNullOrWhiteSpace(name))
                        continue;

                    var type = GetString(item, "type");
                    entities.Add(new ExtractedEntity
                    {
                        Name = name.Trim(),
                        Label = NormalizeLabel(type),
                        Description = GetString(item, "description").Trim(),
                        Confidence = 0.9,
                        StartOffset = 0,
                        EndOffset = 0
                    });
                }
            }

            if (root["relations"] is JArray relationArray)
            {
                foreach (var item in relationArray)
                {
                    var source = GetString(item, "source");
                    var target = GetString(item, "target");
                    if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
                        continue;

                    var relationType = NormalizeRelationType(GetString(item, "relation"));
                    relations.Add(new ExtractedRelation
                    {
                        SourceEntity = source.Trim(),
                        TargetEntity = target.Trim(),
                        RelationType = relationType,
                        Description = GetString(item, "description").Trim(),
                        Confidence = 0.9
                    });
                }
            }

            if (includeClaims && root["claims"] is JArray claimArray)
            {
                foreach (var item in claimArray)
                {
                    var subject = GetString(item, "subject");
                    if (string.IsNullOrWhiteSpace(subject))
                        continue;

                    claims.Add(new ExtractedClaim
                    {
                        Subject = subject.Trim(),
                        Object = GetString(item, "object").Trim(),
                        ClaimType = GetString(item, "type").Trim(),
                        Description = GetString(item, "description").Trim(),
                        Status = GetString(item, "status").Trim()
                    });
                }
            }
        }
        catch (Exception)
        {
            return false;
        }

        // A response whose JSON parsed but contained neither an "entities" nor a "relations"
        // array is treated as malformed so we fall back.
        return root["entities"] != null || root["relations"] != null;
    }

    /// <summary>
    /// Isolates the outermost JSON object (from the first '{' to the last '}') within an LLM
    /// response, discarding any surrounding prose or markdown fences. Returns null if none found.
    /// </summary>
    private static string? ExtractJsonObject(string response)
    {
        if (string.IsNullOrEmpty(response))
            return null;

        int start = response.IndexOf('{');
        int end = response.LastIndexOf('}');
        if (start < 0 || end <= start)
            return null;

        return response.Substring(start, end - start + 1);
    }

    private static string GetString(JToken? token, string key)
    {
        if (token == null)
            return string.Empty;
        var value = token[key];
        if (value == null || value.Type == JTokenType.Null)
            return string.Empty;
        return value.ToString();
    }

    private static string NormalizeLabel(string type)
    {
        if (string.IsNullOrWhiteSpace(type))
            return "ENTITY";
        return type.Trim().ToUpperInvariant();
    }

    private static string NormalizeRelationType(string relation)
    {
        if (string.IsNullOrWhiteSpace(relation))
            return "RELATED_TO";

        var cleaned = relation.Trim().ToUpperInvariant();
        var sb = new StringBuilder(cleaned.Length);
        foreach (var ch in cleaned)
        {
            if (char.IsLetterOrDigit(ch))
                sb.Append(ch);
            else if (ch == ' ' || ch == '-' || ch == '_')
                sb.Append('_');
            // drop other punctuation
        }

        var result = sb.ToString().Trim('_');
        return result.Length == 0 ? "RELATED_TO" : result;
    }

    /// <summary>
    /// Adds a minimal entity for any relation endpoint that was not itself extracted as an entity,
    /// so LLM-declared edges are not dropped during graph construction.
    /// </summary>
    private static void EnsureRelationEndpointsExist(
        List<ExtractedEntity> entities,
        List<ExtractedRelation> relations)
    {
        var known = new HashSet<string>(entities.Select(e => e.Name), StringComparer.OrdinalIgnoreCase);
        foreach (var relation in relations)
        {
            AddMissingEndpoint(entities, known, relation.SourceEntity);
            AddMissingEndpoint(entities, known, relation.TargetEntity);
        }
    }

    private static void AddMissingEndpoint(
        List<ExtractedEntity> entities,
        HashSet<string> known,
        string name)
    {
        if (string.IsNullOrWhiteSpace(name) || known.Contains(name))
            return;

        known.Add(name);
        entities.Add(new ExtractedEntity
        {
            Name = name,
            Label = "ENTITY",
            Confidence = 0.7,
            StartOffset = 0,
            EndOffset = 0
        });
    }

    /// <summary>
    /// Attaches extracted claims to their subject nodes under the "claims" property (as a list of
    /// descriptions). Claims whose subject is not present in the graph are ignored.
    /// </summary>
    private static void AttachClaims(KnowledgeGraph<T> graph, List<ExtractedClaim> claims)
    {
        foreach (var claim in claims)
        {
            if (string.IsNullOrWhiteSpace(claim.Subject))
                continue;

            string nodeId = claim.Subject.ToLowerInvariant().Replace(' ', '_');
            var node = graph.GetNode(nodeId);
            if (node == null)
                continue;

            var existing = node.GetProperty<List<string>>("claims") ?? new List<string>();
            var text = string.IsNullOrWhiteSpace(claim.ClaimType)
                ? claim.Description
                : $"[{claim.ClaimType}] {claim.Description}";
            if (!string.IsNullOrWhiteSpace(claim.Status))
                text += $" (status: {claim.Status})";
            existing.Add(text.Trim());
            node.SetProperty("claims", existing);
        }
    }

    #endregion

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
                Description = entity.Description,
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
            Description = r.Description,
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
            if (!string.IsNullOrWhiteSpace(entity.Description))
                node.SetProperty("description", entity.Description);
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
                if (!string.IsNullOrWhiteSpace(relation.Description))
                    edge.SetProperty("description", relation.Description);
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
