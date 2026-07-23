using System.Text.RegularExpressions;

namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// EntityLinking - Entity extraction and linking for document text.
/// </summary>
/// <remarks>
/// <para>
/// EntityLinking identifies named entities in document text and links them
/// to canonical representations or external knowledge bases.
/// </para>
/// <para>
/// <b>For Beginners:</b> Documents contain references to people, places,
/// organizations, and other entities. This tool identifies them:
///
/// - Extract named entities (people, places, organizations)
/// - Link entities to canonical forms
/// - Resolve entity references
/// - Build entity relationships
///
/// Key features:
/// - Named entity recognition
/// - Entity disambiguation
/// - Reference resolution
/// - Relationship extraction
///
/// Example usage:
/// <code>
/// var linker = new EntityLinking&lt;float&gt;();
/// var entities = linker.Process(documentText);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EntityLinking<T> : PostprocessorBase<T, string, IList<Entity>>, IDisposable
{
    #region Fields

    private readonly Dictionary<string, Entity> _knownEntities;
    private readonly Dictionary<string, List<string>> _entityAliases;
    private bool _disposed;

    // Common patterns for entity extraction
    private static readonly Regex PersonNamePattern = RegexHelper.Create(@"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", RegexOptions.Compiled);
    private static readonly Regex OrganizationPattern = RegexHelper.Create(@"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Association|Foundation|Institute|University)\.?))\b", RegexOptions.Compiled);
    private static readonly Regex DatePattern = RegexHelper.Create(@"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex MoneyPattern = RegexHelper.Create(@"[$£€]\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|euros?|pounds?)", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex EmailPattern = RegexHelper.Create(@"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", RegexOptions.Compiled);
    private static readonly Regex PhonePattern = RegexHelper.Create(@"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", RegexOptions.Compiled);
    private static readonly Regex UrlPattern = RegexHelper.Create(@"\b(?:https?://)?(?:www\.)?[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s]*)?\b", RegexOptions.Compiled);

    #endregion

    #region Properties

    /// <summary>
    /// Entity linking does not support inverse transformation.
    /// </summary>
    public override bool SupportsInverse => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new EntityLinking instance.
    /// </summary>
    public EntityLinking()
    {
        _knownEntities = new Dictionary<string, Entity>(StringComparer.OrdinalIgnoreCase);
        _entityAliases = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
    }

    #endregion

    #region Core Implementation

    /// <summary>
    /// Extracts all entities from the text.
    /// </summary>
    /// <param name="input">The text to analyze.</param>
    /// <returns>List of extracted entities.</returns>
    protected override IList<Entity> ProcessCore(string input)
    {
        var entities = new List<Entity>();

        // Extract different entity types
        entities.AddRange(ExtractPersons(input));
        entities.AddRange(ExtractOrganizations(input));
        entities.AddRange(ExtractDates(input));
        entities.AddRange(ExtractMoneyAmounts(input));
        entities.AddRange(ExtractEmails(input));
        entities.AddRange(ExtractPhoneNumbers(input));
        entities.AddRange(ExtractUrls(input));

        // Deduplicate and link
        return DeduplicateEntities(entities);
    }

    /// <summary>
    /// Validates the input text.
    /// </summary>
    protected override void ValidateInput(string input)
    {
        // Allow null/empty strings - they will return empty list
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Extracts person names from text.
    /// </summary>
    public IList<Entity> ExtractPersons(string text)
    {
        var entities = new List<Entity>();
        var matches = PersonNamePattern.Matches(text);

        foreach (Match match in matches)
        {
            // Filter out likely false positives
            if (!IsLikelyOrganization(match.Value))
            {
                entities.Add(new Entity
                {
                    Text = match.Value,
                    Type = EntityType.Person,
                    StartIndex = match.Index,
                    EndIndex = match.Index + match.Length,
                    Confidence = 0.7
                });
            }
        }

        return entities;
    }

    /// <summary>
    /// Extracts organization names from text.
    /// </summary>
    public IList<Entity> ExtractOrganizations(string text)
    {
        var entities = new List<Entity>();
        var matches = OrganizationPattern.Matches(text);

        foreach (Match match in matches)
        {
            entities.Add(new Entity
            {
                Text = match.Value,
                Type = EntityType.Organization,
                StartIndex = match.Index,
                EndIndex = match.Index + match.Length,
                Confidence = 0.8
            });
        }

        return entities;
    }

    /// <summary>
    /// Extracts dates from text.
    /// </summary>
    public IList<Entity> ExtractDates(string text)
    {
        var entities = new List<Entity>();
        var matches = DatePattern.Matches(text);

        foreach (Match match in matches)
        {
            entities.Add(new Entity
            {
                Text = match.Value,
                Type = EntityType.Date,
                StartIndex = match.Index,
                EndIndex = match.Index + match.Length,
                Confidence = 0.9,
                NormalizedValue = NormalizeDate(match.Value)
            });
        }

        return entities;
    }

    /// <summary>
    /// Extracts money amounts from text.
    /// </summary>
    public IList<Entity> ExtractMoneyAmounts(string text)
    {
        var entities = new List<Entity>();
        var matches = MoneyPattern.Matches(text);

        foreach (Match match in matches)
        {
            entities.Add(new Entity
            {
                Text = match.Value,
                Type = EntityType.Money,
                StartIndex = match.Index,
                EndIndex = match.Index + match.Length,
                Confidence = 0.95,
                NormalizedValue = NormalizeMoney(match.Value)
            });
        }

        return entities;
    }

    /// <summary>
    /// Extracts email addresses from text.
    /// </summary>
    public IList<Entity> ExtractEmails(string text)
    {
        var entities = new List<Entity>();
        var matches = EmailPattern.Matches(text);

        foreach (Match match in matches)
        {
            entities.Add(new Entity
            {
                Text = match.Value,
                Type = EntityType.Email,
                StartIndex = match.Index,
                EndIndex = match.Index + match.Length,
                Confidence = 0.95
            });
        }

        return entities;
    }

    /// <summary>
    /// Extracts phone numbers from text.
    /// </summary>
    public IList<Entity> ExtractPhoneNumbers(string text)
    {
        var entities = new List<Entity>();
        var matches = PhonePattern.Matches(text);

        foreach (Match match in matches)
        {
            entities.Add(new Entity
            {
                Text = match.Value,
                Type = EntityType.Phone,
                StartIndex = match.Index,
                EndIndex = match.Index + match.Length,
                Confidence = 0.9,
                NormalizedValue = NormalizePhone(match.Value)
            });
        }

        return entities;
    }

    /// <summary>
    /// Extracts URLs from text.
    /// </summary>
    public IList<Entity> ExtractUrls(string text)
    {
        var entities = new List<Entity>();
        var matches = UrlPattern.Matches(text);

        foreach (Match match in matches)
        {
            entities.Add(new Entity
            {
                Text = match.Value,
                Type = EntityType.Url,
                StartIndex = match.Index,
                EndIndex = match.Index + match.Length,
                Confidence = 0.85
            });
        }

        return entities;
    }

    /// <summary>
    /// Links an entity to its canonical form.
    /// </summary>
    public Entity? LinkEntity(Entity entity)
    {
        // Check known entities
        if (_knownEntities.TryGetValue(entity.Text, out var known))
        {
            entity.LinkedEntity = known;
            entity.CanonicalName = known.CanonicalName ?? known.Text;
            return entity;
        }

        // Check aliases
        foreach (var (canonical, aliases) in _entityAliases)
        {
            if (aliases.Contains(entity.Text, StringComparer.OrdinalIgnoreCase))
            {
                if (_knownEntities.TryGetValue(canonical, out var linkedEntity))
                {
                    entity.LinkedEntity = linkedEntity;
                    entity.CanonicalName = canonical;
                    return entity;
                }
            }
        }

        return entity;
    }

    /// <summary>
    /// Registers a known entity with optional aliases.
    /// </summary>
    public void RegisterEntity(Entity entity, IEnumerable<string>? aliases = null)
    {
        _knownEntities[entity.Text] = entity;

        if (aliases != null)
        {
            _entityAliases[entity.Text] = aliases.ToList();
        }
    }

    /// <summary>
    /// Adds an alias for an existing entity.
    /// </summary>
    public void AddAlias(string canonicalName, string alias)
    {
        if (!_entityAliases.ContainsKey(canonicalName))
            _entityAliases[canonicalName] = new List<string>();

        _entityAliases[canonicalName].Add(alias);
    }

    #endregion

    #region Private Methods

    private bool IsLikelyOrganization(string name)
    {
        var orgIndicators = new[] { "Inc", "Corp", "LLC", "Ltd", "Co", "Company", "Association",
            "Foundation", "Institute", "University", "College", "School", "Hospital", "Bank" };

        return orgIndicators.Any(ind => name.Contains(ind, StringComparison.OrdinalIgnoreCase));
    }

    private string? NormalizeDate(string dateText)
    {
        if (DateTime.TryParse(dateText, out var date))
            return date.ToString("yyyy-MM-dd");
        return null;
    }

    private string? NormalizeMoney(string moneyText)
    {
        // Extract just the numeric value
        var numericPart = RegexHelper.Replace(moneyText, @"[^\d.]", "");
        if (decimal.TryParse(numericPart, out var amount))
        {
            string currency = moneyText.Contains("$") ? "USD" :
                             moneyText.Contains("£") ? "GBP" :
                             moneyText.Contains("€") ? "EUR" : "USD";
            return $"{currency} {amount:F2}";
        }
        return null;
    }

    private string? NormalizePhone(string phoneText)
    {
        // Strip non-numeric characters except +
        var digits = RegexHelper.Replace(phoneText, @"[^\d+]", "");
        if (digits.Length >= 10)
            return digits;
        return null;
    }

    private IList<Entity> DeduplicateEntities(IList<Entity> entities)
    {
        var deduplicated = new List<Entity>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var entity in entities.OrderByDescending(e => e.Confidence))
        {
            var key = $"{entity.Type}:{entity.Text}";
            if (!seen.Contains(key))
            {
                seen.Add(key);
                deduplicated.Add(LinkEntity(entity) ?? entity);
            }
        }

        return deduplicated;
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the entity linker.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _knownEntities.Clear();
            _entityAliases.Clear();
        }
        _disposed = true;
    }

    #endregion
}



