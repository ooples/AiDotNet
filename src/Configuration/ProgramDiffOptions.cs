namespace AiDotNet.Configuration;

/// <summary>Configures how SEARCH/REPLACE edit blocks are parsed from a model response and applied to a program.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation exposes one regular expression,
/// <c>&lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH\n(.*?)=======\n(.*?)&gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE</c>, which requires
/// bare line feeds and therefore silently matches nothing at all when the model emits Windows line endings or
/// leaves a trailing space after a marker. These options drive a line-based parser instead, so the marker text is
/// still configurable but carriage returns, indentation, and trailing spaces no longer decide whether an edit is
/// seen. <see cref="AllowCarriageReturns"/> exists to opt back into the strict line-feed-only behaviour when a
/// pipeline wants byte-for-byte parity with the reference parser.
/// </para>
/// <para>
/// <see cref="FuzzyWhitespace"/> and <see cref="RejectWhenNoBlockApplied"/> address the second reference-parser
/// weakness: a block whose SEARCH text is absent is skipped without a word, so a child program can be byte
/// identical to its parent and still consume evaluation budget. With the default settings a run instead learns
/// that nothing applied and can retry with feedback.
/// </para>
/// <para><b>For Beginners:</b> Language models edit code by sending "find this text, replace it with that text"
/// instructions wrapped in special marker lines. These settings control how forgiving the reader of those
/// instructions is. The defaults accept Windows and Unix line endings, require the text being searched for to
/// actually exist in the file, and refuse an edit that would change nothing. Leave them alone unless you are
/// reproducing another tool's exact behaviour or your model writes markers in an unusual style.</para>
/// </remarks>
public sealed class ProgramDiffOptions
{
    /// <summary>The marker line that opens the text to search for.</summary>
    public const string DefaultSearchMarker = "<<<<<<< SEARCH";

    /// <summary>The marker line that separates the search text from the replacement text.</summary>
    public const string DefaultDividerMarker = "=======";

    /// <summary>The marker line that closes the replacement text.</summary>
    public const string DefaultReplaceMarker = ">>>>>>> REPLACE";

    /// <summary>Gets or sets the marker line that opens the text to search for.</summary>
    public string SearchMarker { get; set; } = DefaultSearchMarker;

    /// <summary>Gets or sets the marker line that separates the search text from the replacement text.</summary>
    public string DividerMarker { get; set; } = DefaultDividerMarker;

    /// <summary>Gets or sets the marker line that closes the replacement text.</summary>
    public string ReplaceMarker { get; set; } = DefaultReplaceMarker;

    /// <summary>Gets or sets whether responses containing carriage returns are accepted.</summary>
    /// <remarks>
    /// The default <c>true</c> normalizes CRLF and CR to line feeds before parsing. Set it to <c>false</c> to
    /// reproduce the reference implementation, which only matches its pattern against line-feed-separated text.
    /// </remarks>
    public bool AllowCarriageReturns { get; set; } = true;

    /// <summary>Gets or sets whether an exact match failure retries with white space normalized per line.</summary>
    /// <remarks>
    /// Leading indentation is preserved by the comparison; only trailing white space and internal runs of spaces
    /// and tabs are collapsed, so a block cannot silently attach to structurally different code.
    /// </remarks>
    public bool FuzzyWhitespace { get; set; }

    /// <summary>Gets or sets whether applying zero blocks, or producing an unchanged program, is a failure.</summary>
    public bool RejectWhenNoBlockApplied { get; set; } = true;

    /// <summary>Gets or sets the maximum number of edit blocks accepted from one response.</summary>
    public int MaxBlocks { get; set; } = 64;

    /// <summary>Gets or sets the maximum number of characters of search text echoed into a failure message.</summary>
    public int MaxFailureExcerptLength { get; set; } = 240;

    /// <summary>Creates an independent copy so a running component is unaffected by later mutation.</summary>
    /// <returns>A new options instance carrying the same values.</returns>
    public ProgramDiffOptions Clone() => new()
    {
        SearchMarker = SearchMarker,
        DividerMarker = DividerMarker,
        ReplaceMarker = ReplaceMarker,
        AllowCarriageReturns = AllowCarriageReturns,
        FuzzyWhitespace = FuzzyWhitespace,
        RejectWhenNoBlockApplied = RejectWhenNoBlockApplied,
        MaxBlocks = MaxBlocks,
        MaxFailureExcerptLength = MaxFailureExcerptLength
    };

    /// <summary>Validates the marker text and numeric bounds.</summary>
    /// <exception cref="ArgumentException">A marker is <c>null</c>, empty, white space, or duplicated.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A numeric bound is not positive.</exception>
    public void Validate()
    {
        ValidateMarker(SearchMarker, nameof(SearchMarker));
        ValidateMarker(DividerMarker, nameof(DividerMarker));
        ValidateMarker(ReplaceMarker, nameof(ReplaceMarker));
        if (string.Equals(SearchMarker, DividerMarker, StringComparison.Ordinal)
            || string.Equals(SearchMarker, ReplaceMarker, StringComparison.Ordinal)
            || string.Equals(DividerMarker, ReplaceMarker, StringComparison.Ordinal))
        {
            throw new ArgumentException("The search, divider, and replace markers must all differ.", nameof(SearchMarker));
        }

        if (MaxBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(MaxBlocks), MaxBlocks, "Value must be positive.");
        if (MaxFailureExcerptLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxFailureExcerptLength), MaxFailureExcerptLength,
                "Value must be positive.");
    }

    private static void ValidateMarker(string marker, string parameterName)
    {
        if (marker is null) throw new ArgumentException("Markers cannot be null.", parameterName);
        if (string.IsNullOrWhiteSpace(marker))
            throw new ArgumentException("Markers cannot be empty or white space.", parameterName);
        if (marker.IndexOf('\n') >= 0 || marker.IndexOf('\r') >= 0)
            throw new ArgumentException("Markers cannot contain line breaks.", parameterName);
    }
}
