using System.Text;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Prompts;

/// <summary>One piece of prompt text with named <c>{placeholder}</c> slots, parsed once and rendered many times.</summary>
/// <remarks>
/// <para>
/// A template is immutable and its placeholders are discovered when it is constructed, not when it is rendered.
/// That single decision is what allows a whole template set to be checked at configure time: the set knows which
/// names each template asks for and can compare them against the names the prompt builder is able to supply,
/// long before any model is called. Rendering itself is a pure function of the text and the supplied values — no
/// clocks, no culture, no ambient state — so the same inputs always produce the same characters.
/// </para>
/// <para>
/// The placeholder syntax matches the one the reference OpenEvolve templates use, so existing template files
/// transfer unchanged: <c>{name}</c> marks a slot, and <c>{{</c> and <c>}}</c> escape literal braces (which is
/// how the JSON example inside the shipped evaluation template survives). Unlike upstream, an unbalanced or
/// malformed brace is rejected here rather than raising an error from deep inside a formatting call during a run.
/// </para>
/// <para>
/// Line endings are normalized to line feeds when the template is created. Without that, a template file checked
/// out with Windows line endings and the same file checked out with Unix ones would produce different prompt text
/// and different template hashes, which would break both cross-machine reproducibility and checkpoint resume for a
/// reason that has nothing to do with the wording.
/// </para>
/// <para><b>For Beginners:</b> This is a fill-in-the-blanks piece of text. The blanks are written as
/// <c>{name}</c>, and rendering means handing over a value for each blank and getting the finished text back. If
/// you need a real curly brace in the output — for example when the text shows a JSON example — write it twice,
/// as <c>{{</c> or <c>}}</c>. The list of blanks a template contains is worked out as soon as the template is
/// created, so mistakes surface while you are setting things up rather than in the middle of a long run.</para>
/// </remarks>
public sealed class ProgramPromptTemplate
{
    /// <summary>The largest template text accepted, in characters.</summary>
    public const int MaxTextLength = 262_144;

    private readonly IReadOnlyList<Segment> _segments;

    /// <summary>Initializes a template from its text, normalizing line endings and parsing its placeholders.</summary>
    /// <param name="text">The template text; may be empty but not <c>null</c>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="text"/> exceeds <see cref="MaxTextLength"/>, contains an unclosed <c>{</c>, an unescaped
    /// <c>}</c>, or a placeholder whose name is empty or contains a character other than a letter, a digit, or an
    /// underscore.
    /// </exception>
    public ProgramPromptTemplate(string text)
    {
        Guard.NotNull(text);
        if (text.Length > MaxTextLength)
        {
            throw new ArgumentException(
                $"Prompt template text cannot exceed {MaxTextLength} characters.", nameof(text));
        }

        string normalized = NormalizeLineEndings(text);
        Text = normalized;
        var segments = new List<Segment>();
        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        Parse(normalized, segments, names, seen);
        _segments = segments;
        Placeholders = names;
    }

    /// <summary>Gets the template text with every line ending normalized to a line feed.</summary>
    public string Text { get; }

    /// <summary>Gets the distinct placeholder names this template contains, in first-appearance order.</summary>
    public IReadOnlyList<string> Placeholders { get; }

    /// <summary>Gets whether this template contains no placeholders and renders to a constant string.</summary>
    public bool IsConstant => Placeholders.Count == 0;

    /// <summary>Reports whether this template contains a placeholder with the given name.</summary>
    /// <param name="name">The placeholder name to look for.</param>
    /// <returns><c>true</c> when the name appears as a <c>{placeholder}</c> in the text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <c>null</c>.</exception>
    public bool ContainsPlaceholder(string name)
    {
        Guard.NotNull(name);
        for (int index = 0; index < Placeholders.Count; index++)
        {
            if (string.Equals(Placeholders[index], name, StringComparison.Ordinal)) return true;
        }

        return false;
    }

    /// <summary>Renders the template by substituting a value for every placeholder.</summary>
    /// <param name="values">The value for each placeholder name; extra entries are ignored.</param>
    /// <returns>The rendered text with escaped braces reduced to single braces.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <c>null</c>.</exception>
    /// <exception cref="KeyNotFoundException">A placeholder in the template has no entry in <paramref name="values"/>.</exception>
    public string Render(IReadOnlyDictionary<string, string> values)
    {
        Guard.NotNull(values);
        var builder = new StringBuilder(Text.Length);
        foreach (Segment segment in _segments)
        {
            if (segment.IsPlaceholder)
            {
                if (!values.TryGetValue(segment.Value, out string? replacement))
                {
                    throw new KeyNotFoundException(
                        $"No value was supplied for the prompt placeholder '{segment.Value}'.");
                }

                builder.Append(replacement ?? string.Empty);
            }
            else
            {
                builder.Append(segment.Value);
            }
        }

        return builder.ToString();
    }

    /// <summary>Returns a short description that never echoes the template text.</summary>
    /// <returns>The character length and placeholder count.</returns>
    public override string ToString() =>
        $"ProgramPromptTemplate(length={Text.Length}, placeholders={Placeholders.Count})";

    private static string NormalizeLineEndings(string text)
    {
        if (text.IndexOf('\r') < 0) return text;

        var builder = new StringBuilder(text.Length);
        for (int index = 0; index < text.Length; index++)
        {
            char current = text[index];
            if (current != '\r')
            {
                builder.Append(current);
                continue;
            }

            builder.Append('\n');
            if (index + 1 < text.Length && text[index + 1] == '\n') index++;
        }

        return builder.ToString();
    }

    private static void Parse(string text, List<Segment> segments, List<string> names, HashSet<string> seen)
    {
        var literal = new StringBuilder();
        int index = 0;
        while (index < text.Length)
        {
            char current = text[index];
            if (current == '{')
            {
                if (index + 1 < text.Length && text[index + 1] == '{')
                {
                    literal.Append('{');
                    index += 2;
                    continue;
                }

                int close = text.IndexOf('}', index + 1);
                if (close < 0)
                {
                    throw new ArgumentException(
                        $"The prompt template has an unclosed '{{' at character {index}.", nameof(text));
                }

                string name = text.Substring(index + 1, close - index - 1);
                ValidateName(name, index);
                if (literal.Length > 0)
                {
                    segments.Add(Segment.Literal(literal.ToString()));
                    literal.Length = 0;
                }

                segments.Add(Segment.Placeholder(name));
                if (seen.Add(name)) names.Add(name);
                index = close + 1;
                continue;
            }

            if (current == '}')
            {
                if (index + 1 < text.Length && text[index + 1] == '}')
                {
                    literal.Append('}');
                    index += 2;
                    continue;
                }

                throw new ArgumentException(
                    $"The prompt template has an unescaped '}}' at character {index}; write '}}}}' for a literal brace.",
                    nameof(text));
            }

            literal.Append(current);
            index++;
        }

        if (literal.Length > 0) segments.Add(Segment.Literal(literal.ToString()));
    }

    private static void ValidateName(string name, int position)
    {
        if (name.Length == 0)
        {
            throw new ArgumentException(
                $"The prompt template has an empty placeholder '{{}}' at character {position}.", "text");
        }

        foreach (char character in name)
        {
            if (character == '_' || char.IsLetterOrDigit(character)) continue;
            throw new ArgumentException(
                $"The prompt placeholder '{name}' at character {position} may only contain letters, digits, and underscores.",
                "text");
        }
    }

    private readonly struct Segment
    {
        private Segment(string value, bool isPlaceholder)
        {
            Value = value;
            IsPlaceholder = isPlaceholder;
        }

        public string Value { get; }

        public bool IsPlaceholder { get; }

        public static Segment Literal(string value) => new(value, false);

        public static Segment Placeholder(string name) => new(name, true);
    }
}
