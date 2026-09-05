namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>The shared lexical tokenizer the structural novelty metrics agree on.</summary>
/// <remarks>
/// Identifier and number runs stay whole; every other non-white-space character becomes its own token. The rule is
/// language agnostic on purpose — a novelty gate has to work on whatever language a run is evolving, and a real
/// parser per language would be both heavier and no more discriminating for this use.
/// </remarks>
internal static class ProgramTokenizer
{
    internal static HashSet<string> Tokenize(string source)
    {
        var tokens = new HashSet<string>(StringComparer.Ordinal);
        int index = 0;
        while (index < source.Length)
        {
            char character = source[index];
            if (char.IsWhiteSpace(character))
            {
                index++;
                continue;
            }

            if (IsWordCharacter(character))
            {
                int start = index;
                while (index < source.Length && IsWordCharacter(source[index])) index++;
                tokens.Add(source.Substring(start, index - start));
                continue;
            }

            tokens.Add(character.ToString());
            index++;
        }

        return tokens;
    }

    private static bool IsWordCharacter(char character) =>
        char.IsLetterOrDigit(character) || character == '_';
}
