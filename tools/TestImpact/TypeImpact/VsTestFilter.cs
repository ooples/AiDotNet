namespace AiDotNet.TestImpact.TypeImpact;

/// <summary>Kleene three-valued truth: a filter over metadata can be undecidable for one test.</summary>
internal enum Tri { False, Unknown, True }

/// <summary>
/// The VSTest <c>--filter</c> language as the shard manifest uses it: conditions on
/// FullyQualifiedName and Category with <c>=</c>, <c>!=</c>, <c>~</c>, <c>!~</c>, combined by
/// <c>&amp;</c>, <c>|</c> and parentheses. Evaluation is three-valued so that a trait the metadata
/// cannot pin down makes ownership possible rather than excluded: a shard that might own a
/// selected test must run it.
/// </summary>
internal abstract class VsTestFilter
{
    public abstract Tri Evaluate(TestMethod test);

    public static VsTestFilter Parse(string text)
    {
        var parser = new Parser(text);
        var result = parser.ParseOr();
        parser.SkipSpace();
        if (!parser.AtEnd)
        {
            throw new FormatException($"unexpected '{text[parser.Position]}' at {parser.Position} in filter: {text}");
        }

        return result;
    }

    /// <summary>Escapes a value so the filter language reads it literally (a class name with generic or nested punctuation).</summary>
    public static string Escape(string value)
    {
        var builder = new System.Text.StringBuilder(value.Length);
        foreach (char c in value)
        {
            if (c is '(' or ')' or '&' or '|' or '=' or '!' or '~' or '\\')
            {
                builder.Append('\\');
            }

            builder.Append(c);
        }

        return builder.ToString();
    }

    private static Tri And(Tri a, Tri b) => (Tri)Math.Min((int)a, (int)b);
    private static Tri Or(Tri a, Tri b) => (Tri)Math.Max((int)a, (int)b);
    private static Tri Not(Tri a) => (Tri)(2 - (int)a);

    private sealed class AndNode(VsTestFilter left, VsTestFilter right) : VsTestFilter
    {
        public override Tri Evaluate(TestMethod test)
        {
            var l = left.Evaluate(test);
            return l == Tri.False ? Tri.False : And(l, right.Evaluate(test));
        }
    }

    private sealed class OrNode(VsTestFilter left, VsTestFilter right) : VsTestFilter
    {
        public override Tri Evaluate(TestMethod test)
        {
            var l = left.Evaluate(test);
            return l == Tri.True ? Tri.True : Or(l, right.Evaluate(test));
        }
    }

    private sealed class Condition(string property, string op, string value) : VsTestFilter
    {
        public override Tri Evaluate(TestMethod test)
        {
            bool negate = op.StartsWith('!');
            bool contains = op.EndsWith('~');
            Tri positive;
            switch (property)
            {
                case "FullyQualifiedName":
                    bool match = contains
                        ? test.FullyQualifiedName.Contains(value, StringComparison.OrdinalIgnoreCase)
                        : string.Equals(test.FullyQualifiedName, value, StringComparison.OrdinalIgnoreCase);
                    positive = match ? Tri.True : Tri.False;
                    break;
                case "Category":
                    bool Has(IReadOnlySet<string> set) => contains
                        ? set.Any(c => c.Contains(value, StringComparison.OrdinalIgnoreCase))
                        : set.Any(c => string.Equals(c, value, StringComparison.OrdinalIgnoreCase));
                    positive = Has(test.CertainCategories) ? Tri.True
                        : Has(test.PossibleCategories) ? Tri.Unknown
                        : Tri.False;
                    break;
                default:
                    positive = Tri.Unknown;
                    break;
            }

            return negate ? Not(positive) : positive;
        }
    }

    private sealed class Parser(string text)
    {
        public int Position;
        public bool AtEnd => Position >= text.Length;

        public void SkipSpace()
        {
            while (!AtEnd && char.IsWhiteSpace(text[Position]))
            {
                Position++;
            }
        }

        public VsTestFilter ParseOr()
        {
            var left = ParseAnd();
            while (true)
            {
                SkipSpace();
                if (AtEnd || text[Position] != '|')
                {
                    return left;
                }

                Position++;
                left = new OrNode(left, ParseAnd());
            }
        }

        private VsTestFilter ParseAnd()
        {
            var left = ParseFactor();
            while (true)
            {
                SkipSpace();
                if (AtEnd || text[Position] != '&')
                {
                    return left;
                }

                Position++;
                left = new AndNode(left, ParseFactor());
            }
        }

        private VsTestFilter ParseFactor()
        {
            SkipSpace();
            if (!AtEnd && text[Position] == '(')
            {
                Position++;
                var inner = ParseOr();
                SkipSpace();
                if (AtEnd || text[Position] != ')')
                {
                    throw new FormatException($"missing ')' in filter: {text}");
                }

                Position++;
                return inner;
            }

            int start = Position;
            while (!AtEnd && text[Position] is not ('=' or '!' or '~'))
            {
                Position++;
            }

            var property = text[start..Position].Trim();
            string op;
            if (text.AsSpan(Position).StartsWith("!=")) { op = "!="; }
            else if (text.AsSpan(Position).StartsWith("!~")) { op = "!~"; }
            else if (!AtEnd && text[Position] == '=') { op = "="; }
            else if (!AtEnd && text[Position] == '~') { op = "~"; }
            else { throw new FormatException($"expected an operator at {Position} in filter: {text}"); }

            Position += op.Length;
            var value = new System.Text.StringBuilder();
            while (!AtEnd && text[Position] is not ('&' or '|' or ')'))
            {
                if (text[Position] == '\\' && Position + 1 < text.Length)
                {
                    Position++;
                }

                value.Append(text[Position]);
                Position++;
            }

            if (property.Length == 0)
            {
                throw new FormatException($"missing property name at {start} in filter: {text}");
            }

            return new Condition(property, op, value.ToString().Trim());
        }
    }
}
