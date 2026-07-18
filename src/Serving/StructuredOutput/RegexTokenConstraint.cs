using System.Text;
using AiDotNet.Helpers;

namespace AiDotNet.Serving.StructuredOutput;

/// <summary>
/// An <see cref="ITokenConstraint"/> that restricts generated text to match a regular expression. The
/// regex is compiled to a character-level nondeterministic automaton (Thompson construction); the
/// constraint then runs a lazy on-the-fly DFA over that NFA, deciding at each step which vocabulary tokens
/// keep the output on a path that can still complete a full match.
/// </summary>
/// <remarks>
/// <para>
/// This is the standard "regex-guided decoding" approach (outlines / xgrammar style). JSON-schema and
/// grammar compilers lower their languages to a regex or directly to a DFA and reuse this masking logic.
/// The constraint is tokenizer-agnostic: it is constructed with the vocabulary's token→string pieces, so
/// it works with any tokenizer the serving layer registers and on every compute backend.
/// </para>
/// <para><b>Supported syntax:</b> literals, <c>.</c> (any char), character classes <c>[a-z0-9_]</c> with
/// ranges and negation <c>[^...]</c>, shorthands <c>\d \w \s \D \W \S</c>, escapes, quantifiers
/// <c>* + ?</c> and bounded <c>{n} {n,} {n,m}</c>, alternation <c>|</c>, and grouping <c>(...)</c>. The
/// whole pattern is matched (implicitly anchored at both ends).</para>
/// <para><b>For Beginners:</b> give it a pattern like <c>\d{4}-\d{2}-\d{2}</c> and it forces the model to
/// produce text shaped like a date. At each step it hides every token that would make a valid date
/// impossible, so whatever the model emits always fits the pattern.</para>
/// </remarks>
public sealed class RegexTokenConstraint : ITokenConstraint
{
    private readonly RegexNfa _nfa;
    private readonly string[] _tokenText;
    private readonly int _eosTokenId;

    // Lazy-DFA state = the epsilon-closed set of NFA states, represented canonically as a sorted int[].
    // Cached so identical states share their computed allowed-token mask across steps and sequences.
    private readonly Dictionary<StateKey, int[]> _allowedCache = new();
    private int[] _current;
    private bool _finished;

    /// <summary>
    /// Compiles <paramref name="pattern"/> into a token constraint over the given vocabulary.
    /// </summary>
    /// <param name="pattern">The regular expression the full output must match.</param>
    /// <param name="tokenText">The vocabulary: token id -&gt; its decoded string piece. Index is the token id.</param>
    /// <param name="eosTokenId">The end-of-sequence token id, permitted once a full match is reached.</param>
    public RegexTokenConstraint(string pattern, IReadOnlyList<string> tokenText, int eosTokenId)
    {
        Guard.NotNullOrWhiteSpace(pattern);
        Guard.NotNull(tokenText);

        _nfa = RegexNfa.Compile(pattern);
        _tokenText = new string[tokenText.Count];
        for (int i = 0; i < tokenText.Count; i++)
        {
            _tokenText[i] = tokenText[i] ?? string.Empty;
        }
        _eosTokenId = eosTokenId;
        _current = _nfa.StartClosure();
    }

    /// <inheritdoc/>
    public bool IsComplete => _finished || _nfa.IsAccepting(_current);

    /// <inheritdoc/>
    public void ApplyMask(Span<float> logits)
    {
        if (_finished)
        {
            MaskToEosOnly(logits);
            return;
        }

        int[] allowed = GetAllowedTokens(_current);
        bool eosAllowed = _nfa.IsAccepting(_current);

        // allowed is a sorted list of permitted token ids; walk it in lockstep with the vocab.
        int a = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            while (a < allowed.Length && allowed[a] < i) a++;
            bool permitted = (a < allowed.Length && allowed[a] == i) || (eosAllowed && i == _eosTokenId);
            if (!permitted)
            {
                logits[i] = float.NegativeInfinity;
            }
        }

        // Dead-end safety net: if nothing (not even EOS) is permitted, keep EOS open so decoding can stop
        // rather than sampling from an all -inf distribution. A well-formed pattern should not reach here.
        if (allowed.Length == 0 && !eosAllowed && _eosTokenId >= 0 && _eosTokenId < logits.Length)
        {
            logits[_eosTokenId] = 0f;
        }
    }

    /// <inheritdoc/>
    public void Accept(int tokenId)
    {
        if (_finished) return;
        if (tokenId == _eosTokenId)
        {
            _finished = true;
            return;
        }
        if (tokenId >= 0 && tokenId < _tokenText.Length)
        {
            var next = _nfa.Advance(_current, _tokenText[tokenId]);
            if (next.Length > 0)
            {
                _current = next;
            }
        }
    }

    private void MaskToEosOnly(Span<float> logits)
    {
        for (int i = 0; i < logits.Length; i++)
        {
            if (i != _eosTokenId) logits[i] = float.NegativeInfinity;
        }
        if (_eosTokenId >= 0 && _eosTokenId < logits.Length) logits[_eosTokenId] = 0f;
    }

    // Returns the sorted set of token ids that keep the pattern satisfiable from DFA state `state`.
    private int[] GetAllowedTokens(int[] state)
    {
        var key = new StateKey(state);
        if (_allowedCache.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var permitted = new List<int>();
        for (int t = 0; t < _tokenText.Length; t++)
        {
            string text = _tokenText[t];
            if (text.Length == 0)
            {
                continue; // empty pieces (e.g. special tokens) carry no characters to constrain on
            }
            if (_nfa.Advance(state, text).Length > 0)
            {
                permitted.Add(t);
            }
        }

        var arr = permitted.ToArray();
        _allowedCache[key] = arr;
        return arr;
    }

    // Value-equality key over the canonical (sorted) NFA-state set so identical DFA states share a cache slot.
    private readonly struct StateKey : IEquatable<StateKey>
    {
        private readonly int[] _states;
        private readonly int _hash;
        public StateKey(int[] states)
        {
            _states = states;
            int h = 17;
            foreach (int s in states) h = unchecked(h * 31 + s);
            _hash = h;
        }
        public bool Equals(StateKey other)
        {
            if (_states.Length != other._states.Length) return false;
            for (int i = 0; i < _states.Length; i++)
            {
                if (_states[i] != other._states[i]) return false;
            }
            return true;
        }
        public override bool Equals(object? obj) => obj is StateKey k && Equals(k);
        public override int GetHashCode() => _hash;
    }
}

/// <summary>
/// A Thompson-construction NFA over characters, with epsilon transitions and a lazy DFA simulation used by
/// <see cref="RegexTokenConstraint"/>. States are addressed by integer id; a "DFA state" is the sorted,
/// epsilon-closed set of NFA state ids reached so far.
/// </summary>
internal sealed class RegexNfa
{
    // Each NFA state has: an optional character matcher + single target (consuming transition), and a list
    // of epsilon targets. The unique accept state is _accept.
    private readonly List<CharMatcher?> _match = new();
    private readonly List<int> _matchTarget = new();
    private readonly List<List<int>> _epsilon = new();
    private int _start;
    private int _accept;

    private int NewState()
    {
        _match.Add(null);
        _matchTarget.Add(-1);
        _epsilon.Add(new List<int>());
        return _match.Count - 1;
    }

    /// <summary>Compiles a regex pattern into an NFA (implicitly anchored to match the whole string).</summary>
    public static RegexNfa Compile(string pattern)
    {
        var nfa = new RegexNfa();
        var parser = new Parser(pattern, nfa);
        var frag = parser.ParseAlternation();
        parser.ExpectEnd();
        nfa._start = frag.Start;
        nfa._accept = frag.Accept;
        return nfa;
    }

    /// <summary>The start DFA state (epsilon closure of the NFA start).</summary>
    public int[] StartClosure() => Closure(new[] { _start });

    /// <summary>True if the DFA state contains the accept state (a full match is reached here).</summary>
    public bool IsAccepting(int[] state)
    {
        // state is sorted; binary search for the accept id.
        int lo = 0, hi = state.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (state[mid] == _accept) return true;
            if (state[mid] < _accept) lo = mid + 1; else hi = mid - 1;
        }
        return false;
    }

    /// <summary>Advances a DFA state by consuming every character of <paramref name="text"/> in order.
    /// Returns the resulting DFA state, or an empty array if the string drives the automaton to a dead end.</summary>
    public int[] Advance(int[] state, string text)
    {
        var cur = state;
        foreach (char c in text)
        {
            cur = Step(cur, c);
            if (cur.Length == 0) return cur;
        }
        return cur;
    }

    private int[] Step(int[] state, char c)
    {
        var moved = new List<int>();
        foreach (int s in state)
        {
            var m = _match[s];
            if (m is not null && m.Matches(c))
            {
                moved.Add(_matchTarget[s]);
            }
        }
        return moved.Count == 0 ? System.Array.Empty<int>() : Closure(moved);
    }

    private int[] Closure(IEnumerable<int> states)
    {
        var seen = new HashSet<int>();
        var stack = new Stack<int>();
        foreach (int s in states)
        {
            if (seen.Add(s)) stack.Push(s);
        }
        while (stack.Count > 0)
        {
            int s = stack.Pop();
            foreach (int e in _epsilon[s])
            {
                if (seen.Add(e)) stack.Push(e);
            }
        }
        var arr = new int[seen.Count];
        seen.CopyTo(arr);
        System.Array.Sort(arr);
        return arr;
    }

    // A compiled NFA fragment: a start state and a single accept state.
    private readonly struct Fragment
    {
        public readonly int Start;
        public readonly int Accept;
        public Fragment(int start, int accept) { Start = start; Accept = accept; }
    }

    // Matches a single character: a literal, a range set, "any", or a negated set.
    private sealed class CharMatcher
    {
        private readonly bool _any;
        private readonly bool _negate;
        private readonly List<(char lo, char hi)> _ranges;
        private CharMatcher(bool any, bool negate, List<(char, char)> ranges)
        {
            _any = any; _negate = negate; _ranges = ranges;
        }
        public static CharMatcher Any() => new(true, false, new());
        public static CharMatcher Set(List<(char, char)> ranges, bool negate) => new(false, negate, ranges);
        public static CharMatcher Literal(char c) => new(false, false, new() { (c, c) });
        public bool Matches(char c)
        {
            if (_any) return true;
            bool inSet = false;
            foreach (var (lo, hi) in _ranges)
            {
                if (c >= lo && c <= hi) { inSet = true; break; }
            }
            return _negate ? !inSet : inSet;
        }
    }

    // Recursive-descent regex parser building NFA fragments directly (Thompson construction).
    private sealed class Parser
    {
        private readonly string _p;
        private int _i;
        private readonly RegexNfa _n;
        public Parser(string pattern, RegexNfa nfa) { _p = pattern; _n = nfa; }

        private bool More => _i < _p.Length;
        private char Peek => _p[_i];
        private char Next() => _p[_i++];

        public void ExpectEnd()
        {
            if (More)
            {
                throw new FormatException($"Unexpected '{Peek}' at position {_i} in regex '{_p}'.");
            }
        }

        // alternation := concat ('|' concat)*
        public Fragment ParseAlternation()
        {
            var left = ParseConcat();
            while (More && Peek == '|')
            {
                _i++;
                var right = ParseConcat();
                int s = _n.NewState();
                int a = _n.NewState();
                _n._epsilon[s].Add(left.Start);
                _n._epsilon[s].Add(right.Start);
                _n._epsilon[left.Accept].Add(a);
                _n._epsilon[right.Accept].Add(a);
                left = new Fragment(s, a);
            }
            return left;
        }

        // concat := repeat*
        private Fragment ParseConcat()
        {
            // Empty concatenation (e.g. "a|") = epsilon fragment.
            if (!More || Peek == '|' || Peek == ')')
            {
                int s = _n.NewState();
                return new Fragment(s, s);
            }
            var frag = ParseRepeat();
            while (More && Peek != '|' && Peek != ')')
            {
                var next = ParseRepeat();
                _n._epsilon[frag.Accept].Add(next.Start);
                frag = new Fragment(frag.Start, next.Accept);
            }
            return frag;
        }

        // repeat := atom ('*' | '+' | '?' | '{n}' | '{n,}' | '{n,m}')?
        private Fragment ParseRepeat()
        {
            var atom = ParseAtom();
            if (!More) return atom;
            char c = Peek;
            if (c == '*') { _i++; return Star(atom); }
            if (c == '+') { _i++; return Plus(atom); }
            if (c == '?') { _i++; return Optional(atom); }
            if (c == '{') return ParseBounded(atom);
            return atom;
        }

        private Fragment Star(Fragment f)
        {
            int s = _n.NewState();
            int a = _n.NewState();
            _n._epsilon[s].Add(f.Start);
            _n._epsilon[s].Add(a);
            _n._epsilon[f.Accept].Add(f.Start);
            _n._epsilon[f.Accept].Add(a);
            return new Fragment(s, a);
        }

        private Fragment Plus(Fragment f)
        {
            int a = _n.NewState();
            _n._epsilon[f.Accept].Add(f.Start);
            _n._epsilon[f.Accept].Add(a);
            return new Fragment(f.Start, a);
        }

        private Fragment Optional(Fragment f)
        {
            int s = _n.NewState();
            int a = _n.NewState();
            _n._epsilon[s].Add(f.Start);
            _n._epsilon[s].Add(a);
            _n._epsilon[f.Accept].Add(a);
            return new Fragment(s, a);
        }

        // Parses a single atom into a fresh fragment. Cloning for quantifiers is done by re-parsing spans,
        // but bounded repetition re-materializes the atom by remembering its source span.
        private int _atomStart;
        private int _atomEnd;
        private Fragment ParseAtom()
        {
            _atomStart = _i;
            Fragment frag;
            char c = Peek;
            if (c == '(')
            {
                _i++;
                // Support (?:...) non-capturing groups by skipping the "?:".
                if (More && Peek == '?' && _i + 1 < _p.Length && _p[_i + 1] == ':')
                {
                    _i += 2;
                }
                frag = ParseAlternation();
                Expect(')');
            }
            else if (c == '[')
            {
                frag = ParseClass();
            }
            else if (c == '.')
            {
                _i++;
                frag = Single(CharMatcher.Any());
            }
            else if (c == '\\')
            {
                frag = ParseEscape();
            }
            else if (c == '*' || c == '+' || c == '?' || c == '{' || c == '|' || c == ')')
            {
                throw new FormatException($"Unexpected quantifier/operator '{c}' at position {_i} in regex '{_p}'.");
            }
            else
            {
                _i++;
                frag = Single(CharMatcher.Literal(c));
            }
            _atomEnd = _i;
            return frag;
        }

        // {n}, {n,}, {n,m}
        private Fragment ParseBounded(Fragment first)
        {
            int save = _i;
            _i++; // consume '{'
            int min = ParseInt();
            int max = min;
            bool hasMax = true;
            if (More && Peek == ',')
            {
                _i++;
                if (More && Peek == '}') { hasMax = false; }
                else { max = ParseInt(); }
            }
            if (!More || Peek != '}')
            {
                // Not a valid quantifier — treat '{' literally by rewinding.
                _i = save;
                return first;
            }
            _i++; // consume '}'

            string atomSrc = _p.Substring(_atomStart, _atomEnd - _atomStart);

            // Build: `min` mandatory copies, then either unbounded star (no max) or (max-min) optional copies.
            Fragment result = CloneAtomSequence(atomSrc, min, out int lastAccept, first);
            if (!hasMax)
            {
                var starTail = Star(ReparseAtom(atomSrc));
                if (result.Start < 0)
                {
                    result = starTail;
                }
                else
                {
                    _n._epsilon[lastAccept].Add(starTail.Start);
                    result = new Fragment(result.Start, starTail.Accept);
                }
            }
            else
            {
                for (int k = min; k < max; k++)
                {
                    var opt = Optional(ReparseAtom(atomSrc));
                    if (result.Start < 0)
                    {
                        result = opt; lastAccept = opt.Accept;
                    }
                    else
                    {
                        _n._epsilon[lastAccept].Add(opt.Start);
                        lastAccept = opt.Accept;
                        result = new Fragment(result.Start, lastAccept);
                    }
                }
            }

            if (result.Start < 0)
            {
                // {0} with no max: matches empty.
                int s = _n.NewState();
                return new Fragment(s, s);
            }
            return result;
        }

        // Concatenates `count` freshly parsed copies of the atom source. Returns Fragment(-1,-1) if count==0.
        private Fragment CloneAtomSequence(string atomSrc, int count, out int lastAccept, Fragment firstAlready)
        {
            lastAccept = -1;
            if (count == 0) return new Fragment(-1, -1);

            // Reuse the already-parsed `firstAlready` as copy #1 to avoid double-emitting it.
            Fragment chain = firstAlready;
            lastAccept = firstAlready.Accept;
            for (int k = 1; k < count; k++)
            {
                var copy = ReparseAtom(atomSrc);
                _n._epsilon[lastAccept].Add(copy.Start);
                lastAccept = copy.Accept;
                chain = new Fragment(chain.Start, lastAccept);
            }
            return chain;
        }

        private Fragment ReparseAtom(string atomSrc)
        {
            var sub = new Parser(atomSrc, _n);
            var f = sub.ParseAlternation();
            sub.ExpectEnd();
            return f;
        }

        private Fragment ParseEscape()
        {
            _i++; // consume '\'
            if (!More) throw new FormatException($"Dangling escape at end of regex '{_p}'.");
            char e = Next();
            switch (e)
            {
                case 'd': return Single(CharMatcher.Set(new() { ('0', '9') }, false));
                case 'D': return Single(CharMatcher.Set(new() { ('0', '9') }, true));
                case 'w': return Single(CharMatcher.Set(WordRanges(), false));
                case 'W': return Single(CharMatcher.Set(WordRanges(), true));
                case 's': return Single(CharMatcher.Set(SpaceRanges(), false));
                case 'S': return Single(CharMatcher.Set(SpaceRanges(), true));
                case 'n': return Single(CharMatcher.Literal('\n'));
                case 't': return Single(CharMatcher.Literal('\t'));
                case 'r': return Single(CharMatcher.Literal('\r'));
                default: return Single(CharMatcher.Literal(e)); // escaped metachar or literal
            }
        }

        private Fragment ParseClass()
        {
            _i++; // consume '['
            bool negate = false;
            if (More && Peek == '^') { negate = true; _i++; }
            var ranges = new List<(char, char)>();
            while (More && Peek != ']')
            {
                char lo = ReadClassChar(ranges);
                if (lo == '\0' && ranges.Count > 0 && _lastWasShorthand)
                {
                    // shorthand (\d etc.) already appended its ranges
                    continue;
                }
                if (More && Peek == '-' && _i + 1 < _p.Length && _p[_i + 1] != ']')
                {
                    _i++; // consume '-'
                    char hi = ReadClassChar(ranges);
                    ranges.Add((lo, hi));
                }
                else
                {
                    ranges.Add((lo, lo));
                }
            }
            Expect(']');
            return Single(CharMatcher.Set(ranges, negate));
        }

        private bool _lastWasShorthand;
        private char ReadClassChar(List<(char, char)> ranges)
        {
            _lastWasShorthand = false;
            char c = Next();
            if (c == '\\' && More)
            {
                char e = Next();
                switch (e)
                {
                    case 'd': ranges.Add(('0', '9')); _lastWasShorthand = true; return '\0';
                    case 'w': ranges.AddRange(WordRanges()); _lastWasShorthand = true; return '\0';
                    case 's': ranges.AddRange(SpaceRanges()); _lastWasShorthand = true; return '\0';
                    case 'n': return '\n';
                    case 't': return '\t';
                    case 'r': return '\r';
                    default: return e;
                }
            }
            return c;
        }

        private static List<(char, char)> WordRanges() =>
            new() { ('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_') };
        private static List<(char, char)> SpaceRanges() =>
            new() { (' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r'), ('\f', '\f'), ('\v', '\v') };

        private Fragment Single(CharMatcher matcher)
        {
            int s = _n.NewState();
            int a = _n.NewState();
            _n._match[s] = matcher;
            _n._matchTarget[s] = a;
            return new Fragment(s, a);
        }

        private void Expect(char c)
        {
            if (!More || Next() != c)
            {
                throw new FormatException($"Expected '{c}' at position {_i} in regex '{_p}'.");
            }
        }

        // Upper bound on a single quantifier's repetition count. `response_format.regex` is request-controlled,
        // so an unbounded bound (e.g. a{2147483647}) would materialize billions of NFA fragments and exhaust
        // memory/CPU before generation even starts. Bounds larger than this — or numerically too large to be an
        // int — are rejected with ArgumentException, which the serving factory maps to an HTTP 400.
        private const int MaxQuantifierRepetitions = 1000;

        private int ParseInt()
        {
            var sb = new StringBuilder();
            while (More && char.IsDigit(Peek)) sb.Append(Next());
            if (sb.Length == 0) throw new FormatException($"Expected a number at position {_i} in regex '{_p}'.");
            if (!int.TryParse(sb.ToString(), out int value) || value > MaxQuantifierRepetitions)
            {
                throw new ArgumentException(
                    $"Quantifier repetition '{sb}' exceeds the maximum allowed count of {MaxQuantifierRepetitions} in regex '{_p}'.");
            }
            return value;
        }
    }
}
