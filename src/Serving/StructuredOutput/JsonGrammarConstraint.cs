using AiDotNet.Helpers;

namespace AiDotNet.Serving.StructuredOutput;

/// <summary>
/// An <see cref="ITokenConstraint"/> that restricts output to any well-formed JSON value of arbitrary
/// nesting depth. JSON is context-free (unbounded nesting), so it is enforced with a pushdown automaton
/// rather than a regular expression — this is the unbounded counterpart of
/// <see cref="JsonSchemaConstraint.AnyJsonObject"/> (which bounds depth with a finite regex).
/// </summary>
/// <remarks>
/// <para>
/// The automaton validates JSON character-by-character with a stack of open containers. Per decode step it
/// asks, for each vocabulary token, whether appending that token keeps the output a valid JSON prefix that
/// can still complete; the end-of-sequence token is permitted once a complete top-level value has been read.
/// Tokenizer-agnostic (constructed with the vocabulary's token pieces), so it works on every backend.
/// </para>
/// <para><b>For Beginners:</b> this forces the model to emit valid JSON — every brace, quote, comma and
/// value in the right place — no matter how deeply nested, so the result always parses.</para>
/// </remarks>
public sealed class JsonGrammarConstraint : ITokenConstraint
{
    private readonly string[] _tokenText;
    private readonly int _eosTokenId;
    private readonly JsonPda _pda;
    private bool _finished;

    /// <summary>Creates a JSON-grammar constraint over the given vocabulary.</summary>
    /// <param name="tokenText">The vocabulary: token id -&gt; decoded string piece.</param>
    /// <param name="eosTokenId">The end-of-sequence token id, permitted once a full JSON value is complete.</param>
    public JsonGrammarConstraint(IReadOnlyList<string> tokenText, int eosTokenId)
    {
        Guard.NotNull(tokenText);
        _tokenText = new string[tokenText.Count];
        for (int i = 0; i < tokenText.Count; i++)
        {
            _tokenText[i] = tokenText[i] ?? string.Empty;
        }
        _eosTokenId = eosTokenId;
        _pda = new JsonPda();
    }

    /// <inheritdoc/>
    public bool IsComplete => _finished || _pda.CanEnd;

    /// <inheritdoc/>
    // Terminal only when the automaton can accept no further non-EOS token. A complete top-level value is
    // still followed by valid insignificant whitespace, so at End the constraint is accepting (EOS permitted
    // in ApplyMask) but NOT terminal — the model ends generation by emitting EOS. The scan short-circuits on
    // the first acceptable token, which the common (non-dead) states hit almost immediately.
    public bool IsTerminal
    {
        get
        {
            if (_finished)
            {
                return true;
            }
            for (int i = 0; i < _tokenText.Length; i++)
            {
                if (_tokenText[i].Length > 0 && _pda.Accepts(_tokenText[i]))
                {
                    return false;
                }
            }
            return true;
        }
    }

    /// <inheritdoc/>
    public void ApplyMask(Span<float> logits)
    {
        if (_finished)
        {
            for (int i = 0; i < logits.Length; i++)
            {
                if (i != _eosTokenId) logits[i] = float.NegativeInfinity;
            }
            if (_eosTokenId >= 0 && _eosTokenId < logits.Length) logits[_eosTokenId] = 0f;
            return;
        }

        bool eosAllowed = _pda.CanEnd;
        bool anyAllowed = false;
        for (int i = 0; i < logits.Length; i++)
        {
            bool ok = (eosAllowed && i == _eosTokenId) || (_tokenText[i].Length > 0 && _pda.Accepts(_tokenText[i]));
            if (ok) anyAllowed = true;
            else logits[i] = float.NegativeInfinity;
        }

        // Fail closed: no token is permitted and the JSON value is not yet complete (EOS not allowed), so the
        // input can no longer become valid JSON. Rather than opening EOS to fake a successful stop (emitting
        // malformed JSON), signal the dead-end so the engine fails this sequence with an error. A well-formed
        // automaton reached through this mask never gets here.
        if (!anyAllowed)
        {
            throw new StructuredOutputConstraintException(
                "JSON-grammar constraint reached a dead-end: no token can continue a valid JSON value and " +
                "the output so far is not a complete JSON value.");
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
            _pda.Feed(_tokenText[tokenId]);
        }
    }
}

/// <summary>
/// A character-level pushdown automaton that recognizes well-formed JSON of arbitrary nesting depth.
/// <see cref="Feed(string)"/> advances the committed state; <see cref="Accepts(string)"/> tests whether a
/// string keeps the input a viable JSON prefix without committing.
/// </summary>
internal sealed class JsonPda
{
    private enum State
    {
        Start, End, Value,
        ObjStart,       // after '{': expect a key string or '}'
        ObjKeyExpected, // after ',' in an object: expect a key string
        ObjColon,       // after a key: expect ':'
        ObjComma,       // after a member value: expect ',' or '}'
        ArrStart,       // after '[': expect a value or ']'
        ArrComma,       // after an element: expect ',' or ']'
        Str, StrEsc, StrU,
        Key, KeyEsc, KeyU,
        Num, Kw
    }

    private enum NumSub { Sign, Zero, IntDigits, Dot, FracDigits, Exp, ExpSign, ExpDigits }

    // Container stack: true = object, false = array.
    private readonly List<bool> _stack = new();
    private State _state = State.Start;
    private int _uRemaining;
    private string _kw = string.Empty;
    private int _kwPos;
    private NumSub _num;

    public JsonPda() { }

    private JsonPda(JsonPda other)
    {
        _stack = new List<bool>(other._stack);
        _state = other._state;
        _uRemaining = other._uRemaining;
        _kw = other._kw;
        _kwPos = other._kwPos;
        _num = other._num;
    }

    /// <summary>Whether a complete top-level JSON value has been read (generation may stop).</summary>
    public bool CanEnd => _state == State.End || (_state == State.Num && _stack.Count == 0 && NumberCanEnd(_num));

    /// <summary>Advances the committed automaton over every character of <paramref name="text"/>.</summary>
    public void Feed(string text)
    {
        foreach (char c in text)
        {
            if (!Step(c)) return;
        }
    }

    /// <summary>Tests whether feeding <paramref name="text"/> keeps the input a viable JSON prefix.</summary>
    public bool Accepts(string text)
    {
        var copy = new JsonPda(this);
        foreach (char c in text)
        {
            if (!copy.Step(c)) return false;
        }
        return true;
    }

    private static bool IsWs(char c) => c is ' ' or '\t' or '\n' or '\r';
    private static bool IsDigit(char c) => c >= '0' && c <= '9';
    private static bool IsHex(char c) => IsDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    private static bool NumberCanEnd(NumSub n) => n is NumSub.Zero or NumSub.IntDigits or NumSub.FracDigits or NumSub.ExpDigits;

    // Processes one character; returns false if it is invalid in the current state. Numbers have no explicit
    // terminator, so a non-number character ends the number and is reprocessed in the after-value context.
    private bool Step(char c)
    {
        while (true)
        {
            if (_state == State.Num)
            {
                if (StepNumber(c, out bool consumed))
                {
                    if (consumed) return true;
                    continue; // number ended; reprocess c in the new state
                }
                return false;
            }
            return StepNonNumber(c);
        }
    }

    private bool StepNumber(char c, out bool consumed)
    {
        consumed = true;
        switch (_num)
        {
            case NumSub.Sign:
                if (c == '0') { _num = NumSub.Zero; return true; }
                if (c is >= '1' and <= '9') { _num = NumSub.IntDigits; return true; }
                consumed = false; return false;
            case NumSub.Zero:
                if (c == '.') { _num = NumSub.Dot; return true; }
                if (c is 'e' or 'E') { _num = NumSub.Exp; return true; }
                break;
            case NumSub.IntDigits:
                if (IsDigit(c)) return true;
                if (c == '.') { _num = NumSub.Dot; return true; }
                if (c is 'e' or 'E') { _num = NumSub.Exp; return true; }
                break;
            case NumSub.Dot:
                if (IsDigit(c)) { _num = NumSub.FracDigits; return true; }
                consumed = false; return false;
            case NumSub.FracDigits:
                if (IsDigit(c)) return true;
                if (c is 'e' or 'E') { _num = NumSub.Exp; return true; }
                break;
            case NumSub.Exp:
                if (c is '+' or '-') { _num = NumSub.ExpSign; return true; }
                if (IsDigit(c)) { _num = NumSub.ExpDigits; return true; }
                consumed = false; return false;
            case NumSub.ExpSign:
                if (IsDigit(c)) { _num = NumSub.ExpDigits; return true; }
                consumed = false; return false;
            case NumSub.ExpDigits:
                if (IsDigit(c)) return true;
                break;
        }

        if (!NumberCanEnd(_num)) { consumed = false; return false; }
        _state = AfterValueState();
        consumed = false;
        return true;
    }

    private bool StepNonNumber(char c)
    {
        switch (_state)
        {
            case State.Start:
                return IsWs(c) || BeginValue(c);

            case State.End:
                return IsWs(c);

            case State.Value:
                return IsWs(c) || BeginValue(c);

            case State.ObjStart:
                if (IsWs(c)) return true;
                if (c == '"') { _state = State.Key; return true; }
                if (c == '}') return CloseContainer(expectObject: true);
                return false;

            case State.ObjKeyExpected:
                if (IsWs(c)) return true;
                if (c == '"') { _state = State.Key; return true; }
                return false;

            case State.ObjColon:
                if (IsWs(c)) return true;
                if (c == ':') { _state = State.Value; return true; }
                return false;

            case State.ObjComma:
                if (IsWs(c)) return true;
                if (c == ',') { _state = State.ObjKeyExpected; return true; }
                if (c == '}') return CloseContainer(expectObject: true);
                return false;

            case State.ArrStart:
                if (IsWs(c)) return true;
                if (c == ']') return CloseContainer(expectObject: false);
                return BeginValue(c);

            case State.ArrComma:
                if (IsWs(c)) return true;
                if (c == ',') { _state = State.Value; return true; }
                if (c == ']') return CloseContainer(expectObject: false);
                return false;

            case State.Str:
                if (c == '\\') { _state = State.StrEsc; return true; }
                if (c == '"') { _state = AfterValueState(); return true; }
                return c >= 0x20;
            case State.StrEsc:
                return HandleEscape(c, valueString: true);
            case State.StrU:
                if (!IsHex(c)) return false;
                if (--_uRemaining == 0) _state = State.Str;
                return true;

            case State.Key:
                if (c == '\\') { _state = State.KeyEsc; return true; }
                if (c == '"') { _state = State.ObjColon; return true; }
                return c >= 0x20;
            case State.KeyEsc:
                return HandleEscape(c, valueString: false);
            case State.KeyU:
                if (!IsHex(c)) return false;
                if (--_uRemaining == 0) _state = State.Key;
                return true;

            case State.Kw:
                if (_kwPos < _kw.Length && c == _kw[_kwPos])
                {
                    _kwPos++;
                    if (_kwPos == _kw.Length) _state = AfterValueState();
                    return true;
                }
                return false;
        }
        return false;
    }

    private bool HandleEscape(char c, bool valueString)
    {
        switch (c)
        {
            case '"': case '\\': case '/': case 'b': case 'f':
            case 'n': case 'r': case 't':
                _state = valueString ? State.Str : State.Key;
                return true;
            case 'u':
                _uRemaining = 4;
                _state = valueString ? State.StrU : State.KeyU;
                return true;
            default:
                return false;
        }
    }

    private bool BeginValue(char c)
    {
        switch (c)
        {
            case '{': _stack.Add(true); _state = State.ObjStart; return true;
            case '[': _stack.Add(false); _state = State.ArrStart; return true;
            case '"': _state = State.Str; return true;
            case '-': _state = State.Num; _num = NumSub.Sign; return true;
            case '0': _state = State.Num; _num = NumSub.Zero; return true;
            case >= '1' and <= '9': _state = State.Num; _num = NumSub.IntDigits; return true;
            case 't': _state = State.Kw; _kw = "true"; _kwPos = 1; return true;
            case 'f': _state = State.Kw; _kw = "false"; _kwPos = 1; return true;
            case 'n': _state = State.Kw; _kw = "null"; _kwPos = 1; return true;
            default: return false;
        }
    }

    private State AfterValueState()
    {
        if (_stack.Count == 0) return State.End;
        return _stack[^1] ? State.ObjComma : State.ArrComma;
    }

    private bool CloseContainer(bool expectObject)
    {
        if (_stack.Count == 0 || _stack[^1] != expectObject) return false;
        _stack.RemoveAt(_stack.Count - 1);
        _state = AfterValueState();
        return true;
    }
}
