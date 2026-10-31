using System.IO.Compression;
using System.Text;

namespace AiDotNet.Tokenization.Algorithms;

/// <summary>
/// A pretrained SentencePiece model read from its <c>.model</c> file (Kudo and Richardson 2018): the normalizer of the
/// model's precompiled character map, then the unigram model's Viterbi segmentation (Kudo 2018) or the BPE model's
/// score-ordered merges (Sennrich et al. 2016) with byte fallback, as the reference library computes them, so a
/// pretrained model sees exactly the ids it was trained on.
/// </summary>
/// <remarks>
/// <para>
/// Normalization follows <c>normalizer.cc</c>: the longest rule of the precompiled double-array trie (darts-clone)
/// rewrites each prefix, leading and repeated whitespace is dropped, a dummy prefix "▁" is added and spaces become
/// "▁". Segmentation follows <c>unigram_model.cc</c> <c>EncodeOptimized</c>: the best-scoring path, with a later
/// candidate replacing an earlier one only when strictly better, an unknown node of one character (score: lowest piece
/// score − 10) wherever no single-character piece matches, and runs of unknown pieces merged into one.
/// </para>
/// <para>
/// BPE follows <c>bpe_model.cc</c> <c>Encode</c>: the text split into characters, adjacent pairs whose concatenation is a
/// piece merged highest score first (the leftmost on ties), unused pieces split back into the pair that formed them, and,
/// with byte fallback, every unknown character emitted as its UTF-8 bytes' <c>&lt;0xXX&gt;</c> pieces
/// (<c>sentencepiece_processor.cc</c>).
/// </para>
/// <para>A character model follows <c>char_model.cc</c>: each Unicode character of the normalized text is a piece (or
/// the longest user-defined symbol starting there), unknown ones merged into runs as above.</para>
/// <para>User-defined symbols of a unigram model pass through the normalizer unchanged and score
/// <c>bytes × max score − 0.1</c> in the segmentation, as the reference treats them. BPE models with user-defined symbols,
/// and unigram models with byte fallback, are refused rather than segmented differently from the reference.</para>
/// </remarks>
public sealed class SentencePieceModel
{
    private const string Space = "▁";
    private const double UnknownPenalty = 10.0;

    private readonly string[] _pieces;
    private readonly float[] _scores;
    private readonly int[] _types;
    private readonly Dictionary<string, int> _ids = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _bpePieces = new(StringComparer.Ordinal);
    private readonly PieceTrie _trie = new();
    private readonly uint[] _charsmapTrie;
    private readonly byte[] _charsmapStrings;
    private readonly bool _addDummyPrefix;
    private readonly bool _removeExtraWhitespaces;
    private readonly bool _escapeWhitespaces;
    private readonly double _unknownScore;
    private readonly bool _bpe;
    private readonly bool _character;
    private readonly bool _byteFallback;
    private readonly Dictionary<byte, List<byte[]>> _userDefined = new();
    private readonly float _maxScore;

    private const int Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, ByteType = 6;

    /// <summary>Reads a SentencePiece <c>.model</c> (a serialized <c>ModelProto</c>).</summary>
    public SentencePieceModel(byte[] model)
    {
        Guard.NotNull(model);
        var pieces = new List<(string Piece, float Score, int Type)>();
        byte[]? charsmap = null;
        bool addDummyPrefix = true, removeExtraWhitespaces = true, escapeWhitespaces = true;
        int modelType = 1;
        bool byteFallback = false;
        int unkId = 0;
        var reader = new ProtoReader(model, 0, model.Length);
        while (reader.Next(out int field, out int wire))
        {
            if (field == 1 && wire == 2)
            {
                var piece = reader.Message();
                string text = "";
                float score = 0;
                int type = Normal;
                while (piece.Next(out int f, out int w))
                {
                    if (f == 1 && w == 2) text = piece.String();
                    else if (f == 2 && w == 5) score = piece.Fixed32Float();
                    else if (f == 3 && w == 0) type = (int)piece.Varint();
                    else piece.Skip(w);
                }
                pieces.Add((text, score, type));
            }
            else if (field == 2 && wire == 2)
            {
                var trainer = reader.Message();
                while (trainer.Next(out int f, out int w))
                {
                    if (f == 3 && w == 0) modelType = (int)trainer.Varint();
                    else if (f == 35 && w == 0) byteFallback = trainer.Varint() != 0;
                    else if (f == 40 && w == 0) unkId = (int)trainer.Varint();
                    else if (f == 41 && w == 0) BosId = (int)trainer.Varint();
                    else if (f == 42 && w == 0) EosId = (int)trainer.Varint();
                    else if (f == 43 && w == 0) PadId = (int)trainer.Varint();
                    else trainer.Skip(w);
                }
            }
            else if (field == 3 && wire == 2)
            {
                var normalizer = reader.Message();
                while (normalizer.Next(out int f, out int w))
                {
                    if (f == 2 && w == 2) charsmap = normalizer.Bytes();
                    else if (f == 3 && w == 0) addDummyPrefix = normalizer.Varint() != 0;
                    else if (f == 4 && w == 0) removeExtraWhitespaces = normalizer.Varint() != 0;
                    else if (f == 5 && w == 0) escapeWhitespaces = normalizer.Varint() != 0;
                    else normalizer.Skip(w);
                }
            }
            else reader.Skip(wire);
        }
        if (modelType is not (1 or 2 or 4))
            throw new NotSupportedException($"Only unigram, BPE and character SentencePiece models are supported (model_type {modelType}).");
        if (modelType == 4 && (byteFallback || pieces.Any(p => p.Type == ByteType)))
            throw new NotSupportedException("Character SentencePiece models with byte fallback are not supported.");
        _bpe = modelType == 2;
        _character = modelType == 4;
        _byteFallback = byteFallback;

        _pieces = pieces.Select(p => p.Piece).ToArray();
        _scores = pieces.Select(p => p.Score).ToArray();
        _types = pieces.Select(p => p.Type).ToArray();
        UnknownId = unkId;
        double minScore = double.MaxValue;
        float maxScore = float.MinValue;
        for (int i = 0; i < _pieces.Length; i++)
        {
            if (!_ids.ContainsKey(_pieces[i])) _ids[_pieces[i]] = i;
            if (_types[i] == Normal)
            {
                minScore = Math.Min(minScore, _scores[i]);
                maxScore = Math.Max(maxScore, _scores[i]);
            }
            if (_types[i] is Normal or UserDefined or Unused) _trie.Add(_pieces[i], i);
            if (_types[i] == UserDefined)
            {
                var bytes = Encoding.UTF8.GetBytes(_pieces[i]);
                if (bytes.Length == 0) continue;
                if (!_userDefined.TryGetValue(bytes[0], out var list)) _userDefined[bytes[0]] = list = new List<byte[]>();
                list.Add(bytes);
            }
        }
        _unknownScore = minScore - UnknownPenalty;
        _maxScore = maxScore;
        for (int i = 0; i < _pieces.Length; i++)
            if (_types[i] is Normal or UserDefined or Unused && !_bpePieces.ContainsKey(_pieces[i])) _bpePieces[_pieces[i]] = i;
        _addDummyPrefix = addDummyPrefix;
        _removeExtraWhitespaces = removeExtraWhitespaces;
        _escapeWhitespaces = escapeWhitespaces;
        if (charsmap is { Length: > 0 })
        {
            int trieBytes = BitConverter.ToInt32(charsmap, 0);
            _charsmapTrie = new uint[trieBytes / 4];
            for (int i = 0; i < _charsmapTrie.Length; i++) _charsmapTrie[i] = BitConverter.ToUInt32(charsmap, 4 + 4 * i);
            _charsmapStrings = charsmap.Skip(4 + trieBytes).ToArray();
        }
        else
        {
            _charsmapTrie = Array.Empty<uint>();
            _charsmapStrings = Array.Empty<byte>();
        }
    }

    /// <summary>Reads a gzip-compressed <c>.model</c>.</summary>
    public static SentencePieceModel FromGzip(Stream compressed)
    {
        using var gzip = new GZipStream(compressed, CompressionMode.Decompress);
        using var buffer = new MemoryStream();
        gzip.CopyTo(buffer);
        return new SentencePieceModel(buffer.ToArray());
    }

    /// <summary>The number of pieces.</summary>
    public int Count => _pieces.Length;

    /// <summary>The id of the unknown piece.</summary>
    public int UnknownId { get; }

    /// <summary>The id of the beginning-of-sentence piece, or −1.</summary>
    public int BosId { get; private set; } = 1;

    /// <summary>The id of the end-of-sentence piece, or −1.</summary>
    public int EosId { get; private set; } = 2;

    /// <summary>The id of the padding piece, or −1.</summary>
    public int PadId { get; private set; } = -1;

    /// <summary>The piece with id <paramref name="id"/>.</summary>
    public string IdToPiece(int id) => _pieces[id];

    /// <summary>Whether a piece is a control symbol (&lt;s&gt;, &lt;/s&gt;, …) or a user-defined one, both of which a
    /// Hugging Face tokenizer converted from this model matches as added tokens.</summary>
    public bool IsControl(int id) => _types[id] == Control;

    /// <inheritdoc cref="IsControl"/>
    public bool IsUserDefined(int id) => _types[id] == UserDefined;

    /// <summary>The id of <paramref name="piece"/>, or the unknown id.</summary>
    public int PieceToId(string piece) => _ids.TryGetValue(piece, out int id) ? id : UnknownId;

    /// <summary>The piece ids of <paramref name="text"/> (no beginning- or end-of-sentence ids).</summary>
    public int[] Encode(string text) => EncodePieces(text).Select(p => p.Id).ToArray();

    /// <summary>Whether this is a BPE model (otherwise unigram).</summary>
    public bool IsBpe => _bpe;

    /// <summary>The pieces and ids of <paramref name="text"/>.</summary>
    public List<(string Piece, int Id)> EncodePieces(string text)
    {
        Guard.NotNull(text);
        return EncodeNormalized(Normalize(text));
    }

    /// <summary>The pieces and ids of text already in the model's piece alphabet (spaces as "▁"), with no normalization:
    /// what a Hugging Face fast tokenizer's model step reads after its own normalizer and pre-tokenizer.</summary>
    public List<(string Piece, int Id)> EncodeNormalized(string normalized)
    {
        Guard.NotNull(normalized);
        if (_bpe) return EncodeBpe(normalized);
        if (_character) return MergeUnknown(EncodeCharacters(normalized));
        // Positions are UTF-16 indices of Unicode scalar starts (a surrogate pair is one character).
        var starts = new List<int>();
        for (int i = 0; i < normalized.Length; i++)
            if (!char.IsLowSurrogate(normalized[i])) starts.Add(i);
        int n = starts.Count;
        starts.Add(normalized.Length);
        var bestScore = new double[n + 1];
        var bestStart = new int[n + 1];
        var bestId = new int[n + 1];
        for (int i = 1; i <= n; i++) bestStart[i] = -1;
        var charIndexOfOffset = new Dictionary<int, int>();
        for (int i = 0; i <= n; i++) charIndexOfOffset[starts[i]] = i;
        for (int s = 0; s < n; s++)
        {
            if (s > 0 && bestStart[s] < 0) continue;
            bool singleCharacter = false;
            foreach (var (endOffset, id) in _trie.Prefixes(normalized, starts[s]))
            {
                if (_types[id] == Unused) continue;
                int e = charIndexOfOffset[endOffset];
                if (e == s + 1) singleCharacter = true;
                // unigram_model.cc: a user-defined symbol scores (its byte length × the highest piece score) − 0.1, so
                // it wins over any segmentation of the same span.
                double score = _types[id] == UserDefined
                    ? (float)((float)(Encoding.UTF8.GetByteCount(normalized.Substring(starts[s], endOffset - starts[s])) * _maxScore) - 0.1)
                    : _scores[id];
                double candidate = score + bestScore[s];
                if (bestStart[e] < 0 || candidate > bestScore[e])
                {
                    bestScore[e] = candidate;
                    bestStart[e] = s;
                    bestId[e] = id;
                }
            }
            if (!singleCharacter)
            {
                double candidate = _unknownScore + bestScore[s];
                if (bestStart[s + 1] < 0 || candidate > bestScore[s + 1])
                {
                    bestScore[s + 1] = candidate;
                    bestStart[s + 1] = s;
                    bestId[s + 1] = UnknownId;
                }
            }
        }
        var result = new List<(string Piece, int Id)>();
        for (int e = n; e > 0; e = bestStart[e])
            result.Add((normalized.Substring(starts[bestStart[e]], starts[e] - starts[bestStart[e]]), bestId[e]));
        result.Reverse();
        return ByteFallback(MergeUnknown(result));
    }

    // SentencePieceProcessor's byte fallback: an unknown piece becomes the <0xXX> pieces of its UTF-8 bytes.
    private List<(string Piece, int Id)> ByteFallback(List<(string Piece, int Id)> pieces)
    {
        if (!_byteFallback || pieces.All(p => p.Id != UnknownId)) return pieces;
        var result = new List<(string Piece, int Id)>();
        foreach (var piece in pieces)
        {
            if (piece.Id != UnknownId) { result.Add(piece); continue; }
            foreach (byte b in Encoding.UTF8.GetBytes(piece.Piece))
            {
                string bytePiece = $"<0x{b:X2}>";
                result.Add((bytePiece, PieceToId(bytePiece)));
            }
        }
        return result;
    }

    // Runs of unknown pieces are merged into one.
    private List<(string Piece, int Id)> MergeUnknown(List<(string Piece, int Id)> result)
    {
        var merged = new List<(string Piece, int Id)>();
        foreach (var piece in result)
        {
            if (piece.Id == UnknownId && merged.Count > 0 && merged[^1].Id == UnknownId)
                merged[^1] = (merged[^1].Piece + piece.Piece, UnknownId);
            else merged.Add(piece);
        }
        return merged;
    }

    /// <summary>The character segmentation of normalized text (<c>char_model.cc</c>): the longest user-defined symbol at
    /// each position, otherwise one Unicode character, each piece's id or the unknown id.</summary>
    private List<(string Piece, int Id)> EncodeCharacters(string normalized)
    {
        var result = new List<(string Piece, int Id)>();
        for (int i = 0; i < normalized.Length;)
        {
            int length = char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length ? 2 : 1;
            foreach (var (endOffset, id) in _trie.Prefixes(normalized, i))
                if (_types[id] == UserDefined) length = Math.Max(length, endOffset - i);
            string piece = normalized.Substring(i, length);
            int pieceId = _ids.TryGetValue(piece, out int found) && _types[found] != Unused ? found : UnknownId;
            result.Add((piece, pieceId));
            i += length;
        }
        return result;
    }

    /// <summary>The BPE segmentation of normalized text (<c>bpe_model.cc</c>), with byte fallback for unknown characters.</summary>
    private List<(string Piece, int Id)> EncodeBpe(string normalized)
    {
        // Symbols are the Unicode characters, linked so merged ones drop out; a user-defined piece at a position (the
        // longest) is one frozen symbol that never merges (bpe_model.cc's PrefixMatcher and Symbol::freeze).
        var symbols = new List<string>();
        var frozen = new List<bool>();
        for (int i = 0; i < normalized.Length; i++)
        {
            int length = char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length ? 2 : 1;
            bool userDefined = false;
            foreach (var (endOffset, id) in _trie.Prefixes(normalized, i))
                if (_types[id] == UserDefined && endOffset - i >= length) { length = endOffset - i; userDefined = true; }
            symbols.Add(normalized.Substring(i, length));
            frozen.Add(userDefined);
            i += length - 1;
        }
        int n = symbols.Count;
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }
        var reverseMerges = new Dictionary<string, (string Left, string Right)>(StringComparer.Ordinal);
        var agenda = new List<(int Left, int Right, float Score, int Size)>();
        void Push((int Left, int Right, float Score, int Size) entry)
        {
            agenda.Add(entry);
            int i = agenda.Count - 1;
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (!Before(agenda[i], agenda[parent])) break;
                (agenda[i], agenda[parent]) = (agenda[parent], agenda[i]);
                i = parent;
            }
        }
        (int Left, int Right, float Score, int Size) Pop()
        {
            var top = agenda[0];
            agenda[0] = agenda[^1];
            agenda.RemoveAt(agenda.Count - 1);
            int i = 0;
            while (true)
            {
                int left = 2 * i + 1, right = left + 1, best = i;
                if (left < agenda.Count && Before(agenda[left], agenda[best])) best = left;
                if (right < agenda.Count && Before(agenda[right], agenda[best])) best = right;
                if (best == i) break;
                (agenda[i], agenda[best]) = (agenda[best], agenda[i]);
                i = best;
            }
            return top;
        }
        // The reference's comparator: the higher score first, then the leftmost pair.
        static bool Before((int Left, int Right, float Score, int Size) a, (int Left, int Right, float Score, int Size) b)
            => a.Score > b.Score || (a.Score == b.Score && a.Left < b.Left);
        void MaybeAdd(int left, int right)
        {
            if (left < 0 || right < 0 || frozen[left] || frozen[right]) return;
            string piece = symbols[left] + symbols[right];
            if (!_bpePieces.TryGetValue(piece, out int id)) return;
            Push((left, right, _scores[id], piece.Length));
            if (_types[id] == Unused) reverseMerges[piece] = (symbols[left], symbols[right]);
        }
        for (int i = 0; i + 1 < n; i++) MaybeAdd(i, i + 1);
        while (agenda.Count > 0)
        {
            var top = Pop();
            if (symbols[top.Left].Length == 0 || symbols[top.Right].Length == 0 || symbols[top.Left].Length + symbols[top.Right].Length != top.Size)
                continue;
            symbols[top.Left] += symbols[top.Right];
            symbols[top.Right] = "";
            next[top.Left] = next[top.Right];
            if (next[top.Right] >= 0) prev[next[top.Right]] = top.Left;
            MaybeAdd(prev[top.Left], top.Left);
            MaybeAdd(top.Left, next[top.Left]);
        }
        var result = new List<(string Piece, int Id)>();
        void Resegment(string w)
        {
            int id = PieceToId(w);
            if (id == UnknownId || _types[id] != Unused || !reverseMerges.TryGetValue(w, out var pair))
            {
                Emit(w, id);
                return;
            }
            Resegment(pair.Left);
            Resegment(pair.Right);
        }
        void Emit(string w, int id)
        {
            if (id != UnknownId || !_byteFallback)
            {
                result.Add((w, id));
                return;
            }
            foreach (byte b in Encoding.UTF8.GetBytes(w))
            {
                string bytePiece = $"<0x{b:X2}>";
                result.Add((bytePiece, PieceToId(bytePiece)));
            }
        }
        for (int i = n > 0 ? 0 : -1; i >= 0; i = next[i]) Resegment(symbols[i]);
        return result;
    }

    /// <summary>The text of piece ids (control pieces dropped, byte pieces joined into their UTF-8 text, "▁" read as a
    /// space, the dummy prefix removed).</summary>
    public string Decode(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        var bytes = new List<byte>();
        void Flush()
        {
            if (bytes.Count == 0) return;
            sb.Append(Encoding.UTF8.GetString(bytes.ToArray()));
            bytes.Clear();
        }
        foreach (int id in ids)
        {
            if (id < 0 || id >= _pieces.Length || _types[id] == Control) continue;
            if (_types[id] == ByteType)
            {
                bytes.Add(Convert.ToByte(_pieces[id].Substring(3, 2), 16));
                continue;
            }
            Flush();
            sb.Append(id == UnknownId && _types[id] == Unknown ? " ⁇ " : _pieces[id]);
        }
        Flush();
        var text = sb.ToString().Replace(Space, " ");
        return _addDummyPrefix && text.StartsWith(" ", StringComparison.Ordinal) ? text.Substring(1) : text;
    }

    /// <summary>The normalized text the segmentation runs on (normalizer.cc <c>Normalize</c>).</summary>
    public string Normalize(string text)
    {
        var input = Encoding.UTF8.GetBytes(text);
        var output = new List<byte>(input.Length + 8);
        var spaceBytes = Encoding.UTF8.GetBytes(Space);
        int position = 0;
        if (_removeExtraWhitespaces)
        {
            while (position < input.Length)
            {
                var (rewrite, consumed) = NormalizePrefix(input, position);
                if (!(rewrite.Length == 1 && rewrite[0] == (byte)' ')) break;
                position += consumed;
            }
        }
        if (position >= input.Length) return "";
        if (_addDummyPrefix) output.AddRange(_escapeWhitespaces ? spaceBytes : new[] { (byte)' ' });
        bool previousSpace = _removeExtraWhitespaces;
        while (position < input.Length)
        {
            var (rewrite, consumed) = NormalizePrefix(input, position);
            int start = 0;
            if (_removeExtraWhitespaces && previousSpace)
                while (start < rewrite.Length && rewrite[start] == (byte)' ') start++;
            if (start < rewrite.Length)
            {
                for (int i = start; i < rewrite.Length; i++)
                {
                    if (_escapeWhitespaces && rewrite[i] == (byte)' ') output.AddRange(spaceBytes);
                    else output.Add(rewrite[i]);
                }
                previousSpace = rewrite[rewrite.Length - 1] == (byte)' ';
            }
            position += consumed;
            if (!_removeExtraWhitespaces) previousSpace = false;
        }
        if (_removeExtraWhitespaces)
        {
            var trailing = _escapeWhitespaces ? spaceBytes : new[] { (byte)' ' };
            while (output.Count >= trailing.Length && output.Skip(output.Count - trailing.Length).SequenceEqual(trailing))
                output.RemoveRange(output.Count - trailing.Length, trailing.Length);
        }
        return Encoding.UTF8.GetString(output.ToArray());
    }

    // The longest charsmap rule at `position`, or the character itself (U+FFFD for malformed UTF-8).
    private (byte[] Rewrite, int Consumed) NormalizePrefix(byte[] input, int position)
    {
        // normalizer.cc: a user-defined symbol (the longest at this position) passes through unnormalized.
        int symbol = 0;
        foreach (var bytes in _userDefined.TryGetValue(input[position], out var candidates) ? candidates : new List<byte[]>())
        {
            if (bytes.Length <= symbol || position + bytes.Length > input.Length) continue;
            bool match = true;
            for (int i = 0; i < bytes.Length && match; i++) match = input[position + i] == bytes[i];
            if (match) symbol = bytes.Length;
        }
        if (symbol > 0)
        {
            var passed = new byte[symbol];
            Array.Copy(input, position, passed, 0, symbol);
            return (passed, symbol);
        }
        int longestLength = 0, longestValue = 0;
        if (_charsmapTrie.Length > 0)
        {
            uint node = 0;
            node ^= Offset(_charsmapTrie[0]);
            for (int i = position; i < input.Length; i++)
            {
                byte b = input[i];
                node ^= b;
                if (node >= _charsmapTrie.Length) break;
                uint unit = _charsmapTrie[node];
                if ((unit & ((1u << 31) | 0xFF)) != b) break;
                node ^= Offset(unit);
                if (((unit >> 8) & 1) != 0 && node < _charsmapTrie.Length)
                {
                    longestLength = i - position + 1;
                    longestValue = (int)(_charsmapTrie[node] & ((1u << 31) - 1));
                }
            }
        }
        if (longestLength > 0)
        {
            int end = longestValue;
            while (end < _charsmapStrings.Length && _charsmapStrings[end] != 0) end++;
            var rewrite = new byte[end - longestValue];
            Array.Copy(_charsmapStrings, longestValue, rewrite, 0, rewrite.Length);
            return (rewrite, longestLength);
        }
        int length = Utf8Length(input, position);
        if (length == 0) return (new byte[] { 0xEF, 0xBF, 0xBD }, 1);
        var character = new byte[length];
        Array.Copy(input, position, character, 0, length);
        return (character, length);
    }

    private static uint Offset(uint unit) => (unit >> 10) << (int)((unit & (1u << 9)) >> 6);

    // The byte length of a valid UTF-8 character at `position`, or 0 when malformed.
    private static int Utf8Length(byte[] s, int position)
    {
        byte b = s[position];
        int length = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 0;
        if (length == 0 || position + length > s.Length) return 0;
        for (int i = 1; i < length; i++)
            if ((s[position + i] & 0xC0) != 0x80) return 0;
        return length;
    }

    // A character trie over the pieces, yielding (end offset, id) for every piece that starts at an offset, shortest
    // first, as darts-clone's commonPrefixSearch returns them.
    private sealed class PieceTrie
    {
        private sealed class Node
        {
            public readonly Dictionary<char, Node> Next = new();
            public int Id = -1;
        }

        private readonly Node _root = new();

        public void Add(string piece, int id)
        {
            var node = _root;
            foreach (char c in piece)
            {
                if (!node.Next.TryGetValue(c, out var next)) node.Next[c] = next = new Node();
                node = next;
            }
            if (node.Id < 0) node.Id = id;
        }

        public IEnumerable<(int End, int Id)> Prefixes(string text, int start)
        {
            var node = _root;
            for (int i = start; i < text.Length; i++)
            {
                if (!node.Next.TryGetValue(text[i], out node)) yield break;
                // A piece ends only on a whole character, never between a surrogate pair's halves.
                if (node.Id >= 0 && (i + 1 >= text.Length || !char.IsLowSurrogate(text[i + 1]))) yield return (i + 1, node.Id);
            }
        }
    }

    // A minimal protobuf reader for the fields a ModelProto uses.
    private ref struct ProtoReader
    {
        private readonly byte[] _data;
        private int _position;
        private readonly int _end;

        public ProtoReader(byte[] data, int start, int end)
        {
            _data = data;
            _position = start;
            _end = end;
        }

        public bool Next(out int field, out int wire)
        {
            if (_position >= _end)
            {
                field = wire = 0;
                return false;
            }
            ulong key = Varint();
            field = (int)(key >> 3);
            wire = (int)(key & 7);
            return true;
        }

        public ulong Varint()
        {
            ulong value = 0;
            for (int shift = 0; ; shift += 7)
            {
                byte b = _data[_position++];
                value |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) return value;
            }
        }

        public float Fixed32Float()
        {
            float value = BitConverter.ToSingle(_data, _position);
            _position += 4;
            return value;
        }

        public ProtoReader Message()
        {
            int length = (int)Varint();
            var inner = new ProtoReader(_data, _position, _position + length);
            _position += length;
            return inner;
        }

        public byte[] Bytes()
        {
            int length = (int)Varint();
            var bytes = new byte[length];
            Array.Copy(_data, _position, bytes, 0, length);
            _position += length;
            return bytes;
        }

        public string String() => Encoding.UTF8.GetString(Bytes());

        public void Skip(int wire)
        {
            switch (wire)
            {
                case 0: Varint(); break;
                case 1: _position += 8; break;
                case 2:
                    int length = (int)Varint();                    // read before adding: += would add to the old position
                    _position += length;
                    break;
                case 5: _position += 4; break;
                default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
            }
        }
    }
}
