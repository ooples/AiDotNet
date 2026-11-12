namespace AiDotNet.Helpers;

/// <summary>
/// zlib 1.3.2's <c>compress</c> at the default level (6) with the zlib wrapper, the default window (15 bits) and memory
/// level (8): <c>deflate_slow</c>'s lazy matching over hash chains, and <c>trees.c</c>'s choice among stored, static and
/// dynamic Huffman blocks. Python's <c>zlib.compress(data)</c> produces these bytes, which a measure such as Whisper's
/// compression ratio reads; the platform's deflate (zlib-ng on .NET 9 and later) compresses to other lengths.
/// </summary>
/// <remarks>A line-by-line port of deflate.c and trees.c (Jean-loup Gailly and Mark Adler, zlib license), the paths
/// level 6 takes only.</remarks>
internal sealed class ZlibDeflate
{
    private const int MinMatch = 3, MaxMatch = 258, MinLookahead = MaxMatch + MinMatch + 1;
    private const int WBits = 15, WSize = 1 << WBits, WMask = WSize - 1, MaxDist = WSize - MinLookahead;
    private const int HashBits = 8 + 7, HashSize = 1 << HashBits, HashMask = HashSize - 1, HashShift = (HashBits + MinMatch - 1) / MinMatch;
    private const int LitBufSize = 1 << (8 + 6), SymEnd = (LitBufSize - 1) * 3, WinInit = MaxMatch, TooFar = 4096;
    private const int GoodMatch = 8, MaxLazy = 16, NiceMatch = 128, MaxChain = 128;
    private const int LengthCodes = 29, Literals = 256, LCodes = Literals + 1 + LengthCodes, DCodes = 30, BlCodes = 19;
    private const int HeapSize = 2 * LCodes + 1, MaxBits = 15, MaxBlBits = 7, EndBlock = 256;
    private const int Rep3To6 = 16, Repz3To10 = 17, Repz11To138 = 18;

    private static readonly int[] ExtraLbits = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
    private static readonly int[] ExtraDbits = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
    private static readonly int[] ExtraBlbits = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7 };
    private static readonly byte[] BlOrder = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

    // ct_data: Freq and Code share fc; Dad and Len share dl.
    private struct Node
    {
        public ushort Fc;
        public ushort Dl;
    }

    private sealed class TreeDesc
    {
        public Node[] Dyn = null!;
        public int MaxCode;
        public Node[]? Static;
        public int[] Extra = null!;
        public int ExtraBase;
        public int Elems;
        public int MaxLength;
    }

    private static readonly Node[] StaticLtree = new Node[LCodes + 2];
    private static readonly Node[] StaticDtree = new Node[DCodes];
    private static readonly byte[] DistCode = new byte[512];
    private static readonly byte[] LengthCode = new byte[MaxMatch - MinMatch + 1];
    private static readonly int[] BaseLength = new int[LengthCodes];
    private static readonly int[] BaseDist = new int[DCodes];

    static ZlibDeflate()
    {
        int length = 0, code;
        for (code = 0; code < LengthCodes - 1; code++)
        {
            BaseLength[code] = length;
            for (int n = 0; n < (1 << ExtraLbits[code]); n++) LengthCode[length++] = (byte)code;
        }
        LengthCode[length - 1] = (byte)code;
        int dist = 0;
        for (code = 0; code < 16; code++)
        {
            BaseDist[code] = dist;
            for (int n = 0; n < (1 << ExtraDbits[code]); n++) DistCode[dist++] = (byte)code;
        }
        dist >>= 7;
        for (; code < DCodes; code++)
        {
            BaseDist[code] = dist << 7;
            for (int n = 0; n < (1 << (ExtraDbits[code] - 7)); n++) DistCode[256 + dist++] = (byte)code;
        }
        var blCount = new ushort[MaxBits + 1];
        int k = 0;
        while (k <= 143) { StaticLtree[k++].Dl = 8; blCount[8]++; }
        while (k <= 255) { StaticLtree[k++].Dl = 9; blCount[9]++; }
        while (k <= 279) { StaticLtree[k++].Dl = 7; blCount[7]++; }
        while (k <= 287) { StaticLtree[k++].Dl = 8; blCount[8]++; }
        GenCodes(StaticLtree, LCodes + 1, blCount);
        for (int n = 0; n < DCodes; n++)
        {
            StaticDtree[n].Dl = 5;
            StaticDtree[n].Fc = (ushort)BiReverse((uint)n, 5);
        }
    }

    // ---------------------------------------------------------------- state

    private readonly byte[] _input;
    private int _nextIn, _availIn;
    private uint _adler = 1;
    private readonly List<byte> _output = new();

    private readonly byte[] _window = new byte[2 * WSize];
    private readonly ushort[] _prev = new ushort[WSize];
    private readonly ushort[] _head = new ushort[HashSize];
    private uint _insH;
    private long _blockStart;
    private int _matchLength = MinMatch - 1, _prevMatch, _matchAvailable, _strStart, _matchStart, _lookahead;
    private int _prevLength = MinMatch - 1, _insert;
    private long _highWater;

    private readonly Node[] _dynLtree = new Node[HeapSize];
    private readonly Node[] _dynDtree = new Node[2 * DCodes + 1];
    private readonly Node[] _blTree = new Node[2 * BlCodes + 1];
    private readonly TreeDesc _lDesc, _dDesc, _blDesc;
    private readonly ushort[] _blCount = new ushort[MaxBits + 1];
    private readonly int[] _heap = new int[2 * LCodes + 1];
    private int _heapLen, _heapMax;
    private readonly byte[] _depth = new byte[2 * LCodes + 1];
    private readonly byte[] _symBuf = new byte[LitBufSize * 3];
    private int _symNext;
    private ulong _optLen, _staticLen;
    private ushort _biBuf;
    private int _biValid;

    private ZlibDeflate(byte[] input)
    {
        _input = input;
        _availIn = input.Length;
        _lDesc = new TreeDesc { Dyn = _dynLtree, Static = StaticLtree, Extra = ExtraLbits, ExtraBase = Literals + 1, Elems = LCodes, MaxLength = MaxBits };
        _dDesc = new TreeDesc { Dyn = _dynDtree, Static = StaticDtree, Extra = ExtraDbits, ExtraBase = 0, Elems = DCodes, MaxLength = MaxBits };
        _blDesc = new TreeDesc { Dyn = _blTree, Static = null, Extra = ExtraBlbits, ExtraBase = 0, Elems = BlCodes, MaxLength = MaxBlBits };
        InitBlock();
    }

    /// <summary><c>zlib.compress(data)</c>: the zlib header, the level-6 deflate stream and the Adler-32 trailer.</summary>
    public static byte[] Compress(byte[] data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        var s = new ZlibDeflate(data);
        // Header: deflate, 32K window, level 6 (flags 2), check bits.
        int header = (8 + ((WBits - 8) << 4)) << 8;
        header |= 2 << 6;
        header += 31 - (header % 31);
        s._output.Add((byte)(header >> 8));
        s._output.Add((byte)header);
        s.DeflateSlow();
        uint adler = s._adler;
        s._output.Add((byte)(adler >> 24));
        s._output.Add((byte)(adler >> 16));
        s._output.Add((byte)(adler >> 8));
        s._output.Add((byte)adler);
        return s._output.ToArray();
    }

    private void PutByte(byte b) => _output.Add(b);

    private void PutShort(ushort w)
    {
        PutByte((byte)(w & 0xff));
        PutByte((byte)(w >> 8));
    }

    private void SendBits(int value, int length)
    {
        if (_biValid > 16 - length)
        {
            _biBuf |= unchecked((ushort)(value << _biValid));
            PutShort(_biBuf);
            _biBuf = (ushort)((ushort)value >> (16 - _biValid));
            _biValid += length - 16;
        }
        else
        {
            _biBuf |= unchecked((ushort)(value << _biValid));
            _biValid += length;
        }
    }

    private void SendCode(int c, Node[] tree) => SendBits(tree[c].Fc, tree[c].Dl);

    private static uint BiReverse(uint code, int len)
    {
        uint res = 0;
        do
        {
            res |= code & 1;
            code >>= 1;
            res <<= 1;
        } while (--len > 0);
        return res >> 1;
    }

    private void BiWindup()
    {
        if (_biValid > 8) PutShort(_biBuf);
        else if (_biValid > 0) PutByte((byte)_biBuf);
        _biBuf = 0;
        _biValid = 0;
    }

    private static void GenCodes(Node[] tree, int maxCode, ushort[] blCount)
    {
        var nextCode = new ushort[MaxBits + 1];
        uint code = 0;
        for (int bits = 1; bits <= MaxBits; bits++)
        {
            code = (code + blCount[bits - 1]) << 1;
            nextCode[bits] = (ushort)code;
        }
        for (int n = 0; n <= maxCode; n++)
        {
            int len = tree[n].Dl;
            if (len == 0) continue;
            tree[n].Fc = (ushort)BiReverse(nextCode[len]++, len);
        }
    }

    private void InitBlock()
    {
        for (int n = 0; n < LCodes; n++) _dynLtree[n].Fc = 0;
        for (int n = 0; n < DCodes; n++) _dynDtree[n].Fc = 0;
        for (int n = 0; n < BlCodes; n++) _blTree[n].Fc = 0;
        _dynLtree[EndBlock].Fc = 1;
        _optLen = _staticLen = 0;
        _symNext = 0;
    }

    private static bool Smaller(Node[] tree, int n, int m, byte[] depth) =>
        tree[n].Fc < tree[m].Fc || (tree[n].Fc == tree[m].Fc && depth[n] <= depth[m]);

    private void PqDownHeap(Node[] tree, int k)
    {
        int v = _heap[k];
        int j = k << 1;
        while (j <= _heapLen)
        {
            if (j < _heapLen && Smaller(tree, _heap[j + 1], _heap[j], _depth)) j++;
            if (Smaller(tree, v, _heap[j], _depth)) break;
            _heap[k] = _heap[j];
            k = j;
            j <<= 1;
        }
        _heap[k] = v;
    }

    private void GenBitlen(TreeDesc desc)
    {
        var tree = desc.Dyn;
        int maxCode = desc.MaxCode;
        var stree = desc.Static;
        var extra = desc.Extra;
        int baseIndex = desc.ExtraBase, maxLength = desc.MaxLength;
        int h, n, m, bits, xbits;
        int overflow = 0;
        for (bits = 0; bits <= MaxBits; bits++) _blCount[bits] = 0;
        tree[_heap[_heapMax]].Dl = 0;
        for (h = _heapMax + 1; h < HeapSize; h++)
        {
            n = _heap[h];
            bits = tree[tree[n].Dl].Dl + 1;
            if (bits > maxLength) { bits = maxLength; overflow++; }
            tree[n].Dl = (ushort)bits;
            if (n > maxCode) continue;
            _blCount[bits]++;
            xbits = 0;
            if (n >= baseIndex) xbits = extra[n - baseIndex];
            ushort f = tree[n].Fc;
            _optLen += (ulong)f * (uint)(bits + xbits);
            if (stree is not null) _staticLen += (ulong)f * (uint)(stree[n].Dl + xbits);
        }
        if (overflow == 0) return;
        do
        {
            bits = maxLength - 1;
            while (_blCount[bits] == 0) bits--;
            _blCount[bits]--;
            _blCount[bits + 1] += 2;
            _blCount[maxLength]--;
            overflow -= 2;
        } while (overflow > 0);
        for (bits = maxLength; bits != 0; bits--)
        {
            n = _blCount[bits];
            while (n != 0)
            {
                m = _heap[--h];
                if (m > maxCode) continue;
                if (tree[m].Dl != bits)
                {
                    _optLen = unchecked(_optLen + ((ulong)bits - tree[m].Dl) * tree[m].Fc);
                    tree[m].Dl = (ushort)bits;
                }
                n--;
            }
        }
    }

    private void BuildTree(TreeDesc desc)
    {
        var tree = desc.Dyn;
        var stree = desc.Static;
        int elems = desc.Elems;
        int n, m, maxCode = -1, node;
        _heapLen = 0;
        _heapMax = HeapSize;
        for (n = 0; n < elems; n++)
        {
            if (tree[n].Fc != 0)
            {
                _heap[++_heapLen] = maxCode = n;
                _depth[n] = 0;
            }
            else
            {
                tree[n].Dl = 0;
            }
        }
        while (_heapLen < 2)
        {
            node = _heap[++_heapLen] = maxCode < 2 ? ++maxCode : 0;
            tree[node].Fc = 1;
            _depth[node] = 0;
            _optLen--;
            if (stree is not null) _staticLen -= stree[node].Dl;
        }
        desc.MaxCode = maxCode;
        for (n = _heapLen / 2; n >= 1; n--) PqDownHeap(tree, n);
        node = elems;
        do
        {
            n = _heap[1];
            _heap[1] = _heap[_heapLen--];
            PqDownHeap(tree, 1);
            m = _heap[1];
            _heap[--_heapMax] = n;
            _heap[--_heapMax] = m;
            tree[node].Fc = (ushort)(tree[n].Fc + tree[m].Fc);
            _depth[node] = (byte)((_depth[n] >= _depth[m] ? _depth[n] : _depth[m]) + 1);
            tree[n].Dl = tree[m].Dl = (ushort)node;
            _heap[1] = node++;
            PqDownHeap(tree, 1);
        } while (_heapLen >= 2);
        _heap[--_heapMax] = _heap[1];
        GenBitlen(desc);
        GenCodes(tree, maxCode, _blCount);
    }

    private void ScanTree(Node[] tree, int maxCode)
    {
        int prevlen = -1, curlen, nextlen = tree[0].Dl, count = 0, maxCount = 7, minCount = 4;
        if (nextlen == 0) { maxCount = 138; minCount = 3; }
        tree[maxCode + 1].Dl = 0xffff;
        for (int n = 0; n <= maxCode; n++)
        {
            curlen = nextlen;
            nextlen = tree[n + 1].Dl;
            if (++count < maxCount && curlen == nextlen) continue;
            else if (count < minCount) _blTree[curlen].Fc += (ushort)count;
            else if (curlen != 0)
            {
                if (curlen != prevlen) _blTree[curlen].Fc++;
                _blTree[Rep3To6].Fc++;
            }
            else if (count <= 10) _blTree[Repz3To10].Fc++;
            else _blTree[Repz11To138].Fc++;
            count = 0;
            prevlen = curlen;
            if (nextlen == 0) { maxCount = 138; minCount = 3; }
            else if (curlen == nextlen) { maxCount = 6; minCount = 3; }
            else { maxCount = 7; minCount = 4; }
        }
    }

    private void SendTree(Node[] tree, int maxCode)
    {
        int prevlen = -1, curlen, nextlen = tree[0].Dl, count = 0, maxCount = 7, minCount = 4;
        if (nextlen == 0) { maxCount = 138; minCount = 3; }
        for (int n = 0; n <= maxCode; n++)
        {
            curlen = nextlen;
            nextlen = tree[n + 1].Dl;
            if (++count < maxCount && curlen == nextlen) continue;
            else if (count < minCount)
            {
                do { SendCode(curlen, _blTree); } while (--count != 0);
            }
            else if (curlen != 0)
            {
                if (curlen != prevlen)
                {
                    SendCode(curlen, _blTree);
                    count--;
                }
                SendCode(Rep3To6, _blTree);
                SendBits(count - 3, 2);
            }
            else if (count <= 10)
            {
                SendCode(Repz3To10, _blTree);
                SendBits(count - 3, 3);
            }
            else
            {
                SendCode(Repz11To138, _blTree);
                SendBits(count - 11, 7);
            }
            count = 0;
            prevlen = curlen;
            if (nextlen == 0) { maxCount = 138; minCount = 3; }
            else if (curlen == nextlen) { maxCount = 6; minCount = 3; }
            else { maxCount = 7; minCount = 4; }
        }
    }

    private int BuildBlTree()
    {
        ScanTree(_dynLtree, _lDesc.MaxCode);
        ScanTree(_dynDtree, _dDesc.MaxCode);
        BuildTree(_blDesc);
        int maxBlindex;
        for (maxBlindex = BlCodes - 1; maxBlindex >= 3; maxBlindex--)
            if (_blTree[BlOrder[maxBlindex]].Dl != 0) break;
        _optLen += 3 * ((ulong)maxBlindex + 1) + 5 + 5 + 4;
        return maxBlindex;
    }

    private void SendAllTrees(int lcodes, int dcodes, int blcodes)
    {
        SendBits(lcodes - 257, 5);
        SendBits(dcodes - 1, 5);
        SendBits(blcodes - 4, 4);
        for (int rank = 0; rank < blcodes; rank++) SendBits(_blTree[BlOrder[rank]].Dl, 3);
        SendTree(_dynLtree, lcodes - 1);
        SendTree(_dynDtree, dcodes - 1);
    }

    private void StoredBlock(int start, int storedLen, int last)
    {
        SendBits((0 << 1) + last, 3);
        BiWindup();
        PutShort((ushort)storedLen);
        PutShort((ushort)~storedLen);
        for (int i = 0; i < storedLen; i++) PutByte(_window[start + i]);
    }

    private static int DCode(int dist) => dist < 256 ? DistCode[dist] : DistCode[256 + (dist >> 7)];

    private void CompressBlock(Node[] ltree, Node[] dtree)
    {
        int sx = 0;
        if (_symNext != 0)
            do
            {
                int dist = _symBuf[sx++] & 0xff;
                dist += (_symBuf[sx++] & 0xff) << 8;
                int lc = _symBuf[sx++];
                if (dist == 0)
                {
                    SendCode(lc, ltree);
                }
                else
                {
                    int code = LengthCode[lc];
                    SendCode(code + Literals + 1, ltree);
                    int extra = ExtraLbits[code];
                    if (extra != 0)
                    {
                        lc -= BaseLength[code];
                        SendBits(lc, extra);
                    }
                    dist--;
                    code = DCode(dist);
                    SendCode(code, dtree);
                    extra = ExtraDbits[code];
                    if (extra != 0)
                    {
                        dist -= BaseDist[code];
                        SendBits(dist, extra);
                    }
                }
            } while (sx < _symNext);
        SendCode(EndBlock, ltree);
    }

    private void FlushBlock(bool last)
    {
        int storedLen = (int)(_strStart - _blockStart);
        bool hasBuffer = _blockStart >= 0;
        BuildTree(_lDesc);
        BuildTree(_dDesc);
        int maxBlindex = BuildBlTree();
        ulong optLenb = (_optLen + 3 + 7) >> 3;
        ulong staticLenb = (_staticLen + 3 + 7) >> 3;
        if (staticLenb <= optLenb) optLenb = staticLenb;
        int lastBit = last ? 1 : 0;
        if ((ulong)storedLen + 4 <= optLenb && hasBuffer)
        {
            StoredBlock((int)_blockStart, storedLen, lastBit);
        }
        else if (staticLenb == optLenb)
        {
            SendBits((1 << 1) + lastBit, 3);
            CompressBlock(StaticLtree, StaticDtree);
        }
        else
        {
            SendBits((2 << 1) + lastBit, 3);
            SendAllTrees(_lDesc.MaxCode + 1, _dDesc.MaxCode + 1, maxBlindex + 1);
            CompressBlock(_dynLtree, _dynDtree);
        }
        InitBlock();
        if (last) BiWindup();
        _blockStart = _strStart;
    }

    private bool TallyLiteral(byte c)
    {
        _symBuf[_symNext++] = 0;
        _symBuf[_symNext++] = 0;
        _symBuf[_symNext++] = c;
        _dynLtree[c].Fc++;
        return _symNext == SymEnd;
    }

    private bool TallyDistance(int distance, int length)
    {
        byte len = (byte)length;
        ushort dist = (ushort)distance;
        _symBuf[_symNext++] = (byte)dist;
        _symBuf[_symNext++] = (byte)(dist >> 8);
        _symBuf[_symNext++] = len;
        dist--;
        _dynLtree[LengthCode[len] + Literals + 1].Fc++;
        _dynDtree[DCode(dist)].Fc++;
        return _symNext == SymEnd;
    }

    // ---------------------------------------------------------------- deflate.c

    private void UpdateHash(byte c) => _insH = ((_insH << HashShift) ^ c) & HashMask;

    private int InsertString(int str)
    {
        UpdateHash(_window[str + (MinMatch - 1)]);
        int matchHead = _prev[str & WMask] = _head[_insH];
        _head[_insH] = (ushort)str;
        return matchHead;
    }

    private void SlideHash()
    {
        for (int n = 0; n < HashSize; n++) _head[n] = (ushort)(_head[n] >= WSize ? _head[n] - WSize : 0);
        for (int n = 0; n < WSize; n++) _prev[n] = (ushort)(_prev[n] >= WSize ? _prev[n] - WSize : 0);
    }

    private int ReadBuf(int at, int size)
    {
        int len = Math.Min(_availIn, size);
        if (len == 0) return 0;
        _availIn -= len;
        Array.Copy(_input, _nextIn, _window, at, len);
        _adler = Adler32(_adler, _input, _nextIn, len);
        _nextIn += len;
        return len;
    }

    private static uint Adler32(uint adler, byte[] buffer, int offset, int length)
    {
        uint a = adler & 0xffff, b = adler >> 16;
        for (int i = 0; i < length; i++)
        {
            a = (a + buffer[offset + i]) % 65521;
            b = (b + a) % 65521;
        }
        return (b << 16) | a;
    }

    private void FillWindow()
    {
        int more;
        do
        {
            more = 2 * WSize - _lookahead - _strStart;
            if (_strStart >= WSize + MaxDist)
            {
                Array.Copy(_window, WSize, _window, 0, WSize - more);
                _matchStart -= WSize;
                _strStart -= WSize;
                _blockStart -= WSize;
                if (_insert > _strStart) _insert = _strStart;
                SlideHash();
                more += WSize;
            }
            if (_availIn == 0) break;
            int n = ReadBuf(_strStart + _lookahead, more);
            _lookahead += n;
            if (_lookahead + _insert >= MinMatch)
            {
                int str = _strStart - _insert;
                _insH = _window[str];
                UpdateHash(_window[str + 1]);
                while (_insert != 0)
                {
                    UpdateHash(_window[str + MinMatch - 1]);
                    _prev[str & WMask] = _head[_insH];
                    _head[_insH] = (ushort)str;
                    str++;
                    _insert--;
                    if (_lookahead + _insert < MinMatch) break;
                }
            }
        } while (_lookahead < MinLookahead && _availIn != 0);
        long windowSize = 2L * WSize;
        if (_highWater < windowSize)
        {
            long curr = _strStart + (long)_lookahead;
            long init;
            if (_highWater < curr)
            {
                init = windowSize - curr;
                if (init > WinInit) init = WinInit;
                Array.Clear(_window, (int)curr, (int)init);
                _highWater = curr + init;
            }
            else if (_highWater < curr + WinInit)
            {
                init = curr + WinInit - _highWater;
                if (init > windowSize - _highWater) init = windowSize - _highWater;
                Array.Clear(_window, (int)_highWater, (int)init);
                _highWater += init;
            }
        }
    }

    private int LongestMatch(int curMatch)
    {
        int chainLength = MaxChain;
        int scan = _strStart;
        int bestLen = _prevLength;
        int niceMatch = NiceMatch;
        int limit = _strStart > MaxDist ? _strStart - MaxDist : 0;
        int strend = _strStart + MaxMatch;
        byte scanEnd1 = _window[scan + bestLen - 1];
        byte scanEnd = _window[scan + bestLen];
        if (_prevLength >= GoodMatch) chainLength >>= 2;
        if (niceMatch > _lookahead) niceMatch = _lookahead;
        do
        {
            int match = curMatch;
            if (_window[match + bestLen] != scanEnd || _window[match + bestLen - 1] != scanEnd1 || _window[match] != _window[scan]
                || _window[++match] != _window[scan + 1]) continue;
            scan += 2;
            match++;
            do
            {
            } while (_window[++scan] == _window[++match] && _window[++scan] == _window[++match] && _window[++scan] == _window[++match]
                     && _window[++scan] == _window[++match] && _window[++scan] == _window[++match] && _window[++scan] == _window[++match]
                     && _window[++scan] == _window[++match] && _window[++scan] == _window[++match] && scan < strend);
            int len = MaxMatch - (strend - scan);
            scan = strend - MaxMatch;
            if (len > bestLen)
            {
                _matchStart = curMatch;
                bestLen = len;
                if (len >= niceMatch) break;
                scanEnd1 = _window[scan + bestLen - 1];
                scanEnd = _window[scan + bestLen];
            }
        } while ((curMatch = _prev[curMatch & WMask]) > limit && --chainLength != 0);
        return bestLen <= _lookahead ? bestLen : _lookahead;
    }

    private void DeflateSlow()
    {
        int hashHead;
        while (true)
        {
            if (_lookahead < MinLookahead)
            {
                FillWindow();
                if (_lookahead == 0) break;
            }
            hashHead = 0;
            if (_lookahead >= MinMatch) hashHead = InsertString(_strStart);
            _prevLength = _matchLength;
            _prevMatch = _matchStart;
            _matchLength = MinMatch - 1;
            if (hashHead != 0 && _prevLength < MaxLazy && _strStart - hashHead <= MaxDist)
            {
                _matchLength = LongestMatch(hashHead);
                if (_matchLength <= 5 && _matchLength == MinMatch && _strStart - _matchStart > TooFar) _matchLength = MinMatch - 1;
            }
            if (_prevLength >= MinMatch && _matchLength <= _prevLength)
            {
                int maxInsert = _strStart + _lookahead - MinMatch;
                bool flush = TallyDistance(_strStart - 1 - _prevMatch, _prevLength - MinMatch);
                _lookahead -= _prevLength - 1;
                _prevLength -= 2;
                do
                {
                    if (++_strStart <= maxInsert) hashHead = InsertString(_strStart);
                } while (--_prevLength != 0);
                _matchAvailable = 0;
                _matchLength = MinMatch - 1;
                _strStart++;
                if (flush) FlushBlock(false);
            }
            else if (_matchAvailable != 0)
            {
                if (TallyLiteral(_window[_strStart - 1])) FlushBlock(false);
                _strStart++;
                _lookahead--;
            }
            else
            {
                _matchAvailable = 1;
                _strStart++;
                _lookahead--;
            }
        }
        if (_matchAvailable != 0)
        {
            TallyLiteral(_window[_strStart - 1]);
            _matchAvailable = 0;
        }
        _insert = _strStart < MinMatch - 1 ? _strStart : MinMatch - 1;
        FlushBlock(true);
    }
}
