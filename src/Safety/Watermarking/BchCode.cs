namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// A binary, narrow-sense, shortened BCH code over GF(2^m) that corrects up to <see cref="Capacity"/> bit errors:
/// systematic encoding by the generator polynomial, decoding by syndromes, Berlekamp-Massey and a Chien search.
/// </summary>
/// <remarks>
/// StegaStamp (Tancik et al. 2020) protects its 56 data bits with BCH over GF(2^7), primitive polynomial
/// x^7 + x^3 + 1 (137), correcting 5 errors: 35 parity bits.
/// </remarks>
internal sealed class BchCode
{
    private readonly int _m;
    private readonly int _n;
    private readonly int[] _exp;
    private readonly int[] _log;
    private readonly bool[] _generator; // coefficients, lowest degree first

    /// <summary>Bits of data per codeword.</summary>
    public int DataBits { get; }

    /// <summary>Parity bits per codeword: the generator polynomial's degree.</summary>
    public int ParityBits { get; }

    /// <summary>Correctable bit errors per codeword.</summary>
    public int Capacity { get; }

    /// <summary>Bits per (shortened) codeword: data then parity.</summary>
    public int CodewordBits => DataBits + ParityBits;

    public BchCode(int m, int primitivePolynomial, int capacity, int dataBits)
    {
        if (m < 2 || m > 16) throw new ArgumentOutOfRangeException(nameof(m));
        _m = m;
        _n = (1 << m) - 1;
        _exp = new int[2 * _n];
        _log = new int[_n + 1];
        int x = 1;
        for (int i = 0; i < _n; i++)
        {
            _exp[i] = x;
            _log[x] = i;
            x <<= 1;
            if ((x & (1 << m)) != 0) x ^= primitivePolynomial;
        }

        if (x != 1) throw new ArgumentException($"{primitivePolynomial} is not primitive for GF(2^{m}).", nameof(primitivePolynomial));
        for (int i = _n; i < 2 * _n; i++) _exp[i] = _exp[i - _n];

        // g(x) = lcm of the minimal polynomials of alpha^1 .. alpha^(2t).
        var generator = new List<bool> { true };
        var covered = new HashSet<int>();
        for (int i = 1; i <= 2 * capacity; i++)
        {
            if (covered.Contains(i % _n)) continue;
            var coset = new List<int>();
            int e = i % _n;
            while (!coset.Contains(e))
            {
                coset.Add(e);
                covered.Add(e);
                e = (2 * e) % _n;
            }

            // Minimal polynomial: prod (x - alpha^e) over the coset, coefficients in GF(2^m), ending binary.
            var poly = new List<int> { 1 };
            foreach (int c in coset)
            {
                var next = new int[poly.Count + 1];
                for (int k = 0; k < poly.Count; k++)
                {
                    next[k + 1] ^= poly[k];
                    next[k] ^= Multiply(poly[k], _exp[c]);
                }

                poly = next.ToList();
            }

            var binary = poly.Select(c => c switch
            {
                0 => false,
                1 => true,
                _ => throw new InvalidOperationException("A minimal polynomial had a non-binary coefficient.")
            }).ToList();
            generator = MultiplyBinary(generator, binary);
        }

        _generator = generator.ToArray();
        ParityBits = _generator.Length - 1;
        Capacity = capacity;
        if (dataBits < 1 || dataBits + ParityBits > _n)
            throw new ArgumentOutOfRangeException(nameof(dataBits), dataBits, $"At most {_n - ParityBits} data bits fit this code.");
        DataBits = dataBits;
    }

    /// <summary>The paper's code: GF(2^7), polynomial 137, 5 errors, 56 data bits, 35 parity bits.</summary>
    public static BchCode StegaStamp() => new(7, 137, 5, 56);

    /// <summary>Systematic codeword: the data bits, then the remainder of data(x) x^(n-k) divided by g(x).</summary>
    public bool[] Encode(IReadOnlyList<bool> data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Count != DataBits) throw new ArgumentException($"Expected {DataBits} data bits, got {data.Count}.", nameof(data));
        // Polynomial order: codeword bit j is the coefficient of x^(CodewordBits - 1 - j).
        var remainder = new bool[ParityBits];
        for (int i = 0; i < DataBits; i++)
        {
            bool feedback = data[i] ^ remainder[ParityBits - 1];
            for (int k = ParityBits - 1; k > 0; k--) remainder[k] = remainder[k - 1] ^ (feedback && _generator[k]);
            remainder[0] = feedback && _generator[0];
        }

        var codeword = new bool[CodewordBits];
        for (int i = 0; i < DataBits; i++) codeword[i] = data[i];
        for (int k = 0; k < ParityBits; k++) codeword[DataBits + k] = remainder[ParityBits - 1 - k];
        return codeword;
    }

    /// <summary>
    /// Corrects up to <see cref="Capacity"/> errors in <paramref name="received"/>. Returns the data bits and the number
    /// of bits corrected, or null when the word has more errors than the code can locate.
    /// </summary>
    public (bool[] Data, int Corrected)? Decode(IReadOnlyList<bool> received)
    {
        if (received is null) throw new ArgumentNullException(nameof(received));
        if (received.Count != CodewordBits) throw new ArgumentException($"Expected {CodewordBits} bits, got {received.Count}.", nameof(received));
        int length = CodewordBits;
        var word = received.ToArray();

        // Syndromes S_j = r(alpha^j), j = 1..2t, with bit i the coefficient of x^(length - 1 - i).
        var syndromes = new int[2 * Capacity + 1];
        bool clean = true;
        for (int j = 1; j <= 2 * Capacity; j++)
        {
            int s = 0;
            for (int i = 0; i < length; i++)
                if (word[i]) s ^= _exp[(j * (length - 1 - i)) % _n];
            syndromes[j] = s;
            clean &= s == 0;
        }

        if (clean) return (word.Take(DataBits).ToArray(), 0);

        // Berlekamp-Massey for the error locator Lambda(x).
        var lambda = new int[2 * Capacity + 2];
        var b = new int[2 * Capacity + 2];
        lambda[0] = 1;
        b[0] = 1;
        int degree = 0, shift = 1, lastDiscrepancy = 1;
        for (int r = 1; r <= 2 * Capacity; r++)
        {
            int discrepancy = syndromes[r];
            for (int i = 1; i <= degree; i++) discrepancy ^= Multiply(lambda[i], syndromes[r - i]);
            if (discrepancy == 0)
            {
                shift++;
                continue;
            }

            var previous = (int[])lambda.Clone();
            int scale = Divide(discrepancy, lastDiscrepancy);
            for (int i = 0; i + shift < lambda.Length; i++) lambda[i + shift] ^= Multiply(scale, b[i]);
            if (2 * degree <= r - 1)
            {
                degree = r - degree;
                b = previous;
                lastDiscrepancy = discrepancy;
                shift = 1;
            }
            else
            {
                shift++;
            }
        }

        if (degree > Capacity) return null;

        // Chien search over the shortened positions: an error at bit i means Lambda(alpha^-(length - 1 - i)) = 0.
        int found = 0;
        for (int i = 0; i < length; i++)
        {
            int power = length - 1 - i;
            int value = 0;
            for (int k = 0; k <= degree; k++)
                if (lambda[k] != 0) value ^= _exp[(_log[lambda[k]] + (_n - power) % _n * k) % _n];
            if (value == 0)
            {
                word[i] = !word[i];
                found++;
            }
        }

        return found == degree ? (word.Take(DataBits).ToArray(), found) : null;
    }

    private int Multiply(int a, int b) => a == 0 || b == 0 ? 0 : _exp[_log[a] + _log[b]];

    private int Divide(int a, int b) => a == 0 ? 0 : _exp[(_log[a] - _log[b] + _n) % _n];

    private static List<bool> MultiplyBinary(List<bool> a, List<bool> b)
    {
        var product = new bool[a.Count + b.Count - 1];
        for (int i = 0; i < a.Count; i++)
        {
            if (!a[i]) continue;
            for (int j = 0; j < b.Count; j++) product[i + j] ^= b[j];
        }

        return product.ToList();
    }
}
