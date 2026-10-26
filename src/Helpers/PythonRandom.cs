namespace AiDotNet.Helpers;

/// <summary>
/// Python's <c>random.Random</c> (CPython 3.10's <c>_randommodule.c</c> and <c>random.py</c>): the MT19937 Mersenne
/// Twister seeded from an integer by <c>init_by_array</c>, <c>random()</c> from 53 bits, <c>getrandbits</c>,
/// <c>_randbelow</c> by rejection, <c>choice</c> and the Box–Muller <c>gauss</c> with its cached second value.
/// </summary>
internal sealed class PythonRandom
{
    private const int N = 624, M = 397;
    private readonly uint[] _state = new uint[N];
    private int _index;
    private double? _gaussNext;

    public PythonRandom(long seed) => Seed(seed);

    /// <summary><c>seed(a)</c> for an integer: <c>init_by_array</c> on |a|'s 32-bit words (least significant first), and
    /// the cached Gaussian cleared.</summary>
    public void Seed(long seed)
    {
        ulong n = seed < 0 ? (ulong)(-(seed + 1)) + 1 : (ulong)seed;
        var key = n >> 32 != 0 ? new[] { (uint)n, (uint)(n >> 32) } : new[] { (uint)n };
        InitByArray(key);
        _gaussNext = null;
    }

    private void InitGenrand(uint s)
    {
        _state[0] = s;
        for (int i = 1; i < N; i++) _state[i] = 1812433253u * (_state[i - 1] ^ (_state[i - 1] >> 30)) + (uint)i;
        _index = N;
    }

    private void InitByArray(uint[] key)
    {
        InitGenrand(19650218u);
        int i = 1, j = 0;
        for (int k = Math.Max(N, key.Length); k > 0; k--)
        {
            _state[i] = (_state[i] ^ ((_state[i - 1] ^ (_state[i - 1] >> 30)) * 1664525u)) + key[j] + (uint)j;
            i++;
            j++;
            if (i >= N) { _state[0] = _state[N - 1]; i = 1; }
            if (j >= key.Length) j = 0;
        }
        for (int k = N - 1; k > 0; k--)
        {
            _state[i] = (_state[i] ^ ((_state[i - 1] ^ (_state[i - 1] >> 30)) * 1566083941u)) - (uint)i;
            i++;
            if (i >= N) { _state[0] = _state[N - 1]; i = 1; }
        }
        _state[0] = 0x80000000u;
    }

    /// <summary><c>genrand_uint32</c>.</summary>
    public uint NextUInt32()
    {
        if (_index >= N)
        {
            for (int k = 0; k < N; k++)
            {
                uint y = (_state[k] & 0x80000000u) | (_state[(k + 1) % N] & 0x7fffffffu);
                _state[k] = _state[(k + M) % N] ^ (y >> 1) ^ ((y & 1u) != 0 ? 0x9908b0dfu : 0u);
            }
            _index = 0;
        }
        uint x = _state[_index++];
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680u;
        x ^= (x << 15) & 0xefc60000u;
        x ^= x >> 18;
        return x;
    }

    /// <summary><c>random()</c>: a 53-bit float in [0, 1).</summary>
    public double Random()
    {
        uint a = NextUInt32() >> 5, b = NextUInt32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    /// <summary><c>getrandbits(k)</c> for k ≤ 32.</summary>
    public uint GetRandBits(int k)
    {
        if (k < 0 || k > 32) throw new ArgumentOutOfRangeException(nameof(k));
        return k == 0 ? 0u : NextUInt32() >> (32 - k);
    }

    /// <summary><c>_randbelow(n)</c>: getrandbits of n's bit length until below n.</summary>
    public int RandBelow(int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        int k = 0;
        for (int v = n; v != 0; v >>= 1) k++;
        uint r = GetRandBits(k);
        while (r >= (uint)n) r = GetRandBits(k);
        return (int)r;
    }

    /// <summary><c>choice(seq)</c>.</summary>
    public TItem Choice<TItem>(IReadOnlyList<TItem> items)
    {
        if (items.Count == 0) throw new IndexOutOfRangeException("Cannot choose from an empty sequence.");
        return items[RandBelow(items.Count)];
    }

    /// <summary><c>gauss(mu, sigma)</c>: Box–Muller, the second value kept for the next call.</summary>
    public double Gauss(double mu = 0.0, double sigma = 1.0)
    {
        double z;
        if (_gaussNext is { } held)
        {
            z = held;
            _gaussNext = null;
        }
        else
        {
            double x2pi = Random() * 2 * Math.PI;
            double g2rad = Math.Sqrt(-2.0 * Math.Log(1.0 - Random()));
            z = Math.Cos(x2pi) * g2rad;
            _gaussNext = Math.Sin(x2pi) * g2rad;
        }
        return mu + z * sigma;
    }
}
