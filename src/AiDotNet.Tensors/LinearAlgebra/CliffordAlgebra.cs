using System;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Defines a Clifford algebra signature with p positive, q negative, and r zero basis vectors.
/// </summary>
public sealed class CliffordAlgebra : IEquatable<CliffordAlgebra>
{
    private readonly int[] _metric;
    private readonly int[] _grades;

    public int P { get; }
    public int Q { get; }
    public int R { get; }
    public int Dimension { get; }
    public int BasisCount { get; }

    /// <summary>
    /// Gets the default Clifford algebra Cl(3,0,0) representing 3D Euclidean space.
    /// </summary>
    /// <remarks>
    /// This is a read-only singleton. To use a different algebra, pass it explicitly to constructors.
    /// </remarks>
    public static CliffordAlgebra Default { get; } = new CliffordAlgebra(3, 0, 0);

    public CliffordAlgebra(int p, int q, int r = 0)
    {
        if (p < 0 || q < 0 || r < 0)
            throw new ArgumentOutOfRangeException(nameof(p), "Signature components must be non-negative.");

        int dimension = p + q + r;
        if (dimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(p), "Dimension must be positive.");
        if (dimension > 30)
            throw new ArgumentOutOfRangeException(nameof(p), "Dimension must be 30 or less for 32-bit blade indices.");

        P = p;
        Q = q;
        R = r;
        Dimension = dimension;
        BasisCount = 1 << dimension;

        _metric = new int[dimension];
        for (int i = 0; i < dimension; i++)
        {
            if (i < p)
                _metric[i] = 1;
            else if (i < p + q)
                _metric[i] = -1;
            else
                _metric[i] = 0;
        }

        _grades = new int[BasisCount];
        for (int blade = 0; blade < BasisCount; blade++)
        {
            _grades[blade] = CountBits(blade);
        }
    }

    public int GetGrade(int blade)
    {
        if (blade < 0 || blade >= BasisCount)
            throw new ArgumentOutOfRangeException(nameof(blade));

        return _grades[blade];
    }

    public int GetMetricSign(int basisIndex)
    {
        if (basisIndex < 0 || basisIndex >= Dimension)
            throw new ArgumentOutOfRangeException(nameof(basisIndex));

        return _metric[basisIndex];
    }

    internal int ReorderingSign(int bladeA, int bladeB)
    {
        int sign = 1;
        for (int i = 0; i < Dimension; i++)
        {
            int bit = 1 << i;
            if ((bladeA & bit) != 0)
            {
                int lowerBits = bladeB & (bit - 1);
                if ((CountBits(lowerBits) & 1) == 1)
                    sign = -sign;
            }
        }

        return sign;
    }

    internal bool TryMultiplyBlades(int bladeA, int bladeB, out int resultBlade, out int sign)
    {
        if (bladeA < 0 || bladeA >= BasisCount)
            throw new ArgumentOutOfRangeException(nameof(bladeA));
        if (bladeB < 0 || bladeB >= BasisCount)
            throw new ArgumentOutOfRangeException(nameof(bladeB));

        sign = ReorderingSign(bladeA, bladeB);
        int common = bladeA & bladeB;
        if (common != 0)
        {
            for (int i = 0; i < Dimension; i++)
            {
                int bit = 1 << i;
                if ((common & bit) != 0)
                {
                    int metric = _metric[i];
                    if (metric == 0)
                    {
                        resultBlade = 0;
                        sign = 0;
                        return false;
                    }
                    if (metric < 0)
                        sign = -sign;
                }
            }
        }

        resultBlade = bladeA ^ bladeB;
        return true;
    }

    public bool Equals(CliffordAlgebra? other)
    {
        if (other is null)
            return false;

        return P == other.P && Q == other.Q && R == other.R;
    }

    public override bool Equals(object? obj) => obj is CliffordAlgebra other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + P.GetHashCode();
            hash = hash * 23 + Q.GetHashCode();
            hash = hash * 23 + R.GetHashCode();
            return hash;
        }
    }

    public override string ToString() => $"CliffordAlgebra({P},{Q},{R})";

    private static int CountBits(int value)
    {
        int count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }
}
