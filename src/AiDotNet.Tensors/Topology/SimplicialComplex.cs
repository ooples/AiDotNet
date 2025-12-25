using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Topology;

/// <summary>
/// Represents a simplicial complex with boundary and Laplacian operators.
/// </summary>
public sealed class SimplicialComplex
{
    private readonly Dictionary<int, HashSet<Simplex>> _simplicesByDimension = new();

    public int MaxDimension => _simplicesByDimension.Count == 0 ? -1 : _simplicesByDimension.Keys.Max();

    public void AddSimplex(Simplex simplex, bool includeFaces = true)
    {
        if (simplex is null)
            throw new ArgumentNullException(nameof(simplex));

        AddSimplexInternal(simplex);

        if (!includeFaces || simplex.Dimension <= 0)
            return;

        foreach (var face in simplex.Boundary())
        {
            AddSimplex(face.Face, true);
        }
    }

    public IReadOnlyList<Simplex> GetSimplices(int dimension)
    {
        if (!_simplicesByDimension.TryGetValue(dimension, out var set))
            return Array.Empty<Simplex>();

        return set.OrderBy(s => string.Join(",", s.Vertices)).ToList();
    }

    public Matrix<T> BoundaryOperator<T>(int k)
    {
        var kSimplices = GetSimplices(k);
        var kMinusOneSimplices = GetSimplices(k - 1);
        var ops = MathHelper.GetNumericOperations<T>();

        var boundary = new Matrix<T>(kMinusOneSimplices.Count, kSimplices.Count);
        if (kSimplices.Count == 0 || kMinusOneSimplices.Count == 0)
            return boundary;

        var rowMap = new Dictionary<Simplex, int>();
        for (int i = 0; i < kMinusOneSimplices.Count; i++)
            rowMap[kMinusOneSimplices[i]] = i;

        for (int col = 0; col < kSimplices.Count; col++)
        {
            var simplex = kSimplices[col];
            foreach (var (sign, face) in simplex.Boundary())
            {
                if (!rowMap.TryGetValue(face, out int row))
                    continue;

                boundary[row, col] = ops.FromDouble(sign);
            }
        }

        return boundary;
    }

    public Matrix<T> IncidenceMatrix<T>(int k)
    {
        var boundary = BoundaryOperator<T>(k);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < boundary.Rows; i++)
        {
            for (int j = 0; j < boundary.Columns; j++)
            {
                if (!ops.Equals(boundary[i, j], ops.Zero))
                    boundary[i, j] = ops.One;
            }
        }

        return boundary;
    }

    public Matrix<T> HodgeLaplacian<T>(int k)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var kSimplices = GetSimplices(k);
        if (kSimplices.Count == 0)
            return new Matrix<T>(0, 0);

        Matrix<T> result = new Matrix<T>(kSimplices.Count, kSimplices.Count);
        if (k > 0)
        {
            var bK = BoundaryOperator<T>(k);
            var term = (Matrix<T>)bK.Transpose().Multiply(bK);
            result = (Matrix<T>)result.Add(term);
        }

        if (k < MaxDimension)
        {
            var bKPlus = BoundaryOperator<T>(k + 1);
            if (bKPlus.Rows > 0 && bKPlus.Columns > 0)
            {
                var term = (Matrix<T>)bKPlus.Multiply(bKPlus.Transpose());
                result = (Matrix<T>)result.Add(term);
            }
        }

        return result;
    }

    private void AddSimplexInternal(Simplex simplex)
    {
        if (!_simplicesByDimension.TryGetValue(simplex.Dimension, out var set))
        {
            set = new HashSet<Simplex>();
            _simplicesByDimension[simplex.Dimension] = set;
        }

        set.Add(simplex);
    }
}
