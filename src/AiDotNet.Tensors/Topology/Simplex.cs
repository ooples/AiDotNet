using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Topology;

/// <summary>
/// Represents an oriented simplex defined by an ordered set of vertices.
/// </summary>
public sealed class Simplex : IEquatable<Simplex>
{
    private readonly int[] _vertices;

    /// <summary>
    /// Gets a copy of the vertex indices to prevent external mutation.
    /// </summary>
    public int[] Vertices => (int[])_vertices.Clone();

    public int Dimension => _vertices.Length - 1;

    public Simplex(IEnumerable<int> vertices)
    {
        if (vertices is null)
            throw new ArgumentNullException(nameof(vertices));

        var list = vertices.ToList();
        if (list.Count == 0)
            throw new ArgumentException("A simplex must contain at least one vertex.", nameof(vertices));

        list.Sort();
        for (int i = 1; i < list.Count; i++)
        {
            if (list[i] == list[i - 1])
                throw new ArgumentException("Simplex vertices must be unique.", nameof(vertices));
        }

        _vertices = [.. list];
    }

    public IReadOnlyList<(int Sign, Simplex Face)> Boundary()
    {
        if (_vertices.Length <= 1)
            return Array.Empty<(int, Simplex)>();

        var faces = new List<(int, Simplex)>(_vertices.Length);
        for (int i = 0; i < _vertices.Length; i++)
        {
            var faceVertices = new int[_vertices.Length - 1];
            int index = 0;
            for (int j = 0; j < _vertices.Length; j++)
            {
                if (j == i)
                    continue;
                faceVertices[index++] = _vertices[j];
            }

            int sign = (i % 2 == 0) ? 1 : -1;
            faces.Add((sign, new Simplex(faceVertices)));
        }

        return faces;
    }

    public bool Equals(Simplex? other)
    {
        if (other is null)
            return false;
        if (_vertices.Length != other._vertices.Length)
            return false;

        for (int i = 0; i < _vertices.Length; i++)
        {
            if (_vertices[i] != other._vertices[i])
                return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is Simplex other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            for (int i = 0; i < _vertices.Length; i++)
                hash = hash * 23 + _vertices[i].GetHashCode();
            return hash;
        }
    }

    public override string ToString()
    {
        return $"Simplex(dim={Dimension}, vertices=[{string.Join(",", Vertices)}])";
    }
}
