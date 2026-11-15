using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Iso-surface extraction from a dense scalar grid by marching cubes (Lorensen &amp; Cline, "Marching Cubes: A High
/// Resolution 3D Surface Construction Algorithm", SIGGRAPH 1987).
/// </summary>
/// <remarks>
/// <para>
/// Every cell of the grid is classified by which of its eight corners are inside, giving one of 256 cases. The case
/// indexes an edge table (which of the twelve cell edges the surface crosses) and a triangle table (how the crossing
/// points are joined). Each crossing point is placed by linear interpolation of the field along its edge, and each
/// vertex normal is the interpolated central-difference gradient of the field, as in the paper (its section 4).
/// </para>
/// <para>
/// The 256-case tables are built once, at type initialisation, from the topology of the cube: on every face the
/// crossing points are joined so that the inside corners lie to the left of each segment, and the segments of the six
/// faces close into the polygons a case produces; each polygon is fanned into triangles. Wherever a cell face is not
/// ambiguous this gives the polygons of the paper's 15 base cases. On a face whose corners alternate inside/outside
/// (the ambiguous faces of Durst 1988: the 1987 table resolves a case and its complement the same way there, which
/// leaves holes between some neighbouring cells) the rule always separates the inside corners. Two cells sharing a
/// face apply the same rule to it, so they always agree on the face's segments and the extracted surface has no
/// cracks: it is closed wherever the inside region does not touch the grid boundary.
/// </para>
/// <para>
/// Vertices on a grid edge are shared by every cell that touches the edge, so the mesh is indexed (not a triangle
/// soup) and its edges can be checked for being manifold. A cell with a non-finite corner value (NaN marks an
/// undefined sample, as in Hoppe et al. 1992) is skipped. The geometry is computed in double precision: the field is
/// read once and the vertices are converted to <typeparamref name="T"/> once at the end.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give this a 3D grid of numbers (for example the density of a NeRF sampled on a lattice, or
/// a signed distance function) and the level that separates inside from outside. It returns the triangle mesh of
/// that boundary, with outward-facing triangles and smooth per-vertex normals.
/// </para>
/// </remarks>
public static class MarchingCubes
{
    /// <summary>Corner offsets (x, y, z) of the unit cell, in the corner order of the tables.</summary>
    internal static readonly int[,] CornerOffsets =
    {
        { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 },
        { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }
    };

    /// <summary>The two corners of each of the twelve cell edges.</summary>
    internal static readonly int[,] EdgeCorners =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };

    /// <summary>The corners of each cell face, counter-clockwise when the face is seen from outside the cell.</summary>
    internal static readonly int[,] FaceCorners =
    {
        { 0, 4, 7, 3 }, // x = 0
        { 1, 2, 6, 5 }, // x = 1
        { 0, 1, 5, 4 }, // y = 0
        { 3, 7, 6, 2 }, // y = 1
        { 0, 3, 2, 1 }, // z = 0
        { 4, 5, 6, 7 }  // z = 1
    };

    /// <summary>
    /// For each of the 256 cases (bit c set when corner c is inside), the bit mask of the crossed edges.
    /// </summary>
    internal static readonly int[] EdgeTable = new int[256];

    /// <summary>
    /// For each of the 256 cases, the crossed-edge indices of its triangles, three per triangle, each triangle wound
    /// counter-clockwise when seen from outside the solid.
    /// </summary>
    internal static readonly int[][] TriangleTable = new int[256][];

    static MarchingCubes()
    {
        for (int mask = 0; mask < 256; mask++)
        {
            int edges = 0;
            for (int e = 0; e < 12; e++)
            {
                if (IsInside(mask, EdgeCorners[e, 0]) != IsInside(mask, EdgeCorners[e, 1]))
                    edges |= 1 << e;
            }

            EdgeTable[mask] = edges;
            TriangleTable[mask] = BuildCase(mask);
        }
    }

    /// <summary>
    /// Extracts the iso-surface of a scalar grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of the field and of the returned mesh.</typeparam>
    /// <param name="field">The field, shape [nx, ny, nz]; element (i, j, k) is the sample at
    /// <c>(originX + i * spacing, originY + j * spacing, originZ + k * spacing)</c>.</param>
    /// <param name="isoLevel">The level that separates inside from outside.</param>
    /// <param name="interior">Which side of <paramref name="isoLevel"/> is inside.</param>
    /// <param name="originX">World x of grid index 0.</param>
    /// <param name="originY">World y of grid index 0.</param>
    /// <param name="originZ">World z of grid index 0.</param>
    /// <param name="spacing">Distance between neighbouring samples on every axis.</param>
    /// <returns>The surface mesh: vertices [V, 3], faces [F, 3] and unit outward normals [V, 3].</returns>
    public static Mesh3D<T> Extract<T>(
        Tensor<T> field,
        double isoLevel,
        IsoSurfaceInterior interior,
        double originX,
        double originY,
        double originZ,
        double spacing)
    {
        if (field is null)
            throw new ArgumentNullException(nameof(field));
        if (field.Shape.Length != 3)
            throw new ArgumentException("The field must have shape [nx, ny, nz].", nameof(field));
        if (!(spacing > 0.0) || double.IsInfinity(spacing))
            throw new ArgumentOutOfRangeException(nameof(spacing), "The grid spacing must be positive and finite.");
        if (double.IsNaN(isoLevel) || double.IsInfinity(isoLevel))
            throw new ArgumentOutOfRangeException(nameof(isoLevel), "The iso-level must be finite.");

        int nx = field.Shape[0], ny = field.Shape[1], nz = field.Shape[2];
        double[] values = field.Cast<double>().ToArray();
        var surface = ExtractSurface(values, nx, ny, nz, isoLevel, interior, originX, originY, originZ, spacing);
        return surface.ToMesh<T>();
    }

    /// <summary>
    /// Extracts the iso-surface of a cubic grid that spans <c>[-halfExtent, halfExtent]</c> on every axis, sample
    /// index i lying at <c>-halfExtent + i * 2 * halfExtent / (n - 1)</c> (the layout of <c>linspace(-h, h, n)</c>).
    /// </summary>
    /// <typeparam name="T">The numeric type of the field and of the returned mesh.</typeparam>
    /// <param name="field">The field, shape [n, n, n].</param>
    /// <param name="isoLevel">The level that separates inside from outside.</param>
    /// <param name="interior">Which side of <paramref name="isoLevel"/> is inside.</param>
    /// <param name="halfExtent">Half the side of the cube the grid spans.</param>
    /// <returns>The surface mesh: vertices [V, 3], faces [F, 3] and unit outward normals [V, 3].</returns>
    public static Mesh3D<T> ExtractInCube<T>(
        Tensor<T> field,
        double isoLevel,
        IsoSurfaceInterior interior,
        double halfExtent = 1.0)
    {
        if (field is null)
            throw new ArgumentNullException(nameof(field));
        if (field.Shape.Length != 3 || field.Shape[0] != field.Shape[1] || field.Shape[1] != field.Shape[2])
            throw new ArgumentException("The field must be a cubic grid of shape [n, n, n].", nameof(field));
        if (field.Shape[0] < 2)
            throw new ArgumentException("The grid needs at least two samples per axis.", nameof(field));
        if (!(halfExtent > 0.0) || double.IsInfinity(halfExtent))
            throw new ArgumentOutOfRangeException(nameof(halfExtent), "The half extent must be positive and finite.");

        double spacing = 2.0 * halfExtent / (field.Shape[0] - 1);
        return Extract(field, isoLevel, interior, -halfExtent, -halfExtent, -halfExtent, spacing);
    }

    /// <summary>
    /// The double-precision core: <paramref name="values"/> is the row-major [nx, ny, nz] grid.
    /// </summary>
    internal static MeshSurface ExtractSurface(
        double[] values,
        int nx,
        int ny,
        int nz,
        double isoLevel,
        IsoSurfaceInterior interior,
        double originX,
        double originY,
        double originZ,
        double spacing)
    {
        if (nx < 2 || ny < 2 || nz < 2)
            throw new ArgumentException("The grid needs at least two samples per axis.");
        if ((long)nx * ny * nz != values.Length)
            throw new ArgumentException("The value count does not match the grid shape.");

        // s > 0 inside on either convention, so the classification and the interpolation are written once.
        var s = new double[values.Length];
        bool above = interior == IsoSurfaceInterior.AboveIsoLevel;
        for (int i = 0; i < values.Length; i++)
            s[i] = above ? values[i] - isoLevel : isoLevel - values[i];

        int strideX = ny * nz;
        var coordinates = new List<double>();
        var normals = new List<double>();
        var triangles = new List<int>();
        var vertexOfEdge = new Dictionary<long, int>();
        var corner = new double[8];
        var cornerIndex = new int[8];
        var cellVertex = new int[12];

        for (int x = 0; x < nx - 1; x++)
        {
            for (int y = 0; y < ny - 1; y++)
            {
                for (int z = 0; z < nz - 1; z++)
                {
                    int mask = 0;
                    bool defined = true;
                    for (int c = 0; c < 8; c++)
                    {
                        int index = ((x + CornerOffsets[c, 0]) * strideX) + ((y + CornerOffsets[c, 1]) * nz)
                            + z + CornerOffsets[c, 2];
                        cornerIndex[c] = index;
                        corner[c] = s[index];
                        if (double.IsNaN(corner[c]) || double.IsInfinity(corner[c]))
                        {
                            defined = false;
                            break;
                        }

                        if (corner[c] > 0.0)
                            mask |= 1 << c;
                    }

                    if (!defined)
                        continue;

                    int crossed = EdgeTable[mask];
                    if (crossed == 0)
                        continue;

                    for (int e = 0; e < 12; e++)
                    {
                        if ((crossed & (1 << e)) == 0)
                            continue;

                        // Interpolate from the lower grid point to the higher one, so the cells sharing this edge
                        // compute (and look up) the very same vertex.
                        int a = EdgeCorners[e, 0], b = EdgeCorners[e, 1];
                        if (cornerIndex[a] > cornerIndex[b])
                        {
                            int swap = a;
                            a = b;
                            b = swap;
                        }

                        int lower = cornerIndex[a];
                        int axis = CornerOffsets[b, 0] != CornerOffsets[a, 0] ? 0
                            : CornerOffsets[b, 1] != CornerOffsets[a, 1] ? 1 : 2;
                        long key = ((long)lower * 3) + axis;
                        if (!vertexOfEdge.TryGetValue(key, out int vertex))
                        {
                            vertex = coordinates.Count / 3;
                            vertexOfEdge[key] = vertex;
                            double sa = corner[a], sb = corner[b];
                            double t = sa == sb ? 0.5 : sa / (sa - sb);
                            int ax = x + CornerOffsets[a, 0], ay = y + CornerOffsets[a, 1], az = z + CornerOffsets[a, 2];
                            coordinates.Add(originX + ((ax + (axis == 0 ? t : 0.0)) * spacing));
                            coordinates.Add(originY + ((ay + (axis == 1 ? t : 0.0)) * spacing));
                            coordinates.Add(originZ + ((az + (axis == 2 ? t : 0.0)) * spacing));
                            AddGradientNormal(s, nx, ny, nz, ax, ay, az, axis, t, normals);
                        }

                        cellVertex[e] = vertex;
                    }

                    foreach (int edge in TriangleTable[mask])
                        triangles.Add(cellVertex[edge]);
                }
            }
        }

        var surface = new MeshSurface(coordinates.ToArray(), triangles.ToArray(), normals.ToArray());
        surface.RepairDegenerateNormals();
        return surface;
    }

    private static bool IsInside(int mask, int corner) => (mask & (1 << corner)) != 0;

    /// <summary>
    /// Builds the triangles of one case from the cube's face topology (see the type remarks).
    /// </summary>
    private static int[] BuildCase(int mask)
    {
        // next[e]: the crossed edge that follows e on the polygon boundary; -1 when e is not crossed.
        var next = new int[12];
        for (int e = 0; e < 12; e++)
            next[e] = -1;

        var faceEdges = new int[4];
        var faceExit = new bool[4];
        for (int f = 0; f < 6; f++)
        {
            int count = 0;
            for (int k = 0; k < 4; k++)
            {
                int a = FaceCorners[f, k], b = FaceCorners[f, (k + 1) % 4];
                bool insideA = IsInside(mask, a);
                if (insideA == IsInside(mask, b))
                    continue;

                faceEdges[count] = EdgeOf(a, b);
                faceExit[count] = insideA;
                count++;
            }

            // Walking the face counter-clockwise, each exit (inside -> outside) is joined to the entry just before
            // it. The segment then cuts off the inside corners between that entry and the exit, so they lie on its
            // left, and on an ambiguous face the two inside corners are separated.
            for (int k = 0; k < count; k++)
            {
                if (!faceExit[k])
                    continue;

                int previous = (k + count - 1) % count;
                next[faceEdges[k]] = faceEdges[previous];
            }
        }

        var triangles = new List<int>();
        var visited = new bool[12];
        var loop = new List<int>(12);
        for (int start = 0; start < 12; start++)
        {
            if (next[start] < 0 || visited[start])
                continue;

            loop.Clear();
            int e = start;
            while (!visited[e])
            {
                visited[e] = true;
                loop.Add(e);
                e = next[e];
            }

            // The loop runs with the inside on its left seen from outside the cell, so its own normal points into
            // the solid; the fan is emitted in reverse so every triangle faces outward.
            for (int i = 1; i + 1 < loop.Count; i++)
            {
                triangles.Add(loop[0]);
                triangles.Add(loop[i + 1]);
                triangles.Add(loop[i]);
            }
        }

        return triangles.ToArray();
    }

    private static int EdgeOf(int a, int b)
    {
        for (int e = 0; e < 12; e++)
        {
            if ((EdgeCorners[e, 0] == a && EdgeCorners[e, 1] == b) || (EdgeCorners[e, 0] == b && EdgeCorners[e, 1] == a))
                return e;
        }

        throw new InvalidOperationException("The corners do not share a cell edge.");
    }

    /// <summary>
    /// Appends the outward unit normal at the edge vertex: minus the gradient of s (s grows into the solid), the
    /// central-difference gradients at the two edge ends blended with the interpolation weight.
    /// </summary>
    private static void AddGradientNormal(
        double[] s, int nx, int ny, int nz, int ax, int ay, int az, int axis, double t, List<double> normals)
    {
        int bx = ax + (axis == 0 ? 1 : 0), by = ay + (axis == 1 ? 1 : 0), bz = az + (axis == 2 ? 1 : 0);
        double gx = ((1.0 - t) * Gradient(s, nx, ny, nz, ax, ay, az, 0)) + (t * Gradient(s, nx, ny, nz, bx, by, bz, 0));
        double gy = ((1.0 - t) * Gradient(s, nx, ny, nz, ax, ay, az, 1)) + (t * Gradient(s, nx, ny, nz, bx, by, bz, 1));
        double gz = ((1.0 - t) * Gradient(s, nx, ny, nz, ax, ay, az, 2)) + (t * Gradient(s, nx, ny, nz, bx, by, bz, 2));
        double length = Math.Sqrt((gx * gx) + (gy * gy) + (gz * gz));
        if (length > 0.0 && !double.IsNaN(length) && !double.IsInfinity(length))
        {
            normals.Add(-gx / length);
            normals.Add(-gy / length);
            normals.Add(-gz / length);
        }
        else
        {
            // Flat or undefined field here: MeshSurface.RepairDegenerateNormals fills it from the faces.
            normals.Add(0.0);
            normals.Add(0.0);
            normals.Add(0.0);
        }
    }

    /// <summary>
    /// Central difference of s along an axis at a grid point (one-sided at the grid boundary), in grid units. A
    /// non-finite neighbour falls back to the one-sided difference on the other side.
    /// </summary>
    private static double Gradient(double[] s, int nx, int ny, int nz, int x, int y, int z, int axis)
    {
        int n = axis == 0 ? nx : axis == 1 ? ny : nz;
        int position = axis == 0 ? x : axis == 1 ? y : z;
        int stride = axis == 0 ? ny * nz : axis == 1 ? nz : 1;
        int index = (x * ny * nz) + (y * nz) + z;
        double center = s[index];
        bool hasLow = position > 0 && IsFinite(s[index - stride]);
        bool hasHigh = position < n - 1 && IsFinite(s[index + stride]);
        if (hasLow && hasHigh)
            return 0.5 * (s[index + stride] - s[index - stride]);
        if (hasHigh)
            return s[index + stride] - center;
        if (hasLow)
            return center - s[index - stride];
        return 0.0;
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}
