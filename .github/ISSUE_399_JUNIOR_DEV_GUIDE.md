# Issue #399: Junior Developer Implementation Guide
## 3D AI Models (NeRF, PointNet, MeshCNN)

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [3D Representation Fundamentals](#3d-representation-fundamentals)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Strategy](#implementation-strategy)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Understanding the Problem

### What Are We Building?

We're implementing support for **3D AI models** that can:
- **3D Object Classification**: Recognize objects from 3D shapes (PointNet)
- **3D Scene Reconstruction**: Build 3D scenes from 2D images (NeRF)
- **3D Shape Analysis**: Process mesh structures (MeshCNN)
- **3D Segmentation**: Identify parts of 3D objects
- **Novel View Synthesis**: Generate new viewpoints of scenes

### Why 3D Models Are Special

3D data requires different representations than 2D images:
1. **Point clouds**: Unordered sets of 3D points
2. **Meshes**: Vertices, edges, and faces forming surfaces
3. **Voxels**: 3D grids (like 3D pixels)
4. **Implicit functions**: Neural fields representing surfaces
5. **Permutation invariance**: Order of points shouldn't matter

### Real-World Use Cases

- **Autonomous driving**: Understanding 3D environment from LiDAR
- **Robotics**: Grasping and manipulation of objects
- **AR/VR**: Creating immersive 3D scenes
- **Medical imaging**: CT/MRI scan analysis
- **Architecture**: 3D building modeling
- **Gaming**: Procedural 3D content generation

---

## 3D Representation Fundamentals

### Understanding 3D Data Formats

#### 1. Point Clouds
```csharp
/// <summary>
/// Point cloud: collection of 3D points with optional features.
/// Shape: [num_points, 3+features]
/// </summary>
/// <remarks>
/// For Beginners:
/// A point cloud is a set of 3D points in space:
/// - Each point has (x, y, z) coordinates
/// - Optional: color (R, G, B), normal vector, intensity
/// - Unordered: [point1, point2, point3] same as [point3, point1, point2]
///
/// Example from LiDAR:
/// - Car scanner shoots laser rays
/// - Each ray returns a 3D point
/// - 10,000+ points per scan
/// - Points form the shape of objects
///
/// Challenges:
/// - Irregular: not a grid like images
/// - Unordered: permutation invariance required
/// - Sparse: empty space between points
/// </remarks>
public class PointCloud<T>
{
    // Shape: [num_points, channels]
    // channels: 3 for XYZ, 6 for XYZ+RGB, more for normals/features
    public Tensor<T> Points { get; set; } = new Tensor<T>(new[] { 0, 3 });

    public int NumPoints { get; set; }       // Total points
    public int Channels { get; set; }        // 3=XYZ, 6=XYZ+RGB, etc.
    public bool HasColor { get; set; }       // RGB available?
    public bool HasNormals { get; set; }     // Surface normals available?

    // Bounding box
    public (T minX, T maxX, T minY, T maxY, T minZ, T maxZ) Bounds { get; set; }

    /// <summary>
    /// Get XYZ coordinates only (first 3 channels).
    /// </summary>
    public Tensor<T> GetCoordinates()
    {
        var coords = new Tensor<T>(new[] { NumPoints, 3 });
        for (int i = 0; i < NumPoints; i++)
        {
            coords[i, 0] = Points[i, 0];  // X
            coords[i, 1] = Points[i, 1];  // Y
            coords[i, 2] = Points[i, 2];  // Z
        }
        return coords;
    }

    /// <summary>
    /// Get RGB colors (channels 3-5 if present).
    /// </summary>
    public Tensor<T>? GetColors()
    {
        if (!HasColor || Channels < 6)
            return null;

        var colors = new Tensor<T>(new[] { NumPoints, 3 });
        for (int i = 0; i < NumPoints; i++)
        {
            colors[i, 0] = Points[i, 3];  // R
            colors[i, 1] = Points[i, 4];  // G
            colors[i, 2] = Points[i, 5];  // B
        }
        return colors;
    }
}
```

#### 2. Triangle Meshes
```csharp
/// <summary>
/// Triangle mesh: vertices connected by triangular faces.
/// Common format: .obj, .stl, .ply files.
/// </summary>
/// <remarks>
/// For Beginners:
/// A mesh represents a 3D surface using:
/// - Vertices: 3D points (corners)
/// - Faces: Triangles connecting 3 vertices
/// - Edges: Lines between vertices
///
/// Example cube:
/// - 8 vertices (corners)
/// - 12 triangles (2 per face × 6 faces)
/// - 18 edges
///
/// Why triangles?
/// - Simplest polygon (always planar)
/// - Easy to render
/// - Standard in computer graphics
/// </remarks>
public class TriangleMesh<T>
{
    // Vertices: [num_vertices, 3] for XYZ coordinates
    public Tensor<T> Vertices { get; set; } = new Tensor<T>(new[] { 0, 3 });

    // Faces: [num_faces, 3] with vertex indices
    // Each face is [v1_idx, v2_idx, v3_idx]
    public Tensor<int> Faces { get; set; } = new Tensor<int>(new[] { 0, 3 });

    // Edges: [num_edges, 2] with vertex indices (derived from faces)
    public Tensor<int> Edges { get; set; } = new Tensor<int>(new[] { 0, 2 });

    // Optional: per-vertex normals (for smooth shading)
    public Tensor<T>? VertexNormals { get; set; }

    // Optional: per-vertex colors
    public Tensor<T>? VertexColors { get; set; }

    public int NumVertices => Vertices.Shape[0];
    public int NumFaces => Faces.Shape[0];
    public int NumEdges => Edges.Shape[0];

    /// <summary>
    /// Compute edges from faces (each triangle has 3 edges).
    /// </summary>
    public void ComputeEdges()
    {
        var edgeSet = new HashSet<(int, int)>();

        for (int f = 0; f < NumFaces; f++)
        {
            int v0 = Faces[f, 0];
            int v1 = Faces[f, 1];
            int v2 = Faces[f, 2];

            // Add edges (ensure v_min < v_max for uniqueness)
            AddEdge(edgeSet, v0, v1);
            AddEdge(edgeSet, v1, v2);
            AddEdge(edgeSet, v2, v0);
        }

        Edges = new Tensor<int>(new[] { edgeSet.Count, 2 });
        int idx = 0;
        foreach (var (v0, v1) in edgeSet)
        {
            Edges[idx, 0] = v0;
            Edges[idx, 1] = v1;
            idx++;
        }
    }

    private void AddEdge(HashSet<(int, int)> edges, int v0, int v1)
    {
        if (v0 > v1)
            (v0, v1) = (v1, v0);  // Ensure v0 < v1
        edges.Add((v0, v1));
    }

    /// <summary>
    /// Compute vertex normals (average of adjacent face normals).
    /// </summary>
    public void ComputeVertexNormals()
    {
        VertexNormals = new Tensor<T>(new[] { NumVertices, 3 });

        // Initialize to zero
        for (int v = 0; v < NumVertices; v++)
        {
            for (int d = 0; d < 3; d++)
            {
                VertexNormals[v, d] = (T)(object)0.0;
            }
        }

        // Accumulate face normals
        for (int f = 0; f < NumFaces; f++)
        {
            int v0 = Faces[f, 0];
            int v1 = Faces[f, 1];
            int v2 = Faces[f, 2];

            // Get vertices
            var p0 = GetVertex(v0);
            var p1 = GetVertex(v1);
            var p2 = GetVertex(v2);

            // Compute face normal: (p1-p0) × (p2-p0)
            var normal = CrossProduct(
                Subtract(p1, p0),
                Subtract(p2, p0));

            // Add to vertex normals
            for (int d = 0; d < 3; d++)
            {
                dynamic n = VertexNormals[v0, d];
                VertexNormals[v0, d] = (T)(object)(n + normal[d]);

                n = VertexNormals[v1, d];
                VertexNormals[v1, d] = (T)(object)(n + normal[d]);

                n = VertexNormals[v2, d];
                VertexNormals[v2, d] = (T)(object)(n + normal[d]);
            }
        }

        // Normalize
        for (int v = 0; v < NumVertices; v++)
        {
            double norm = 0;
            for (int d = 0; d < 3; d++)
            {
                double val = Convert.ToDouble(VertexNormals[v, d]);
                norm += val * val;
            }
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int d = 0; d < 3; d++)
                {
                    double val = Convert.ToDouble(VertexNormals[v, d]);
                    VertexNormals[v, d] = (T)(object)(val / norm);
                }
            }
        }
    }

    private T[] GetVertex(int idx)
    {
        return new[]
        {
            Vertices[idx, 0],
            Vertices[idx, 1],
            Vertices[idx, 2]
        };
    }

    private T[] Subtract(T[] a, T[] b)
    {
        return new[]
        {
            (T)(object)(Convert.ToDouble(a[0]) - Convert.ToDouble(b[0])),
            (T)(object)(Convert.ToDouble(a[1]) - Convert.ToDouble(b[1])),
            (T)(object)(Convert.ToDouble(a[2]) - Convert.ToDouble(b[2]))
        };
    }

    private T[] CrossProduct(T[] a, T[] b)
    {
        double ax = Convert.ToDouble(a[0]);
        double ay = Convert.ToDouble(a[1]);
        double az = Convert.ToDouble(a[2]);
        double bx = Convert.ToDouble(b[0]);
        double by = Convert.ToDouble(b[1]);
        double bz = Convert.ToDouble(b[2]);

        return new[]
        {
            (T)(object)(ay * bz - az * by),
            (T)(object)(az * bx - ax * bz),
            (T)(object)(ax * by - ay * bx)
        };
    }
}
```

#### 3. Voxel Grids
```csharp
/// <summary>
/// Voxel grid: 3D grid of occupied/empty cells (like 3D pixels).
/// Shape: [depth, height, width] or [depth, height, width, channels]
/// </summary>
/// <remarks>
/// For Beginners:
/// Voxels are 3D pixels:
/// - Grid divides space into small cubes
/// - Each cube (voxel) is either filled or empty
/// - Like Minecraft blocks
///
/// Example: 32×32×32 voxel grid
/// - 32,768 voxels total
/// - Each voxel: 1 = occupied, 0 = empty
/// - Forms a 3D shape
///
/// Pros:
/// - Regular structure (easy to process with 3D CNNs)
/// - Simple representation
///
/// Cons:
/// - Memory intensive (32³ = 32,768 voxels)
/// - Fixed resolution
/// - Sparse (most voxels are empty)
/// </remarks>
public class VoxelGrid<T>
{
    // Shape: [depth, height, width] for binary occupancy
    // Or: [depth, height, width, channels] for features
    public Tensor<T> Grid { get; set; } = new Tensor<T>(new[] { 32, 32, 32 });

    public int Depth { get; set; }
    public int Height { get; set; }
    public int Width { get; set; }
    public int Channels { get; set; }  // 1 for binary, >1 for features

    // Real-world bounding box
    public (double minX, double maxX) BoundsX { get; set; }
    public (double minY, double maxY) BoundsY { get; set; }
    public (double minZ, double maxZ) BoundsZ { get; set; }

    public double VoxelSize { get; set; }  // Size of each voxel in world units

    /// <summary>
    /// Check if a voxel at (d, h, w) is occupied.
    /// </summary>
    public bool IsOccupied(int d, int h, int w)
    {
        if (d < 0 || d >= Depth || h < 0 || h >= Height || w < 0 || w >= Width)
            return false;

        double val = Convert.ToDouble(Grid[d, h, w]);
        return val > 0.5;
    }

    /// <summary>
    /// Convert 3D world coordinates to voxel indices.
    /// </summary>
    public (int d, int h, int w) WorldToVoxel(double x, double y, double z)
    {
        int d = (int)((z - BoundsZ.minZ) / VoxelSize);
        int h = (int)((y - BoundsY.minY) / VoxelSize);
        int w = (int)((x - BoundsX.minX) / VoxelSize);

        return (d, h, w);
    }

    /// <summary>
    /// Convert voxel indices to 3D world coordinates (center of voxel).
    /// </summary>
    public (double x, double y, double z) VoxelToWorld(int d, int h, int w)
    {
        double x = BoundsX.minX + (w + 0.5) * VoxelSize;
        double y = BoundsY.minY + (h + 0.5) * VoxelSize;
        double z = BoundsZ.minZ + (d + 0.5) * VoxelSize;

        return (x, y, z);
    }
}
```

#### 4. Neural Radiance Fields (Implicit Representation)
```csharp
/// <summary>
/// Neural Radiance Field (NeRF): implicit 3D representation.
/// Maps (x, y, z, theta, phi) → (r, g, b, density).
/// </summary>
/// <remarks>
/// For Beginners:
/// NeRF represents 3D scenes as a continuous function:
/// - Input: 3D position (x, y, z) + viewing direction (theta, phi)
/// - Output: Color (RGB) + density (opacity)
///
/// Instead of storing voxels or meshes, NeRF uses a neural network:
/// - Network learns the function
/// - Can query any point in 3D space
/// - Infinitely high resolution (continuous)
///
/// How it works:
/// 1. For each pixel in an image:
/// 2. Cast a ray through the pixel
/// 3. Sample points along the ray
/// 4. Query NeRF network at each point
/// 5. Composite colors/densities → pixel color
///
/// Applications:
/// - Novel view synthesis (new camera angles)
/// - 3D reconstruction from images
/// - Virtual tours
/// </remarks>
public class NeuralRadianceField<T>
{
    // Position encoding network
    private readonly PositionalEncoder<T> _posEncoder;

    // Direction encoding network
    private readonly PositionalEncoder<T> _dirEncoder;

    // MLP for density and features
    private readonly MultilayerPerceptron<T> _densityMLP;

    // MLP for RGB color
    private readonly MultilayerPerceptron<T> _colorMLP;

    public NeuralRadianceField(
        int posEncodingLevels = 10,
        int dirEncodingLevels = 4,
        int hiddenSize = 256,
        int numLayers = 8)
    {
        Guard.Positive(hiddenSize, nameof(hiddenSize));
        Guard.Positive(numLayers, nameof(numLayers));

        // Positional encoding for XYZ (higher frequency = finer details)
        _posEncoder = new PositionalEncoder<T>(
            inputDim: 3,  // XYZ
            numFrequencies: posEncodingLevels);

        // Positional encoding for viewing direction (lower frequency)
        _dirEncoder = new PositionalEncoder<T>(
            inputDim: 3,  // Direction vector
            numFrequencies: dirEncodingLevels);

        int posEncodedDim = 3 * 2 * posEncodingLevels;  // sin+cos for each frequency
        int dirEncodedDim = 3 * 2 * dirEncodingLevels;

        // MLP: encoded_pos → density + features
        _densityMLP = new MultilayerPerceptron<T>(
            inputSize: posEncodedDim,
            hiddenSizes: Enumerable.Repeat(hiddenSize, numLayers).ToArray(),
            outputSize: hiddenSize + 1,  // +1 for density
            activation: "relu");

        // MLP: features + encoded_dir → RGB
        _colorMLP = new MultilayerPerceptron<T>(
            inputSize: hiddenSize + dirEncodedDim,
            hiddenSizes: new[] { hiddenSize / 2 },
            outputSize: 3,  // RGB
            activation: "relu");
    }

    /// <summary>
    /// Query the radiance field at a 3D point with viewing direction.
    /// </summary>
    /// <param name="position">3D position (x, y, z).</param>
    /// <param name="direction">Viewing direction (normalized).</param>
    /// <returns>RGB color and density.</returns>
    public (Tensor<T> rgb, T density) Query(Tensor<T> position, Tensor<T> direction)
    {
        Guard.NotNull(position, nameof(position));
        Guard.NotNull(direction, nameof(direction));

        // position: [batch, 3]
        // direction: [batch, 3]

        // Step 1: Positional encoding
        var encodedPos = _posEncoder.Forward(position);
        var encodedDir = _dirEncoder.Forward(direction);

        // Step 2: Predict density and features from position
        var densityAndFeatures = _densityMLP.Forward(encodedPos);

        // Split: last channel is density, rest are features
        int batchSize = densityAndFeatures.Shape[0];
        int featureDim = densityAndFeatures.Shape[1] - 1;

        var features = new Tensor<T>(new[] { batchSize, featureDim });
        var density = new Tensor<T>(new[] { batchSize });

        for (int b = 0; b < batchSize; b++)
        {
            density[b] = densityAndFeatures[b, featureDim];  // Last channel

            for (int d = 0; d < featureDim; d++)
            {
                features[b, d] = densityAndFeatures[b, d];
            }
        }

        // Apply activation to density (ensure non-negative)
        density = ReLU(density);

        // Step 3: Predict RGB from features and viewing direction
        var combined = Concatenate(features, encodedDir);
        var rgb = _colorMLP.Forward(combined);

        // Apply sigmoid to RGB (ensure [0, 1] range)
        rgb = Sigmoid(rgb);

        return (rgb, density[0]);  // Simplified - return first batch item
    }

    private Tensor<T> ReLU(Tensor<T> x)
    {
        var result = x.Clone();
        for (int i = 0; i < x.Size; i++)
        {
            double val = Convert.ToDouble(x.Data[i]);
            result.Data[i] = (T)(object)Math.Max(0, val);
        }
        return result;
    }

    private Tensor<T> Sigmoid(Tensor<T> x)
    {
        var result = x.Clone();
        for (int i = 0; i < x.Size; i++)
        {
            double val = Convert.ToDouble(x.Data[i]);
            result.Data[i] = (T)(object)(1.0 / (1.0 + Math.Exp(-val)));
        }
        return result;
    }

    private Tensor<T> Concatenate(Tensor<T> a, Tensor<T> b)
    {
        // Concatenate along last dimension
        int batch = a.Shape[0];
        int dimA = a.Shape[1];
        int dimB = b.Shape[1];

        var result = new Tensor<T>(new[] { batch, dimA + dimB });

        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < dimA; j++)
            {
                result[i, j] = a[i, j];
            }
            for (int j = 0; j < dimB; j++)
            {
                result[i, dimA + j] = b[i, j];
            }
        }

        return result;
    }
}

/// <summary>
/// Positional encoding: map continuous values to high-frequency features.
/// Used in NeRF to help network represent high-frequency details.
/// </summary>
public class PositionalEncoder<T>
{
    private readonly int _inputDim;
    private readonly int _numFrequencies;

    public PositionalEncoder(int inputDim, int numFrequencies)
    {
        Guard.Positive(inputDim, nameof(inputDim));
        Guard.Positive(numFrequencies, nameof(numFrequencies));

        _inputDim = inputDim;
        _numFrequencies = numFrequencies;
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        // x: [batch, input_dim]
        // Output: [batch, input_dim * 2 * num_frequencies]

        int batch = x.Shape[0];
        int outputDim = _inputDim * 2 * _numFrequencies;

        var encoded = new Tensor<T>(new[] { batch, outputDim });

        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int d = 0; d < _inputDim; d++)
            {
                double val = Convert.ToDouble(x[b, d]);

                for (int freq = 0; freq < _numFrequencies; freq++)
                {
                    double frequency = Math.Pow(2, freq) * Math.PI;

                    // sin(2^freq * pi * x)
                    encoded[b, idx++] = (T)(object)Math.Sin(frequency * val);

                    // cos(2^freq * pi * x)
                    encoded[b, idx++] = (T)(object)Math.Cos(frequency * val);
                }
            }
        }

        return encoded;
    }
}
```

---

## Architecture Overview

### Model Taxonomy

```
3D AI Models
├── Point Cloud Models (Unordered points)
│   ├── PointNet (Permutation-invariant classification)
│   ├── PointNet++ (Hierarchical feature learning)
│   ├── DGCNN (Dynamic Graph CNN)
│   └── Point Transformer
│
├── Mesh Models (Vertices + faces)
│   ├── MeshCNN (Convolutions on edges)
│   ├── MeshNet (Face-based features)
│   └── SpiralNet (Spiral convolutions)
│
├── Voxel Models (3D grids)
│   ├── 3D CNN (Standard convolutions)
│   ├── VoxNet (Voxel-based classification)
│   └── OctNet (Hierarchical octrees)
│
└── Implicit Models (Neural fields)
    ├── NeRF (Neural Radiance Fields)
    ├── DeepSDF (Signed Distance Functions)
    └── Occupancy Networks
```

### PointNet Architecture

```csharp
/// <summary>
/// PointNet: Deep learning on point sets for 3D classification and segmentation.
/// Key innovation: Permutation invariance via max pooling.
/// </summary>
/// <remarks>
/// For Beginners:
/// PointNet processes unordered point clouds directly:
///
/// Problem: Point order shouldn't matter
/// - [p1, p2, p3] should give same result as [p3, p1, p2]
///
/// Solution: Symmetric function (max pooling)
/// 1. Transform each point independently (shared MLP)
/// 2. Pool features across all points (max pooling)
/// 3. Global feature is permutation-invariant
///
/// Architecture:
/// Input: [batch, num_points, 3] (XYZ coordinates)
/// → Input Transform (learn optimal rotation)
/// → Shared MLP: [3] → [64] → [64]
/// → Feature Transform (align features)
/// → Shared MLP: [64] → [128] → [1024]
/// → Max Pool: [batch, num_points, 1024] → [batch, 1024]
/// → Classification MLP: [1024] → [512] → [256] → [num_classes]
///
/// Why it works:
/// - max(f(p1), f(p2), ..., f(pN)) is same for any order
/// - Each point processed independently (shared weights)
/// - Global context from max pooling
/// </remarks>
public class PointNetModel<T> : IPointCloudModel<T>
{
    private readonly PointNetConfig _config;
    private readonly TransformNet<T> _inputTransform;
    private readonly SharedMLP<T> _mlp1;
    private readonly TransformNet<T> _featureTransform;
    private readonly SharedMLP<T> _mlp2;
    private readonly MaxPooling<T> _globalPooling;
    private readonly MultilayerPerceptron<T> _classificationHead;

    public PointNetModel(PointNetConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        // Input transform: learn 3×3 rotation matrix
        _inputTransform = new TransformNet<T>(
            inputDim: 3,
            outputDim: 3);

        // First MLP: 3 → 64 → 64
        _mlp1 = new SharedMLP<T>(
            inputSize: 3,
            hiddenSizes: new[] { 64, 64 },
            activation: "relu");

        // Feature transform: learn 64×64 transformation
        _featureTransform = new TransformNet<T>(
            inputDim: 64,
            outputDim: 64);

        // Second MLP: 64 → 128 → 1024
        _mlp2 = new SharedMLP<T>(
            inputSize: 64,
            hiddenSizes: new[] { 128, 1024 },
            activation: "relu");

        // Max pooling over points
        _globalPooling = new MaxPooling<T>(axis: 1);  // Pool over num_points

        // Classification head: 1024 → 512 → 256 → num_classes
        _classificationHead = new MultilayerPerceptron<T>(
            inputSize: 1024,
            hiddenSizes: new[] { 512, 256 },
            outputSize: config.NumClasses,
            activation: "relu");
    }

    public PointCloudOutput<T> Forward(PointCloud<T> pointCloud)
    {
        Guard.NotNull(pointCloud, nameof(pointCloud));

        // Get coordinates: [batch, num_points, 3]
        var points = pointCloud.Points;

        // Step 1: Input transform (align input)
        var transformMatrix = _inputTransform.Forward(points);
        var transformedPoints = ApplyTransform(points, transformMatrix);

        // Step 2: First MLP (point-wise features)
        var features1 = _mlp1.Forward(transformedPoints);
        // features1: [batch, num_points, 64]

        // Step 3: Feature transform (align features)
        var featureMatrix = _featureTransform.Forward(features1);
        var transformedFeatures = ApplyTransform(features1, featureMatrix);

        // Step 4: Second MLP (higher-level features)
        var features2 = _mlp2.Forward(transformedFeatures);
        // features2: [batch, num_points, 1024]

        // Step 5: Global max pooling (permutation-invariant)
        var globalFeatures = _globalPooling.Forward(features2);
        // globalFeatures: [batch, 1024]

        // Step 6: Classification
        var logits = _classificationHead.Forward(globalFeatures);

        return new PointCloudOutput<T>
        {
            Logits = logits,
            GlobalFeatures = globalFeatures,
            PointFeatures = features2
        };
    }

    private Tensor<T> ApplyTransform(Tensor<T> points, Tensor<T> matrix)
    {
        // points: [batch, num_points, dim]
        // matrix: [batch, dim, dim]
        // result: [batch, num_points, dim]

        int batch = points.Shape[0];
        int numPoints = points.Shape[1];
        int dim = points.Shape[2];

        var result = new Tensor<T>(points.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPoints; p++)
            {
                for (int i = 0; i < dim; i++)
                {
                    dynamic sum = (T)(object)0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += points[b, p, j] * matrix[b, i, j];
                    }
                    result[b, p, i] = (T)(object)sum;
                }
            }
        }

        return result;
    }
}

public class PointNetConfig
{
    public int NumClasses { get; set; } = 40;  // ModelNet40
    public bool UseFeatureTransform { get; set; } = true;
}
```

### TransformNet (T-Net)

```csharp
/// <summary>
/// T-Net: Learns transformation matrix to align inputs/features.
/// Ensures invariance to geometric transformations.
/// </summary>
public class TransformNet<T>
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly SharedMLP<T> _mlp;
    private readonly MaxPooling<T> _pooling;
    private readonly MultilayerPerceptron<T> _fcLayers;

    public TransformNet(int inputDim, int outputDim)
    {
        Guard.Positive(inputDim, nameof(inputDim));
        Guard.Positive(outputDim, nameof(outputDim));

        _inputDim = inputDim;
        _outputDim = outputDim;

        // Shared MLP for point-wise features
        _mlp = new SharedMLP<T>(
            inputSize: inputDim,
            hiddenSizes: new[] { 64, 128, 1024 },
            activation: "relu");

        // Max pooling over points
        _pooling = new MaxPooling<T>(axis: 1);

        // FC layers to predict transformation matrix
        _fcLayers = new MultilayerPerceptron<T>(
            inputSize: 1024,
            hiddenSizes: new[] { 512, 256 },
            outputSize: outputDim * outputDim,
            activation: "relu");
    }

    public Tensor<T> Forward(Tensor<T> points)
    {
        // points: [batch, num_points, input_dim]

        // Extract features
        var features = _mlp.Forward(points);
        // features: [batch, num_points, 1024]

        // Global pooling
        var globalFeatures = _pooling.Forward(features);
        // globalFeatures: [batch, 1024]

        // Predict transformation matrix
        var matrixFlat = _fcLayers.Forward(globalFeatures);
        // matrixFlat: [batch, output_dim * output_dim]

        // Reshape to matrix and add identity
        int batch = matrixFlat.Shape[0];
        var matrix = new Tensor<T>(new[] { batch, _outputDim, _outputDim });

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < _outputDim; i++)
            {
                for (int j = 0; j < _outputDim; j++)
                {
                    int idx = i * _outputDim + j;
                    double val = Convert.ToDouble(matrixFlat[b, idx]);

                    // Add identity matrix (helps training stability)
                    if (i == j)
                        val += 1.0;

                    matrix[b, i, j] = (T)(object)val;
                }
            }
        }

        return matrix;
    }
}
```

### MeshCNN Architecture

```csharp
/// <summary>
/// MeshCNN: Convolutional neural network for triangle meshes.
/// Operates on edges (unique to each mesh architecture).
/// </summary>
/// <remarks>
/// For Beginners:
/// MeshCNN processes meshes using edge-based convolutions:
///
/// Key idea: Edges are the fundamental unit
/// - Each edge connects two triangles
/// - Edge features: dihedral angle, edge length, etc.
/// - Convolution: aggregate features from neighboring edges
///
/// Edge pooling (like max pooling for images):
/// - Collapse edges to reduce mesh complexity
/// - Similar to image downsampling
/// - Maintains mesh connectivity
///
/// Architecture:
/// Input: Mesh with N edges
/// → Edge feature extraction (5 features per edge)
/// → Mesh Conv blocks (learn edge features)
/// → Edge pooling (reduce from N to N/2 edges)
/// → More conv blocks
/// → Global pooling
/// → Classification
///
/// Applications:
/// - 3D shape classification
/// - Mesh segmentation (label each face)
/// - Shape correspondence
/// </remarks>
public class MeshCNNModel<T> : IMeshModel<T>
{
    private readonly MeshCNNConfig _config;
    private readonly EdgeFeatureExtractor<T> _featureExtractor;
    private readonly List<MeshConvBlock<T>> _convBlocks;
    private readonly List<EdgePooling<T>> _poolingLayers;
    private readonly GlobalPooling<T> _globalPooling;
    private readonly MultilayerPerceptron<T> _classifier;

    public MeshCNNModel(MeshCNNConfig config)
    {
        Guard.NotNull(config, nameof(config));

        _config = config;

        // Extract initial edge features
        _featureExtractor = new EdgeFeatureExtractor<T>();

        // Convolution and pooling blocks
        _convBlocks = new List<MeshConvBlock<T>>();
        _poolingLayers = new List<EdgePooling<T>>();

        int currentChannels = 5;  // Initial edge features
        foreach (int channels in config.ConvChannels)
        {
            _convBlocks.Add(new MeshConvBlock<T>(
                inChannels: currentChannels,
                outChannels: channels));

            _poolingLayers.Add(new EdgePooling<T>(
                targetReduction: 0.5));  // Reduce by 50%

            currentChannels = channels;
        }

        // Global pooling over all edges
        _globalPooling = new GlobalPooling<T>();

        // Classification head
        _classifier = new MultilayerPerceptron<T>(
            inputSize: currentChannels,
            hiddenSizes: new[] { 256, 128 },
            outputSize: config.NumClasses,
            activation: "relu");
    }

    public MeshOutput<T> Forward(TriangleMesh<T> mesh)
    {
        Guard.NotNull(mesh, nameof(mesh));

        // Step 1: Extract edge features
        var edgeFeatures = _featureExtractor.Extract(mesh);
        // edgeFeatures: [num_edges, 5]

        var currentMesh = mesh;
        var currentFeatures = edgeFeatures;

        // Step 2: Apply conv and pooling blocks
        for (int i = 0; i < _convBlocks.Count; i++)
        {
            // Convolution
            currentFeatures = _convBlocks[i].Forward(
                currentMesh,
                currentFeatures);

            // Pooling (reduces mesh complexity)
            (currentMesh, currentFeatures) = _poolingLayers[i].Forward(
                currentMesh,
                currentFeatures);
        }

        // Step 3: Global pooling
        var globalFeatures = _globalPooling.Forward(currentFeatures);
        // globalFeatures: [1, channels]

        // Step 4: Classification
        var logits = _classifier.Forward(globalFeatures);

        return new MeshOutput<T>
        {
            Logits = logits,
            GlobalFeatures = globalFeatures,
            EdgeFeatures = currentFeatures
        };
    }
}

public class MeshCNNConfig
{
    public int[] ConvChannels { get; set; } = new[] { 32, 64, 128, 256 };
    public int NumClasses { get; set; } = 30;  // SHREC dataset
}
```

### EdgeFeatureExtractor

```csharp
/// <summary>
/// Extracts geometric features from mesh edges.
/// Features: dihedral angle, edge length, edge ratios, etc.
/// </summary>
public class EdgeFeatureExtractor<T>
{
    public Tensor<T> Extract(TriangleMesh<T> mesh)
    {
        Guard.NotNull(mesh, nameof(mesh));

        if (mesh.Edges.Shape[0] == 0)
        {
            mesh.ComputeEdges();
        }

        int numEdges = mesh.NumEdges;

        // 5 features per edge:
        // 1. Dihedral angle (angle between adjacent faces)
        // 2. Edge length
        // 3. Edge length ratio to adjacent edges
        // 4-5. Additional geometric features
        var features = new Tensor<T>(new[] { numEdges, 5 });

        for (int e = 0; e < numEdges; e++)
        {
            int v0 = mesh.Edges[e, 0];
            int v1 = mesh.Edges[e, 1];

            // Feature 1: Dihedral angle
            double dihedralAngle = ComputeDihedralAngle(mesh, v0, v1);
            features[e, 0] = (T)(object)dihedralAngle;

            // Feature 2: Edge length
            double edgeLength = ComputeEdgeLength(mesh, v0, v1);
            features[e, 1] = (T)(object)edgeLength;

            // Features 3-5: Additional geometric properties
            // (Simplified - real implementation would compute more features)
            features[e, 2] = (T)(object)1.0;
            features[e, 3] = (T)(object)1.0;
            features[e, 4] = (T)(object)1.0;
        }

        return features;
    }

    private double ComputeDihedralAngle(TriangleMesh<T> mesh, int v0, int v1)
    {
        // Find two faces sharing this edge
        // Compute normal vectors of both faces
        // Dihedral angle = angle between normals

        // Simplified - real implementation would find adjacent faces
        return Math.PI / 4;  // Placeholder
    }

    private double ComputeEdgeLength(TriangleMesh<T> mesh, int v0, int v1)
    {
        double dx = Convert.ToDouble(mesh.Vertices[v0, 0]) -
                    Convert.ToDouble(mesh.Vertices[v1, 0]);
        double dy = Convert.ToDouble(mesh.Vertices[v0, 1]) -
                    Convert.ToDouble(mesh.Vertices[v1, 1]);
        double dz = Convert.ToDouble(mesh.Vertices[v0, 2]) -
                    Convert.ToDouble(mesh.Vertices[v1, 2]);

        return Math.Sqrt(dx * dx + dy * dy + dz * dz);
    }
}
```

---

## Implementation Strategy

### Project Structure

```
src/
├── ThreeD/
│   ├── IPointCloudModel.cs
│   ├── IMeshModel.cs
│   ├── PointCloud.cs
│   ├── TriangleMesh.cs
│   ├── VoxelGrid.cs
│   └── Preprocessing/
│       ├── PointCloudNormalizer.cs
│       ├── MeshSimplifier.cs
│       └── VoxelConverter.cs
│
├── ThreeD/Models/
│   ├── PointNet/
│   │   ├── PointNetModel.cs
│   │   ├── PointNetConfig.cs
│   │   ├── TransformNet.cs
│   │   ├── SharedMLP.cs
│   │   └── PointNetProcessor.cs
│   │
│   ├── MeshCNN/
│   │   ├── MeshCNNModel.cs
│   │   ├── MeshCNNConfig.cs
│   │   ├── MeshConvBlock.cs
│   │   ├── EdgePooling.cs
│   │   └── EdgeFeatureExtractor.cs
│   │
│   └── NeRF/
│       ├── NeRFModel.cs
│       ├── NeRFConfig.cs
│       ├── VolumeRenderer.cs
│       ├── RayCaster.cs
│       └── PositionalEncoder.cs
│
└── ThreeD/Utils/
    ├── PointCloudIO.cs (Load/save .ply, .pcd)
    ├── MeshIO.cs (Load/save .obj, .stl)
    ├── GeometricUtils.cs
    └── Visualization.cs
```

---

## Testing Strategy

### Unit Tests

```csharp
namespace AiDotNetTests.ThreeD;

public class PointCloudTests
{
    [Fact]
    public void PointCloud_GetCoordinates_ReturnsXYZ()
    {
        // Arrange
        var pointCloud = CreateTestPointCloud(numPoints: 100);

        // Act
        var coords = pointCloud.GetCoordinates();

        // Assert
        Assert.NotNull(coords);
        Assert.Equal(100, coords.Shape[0]);
        Assert.Equal(3, coords.Shape[1]);  // XYZ
    }

    [Fact]
    public void PointNet_ProcessPointCloud_ReturnsLogits()
    {
        // Arrange
        var config = new PointNetConfig { NumClasses = 40 };
        var model = new PointNetModel<double>(config);

        var pointCloud = CreateTestPointCloud(numPoints: 1024);

        // Act
        var output = model.Forward(pointCloud);

        // Assert
        Assert.NotNull(output.Logits);
        Assert.Equal(40, output.Logits.Shape[1]);  // 40 classes
    }

    private PointCloud<double> CreateTestPointCloud(int numPoints)
    {
        var points = new Tensor<double>(new[] { numPoints, 3 });

        // Generate random points
        var random = new Random();
        for (int i = 0; i < numPoints; i++)
        {
            points[i, 0] = random.NextDouble();  // X
            points[i, 1] = random.NextDouble();  // Y
            points[i, 2] = random.NextDouble();  // Z
        }

        return new PointCloud<double>
        {
            Points = points,
            NumPoints = numPoints,
            Channels = 3
        };
    }
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Core 3D Infrastructure (8 hours)

#### AC 1.1: 3D Data Structures
**Files**:
- `src/ThreeD/PointCloud.cs`
- `src/ThreeD/TriangleMesh.cs`
- `src/ThreeD/VoxelGrid.cs`

#### AC 1.2: Preprocessing
**Files**:
- `src/ThreeD/Preprocessing/PointCloudNormalizer.cs`
- `src/ThreeD/Preprocessing/MeshSimplifier.cs`

**Tests**: `tests/ThreeD/PreprocessingTests.cs`

### Phase 2: PointNet Implementation (12 hours)

#### AC 2.1: Transform Networks
**File**: `src/ThreeD/Models/PointNet/TransformNet.cs`

#### AC 2.2: Shared MLP
**File**: `src/ThreeD/Models/PointNet/SharedMLP.cs`

#### AC 2.3: Complete PointNet
**File**: `src/ThreeD/Models/PointNet/PointNetModel.cs`

**Tests**: `tests/ThreeD/Models/PointNet/PointNetTests.cs`

### Phase 3: MeshCNN Implementation (14 hours)

#### AC 3.1: Edge Features
**File**: `src/ThreeD/Models/MeshCNN/EdgeFeatureExtractor.cs`

#### AC 3.2: Mesh Convolutions
**File**: `src/ThreeD/Models/MeshCNN/MeshConvBlock.cs`

#### AC 3.3: Edge Pooling
**File**: `src/ThreeD/Models/MeshCNN/EdgePooling.cs`

#### AC 3.4: Complete MeshCNN
**File**: `src/ThreeD/Models/MeshCNN/MeshCNNModel.cs`

**Tests**: `tests/ThreeD/Models/MeshCNN/MeshCNNTests.cs`

### Phase 4: NeRF Implementation (16 hours)

#### AC 4.1: Positional Encoding
**File**: `src/ThreeD/Models/NeRF/PositionalEncoder.cs`

#### AC 4.2: Volume Rendering
**File**: `src/ThreeD/Models/NeRF/VolumeRenderer.cs`

#### AC 4.3: Ray Casting
**File**: `src/ThreeD/Models/NeRF/RayCaster.cs`

#### AC 4.4: Complete NeRF
**File**: `src/ThreeD/Models/NeRF/NeRFModel.cs`

**Tests**: `tests/ThreeD/Models/NeRF/NeRFTests.cs`

### Phase 5: Documentation (4 hours)

#### AC 5.1: XML Documentation
Complete API documentation.

#### AC 5.2: Usage Examples
Create examples for 3D classification, mesh processing, novel view synthesis.

---

## Checklist Summary

### Phase 1: Core Infrastructure (8 hours)
- [ ] Implement PointCloud, TriangleMesh, VoxelGrid
- [ ] Implement preprocessing utilities
- [ ] Write unit tests
- [ ] Test with real 3D data files

### Phase 2: PointNet (12 hours)
- [ ] Implement TransformNet
- [ ] Implement SharedMLP
- [ ] Create PointNetModel
- [ ] Write integration tests
- [ ] Test on ModelNet40

### Phase 3: MeshCNN (14 hours)
- [ ] Implement edge feature extraction
- [ ] Implement mesh convolutions
- [ ] Implement edge pooling
- [ ] Create MeshCNNModel
- [ ] Write integration tests
- [ ] Test on SHREC dataset

### Phase 4: NeRF (16 hours)
- [ ] Implement positional encoding
- [ ] Implement volume rendering
- [ ] Implement ray casting
- [ ] Create NeRFModel
- [ ] Write integration tests
- [ ] Test novel view synthesis

### Phase 5: Documentation (4 hours)
- [ ] Add XML documentation
- [ ] Create usage examples
- [ ] Write performance benchmarks

### Total Estimated Time: 54 hours

---

## Success Criteria

1. **PointNet**: Achieves >85% accuracy on ModelNet40
2. **MeshCNN**: Segments meshes accurately
3. **NeRF**: Renders novel views with high quality
4. **Tests**: 80%+ coverage
5. **Performance**: Real-time or near real-time
6. **Documentation**: Complete XML docs and examples

---

## Common Pitfalls

### Pitfall 1: Ignoring Permutation Invariance
**Problem**: Point order affects results.
**Solution**: Use symmetric functions (max pooling).

### Pitfall 2: Memory Issues with Large Meshes
**Problem**: Millions of vertices.
**Solution**: Mesh simplification, edge pooling.

### Pitfall 3: Slow NeRF Rendering
**Problem**: Querying network for every ray sample.
**Solution**: Hierarchical sampling, caching.

---

## Resources

- [PointNet Paper](https://arxiv.org/abs/1612.00593)
- [MeshCNN Paper](https://arxiv.org/abs/1809.05910)
- [NeRF Paper](https://arxiv.org/abs/2003.08934)
- [ModelNet Dataset](http://modelnet.cs.princeton.edu/)
- [3D Deep Learning Survey](https://arxiv.org/abs/1912.12033)
