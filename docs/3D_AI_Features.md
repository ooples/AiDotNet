# 3D AI Features - Point Clouds and Neural Radiance Fields

This document describes the 3D AI capabilities added to AiDotNet, including point cloud processing and neural radiance fields for novel view synthesis.

## Overview

This implementation adds two major categories of 3D AI functionality:

1. **Point Cloud Processing**: Deep learning models for processing 3D point cloud data
2. **Neural Radiance Fields**: Methods for 3D scene representation and novel view synthesis

## Point Cloud Processing

Point clouds are collections of 3D points representing object surfaces or scenes, commonly captured by LIDAR sensors, depth cameras, or 3D scanners.

### Models Implemented

#### 1. PointNet

The pioneering architecture for directly processing unordered point sets.

**Key Features:**
- Permutation invariant (order of points doesn't matter)
- Spatial transformer networks (T-Net) for alignment
- Global and local feature extraction
- Suitable for classification and segmentation

**Example Usage:**
```csharp
using AiDotNet.PointCloud.Models;
using AiDotNet.PointCloud.Data;

// Create PointNet model for 40-class classification (e.g., ModelNet40)
var pointNet = new PointNet<double>(
    numClasses: 40,
    useInputTransform: true,
    useFeatureTransform: true
);

// Load point cloud data
var pointCloud = PointCloudData<double>.FromCoordinates(coordinates);

// Classify point cloud
var predictions = pointNet.ClassifyPointCloud(pointCloud.Points);

// Extract global features
var globalFeatures = pointNet.ExtractGlobalFeatures(pointCloud.Points);
```

**Reference:** "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (Qi et al., CVPR 2017)

#### 2. PointNet++

Hierarchical extension of PointNet with multi-scale feature learning.

**Key Features:**
- Hierarchical feature learning at multiple resolutions
- Set abstraction layers with local grouping
- Better handling of non-uniform point density
- Improved performance on complex shapes

**Example Usage:**
```csharp
using AiDotNet.PointCloud.Models;

// Create PointNet++ with hierarchical sampling
var pointNetPP = new PointNetPlusPlus<double>(
    numClasses: 40,
    samplingRates: new[] { 512, 128, 32 },
    searchRadii: new[] { 0.1, 0.2, 0.4 },
    mlpDimensions: new[] {
        new[] { 64, 64, 128 },
        new[] { 128, 128, 256 },
        new[] { 256, 512, 1024 }
    },
    useMultiScaleGrouping: false
);

// Classify point cloud
var predictions = pointNetPP.ClassifyPointCloud(pointCloud.Points);

// Segment point cloud (per-point labels)
var segmentation = pointNetPP.SegmentPointCloud(pointCloud.Points);
```

**Reference:** "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (Qi et al., NeurIPS 2017)

#### 3. DGCNN (Dynamic Graph CNN)

Graph-based approach using edge convolutions and dynamic k-NN graphs.

**Key Features:**
- Dynamic graph construction based on learned features
- Edge convolution for capturing local geometry
- Adapts neighborhood structure at each layer
- State-of-the-art classification performance

**Example Usage:**
```csharp
using AiDotNet.PointCloud.Models;

// Create DGCNN model
var dgcnn = new DGCNN<double>(
    numClasses: 40,
    knnK: 20,  // Number of nearest neighbors
    edgeConvChannels: new[] { 64, 64, 128, 256 },
    useDropout: true,
    dropoutRate: 0.5
);

// Classify point cloud
var predictions = dgcnn.ClassifyPointCloud(pointCloud.Points);

// Extract hierarchical features
var features = dgcnn.ExtractPointFeatures(pointCloud.Points);
```

**Reference:** "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., ACM TOG 2019)

### Point Cloud Data Structure

```csharp
using AiDotNet.PointCloud.Data;

// Create point cloud with XYZ coordinates only
var coordinates = new Matrix<double>(1000, 3);  // 1000 points
var pointCloud = PointCloudData<double>.FromCoordinates(coordinates);

// Create point cloud with additional features (e.g., RGB colors)
var pointsWithFeatures = new Tensor<double>(data, new[] { 1000, 6 });  // XYZ + RGB
var pointCloudWithColors = new PointCloudData<double>(pointsWithFeatures);

// Extract coordinates
var coords = pointCloud.GetCoordinates();

// Extract additional features
var features = pointCloudWithColors.GetFeatures();  // Returns RGB features
```

### Interfaces

All point cloud models implement these interfaces:

- `IPointCloudModel<T>`: Base interface for point cloud processing
- `IPointCloudClassification<T>`: For whole-cloud classification
- `IPointCloudSegmentation<T>`: For per-point segmentation

## Neural Radiance Fields (NeRF)

Neural radiance fields represent 3D scenes as continuous functions, enabling photorealistic novel view synthesis.

### Models Implemented

#### 1. NeRF (Neural Radiance Fields)

The original NeRF architecture for representing scenes as neural networks.

**Key Features:**
- Continuous 5D function: (x, y, z, θ, φ) → (r, g, b, σ)
- Positional encoding for high-frequency details
- Volume rendering for photorealistic images
- Hierarchical sampling for efficiency

**Example Usage:**
```csharp
using AiDotNet.NeuralRadianceFields.Models;

// Create NeRF model
var nerf = new NeRF<double>(
    positionEncodingLevels: 10,
    directionEncodingLevels: 4,
    hiddenDim: 256,
    numLayers: 8,
    useHierarchicalSampling: true
);

// Query radiance field at specific positions
var positions = CreatePositionTensor();  // [N, 3]
var viewingDirections = CreateDirectionTensor();  // [N, 3]
var (rgb, density) = nerf.QueryField(positions, viewingDirections);

// Render image from camera view
var cameraPosition = new Vector<double>(new[] { 0.0, 0.0, 5.0 });
var cameraRotation = Matrix<double>.Identity(3);
var focalLength = 50.0;

var renderedImage = nerf.RenderImage(
    cameraPosition,
    cameraRotation,
    imageWidth: 512,
    imageHeight: 512,
    focalLength: focalLength
);
```

**Reference:** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall et al., ECCV 2020)

#### 2. Instant-NGP (Instant Neural Graphics Primitives)

Ultra-fast NeRF variant using multiresolution hash encoding.

**Key Features:**
- 100× faster training than NeRF (minutes vs hours)
- 1000× faster rendering (milliseconds vs seconds)
- Multiresolution hash encoding
- Tiny MLP (2-4 layers vs 8)
- Occupancy grids for efficient sampling

**Example Usage:**
```csharp
using AiDotNet.NeuralRadianceFields.Models;

// Create Instant-NGP model
var instantNGP = new InstantNGP<double>(
    hashTableSize: 524288,  // 2^19 entries
    numLevels: 16,
    featuresPerLevel: 2,
    finestResolution: 2048,
    coarsestResolution: 16,
    mlpHiddenDim: 64,
    mlpNumLayers: 2,
    occupancyGridResolution: 128
);

// Query radiance field (same interface as NeRF)
var (rgb, density) = instantNGP.QueryField(positions, viewingDirections);

// Render image (much faster than NeRF)
var image = instantNGP.RenderImage(
    cameraPosition,
    cameraRotation,
    imageWidth: 512,
    imageHeight: 512,
    focalLength: focalLength
);
```

**Reference:** "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" (Müller et al., ACM TOG 2022)

#### 3. 3D Gaussian Splatting

State-of-the-art real-time rendering using explicit 3D Gaussians.

**Key Features:**
- Real-time rendering (100+ FPS)
- Explicit representation (no neural network evaluation)
- Photorealistic quality
- Adaptive Gaussian densification
- Easy scene editing and manipulation

**Example Usage:**
```csharp
using AiDotNet.NeuralRadianceFields.Models;

// Initialize from Structure-from-Motion point cloud
var initialPoints = LoadCOLMAPPointCloud("scene.ply");
var initialColors = LoadCOLMAPColors("scene.ply");

var gaussianSplatting = new GaussianSplatting<double>(
    initialPointCloud: initialPoints,
    initialColors: initialColors,
    useSphericalHarmonics: true,
    shDegree: 3  // Spherical harmonics degree for view-dependence
);

// Render image (real-time performance)
var image = gaussianSplatting.RenderImage(
    cameraPosition,
    cameraRotation,
    imageWidth: 1920,
    imageHeight: 1080,
    focalLength: focalLength
);

// Get number of Gaussians
var numGaussians = gaussianSplatting.GaussianCount;
```

**Reference:** "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., SIGGRAPH 2023)

### Ray Data Structure

```csharp
using AiDotNet.NeuralRadianceFields.Data;

// Create a ray
var origin = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
var direction = new Vector<double>(new[] { 0.0, 0.0, -1.0 });
var ray = new Ray<double>(
    origin: origin,
    direction: direction,
    nearBound: 2.0,
    farBound: 6.0
);

// Get point along ray at distance t
var point = ray.PointAt(5.0);  // Returns 3D position
```

### Interfaces

All radiance field models implement:

- `IRadianceField<T>`: Base interface for radiance fields
  - `QueryField()`: Query RGB and density at positions
  - `RenderImage()`: Render full image from camera
  - `RenderRays()`: Render specific rays

## Performance Comparison

| Model | Training Time | Rendering Speed | Memory | Quality |
|-------|--------------|-----------------|--------|---------|
| NeRF | 1-2 days | 30 sec/image | 5 MB | High |
| Instant-NGP | 5-10 min | 30 ms/image | 50-100 MB | High |
| Gaussian Splatting | 10-30 min | 10 ms/image (100+ FPS) | 200-500 MB | Very High |

## Applications

### Point Clouds
- Autonomous driving (LIDAR processing)
- Robotics (object recognition and grasping)
- AR/VR (3D scene understanding)
- 3D object classification and retrieval
- Part segmentation
- Semantic scene segmentation

### Neural Radiance Fields
- Virtual reality and AR
- Film and gaming (photorealistic asset capture)
- Real estate (virtual property tours)
- Cultural heritage preservation
- Robotics (3D mapping and navigation)
- Telepresence and remote collaboration

## Directory Structure

```
src/
├── PointCloud/
│   ├── Interfaces/
│   │   ├── IPointCloudModel.cs
│   │   ├── IPointCloudClassification.cs
│   │   └── IPointCloudSegmentation.cs
│   ├── Layers/
│   │   ├── PointConvolutionLayer.cs
│   │   ├── MaxPoolingLayer.cs
│   │   └── TNetLayer.cs
│   ├── Models/
│   │   ├── PointNet.cs
│   │   ├── PointNetPlusPlus.cs
│   │   └── DGCNN.cs
│   ├── Data/
│   │   └── PointCloudData.cs
│   └── Tasks/
│
└── NeuralRadianceFields/
    ├── Interfaces/
    │   └── IRadianceField.cs
    ├── Models/
    │   ├── NeRF.cs
    │   ├── InstantNGP.cs
    │   └── GaussianSplatting.cs
    ├── Layers/
    ├── Data/
    │   └── Ray.cs
    └── Rendering/

tests/
└── AiDotNet.Tests/
    └── UnitTests/
        ├── PointCloud/
        │   └── PointNetTests.cs
        └── NeuralRadianceFields/
            └── NeRFTests.cs
```

## Future Enhancements

### Point Cloud Processing
- 3D object detection implementations
- Instance segmentation
- Point cloud completion
- Point cloud upsampling
- Integration with more benchmarks (ScanNet, S3DIS)

### Neural Radiance Fields
- Dynamic NeRF (time-varying scenes)
- NeRF for unbounded scenes
- NeRF editing and manipulation tools
- Multi-view consistency losses
- Mip-NeRF (anti-aliasing)
- TensoRF (tensor factorization)

## References

1. Qi et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" CVPR 2017
2. Qi et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" NeurIPS 2017
3. Wang et al. "Dynamic Graph CNN for Learning on Point Clouds" ACM TOG 2019
4. Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" ECCV 2020
5. Müller et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" ACM TOG 2022
6. Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" SIGGRAPH 2023
