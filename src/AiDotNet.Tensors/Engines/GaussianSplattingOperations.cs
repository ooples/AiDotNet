using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides vectorized implementations of 3D Gaussian Splatting operations.
/// These operations are shared between CPU and GPU engines.
/// </summary>
public static class GaussianSplattingOperations
{
    #region Gaussian Projection

    /// <summary>
    /// Projects 3D Gaussians to 2D screen space for rasterization.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="means3D">3D Gaussian center positions in world space, shape [N, 3].</param>
    /// <param name="covariances3D">
    /// 3D covariance matrices in CAMERA SPACE (must be pre-transformed by view matrix rotation).
    /// Shape [N, 6] for upper-triangular storage (c00, c01, c02, c11, c12, c22) or [N, 3, 3] for full matrix.
    /// </param>
    /// <param name="viewMatrix">4x4 view matrix to transform world coordinates to camera space.</param>
    /// <param name="projMatrix">4x4 projection matrix containing focal lengths (p00=fx, p11=fy).</param>
    /// <param name="imageWidth">Output image width in pixels.</param>
    /// <param name="imageHeight">Output image height in pixels.</param>
    /// <param name="means2D">Output 2D screen positions, shape [N, 2].</param>
    /// <param name="covariances2D">Output 2D covariances (a, b, c for ax² + 2bxy + cy²), shape [N, 3].</param>
    /// <param name="depths">Output depth values in camera space, shape [N].</param>
    /// <param name="visible">Output visibility flags, shape [N].</param>
    /// <remarks>
    /// The 2D covariance is computed using the Jacobian of perspective projection: Σ_2D = J * Σ_3D * J^T.
    /// This is the standard approach used in 3D Gaussian Splatting (Kerbl et al., 2023).
    /// </remarks>
    public static void ProjectGaussians3DTo2D<T>(
        Tensor<T> means3D,
        Tensor<T> covariances3D,
        Matrix<T> viewMatrix,
        Matrix<T> projMatrix,
        int imageWidth,
        int imageHeight,
        out Tensor<T> means2D,
        out Tensor<T> covariances2D,
        out Tensor<T> depths,
        out Tensor<bool> visible)
    {
        if (means3D == null) throw new ArgumentNullException(nameof(means3D));
        if (covariances3D == null) throw new ArgumentNullException(nameof(covariances3D));
        if (viewMatrix == null) throw new ArgumentNullException(nameof(viewMatrix));
        if (projMatrix == null) throw new ArgumentNullException(nameof(projMatrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = means3D.Shape[0];

        // Validate against integer overflow for large datasets
        // Maximum safe size is int.MaxValue / 9 (for 3x3 covariance matrix indexing)
        const int maxSafeGaussians = int.MaxValue / 9;
        if (numGaussians > maxSafeGaussians)
        {
            throw new ArgumentException(
                $"Number of Gaussians ({numGaussians}) exceeds maximum safe limit ({maxSafeGaussians}) to prevent integer overflow in index calculations.",
                nameof(means3D));
        }

        var means2DData = new T[numGaussians * 2];
        var cov2DData = new T[numGaussians * 3]; // a, b, c for ax² + 2bxy + cy²
        var depthsData = new T[numGaussians];
        var visibleData = new bool[numGaussians];

        // Extract view matrix elements
        double v00 = numOps.ToDouble(viewMatrix[0, 0]);
        double v01 = numOps.ToDouble(viewMatrix[0, 1]);
        double v02 = numOps.ToDouble(viewMatrix[0, 2]);
        double v03 = numOps.ToDouble(viewMatrix[0, 3]);
        double v10 = numOps.ToDouble(viewMatrix[1, 0]);
        double v11 = numOps.ToDouble(viewMatrix[1, 1]);
        double v12 = numOps.ToDouble(viewMatrix[1, 2]);
        double v13 = numOps.ToDouble(viewMatrix[1, 3]);
        double v20 = numOps.ToDouble(viewMatrix[2, 0]);
        double v21 = numOps.ToDouble(viewMatrix[2, 1]);
        double v22 = numOps.ToDouble(viewMatrix[2, 2]);
        double v23 = numOps.ToDouble(viewMatrix[2, 3]);

        // Extract projection matrix elements
        double p00 = numOps.ToDouble(projMatrix[0, 0]);
        double p11 = numOps.ToDouble(projMatrix[1, 1]);

        double cx = imageWidth * 0.5;
        double cy = imageHeight * 0.5;

        Parallel.For(0, numGaussians, i =>
        {
            // Get 3D mean
            double mx = numOps.ToDouble(means3D.GetFlat(i * 3));
            double my = numOps.ToDouble(means3D.GetFlat(i * 3 + 1));
            double mz = numOps.ToDouble(means3D.GetFlat(i * 3 + 2));

            // Transform to camera space
            double camX = v00 * mx + v01 * my + v02 * mz + v03;
            double camY = v10 * mx + v11 * my + v12 * mz + v13;
            double camZ = v20 * mx + v21 * my + v22 * mz + v23;

            // Check if behind camera
            if (camZ <= 0.001)
            {
                visibleData[i] = false;
                return;
            }

            // Project to screen space
            double invZ = 1.0 / camZ;
            double screenX = p00 * camX * invZ * cx + cx;
            double screenY = p11 * camY * invZ * cy + cy;

            // Check if in frustum (explicit casts to avoid potential integer overflow)
            if (screenX < -(double)imageWidth || screenX > 2.0 * (double)imageWidth ||
                screenY < -(double)imageHeight || screenY > 2.0 * (double)imageHeight)
            {
                visibleData[i] = false;
                return;
            }

            // Get 3D covariance (assume upper triangular storage: c00, c01, c02, c11, c12, c22)
            int covOffset = i * 6;
            double c00, c01, c02, c11, c12, c22;
            if (covariances3D.Shape[1] == 6)
            {
                c00 = numOps.ToDouble(covariances3D.GetFlat(covOffset));
                c01 = numOps.ToDouble(covariances3D.GetFlat(covOffset + 1));
                c02 = numOps.ToDouble(covariances3D.GetFlat(covOffset + 2));
                c11 = numOps.ToDouble(covariances3D.GetFlat(covOffset + 3));
                c12 = numOps.ToDouble(covariances3D.GetFlat(covOffset + 4));
                c22 = numOps.ToDouble(covariances3D.GetFlat(covOffset + 5));
            }
            else if (covariances3D.Shape.Length == 3) // [N, 3, 3] format
            {
                c00 = numOps.ToDouble(covariances3D.GetFlat(i * 9));
                c01 = numOps.ToDouble(covariances3D.GetFlat(i * 9 + 1));
                c02 = numOps.ToDouble(covariances3D.GetFlat(i * 9 + 2));
                c11 = numOps.ToDouble(covariances3D.GetFlat(i * 9 + 4));
                c12 = numOps.ToDouble(covariances3D.GetFlat(i * 9 + 5));
                c22 = numOps.ToDouble(covariances3D.GetFlat(i * 9 + 8));
            }
            else
            {
                throw new ArgumentException(
                    $"Unsupported covariance shape: expected [N, 6] or [N, 3, 3], got [{string.Join(", ", covariances3D.Shape)}]",
                    nameof(covariances3D));
            }

            // Project 3D covariance to 2D using Jacobian of perspective projection
            // NOTE: Input covariances must already be in camera space (pre-transformed by view matrix)
            //
            // Jacobian J of perspective projection (u = fx*X/Z, v = fy*Y/Z):
            //   J = [[fx/Z,    0, -fx*X/Z²],
            //        [   0, fy/Z, -fy*Y/Z²]]
            double j00 = p00 * invZ;                    // ∂u/∂X = fx/Z
            double j02 = -p00 * camX * invZ * invZ;     // ∂u/∂Z = -fx*X/Z²
            double j11 = p11 * invZ;                    // ∂v/∂Y = fy/Z
            double j12 = -p11 * camY * invZ * invZ;     // ∂v/∂Z = -fy*Y/Z²

            // Full 2D covariance projection: Σ_2D = J * Σ_3D * J^T
            // For symmetric 3x3 Σ with elements [c00,c01,c02; c01,c11,c12; c02,c12,c22]:
            //   Σ_2D[0,0] = j00²*c00 + 2*j00*j02*c02 + j02²*c22
            //   Σ_2D[0,1] = j00*j11*c01 + j00*j12*c02 + j02*j11*c12 + j02*j12*c22
            //   Σ_2D[1,1] = j11²*c11 + 2*j11*j12*c12 + j12²*c22
            double cov2D_00 = j00 * j00 * c00 + 2.0 * j00 * j02 * c02 + j02 * j02 * c22;
            double cov2D_01 = j00 * j11 * c01 + j00 * j12 * c02 + j02 * j11 * c12 + j02 * j12 * c22;
            double cov2D_11 = j11 * j11 * c11 + 2.0 * j11 * j12 * c12 + j12 * j12 * c22;

            // Add small regularization for numerical stability
            cov2D_00 += 0.3;
            cov2D_11 += 0.3;

            // Store results
            means2DData[i * 2] = numOps.FromDouble(screenX);
            means2DData[i * 2 + 1] = numOps.FromDouble(screenY);
            cov2DData[i * 3] = numOps.FromDouble(cov2D_00);
            cov2DData[i * 3 + 1] = numOps.FromDouble(cov2D_01);
            cov2DData[i * 3 + 2] = numOps.FromDouble(cov2D_11);
            depthsData[i] = numOps.FromDouble(camZ);
            visibleData[i] = true;
        });

        means2D = new Tensor<T>(means2DData, [numGaussians, 2]);
        covariances2D = new Tensor<T>(cov2DData, [numGaussians, 3]);
        depths = new Tensor<T>(depthsData, [numGaussians]);
        visible = new Tensor<bool>(visibleData, [numGaussians]);
    }

    #endregion

    #region Gaussian Rasterization

    /// <summary>
    /// Rasterizes 2D Gaussians onto an image using alpha blending.
    /// Uses tiled rendering for efficiency.
    /// </summary>
    public static Tensor<T> RasterizeGaussians<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        int tileSize = 16)
    {
        if (means2D == null) throw new ArgumentNullException(nameof(means2D));
        if (covariances2D == null) throw new ArgumentNullException(nameof(covariances2D));
        if (colors == null) throw new ArgumentNullException(nameof(colors));
        if (opacities == null) throw new ArgumentNullException(nameof(opacities));
        if (depths == null) throw new ArgumentNullException(nameof(depths));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = means2D.Shape[0];
        int numChannels = colors.Shape[1];

        // Validate against integer overflow for large datasets
        const int maxSafeGaussians = int.MaxValue / 9;
        if (numGaussians > maxSafeGaussians)
        {
            throw new ArgumentException(
                $"Number of Gaussians ({numGaussians}) exceeds maximum safe limit ({maxSafeGaussians}) to prevent integer overflow.",
                nameof(means2D));
        }

        // Validate image dimensions to prevent overflow
        if ((long)imageHeight * imageWidth * numChannels > int.MaxValue)
        {
            throw new ArgumentException(
                $"Image dimensions ({imageWidth}x{imageHeight}x{numChannels}) exceed maximum safe array size.",
                nameof(imageWidth));
        }

        var image = new double[imageHeight * imageWidth * numChannels];

        // Sort Gaussians by depth (front to back for alpha blending)
        var sortedIndices = Enumerable.Range(0, numGaussians)
            .OrderBy(i => numOps.ToDouble(depths.GetFlat(i)))
            .ToArray();

        // Precompute Gaussian parameters
        var gaussianParams = new (double mx, double my, double invA, double invB, double invC, double opacity, double[] color, double radius)[numGaussians];

        for (int i = 0; i < numGaussians; i++)
        {
            int idx = sortedIndices[i];
            double mx = numOps.ToDouble(means2D.GetFlat(idx * 2));
            double my = numOps.ToDouble(means2D.GetFlat(idx * 2 + 1));

            double a = numOps.ToDouble(covariances2D.GetFlat(idx * 3));
            double b = numOps.ToDouble(covariances2D.GetFlat(idx * 3 + 1));
            double c = numOps.ToDouble(covariances2D.GetFlat(idx * 3 + 2));

            // Invert 2x2 covariance for evaluation
            double det = a * c - b * b;
            if (det <= 1e-10)
            {
                det = 1e-10;
            }
            double invDet = 1.0 / det;
            double invA = c * invDet;
            double invB = -b * invDet;
            double invC = a * invDet;

            double opacity = Sigmoid(numOps.ToDouble(opacities.GetFlat(idx)));

            var color = new double[numChannels];
            for (int ch = 0; ch < numChannels; ch++)
            {
                color[ch] = numOps.ToDouble(colors.GetFlat(idx * numChannels + ch));
            }

            // Compute approximate radius (3 sigma)
            double radius = 3.0 * Math.Sqrt(Math.Max(a, c));

            gaussianParams[i] = (mx, my, invA, invB, invC, opacity, color, radius);
        }

        // Tiled rasterization
        int numTilesX = (imageWidth + tileSize - 1) / tileSize;
        int numTilesY = (imageHeight + tileSize - 1) / tileSize;

        Parallel.For(0, numTilesX * numTilesY, tileIdx =>
        {
            int tileX = tileIdx % numTilesX;
            int tileY = tileIdx / numTilesX;

            int startX = tileX * tileSize;
            int startY = tileY * tileSize;
            int endX = Math.Min(startX + tileSize, imageWidth);
            int endY = Math.Min(startY + tileSize, imageHeight);

            // For each pixel in tile
            for (int py = startY; py < endY; py++)
            {
                for (int px = startX; px < endX; px++)
                {
                    int pixelIdx = py * imageWidth + px;
                    double transmittance = 1.0;
                    var accumColor = new double[numChannels];

                    // Process Gaussians front to back
                    for (int gi = 0; gi < numGaussians; gi++)
                    {
                        var (mx, my, invA, invB, invC, opacity, color, radius) = gaussianParams[gi];

                        // Quick bounds check
                        double dx = px - mx;
                        double dy = py - my;
                        if (Math.Abs(dx) > radius || Math.Abs(dy) > radius)
                            continue;

                        // Evaluate Gaussian: exp(-0.5 * (dx, dy)^T * Σ^(-1) * (dx, dy))
                        double exponent = -0.5 * (invA * dx * dx + 2 * invB * dx * dy + invC * dy * dy);
                        if (exponent < -20.0) continue;

                        double weight = Math.Exp(exponent);
                        double alpha = opacity * weight;

                        if (alpha < 1e-6) continue;

                        // Alpha blending
                        for (int ch = 0; ch < numChannels; ch++)
                        {
                            accumColor[ch] += transmittance * alpha * color[ch];
                        }

                        transmittance *= (1.0 - alpha);
                        if (transmittance < 1e-4) break;
                    }

                    // Store result
                    for (int ch = 0; ch < numChannels; ch++)
                    {
                        image[pixelIdx * numChannels + ch] = Clamp01(accumColor[ch]);
                    }
                }
            }
        });

        // Convert to tensor
        var imageData = new T[imageHeight * imageWidth * numChannels];
        for (int i = 0; i < imageData.Length; i++)
        {
            imageData[i] = numOps.FromDouble(image[i]);
        }

        return new Tensor<T>(imageData, [imageHeight, imageWidth, numChannels]);
    }

    /// <summary>
    /// Computes the backward pass for Gaussian rasterization.
    /// </summary>
    public static void RasterizeGaussiansBackward<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        Tensor<T> outputGradient,
        int tileSize,
        out Tensor<T> means2DGrad,
        out Tensor<T> covariances2DGrad,
        out Tensor<T> colorsGrad,
        out Tensor<T> opacitiesGrad)
    {
        if (means2D == null) throw new ArgumentNullException(nameof(means2D));
        if (outputGradient == null) throw new ArgumentNullException(nameof(outputGradient));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = means2D.Shape[0];
        int numChannels = colors.Shape[1];

        // Validate against integer overflow for large datasets
        const int maxSafeGaussians = int.MaxValue / 9;
        if (numGaussians > maxSafeGaussians)
        {
            throw new ArgumentException(
                $"Number of Gaussians ({numGaussians}) exceeds maximum safe limit ({maxSafeGaussians}) to prevent integer overflow in index calculations.",
                nameof(means2D));
        }

        var means2DGradData = new double[numGaussians * 2];
        var cov2DGradData = new double[numGaussians * 3];
        var colorsGradData = new double[numGaussians * numChannels];
        var opacitiesGradData = new double[numGaussians];

        // Sort Gaussians by depth
        var sortedIndices = Enumerable.Range(0, numGaussians)
            .OrderBy(i => numOps.ToDouble(depths.GetFlat(i)))
            .ToArray();

        // Simplified backward pass - accumulate gradients per Gaussian
        var lockObjects = new object[numGaussians];
        for (int i = 0; i < numGaussians; i++)
        {
            lockObjects[i] = new object();
        }

        // Tiled gradient computation
        int numTilesX = (imageWidth + tileSize - 1) / tileSize;
        int numTilesY = (imageHeight + tileSize - 1) / tileSize;

        Parallel.For(0, numTilesX * numTilesY, tileIdx =>
        {
            int tileX = tileIdx % numTilesX;
            int tileY = tileIdx / numTilesX;

            int startX = tileX * tileSize;
            int startY = tileY * tileSize;
            int endX = Math.Min(startX + tileSize, imageWidth);
            int endY = Math.Min(startY + tileSize, imageHeight);

            for (int py = startY; py < endY; py++)
            {
                for (int px = startX; px < endX; px++)
                {
                    int pixelIdx = py * imageWidth + px;

                    // Get output gradient for this pixel
                    var pixelGrad = new double[numChannels];
                    for (int ch = 0; ch < numChannels; ch++)
                    {
                        pixelGrad[ch] = numOps.ToDouble(outputGradient.GetFlat(pixelIdx * numChannels + ch));
                    }

                    // Compute gradients for each Gaussian affecting this pixel
                    double transmittance = 1.0;

                    for (int gi = 0; gi < numGaussians; gi++)
                    {
                        int idx = sortedIndices[gi];

                        double mx = numOps.ToDouble(means2D.GetFlat(idx * 2));
                        double my = numOps.ToDouble(means2D.GetFlat(idx * 2 + 1));

                        double a = numOps.ToDouble(covariances2D.GetFlat(idx * 3));
                        double b = numOps.ToDouble(covariances2D.GetFlat(idx * 3 + 1));
                        double c = numOps.ToDouble(covariances2D.GetFlat(idx * 3 + 2));

                        double det = a * c - b * b;
                        if (det <= 1e-10) continue;

                        double invDet = 1.0 / det;
                        double invA = c * invDet;
                        double invB = -b * invDet;
                        double invC = a * invDet;

                        double dx = px - mx;
                        double dy = py - my;

                        double radius = 3.0 * Math.Sqrt(Math.Max(a, c));
                        if (Math.Abs(dx) > radius || Math.Abs(dy) > radius)
                            continue;

                        double exponent = -0.5 * (invA * dx * dx + 2 * invB * dx * dy + invC * dy * dy);
                        if (exponent < -20.0) continue;

                        double weight = Math.Exp(exponent);
                        double opacityParam = numOps.ToDouble(opacities.GetFlat(idx));
                        double opacity = Sigmoid(opacityParam);
                        double alpha = opacity * weight;

                        if (alpha < 1e-6) continue;

                        // Gradient computations
                        double dL_dalpha = 0.0;
                        for (int ch = 0; ch < numChannels; ch++)
                        {
                            double colorVal = numOps.ToDouble(colors.GetFlat(idx * numChannels + ch));
                            dL_dalpha += pixelGrad[ch] * transmittance * colorVal;
                        }

                        // Gradient w.r.t. color
                        lock (lockObjects[idx])
                        {
                            for (int ch = 0; ch < numChannels; ch++)
                            {
                                colorsGradData[idx * numChannels + ch] += pixelGrad[ch] * transmittance * alpha;
                            }

                            // Gradient w.r.t. opacity (through sigmoid)
                            double dAlpha_dOpacity = weight * opacity * (1.0 - opacity);
                            opacitiesGradData[idx] += dL_dalpha * dAlpha_dOpacity;

                            // Gradient w.r.t. mean (through weight)
                            double dWeight_dMx = weight * (invA * dx + invB * dy);
                            double dWeight_dMy = weight * (invB * dx + invC * dy);
                            means2DGradData[idx * 2] += dL_dalpha * opacity * dWeight_dMx;
                            means2DGradData[idx * 2 + 1] += dL_dalpha * opacity * dWeight_dMy;

                            // Gradient w.r.t. 2D covariance [a, b, c] where matrix is [[a,b],[b,c]]
                            // Chain rule through: exponent = -0.5 * (invA*dx^2 + 2*invB*dx*dy + invC*dy^2)
                            // where invA = c/det, invB = -b/det, invC = a/det, det = a*c - b^2
                            double dL_dWeight = dL_dalpha * opacity;
                            double dL_dExp = dL_dWeight * weight;

                            // Gradients w.r.t. inverse covariance elements
                            double dL_dInvA = dL_dExp * (-0.5) * dx * dx;
                            double dL_dInvB = dL_dExp * (-1.0) * dx * dy;
                            double dL_dInvC = dL_dExp * (-0.5) * dy * dy;

                            // Gradients w.r.t. original covariance elements via inverse derivatives
                            double invDet2 = invDet * invDet;
                            double dL_da = invDet2 * (-dL_dInvA * c * c + dL_dInvB * b * c - dL_dInvC * b * b);
                            double dL_db = invDet2 * (2.0 * dL_dInvA * b * c + dL_dInvB * (-a * c - b * b) + 2.0 * dL_dInvC * a * b);
                            double dL_dc = invDet2 * (-dL_dInvA * b * b + dL_dInvB * a * b - dL_dInvC * a * a);

                            cov2DGradData[idx * 3] += dL_da;
                            cov2DGradData[idx * 3 + 1] += dL_db;
                            cov2DGradData[idx * 3 + 2] += dL_dc;
                        }

                        transmittance *= (1.0 - alpha);
                        if (transmittance < 1e-4) break;
                    }
                }
            }
        });

        // Convert to tensors
        var m2DGrad = new T[numGaussians * 2];
        var c2DGrad = new T[numGaussians * 3];
        var cGrad = new T[numGaussians * numChannels];
        var oGrad = new T[numGaussians];

        for (int i = 0; i < numGaussians; i++)
        {
            m2DGrad[i * 2] = numOps.FromDouble(means2DGradData[i * 2]);
            m2DGrad[i * 2 + 1] = numOps.FromDouble(means2DGradData[i * 2 + 1]);
            c2DGrad[i * 3] = numOps.FromDouble(cov2DGradData[i * 3]);
            c2DGrad[i * 3 + 1] = numOps.FromDouble(cov2DGradData[i * 3 + 1]);
            c2DGrad[i * 3 + 2] = numOps.FromDouble(cov2DGradData[i * 3 + 2]);
            oGrad[i] = numOps.FromDouble(opacitiesGradData[i]);
        }

        for (int i = 0; i < numGaussians * numChannels; i++)
        {
            cGrad[i] = numOps.FromDouble(colorsGradData[i]);
        }

        means2DGrad = new Tensor<T>(m2DGrad, [numGaussians, 2]);
        covariances2DGrad = new Tensor<T>(c2DGrad, [numGaussians, 3]);
        colorsGrad = new Tensor<T>(cGrad, [numGaussians, numChannels]);
        opacitiesGrad = new Tensor<T>(oGrad, [numGaussians]);
    }

    #endregion

    #region Spherical Harmonics

    /// <summary>
    /// Evaluates spherical harmonics for view-dependent color.
    /// </summary>
    public static Tensor<T> EvaluateSphericalHarmonics<T>(
        Tensor<T> shCoefficients,
        Tensor<T> viewDirections,
        int degree)
    {
        if (shCoefficients == null) throw new ArgumentNullException(nameof(shCoefficients));
        if (viewDirections == null) throw new ArgumentNullException(nameof(viewDirections));
        if (degree < 0 || degree > 3)
            throw new ArgumentOutOfRangeException(nameof(degree), "Degree must be between 0 and 3.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = shCoefficients.Shape[0];
        int basisCount = (degree + 1) * (degree + 1);
        int numChannels = shCoefficients.Shape[2]; // Typically 3 for RGB

        bool broadcastDir = viewDirections.Shape[0] == 1;
        var colors = new T[numGaussians * numChannels];

        Parallel.For(0, numGaussians, i =>
        {
            int dirIdx = broadcastDir ? 0 : i;
            double dx = numOps.ToDouble(viewDirections.GetFlat(dirIdx * 3));
            double dy = numOps.ToDouble(viewDirections.GetFlat(dirIdx * 3 + 1));
            double dz = numOps.ToDouble(viewDirections.GetFlat(dirIdx * 3 + 2));

            // Normalize direction
            double norm = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            if (norm > 0)
            {
                double inv = 1.0 / norm;
                dx *= inv;
                dy *= inv;
                dz *= inv;
            }

            // Compute SH basis functions
            var basis = ComputeSphericalHarmonicsBasis(dx, dy, dz, degree);

            // Evaluate for each color channel
            for (int ch = 0; ch < numChannels; ch++)
            {
                double color = 0.0;
                for (int b = 0; b < basisCount; b++)
                {
                    double coeff = numOps.ToDouble(shCoefficients.GetFlat(i * basisCount * numChannels + b * numChannels + ch));
                    color += coeff * basis[b];
                }
                colors[i * numChannels + ch] = numOps.FromDouble(Clamp01(color));
            }
        });

        return new Tensor<T>(colors, [numGaussians, numChannels]);
    }

    /// <summary>
    /// Computes the backward pass for spherical harmonics evaluation.
    /// </summary>
    public static Tensor<T> EvaluateSphericalHarmonicsBackward<T>(
        Tensor<T> shCoefficients,
        Tensor<T> viewDirections,
        int degree,
        Tensor<T> outputGradient)
    {
        if (shCoefficients == null) throw new ArgumentNullException(nameof(shCoefficients));
        if (viewDirections == null) throw new ArgumentNullException(nameof(viewDirections));
        if (outputGradient == null) throw new ArgumentNullException(nameof(outputGradient));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = shCoefficients.Shape[0];
        int basisCount = (degree + 1) * (degree + 1);
        int numChannels = shCoefficients.Shape[2];

        bool broadcastDir = viewDirections.Shape[0] == 1;
        var shGrad = new T[numGaussians * basisCount * numChannels];

        Parallel.For(0, numGaussians, i =>
        {
            int dirIdx = broadcastDir ? 0 : i;
            double dx = numOps.ToDouble(viewDirections.GetFlat(dirIdx * 3));
            double dy = numOps.ToDouble(viewDirections.GetFlat(dirIdx * 3 + 1));
            double dz = numOps.ToDouble(viewDirections.GetFlat(dirIdx * 3 + 2));

            double norm = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            if (norm > 0)
            {
                double inv = 1.0 / norm;
                dx *= inv;
                dy *= inv;
                dz *= inv;
            }

            var basis = ComputeSphericalHarmonicsBasis(dx, dy, dz, degree);

            for (int ch = 0; ch < numChannels; ch++)
            {
                // Recompute pre-clamped color to check if it was saturated
                double preclampColor = 0.0;
                for (int b = 0; b < basisCount; b++)
                {
                    double coeff = numOps.ToDouble(shCoefficients.GetFlat(i * basisCount * numChannels + b * numChannels + ch));
                    preclampColor += coeff * basis[b];
                }

                // Mask gradient if color was clamped (gradient is 0 outside [0,1])
                double colorGrad = numOps.ToDouble(outputGradient.GetFlat(i * numChannels + ch));
                if (preclampColor < 0.0 || preclampColor > 1.0)
                {
                    colorGrad = 0.0;
                }

                for (int b = 0; b < basisCount; b++)
                {
                    // dL/d(coeff) = dL/dcolor * basis
                    shGrad[i * basisCount * numChannels + b * numChannels + ch] =
                        numOps.FromDouble(colorGrad * basis[b]);
                }
            }
        });

        return new Tensor<T>(shGrad, shCoefficients.Shape);
    }

    /// <summary>
    /// Computes 3D covariance matrices from rotation quaternions and scale vectors.
    /// </summary>
    public static Tensor<T> ComputeGaussianCovariance<T>(Tensor<T> rotations, Tensor<T> scales)
    {
        if (rotations == null) throw new ArgumentNullException(nameof(rotations));
        if (scales == null) throw new ArgumentNullException(nameof(scales));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = rotations.Shape[0];

        // Output: upper triangular covariance [N, 6] = c00, c01, c02, c11, c12, c22
        var covariances = new T[numGaussians * 6];

        Parallel.For(0, numGaussians, i =>
        {
            // Get quaternion (w, x, y, z)
            double qw = numOps.ToDouble(rotations.GetFlat(i * 4));
            double qx = numOps.ToDouble(rotations.GetFlat(i * 4 + 1));
            double qy = numOps.ToDouble(rotations.GetFlat(i * 4 + 2));
            double qz = numOps.ToDouble(rotations.GetFlat(i * 4 + 3));

            // Normalize quaternion
            double qNorm = Math.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            if (qNorm > 0)
            {
                double inv = 1.0 / qNorm;
                qw *= inv;
                qx *= inv;
                qy *= inv;
                qz *= inv;
            }

            // Convert quaternion to rotation matrix
            double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
            double r01 = 2.0 * (qx * qy - qw * qz);
            double r02 = 2.0 * (qx * qz + qw * qy);
            double r10 = 2.0 * (qx * qy + qw * qz);
            double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
            double r12 = 2.0 * (qy * qz - qw * qx);
            double r20 = 2.0 * (qx * qz - qw * qy);
            double r21 = 2.0 * (qy * qz + qw * qx);
            double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

            // Get scales
            double sx = Math.Max(1e-6, Math.Abs(numOps.ToDouble(scales.GetFlat(i * 3))));
            double sy = Math.Max(1e-6, Math.Abs(numOps.ToDouble(scales.GetFlat(i * 3 + 1))));
            double sz = Math.Max(1e-6, Math.Abs(numOps.ToDouble(scales.GetFlat(i * 3 + 2))));

            // Compute covariance: Σ = R * S * S^T * R^T
            // where S is diagonal scale matrix
            double sx2 = sx * sx;
            double sy2 = sy * sy;
            double sz2 = sz * sz;

            // M = R * S² (column-wise multiplication)
            double m00 = r00 * sx2;
            double m01 = r01 * sy2;
            double m02 = r02 * sz2;
            double m10 = r10 * sx2;
            double m11 = r11 * sy2;
            double m12 = r12 * sz2;
            double m20 = r20 * sx2;
            double m21 = r21 * sy2;
            double m22 = r22 * sz2;

            // Σ = M * R^T
            double c00 = m00 * r00 + m01 * r01 + m02 * r02;
            double c01 = m00 * r10 + m01 * r11 + m02 * r12;
            double c02 = m00 * r20 + m01 * r21 + m02 * r22;
            double c11 = m10 * r10 + m11 * r11 + m12 * r12;
            double c12 = m10 * r20 + m11 * r21 + m12 * r22;
            double c22 = m20 * r20 + m21 * r21 + m22 * r22;

            // Store upper triangular
            int offset = i * 6;
            covariances[offset] = numOps.FromDouble(c00);
            covariances[offset + 1] = numOps.FromDouble(c01);
            covariances[offset + 2] = numOps.FromDouble(c02);
            covariances[offset + 3] = numOps.FromDouble(c11);
            covariances[offset + 4] = numOps.FromDouble(c12);
            covariances[offset + 5] = numOps.FromDouble(c22);
        });

        return new Tensor<T>(covariances, [numGaussians, 6]);
    }

    /// <summary>
    /// Computes the backward pass for Gaussian covariance computation.
    /// Implements full analytical gradients for both rotation quaternions and scale vectors.
    /// </summary>
    /// <remarks>
    /// The covariance matrix Σ is computed as Σ = R * S² * R^T where R is the rotation matrix
    /// derived from the quaternion and S is a diagonal scale matrix. This method computes
    /// the gradients dL/dq and dL/ds given dL/dΣ using the chain rule.
    /// </remarks>
    public static void ComputeGaussianCovarianceBackward<T>(
        Tensor<T> rotations,
        Tensor<T> scales,
        Tensor<T> covarianceGradient,
        out Tensor<T> rotationsGrad,
        out Tensor<T> scalesGrad)
    {
        if (rotations == null) throw new ArgumentNullException(nameof(rotations));
        if (scales == null) throw new ArgumentNullException(nameof(scales));
        if (covarianceGradient == null) throw new ArgumentNullException(nameof(covarianceGradient));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numGaussians = rotations.Shape[0];

        var rotGrad = new T[numGaussians * 4];
        var scaleGrad = new T[numGaussians * 3];

        Parallel.For(0, numGaussians, i =>
        {
            // Get quaternion (w, x, y, z) and normalize in a single expression
            double rawQw = numOps.ToDouble(rotations.GetFlat(i * 4));
            double rawQx = numOps.ToDouble(rotations.GetFlat(i * 4 + 1));
            double rawQy = numOps.ToDouble(rotations.GetFlat(i * 4 + 2));
            double rawQz = numOps.ToDouble(rotations.GetFlat(i * 4 + 3));

            double qNorm = Math.Sqrt(rawQw * rawQw + rawQx * rawQx + rawQy * rawQy + rawQz * rawQz);
            double normInv = qNorm > 1e-10 ? 1.0 / qNorm : 1.0;
            double qw = rawQw * normInv;
            double qx = rawQx * normInv;
            double qy = rawQy * normInv;
            double qz = rawQz * normInv;

            // Get scales with minimum threshold
            double sx = Math.Max(1e-6, Math.Abs(numOps.ToDouble(scales.GetFlat(i * 3))));
            double sy = Math.Max(1e-6, Math.Abs(numOps.ToDouble(scales.GetFlat(i * 3 + 1))));
            double sz = Math.Max(1e-6, Math.Abs(numOps.ToDouble(scales.GetFlat(i * 3 + 2))));

            double sx2 = sx * sx;
            double sy2 = sy * sy;
            double sz2 = sz * sz;

            // Get covariance gradients (symmetric, so gc01 = gc10, etc.)
            int offset = i * 6;
            double gc00 = numOps.ToDouble(covarianceGradient.GetFlat(offset));
            double gc01 = numOps.ToDouble(covarianceGradient.GetFlat(offset + 1));
            double gc02 = numOps.ToDouble(covarianceGradient.GetFlat(offset + 2));
            double gc11 = numOps.ToDouble(covarianceGradient.GetFlat(offset + 3));
            double gc12 = numOps.ToDouble(covarianceGradient.GetFlat(offset + 4));
            double gc22 = numOps.ToDouble(covarianceGradient.GetFlat(offset + 5));

            // Rotation matrix elements from quaternion
            double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
            double r01 = 2.0 * (qx * qy - qw * qz);
            double r02 = 2.0 * (qx * qz + qw * qy);
            double r10 = 2.0 * (qx * qy + qw * qz);
            double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
            double r12 = 2.0 * (qy * qz - qw * qx);
            double r20 = 2.0 * (qx * qz - qw * qy);
            double r21 = 2.0 * (qy * qz + qw * qx);
            double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

            // Compute scale gradients: dL/ds_k = 2 * s_k * sum_ij(dL/dΣ_ij * R_ik * R_jk)
            // For scale x (k=0): dL/ds_x = 2*s_x * (gc00*r00*r00 + 2*gc01*r00*r10 + 2*gc02*r00*r20 + gc11*r10*r10 + 2*gc12*r10*r20 + gc22*r20*r20)
            double dL_dsx = 2.0 * sx * (
                gc00 * r00 * r00 +
                2.0 * gc01 * r00 * r10 +
                2.0 * gc02 * r00 * r20 +
                gc11 * r10 * r10 +
                2.0 * gc12 * r10 * r20 +
                gc22 * r20 * r20);

            double dL_dsy = 2.0 * sy * (
                gc00 * r01 * r01 +
                2.0 * gc01 * r01 * r11 +
                2.0 * gc02 * r01 * r21 +
                gc11 * r11 * r11 +
                2.0 * gc12 * r11 * r21 +
                gc22 * r21 * r21);

            double dL_dsz = 2.0 * sz * (
                gc00 * r02 * r02 +
                2.0 * gc01 * r02 * r12 +
                2.0 * gc02 * r02 * r22 +
                gc11 * r12 * r12 +
                2.0 * gc12 * r12 * r22 +
                gc22 * r22 * r22);

            scaleGrad[i * 3] = numOps.FromDouble(dL_dsx);
            scaleGrad[i * 3 + 1] = numOps.FromDouble(dL_dsy);
            scaleGrad[i * 3 + 2] = numOps.FromDouble(dL_dsz);

            // Compute rotation gradients using chain rule through rotation matrix
            // dL/dq = sum_ij(dL/dΣ_ij * dΣ_ij/dR * dR/dq)
            // For quaternion components, we need to compute dR/dq_k for each rotation matrix element

            // Compute M = R * S² (column-wise multiplication)
            double m00 = r00 * sx2; double m01 = r01 * sy2; double m02 = r02 * sz2;
            double m10 = r10 * sx2; double m11 = r11 * sy2; double m12 = r12 * sz2;
            double m20 = r20 * sx2; double m21 = r21 * sy2; double m22 = r22 * sz2;

            // Gradient of covariance w.r.t. rotation matrix elements
            // Σ = M * R^T, so dL/dM_ij = sum_k(dL/dΣ_ik * R_jk)
            // And dL/dR_ij (through M) = S_j² * sum_k(dL/dΣ_ik * R_jk)
            // Plus dL/dR_ij (through R^T) = sum_k(dL/dΣ_ki * M_kj)

            // Combined gradient for rotation matrix
            // dL/dR_ij = dL/dΣ * (M * δR)^T + (δR * M)^T where we accumulate through both paths
            double dL_dr00 = sx2 * (gc00 * r00 + gc01 * r10 + gc02 * r20) + (gc00 * m00 + gc01 * m10 + gc02 * m20);
            double dL_dr01 = sy2 * (gc00 * r01 + gc01 * r11 + gc02 * r21) + (gc00 * m01 + gc01 * m11 + gc02 * m21);
            double dL_dr02 = sz2 * (gc00 * r02 + gc01 * r12 + gc02 * r22) + (gc00 * m02 + gc01 * m12 + gc02 * m22);
            double dL_dr10 = sx2 * (gc01 * r00 + gc11 * r10 + gc12 * r20) + (gc01 * m00 + gc11 * m10 + gc12 * m20);
            double dL_dr11 = sy2 * (gc01 * r01 + gc11 * r11 + gc12 * r21) + (gc01 * m01 + gc11 * m11 + gc12 * m21);
            double dL_dr12 = sz2 * (gc01 * r02 + gc11 * r12 + gc12 * r22) + (gc01 * m02 + gc11 * m12 + gc12 * m22);
            double dL_dr20 = sx2 * (gc02 * r00 + gc12 * r10 + gc22 * r20) + (gc02 * m00 + gc12 * m10 + gc22 * m20);
            double dL_dr21 = sy2 * (gc02 * r01 + gc12 * r11 + gc22 * r21) + (gc02 * m01 + gc12 * m11 + gc22 * m21);
            double dL_dr22 = sz2 * (gc02 * r02 + gc12 * r12 + gc22 * r22) + (gc02 * m02 + gc12 * m12 + gc22 * m22);

            // Derivatives of rotation matrix elements w.r.t. quaternion components
            // r00 = 1 - 2(qy² + qz²), r01 = 2(qx*qy - qw*qz), r02 = 2(qx*qz + qw*qy), etc.

            // dL/dqw
            double dL_dqw =
                dL_dr01 * (-2.0 * qz) +  // dr01/dqw = -2qz
                dL_dr02 * (2.0 * qy) +   // dr02/dqw = 2qy
                dL_dr10 * (2.0 * qz) +   // dr10/dqw = 2qz
                dL_dr12 * (-2.0 * qx) +  // dr12/dqw = -2qx
                dL_dr20 * (-2.0 * qy) +  // dr20/dqw = -2qy
                dL_dr21 * (2.0 * qx);    // dr21/dqw = 2qx

            // dL/dqx
            double dL_dqx =
                dL_dr01 * (2.0 * qy) +   // dr01/dqx = 2qy
                dL_dr02 * (2.0 * qz) +   // dr02/dqx = 2qz
                dL_dr10 * (2.0 * qy) +   // dr10/dqx = 2qy
                dL_dr11 * (-4.0 * qx) +  // dr11/dqx = -4qx
                dL_dr12 * (-2.0 * qw) +  // dr12/dqx = -2qw
                dL_dr20 * (2.0 * qz) +   // dr20/dqx = 2qz
                dL_dr21 * (2.0 * qw) +   // dr21/dqx = 2qw
                dL_dr22 * (-4.0 * qx);   // dr22/dqx = -4qx

            // dL/dqy
            double dL_dqy =
                dL_dr00 * (-4.0 * qy) +  // dr00/dqy = -4qy
                dL_dr01 * (2.0 * qx) +   // dr01/dqy = 2qx
                dL_dr02 * (2.0 * qw) +   // dr02/dqy = 2qw
                dL_dr10 * (2.0 * qx) +   // dr10/dqy = 2qx
                dL_dr12 * (2.0 * qz) +   // dr12/dqy = 2qz
                dL_dr20 * (-2.0 * qw) +  // dr20/dqy = -2qw
                dL_dr21 * (2.0 * qz) +   // dr21/dqy = 2qz
                dL_dr22 * (-4.0 * qy);   // dr22/dqy = -4qy

            // dL/dqz
            double dL_dqz =
                dL_dr00 * (-4.0 * qz) +  // dr00/dqz = -4qz
                dL_dr01 * (-2.0 * qw) +  // dr01/dqz = -2qw
                dL_dr02 * (2.0 * qx) +   // dr02/dqz = 2qx
                dL_dr10 * (2.0 * qw) +   // dr10/dqz = 2qw
                dL_dr11 * (-4.0 * qz) +  // dr11/dqz = -4qz
                dL_dr12 * (2.0 * qy) +   // dr12/dqz = 2qy
                dL_dr20 * (2.0 * qx) +   // dr20/dqz = 2qx
                dL_dr21 * (2.0 * qy);    // dr21/dqz = 2qy

            rotGrad[i * 4] = numOps.FromDouble(dL_dqw);
            rotGrad[i * 4 + 1] = numOps.FromDouble(dL_dqx);
            rotGrad[i * 4 + 2] = numOps.FromDouble(dL_dqy);
            rotGrad[i * 4 + 3] = numOps.FromDouble(dL_dqz);
        });

        rotationsGrad = new Tensor<T>(rotGrad, [numGaussians, 4]);
        scalesGrad = new Tensor<T>(scaleGrad, [numGaussians, 3]);
    }

    #endregion

    #region Helper Methods

    private static double[] ComputeSphericalHarmonicsBasis(double x, double y, double z, int degree)
    {
        int count = (degree + 1) * (degree + 1);
        var basis = new double[count];

        // Band 0 (constant)
        basis[0] = 0.282095;

        if (degree >= 1)
        {
            // Band 1 (linear)
            basis[1] = 0.488603 * y;
            basis[2] = 0.488603 * z;
            basis[3] = 0.488603 * x;
        }

        if (degree >= 2)
        {
            // Band 2 (quadratic)
            basis[4] = 1.092548 * x * y;
            basis[5] = 1.092548 * y * z;
            basis[6] = 0.315392 * (3.0 * z * z - 1.0);
            basis[7] = 1.092548 * x * z;
            basis[8] = 0.546274 * (x * x - y * y);
        }

        if (degree >= 3)
        {
            // Band 3 (cubic)
            basis[9] = 0.590044 * y * (3.0 * x * x - y * y);
            basis[10] = 2.890611 * x * y * z;
            basis[11] = 0.457046 * y * (5.0 * z * z - 1.0);
            basis[12] = 0.373176 * z * (5.0 * z * z - 3.0);
            basis[13] = 0.457046 * x * (5.0 * z * z - 1.0);
            basis[14] = 1.445306 * z * (x * x - y * y);
            basis[15] = 0.590044 * x * (x * x - 3.0 * y * y);
        }

        return basis;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private static double Clamp01(double value)
    {
        if (value <= 0.0) return 0.0;
        if (value >= 1.0) return 1.0;
        return value;
    }

    #endregion
}
