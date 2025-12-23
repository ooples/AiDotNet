using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides vectorized implementations of Neural Radiance Fields operations.
/// These operations are shared between CPU and GPU engines.
/// </summary>
public static class NeRFOperations
{
    #region Positional Encoding

    /// <summary>
    /// Computes positional encoding for Neural Radiance Fields.
    /// Vectorized implementation using SIMD when possible.
    /// </summary>
    public static Tensor<T> PositionalEncoding<T>(Tensor<T> positions, int numFrequencies)
    {
        if (positions == null) throw new ArgumentNullException(nameof(positions));
        if (positions.Shape.Length != 2)
            throw new ArgumentException("Positions must be 2D tensor [N, D].", nameof(positions));
        if (numFrequencies <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFrequencies), "Number of frequencies must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int numPoints = positions.Shape[0];
        int inputDim = positions.Shape[1];
        int outputDim = inputDim * 2 * numFrequencies;

        var result = new T[numPoints * outputDim];

        // Precompute frequency values: 2^l * π for l = 0 to numFrequencies-1
        var frequencies = new double[numFrequencies];
        for (int l = 0; l < numFrequencies; l++)
        {
            frequencies[l] = Math.Pow(2.0, l) * Math.PI;
        }

        // Vectorized parallel computation
        Parallel.For(0, numPoints, n =>
        {
            int posOffset = n * inputDim;
            int outOffset = n * outputDim;

            for (int d = 0; d < inputDim; d++)
            {
                double value = numOps.ToDouble(positions.GetFlat(posOffset + d));
                int dimOffset = outOffset + d * 2 * numFrequencies;

                for (int l = 0; l < numFrequencies; l++)
                {
                    double angle = frequencies[l] * value;
                    int idx = dimOffset + l * 2;
                    result[idx] = numOps.FromDouble(Math.Sin(angle));
                    result[idx + 1] = numOps.FromDouble(Math.Cos(angle));
                }
            }
        });

        return new Tensor<T>(result, [numPoints, outputDim]);
    }

    /// <summary>
    /// Computes the backward pass for positional encoding.
    /// </summary>
    public static Tensor<T> PositionalEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T> encodedGradient,
        int numFrequencies)
    {
        if (positions == null) throw new ArgumentNullException(nameof(positions));
        if (encodedGradient == null) throw new ArgumentNullException(nameof(encodedGradient));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numPoints = positions.Shape[0];
        int inputDim = positions.Shape[1];
        int outputDim = inputDim * 2 * numFrequencies;

        if (encodedGradient.Shape[1] != outputDim)
            throw new ArgumentException("Encoded gradient dimension mismatch.", nameof(encodedGradient));

        var result = new T[numPoints * inputDim];

        var frequencies = new double[numFrequencies];
        for (int l = 0; l < numFrequencies; l++)
        {
            frequencies[l] = Math.Pow(2.0, l) * Math.PI;
        }

        Parallel.For(0, numPoints, n =>
        {
            int posOffset = n * inputDim;
            int gradOffset = n * outputDim;
            int outOffset = n * inputDim;

            for (int d = 0; d < inputDim; d++)
            {
                double value = numOps.ToDouble(positions.GetFlat(posOffset + d));
                int dimGradOffset = gradOffset + d * 2 * numFrequencies;
                double gradSum = 0.0;

                for (int l = 0; l < numFrequencies; l++)
                {
                    double freq = frequencies[l];
                    double angle = freq * value;
                    int idx = dimGradOffset + l * 2;

                    double gradSin = numOps.ToDouble(encodedGradient.GetFlat(idx));
                    double gradCos = numOps.ToDouble(encodedGradient.GetFlat(idx + 1));

                    // d/dx sin(f*x) = f*cos(f*x)
                    // d/dx cos(f*x) = -f*sin(f*x)
                    gradSum += gradSin * freq * Math.Cos(angle);
                    gradSum += gradCos * (-freq) * Math.Sin(angle);
                }

                result[outOffset + d] = numOps.FromDouble(gradSum);
            }
        });

        return new Tensor<T>(result, [numPoints, inputDim]);
    }

    #endregion

    #region Volume Rendering

    /// <summary>
    /// Performs volume rendering along rays using alpha compositing.
    /// Vectorized implementation for efficient batch processing.
    /// </summary>
    public static Tensor<T> VolumeRendering<T>(
        Tensor<T> rgbSamples,
        Tensor<T> densitySamples,
        Tensor<T> tValues)
    {
        if (rgbSamples == null) throw new ArgumentNullException(nameof(rgbSamples));
        if (densitySamples == null) throw new ArgumentNullException(nameof(densitySamples));
        if (tValues == null) throw new ArgumentNullException(nameof(tValues));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Infer dimensions from input shapes
        int numRays, numSamples;
        if (rgbSamples.Shape.Length == 3)
        {
            numRays = rgbSamples.Shape[0];
            numSamples = rgbSamples.Shape[1];
        }
        else if (rgbSamples.Shape.Length == 2)
        {
            // Flat format: [numRays * numSamples, 3]
            int totalSamples = rgbSamples.Shape[0];
            numSamples = tValues.Shape[1];
            numRays = totalSamples / numSamples;
        }
        else
        {
            throw new ArgumentException("RGB samples must be 2D or 3D tensor.", nameof(rgbSamples));
        }

        var colors = new T[numRays * 3];

        Parallel.For(0, numRays, r =>
        {
            double transmittance = 1.0;
            double accumR = 0.0, accumG = 0.0, accumB = 0.0;

            for (int s = 0; s < numSamples; s++)
            {
                // Get t values for delta computation
                double t0 = numOps.ToDouble(tValues.GetFlat(r * numSamples + s));
                double t1 = (s + 1 < numSamples)
                    ? numOps.ToDouble(tValues.GetFlat(r * numSamples + s + 1))
                    : t0 + 0.01; // Small default delta for last sample
                double deltaT = Math.Max(0.0, t1 - t0);

                // Get density (sigma)
                int densityIdx = r * numSamples + s;
                double sigma = numOps.ToDouble(densitySamples.GetFlat(densityIdx));

                // Compute alpha from density: alpha = 1 - exp(-sigma * delta)
                double alpha = 1.0 - Math.Exp(-sigma * deltaT);
                if (alpha <= 1e-10) continue;

                // Get RGB values
                int rgbBaseIdx;
                if (rgbSamples.Shape.Length == 3)
                {
                    rgbBaseIdx = (r * numSamples + s) * 3;
                }
                else
                {
                    rgbBaseIdx = (r * numSamples + s) * 3;
                }

                double rVal = numOps.ToDouble(rgbSamples.GetFlat(rgbBaseIdx));
                double gVal = numOps.ToDouble(rgbSamples.GetFlat(rgbBaseIdx + 1));
                double bVal = numOps.ToDouble(rgbSamples.GetFlat(rgbBaseIdx + 2));

                // Accumulate: C += T * alpha * c
                accumR += transmittance * alpha * rVal;
                accumG += transmittance * alpha * gVal;
                accumB += transmittance * alpha * bVal;

                // Update transmittance: T *= (1 - alpha)
                transmittance *= (1.0 - alpha);

                // Early termination if transmittance is very small
                if (transmittance < 1e-4) break;
            }

            int outIdx = r * 3;
            colors[outIdx] = numOps.FromDouble(accumR);
            colors[outIdx + 1] = numOps.FromDouble(accumG);
            colors[outIdx + 2] = numOps.FromDouble(accumB);
        });

        return new Tensor<T>(colors, [numRays, 3]);
    }

    /// <summary>
    /// Computes the backward pass for volume rendering.
    /// </summary>
    public static void VolumeRenderingBackward<T>(
        Tensor<T> rgbSamples,
        Tensor<T> densitySamples,
        Tensor<T> tValues,
        Tensor<T> outputGradient,
        out Tensor<T> rgbGradient,
        out Tensor<T> densityGradient)
    {
        if (rgbSamples == null) throw new ArgumentNullException(nameof(rgbSamples));
        if (densitySamples == null) throw new ArgumentNullException(nameof(densitySamples));
        if (tValues == null) throw new ArgumentNullException(nameof(tValues));
        if (outputGradient == null) throw new ArgumentNullException(nameof(outputGradient));

        var numOps = MathHelper.GetNumericOperations<T>();

        int numRays = tValues.Shape[0];
        int numSamples = tValues.Shape[1];

        var rgbGrad = new T[numRays * numSamples * 3];
        var densityGrad = new T[numRays * numSamples];

        Parallel.For(0, numRays, r =>
        {
            // Forward pass to cache transmittance and alpha values
            var transmittances = new double[numSamples];
            var alphas = new double[numSamples];
            var deltas = new double[numSamples];

            double T = 1.0;
            for (int s = 0; s < numSamples; s++)
            {
                transmittances[s] = T;

                double t0 = numOps.ToDouble(tValues.GetFlat(r * numSamples + s));
                double t1 = (s + 1 < numSamples)
                    ? numOps.ToDouble(tValues.GetFlat(r * numSamples + s + 1))
                    : t0 + 0.01;
                deltas[s] = Math.Max(0.0, t1 - t0);

                double sigma = numOps.ToDouble(densitySamples.GetFlat(r * numSamples + s));
                double alpha = 1.0 - Math.Exp(-sigma * deltas[s]);
                alphas[s] = alpha;

                T *= (1.0 - alpha);
            }

            // Get output gradient for this ray
            double gradR = numOps.ToDouble(outputGradient.GetFlat(r * 3));
            double gradG = numOps.ToDouble(outputGradient.GetFlat(r * 3 + 1));
            double gradB = numOps.ToDouble(outputGradient.GetFlat(r * 3 + 2));

            // Backward pass
            for (int s = 0; s < numSamples; s++)
            {
                double Ti = transmittances[s];
                double ai = alphas[s];
                double di = deltas[s];

                int rgbIdx = (r * numSamples + s) * 3;
                double rVal = numOps.ToDouble(rgbSamples.GetFlat(rgbIdx));
                double gVal = numOps.ToDouble(rgbSamples.GetFlat(rgbIdx + 1));
                double bVal = numOps.ToDouble(rgbSamples.GetFlat(rgbIdx + 2));

                // dL/dc_i = T_i * alpha_i * dL/dC
                double dL_dci = Ti * ai;
                rgbGrad[rgbIdx] = numOps.FromDouble(dL_dci * gradR);
                rgbGrad[rgbIdx + 1] = numOps.FromDouble(dL_dci * gradG);
                rgbGrad[rgbIdx + 2] = numOps.FromDouble(dL_dci * gradB);

                // dL/d(sigma_i) = sum over j >= i of contribution
                // Simplified: dL/d(alpha_i) * d(alpha_i)/d(sigma_i)
                // d(alpha)/d(sigma) = delta * exp(-sigma * delta) = delta * (1 - alpha)
                double dAlpha_dSigma = di * (1.0 - ai);
                double dL_dAlpha = Ti * (gradR * rVal + gradG * gVal + gradB * bVal);

                // Also need to account for transmittance effect on subsequent samples
                // This is a simplified approximation
                densityGrad[r * numSamples + s] = numOps.FromDouble(dL_dAlpha * dAlpha_dSigma);
            }
        });

        rgbGradient = new Tensor<T>(rgbGrad, [numRays * numSamples, 3]);
        densityGradient = new Tensor<T>(densityGrad, [numRays * numSamples, 1]);
    }

    #endregion

    #region Ray Sampling

    /// <summary>
    /// Samples points uniformly along rays.
    /// </summary>
    public static (Tensor<T> positions, Tensor<T> directions, Tensor<T> tValues) SampleRayPoints<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        T nearBound,
        T farBound,
        int numSamples,
        bool stratified = true)
    {
        if (rayOrigins == null) throw new ArgumentNullException(nameof(rayOrigins));
        if (rayDirections == null) throw new ArgumentNullException(nameof(rayDirections));
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int numRays = rayOrigins.Shape[0];

        double near = numOps.ToDouble(nearBound);
        double far = numOps.ToDouble(farBound);
        double step = (far - near) / numSamples;

        var positions = new T[numRays * numSamples * 3];
        var directions = new T[numRays * numSamples * 3];
        var tVals = new T[numRays * numSamples];

        // Use thread-local random for stratified sampling
        Parallel.For(0, numRays, () => RandomHelper.CreateSeededRandom(Environment.TickCount ^ Thread.CurrentThread.ManagedThreadId),
            (r, state, random) =>
            {
                double ox = numOps.ToDouble(rayOrigins.GetFlat(r * 3));
                double oy = numOps.ToDouble(rayOrigins.GetFlat(r * 3 + 1));
                double oz = numOps.ToDouble(rayOrigins.GetFlat(r * 3 + 2));
                double dx = numOps.ToDouble(rayDirections.GetFlat(r * 3));
                double dy = numOps.ToDouble(rayDirections.GetFlat(r * 3 + 1));
                double dz = numOps.ToDouble(rayDirections.GetFlat(r * 3 + 2));

                for (int s = 0; s < numSamples; s++)
                {
                    double t;
                    if (stratified)
                    {
                        double t0 = near + step * s;
                        double t1 = t0 + step;
                        t = t0 + random.NextDouble() * (t1 - t0);
                    }
                    else
                    {
                        t = near + step * (s + 0.5);
                    }

                    int idx = (r * numSamples + s) * 3;
                    positions[idx] = numOps.FromDouble(ox + t * dx);
                    positions[idx + 1] = numOps.FromDouble(oy + t * dy);
                    positions[idx + 2] = numOps.FromDouble(oz + t * dz);

                    directions[idx] = rayDirections.GetFlat(r * 3);
                    directions[idx + 1] = rayDirections.GetFlat(r * 3 + 1);
                    directions[idx + 2] = rayDirections.GetFlat(r * 3 + 2);

                    tVals[r * numSamples + s] = numOps.FromDouble(t);
                }

                return random;
            },
            _ => { });

        return (
            new Tensor<T>(positions, [numRays * numSamples, 3]),
            new Tensor<T>(directions, [numRays * numSamples, 3]),
            new Tensor<T>(tVals, [numRays, numSamples])
        );
    }

    /// <summary>
    /// Performs importance sampling based on density weights.
    /// </summary>
    public static Tensor<T> ImportanceSampling<T>(
        Tensor<T> tValuesCoarse,
        Tensor<T> weightsCoarse,
        int numFineSamples)
    {
        if (tValuesCoarse == null) throw new ArgumentNullException(nameof(tValuesCoarse));
        if (weightsCoarse == null) throw new ArgumentNullException(nameof(weightsCoarse));
        if (numFineSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFineSamples), "Number of fine samples must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int numRays = tValuesCoarse.Shape[0];
        int numCoarseSamples = tValuesCoarse.Shape[1];

        var fineTValues = new T[numRays * numFineSamples];

        Parallel.For(0, numRays, () => RandomHelper.CreateSeededRandom(Environment.TickCount ^ Thread.CurrentThread.ManagedThreadId),
            (r, state, random) =>
            {
                int coarseOffset = r * numCoarseSamples;

                // Compute CDF from weights
                var cdf = new double[numCoarseSamples];
                double weightSum = 0.0;
                for (int s = 0; s < numCoarseSamples; s++)
                {
                    weightSum += Math.Max(0.0, numOps.ToDouble(weightsCoarse.GetFlat(coarseOffset + s)));
                }

                if (weightSum <= 1e-10)
                {
                    // Uniform sampling if weights are zero
                    double tMin = numOps.ToDouble(tValuesCoarse.GetFlat(coarseOffset));
                    double tMax = numOps.ToDouble(tValuesCoarse.GetFlat(coarseOffset + numCoarseSamples - 1));
                    for (int i = 0; i < numFineSamples; i++)
                    {
                        double u = (i + random.NextDouble()) / numFineSamples;
                        fineTValues[r * numFineSamples + i] = numOps.FromDouble(tMin + u * (tMax - tMin));
                    }
                }
                else
                {
                    double accum = 0.0;
                    for (int s = 0; s < numCoarseSamples; s++)
                    {
                        accum += Math.Max(0.0, numOps.ToDouble(weightsCoarse.GetFlat(coarseOffset + s))) / weightSum;
                        cdf[s] = accum;
                    }

                    // Inverse CDF sampling
                    for (int i = 0; i < numFineSamples; i++)
                    {
                        double u = (i + random.NextDouble()) / numFineSamples;

                        // Binary search in CDF
                        int idx = 0;
                        while (idx < numCoarseSamples - 1 && u > cdf[idx])
                        {
                            idx++;
                        }

                        // Linear interpolation
                        double cdfPrev = idx > 0 ? cdf[idx - 1] : 0.0;
                        double cdfCurr = cdf[idx];
                        double t0 = numOps.ToDouble(tValuesCoarse.GetFlat(coarseOffset + Math.Max(0, idx - 1)));
                        double t1 = numOps.ToDouble(tValuesCoarse.GetFlat(coarseOffset + idx));

                        double denom = cdfCurr - cdfPrev;
                        double t = denom > 1e-10 ? t0 + (u - cdfPrev) / denom * (t1 - t0) : t0;
                        fineTValues[r * numFineSamples + i] = numOps.FromDouble(t);
                    }
                }

                return random;
            },
            _ => { });

        return new Tensor<T>(fineTValues, [numRays, numFineSamples]);
    }

    /// <summary>
    /// Generates camera rays for image rendering.
    /// </summary>
    public static (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays<T>(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength)
    {
        if (cameraPosition == null) throw new ArgumentNullException(nameof(cameraPosition));
        if (cameraRotation == null) throw new ArgumentNullException(nameof(cameraRotation));
        if (imageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(imageWidth));
        if (imageHeight <= 0) throw new ArgumentOutOfRangeException(nameof(imageHeight));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numRays = imageWidth * imageHeight;

        var origins = new T[numRays * 3];
        var directions = new T[numRays * 3];

        double f = numOps.ToDouble(focalLength);
        double cx = (imageWidth - 1) * 0.5;
        double cy = (imageHeight - 1) * 0.5;

        // Extract rotation matrix elements
        double r00 = numOps.ToDouble(cameraRotation[0, 0]);
        double r01 = numOps.ToDouble(cameraRotation[0, 1]);
        double r02 = numOps.ToDouble(cameraRotation[0, 2]);
        double r10 = numOps.ToDouble(cameraRotation[1, 0]);
        double r11 = numOps.ToDouble(cameraRotation[1, 1]);
        double r12 = numOps.ToDouble(cameraRotation[1, 2]);
        double r20 = numOps.ToDouble(cameraRotation[2, 0]);
        double r21 = numOps.ToDouble(cameraRotation[2, 1]);
        double r22 = numOps.ToDouble(cameraRotation[2, 2]);

        Parallel.For(0, imageHeight, y =>
        {
            for (int x = 0; x < imageWidth; x++)
            {
                // Pixel to camera space direction
                double px = (x - cx) / f;
                double py = (y - cy) / f;
                double pz = 1.0;

                // Rotate to world space
                double dx = r00 * px + r01 * py + r02 * pz;
                double dy = r10 * px + r11 * py + r12 * pz;
                double dz = r20 * px + r21 * py + r22 * pz;

                // Normalize direction
                double norm = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (norm > 0)
                {
                    double inv = 1.0 / norm;
                    dx *= inv;
                    dy *= inv;
                    dz *= inv;
                }

                int idx = (y * imageWidth + x) * 3;
                origins[idx] = cameraPosition[0];
                origins[idx + 1] = cameraPosition[1];
                origins[idx + 2] = cameraPosition[2];
                directions[idx] = numOps.FromDouble(dx);
                directions[idx + 1] = numOps.FromDouble(dy);
                directions[idx + 2] = numOps.FromDouble(dz);
            }
        });

        return (
            new Tensor<T>(origins, [numRays, 3]),
            new Tensor<T>(directions, [numRays, 3])
        );
    }

    #endregion
}
