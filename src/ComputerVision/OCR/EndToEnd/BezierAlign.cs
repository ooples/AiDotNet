using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.OCR.EndToEnd;

/// <summary>
/// BezierAlign: rectifies an arbitrarily curved text instance into a rectangular feature map by
/// sampling along its two boundary Bezier curves.
/// </summary>
/// <remarks>
/// <para>
/// This is the second of ABCNet's two contributions (Liu et al., CVPR 2020, arXiv:2002.10200), and the
/// one that makes the Bezier representation useful rather than merely compact. RoIAlign samples an
/// AXIS-ALIGNED grid and RoIRotate samples a rotated quadrilateral; both are the wrong shape for curved
/// text, so they drag in background and cut off ascenders, and the recognizer then has to cope with a
/// warped, partly-occluded strip. BezierAlign instead samples a grid that FOLLOWS the curve, so the
/// rectified output is upright text regardless of how the instance bends.
/// </para>
/// <para>
/// The sampling rule, for output pixel <c>(i, j)</c> of an <c>outputHeight x outputWidth</c> map:
/// <c>t = (j + 0.5) / outputWidth</c> walks along the curve, <c>s = (i + 0.5) / outputHeight</c> walks
/// across it, and the source location is
/// <c>(1 - s) * TopCurve(t) + s * BottomCurve(t)</c> — the point s of the way down the segment joining
/// the two curves at the same t. Half-pixel centres rather than <c>j / outputWidth</c>, matching
/// RoIAlign's convention: sampling at pixel corners biases the whole map half a pixel toward the origin.
/// </para>
/// <para>
/// GRADIENTS REACH THE CONTROL POINTS. Because the Bernstein coefficients depend only on
/// <c>(i, j)</c>, the sampling grid is a LINEAR map of the eight control points, so it is built as a
/// constant matrix times the control-point tensor through engine ops rather than being precomputed into
/// a constant grid. A precomputed grid is the obvious simplification and it silently severs the
/// detection branch: the control points would receive gradient only from their own regression loss and
/// none from the recognition loss, so the two branches would stop informing each other and the
/// end-to-end training the paper reports would not be happening.
/// </para>
/// <para><b>For Beginners:</b> Curved text in a photo is hard to read directly. This straightens it
/// out — it walks along the top and bottom edges of the text and resamples the image into a neat
/// rectangle, so the text recognizer only ever sees flat, horizontal text.</para>
/// </remarks>
public static class BezierAlign
{
    /// <summary>Control points per instance: four along the top edge, then four along the bottom.</summary>
    public const int ControlPointCount = 8;

    /// <summary>
    /// Builds the <c>[8, 2]</c> control-point tensor BezierAlign consumes, top curve first.
    /// </summary>
    /// <remarks>
    /// The ORDER is part of the contract and is not self-checking: swapping the two curves flips the
    /// rectified output vertically, which produces upside-down text that still has the right shape and
    /// therefore still looks plausible in a feature map.
    /// </remarks>
    public static Tensor<T> ControlPointTensor<T>(CubicBezierCurve<T> topCurve, CubicBezierCurve<T> bottomCurve)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(new[] { ControlPointCount, 2 });

        for (int k = 0; k < CubicBezierCurve<T>.ControlPointCount; k++)
        {
            var (tx, ty) = topCurve.ControlPoint(k);
            t[(k * 2) + 0] = numOps.FromDouble(tx);
            t[(k * 2) + 1] = numOps.FromDouble(ty);

            var (bx, by) = bottomCurve.ControlPoint(k);
            int row = CubicBezierCurve<T>.ControlPointCount + k;
            t[(row * 2) + 0] = numOps.FromDouble(bx);
            t[(row * 2) + 1] = numOps.FromDouble(by);
        }

        return t;
    }

    /// <summary>
    /// The constant coefficient matrix <c>[outputHeight * outputWidth, 8]</c> mapping control points to
    /// sampling locations.
    /// </summary>
    /// <remarks>
    /// Row <c>q = i * outputWidth + j</c> holds <c>(1 - s) * B_k(t)</c> in its first four entries and
    /// <c>s * B_k(t)</c> in its last four. Exposed because it is the whole of the geometry: every row
    /// must sum to 1 (the Bernstein basis is a partition of unity and the top/bottom blend is convex),
    /// which is a directly checkable invariant that a sign or index slip breaks.
    /// </remarks>
    public static Tensor<T> SamplingCoefficients<T>(int outputHeight, int outputWidth)
    {
        if (outputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputHeight), outputHeight, "outputHeight must be positive.");
        if (outputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputWidth), outputWidth, "outputWidth must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var coeff = new Tensor<T>(new[] { outputHeight * outputWidth, ControlPointCount });

        for (int i = 0; i < outputHeight; i++)
        {
            double s = (i + 0.5) / outputHeight;
            for (int j = 0; j < outputWidth; j++)
            {
                double t = (j + 0.5) / outputWidth;
                var basis = CubicBezierCurve<T>.BernsteinBasis(t);

                int row = (i * outputWidth) + j;
                int off = row * ControlPointCount;
                for (int k = 0; k < 4; k++)
                {
                    coeff[off + k] = numOps.FromDouble((1.0 - s) * basis[k]);
                    coeff[off + 4 + k] = numOps.FromDouble(s * basis[k]);
                }
            }
        }

        return coeff;
    }

    /// <summary>
    /// Rectifies a curved text instance out of a feature map.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="engine">
    /// The tensor engine, so the sampling is recorded on the caller's tape. Read the calling type's own
    /// engine member per use rather than capturing one.
    /// </param>
    /// <param name="features">Feature map, <c>[C, H, W]</c> or <c>[B, C, H, W]</c>.</param>
    /// <param name="controlPoints">
    /// <c>[8, 2]</c> control points in FEATURE-MAP pixel coordinates — four along the top edge then four
    /// along the bottom, as produced by <see cref="ControlPointTensor{T}"/>. Coordinates are (x, y).
    /// </param>
    /// <param name="outputHeight">Rectified height, in pixels.</param>
    /// <param name="outputWidth">Rectified width, in pixels.</param>
    /// <returns>The rectified features, <c>[C, outputHeight, outputWidth]</c> (or batched).</returns>
    public static Tensor<T> Sample<T>(
        IEngine engine,
        Tensor<T> features,
        Tensor<T> controlPoints,
        int outputHeight,
        int outputWidth)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (controlPoints is null) throw new ArgumentNullException(nameof(controlPoints));

        var featureShape = features.Shape;
        if (featureShape.Length is not (3 or 4))
        {
            throw new ArgumentException(
                $"Features must be [C,H,W] or [B,C,H,W]; got rank {featureShape.Length}.", nameof(features));
        }

        var cpShape = controlPoints.Shape;
        if (cpShape.Length != 2 || cpShape[0] != ControlPointCount || cpShape[1] != 2)
        {
            throw new ArgumentException(
                $"Control points must be [{ControlPointCount}, 2]; got [{string.Join(", ", cpShape.ToArray())}].",
                nameof(controlPoints));
        }

        bool batched = featureShape.Length == 4;
        int batch = batched ? featureShape[0] : 1;
        int channels = batched ? featureShape[1] : featureShape[0];
        int height = featureShape[featureShape.Length - 2];
        int width = featureShape[featureShape.Length - 1];

        if (batch != 1)
        {
            // One instance's control points cannot describe several images. Silently sampling the same
            // curve out of every image in the batch would be wrong in a way that produces plausible
            // output, so it is refused instead.
            throw new ArgumentException(
                $"BezierAlign samples ONE text instance, so the batch must be 1; got {batch}. Call it per "
                + "instance and stack the results.", nameof(features));
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // grid_pixels = coeff @ controlPoints, so gradients flow back into the control points.
        var coeff = SamplingCoefficients<T>(outputHeight, outputWidth);
        var pixels = engine.TensorMatMul(coeff, controlPoints);   // [outH*outW, 2]

        var pixelX = engine.TensorNarrow(pixels, 1, 0, 1);        // [outH*outW, 1]
        var pixelY = engine.TensorNarrow(pixels, 1, 1, 1);

        var normX = NormalizeToGrid(engine, pixelX, width, numOps);
        var normY = NormalizeToGrid(engine, pixelY, height, numOps);

        var gridX = engine.Reshape(normX, new[] { 1, outputHeight, outputWidth, 1 });
        var gridY = engine.Reshape(normY, new[] { 1, outputHeight, outputWidth, 1 });
        var grid = engine.Concat(new[] { gridX, gridY }, 3);      // [1, outH, outW, 2]

        var featuresNchw = batched
            ? features
            : engine.Reshape(features, new[] { 1, channels, height, width });

        var sampled = engine.GridSample(featuresNchw, grid);      // [1, C, outH, outW]

        return batched
            ? sampled
            : engine.Reshape(sampled, new[] { channels, outputHeight, outputWidth });
    }

    /// <summary>
    /// Maps pixel coordinates onto the <c>[-1, 1]</c> range GridSample samples in, using the
    /// <c>align_corners=false</c> convention: <c>norm = 2 * (p + 0.5) / extent - 1</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// THE CONVENTION MATTERS AND IS NOT THE OBVIOUS ONE. The two-argument
    /// <c>IEngine.GridSample(input, grid)</c> is documented as a torchvision-default shim, and
    /// torchvision's default is <c>align_corners=false</c>, which places normalized -1 and +1 at the
    /// OUTER EDGES of the border pixels rather than at their centres. The intuitive
    /// <c>2p / (extent - 1) - 1</c> is the <c>align_corners=true</c> mapping; feeding it to this sampler
    /// makes it decode every coordinate as <c>p * extent / (extent - 1) - 0.5</c> — a half-pixel shift
    /// plus a slight outward stretch that grows toward the edges.
    /// </para>
    /// <para>
    /// MEASURED rather than reasoned about: with the true-convention mapping, sampling an affine feature
    /// map <c>f(y,x) = x + 10y</c> at intended (3, 5) returned 51.033333333333335 instead of 53, which is
    /// exactly <c>2.7 + 10 * 4.8333</c> — the false-convention decode of the true-convention input.
    /// </para>
    /// <para>
    /// A degenerate axis of extent 1 has no interior to interpolate across, so its coordinate is pinned
    /// to the centre.
    /// </para>
    /// </remarks>
    private static Tensor<T> NormalizeToGrid<T>(
        IEngine engine, Tensor<T> pixels, int extent, INumericOperations<T> numOps)
    {
        if (extent <= 1) return engine.TensorMultiplyScalar(pixels, numOps.Zero);

        var scaled = engine.TensorMultiplyScalar(pixels, numOps.FromDouble(2.0 / extent));
        return engine.TensorAddScalar(scaled, numOps.FromDouble((1.0 / extent) - 1.0));
    }
}
