using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Static factory for image-space training data loaders (#1834). Reference implementations
/// require the caller to construct a full data pipeline manually — read image files, parse a
/// COLMAP transforms.json, build camera intrinsics, sample rays, batch them. These factories
/// give consumers a one-liner: <c>ImageTrainingDataLoaders.FromViews(views)</c>.
/// </summary>
/// <summary>
/// Optional interface for image-space loaders that hold their view set in memory. Allows
/// pose-based scene-bounds estimation without materializing the full loader.
/// </summary>
internal interface IViewSetProvider<T>
{
    /// <summary>The full in-memory view set backing this loader.</summary>
    IReadOnlyList<ImageView<T>> Views { get; }
}

public static class ImageTrainingDataLoaders
{
    /// <summary>
    /// Builds an image-space loader from an in-memory sequence of <see cref="ImageView{T}"/>
    /// records. Each <c>IterateBatches</c> call samples <c>raysPerBatch</c> pixels randomly
    /// across the full view set (uniform pixel sampling — paper standard).
    /// </summary>
    /// <param name="views">The photo + camera-pose set to train on.</param>
    /// <param name="seed">Optional RNG seed for reproducible pixel sampling.</param>
    public static IDataLoader<ImageView<T>, PixelBatch<T>> FromViews<T>(
        IEnumerable<ImageView<T>> views,
        int? seed = null)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        var array = views.ToArray();
        if (array.Length == 0) throw new ArgumentException("View set must be non-empty.", nameof(views));
        return new InMemoryImageDataLoader<T>(array, seed);
    }

    private sealed class InMemoryImageDataLoader<T> : IDataLoader<ImageView<T>, PixelBatch<T>>, IViewSetProvider<T>
    {
        public IReadOnlyList<ImageView<T>> Views => _views;
        private readonly ImageView<T>[] _views;
        private readonly int? _seed;
        private Random _rng;

        public InMemoryImageDataLoader(ImageView<T>[] views, int? seed)
        {
            _views = views;
            _seed  = seed;
            _rng   = seed.HasValue
                ? RandomHelper.CreateSeededRandom(seed.Value)
                : RandomHelper.CreateSecureRandom();
            IsLoaded = true;
        }

        public string Name        => "InMemoryImageDataLoader";
        public string Description => $"{_views.Length} photo(s) + camera pose(s) for image-space radiance-field training.";
        public bool   IsLoaded    { get; private set; }
        public int    Count       => _views.Length;

        // ICountable — image-space iteration is per-batch of rays (raysPerBatch), not per-
        // view, so the "sample" unit is the ray. We advertise TotalCount = view count for
        // caller-side progress bars (which typically want epochs measured in view counts).
        // Batch tracking is best-effort — image loaders emit exactly one batch per call.
        public int TotalCount => _views.Length;
        public int CurrentIndex => _iterations;
        public int BatchCount => 1;
        public int CurrentBatchIndex => 0;
        public double Progress => TotalCount > 0 ? Math.Min(1.0, (double)_iterations / TotalCount) : 1.0;
        private int _iterations;

        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            IsLoaded = true;
            return Task.CompletedTask;
        }

        public void Unload()
        {
            // In-memory: nothing to unload; the ImageView array stays referenced.
            IsLoaded = false;
        }

        public void Reset()
        {
            // Fresh RNG so seeded loaders are deterministic across Reset — the next batch
            // sequence matches the original construction. Unseeded loaders re-secure-seed.
            _rng = _seed.HasValue
                ? RandomHelper.CreateSeededRandom(_seed.Value)
                : RandomHelper.CreateSecureRandom();
        }

        public IEnumerable<(ImageView<T> Input, PixelBatch<T> Output)> IterateBatches(int batchSize)
        {
            if (batchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize), "batchSize must be positive.");
            }

            // Excellence goal #2 — MIXED-RESOLUTION proportional sampling. Reference impls
            // (nerfstudio, tiny-cuda-nn) require every photo to be pre-resized to a common
            // resolution because they pick views uniformly at random and would starve
            // high-res photos of rays if pixel counts differ. Here we sample view weights
            // proportional to their pixel counts, so a 4K photo gets ~16x the rays of a
            // 1K photo per epoch. Callers drop in raw photos of any resolution.
            _iterations++;
            var view = SampleViewProportional();
            var (origins, dirs, colors) = SampleRays(view, batchSize, _rng);
            yield return (view, new PixelBatch<T>(origins, dirs, colors));
        }

        private ImageView<T> SampleViewProportional()
        {
            // Pick a view weighted by its pixel count so mixed-resolution view sets sample
            // rays proportionally. For a uniform-resolution set this collapses to the
            // reference impl's uniform-random pick (which is what most callers see).
            long totalPixels = 0;
            for (int i = 0; i < _views.Length; i++)
            {
                totalPixels += (long)_views[i].Height * _views[i].Width;
            }
            long target = (long)(_rng.NextDouble() * totalPixels);
            long cum = 0;
            for (int i = 0; i < _views.Length; i++)
            {
                cum += (long)_views[i].Height * _views[i].Width;
                if (target < cum) return _views[i];
            }
            return _views[_views.Length - 1];
        }

        private static (Tensor<T> origins, Tensor<T> dirs, Tensor<T> colors) SampleRays(
            ImageView<T> view, int rayCount, Random rng)
        {
            var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var engine = AiDotNetEngine.Current;
            int H = view.Height;
            int W = view.Width;

            // Three-tier focal resolution (excellence goal #1): explicit -> EXIF -> nerfstudio
            // fallback. ImageView owns the precedence order; the loader just consumes.
            double focalPx = view.ResolveFocalLengthInPixels();

            // Pixel draws keep the seeded order: y then x for every ray.
            var rows = new int[rayCount];
            var columns = new int[rayCount];
            for (int i = 0; i < rayCount; i++)
            {
                rows[i] = rng.Next(H);
                columns[i] = rng.Next(W);
            }

            var rowIndex = new Tensor<int>(rows, new[] { rayCount });
            var columnIndex = new Tensor<int>(columns, new[] { rayCount });
            var ys = engine.TensorIndexSelect(engine.TensorArange<T>(0, H, 1), rowIndex, 0);
            var xs = engine.TensorIndexSelect(engine.TensorArange<T>(0, W, 1), columnIndex, 0);

            // Camera-space ray direction (right-handed, +z into scene) via pinhole intrinsics.
            var focal = numOps.FromDouble(focalPx);
            var dxCam = engine.TensorDivideScalar(engine.TensorSubtractScalar(xs, numOps.FromDouble(W * 0.5)), focal);
            var dyCam = engine.TensorDivideScalar(
                engine.TensorNegate(engine.TensorSubtractScalar(ys, numOps.FromDouble(H * 0.5))), focal);
            var dzCam = engine.TensorOnesLike(dxCam);
            var cameraDirs = engine.TensorStack(new[] { dxCam, dyCam, dzCam }, 1);
            var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(cameraDirs), new[] { 1 }, true));
            cameraDirs = engine.TensorDivide(cameraDirs, engine.TensorBroadcastTo(norm, cameraDirs.Shape.ToArray()));

            // Rotate into world coordinates via CameraRotation columns: world = R * cam for every ray.
            var rotation = engine.TensorSlice(Tensor<T>.FromMatrix(view.CameraRotation), new[] { 0, 0 }, new[] { 3, 3 });
            var dirs = engine.TensorMatMul(cameraDirs, engine.TensorTranspose(rotation));

            var position = Tensor<T>.FromVector(view.CameraPosition.GetSubVector(0, 3)).Reshape(1, 3);
            var origins = engine.TensorBroadcastTo(position, new[] { rayCount, 3 }).Contiguous();

            var pixels = engine.TensorIndex(view.Photo, new[] { rowIndex, columnIndex });
            var colors = engine.TensorSlice(pixels, new[] { 0, 0 }, new[] { rayCount, 3 });

            return (origins, dirs, colors);
        }
    }
}
