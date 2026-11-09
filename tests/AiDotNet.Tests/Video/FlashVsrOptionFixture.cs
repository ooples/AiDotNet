using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;

namespace AiDotNet.Tests.Video;

internal static class FlashVsrOptionFixture
{
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    private static readonly ConditionalWeakTable<object, object> PreparedStages = new();

    internal static bool Handles(string option) => option is "ScaleFactor" or "LqProjectionLayers"
        or "SigmaShift" or "NumTrainTimesteps" or "TeacherIterations" or "SparseIterations"
        or "SparseRatio" or "LocalRange" or "KvCacheChunks";

    internal static void Prepare(FlashVSROptions options, string option)
    {
        if (option == "LqProjectionLayers") options.NumLayers = 3;
        if (option is "SigmaShift" or "NumTrainTimesteps") options.DecoderIterations = 0;
        if (option is "TeacherIterations" or "SparseIterations")
        {
            options.DecoderIterations = 0;
            options.TeacherIterations = option == "TeacherIterations" ? 2 : 0;
            options.SparseIterations = 2;
        }
        // At smoke resolution the released sparsity budget and local window both saturate.
        if (option == "SparseRatio") options.SparseRatio = .02;
        if (option == "LocalRange") options.LocalRange = 1;
    }

    internal static long Observe<T>(FlashVSR<T> model, Tensor<T> input, string option)
    {
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        long Fingerprint(Tensor<T> tensor) => TtsPaperNumerics.Combine(
            TtsPaperNumerics.Fingerprint(tensor.Shape.Select(size => (double)size)),
            TtsPaperNumerics.Fingerprint(tensor.Select(numeric.ToDouble)));
        var options = (FlashVSROptions)typeof(FlashVSR<T>).GetField("_options", PrivateInstance)!.GetValue(model)!;
        if (option is "SigmaShift" or "NumTrainTimesteps")
            return TtsPaperNumerics.Fingerprint(FlashVSR<T>.Sigmas(options));
        if (option is "TeacherIterations" or "SparseIterations")
        {
            PreparedStages.GetValue(model, instance =>
            {
                typeof(FlashVSR<T>).GetField("_trainingSteps", PrivateInstance)!.SetValue(instance, 2);
                return new object();
            });
            return TtsPaperNumerics.Fingerprint(new[] { (double)model.TrainingStage });
        }
        if (option == "ScaleFactor")
        {
            // Exercise the actual bicubic preprocessing without padding an odd scale to hundreds of pixels.
            var clips = input.Rank == 5 ? input : AiDotNetEngine.Current.Reshape(input,
                new[] { 1 }.Concat(input.Shape).ToArray());
            var upsampled = (Tensor<T>)typeof(FlashVSR<T>).GetMethod("Upsample", PrivateInstance)!
                .Invoke(model, new object[] { clips })!;
            return Fingerprint(upsampled);
        }
        if (option == "LqProjectionLayers")
        {
            var projection = (FlashVsrLqProjection<T>)typeof(FlashVSR<T>).GetMethod("Projection", PrivateInstance)!
                .Invoke(model, new object[] { 0 })!;
            var video = new Tensor<T>(new[] { 1, 3, 9, 16, 16 });
            for (int i = 0; i < video.Length; i++) video[i] = numeric.FromDouble(.2 * Math.Sin(i * .017));
            return TtsPaperNumerics.Combine(projection.Forward(video).Select(Fingerprint).ToArray());
        }
        // Five temporal chunks expose cache eviction; two spatial blocks per side expose locality.
        const int frames = 10, height = 16, width = 16, headDimension = 4;
        var queries = new Tensor<T>(new[] { 1, frames * height * width, headDimension });
        var keys = new Tensor<T>(queries.Shape.ToArray());
        for (int i = 0; i < queries.Length; i++)
        {
            queries[i] = numeric.FromDouble(Math.Sin(i * .013) + .2 * Math.Cos(i * .007));
            keys[i] = numeric.FromDouble(Math.Cos(i * .019) + .3 * Math.Sin(i * .011));
        }
        var plan = (WanSparsePlan)typeof(FlashVSR<T>).GetMethod("SparsePlan", PrivateInstance)!
            .Invoke(model, new object[] { frames, height, width })!;
        return Fingerprint(WanSparseMask.Build(queries, keys, plan));
    }
}
