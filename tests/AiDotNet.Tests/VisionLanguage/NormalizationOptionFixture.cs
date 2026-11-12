using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.VisionLanguage.Foundational;

namespace AiDotNet.Tests.VisionLanguage;

internal static class NormalizationOptionFixture
{
    internal static bool Handles<T>(INeuralNetworkModel<T> model, string option)
        => (model is VinVL<T> or UNITER<T>) && (option is "ImageMean" or "ImageStd");

    internal static long Observe<T>(INeuralNetworkModel<T> model)
    {
        // Image normalization acts before the encoder. Probe its real image-input stage instead
        // of training the whole multimodal network for each channel of a preprocessing setting.
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var image = new Tensor<T>(new[] { 1, 3, 2, 2 });
        for (int i = 0; i < image.Length; i++) image[i] = ops.FromDouble((i + 1) / 13.0);
        var preprocess = model.GetType().GetMethod("PreprocessImage", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var result = (Tensor<T>)preprocess.Invoke(model, new object[] { image })!;
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(result.ToArray().Select(value => ops.ToDouble(value)));
    }
}
