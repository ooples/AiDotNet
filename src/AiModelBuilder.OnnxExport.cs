using AiDotNet.Models.Results;
using AiDotNet.Onnx;

namespace AiDotNet;

/// <summary>
/// Opt-in ONNX-export validation surface on <see cref="AiModelBuilder{T, TInput, TOutput}"/>.
///
/// Calling <see cref="RequireOnnxExportable"/> on the builder marks the build pipeline
/// as requiring an ONNX-exportable result. The check is *advisory* in v0.1 — the
/// flag is read after Build() to validate the resulting AiModelResult and throw
/// if any layer lacks a ConvertToOnnx override.
/// </summary>
/// <remarks>
/// v0.1 limitation: the flag does NOT auto-fire from inside <c>Build()</c>. Users
/// call <see cref="ValidateOnnxExportableIfRequired"/> on the resulting model, or
/// simply rely on <see cref="AiModelResultOnnxExtensions.ExportToOnnx"/> to throw
/// at export time. A future PR can wire the flag into the Build() pipeline so
/// the check fires eagerly.
/// </remarks>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private bool _requireOnnxExportable;

    /// <summary>
    /// Marks the builder as requiring an ONNX-exportable result. After
    /// <see cref="ValidateOnnxExportableIfRequired"/> is invoked on the build
    /// output, an <see cref="OnnxExportUnsupportedException"/> will be raised
    /// if any layer in the resulting model lacks ONNX export support.
    /// </summary>
    public AiModelBuilder<T, TInput, TOutput> RequireOnnxExportable()
    {
        _requireOnnxExportable = true;
        return this;
    }

    /// <summary>True if the user opted into ONNX-export validation on this builder.</summary>
    public bool IsOnnxExportableRequired => _requireOnnxExportable;

    /// <summary>
    /// If the builder was marked with <see cref="RequireOnnxExportable"/>,
    /// throws <see cref="OnnxExportUnsupportedException"/> when the supplied
    /// trained model contains a non-exportable layer. No-op otherwise.
    /// </summary>
    public void ValidateOnnxExportableIfRequired(AiModelResult<T, TInput, TOutput> result)
    {
        if (!_requireOnnxExportable) return;
        if (result is null) throw new ArgumentNullException(nameof(result));

        if (!result.CanExportToOnnx())
        {
            throw new OnnxExportUnsupportedException(
                "AiModelResult (per RequireOnnxExportable)",
                "Builder was configured with RequireOnnxExportable() but the trained model " +
                "contains one or more layers without an ONNX export override. Inspect the " +
                "model's Layers and replace or remove the unsupported layer types.");
        }
    }
}
