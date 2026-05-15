using System;

namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// Translates a model file (TFLite / ONNX / etc.) into an NNAPI operation
/// graph by adding operands and operations to a freshly-created
/// <c>ANeuralNetworksModel</c> handle. Plug an implementation into
/// <see cref="NNAPIBackend{T}"/> via its constructor so <c>LoadModel</c>
/// can actually compile a real NNAPI graph instead of falling back to the
/// managed CPU executor.
/// </summary>
/// <remarks>
/// <para>
/// Each NNAPI backend instance owns its own <c>_model</c> IntPtr. The
/// builder is invoked once per <see cref="NNAPIBackend{T}.LoadModel"/>
/// call, after the backend has called <c>ANeuralNetworksModel_create</c>
/// but before <c>ANeuralNetworksModel_finish</c>. The builder is
/// responsible for:
/// </para>
/// <list type="number">
/// <item><description>Parsing <paramref name="modelBytes"/> into the
/// caller's per-op IR.</description></item>
/// <item><description>For each tensor in the IR, calling
/// <c>ANeuralNetworksModel_addOperand</c> via
/// <see cref="NNAPIInterop.ModelAddOperand"/>.</description></item>
/// <item><description>For each op in the IR, calling
/// <c>ANeuralNetworksModel_addOperation</c> via
/// <see cref="NNAPIInterop.ModelAddOperation"/>.</description></item>
/// <item><description>Calling
/// <see cref="NNAPIInterop.ModelIdentifyInputsAndOutputs"/> to mark the
/// graph's external IO operands.</description></item>
/// </list>
/// <para>
/// Splitting graph construction out of the backend keeps the backend
/// model-format-agnostic — TFLite, ONNX, and bespoke IRs all plug in via
/// the same interface.
/// </para>
/// </remarks>
public interface INNAPIGraphBuilder
{
    /// <summary>
    /// Populates the supplied <paramref name="modelHandle"/> with operands
    /// and operations decoded from <paramref name="modelBytes"/>.
    /// </summary>
    /// <param name="modelHandle">
    /// Native <c>ANeuralNetworksModel*</c> handle created by the backend.
    /// The builder must NOT call <c>ANeuralNetworksModel_finish</c> on this
    /// handle — the backend invokes that itself once the builder returns.
    /// </param>
    /// <param name="modelBytes">Raw model file contents (TFLite / ONNX / etc.).</param>
    /// <returns>
    /// <c>true</c> if the model was populated successfully and the backend
    /// should proceed to compile it. <c>false</c> if the builder couldn't
    /// translate the file — the backend will then Trace-warn and route to
    /// the managed CPU fallback.
    /// </returns>
    bool BuildGraph(IntPtr modelHandle, byte[] modelBytes);
}
