using AiDotNet;
using AiDotNet.Onnx;
using Xunit;

namespace AiDotNet.Tests.Onnx;

/// <summary>
/// Verifies the AiModelBuilder.RequireOnnxExportable() opt-in surface:
///   - Flag is off by default
///   - Method is fluent (returns the builder)
///   - Method flips the flag to true
///
/// v0.1 doesn't auto-fire the validation from Build() — that's a future PR.
/// These tests cover the flag plumbing only.
/// </summary>
public class OnnxRequireExportableTests
{
    [Fact]
    public void RequireOnnxExportable_IsFalseByDefault()
    {
        var builder = new AiModelBuilder<float, AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>();
        Assert.False(builder.IsOnnxExportableRequired);
    }

    [Fact]
    public void RequireOnnxExportable_FlipsTheFlag()
    {
        var builder = new AiModelBuilder<float, AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>();
        builder.RequireOnnxExportable();
        Assert.True(builder.IsOnnxExportableRequired);
    }

    [Fact]
    public void RequireOnnxExportable_IsFluent()
    {
        var builder = new AiModelBuilder<float, AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>();
        var returned = builder.RequireOnnxExportable();
        Assert.Same(builder, returned);
    }
}
