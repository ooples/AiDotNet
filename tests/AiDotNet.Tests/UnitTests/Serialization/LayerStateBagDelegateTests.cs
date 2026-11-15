using System;
using System.Collections.Generic;
using AiDotNet.Serialization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

/// <summary>
/// The delegate channel of <see cref="LayerStateBag.CloneObject{TObject}"/>: a saved tensor program rebuilds the
/// function, and a corrupt one fails with the documented <see cref="InvalidOperationException"/> and its reason.
/// </summary>
public class LayerStateBagDelegateTests
{
    [Fact]
    public void CloneObject_RebuildsASavedTensorProgram()
    {
        var scale = new Tensor<float>(new[] { 0.5f, -2f, 3f }, new[] { 3 });
        Func<Tensor<float>, Tensor<float>> f = x => AiDotNetEngine.Current.TensorMultiply(x, scale);
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["forward"] = LayerStateBag.FormatDelegate(f, new[] { 3 }),
        }, "LambdaLayer");

        var restored = bag.CloneObject<Func<Tensor<float>, Tensor<float>>>("forward");

        var input = new Tensor<float>(new[] { 1f, 2f, 3f, -1f, -2f, -3f }, new[] { 2, 3 });
        var expected = f(input);
        var actual = restored(input);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 6);
    }

    [Fact]
    public void CloneObject_CorruptTensorProgram_ThrowsInvalidOperationWithTheReason()
    {
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["forward"] = DelegateState.ProgramScheme + "%%% not base64 %%%",
        }, "LambdaLayer");

        var error = Assert.Throws<InvalidOperationException>(
            () => bag.CloneObject<Func<Tensor<float>, Tensor<float>>>("forward"));
        Assert.Contains("not valid base64", error.Message);
        Assert.Contains("LambdaLayer", error.Message);
    }
}
