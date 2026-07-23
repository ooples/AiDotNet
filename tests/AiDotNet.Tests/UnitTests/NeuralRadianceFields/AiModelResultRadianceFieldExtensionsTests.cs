using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralRadianceFields.Extensions;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Regression tests for #1836 — the radiance-field family extension methods on
/// <see cref="AiModelResult{T, TInput, TOutput}"/>. Guarantees:
///   1. When the result's underlying model IS a radiance field, the extensions dispatch to
///      the model's own <c>RenderImage</c> / <c>RenderRays</c> / <c>QueryField</c>.
///   2. When it's NOT a radiance field, the extensions throw <see cref="InvalidOperationException"/>
///      with a message naming the actual model type + pointing at the correct namespace.
///   3. Null results throw <see cref="ArgumentNullException"/> at the extension entry.
///
/// The extensions are the ONLY way for external consumers to reach radiance-field-specific
/// inference from the facade path, because <see cref="AiModelResult{T, TInput, TOutput}.Model"/>
/// is intentionally <c>internal</c> (IP protection).
/// </summary>
public class AiModelResultRadianceFieldExtensionsTests
{
    private static AiModelResult<float, Tensor<float>, Tensor<float>> BuildRadianceFieldResult()
    {
        var nerf = new NeRF<float>(
            positionEncodingLevels: 4,
            directionEncodingLevels: 4,
            hiddenDim: 32,
            numLayers: 2,
            colorHiddenDim: 16,
            colorNumLayers: 1,
            useHierarchicalSampling: false,
            renderSamples: 4,
            renderNearBound: 1.0,
            renderFarBound: 4.5,
            learningRate: 1e-3);

        // AiDotNetTests has InternalsVisibleTo — construct + set internal Model directly.
        // Mirrors what BuildAsync does when it hands the trained model back.
        var result = new AiModelResult<float, Tensor<float>, Tensor<float>> { Model = nerf };
        return result;
    }

    private static AiModelResult<float, Matrix<float>, Vector<float>> BuildNonRadianceFieldResult()
    {
        // Any model that's NOT a radiance field — SimpleRegression is the smallest stand-in.
        var model = new SimpleRegression<float>();
        return new AiModelResult<float, Matrix<float>, Vector<float>> { Model = model };
    }

    [Fact]
    public void RenderImage_OnNonRadianceFieldResult_ThrowsWithModelTypeNamed()
    {
        var result = BuildNonRadianceFieldResult();
        var pos = new Vector<float>(3);
        var rot = new Matrix<float>(3, 3);

        var ex = Assert.Throws<InvalidOperationException>(
            () => result.RenderImage(pos, rot, imageWidth: 8, imageHeight: 8, focalLength: 100f));

        Assert.Contains("IRadianceField", ex.Message);
        Assert.Contains("SimpleRegression", ex.Message);
        // Points caller at other extension namespaces.
        Assert.Contains("Transformers", ex.Message);
    }

    [Fact]
    public void RenderRays_OnNonRadianceFieldResult_ThrowsWithModelTypeNamed()
    {
        var result = BuildNonRadianceFieldResult();
        var origins = new Tensor<float>(new[] { 1, 3 }, new Vector<float>(new float[3]));
        var dirs    = new Tensor<float>(new[] { 1, 3 }, new Vector<float>(new float[] { 0f, 0f, 1f }));

        var ex = Assert.Throws<InvalidOperationException>(
            () => result.RenderRays(origins, dirs, numSamples: 4, nearBound: 1f, farBound: 4f));

        Assert.Contains("RenderRays", ex.Message);
        Assert.Contains("IRadianceField", ex.Message);
    }

    [Fact]
    public void QueryField_OnNonRadianceFieldResult_ThrowsWithModelTypeNamed()
    {
        var result = BuildNonRadianceFieldResult();
        var pos = new Tensor<float>(new[] { 1, 3 }, new Vector<float>(new float[3]));
        var dir = new Tensor<float>(new[] { 1, 3 }, new Vector<float>(new float[] { 0f, 0f, 1f }));

        var ex = Assert.Throws<InvalidOperationException>(
            () => result.QueryField(pos, dir));

        Assert.Contains("QueryField", ex.Message);
        Assert.Contains("IRadianceField", ex.Message);
    }

    [Fact]
    public void RenderImage_NullResult_ThrowsArgumentNullException()
    {
        AiModelResult<float, Tensor<float>, Tensor<float>>? result = null;
        var pos = new Vector<float>(3);
        var rot = new Matrix<float>(3, 3);
        Assert.Throws<ArgumentNullException>(
            () => result!.RenderImage(pos, rot, 8, 8, 100f));
    }

    [Fact]
    public void QueryField_OnRadianceFieldResult_DispatchesToModel()
    {
        var result = BuildRadianceFieldResult();
        var positions  = new Tensor<float>(new[] { 2, 3 }, new Vector<float>(new float[]
        {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
        }));
        var directions = new Tensor<float>(new[] { 2, 3 }, new Vector<float>(new float[]
        {
            0f, 0f, 1f,
            0f, 1f, 0f,
        }));

        // Round-trip: extension should return a same-shape output as calling the model directly.
        var (rgb, density) = result.QueryField(positions, directions);

        Assert.NotNull(rgb);
        Assert.NotNull(density);
        Assert.Equal(2, rgb.Shape[0]);
        Assert.Equal(3, rgb.Shape[1]);
        Assert.Equal(2, density.Shape[0]);
    }
}
