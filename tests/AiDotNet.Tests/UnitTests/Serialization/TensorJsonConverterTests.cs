namespace AiDotNet.Tests.UnitTests.Serialization;

using System.Collections.Generic;
using System.IO;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

public class TensorJsonConverterTests
{
    [Fact]
    public void CanConvert_ReturnsTrue_ForTensorAndDerivedTensor()
    {
        var converter = new TensorJsonConverter();

        Assert.True(converter.CanConvert(typeof(Tensor<double>)));
        Assert.True(converter.CanConvert(typeof(NoVectorConstructorTensor<double>)));
    }

    [Fact]
    public void CanConvert_ReturnsFalse_ForNonTensorType()
    {
        var converter = new TensorJsonConverter();

        Assert.False(converter.CanConvert(typeof(string)));
    }

    [Fact]
    public void WriteJson_WritesNull_WhenValueIsNull()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();
        var writer = new JTokenWriter();

        converter.WriteJson(writer, value: null, serializer);

        Assert.NotNull(writer.Token);
        Assert.Equal(JTokenType.Null, writer.Token!.Type);
    }

    [Fact]
    public void WriteJson_Throws_WhenMissingRequiredTensorProperties()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();
        var writer = new JTokenWriter();

        Assert.Throws<JsonSerializationException>(() => converter.WriteJson(writer, new object(), serializer));
    }

    [Fact]
    public void ReadJson_ReturnsNull_WhenJsonNull()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();

        using var reader = new JsonTextReader(new StringReader("null"));
        Assert.True(reader.Read());

        var result = converter.ReadJson(reader, typeof(Tensor<double>), existingValue: null, serializer);

        Assert.Null(result);
    }

    [Fact]
    public void ReadJson_Throws_WhenShapeMissing()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();

        using var reader = new JsonTextReader(new StringReader("{\"data\":[1.0,2.0]}"));
        Assert.True(reader.Read());

        Assert.Throws<JsonSerializationException>(() =>
            converter.ReadJson(reader, typeof(Tensor<double>), existingValue: null, serializer));
    }

    [Fact]
    public void ReadJson_Throws_WhenDataMissing()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();

        using var reader = new JsonTextReader(new StringReader("{\"shape\":[2]}"));
        Assert.True(reader.Read());

        Assert.Throws<JsonSerializationException>(() =>
            converter.ReadJson(reader, typeof(Tensor<double>), existingValue: null, serializer));
    }

    [Fact]
    public void ReadJson_Throws_WhenDataLengthMismatch()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();

        using var reader = new JsonTextReader(new StringReader("{\"shape\":[2,2],\"data\":[1.0,2.0,3.0]}"));
        Assert.True(reader.Read());

        Assert.Throws<JsonSerializationException>(() =>
            converter.ReadJson(reader, typeof(Tensor<double>), existingValue: null, serializer));
    }

    [Fact]
    public void RoundTrip_SerializesAndDeserializesTensor()
    {
        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { new TensorJsonConverter() }
        };

        var tensor = new Tensor<double>(
            new[] { 2, 2 },
            new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));

        var json = JsonConvert.SerializeObject(tensor, settings);
        var parsed = JObject.Parse(json);

        Assert.NotNull(parsed["shape"]);
        Assert.NotNull(parsed["data"]);

        var roundTrip = JsonConvert.DeserializeObject<Tensor<double>>(json, settings);

        Assert.NotNull(roundTrip);
        Assert.Equal(tensor.Shape, roundTrip!.Shape);
        Assert.Equal(tensor.ToArray(), roundTrip.ToArray());
    }

    [Fact]
    public void ReadJson_UsesFallbackConstructor_WhenVectorConstructorMissing()
    {
        var settings = new JsonSerializerSettings
        {
            Converters = new List<JsonConverter> { new TensorJsonConverter() }
        };

        var tensor = new Tensor<double>(
            new[] { 2, 2 },
            new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));

        var json = JsonConvert.SerializeObject(tensor, settings);
        var fallback = JsonConvert.DeserializeObject<NoVectorConstructorTensor<double>>(json, settings);

        Assert.NotNull(fallback);
        Assert.Equal(tensor.Shape, fallback!.Shape);
        Assert.Equal(tensor.ToArray(), fallback.ToArray());
    }

    [Fact]
    public void ReadJson_Throws_WhenNoSuitableConstructor()
    {
        var converter = new TensorJsonConverter();
        var serializer = JsonSerializer.CreateDefault();

        using var reader = new JsonTextReader(new StringReader("{\"shape\":[1],\"data\":[1.0]}"));
        Assert.True(reader.Read());

        Assert.Throws<JsonSerializationException>(() =>
            converter.ReadJson(reader, typeof(List<double>), existingValue: null, serializer));
    }
}

