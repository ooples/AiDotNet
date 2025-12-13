using AiDotNet.InferenceOptimization.IR.Common;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization.IR;

/// <summary>
/// Tests for IR type system classes.
/// </summary>
public class IRTypesTests
{
    #region IRDataType Tests

    [Theory]
    [InlineData(IRDataType.Float32, true)]
    [InlineData(IRDataType.Float64, true)]
    [InlineData(IRDataType.Float16, true)]
    [InlineData(IRDataType.BFloat16, true)]
    [InlineData(IRDataType.Int32, false)]
    [InlineData(IRDataType.QInt8, false)]
    public void IsFloatingPoint_ReturnsCorrectResult(IRDataType type, bool expected)
    {
        Assert.Equal(expected, type.IsFloatingPoint());
    }

    [Theory]
    [InlineData(IRDataType.Int8, true)]
    [InlineData(IRDataType.Int32, true)]
    [InlineData(IRDataType.UInt64, true)]
    [InlineData(IRDataType.Float32, false)]
    [InlineData(IRDataType.QInt8, false)]
    public void IsInteger_ReturnsCorrectResult(IRDataType type, bool expected)
    {
        Assert.Equal(expected, type.IsInteger());
    }

    [Theory]
    [InlineData(IRDataType.QInt8, true)]
    [InlineData(IRDataType.QUInt8, true)]
    [InlineData(IRDataType.QInt4, true)]
    [InlineData(IRDataType.QInt2, true)]
    [InlineData(IRDataType.Int8, false)]
    [InlineData(IRDataType.Float32, false)]
    public void IsQuantized_ReturnsCorrectResult(IRDataType type, bool expected)
    {
        Assert.Equal(expected, type.IsQuantized());
    }

    [Theory]
    [InlineData(typeof(float), IRDataType.Float32)]
    [InlineData(typeof(double), IRDataType.Float64)]
    [InlineData(typeof(int), IRDataType.Int32)]
    [InlineData(typeof(byte), IRDataType.UInt8)]
    [InlineData(typeof(bool), IRDataType.Bool)]
    public void FromSystemType_ConvertsCorrectly(Type systemType, IRDataType expected)
    {
        Assert.Equal(expected, IRDataTypeExtensions.FromSystemType(systemType));
    }

    [Theory]
    [InlineData(IRDataType.Float32, typeof(float))]
    [InlineData(IRDataType.Float64, typeof(double))]
    [InlineData(IRDataType.Int32, typeof(int))]
    [InlineData(IRDataType.UInt8, typeof(byte))]
    [InlineData(IRDataType.Bool, typeof(bool))]
    public void ToSystemType_ConvertsCorrectly(IRDataType irType, Type expected)
    {
        Assert.Equal(expected, irType.ToSystemType());
    }

    #endregion

    #region TensorType Tests

    [Fact]
    public void TensorType_DefaultValues_AreCorrect()
    {
        var tensorType = new TensorType();

        Assert.Equal(IRDataType.Float32, tensorType.DataType);
        Assert.Empty(tensorType.Shape);
        Assert.Equal(MemoryLayout.RowMajor, tensorType.Layout);
        Assert.Equal(DeviceType.Auto, tensorType.Device);
        Assert.Null(tensorType.Quantization);
    }

    [Theory]
    [InlineData(new int[] { 2, 3, 4 }, 24)]
    [InlineData(new int[] { 10 }, 10)]
    [InlineData(new int[] { }, 1)] // scalar
    public void TensorType_NumElements_CalculatesCorrectly(int[] shape, long expected)
    {
        var tensorType = new TensorType { Shape = shape };
        Assert.Equal(expected, tensorType.NumElements);
    }

    [Fact]
    public void TensorType_NumElements_ReturnsMinusOneForDynamicShape()
    {
        var tensorType = new TensorType { Shape = new[] { 2, -1, 4 } };
        Assert.Equal(-1, tensorType.NumElements);
    }

    [Fact]
    public void TensorType_HasDynamicShape_DetectsDynamicDimensions()
    {
        var staticType = new TensorType { Shape = new[] { 2, 3, 4 } };
        var dynamicType = new TensorType { Shape = new[] { 2, -1, 4 } };

        Assert.False(staticType.HasDynamicShape);
        Assert.True(dynamicType.HasDynamicShape);
    }

    [Theory]
    [InlineData(IRDataType.Float32, 4)]
    [InlineData(IRDataType.Float64, 8)]
    [InlineData(IRDataType.Float16, 2)]
    [InlineData(IRDataType.Int8, 1)]
    [InlineData(IRDataType.Complex128, 16)]
    public void TensorType_ElementSize_ReturnsCorrectSize(IRDataType dataType, int expected)
    {
        var tensorType = new TensorType { DataType = dataType };
        Assert.Equal(expected, tensorType.ElementSize);
    }

    [Fact]
    public void TensorType_TotalBytes_CalculatesCorrectly()
    {
        var tensorType = new TensorType
        {
            DataType = IRDataType.Float32,
            Shape = new[] { 2, 3, 4 }
        };

        Assert.Equal(24 * 4, tensorType.TotalBytes); // 24 elements * 4 bytes
    }

    [Theory]
    [InlineData(new[] { 3, 4 }, new[] { 3, 4 }, true)]
    [InlineData(new[] { 1, 4 }, new[] { 3, 4 }, true)]
    [InlineData(new[] { 3, 1 }, new[] { 3, 4 }, true)]
    [InlineData(new[] { 4 }, new[] { 3, 4 }, true)]
    [InlineData(new[] { 3, 4 }, new[] { 2, 4 }, false)]
    public void TensorType_IsBroadcastCompatible_ReturnsCorrectResult(int[] shape1, int[] shape2, bool expected)
    {
        var type1 = new TensorType { Shape = shape1 };
        var type2 = new TensorType { Shape = shape2 };

        Assert.Equal(expected, type1.IsBroadcastCompatible(type2));
    }

    [Fact]
    public void TensorType_Clone_CreatesIndependentCopy()
    {
        var original = new TensorType
        {
            DataType = IRDataType.Float32,
            Shape = new[] { 2, 3 },
            Layout = MemoryLayout.NCHW,
            Device = DeviceType.GPU
        };

        var clone = original.Clone();

        Assert.Equal(original.DataType, clone.DataType);
        Assert.Equal(original.Shape, clone.Shape);
        Assert.Equal(original.Layout, clone.Layout);
        Assert.Equal(original.Device, clone.Device);

        // Modify clone and verify original is unchanged
        clone.Shape[0] = 999;
        Assert.NotEqual(original.Shape[0], clone.Shape[0]);
    }

    [Fact]
    public void TensorType_ToString_ReturnsFormattedString()
    {
        var tensorType = new TensorType
        {
            DataType = IRDataType.Float32,
            Shape = new[] { 2, 3, 4 },
            Device = DeviceType.GPU
        };

        var str = tensorType.ToString();
        Assert.Contains("Float32", str);
        Assert.Contains("2", str);
        Assert.Contains("GPU", str);
    }

    #endregion

    #region QuantizationParams Tests

    [Fact]
    public void QuantizationParams_DefaultValues_AreCorrect()
    {
        var qParams = new QuantizationParams();

        Assert.Equal(1.0, qParams.Scale);
        Assert.Equal(0, qParams.ZeroPoint);
        Assert.False(qParams.PerChannel);
        Assert.Equal(-1, qParams.QuantizationAxis);
    }

    [Fact]
    public void QuantizationParams_PerChannel_CanBeConfigured()
    {
        var qParams = new QuantizationParams
        {
            PerChannel = true,
            QuantizationAxis = 0,
            PerChannelScales = new[] { 0.1, 0.2, 0.3 },
            PerChannelZeroPoints = new[] { 0, 1, 2 }
        };

        Assert.True(qParams.PerChannel);
        Assert.Equal(0, qParams.QuantizationAxis);
        Assert.Equal(3, qParams.PerChannelScales!.Length);
        Assert.Equal(3, qParams.PerChannelZeroPoints!.Length);
    }

    #endregion

    #region MemoryLayout Tests

    [Fact]
    public void MemoryLayout_AllValuesAreDefined()
    {
        Assert.True(Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.RowMajor));
        Assert.True(Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.ColumnMajor));
        Assert.True(Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.NCHW));
        Assert.True(Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.NHWC));
        Assert.True(Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.Tiled4x4));
        Assert.True(Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.Blocked));
    }

    #endregion

    #region DeviceType Tests

    [Fact]
    public void DeviceType_AllValuesAreDefined()
    {
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.CPU));
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.GPU));
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.TPU));
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.NPU));
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.FPGA));
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.Auto));
        Assert.True(Enum.IsDefined(typeof(DeviceType), DeviceType.Any));
    }

    #endregion
}
