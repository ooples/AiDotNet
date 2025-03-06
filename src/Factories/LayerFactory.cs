global using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Factories;

public static class LayerFactory<T>
{
    //public static ILayer<T> CreateLayer(LayerType layerType, LayerOptions<T> options)
    //{
    //    return layerType switch
    //    {
    //        LayerType.Pooling => new PoolingLayer<T>(options.PoolSize, options.Stride, options.PoolingType),
    //        LayerType.Convolutional => CreateConvolutionalLayer(options),
    //        _ => throw new ArgumentException("Unsupported layer type", nameof(layerType)),
    //    };
    //}

    //private static ConvolutionalLayer<T> CreateConvolutionalLayer(LayerOptions<T> options)
    //{
    //    if (options.VectorActivationFunction != null)
    //    {
    //        return new ConvolutionalLayer<T>(
    //            options.InputChannels, 
    //            options.OutputChannels, 
    //            options.KernelSize, 
    //            options.Stride, 
    //            options.Padding, 
    //            options.VectorActivationFunction);
    //    }
    //    else if (options.ScalarActivationFunction != null)
    //    {
    //        return new ConvolutionalLayer<T>(
    //            options.InputChannels, 
    //            options.OutputChannels, 
    //            options.KernelSize, 
    //            options.Stride, 
    //            options.Padding, 
    //            options.ScalarActivationFunction);
    //    }

    //    return new ConvolutionalLayer<T>(
    //            options.InputChannels, 
    //            options.OutputChannels, 
    //            options.KernelSize, 
    //            options.Stride, 
    //            options.Padding, 
    //            (IActivationFunction<T>)new IdentityActivation<T>());
    //}
}