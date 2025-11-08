namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Types of edge devices for optimization targeting.
/// </summary>
public enum EdgeDeviceType
{
    /// <summary>Generic edge device</summary>
    Generic,

    /// <summary>Raspberry Pi</summary>
    RaspberryPi,

    /// <summary>NVIDIA Jetson</summary>
    Jetson,

    /// <summary>Google Coral Edge TPU</summary>
    CoralTPU,

    /// <summary>Intel Movidius</summary>
    Movidius,

    /// <summary>Microcontroller (STM32, ESP32, etc.)</summary>
    Microcontroller,

    /// <summary>Android phone</summary>
    AndroidPhone,

    /// <summary>iPhone/iPad</summary>
    iOS
}
