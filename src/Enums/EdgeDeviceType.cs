namespace AiDotNet.Enums;

/// <summary>
/// Types of edge devices for optimization targeting.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Edge devices are small computers that run AI models locally instead of
/// in the cloud. Each type of edge device has different hardware capabilities, so the library
/// can optimize your model specifically for the device you're targeting:
///
/// - **Generic**: Works on any edge device, but not specifically optimized
/// - **RaspberryPi**: Popular single-board computer for hobbyists and education
/// - **Jetson**: NVIDIA's edge AI platform with GPU acceleration
/// - **CoralTPU**: Google's Edge TPU for fast AI inference
/// - **Movidius**: Intel's vision processing unit for cameras and drones
/// - **Microcontroller**: Very small devices like Arduino or ESP32 with minimal resources
/// - **AndroidPhone**: Android smartphones and tablets
/// - **iOS**: iPhones and iPads
///
/// Choosing the right device type helps the library apply device-specific optimizations.
/// </remarks>
public enum EdgeDeviceType
{
    /// <summary>Generic edge device - works everywhere but not specifically optimized</summary>
    Generic,

    /// <summary>Raspberry Pi - popular single-board computer ($35-$100)</summary>
    RaspberryPi,

    /// <summary>NVIDIA Jetson - edge AI platform with GPU acceleration ($99-$500)</summary>
    Jetson,

    /// <summary>Google Coral Edge TPU - specialized AI accelerator ($25-$150)</summary>
    CoralTPU,

    /// <summary>Intel Movidius - vision processing unit for cameras and drones</summary>
    Movidius,

    /// <summary>Microcontroller (STM32, ESP32, etc.) - very limited resources</summary>
    Microcontroller,

    /// <summary>Android phone - smartphone with potential GPU/NPU acceleration</summary>
    AndroidPhone,

    /// <summary>iPhone/iPad - iOS devices with Neural Engine</summary>
    iOS
}
