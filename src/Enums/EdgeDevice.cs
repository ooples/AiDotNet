namespace AiDotNet.Enums;

/// <summary>
/// Defines types of edge devices for model deployment.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Edge devices are computers that run AI models close to where data is 
/// collected, instead of sending everything to the cloud. This enum lists different types of 
/// edge hardware.
/// </para>
/// </remarks>
public enum EdgeDevice
{
    /// <summary>
    /// No edge device (cloud or local deployment).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Not using edge computing - running models on regular computers 
    /// or in the cloud instead.
    /// </remarks>
    None,

    /// <summary>
    /// Mobile device deployment (phones, tablets, etc.).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Running AI models on mobile devices - like smartphones and tablets.
    /// This includes face recognition, voice assistants, and other apps that work offline.
    /// </remarks>
    Mobile,

    /// <summary>
    /// Mobile phone deployment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Running AI models directly on smartphones - like face recognition
    /// or voice assistants that work offline.
    /// </remarks>
    MobilePhone,

    /// <summary>
    /// Tablet device deployment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Running models on tablets - similar to phones but with more 
    /// screen space and sometimes more processing power.
    /// </remarks>
    Tablet,

    /// <summary>
    /// Raspberry Pi deployment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A small, affordable computer popular for hobby projects and 
    /// IoT applications - about the size of a credit card.
    /// </remarks>
    RaspberryPi,

    /// <summary>
    /// NVIDIA Jetson series.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Specialized small computers from NVIDIA designed for AI - much 
    /// more powerful than Raspberry Pi for machine learning tasks.
    /// </remarks>
    NvidiaJetson,

    /// <summary>
    /// Intel Neural Compute Stick.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A USB stick that adds AI processing capabilities to any computer - 
    /// plug it in to accelerate inference.
    /// </remarks>
    IntelNCS,

    /// <summary>
    /// Google Coral devices.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Google's edge AI hardware with specialized chips for fast, 
    /// efficient machine learning inference.
    /// </remarks>
    GoogleCoral,

    /// <summary>
    /// Arduino with ML capabilities.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Simple microcontrollers that can run tiny machine learning models - 
    /// good for basic sensors and controls.
    /// </remarks>
    Arduino,

    /// <summary>
    /// IoT sensors with processing.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Smart sensors that can process data locally - like security cameras 
    /// that detect motion without cloud connection.
    /// </remarks>
    IoTSensor,

    /// <summary>
    /// Smartwatch deployment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Running models on wearable devices - like fitness tracking or 
    /// health monitoring on your wrist.
    /// </remarks>
    Smartwatch,

    /// <summary>
    /// Automotive ECU deployment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Car computers that run AI for features like lane detection, 
    /// automatic braking, or self-driving capabilities.
    /// </remarks>
    AutomotiveECU,

    /// <summary>
    /// Drone onboard computer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Computers on drones that process vision and sensor data for 
    /// navigation and obstacle avoidance.
    /// </remarks>
    Drone,

    /// <summary>
    /// Industrial PLC with ML.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Factory control computers enhanced with AI - for quality control, 
    /// predictive maintenance, or process optimization.
    /// </remarks>
    IndustrialPLC,

    /// <summary>
    /// Smart camera systems.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Cameras with built-in AI processing - can recognize objects or 
    /// people without sending video elsewhere.
    /// </remarks>
    SmartCamera,

    /// <summary>
    /// FPGA-based edge device.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Programmable chips that can be customized for specific AI tasks - 
    /// very fast but require specialized knowledge.
    /// </remarks>
    FPGA,

    /// <summary>
    /// Custom edge hardware.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Any other edge device not listed here - allows for specialized 
    /// or proprietary hardware.
    /// </remarks>
    Custom
}