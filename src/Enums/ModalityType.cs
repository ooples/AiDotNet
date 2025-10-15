namespace AiDotNet.Enums
{
    /// <summary>
    /// Types of modalities supported by multimodal models
    /// </summary>
    public enum ModalityType
    {
        /// <summary>
        /// Text data modality
        /// </summary>
        Text,

        /// <summary>
        /// Image data modality
        /// </summary>
        Image,

        /// <summary>
        /// Audio data modality
        /// </summary>
        Audio,

        /// <summary>
        /// Numerical/quantitative data modality
        /// </summary>
        Numerical,

        /// <summary>
        /// Video data modality
        /// </summary>
        Video,

        /// <summary>
        /// Tabular/structured data modality
        /// </summary>
        Tabular,

        /// <summary>
        /// Time series data modality
        /// </summary>
        TimeSeries,

        /// <summary>
        /// Graph/network data modality
        /// </summary>
        Graph,

        /// <summary>
        /// 3D point cloud data modality
        /// </summary>
        PointCloud
    }
}