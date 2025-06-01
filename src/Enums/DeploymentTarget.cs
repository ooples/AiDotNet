namespace AiDotNet.Enums
{
    /// <summary>
    /// Specifies the deployment target for model optimization
    /// </summary>
    public enum DeploymentTarget
    {
        /// <summary>
        /// Desktop/server deployment
        /// </summary>
        Server,
        
        /// <summary>
        /// Cloud deployment (AWS, Azure, GCP)
        /// </summary>
        CloudDeployment,
        
        /// <summary>
        /// Mobile devices (iOS, Android)
        /// </summary>
        Mobile,
        
        /// <summary>
        /// Edge devices (IoT, embedded)
        /// </summary>
        Edge,
        
        /// <summary>
        /// Web browser deployment
        /// </summary>
        WebAssembly,
        
        /// <summary>
        /// Specialized hardware (TPU, FPGA)
        /// </summary>
        SpecializedHardware
    }
}