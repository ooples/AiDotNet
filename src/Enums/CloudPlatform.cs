namespace AiDotNet.Enums;

/// <summary>
/// Defines supported cloud platforms for model deployment and distributed training.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cloud platforms are like powerful computer centers you can rent. 
/// This enum lists the different cloud services where you can deploy and run your models.
/// </para>
/// </remarks>
public enum CloudPlatform
{
    /// <summary>
    /// No cloud platform (local deployment only).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Running everything on your own computer or servers - no cloud 
    /// services involved.
    /// </remarks>
    None,

    /// <summary>
    /// Amazon Web Services.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Amazon's cloud platform - one of the largest, offering services 
    /// like SageMaker for machine learning.
    /// </remarks>
    AWS,

    /// <summary>
    /// Microsoft Azure.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Microsoft's cloud platform - well integrated with Windows and 
    /// offers Azure Machine Learning services.
    /// </remarks>
    Azure,

    /// <summary>
    /// Google Cloud Platform.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Google's cloud platform - known for strong AI/ML capabilities
    /// and services like Vertex AI.
    /// </remarks>
    GoogleCloud,

    /// <summary>
    /// Google Cloud Platform (GCP) - alias for GoogleCloud.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> GCP is the abbreviated name for Google Cloud Platform.
    /// </remarks>
    GCP,

    /// <summary>
    /// IBM Cloud.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> IBM's cloud platform - offers Watson services for AI and 
    /// enterprise-focused solutions.
    /// </remarks>
    IBMCloud,

    /// <summary>
    /// Oracle Cloud Infrastructure.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Oracle's cloud platform - focused on enterprise database and 
    /// application services.
    /// </remarks>
    OracleCloud,

    /// <summary>
    /// Alibaba Cloud.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> China's largest cloud platform - offers machine learning services 
    /// similar to other major providers.
    /// </remarks>
    AlibabaCloud,

    /// <summary>
    /// Salesforce Cloud.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Salesforce's platform - specialized for CRM and business applications 
    /// with Einstein AI capabilities.
    /// </remarks>
    Salesforce,

    /// <summary>
    /// Digital Ocean.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A simpler, developer-friendly cloud platform - good for smaller 
    /// projects and startups.
    /// </remarks>
    DigitalOcean,

    /// <summary>
    /// Linode (Akamai).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A straightforward cloud platform now owned by Akamai - known for 
    /// simplicity and good pricing.
    /// </remarks>
    Linode,

    /// <summary>
    /// Vultr cloud platform.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A cloud platform offering high-performance instances - popular for 
    /// GPU-based machine learning workloads.
    /// </remarks>
    Vultr,

    /// <summary>
    /// Private cloud infrastructure.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Your organization's own cloud setup - like having your own data 
    /// center with cloud-like capabilities.
    /// </remarks>
    PrivateCloud,

    /// <summary>
    /// Hybrid cloud (mix of public and private).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Using both public cloud services and private infrastructure - 
    /// keeping sensitive data private while using public cloud for scaling.
    /// </remarks>
    HybridCloud,

    /// <summary>
    /// Multi-cloud (using multiple providers).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Using services from multiple cloud providers - like shopping at 
    /// different stores for the best deals on each item.
    /// </remarks>
    MultiCloud,

    /// <summary>
    /// Edge cloud infrastructure.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Cloud services located close to where data is generated - reduces 
    /// delays for real-time applications.
    /// </remarks>
    EdgeCloud,

    /// <summary>
    /// Custom cloud platform.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Any other cloud platform not listed here - allows flexibility for 
    /// new or specialized providers.
    /// </remarks>
    Custom
}