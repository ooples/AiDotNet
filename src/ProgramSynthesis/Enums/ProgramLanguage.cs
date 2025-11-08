namespace AiDotNet.ProgramSynthesis.Enums;

/// <summary>
/// Defines the programming languages that can be synthesized or processed.
/// </summary>
/// <remarks>
/// <para>
/// This enumeration specifies the target programming languages for code synthesis,
/// translation, and analysis tasks. Each language has its own syntax, semantics,
/// and typical use cases.
/// </para>
/// <para><b>For Beginners:</b> This lists the different programming languages the system can work with.
///
/// Just like human languages (English, Spanish, French), there are many programming languages
/// (Python, C#, Java). Each has its own rules and is better suited for different tasks.
/// This enum helps the system know which language you want to work with.
/// </para>
/// </remarks>
public enum ProgramLanguage
{
    /// <summary>
    /// Python programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Python is a high-level, interpreted language known for its readability and extensive
    /// ecosystem. It's widely used in data science, machine learning, web development,
    /// and automation.
    /// </para>
    /// <para><b>For Beginners:</b> Python is known for being easy to read and beginner-friendly.
    ///
    /// It's popular for AI, data analysis, and general programming. Code looks clean and
    /// is relatively easy to understand, making it a great choice for many applications.
    /// </para>
    /// </remarks>
    Python,

    /// <summary>
    /// C# programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// C# is a modern, object-oriented language developed by Microsoft. It's used for
    /// Windows applications, game development (Unity), web services, and enterprise software.
    /// </para>
    /// <para><b>For Beginners:</b> C# is a powerful language used for many types of applications.
    ///
    /// It's particularly popular for Windows programs, games (especially with Unity),
    /// and business applications. It has strong typing which helps catch errors early.
    /// </para>
    /// </remarks>
    CSharp,

    /// <summary>
    /// Java programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Java is a widely-used, object-oriented language known for its "write once, run anywhere"
    /// philosophy. It's popular for enterprise applications, Android development, and large-scale systems.
    /// </para>
    /// <para><b>For Beginners:</b> Java is one of the most popular languages in the world.
    ///
    /// It's used for Android apps, large business systems, and web applications. Code written
    /// in Java can run on different types of computers without modification.
    /// </para>
    /// </remarks>
    Java,

    /// <summary>
    /// JavaScript programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// JavaScript is the primary language for web browser programming and has expanded to
    /// server-side development (Node.js). It's essential for interactive web applications
    /// and is one of the most widely used languages.
    /// </para>
    /// <para><b>For Beginners:</b> JavaScript makes websites interactive and dynamic.
    ///
    /// It runs in web browsers and powers most of the interactive features you see on websites.
    /// It's also used for server-side programming and mobile app development.
    /// </para>
    /// </remarks>
    JavaScript,

    /// <summary>
    /// TypeScript programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TypeScript is a superset of JavaScript that adds static typing. It helps catch
    /// errors during development and is increasingly popular for large JavaScript applications.
    /// </para>
    /// <para><b>For Beginners:</b> TypeScript is JavaScript with extra type checking.
    ///
    /// It helps prevent bugs by checking your code before it runs. Think of it as JavaScript
    /// with helpful guardrails that catch mistakes early.
    /// </para>
    /// </remarks>
    TypeScript,

    /// <summary>
    /// C++ programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// C++ is a powerful, high-performance language used for system software, game engines,
    /// and applications where speed is critical. It provides low-level control while supporting
    /// high-level abstractions.
    /// </para>
    /// <para><b>For Beginners:</b> C++ is known for speed and control over computer resources.
    ///
    /// It's used when performance is critical, like in game engines, operating systems,
    /// and high-frequency trading systems. It's more complex but very powerful.
    /// </para>
    /// </remarks>
    CPlusPlus,

    /// <summary>
    /// C programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// C is a low-level language that provides fine-grained control over computer resources.
    /// It's used for operating systems, embedded systems, and performance-critical applications.
    /// </para>
    /// <para><b>For Beginners:</b> C is a foundational language that's close to how computers work.
    ///
    /// Many other languages are based on C. It's used for operating systems and programs
    /// that need direct control over computer hardware.
    /// </para>
    /// </remarks>
    C,

    /// <summary>
    /// Go (Golang) programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Go is a modern language designed at Google for building scalable network services
    /// and concurrent applications. It emphasizes simplicity and has built-in concurrency support.
    /// </para>
    /// <para><b>For Beginners:</b> Go is designed for building fast, reliable network services.
    ///
    /// It's simpler than some languages but still powerful, especially good for programs
    /// that need to do many things at once (like web servers handling many users).
    /// </para>
    /// </remarks>
    Go,

    /// <summary>
    /// Rust programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Rust is a systems programming language focused on safety, concurrency, and performance.
    /// It prevents many common bugs through its unique ownership system.
    /// </para>
    /// <para><b>For Beginners:</b> Rust helps you write safe and fast programs.
    ///
    /// It has special rules that prevent common programming errors (like memory bugs)
    /// while still being very fast. Popular for system programming and security-critical applications.
    /// </para>
    /// </remarks>
    Rust,

    /// <summary>
    /// SQL (Structured Query Language) for database operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SQL is a domain-specific language for managing and querying relational databases.
    /// It's essential for data manipulation and retrieval in database systems.
    /// </para>
    /// <para><b>For Beginners:</b> SQL is for working with databases.
    ///
    /// It's not a general programming language but a specialized language for storing,
    /// retrieving, and managing data in databases. Used everywhere data is stored.
    /// </para>
    /// </remarks>
    SQL,

    /// <summary>
    /// Generic or language-agnostic representation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This option is used when working with abstract program representations that aren't
    /// tied to a specific programming language, or when the language is not yet determined.
    /// </para>
    /// <para><b>For Beginners:</b> Generic means not specific to any one language.
    ///
    /// Sometimes you want to work with the logic of a program without worrying about
    /// which language it will eventually be written in. This option represents that.
    /// </para>
    /// </remarks>
    Generic
}
