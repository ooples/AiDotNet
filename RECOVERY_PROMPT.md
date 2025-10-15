# Multi-Agent Build Error Fix System - Recovery Prompt

## Situation
My computer shut down and I lost the multi-agent build error fix system that was helping coordinate fixes for my AiDotNet project. I need to recreate the entire system and resume where I left off.

## Request
Please recreate the complete multi-agent coordination system for fixing build errors in my AiDotNet C# project. The system should include:

### 1. Multi-Agent Architecture
Create a 4-agent system:
- **Build Checker Agent** (Supervisor): Runs builds, counts errors, validates changes, coordinates other agents
- **Agent 1 - Duplicate Resolution**: Fixes CS0101 (duplicate classes) and CS0111 (duplicate members) 
- **Agent 2 - Constraints Specialist**: Fixes CS8377 (generic constraints) and CS0104 (ambiguous types)
- **Agent 3 - Inheritance Specialist**: Fixes CS0462 (inheritance conflicts) and CS0115 (missing overrides)

### 2. Key Requirements
- **File Locking System**: Prevent agents from working on same files simultaneously
- **Communication Protocol**: CLAIM_FILE/RELEASE_FILE messaging between agents
- **Progress Tracking**: Detailed logging and state persistence
- **Build Validation**: Run builds before/after changes to prevent regressions
- **.NET Framework Compatibility**: Ensure all fixes work with .NET Framework 4.6.2+
- **Recovery Capability**: System can resume from any interruption point

### 3. Target Error Types
The system should prioritize these error patterns:
- CS8377: Generic type constraints (remove INumber<T> constraints for framework compatibility)
- CS0104: Ambiguous type references (fix Vector<> namespace conflicts)
- CS0111: Duplicate member definitions 
- CS0101: Duplicate class definitions (remove inline classes that exist in separate files)
- CS0462: Inherited member conflicts
- CS0115: Override method signature mismatches

### 4. Key Files to Target
Based on previous analysis, prioritize:
- `src/Deployment/CloudOptimizer.cs` (contains duplicate CachedModel class)
- `src/Deployment/Techniques/ModelQuantizer.cs` (contains multiple duplicate quantization strategies)
- `src/FederatedLearning/Communication/CommunicationManager.cs` (INumber constraints)
- `src/ReinforcementLearning/Models/ActorCriticModel.cs` (duplicate GetModelMetadata)
- Files with Vector<> ambiguity between AiDotNet.LinearAlgebra and System.Numerics

### 5. Implementation Details
Create these scripts:
- `multi_agent_coordinator.sh` - Main coordination script
- `build_checker_agent.sh` - Build validation and supervision
- `agent1_duplicate_resolver.sh` - Duplicate class/member resolution
- `agent2_constraints_specialist.sh` - Constraint and compatibility fixes
- `agent3_inheritance_specialist.sh` - Inheritance and override fixes
- `AGENT_COORDINATION.md` - State tracking and coordination file

### 6. Safety Features
- **Simulation Mode**: Test changes before applying them
- **Backup Strategy**: Save state before each change
- **Rollback Capability**: Undo changes if errors increase
- **Incremental Progress**: Work on small sets of files at a time

### 7. Usage Pattern
```bash
# Test the system safely
./multi_agent_coordinator.sh simulation

# Execute actual fixes
./multi_agent_coordinator.sh execute

# Resume from interruption
./multi_agent_coordinator.sh resume
```

### 8. Recovery Information
If you have access to any previous state files, restore from:
- `AGENT_COORDINATION.md` - Contains previous progress and file assignments
- `agent_coordination.log` - Contains detailed agent activity logs
- `build_output_*.txt` - Contains build error baselines

### 9. Current Project Context
- **Project**: AiDotNet - C# machine learning library
- **Targets**: .NET Framework 4.6.2, .NET 6.0, .NET 8.0
- **Current Status**: ~33 unique build errors (66 total across frameworks)
- **Previous Progress**: Reduced from 624+ errors through systematic fixes
- **Priority**: Remove INumber constraints and duplicate class definitions

### 10. Success Criteria
- Reduce build errors by 50%+ per iteration
- Maintain .NET Framework 4.6.2 compatibility
- Zero new errors introduced
- Complete documentation of all changes
- Resumable process that can handle interruptions

Please recreate this entire system, run the initial build analysis, and start the first iteration of coordinated error fixes. Focus on the highest-impact errors first (duplicate classes and incompatible constraints).

## Expected Outcome
A fully functional multi-agent system that can systematically fix build errors, track progress, coordinate between agents, and resume from any interruption point. The system should be robust enough to handle computer shutdowns and continue where it left off.