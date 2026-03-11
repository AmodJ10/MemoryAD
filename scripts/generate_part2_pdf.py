import os
import glob
from pathlib import Path

# Current session's brain directory
brain_dir = Path(r"C:\Users\pramo\.gemini\antigravity\brain\7357876d-c5e6-43d2-ab66-a4ef72a698d5")
logs_dir = sorted(glob.glob(str(brain_dir / ".system_generated" / "logs" / "*.txt")))

# Artifacts generated in this session
artifacts = [
    "task.md",
    "implementation_plan.md",
    "walkthrough.md"
]

output_md = brain_dir / "comprehensive_report.md"

with open(output_md, "w", encoding="utf-8") as out:
    out.write("---\n")
    out.write("title: MemoryAD Part 2 Comprehensive Archive (with Model Thinking!)\n")
    out.write("author: Antigravity AI & User\n")
    out.write("geometry: margin=1in\n")
    out.write("toc: true\n")
    out.write("colorlinks: true\n")
    out.write("---\n\n")

    out.write("# Part 1: Project Artifacts & Reports (Phase 2)\n\n")
    for art in artifacts:
        art_path = brain_dir / art
        if art_path.exists():
            out.write(f"## Artifact: {art}\n\n")
            with open(art_path, "r", encoding="utf-8") as f:
                content = f.read()
                content = content.replace("file:///", "")
                out.write(content)
            out.write("\n\n\\newpage\n\n")

    out.write("# Part 2: Conversation Logs & Reasoning (Phase 2)\n\n")
    out.write("*Note: This section includes the raw tool calls and internal model thinking (reasoning) steps that powered the project finalization.* \n\n")
    for log in logs_dir:
        log_name = os.path.basename(log)
        out.write(f"## Session Log: {log_name}\n\n")
        
        try:
            with open(log, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            with open(log, "r", encoding="latin-1") as f:
                content = f.read()
                
        # String replacement logic to highlight thought process
        content = content.replace("<thought>", "\n[INTERNAL REASONING / THOUGHT]:\n")
        content = content.replace("</thought>", "\n[END REASONING]\n")
        
        out.write("```text\n")
        out.write(content.replace('`', "'")) 
        out.write("\n```\n\n\\newpage\n\n")

print(f"Created {output_md}")
