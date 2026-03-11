import os
import glob
from pathlib import Path

# Paths to the brain directories for Part 1 and Part 2
brain_dir_part1 = Path(r"C:\Users\pramo\.gemini\antigravity\brain\e140f1de-87c6-46e2-9816-828112e00eed")
brain_dir_part2 = Path(r"C:\Users\pramo\.gemini\antigravity\brain\7357876d-c5e6-43d2-ab66-a4ef72a698d5")

logs_dir_part1 = sorted(glob.glob(str(brain_dir_part1 / ".system_generated" / "logs" / "*.txt")))
logs_dir_part2 = sorted(glob.glob(str(brain_dir_part2 / ".system_generated" / "logs" / "*.txt")))

artifacts_part1 = [
    "phase1_step1.1_taxonomy.md",
    "phase1_step1.2_review_papers.md",
    "phase1_step1.3_gap_analysis.md",
    "phase2_gap_selection.md",
    "phase3_gap_validation.md",
    "phase4_deep_dive.md",
    "phase5_novel_idea.md",
    "literature_review_report.md",
    "implementation_roadmap.md",
    "implementation_plan.md",
    "task.md",
    "walkthrough.md"
]

artifacts_part2 = [
    "task.md",
    "implementation_plan.md",
    "walkthrough.md"
]

output_md = brain_dir_part2 / "comprehensive_report_combined.md"

with open(output_md, "w", encoding="utf-8") as out:
    out.write("---\n")
    out.write("title: MemoryAD Comprehensive Archive (Part 1 & 2 combined)\n")
    out.write("author: Antigravity AI & User\n")
    out.write("geometry: margin=1in\n")
    out.write("toc: true\n")
    out.write("colorlinks: true\n")
    out.write("---\n\n")

    out.write("# Part 1: Project Artifacts & Reports (Phase 1)\n\n")
    for art in artifacts_part1:
        art_path = brain_dir_part1 / art
        if art_path.exists():
            out.write(f"## Artifact: {art}\n\n")
            with open(art_path, "r", encoding="utf-8") as f:
                content = f.read()
                content = content.replace("file:///", "")
                out.write(content)
            out.write("\n\n\\newpage\n\n")
            
    out.write("# Part 2: Project Artifacts & Reports (Phase 2)\n\n")
    for art in artifacts_part2:
        art_path = brain_dir_part2 / art
        if art_path.exists():
            out.write(f"## Artifact: {art}\n\n")
            with open(art_path, "r", encoding="utf-8") as f:
                content = f.read()
                content = content.replace("file:///", "")
                out.write(content)
            out.write("\n\n\\newpage\n\n")

    out.write("# Part 3: Conversation Logs & Reasoning (Phase 1)\n\n")
    out.write("*Note: This section includes the raw tool calls and internal model thinking (reasoning) steps that powered the project.* \n\n")
    for log in logs_dir_part1:
        log_name = os.path.basename(log)
        out.write(f"## Session Log: {log_name}\n\n")
        
        try:
            with open(log, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            with open(log, "r", encoding="latin-1") as f:
                content = f.read()
                
        # To ensure the markdown parser doesn't swallow HTML-like tags (like <thought>), 
        # we will render the whole log as a text block, but we'll prepend an explicit marker for thoughts
        # so they are very visible.
        content = content.replace("<thought>", "\n[INTERNAL REASONING / THOUGHT]:\n")
        content = content.replace("</thought>", "\n[END REASONING]\n")
        
        out.write("```text\n")
        out.write(content.replace('`', "'")) 
        out.write("\n```\n\n\\newpage\n\n")
        
    out.write("# Part 4: Conversation Logs & Reasoning (Phase 2)\n\n")
    out.write("*Note: This section includes the raw tool calls and internal model thinking (reasoning) steps that powered the project.* \n\n")
    for log in logs_dir_part2:
        log_name = os.path.basename(log)
        out.write(f"## Session Log: {log_name}\n\n")
        
        try:
            with open(log, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            with open(log, "r", encoding="latin-1") as f:
                content = f.read()
                
        content = content.replace("<thought>", "\n[INTERNAL REASONING / THOUGHT]:\n")
        content = content.replace("</thought>", "\n[END REASONING]\n")
        
        out.write("```text\n")
        out.write(content.replace('`', "'")) 
        out.write("\n```\n\n\\newpage\n\n")

print(f"Created {output_md}")
