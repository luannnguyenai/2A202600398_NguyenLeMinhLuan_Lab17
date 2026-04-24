"""
agent/prompt.py
"""
from typing import Any

def build_prompt(state: dict[str, Any]) -> list[dict[str, str]]:
    profile = state.get("user_profile", {})
    if profile:
        profile_str = "\n".join(f"- {k}: {v}" for k, v in profile.items())
    else:
        profile_str = "(none)"

    episodes = state.get("episodes", [])
    if episodes:
        episode_lines = []
        for ep in episodes:
            meta = ep.get("metadata", {})
            lesson = meta.get("lesson") or "No lesson"
            summary = ep.get("content", "").split(" | ")[0] if " | " in ep.get("content", "") else ep.get("content", "")
            episode_lines.append(f"- ID: {meta.get('id', 'unknown')} | {summary} | Lesson: {lesson}")
        episodes_str = "\n".join(episode_lines)
    else:
        episodes_str = "(none)"

    semantic = state.get("semantic_hits", [])
    if semantic:
        sem_lines = [f"{i+1}. {hit.get('content', '')}" for i, hit in enumerate(semantic)]
        semantic_str = "\n".join(sem_lines)
    else:
        semantic_str = "(none)"

    short_term = state.get("messages", [])
    if short_term:
        st_lines = []
        for m in short_term:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            st_lines.append(f"{role.capitalize()}: {content}")
        recent_str = "\n".join(st_lines)
    else:
        recent_str = "(none)"

    system_content = f"""[SYSTEM INSTRUCTIONS]
You are a helpful AI assistant with memory. Use the context below to answer.
Always respond naturally based on the current user input and memory context.

[USER PROFILE]
{profile_str}

[RELEVANT EPISODES]
{episodes_str}

[SEMANTIC CONTEXT]
{semantic_str}

[RECENT CONVERSATION]
{recent_str}
"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": state["user_input"]}
    ]
