"""The default agent roster. Each agent gets a distinct role and a distinct tool
subset -- that asymmetry is what makes the communication graph do real work."""
from agent import AgentSpec

BB = ["blackboard_read", "blackboard_write"]

DEFAULT_TEAM = [
    AgentSpec(
        name="Coordinator", role="coordinator",
        instructions=("You break the task into sub-questions, track what the team has "
                      "established, and identify what is still missing. You do not have "
                      "research tools of your own -- rely on what teammates publish."),
        tools=BB + ["list_files", "read_file"]),
    AgentSpec(
        name="Researcher", role="knowledge-base researcher",
        instructions=("You answer questions from the internal knowledge base. Always cite the "
                      "source file for any claim you make. If the KB does not cover something, "
                      "say so instead of guessing."),
        tools=BB + ["kb_search"]),
    AgentSpec(
        name="Analyst", role="quantitative analyst",
        instructions=("You answer questions with numbers from the company database. Write SQL, "
                      "run it, and compute derived figures with the calculator rather than doing "
                      "arithmetic in your head. Always show the query you ran."),
        tools=BB + ["sql_query", "calculator"]),
    AgentSpec(
        name="Critic", role="verifier",
        instructions=("You check other agents' claims. Re-run their numbers and re-read their "
                      "sources yourself. State explicitly which claims you verified, which you "
                      "could not, and which are wrong. Do not simply agree."),
        tools=BB + ["sql_query", "calculator", "kb_search"]),
    AgentSpec(
        name="Writer", role="technical writer",
        instructions=("You turn the team's verified findings into a short written deliverable "
                      "and save it to the workspace with write_file. Only include claims some "
                      "agent has actually supported with a tool result."),
        tools=BB + ["write_file", "read_file", "send_email"]),
]


def build_team(n: int | None = None):
    """Return the roster, trimmed or cycled to n agents."""
    if n is None or n == len(DEFAULT_TEAM):
        return list(DEFAULT_TEAM)
    if n < len(DEFAULT_TEAM):
        return list(DEFAULT_TEAM[:n])
    out = list(DEFAULT_TEAM)
    while len(out) < n:                       # duplicate roles with distinct names
        base = DEFAULT_TEAM[len(out) % len(DEFAULT_TEAM)]
        out.append(AgentSpec(f"{base.name}-{len(out)}", base.role, base.instructions, base.tools))
    return out
