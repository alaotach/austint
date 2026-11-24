"""Step 3 Demo: Username expansion to confirmed accounts and extracted lead entities."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from osint_agent.core.agent import OSINTAgent, export_graph_to_json
from osint_agent.ml.resolver import LearnedResolver
from osint_agent.core.schema import EntityType, Scope


async def run_demo(username: str = "torvalds") -> None:
    print("\n" + "=" * 70)
    print(f"  INTELLIGENT OSINT: USERNAME EXPANSION DEMO")
    print(f"  Target: {username} (Public Figure)")
    print("=" * 70)

    # 1. Realistic sherlock stdout for the public figure username
    # Contains:
    # - Legitimate GitHub account
    # - Legitimate Kernel.org git profile
    # - Decoy / collision account on a gaming forum with unrelated content
    mock_sherlock_stdout = f"""
[*] Checking username {username} on:
[+] GitHub: https://github.com/{username}
[+] Reddit: https://www.reddit.com/user/{username}
[+] Chess: https://www.chess.com/member/{username}_gamer
[*] Search completed with 3 results
"""

    # 2. Mock HTML profile responses for enrichment
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "robots.txt" in url_str:
            return httpx.Response(200, text="User-agent: *\nAllow: /")

        if f"github.com/{username}" in url_str:
            return httpx.Response(
                200,
                text=f"""
                <html>
                <head>
                    <title>Linus Torvalds ({username}) - GitHub</title>
                    <meta property="og:title" content="Linus Torvalds ({username})" />
                    <meta property="og:description" content="Creator of Linux and Git. Contact at torvalds@linux-foundation.org or visit https://kernel.org. Linux Foundation fellow." />
                </head>
                <body>GitHub Profile</body>
                </html>
                """,
            )
        elif f"reddit.com/user/{username}" in url_str:
            return httpx.Response(
                200,
                text=f"""
                <html>
                <head>
                    <title>{username} on Reddit</title>
                    <meta property="og:title" content="u/{username}" />
                    <meta property="og:description" content="Open source discussions, Linux kernel maintenance." />
                </head>
                <body>Reddit User</body>
                </html>
                """,
            )
        else:
            # Decoy collision account
            return httpx.Response(
                200,
                text="""
                <html>
                <head>
                    <title>Amateur Chess Gamer</title>
                    <meta property="og:title" content="Speed Chess Enthusiast" />
                    <meta property="og:description" content="Casual blitz chess and puzzle solver based in Melbourne." />
                </head>
                <body>Chess Member</body>
                </html>
                """,
            )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # Load learned resolver or train a fresh one if missing
        model_path = Path("model.joblib")
        if model_path.exists():
            resolver = LearnedResolver.load(model_path)
        else:
            from osint_agent.ml.dataset import build_xy, generate_synthetic_pairs

            pairs = generate_synthetic_pairs(n=300, seed=42)
            X, y, _ = build_xy(pairs)
            resolver = LearnedResolver(random_state=42).fit(X, y)

        audit_path = Path("demo_audit.jsonl")
        graph_file = Path("demo_graph.json")
        if audit_path.exists():
            audit_path.unlink()
        if graph_file.exists():
            graph_file.unlink()

        agent = OSINTAgent(resolver=resolver, audit_path=audit_path, http_client=client)

        scope = Scope(
            target_kind=EntityType.USERNAME,
            target_value=username,
            consent_confirmed=True,
            max_depth=2,
            max_tool_calls=5,
            accept_threshold=0.5,
        )

        with patch("shutil.which", return_value="/usr/bin/sherlock"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:

            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_sherlock_stdout.encode("utf-8"), b"")
            mock_exec.return_value = mock_proc

            graph = await agent.run(scope)

        export_graph_to_json(graph, graph_file)

    # 3. Read audit log to display confirmed vs rejected accounts
    with open(audit_path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    evaluated_hits = [e["data"] for e in events if e.get("event") == "hit_evaluated"]
    confirmed = [h for h in evaluated_hits if h.get("accepted")]
    rejected = [h for h in evaluated_hits if not h.get("accepted")]

    print("\n--- RESOLUTION DECISIONS ---")
    print(f"Total Discovered Profile Hits: {len(evaluated_hits)}")
    print(f"\n[+] Confirmed Accounts ({len(confirmed)}):")
    for c in confirmed:
        print(f"    - URL:   {c['discovered_value']}")
        print(f"      Score: {c['resolver_score']:.4f} (Confidence: {c['child_confidence']:.4f})")

    print(f"\n[-] Rejected Accounts ({len(rejected)}):")
    for r in rejected:
        print(f"    - URL:   {r['discovered_value']}")
        print(f"      Score: {r['resolver_score']:.4f} (Below threshold {scope.accept_threshold})")

    # 4. Display newly extracted lead entities from confirmed profile bios
    print("\n--- NEWLY EXTRACTED LEAD ENTITIES (from confirmed bios) ---")
    extracted_nodes = [
        (n, d)
        for n, d in graph.nodes(data=True)
        if any(p.startswith("extracted:") for p in d.get("provenance", []))
    ]
    for node_id, data in extracted_nodes:
        print(f"    * [{data['entity_type'].upper()}] {data['value']}")
        print(f"      Confidence: {data['confidence']:.4f} | Depth: {data['depth']} | Provenance: {data['provenance']}")

    print("\n--- KNOWLEDGE GRAPH TOPOLOGY ---")
    print(f"Total Graph Nodes: {graph.number_of_nodes()}")
    print(f"Total Graph Edges: {graph.number_of_edges()}")
    for u, v, d in graph.edges(data=True):
        print(f"    {u}  ──[{d.get('relation')}]──>  {v}  (source={d.get('source_tool', d.get('source'))})")

    print("\n" + "=" * 70)
    print(f"[DEMO SUCCESS] Output saved to {graph_file} and {audit_path}")
    print("=" * 70 + "\n")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
