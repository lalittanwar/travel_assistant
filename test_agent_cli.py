"""
Quick manual test of the agent from the command line, with multi-turn
memory via a fixed thread_id.

Usage:
    python test_agent_cli.py
"""

import asyncio
from agent import build_agent


async def main():
    agent, cleanup = await build_agent()
    config = {"configurable": {"thread_id": "cli-test-session"}}

    print("Singapore Travel Assistant (type 'quit' to exit)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            response = await agent.ainvoke(
                {"messages": [("user", user_input)]},
                config=config,
            )
            print(f"\nAssistant: {response['messages'][-1].content}\n")
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
