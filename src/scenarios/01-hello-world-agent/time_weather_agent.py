# Copyright (c) Microsoft. All rights reserved.
"""
Scenario 1 - Your First Time & Weather Agent

This agent can:
- Remember the user's location from conversation context
- Answer questions about the current time based on user location
- Provide weather information for the user's location
- Track location changes and recall previous locations mentioned

It uses two MCP servers:
- User Server (port 8002): Provides user info, location, and time functions
- Weather Server (port 8003): Provides weather information based on location and time

Run the MCP servers first:
    cd src/mcp-server/02-user-server && python server-mcp-sse-user.py
    cd src/mcp-server/04-weather-server && python server-mcp-sse-weather.py

Then run this agent:
    python time_weather_agent.py
"""

import os
import asyncio
from typing import Annotated
from datetime import datetime

import pytz
from pydantic import Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

from agent_framework import ChatAgent, HostedMCPTool, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

load_dotenv()

# --- Client Configuration ---

if os.environ.get("GITHUB_TOKEN") is not None:
    token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.github.ai/inference"
    print("Using GitHub Token for authentication")
elif os.environ.get("AZURE_OPENAI_API_KEY") is not None:
    token = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    print("Using Azure OpenAI Token for authentication")
else:
    raise ValueError("No API key found. Set GITHUB_TOKEN or AZURE_OPENAI_API_KEY.")

async_openai_client = AsyncOpenAI(
    base_url=endpoint,
    api_key=token
)

# Model names from environment
completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")
small_model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")

# Create chat clients for different model sizes
completion_client = OpenAIChatClient(
    model_id=completion_model_name,
    api_key=token,
    async_client=async_openai_client
)

medium_client = OpenAIChatClient(
    model_id=medium_model_name,
    api_key=token,
    async_client=async_openai_client
)

small_client = OpenAIChatClient(
    model_id=small_model_name,
    api_key=token,
    async_client=async_openai_client
)

# --- MCP Server Configuration ---

# The user server provides: get_current_user, get_current_location, get_current_time, move
USER_MCP_URL = os.environ.get("USER_MCP_URL", "http://localhost:8002/mcp")

# The weather server provides: list_supported_locations, get_weather_at_location, get_weather_for_multiple_locations
WEATHER_MCP_URL = os.environ.get("WEATHER_MCP_URL", "http://localhost:8003/mcp")

# --- Agent Instructions ---

AGENT_INSTRUCTIONS = """You are a helpful Time & Weather assistant that helps users know the current time and weather for their location.

Use the MCP tools for fullfilling your tasks."""


async def run_interactive_conversation():
    """Run an interactive conversation with the Time & Weather agent."""
    
    print("\n" + "=" * 60)
    print("Time & Weather Agent - Scenario 1")
    print("=" * 60)
    print("\nThis agent remembers your location and can tell you the time")
    print("and weather. Try these example queries:")
    print("  - 'I am currently in London'")
    print("  - 'What is the weather now here?'")
    print("  - 'What time is it for me right now?'")
    print("  - 'I moved to Berlin, what is the weather like today?'")
    print("  - 'Can you remind me where I said I am based?'")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("=" * 60 + "\n")

    # Create the agent with MCP tools and local tools
    agent = ChatAgent(
        chat_client=medium_client,
        name="TimeWeatherAgent",
        instructions=AGENT_INSTRUCTIONS,
        tools=[
            # MCP tools from external servers
            MCPStreamableHTTPTool(
                name="UserService",
                url=USER_MCP_URL,
                approval_mode="never_require",
            ),
            MCPStreamableHTTPTool(
                name="WeatherService", 
                url=WEATHER_MCP_URL,
                approval_mode="never_require",
            ),
        ],
    )

    # Create a thread to maintain conversation context
    thread = agent.get_new_thread()

    while True:
        try:
            # Get user input
            user_input = input("\nYou 2: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Have a great day!")
                break

            # Run the agent with the thread to maintain context
            result = await agent.run(user_input, thread=thread)
            # result = await handle_approvals_with_thread(user_input, agent, thread)
            
            print(f"\nAgent: {result.text}")

        except KeyboardInterrupt:
            print("\n\nConversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.")


async def run_demo_conversation():
    """Run a demo conversation showing all the scenario's input queries."""
    
    print("\n" + "=" * 60)
    print("Time & Weather Agent - Demo Mode")
    print("=" * 60)
    print("Running through the scenario's example queries...\n")

    # Create the agent with MCP tools and local tools
    agent = ChatAgent(
        chat_client=medium_client,
        name="TimeWeatherAgent",
        instructions=AGENT_INSTRUCTIONS,
        tools=[
            # MCP tools from external servers
            HostedMCPTool(
                name="UserService",
                url=USER_MCP_URL,
            ),
            HostedMCPTool(
                name="WeatherService", 
                url=WEATHER_MCP_URL,
            ),
            # Local Python function tools
            get_current_time_for_timezone,
            get_timezone_for_city,
        ],
    )

    # Create a thread to maintain conversation context
    thread = agent.get_new_thread()

    # The demo queries from the scenario
    demo_queries = [
        "I am currently in London",
        "What is the weather now here?",
        "What time is it for me right now?",
        "I moved to Berlin, what is the weather like today?",
        "Can you remind me where I said I am based?",
    ]

    for query in demo_queries:
        print(f"\n{'─' * 50}")
        print(f"User: {query}")
        
        try:
            result = await agent.run(query, thread=thread)
            print(f"\nAgent: {result.text}")
        except Exception as e:
            print(f"\nError: {e}")
        
        # Small delay for readability
        await asyncio.sleep(1)

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)


async def main():
    """Main entry point - choose between demo and interactive modes."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        await run_demo_conversation()
    else:
        await run_interactive_conversation()


if __name__ == "__main__":
    asyncio.run(main())
