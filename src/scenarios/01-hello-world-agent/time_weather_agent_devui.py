# Copyright (c) Microsoft. All rights reserved.
"""
Scenario 1 - Time & Weather Agent with Dev UI

This version of the agent integrates with the Agent Framework Dev UI
for exploring activities, metrics, and traces to troubleshoot multi-tool
agent executions.

Prerequisites:
1. Install the Dev UI: pip install agent-framework-devui --pre
2. Start the MCP servers:
   - cd src/mcp-server/02-user-server && python run-mcp-user.py
   - cd src/mcp-server/04-weather-server && python run-mcp-weather.py
3. Run this script: python time_weather_agent_devui.py
4. Open the Dev UI in your browser (URL shown in console)

The Dev UI allows you to:
- See all agent activities and tool calls
- Inspect message flow and responses
- View metrics and performance data
- Debug multi-tool execution sequences
"""

import os
import asyncio
from typing import Annotated
from datetime import datetime

import pytz
from pydantic import Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

from agent_framework import ChatAgent, MCPStreamableHTTPTool
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
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")

medium_client = OpenAIChatClient(
    model_id=medium_model_name,
    api_key=token,
    async_client=async_openai_client
)

# --- MCP Server Configuration ---

USER_MCP_URL = os.environ.get("USER_MCP_URL", "http://localhost:8002/sse")
WEATHER_MCP_URL = os.environ.get("WEATHER_MCP_URL", "http://localhost:8003/sse")


# --- Local Tools ---

def get_current_time_for_timezone(
    timezone_name: Annotated[str, Field(description="IANA timezone name like 'Europe/London', 'America/New_York', 'Europe/Berlin'")]
) -> str:
    """Get the current time for a given IANA timezone."""
    try:
        timezone = pytz.timezone(timezone_name)
        now = datetime.now(timezone)
        return f"The current time in {timezone_name} is {now.strftime('%I:%M:%S %p')} on {now.strftime('%A, %B %d, %Y')}."
    except Exception as e:
        return f"Could not determine time for timezone '{timezone_name}': {str(e)}"


def get_timezone_for_city(
    city: Annotated[str, Field(description="City name like 'London', 'Berlin', 'New York', 'Tokyo'")]
) -> str:
    """Get the IANA timezone for a well-known city."""
    city_timezones = {
        "london": "Europe/London",
        "berlin": "Europe/Berlin",
        "new york": "America/New_York",
        "tokyo": "Asia/Tokyo",
        "sydney": "Australia/Sydney",
        "seattle": "America/Los_Angeles",
        "paris": "Europe/Paris",
    }
    
    city_lower = city.lower().strip()
    if city_lower in city_timezones:
        return f"The timezone for {city} is {city_timezones[city_lower]}."
    else:
        return f"Unknown city '{city}'. Try a major city name."


# --- Agent Instructions ---

AGENT_INSTRUCTIONS = """You are a helpful Time & Weather assistant.

## Capabilities:
1. **Remember Context**: Remember the user's location from our conversation.
2. **Location Tracking**: When a user says "I am in [city]", remember this as their current location.
3. **Time Information**: Tell users the current time in their location.
4. **Weather Information**: Provide weather for supported locations (Seattle, New York, London, Berlin, Tokyo, Sydney).

## Guidelines:
- When the user says "here" or "my location", use the location they mentioned earlier.
- If the user says they "moved", update their current location.
- If asked to recall their location, tell them where they said they are.
- Be concise and friendly.
"""


def create_time_weather_agent() -> ChatAgent:
    """Create and return the Time & Weather agent.
    
    This function creates the agent without the async context manager,
    which is required for use with the Dev UI.
    """
    return ChatAgent(
        chat_client=medium_client,
        name="TimeWeatherAgent",
        instructions=AGENT_INSTRUCTIONS,
        tools=[
            MCPStreamableHTTPTool(
                name="UserService",
                url=USER_MCP_URL,
            ),
            MCPStreamableHTTPTool(
                name="WeatherService", 
                url=WEATHER_MCP_URL,
            ),
    )


def main_with_devui():
    """Run the agent with the Dev UI for debugging and exploration."""
    try:
        from agent_framework.devui import serve
    except ImportError as e:
        print("=" * 60)
        print("Dev UI import failed!")
        print(f"Import error details: {e}")
        print("=" * 60)
        print("\nInstall it with: pip install agent-framework-devui --pre")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("Time & Weather Agent - Dev UI Mode")
    print("=" * 60)
    
    # Create the agent
    agent = create_time_weather_agent()
    
    print("\nStarting Dev UI...")
    print("Open the URL shown below in your browser to interact with the agent.")
    print("The Dev UI lets you see all tool calls, message flow, and metrics.")
    print("=" * 60 + "\n")

    # Start the Dev UI with the agent
    # The Dev UI handles the conversation and provides a web interface
    serve(entities=[agent], port=8093, auto_open=True)
    


async def main_console():
    """Run the agent in console mode (fallback if Dev UI not available)."""
    print("\n" + "=" * 60)
    print("Time & Weather Agent - Console Mode")
    print("=" * 60)
    print("\nType your messages below. Type 'quit' to exit.\n")

    agent = create_time_weather_agent()
    thread = agent.get_new_thread()

    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            result = await agent.run(user_input, thread=thread)
            print(f"Agent: {result.text}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        asyncio.run(main_console())
    else:
        try:
            main_with_devui()
        except Exception as e:
            print(f"Dev UI failed: {e}")
            print("Falling back to console mode...")
            asyncio.run(main_console())


if __name__ == "__main__":
    main()
