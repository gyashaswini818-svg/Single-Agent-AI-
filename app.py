
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.tools import tool

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("Ask me anything! I can search the web and check weather.")

# -----------------------------
# Tools
# -----------------------------
search_tool = TavilySearch(max_results=3)


@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set."

    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}."

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Create Agent
# -----------------------------
tools = [search_tool, get_weather]

agent = create_agent(
    model=llm,
    tools=tools
)

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User input
# -----------------------------
user_input = st.chat_input("Ask your AI agent...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Agent response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            })

            answer = response["messages"][-1].content

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

