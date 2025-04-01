import numpy as np
import os 
from dotenv import load_dotenv
load_dotenv()
import yfinance
from phi.model.groq import Groq
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# AGENT A
sentiment_analysis_agent = Agent(
    name="SENTIMENT_ANALYSIS_AGENT",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Analyze sentiment in news and discussions about stocks"],
    show_tools_calls=True,
    markdown=True,
)

# AGENT B
forecasting_agent = Agent(
    name="FORECASTING_AGENT",
    model=Groq(id="llama3-70b-8192"),
    tools = [
    YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)
],
    instructions=["Predict stock trends based on historical data"],
    show_tools_calls=True,
    markdown=True,
)

# AGENT C
portfolio_management_agent = Agent(
    name="PORTFOLIO_MANAGEMENT_AGENT",
    model=Groq(id="llama3-70b-8192"),
    instructions=["Advise on investment diversification and risk management"],
    show_tools_calls=True,
    markdown=True,
)

# AGENT D
web_search_agent=Agent(
    name="web_search_agent",
    role=" search the web in the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always includes sources"],
    show_tools_calls=True,
    markdown=True,
)

# AGENT E
financial_agent=Agent(
    name="FINANCE_AI_AGENT",
    model=Groq(id="llama3-70b-8192"),
    tools = [
    YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)
],
    instructions=["Use tables to describe data"],
    show_tools_calls=True,
    markdown=True,

)

# GLOBAL AGENTS
multy_ai_agent=Agent(
    model=Groq(id="llama3-70b-8192"),
    team=[sentiment_analysis_agent,forecasting_agent,portfolio_management_agent,web_search_agent,financial_agent],
    instructions=["Always includes sources","Use tables to describe data"],
    show_tools_calls=True,
    markdown=True,

)