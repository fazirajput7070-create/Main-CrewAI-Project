# Multi_agent_system_with_Crewai
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew,LLM
from crewai_tools import TavilySearchTool
search_tool = TavilySearchTool()

# 1. Define LLM Configuration

llm = LLM(
    model="ollama/llama3.2"
)

# define Agents
researcher = Agent(
    role="Dehydrated fruit researcher",
    goal="find accurate information about the dehydrated fruit market in America",
    backstory="You are an experienced dehydraetd fruit market researcher who specializes in researching technology markets.",
    llm=llm,
    tools=[search_tool]

    
)

analyst = Agent(
    role="Dehydrated fruit Analyst",
    goal="Analyze and compare information about Dehydrated companies in America",
    backstory="You are an experienced market analyst who analyzes companies and identifies important trends.",
    llm=llm
)   

writer=Agent(
role="Dehydrated Report Writer",
goal="Create a clear and professional report using the research analysis provide by the other agents.",
backstory="You are an experienced report writer who turns research and analysis into clear, well-structured reports.",
llm=llm

)


financial_researcher = Agent(
    role="Financial researcher",
    goal="Research and collect financial inforamtion about companies",
    backstory="You are an experienced financial researcher who finds and organizes relevant financial information",
    llm=llm
)


financial_analyst = Agent(
    role="Financial Analyst",
    goal="Analyze and compare financial inforamtion and compare the financial performance of companies",
    backstory="You are an ecperienced financial analyst who analyzes financial data,compares companies, and identifies important financial strengths and weaknesses.",
    llm=llm
)


# Define task

research_task = Task(
description="Identify major dehydrated fruit companies and startups in America, and explain what each one does.",
expected_output="A clear list of major America dehydrate fruit companies and startups with a short description of their work.",
agent=researcher
)

company_task = Task(
    description="Identify major dehydrated fruit companies and startups in America.",
    expected_output="A list of major dehydrated fruit companies and startups.",
    agent=researcher
)

analysis_task = Task(
    description="Analyze and compare the Dehydrated fruit companies and startups identified by the researcher. Determine which companies appear to have stronger business potential and explain the reasons based on their areas of work, strengths, and differences.",
    expected_output="A detailed comparison of the major Dehydrated fruit  companies and startups in America, identifying their strengths, weaknesses, differences, and which companies appear stronger from a business perspective.",
    context=[company_task],
    agent=analyst
)


writer_task=Task(
    description="create a clear amd professional final report using the research and analysis provided by the other aagents.",
    expected_output="A well-structured final report about america's Dehydrated market, including major companies, their comparison,business strengths, and important market insights.",
    context=[research_task,analysis_task],
    agent=writer
)


financial_researcher_task=Task(
    description="Research and collect important financial information about the selected companies, including growth, financial performance, and other relevant financial details.",
    expected_output="A clear and organized report containing the important financial information collected about the  companies.",
    agent=financial_researcher
)


financial_analyst_task=Task(
    description="Analyze and compare the financial information collected about companies. Identify financial strengths, weaknesses, and important differences.",
    expected_output="A detailed financial analysis comparing the companies, including their strengths, weaknesses, and overall financial performance",
    context=[financial_researcher_task],
    agent=financial_analyst
)



crew = Crew(
    agents=[researcher, analyst, writer, financial_researcher, financial_analyst],
    tasks=[research_task,  company_task, analysis_task,  writer_task, financial_researcher_task, financial_analyst_task]

)

result = crew.kickoff()

print(result)

