from crewai import Task
from agents import Agents
from tools import search_tool

class Tasks:
    def __init__(self, event):
        self.historical_events = event

    def research_task(self):
        agent = Agents().historian_agent()
        search_task = Task (
            description = """Research and provide detailed information about {}, figure, or period.
                            use search_tool to search for the answers.
                            Use reliable sources to ensure accuracy and include relevant historical context, causes, and consequences. 
                            If needed, connect the event or figure to broader historical trends or themes.""".format(self.historical_events), 
            expected_output="""A comprehensive and well-researched report that includes:
                            - An overview of the historical event, figure, or period.
                            - Key dates, locations, and individuals involved.
                            - The causes and consequences of the event or significance of the figure.
                            - Relevant historical context and connections to broader themes or trends.
                            - Citations from reliable sources to support the information provided.""", 
            agent=agent, 
            tools=[search_tool]
        )
        
        return search_task
