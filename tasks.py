from crewai import Task
from agents import Agents
from tools import search_tool
class Tasks:
    def __init__(self, event):
        self.historical_events = event

    def research_task(self):
        search_task = Task(
            description = """Research and provide a detailed and nuanced report on the historical event, figure, or period specified: {}. 
                            Use the search_tool to find comprehensive and accurate information. 
                            Your research should include:
                            - A thorough overview of the event, figure, or period, highlighting its significance in the broader historical context.
                            - Key dates, locations, and notable individuals involved.
                            - A detailed analysis of the causes and consequences, including direct and indirect impacts.
                            - Connections to larger historical themes or trends, such as social, economic, or political changes.
                            - Cross-reference information from multiple reliable sources such as primary documents, peer-reviewed journals, and reputable historical databases.
                            - Provide a critical evaluation of the sources used, noting any biases or limitations.
                            - Ensure all information is accurately cited to allow verification.
                            """.format(self.historical_events),
            expected_output="""A detailed report containing:
                            - An in-depth overview of the historical event, figure, or period.
                            - Important dates, locations, and people.
                            - Analysis of causes, effects, and broader historical significance.
                            - Connections to major historical trends or themes.
                            - Evaluation of sources with proper citations.
                            """,
            agent=Agents().historian_agent(),
            tools=[search_tool]
        )
        
        return search_task
