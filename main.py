import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from crewai import Process, Crew
from agents import Agents
from tasks import Tasks
from tools import file_writer_tool

load_dotenv()

def main():
    st.title(" === World Historian Agent === ")
    
    question = st.text_area("What History do you want to know:", """ Explain The History About Borobudur Temple """)
    st.write(' This Is The Answer From your Question')
    st.write(question)

    openaigpt4 = ChatOpenAI (
                    model=os.getenv("OPENAI_MODEL_NAME"),
                    temperature=0.2,
                    api_key=os.getenv("OPENAI_API_KEY")
                )

    historian_crew = Crew(
                       agents = [ Agents().historian_agent() ], 
                       tasks = [ Tasks(question).research_task() ], 
                       process = Process.sequential, 
                       manager_llm = openaigpt4
                    )
    
    results = historian_crew.kickoff()
    
    st.write("Results obtained:")
    st.write(results)

    results_str = str(results)

    runner = file_writer_tool._run(filename= 'example.txt', content=results_str, directory='./note', overwrite=True)
    print(runner)
    
    # try:
    #     file_writer_tool._run(filename= 'example.txt', content=results_str, directory='./note', overwrite=True)
    #     st.write(f"File successfully written to {output_file}.")
    # except Exception as e:
    #     st.error(f"Error writing file: {e}")

    # if os.path.exists(output_file):
    #     st.write(f"File {output_file} found.")
    #     with open(output_file, 'r') as f:
    #         content = f.read()
    #     st.write("File content:")
    #     st.write(content)
    # else:
    #     st.write(f"File {output_file} not found.")

if __name__ == "__main__":
    main()