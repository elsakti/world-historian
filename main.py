import os
from dotenv import load_dotenv
import streamlit as st
from streamlit.components.v1 import html
from langchain_openai import ChatOpenAI
from crewai import Process, Crew
from agents import Agents
from tasks import Tasks
from tools import file_writer_tool

load_dotenv()

def main():

    st.set_page_config(
        page_title="World Historian GPT",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #4b4b4b;
            text-align: center;
            padding: 20px 0;
        }
        .stTextArea {
            background-color: #ffffff;
        }
        p, li, h2, h3 {
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🌍 World Historian Agent 🌍")

    question = st.text_area("What History do you want to know:", "Tell me a common world truth")
    st.write(f"**Your Question:** {question}")

    st.markdown("### Processing...")

    openaigpt4 = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME"),
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    historian_crew = Crew(
        agents=[Agents().historian_agent()],
        tasks=[Tasks(question).research_task()],
        process=Process.sequential,
        manager_llm=openaigpt4
    )
        
    results = historian_crew.kickoff()
    
    st.markdown("## Results obtained:")
    st.write(f"""
        **Answer:**
        {results}
    """)

    results_str = str(results)
        
    runner = file_writer_tool._run(filename='example.txt', content=results_str, directory='./note', overwrite=True)
    print(runner)
    
    st.success("Results saved successfully!")
    st.balloons()

if __name__ == "__main__":
    main()