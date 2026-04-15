##Step 1: Load the libraries of Crew AI and LiteLLM
from crewai import Agent, Task, Crew
from crewai.tools import tool

##Step 2: Configuration by reading config.json file for the API key of AI core
import os
import json

##New Dependency
from gen_ai_hub.document_grounding.client import RetrievalAPIClient
from gen_ai_hub.document_grounding.models.retrieval import (
    RetrievalSearchInput,
    RetrievalSearchFilter,
)
from gen_ai_hub.orchestration.models.document_grounding import DataRepositoryType

DATA_REPOSITORY_ID = "d7834fe4-83f0-4d6e-8583-1be833a2ca0b"

home_dir = os.path.expanduser('~')
aicore_dir = os.path.join(home_dir, '.aicore')
os.makedirs(aicore_dir, exist_ok=True)

config_path = os.path.join(aicore_dir, 'config.json')
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)

    os.environ["AICORE_AUTH_URL"]=config_data["AICORE_AUTH_URL"]+"/oauth/token"
    os.environ["AICORE_CLIENT_ID"]=config_data["AICORE_CLIENT_ID"]
    os.environ["AICORE_CLIENT_SECRET"]=config_data["AICORE_CLIENT_SECRET"]
    os.environ["AICORE_BASE_URL"]=config_data["AICORE_BASE_URL"]

    os.environ["LITELLM_PROVIDER"]= "sap"
    os.environ["AICORE_RESOURCE_GROUP"]="doc-grounding"

##Accept user question as input usage the rag api client from Gen Ai Cloud SDK
@tool("call_grounding_service")
def call_grounding_service(user_question: str) -> str:
    """Function to call the grounding service and retrieve relevant information based on the user's question."""
    retrieval_client = RetrievalAPIClient()

    search_filter = RetrievalSearchFilter(
        id="vector",
        dataRepositoryType=DataRepositoryType.VECTOR.value,
        dataRepositories=[DATA_REPOSITORY_ID],
        searchConfiguration={
            "maxChunkCount": 2
        },
    )

    search_input = RetrievalSearchInput(
        query=user_question,
        filters=[search_filter],
    )

    response = retrieval_client.search(search_input)

    response_dict = json.dumps(response.model_dump(), indent=2)
    return response_dict


##Step 3: Create a Hamburg Social Welfare Case Manager Agent

welfare_agent = Agent(
    role="Hamburg Social Welfare Case Manager",
    goal="Process and manage social welfare applications for the city of Hamburg, matching citizens with appropriate social services and benefits based on their needs and circumstances.",
    backstory="You are an experienced social welfare case manager working for the city of Hamburg. You have deep knowledge of Hamburg's social services programs, eligibility requirements, local resources, and citizen support initiatives. Your role is to help vulnerable populations access the social welfare benefits they need.",
    llm="sap/gpt-4o",  # provider/llm - Using one of the models from SAP's model library in Generative AI Hub
    tools=[call_grounding_service],
    verbose=True
)

## Step 4:  Get user input
print("Hamburg Social Welfare Case Management System")
print("=" * 50)
user_question = input("\nPlease describe your social welfare inquiry or situation:\n> ")

## Step 5: Create a task for the welfare case manager with user input
process_welfare_task = Task(
    description=f"Process the following social welfare inquiry for a Hamburg citizen: {user_question}\n\nBased on this inquiry, provide more information regarding eligibility and available services.",
    expected_output="Structured social welfare assessment with eligibility determination, service recommendations, and personalized support plan for Hamburg citizens.",
    agent=welfare_agent
)

## Step 6: Create a crew with the welfare agent
crew = Crew(
    agents=[welfare_agent],
    tasks=[process_welfare_task],
    verbose=True
)

#Step 7: Execute the crew
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n" + "="*50)
    print("Hamburg Social Welfare Assessment Report:")
    print("="*50)
    print(result)



