##Step 1: Importing all the python libs for grounding, orchestration, LLM
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import GroundingModule, OrchestrationConfig
from gen_ai_hub.orchestration.models.document_grounding import DocumentGrounding, DocumentGroundingFilter, DataRepositoryType, GroundingFilterSearch
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.service import OrchestrationService

##Step 2: Set the DATA Repository ID - Pipeline configured in AI Core using the API or Launchpad
DATA_REPOSITORY_ID = "d7834fe4-83f0-4d6e-8583-1be833a2ca0b"

##Step 3: Initialize the LLM model along with Prompt template
llm = LLM(
    name="gemini-2.5-flash",
    parameters={
        'temperature': 0.0,
    }
)

template = Template(
            messages=[
                SystemMessage("You are a helpful assistant for the German citizen center."),
                UserMessage("""Answer the request by providing relevant answers that fit to the request.
                Request: {{ ?user_query }}
                Context:{{ ?grounding_response }}
                """),
            ]
        )

#Step 4:  Set up Document Grounding
filters = [
            DocumentGroundingFilter(
                id="vector", data_repositories=[DATA_REPOSITORY_ID],
                search_config=GroundingFilterSearch(max_chunk_count=5),
                data_repository_type=DataRepositoryType.VECTOR.value
            )
        ]

grounding_config = GroundingModule(
            type="document_grounding_service",
            config=DocumentGrounding(input_params=["user_query"], output_param="grounding_response", filters=filters)
        )

##Step 5: Setup the Orchestration service config
config = OrchestrationConfig(
    template=template,
    llm=llm,
    grounding=grounding_config
)

orchestration_service = OrchestrationService(
#    api_url=AICORE_ORCHESTRATION_DEPLOYMENT_URL,
    config=config
)

##Step 6: Trigger our workflow for AI 
response = orchestration_service.stream(
    template_values=[
        TemplateValue(
            name="user_query",
            #TODO Here you can change the user prompt into whatever you want to ask the model
            value="What is die Tafel in Germany?"
        )
    ]
)

for chunk in response:
    print(chunk.orchestration_result.choices[0].delta.content)
