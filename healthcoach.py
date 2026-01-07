# -*- coding: utf-8 -*-
"""HealthCoach.ipynb

"""

!pip install -U langchain langchain-core langchain-aws

# To Do: Update the access key ID and access key

import os
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

import boto3
import botocore

session = boto3.Session()
bedrock_client = session.client('bedrock-agent')
bedrock_runtime = boto3.client('bedrock-runtime')

try:
    response = bedrock_client.list_knowledge_bases(maxResults=1)  # Retrieve the first knowledge base
    knowledge_base_summaries = response.get('knowledgeBaseSummaries', [])

    if knowledge_base_summaries:
        kb_id = knowledge_base_summaries[0]['knowledgeBaseId']
        print(f"Knowledge Base ID: {kb_id}")
    else:
        print("No Knowledge Base summaries found.")

except botocore.exceptions.ClientError as e:
    print(f"Error: {e}")

import boto3
from botocore.client import Config
import pprint
import json

pp = pprint.PrettyPrinter(indent=2)

session = boto3.session.Session()
region = session.region_name

bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime', region_name = region)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config, region_name = region)

import boto3

REGION = "us-west-2"
bedrock = boto3.client("bedrock", region_name=REGION)

target_model_id = "anthropic.claude-haiku-4-5-20251001-v1:0"

profiles = []
resp = bedrock.list_inference_profiles(maxResults=100)
profiles += resp.get("inferenceProfileSummaries", [])
while "nextToken" in resp:
    resp = bedrock.list_inference_profiles(maxResults=100, nextToken=resp["nextToken"])
    profiles += resp.get("inferenceProfileSummaries", [])

match = None
for p in profiles:
    for m in p.get("models", []):
        if m.get("modelArn", "").endswith(target_model_id):
            match = p
            break
    if match:
        break

if not match:
    raise RuntimeError("No inference profile found that contains the target model.")

inference_profile_id_or_arn = match.get("inferenceProfileArn") or match.get("inferenceProfileId")
print("Using inference profile:", inference_profile_id_or_arn)

modelId = inference_profile_id_or_arn

import langchain
from langchain_aws import ChatBedrock
from langchain_aws.retrievers.bedrock import AmazonKnowledgeBasesRetriever

llm = ChatBedrock(model_id=modelId, provider="anthropic", client=bedrock_runtime)

query = "What helps headaches?"
retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4, 'overrideSearchType': "SEMANTIC"}})
docs = retriever.invoke(
        input=query
    )
#for doc in docs:
#    print(doc.page_content)
#    print("------")

from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
You are a behavioral health coach who has been provided mental health information about a specific patient.
You should speak in a compassionate, professional tone to support the user. Make sure to not share
information about the patient in the context, and only focus on providing advice.

Context:
{context}

Question: {question}

Make sure the response contains:
-Actionable advice
-Less than 150 words

If certain information is not available in the provided context, explicitly state: "This information is not provided."
Stick to the responses provided in the context.
"""

claude_prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                               input_variables=["context","question"])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | claude_prompt
    | llm
    | StrOutputParser()
)

response=chain.invoke(query)
print(response)

### Gradio ###

import os
import boto3
import botocore
from botocore.config import Config
import gradio as gr

from langchain_aws import ChatBedrock
from langchain_aws.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

REGION = "us-west-2"
TARGET_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"

cfg = Config(connect_timeout=120, read_timeout=120, retries={"max_attempts": 0})

bedrock_control = boto3.client("bedrock", region_name=REGION)                  # list inference profiles
bedrock_agent = boto3.client("bedrock-agent", region_name=REGION)              # list KBs
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION, config=cfg)  # retrieve
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)             # invoke model

def get_first_kb_id() -> str:
    resp = bedrock_agent.list_knowledge_bases(maxResults=1)
    summaries = resp.get("knowledgeBaseSummaries", [])
    if not summaries:
        raise RuntimeError("No Knowledge Bases found in this account/region.")
    return summaries[0]["knowledgeBaseId"]

def get_inference_profile_for_model(target_model_id: str) -> str:
    profiles = []
    resp = bedrock_control.list_inference_profiles(maxResults=100)
    profiles += resp.get("inferenceProfileSummaries", [])

    while "nextToken" in resp:
        resp = bedrock_control.list_inference_profiles(maxResults=100, nextToken=resp["nextToken"])
        profiles += resp.get("inferenceProfileSummaries", [])

    for p in profiles:
        for m in p.get("models", []):
            if m.get("modelArn", "").endswith(target_model_id):
                return p.get("inferenceProfileArn") or p.get("inferenceProfileId")

    raise RuntimeError(f"No inference profile found that contains: {target_model_id}")

kb_id = get_first_kb_id()
inference_profile_id_or_arn = get_inference_profile_for_model(TARGET_MODEL_ID)

llm = ChatBedrock(model_id=inference_profile_id_or_arn, provider="anthropic", client=bedrock_runtime)

retriever = AmazonKnowledgeBasesRetriever(knowledge_base_id=kb_id, retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4,
            "overrideSearchType": "SEMANTIC",
        }
    },
)

PROMPT_TEMPLATE = """
You are a behavioral health coach who has been provided mental health information about a specific patient.
You should speak in a compassionate, professional tone to support the user. Make sure to not share
information about the patient in the context, and only focus on providing advice.

Context:
{context}

Question: {question}

Make sure the response contains:
- Actionable advice
- Less than 150 words

If certain information is not available in the provided context, explicitly state: "This information is not provided."
Stick to the responses provided in the context.
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def chat(user_message: str, history):
    user_message = (user_message or "").strip()
    if not user_message:
        return "", history

    try:
        answer = chain.invoke(user_message)
        history = history + [[user_message, answer]]
        return "", history
    except botocore.exceptions.ClientError as e:
        err = f"AWS error: {e}"
        history = history + [[user_message, err]]
        return "", history
    except Exception as e:
        err = f"Error: {repr(e)}"
        history = history + [[user_message, err]]
        return "", history

with gr.Blocks(title="Bedrock Knowledge Base Coach") as demo:
    gr.Markdown(
        f"""
# Bedrock Knowledge Base Chat (Claude via Inference Profile)

**Region:** `{REGION}`
**Knowledge Base:** `{kb_id}`
**Model (Inference Profile):** `{inference_profile_id_or_arn}`

Ask a question and I’ll answer using your Knowledge Base context.
"""
    )

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(label="Your message", placeholder="e.g., In what ways is green tea helpful?", lines=2)
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(debug=True)

