import os
import json
import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()


# Required Mock Tool Execution Function
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")

# Knowledge Base (RAG context)
try:
    with open('knowledge_base.json', 'r') as f:
        KNOWLEDGE_BASE = f.read()
except FileNotFoundError:
    KNOWLEDGE_BASE = "Knowledge base not found."

# State & Schemas Definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool

class IntentClassification(BaseModel):
    intent: Literal['greeting', 'product_inquiry', 'high_intent'] = Field(
        description="Categorize the user's latest intent based on the conversation."
    )

class LeadExtraction(BaseModel):
    name: Optional[str] = Field(description="The full name of the user, if provided. Leave null if not provided.")
    email: Optional[str] = Field(description="The email address of the user, if provided. Leave null if not provided.")
    platform: Optional[str] = Field(description="The creator platform (e.g., YouTube, Instagram, TikTok), if provided. Leave null if not provided.")

# Agent Nodes & Logic
# Initialize LLM with Groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Create structured output wrappers
intent_llm = llm.with_structured_output(IntentClassification)
extractor_llm = llm.with_structured_output(LeadExtraction)

def classify_intent(state: AgentState):
    """Identifies the user's intent to route the conversation."""
    context_msgs = state["messages"]
    system_prompt = SystemMessage(content="""You are an intent classifier for AutoStream (a video editing SaaS). 
Analyze the conversation and classify the user's LATEST intent into one of three categories:
1. 'greeting': Casual greetings, hello, hi.
2. 'product_inquiry': Asking about pricing, plans, features, refunds, support.
3. 'high_intent': The user expresses desire to buy, subscribe, try a plan, OR they are currently providing their lead details (name, email, platform) because you asked for them.

If the user is answering a request for their name/email/platform, ALWAYS classify it as 'high_intent'."""
    )
    
    classification = intent_llm.invoke([system_prompt] + context_msgs)
    return {"intent": classification.intent}

def handle_greeting(state: AgentState):
    """Responds to casual greetings."""
    sys_msg = SystemMessage(content="You are a polite AI assistant for AutoStream, a video editing SaaS. The user just greeted you. Keep your response brief, friendly, and ask how you can help them today.")
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def handle_inquiry(state: AgentState):
    """Retrieves context from knowledge base (RAG) and answers."""
    sys_msg = SystemMessage(content=f"""You are a helpful assistant for AutoStream. 
Answer the user's question using ONLY the following knowledge base: 
{KNOWLEDGE_BASE}

Keep answers concise and accurate based purely on the provided text.""")
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def handle_lead_capture(state: AgentState):
    """Extracts lead info, asks for missing fields, and triggers the tool when done."""
    # Check if we already captured the lead to prevent duplicate capture
    if state.get("lead_captured"):
        return {"messages": [AIMessage(content="We've already saved your details! Our team will reach out to you shortly.")]}

    # Extract details from the conversation
    sys_msg = SystemMessage(content="""Extract the user's name, email, and creator platform. 
If a field is not explicitly mentioned, return null (do not return the string "null" or "NA"). 
Only extract info if the user provided it. Do not guess.""")
    extraction = extractor_llm.invoke([sys_msg] + state["messages"])
    
    # Helper to sanitize model output (some models return "null" as string)
    def sanitize(val):
        if val is None: return None
        s_val = str(val).strip().lower()
        if s_val in ["null", "none", "n/a", "", "undefined"]:
            return None
        return str(val)

    # Merge newly extracted info with existing state info
    current_name = sanitize(state.get("lead_name")) or sanitize(extraction.name)
    current_email = sanitize(state.get("lead_email")) or sanitize(extraction.email)
    current_platform = sanitize(state.get("lead_platform")) or sanitize(extraction.platform)

    missing = []
    if not current_name: missing.append("name")
    if not current_email: missing.append("email")
    if not current_platform: missing.append("creator platform (e.g. YouTube, Instagram)")

    if missing:
        # Prompt for missing info
        ask_msg = f"Awesome! To get you started, I just need a few more details: {', '.join(missing)}. Could you please provide those?"
        return {
            "messages": [AIMessage(content=ask_msg)],
            "lead_name": current_name,
            "lead_email": current_email,
            "lead_platform": current_platform
        }
    else:
        # All required details collected -> Execute Tool
        mock_lead_capture(current_name, current_email, current_platform)
        
        success_msg = f"Thanks {current_name}! We've successfully registered your interest for your {current_platform} channel. A confirmation has been sent to {current_email}."
        return {
            "messages": [AIMessage(content=success_msg)],
            "lead_name": current_name,
            "lead_email": current_email,
            "lead_platform": current_platform,
            "lead_captured": True
        }

def route_intent(state: AgentState):
    """Edge routing function based on state intent."""
    return state.get("intent", "product_inquiry")

# ==========================================
# 5. Build Graph
# ==========================================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify", classify_intent)
workflow.add_node("greeting", handle_greeting)
workflow.add_node("product_inquiry", handle_inquiry)
workflow.add_node("high_intent", handle_lead_capture)

# Build graph topology
workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify",
    route_intent,
    {
        "greeting": "greeting",
        "product_inquiry": "product_inquiry",
        "high_intent": "high_intent"
    }
)

workflow.add_edge("greeting", END)
workflow.add_edge("product_inquiry", END)
workflow.add_edge("high_intent", END)

# Compile graph with Memory component for state retention across turns
memory = MemorySaver()
agent_executor = workflow.compile(checkpointer=memory)
