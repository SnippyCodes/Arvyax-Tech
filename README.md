# AutoStream Agentic Workflow

This repository contains the solution for the "Social-to-Lead Agentic Workflow" assignment. It features a conversational AI agent built for the fictional SaaS product **AutoStream**.

## Features
- **Intent Detection**: Accurately classifies user queries as greetings, product inquiries, or high-intent lead actions.
- **Local RAG**: Uses a JSON knowledge base to accurately retrieve pricing and policy details.
- **Lead Capture Mechanism**: Gathers Name, Email, and Creator Platform asynchronously and triggers a mock database execution once all constraints are met.
- **Stateful Memory**: Employs LangGraph's native checkpointer to persist context across 5-6 conversational turns.
- **Rich Terminal UI**: Beautiful, styled interactive command-line interface with Markdown support and thinking animations using `rich`.

## How to run locally

1. **Clone the repository and enter the directory**:
```bash
git clone <your-repo-link>
cd <repo>
```

2. **Set up a virtual environment (Optional but Recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**:
Copy the `.env.example` file to `.env` and add your Groq API Key:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
`GROQ_API_KEY=your_groq_api_key_here`

5. **Run the Agent**:
```bash
python run_terminal.py
```

## Architecture Explanation

This project leverages **LangGraph** due to its explicit, graph-based approach to defining state machines. Unlike traditional chain-based LLM workflows which can be implicit and unpredictable in multi-turn routing scenarios, LangGraph provides granular control over node transitions and logic segregation. This is crucial for a real-world SaaS deployment where deterministic handling of tool calling (like Lead Capture) needs to be strictly separated from plain conversational steps to avoid premature execution.

**State Management** is handled intrinsically by a `TypedDict` (`AgentState`). At every turn, the user's message is routed through an `IntentClassification` LLM call which identifies the path. During lead capture, missing user details (Name, Email, Platform) are updated iteratively within the state. A `MemorySaver` checkpointer binds the conversation history to a unique `thread_id`. Because of the graph's reducer (`operator.add` for messages, overwrite for the strings), the agent effortlessly maintains its context over 5-6 turns, accurately prompting the user for *only* the details they have not provided yet, leading up to the final tool trigger.

## WhatsApp Webhooks Integration

To integrate this agent with WhatsApp, I would deploy a backend service (e.g., FastAPI or Express.js) linked to the official WhatsApp Business API via Meta Developer platform. The workflow would be:
1. **Webhook Registration**: Expose an HTTPS `POST` route (`/webhook`) from the server to Meta. 
2. **Ingestion**: When a user messages the WhatsApp number, Meta posts a JSON payload to the webhook containing the user's phone number, profile details, and message text.
3. **Session Management**: Extract the user's phone number to serve as the LangGraph `thread_id` to route their state independently.
4. **Agent Invocation**: Send the incoming text into `agent_executor.invoke()` with the specific `thread_id`.
5. **Response Dispatch**: The agent yields a final string response (or tool execution success message). The backend wraps this string into a payload and sends an asynchronous API request back to the WhatsApp Cloud API (`messages` endpoint) targeting the user's phone number.
