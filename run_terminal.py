import uuid
from langchain_core.messages import HumanMessage
from agent import agent_executor

def run_chat():
    print("Welcome to AutoStream AI Assistant! Type 'quit' or 'exit' to stop.")
    
    # Generate a unique thread ID for this session to maintain state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        try:
            state_update = {"messages": [HumanMessage(content=user_input)]}
            result = agent_executor.invoke(state_update, config)
            
            # The last message is from the AI
            messages = result.get("messages", [])
            if messages:
                print(f"\nAutoStream Agent: {messages[-1].content}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_chat()
