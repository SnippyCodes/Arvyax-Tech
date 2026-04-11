import uuid
from langchain_core.messages import HumanMessage
from agent import agent_executor

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Initialize console for rich UI
console = Console()

def run_chat():
    console.print(Panel.fit("[bold cyan]Welcome to AutoStream AI Assistant![/]\nType [bold red]'quit'[/] or [bold red]'exit'[/] to stop.", border_style="cyan"))
    
    # Generate a unique thread ID for this session to maintain state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        # Get user input with a styled prompt
        user_input = console.input("\n[bold green]You:[/bold green] ")
        
        if user_input.lower() in ["quit", "exit"]:
            console.print("[bold red]Goodbye![/bold red]")
            break
            
        if not user_input.strip():
            continue
            
        try:
            state_update = {"messages": [HumanMessage(content=user_input)]}
            
            # Show a thinking spinner while LLM is generating
            with console.status("[bold yellow]Agent is thinking...[/bold yellow]", spinner="dots"):
                result = agent_executor.invoke(state_update, config)
            
            # Extract and display the response
            messages = result.get("messages", [])
            if messages:
                content = messages[-1].content
                # Handle instances where model returns list of dicts instead of string
                if isinstance(content, list):
                    text_parts = [part['text'] for part in content if 'text' in part]
                    content = "".join(text_parts)
                
                # Print response inside a nice panel and render any markdown seamlessly
                console.print(Panel(Markdown(content), title="[bold blue]AutoStream Agent[/]", border_style="blue"))
                
        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/] {e}")

if __name__ == "__main__":
    run_chat()
