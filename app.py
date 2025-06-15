# ==============================================================================
# Original Streamlit App (Commented Out)
# ==============================================================================
"""
import streamlit as st
import os
import json
import re
import uuid
import extra_streamlit_components as esc
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
import stui
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
import io # Import io module for BytesIO
from llama_index.core.tools import FunctionTool # Import FunctionTool

# Import necessary libraries for Hugging Face integration
from huggingface_hub import HfFileSystem
import os # Import os to access environment variables

# Initialize HfFileSystem globally
fs = HfFileSystem()
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

cookies = esc.CookieManager(key="esi_cookie_manager")

SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

# Import UI_ACCESSIBLE_WORKSPACE from tools.py
from tools import UI_ACCESSIBLE_WORKSPACE
# Import HF_USER_MEMORIES_DATASET_ID from config.py
from config import HF_USER_MEMORIES_DATASET_ID

# Constant to control the maximum number of messages sent in chat history to the LLM
MAX_CHAT_HISTORY_MESSAGES = 15 # Keep the last N messages to manage context length

@st.cache_resource
def setup_global_llm_settings() -> tuple[bool, str | None]:
    \"\"\"Initializes global LLM settings using st.cache_resource to run only once.\"\"\"
    print("Initializing LLM settings...")
    try:
        initialize_agent_settings()
        print("LLM settings initialized successfully.")
        return True, None
    except Exception as e:
        error_message = f"Fatal Error: Could not initialize LLM settings. {e}"
        print(error_message)
        return False, error_message

# New cached function for initial greeting
@st.cache_data(show_spinner=False)
def _get_initial_greeting_text():
    \"\"\"Generates and caches the initial LLM greeting text for startup.\"\"\"
    return generate_llm_greeting()

# New cached wrapper for suggested prompts
@st.cache_data(show_spinner=False)
def _cached_generate_suggested_prompts(chat_history: List[Dict[str, Any]]) -> List[str]:
    \"\"\"
    Generates suggested prompts based on chat history, cached to avoid redundant LLM calls.
    The cache key is based on the content of chat_history.
    \"\"\"
    print("Generating suggested prompts...")
    return generate_suggested_prompts(chat_history)

# Define dynamic tool functions that can access st.session_state
def read_uploaded_document_tool_fn(filename: str) -> str:
    \"\"\"Reads the full text content of a document previously uploaded by the user.
    Input is the exact filename (e.g., 'my_dissertation.pdf').\"\"\"
    if "uploaded_documents" not in st.session_state or filename not in st.session_state.uploaded_documents:
        return f"Error: Document '{filename}' not found in uploaded documents. Available documents: {list(st.session_state.uploaded_documents.keys())}"
    return st.session_state.uploaded_documents[filename]

def analyze_dataframe_tool_fn(filename: str, head_rows: int = 5) -> str:
    \"\"\"Provides summary information (shape, columns, dtypes, head, describe) about a pandas DataFrame
    previously uploaded by the user. Input is the exact filename (e.g., 'my_data.csv').
    For more complex analysis, use the 'code_interpreter' tool.\"\"\"
    if "uploaded_dataframes" not in st.session_state or filename not in st.session_state.uploaded_dataframes:
        return f"Error: DataFrame '{filename}' not found in uploaded dataframes. Available dataframes: {list(st.session_state.uploaded_dataframes.keys())}"

    df = st.session_state.uploaded_dataframes[filename]

    info_str = f"DataFrame: {filename}\n"
    info_str += f"Shape: {df.shape}\n"
    info_str += f"Columns: {', '.join(df.columns)}\n"
    info_str += f"Data Types:\n{df.dtypes.to_string()}\n"

    # Ensure head_rows is not negative and not too large
    head_rows = max(0, min(head_rows, len(df)))
    if head_rows > 0:
        info_str += f"First {head_rows} rows:\n{df.head(head_rows).to_string()}\n"
    else:
        info_str += "No head rows requested or available.\n"

    info_str += f"Summary Statistics:\n{df.describe().to_string()}\n"

    return info_str

@st.cache_resource
def setup_agent(max_search_results: int) -> tuple[Any | None, str | None]:
    \"\"\"Initializes the orchestrator agent using st.cache_resource to run only once per max_search_results value.
    Returns a tuple (agent_instance, error_message).
    agent_instance is None if an error occurred.
    error_message is None if successful.
    \"\"\"
    print("Initializing AI agent...")
    try:
        # Create dynamic tools here, passing the functions defined above
        uploaded_doc_reader_tool = FunctionTool.from_defaults(
            fn=read_uploaded_document_tool_fn,
            name="read_uploaded_document",
            description="Reads the full text content of a document previously uploaded by the user. Input is the exact filename (e.g., 'my_dissertation.pdf'). Use this to answer questions about the content of uploaded documents."
        )

        dataframe_analyzer_tool = FunctionTool.from_defaults(
            fn=analyze_dataframe_tool_fn,
            name="analyze_uploaded_dataframe",
            description="Provides summary information (shape, columns, dtypes, head, describe) about a pandas DataFrame previously uploaded by the user. Input is the exact filename (e.g., 'my_data.csv'). Use this to understand the structure and basic statistics of uploaded datasets. For more complex analysis, use the 'code_interpreter' tool."
        )

        # Pass these dynamic tools and max_search_results to the agent creation function
        agent_instance = create_orchestrator_agent(
            dynamic_tools=[uploaded_doc_reader_tool, dataframe_analyzer_tool],
            max_search_results=max_search_results # Pass the parameter here
        )
        print("AI agent initialized successfully.")
        return agent_instance, None
    except Exception as e:
        error_message = f"Failed to initialize the AI agent. Please check configurations. Error: {e}"
        print(f"Error initializing AI agent: {e}")
        return None, error_message

# ... (rest of the Streamlit app code is omitted for brevity but available in previous turns) ...
"""

# ==============================================================================
# Shiny App Implementation
# ==============================================================================

import os
import json
import re
import uuid
from typing import List, Dict, Any, Tuple

from shiny import App, ui, render, reactive
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool

from agent import (
    create_orchestrator_agent,
    initialize_settings as initialize_agent_settings,
    DEFAULT_PROMPTS,
    generate_llm_greeting as agent_generate_llm_greeting,
    generate_suggested_prompts as agent_generate_suggested_prompts,
)
from tools import UI_ACCESSIBLE_WORKSPACE
from config import HF_USER_MEMORIES_DATASET_ID
from shiny_ui import get_app_ui

# --- Global Variables and Constants ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SIMPLE_STORE_PATH_RELATIVE = os.getenv("SIMPLE_STORE_PATH", "ragdb/simple_vector_store")
DB_PATH = os.path.join(PROJECT_ROOT, SIMPLE_STORE_PATH_RELATIVE)
AGENT_SESSION_KEY = "esi_orchestrator_agent"
MAX_CHAT_HISTORY_MESSAGES = 15

load_dotenv()
fs = HfFileSystem()

llm_settings_initialized: bool = False
llm_settings_error: str | None = None
orchestrator_agent: Any | None = None # Will hold the LlamaIndex agent instance
agent_init_error: str | None = None

# --- Core Logic Functions (Adapted for Shiny) ---
def setup_global_llm_settings() -> Tuple[bool, str | None]:
    print("Attempting to initialize LLM settings...")
    try:
        initialize_agent_settings()
        print("LLM settings initialized successfully.")
        return True, None
    except Exception as e:
        error_message = f"Fatal Error: Could not initialize LLM settings. {e}"
        print(error_message)
        return False, error_message

def read_uploaded_document_tool_fn(filename: str) -> str:
    # TODO: Access rv_uploaded_documents.get()
    print(f"TODO: Implement read_uploaded_document_tool_fn for {filename}")
    return f"Error: Document '{filename}' not found. Functionality pending."

def analyze_dataframe_tool_fn(filename: str, head_rows: int = 5) -> str:
    # TODO: Access rv_uploaded_dataframes.get()
    print(f"TODO: Implement analyze_dataframe_tool_fn for {filename}")
    return f"Error: DataFrame '{filename}' not found. Functionality pending."

def setup_agent(max_search_results: int = 10) -> Tuple[Any | None, str | None]:
    print("Attempting to initialize AI agent...")
    if not llm_settings_initialized:
        return None, "LLM settings must be initialized before setting up the agent."
    try:
        # These tools will be non-functional until their respective reactive values are used
        uploaded_doc_reader_tool = FunctionTool.from_defaults(fn=read_uploaded_document_tool_fn, name="read_uploaded_document", description="Reads...")
        dataframe_analyzer_tool = FunctionTool.from_defaults(fn=analyze_dataframe_tool_fn, name="analyze_uploaded_dataframe", description="Analyzes...")
        
        agent_instance = create_orchestrator_agent(
            dynamic_tools=[uploaded_doc_reader_tool, dataframe_analyzer_tool],
            max_search_results=max_search_results
        )
        print("AI agent initialized successfully.")
        return agent_instance, None
    except Exception as e:
        error_message = f"Failed to initialize the AI agent. Error: {e}"
        print(f"Error initializing AI agent: {e}")
        return None, error_message

def generate_llm_greeting_shiny() -> str:
    # return agent_generate_llm_greeting() # If simple enough, otherwise use static
    return "Welcome to ESI! How can I assist you with your research today?"

def generate_suggested_prompts_shiny(chat_history: List[Dict[str, Any]]) -> List[str]:
    print(f"generate_suggested_prompts_shiny called. History length: {len(chat_history)}. Returning defaults for now.")
    # In future, could call: return agent_generate_suggested_prompts(chat_history)
    return list(DEFAULT_PROMPTS)

def format_chat_history_shiny(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Converts Shiny message history to LlamaIndex ChatMessage list, truncating."""
    truncated_messages = messages[-MAX_CHAT_HISTORY_MESSAGES:]
    history = []
    for msg in truncated_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history

def get_agent_response_shiny(query: str, chat_history: List[ChatMessage], temp: float, verbosity: int) -> str:
    """Gets response from the agent. Returns canned response for now."""
    global orchestrator_agent # Access the global agent instance
    print(f"get_agent_response_shiny called. Query: '{query}', Temp: {temp}, Verbosity: {verbosity}")
    if orchestrator_agent is None:
        print("Agent not initialized. Returning error message.")
        return "Error: The AI agent is not available. Please check startup logs."

    # TODO: Full agent interaction logic will be enabled in a later step.
    # For now, returning a canned response to test the UI flow.
    # try:
    #     if Settings.llm and hasattr(Settings.llm, 'temperature'):
    #         Settings.llm.temperature = temp
    #     else:
    #         print("Warning: Settings.llm not configured or no temperature attribute.")
        
    #     modified_query = f"Verbosity Level: {verbosity}. {query}"
    #     response = orchestrator_agent.chat(modified_query, chat_history=chat_history)
    #     response_text = response.response if hasattr(response, 'response') else str(response)
    #     return response_text
    # except Exception as e:
    #     print(f"Error getting agent response: {type(e).__name__} - {e}")
    #     return f"Apologies, an error occurred: {e}"
    
    return f"Agent received: '{query}'. Response generation is currently a placeholder."


# --- Application Startup ---
print("Shiny App Startup: Initializing settings and agent...")
try:
    llm_settings_initialized, llm_settings_error = setup_global_llm_settings()
    if llm_settings_initialized:
        # Initialize agent with default search results count from a reactive value later
        orchestrator_agent, agent_init_error = setup_agent(10) # Defaulting to 10 for now
    else:
        agent_init_error = "Agent setup skipped: LLM settings failed."
        print(agent_init_error)
except Exception as e:
    print(f"Unhandled exception during startup: {e}")
    agent_init_error = f"Unhandled startup exception: {e}"

if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY environment variable not set.")

app_ui = get_app_ui()

def server(input, output, session):
    print("--- Server Function Initializing ---")

    rv_user_id = reactive.Value(str(uuid.uuid4()))
    rv_long_term_memory_enabled = reactive.Value(True)
    rv_chat_metadata = reactive.Value({})
    rv_all_chat_messages = reactive.Value({})
    rv_current_chat_id = reactive.Value(None)
    rv_messages = reactive.Value([])
    rv_chat_modified = reactive.Value(False)
    rv_suggested_prompts = reactive.Value(list(DEFAULT_PROMPTS))
    rv_llm_temperature = reactive.Value(0.7) # Default from UI
    rv_llm_verbosity = reactive.Value(3)   # Default from UI
    rv_search_results_count = reactive.Value(10) # Default from UI
    rv_uploaded_documents = reactive.Value({})
    rv_uploaded_dataframes = reactive.Value({})

    print(f"Reactive values initialized. User ID: {rv_user_id.get()}")

    def create_new_chat_session_shiny():
        # (Implementation from previous step - kept for brevity)
        print("Creating new chat session...")
        new_chat_id = str(uuid.uuid4())
        current_meta = rv_chat_metadata.get().copy()
        new_chat_name = "Current Session"
        if rv_long_term_memory_enabled.get():
            existing_idea_nums = [int(m.group(1)) for name in current_meta.values() if (m := re.match(r"Idea (\d+)", name))]
            next_idea_num = max(existing_idea_nums) + 1 if existing_idea_nums else 1
            new_chat_name = f"Idea {next_idea_num}"
        current_meta[new_chat_id] = new_chat_name
        rv_chat_metadata.set(current_meta)
        current_all_messages = rv_all_chat_messages.get().copy()
        initial_greeting_msg = {"role": "assistant", "content": generate_llm_greeting_shiny()}
        current_all_messages[new_chat_id] = [initial_greeting_msg]
        rv_all_chat_messages.set(current_all_messages)
        rv_current_chat_id.set(new_chat_id)
        rv_messages.set([initial_greeting_msg])
        rv_chat_modified.set(False)
        rv_suggested_prompts.set(generate_suggested_prompts_shiny([initial_greeting_msg]))
        print(f"Created new chat: '{new_chat_name}' (ID: {new_chat_id})")


    def switch_chat_shiny(chat_id_to_switch: str):
        # (Implementation from previous step - kept for brevity)
        print(f"Attempting to switch to chat ID: {chat_id_to_switch}")
        if not rv_long_term_memory_enabled.get():
            create_new_chat_session_shiny()
            return
        if chat_id_to_switch not in rv_chat_metadata.get():
            print(f"Error: Chat ID '{chat_id_to_switch}' not found.")
            return
        rv_current_chat_id.set(chat_id_to_switch)
        messages_for_chat = rv_all_chat_messages.get().get(chat_id_to_switch, [])
        rv_messages.set(messages_for_chat)
        rv_suggested_prompts.set(generate_suggested_prompts_shiny(messages_for_chat))
        rv_chat_modified.set(True)
        print(f"Switched to chat: '{rv_chat_metadata.get().get(chat_id_to_switch, 'Unknown')}'")

    if rv_current_chat_id.get() is None:
        create_new_chat_session_shiny()

    @reactive.Effect
    @reactive.event(input.new_chat_btn)
    def handle_new_chat():
        create_new_chat_session_shiny()

    def process_and_send_query(query_text: str):
        """Helper function to process a query, get response, and update state."""
        if not query_text.strip():
            return

        current_messages = rv_messages.get().copy()
        current_messages.append({"role": "user", "content": query_text})
        rv_messages.set(current_messages)
        
        # TODO: Scroll to bottom of chat logic would be needed here in a real UI

        formatted_hist = format_chat_history_shiny(current_messages)
        
        # Use reactive values for temperature and verbosity
        temp = rv_llm_temperature.get()
        verbosity = rv_llm_verbosity.get()
        
        assistant_response = get_agent_response_shiny(query_text, formatted_hist, temp, verbosity)
        
        current_messages_after_assist = rv_messages.get().copy() # Re-fetch in case of async changes
        current_messages_after_assist.append({"role": "assistant", "content": assistant_response})
        rv_messages.set(current_messages_after_assist)
        
        rv_suggested_prompts.set(generate_suggested_prompts_shiny(current_messages_after_assist))
        rv_chat_modified.set(True)
        
        if rv_long_term_memory_enabled.get():
            print(f"TODO: Save chat history for chat ID: {rv_current_chat_id.get()}")

    @reactive.Effect
    @reactive.event(input.send_message_btn)
    def handle_send_message():
        user_query = input.chat_input()
        print(f"Send button clicked. Query: '{user_query}'")
        if user_query:
            process_and_send_query(user_query)
            ui.update_text_area("chat_input", value="") # Clear input after sending

    # Handle first suggested prompt click (example)
    # This pattern would need to be repeated or generalized for all suggested prompts
    @reactive.Effect
    @reactive.event(input.suggested_prompt_0) # Assuming DEFAULT_PROMPTS has at least one
    def handle_suggested_prompt_0():
        if DEFAULT_PROMPTS:
            prompt_text = DEFAULT_PROMPTS[0]
            print(f"Suggested prompt 0 ('{prompt_text}') clicked.")
            # Option 1: Populate input (less direct)
            # ui.update_text_area("chat_input", value=prompt_text)
            # Option 2: Directly process (more direct)
            process_and_send_query(prompt_text)
            ui.update_text_area("chat_input", value="") # Clear input if needed, or let user edit


    # --- Output Renderers (UI updates) ---
    @output
    @render.ui
    def chat_history_output():
        # (Implementation from previous step - kept for brevity)
        history_items = []
        new_chat_label = "➕ New Chat"
        if not rv_long_term_memory_enabled.get(): new_chat_label += " (Temporary)"
        history_items.append(ui.input_action_button("new_chat_btn", new_chat_label, class_="btn-primary w-100 mb-2"))
        if not rv_long_term_memory_enabled.get(): history_items.append(ui.tags.div(ui.tags.strong("Warning:"), " LTM disabled.", style="color: orange; font-size: 0.9em;"))
        if rv_long_term_memory_enabled.get():
            current_metadata = rv_chat_metadata.get()
            if not current_metadata: history_items.append(ui.tags.p("No saved chats.", style="text-align: center; color: grey;"))
            else:
                for chat_id, chat_name in sorted(current_metadata.items(), key=lambda item: item[1]):
                    is_current = chat_id == rv_current_chat_id.get()
                    history_items.append(ui.tags.div(ui.row(
                        ui.column(9, ui.input_action_button(id=f"select_chat_{chat_id}", label=chat_name, class_=f"{'btn-primary' if is_current else 'btn-light'} w-100 text-start")),
                        ui.column(3, ui.input_action_button(id=f"options_chat_{chat_id}", label="⚙️", class_="btn-secondary"))
                    ), style="margin-bottom: 5px;"))
        return ui.tags.div(*history_items)

    @output
    @render.ui
    def uploaded_files_output():
        # (Implementation from previous step - kept for brevity)
        items = []
        docs = rv_uploaded_documents.get()
        dfs = rv_uploaded_dataframes.get()
        if not docs and not dfs: items.append(ui.tags.p("No files uploaded.", style="text-align: center; color: grey;"))
        if docs:
            items.append(ui.tags.h6("Documents:", style="margin-top: 10px;"))
            for i, name in enumerate(docs.keys()): items.append(ui.tags.div(ui.row(ui.column(1, ui.tags.span("📄")), ui.column(8, ui.tags.span(name)), ui.column(3, ui.input_action_button(f"remove_doc_{i}_{name}", "🗑️", class_="btn-danger btn-sm"))), style="padding: 3px 0;"))
        if dfs:
            items.append(ui.tags.h6("Datasets:", style="margin-top: 10px;"))
            for i, name in enumerate(dfs.keys()): items.append(ui.tags.div(ui.row(ui.column(1, ui.tags.span("📊")), ui.column(8, ui.tags.span(name)), ui.column(3, ui.input_action_button(f"remove_df_{i}_{name}", "🗑️", class_="btn-danger btn-sm"))), style="padding: 3px 0;"))
        return ui.tags.div(*items)

    @output
    @render.ui
    def chat_display_output():
        # (Implementation from previous step - kept for brevity)
        message_list = rv_messages.get()
        if not message_list: return ui.tags.div("No messages yet.", style="text-align: center; color: grey; padding: 20px; min-height: 300px;")
        elements = []
        for i, msg in enumerate(message_list):
            role, content = msg.get("role", "u"), msg.get("content", "")
            is_user = role == "user"
            style = "background-color: #DCF8C6; margin-left: auto; margin-right: 10px;" if is_user else "background-color: #E0E0E0; margin-left: 10px; margin-right: auto;"
            style += "padding: 10px; margin: 5px; border-radius: 15px; max-width: 70%;"
            btns = [ui.input_action_button(f"copy_msg_{i}", "📄", class_="btn-sm btn-outline-secondary m-1")]
            if role == "assistant" and i == len(message_list) - 1: btns.append(ui.input_action_button(f"regenerate_msg_{i}", "🔄", class_="btn-sm btn-outline-warning m-1"))
            elements.append(ui.tags.div(ui.tags.div(ui.tags.strong(f"{role.capitalize()}:")), ui.markdown(content), ui.tags.div(*btns, style="margin-top: 5px; text-align: right;"), style=style, class_=f"message-bubble {'user' if is_user else 'assistant'}-bubble"))
        return ui.tags.div(*elements, style="display: flex; flex-direction: column; padding: 10px; min-height: 300px;")


    @output
    @render.ui
    def suggested_prompts_output():
        # (Implementation from previous step - kept for brevity)
        prompts = rv_suggested_prompts.get()
        if not prompts: return ui.tags.div()
        btns = [ui.input_action_button(f"suggested_prompt_{i}", p, class_="btn btn-outline-secondary m-1") for i, p in enumerate(prompts)]
        return ui.tags.div(ui.tags.h5("Suggested Prompts:", style="margin-bottom: 5px;"), ui.tags.div(*btns, style="display: flex; flex-wrap: wrap; justify-content: center; padding: 5px;"), style="padding-top: 10px; border-top: 1px solid #eee; margin-top: 10px;")
    
    @output
    @render.text
    def startup_status():
        # (Implementation from previous step - kept for brevity)
        status = []
        if llm_settings_error: status.append(f"LLM Error: {llm_settings_error}")
        elif llm_settings_initialized: status.append("LLM OK.")
        if agent_init_error: status.append(f"Agent Error: {agent_init_error}")
        elif orchestrator_agent: status.append("Agent OK.")
        if not os.getenv("GOOGLE_API_KEY"): status.append("GOOG_KEY Missing.")
        print(f"Console Startup: {status}")
        return " | ".join(status) if status else "System Ready."

    print("--- Server Function Initialized ---")

app = App(app_ui, server)
