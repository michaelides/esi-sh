from shiny import ui
from shiny.experimental import ui as xui # For accordion

def get_app_ui():
    """
    Defines the main application UI layout.
    """
    return ui.page_sidebar(
        ui.sidebar(
            xui.accordion(
                xui.accordion_panel(
                    "LLM Settings",
                    ui.input_slider("llm_temperature", "Creativity (Temperature)", min=0, max=1, value=0.7, step=0.1),
                    ui.input_slider("llm_verbosity", "Verbosity", min=1, max=5, value=3, step=1),
                    ui.input_slider("search_results_count", "Number of Search Results", min=1, max=20, value=10, step=1),
                    ui.input_switch("long_term_memory_enabled", "Enable Long-term Memory", value=True),
                ),
                xui.accordion_panel(
                    "Chat History",
                    ui.output_ui("chat_history_output")
                ),
                xui.accordion_panel(
                    "Upload Files",
                    ui.input_file("file_uploader",
                                  "Upload a document or dataset",
                                  multiple=False, # Changed to False as per common use case, can be True if batch uploads are desired later
                                  accept=[".pdf", ".docx", ".md", ".txt", ".csv", ".xlsx", ".sav"]),
                    ui.hr(), # Visual separator
                    ui.output_ui("uploaded_files_output") # To display the list of uploaded files
                ),
                xui.accordion_panel(
                    "About ESI",
                    ui.markdown(
                        """
                        **ESI (Experimental Shiny Interface)** is a prototype demonstrating
                        how to build a conversational AI research assistant using Python Shiny.

                        Features include:
                        - Interaction with a LlamaIndex-powered agent.
                        - Dynamic LLM parameter adjustment.
                        - File upload and basic analysis capabilities (planned).
                        - Chat history management (planned).
                        """
                    ),
                ),
                id="sidebar_accordion",
                # Open "Upload Files" and "LLM Settings" by default for easier access during dev
                open=["Upload Files", "LLM Settings"]
            ),
            width="350px"
        ),
        ui.output_ui("chat_display_output"),
        ui.output_ui("suggested_prompts_output"),
        ui.tags.div(
            ui.input_text_area("chat_input", label="", placeholder="Ask ESI anything...", rows=3, width="90%"), # Give chat input more space
            ui.input_action_button("send_message", "Send", class_="btn-primary"), # Style send button
            style="display: flex; align-items: center; padding: 5px; gap: 5px;" # Added gap
        ),
        title="ESI - Experimental Shiny Interface",
    )
