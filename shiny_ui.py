from shiny import ui
from shiny.experimental import ui as xui # For accordion

def chat_options_modal_ui(chat_id: str, chat_name: str) -> ui.Modal:
    """
    Creates the UI for the chat options modal.
    """
    return ui.modal(
        ui.tags.h5(f"Options for: {chat_name}"),
        ui.hr(),
        ui.input_text("modal_rename_chat_input", "Rename Chat:", value=chat_name),
        ui.input_action_button("modal_save_rename_btn", "Save Name", class_="btn-primary btn-sm"),
        ui.hr(),
        ui.tags.h6("Download Chat:"),
        ui.download_button("download_markdown_handler", "Download as Markdown (.md)", class_="btn-success btn-sm w-100 my-1"),
        ui.download_button("download_docx_handler", "Download as Word (.docx)", class_="btn-success btn-sm w-100 my-1"),
        ui.hr(),
        ui.input_action_button("modal_delete_chat_btn", "Delete Chat", class_="btn-danger btn-sm"),
        title=False,
        easy_close=True,
        footer=ui.modal_button("Dismiss", class_="btn-secondary btn-sm")
    )

def delete_chat_confirmation_modal_ui(chat_name: str) -> ui.Modal:
    """
    Creates the UI for the delete chat confirmation modal.
    """
    return ui.modal(
        ui.modal_title("Confirm Deletion"),
        ui.tags.p(f"Are you sure you want to delete the chat '{chat_name}'? This action cannot be undone."),
        footer=ui.tag_list(
            ui.input_action_button("modal_confirm_delete_btn", "Yes, Delete", class_="btn-danger"),
            ui.modal_button("Cancel", class_="btn-secondary")
        ),
        easy_close=True
    )

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
                    ui.input_switch("long_term_memory_enabled", "Enable Long-term Memory", value=True), # Initial value, server will update from cookie
                ),
                xui.accordion_panel(
                    "Chat History",
                    ui.output_ui("chat_history_output")
                ),
                xui.accordion_panel(
                    "Upload Files",
                    ui.input_file("file_uploader",
                                  "Upload a document or dataset",
                                  multiple=False,
                                  accept=[".pdf", ".docx", ".md", ".txt", ".csv", ".xlsx", ".sav"]),
                    ui.hr(),
                    ui.output_ui("uploaded_files_output")
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
                open=["Upload Files", "LLM Settings"]
            ),
            width="350px"
        ),
        ui.output_ui("chat_display_output"),
        ui.output_ui("suggested_prompts_output"),
        ui.tags.div(
            ui.input_text_area("chat_input", label="", placeholder="Ask ESI anything...", rows=3, width="90%"),
            ui.input_action_button("send_message", "Send", class_="btn-primary"),
            style="display: flex; align-items: center; padding: 5px; gap: 5px;"
        ),
        ui.output_ui("clipboard_trigger_output"),
        # Removed the large inline script block that was here.
        # Adding script tags for jQuery and the external cookie_handler.js
        ui.tags.script(src="https://code.jquery.com/jquery-3.6.0.min.js"),
        ui.tags.script(src="js/cookie_handler.js"), # Assumes js folder is in www
        title="ESI - Experimental Shiny Interface",
    )
