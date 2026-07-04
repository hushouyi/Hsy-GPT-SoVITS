"""
GPT-SoVITS Sentence Recording + One-Click Training WebUI

Main entry point: Gradio application with:
  - Training tab (sub-tabs: Recording, Training)
  - Inference tab
  - Global status bar
  - Port cleanup on startup
  - Auto-open browser
  - /quit endpoint for graceful shutdown

Usage:
    set PYTHONIOENCODING=utf-8
    .\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe sentence_webui.py
"""

import os
import sys
import time
import threading
import webbrowser
import gradio as gr
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_recorder.state import AppState
from sentence_recorder.project_manager import ProjectManager
from sentence_recorder.model_utils import cleanup_all_ports, stop_inference_api
from sentence_tabs.tab_recording import create_recording_tab
from sentence_tabs.tab_training import create_training_tab
from sentence_tabs.tab_inference import create_inference_tab


# ==================== Port Cleanup ====================

def startup_cleanup():
    """Clean up ports before starting."""
    print("[OK] Cleaning up ports...")
    for port in [7860, 9880, 17860]:
        _kill_process_on_port(port)


def _kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given port."""
    import re
    import subprocess
    try:
        result = subprocess.run(
            f'netstat -ano | findstr ":{port} "',
            capture_output=True, text=True, shell=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 5 and 'LISTENING' in line:
                pid = parts[-1]
                subprocess.run(f'taskkill /f /pid {pid}', shell=True,
                              capture_output=True, timeout=5)
                print(f"[OK] Killed process {pid} on port {port}")
    except Exception:
        pass


# ==================== /quit HTTP Server ====================

_should_exit = False


class QuitHandler(BaseHTTPRequestHandler):
    """HTTP server that handles /quit for graceful shutdown."""

    def do_GET(self):
        global _should_exit
        if self.path == '/quit':
            _should_exit = True
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'OK')
            # Trigger shutdown in a thread
            threading.Thread(target=self._do_shutdown, daemon=True).start()
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'QUIT Server Running')

    def _do_shutdown(self):
        time.sleep(0.5)
        os._exit(0)

    def log_message(self, format, *args):
        pass  # Suppress HTTP log


def start_quit_server(port: int = 17860):
    """Start the /quit HTTP server in background."""
    try:
        server = HTTPServer(('127.0.0.1', port), QuitHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[OK] QUIT server listening on port {port}")
        return server
    except Exception as e:
        print(f"[WARN] Could not start QUIT server: {e}")
        return None


# ==================== Global Status Bar ====================

def get_status_html() -> str:
    """Generate global status bar HTML."""
    state = AppState.get_status_text()
    project = AppState.get_current_project()
    training = AppState.get_training()
    inference = AppState.get_inference_api()
    recording = AppState.get_recording()

    # Determine icon
    if training:
        icon = "🟡"
    elif recording:
        icon = "🔴"
    elif inference:
        icon = "🔵"
    else:
        icon = "🟢"

    # Recording count
    from sentence_recorder.mapping import MappingManager
    from sentence_recorder.project_manager import ProjectManager as PM
    mapping_path = PM.get_mapping_path(project)
    count = 0
    total = 0
    if os.path.exists(mapping_path):
        mm = MappingManager()
        mm.load(mapping_path)
        count = len(mm.get_confirmed())
        total = len(mm.get_all())

    return f'''
    <div style="display:flex;align-items:center;gap:16px;padding:4px 12px;
         background:#f0f0f0;border-radius:4px;font-size:13px;">
        <span>{icon} <b>{state.split("]")[0].strip("[").strip() if "]" in state else state}</b></span>
        <span>|</span>
        <span>Project: <b>{project}</b></span>
        <span>|</span>
        <span>Recorded: <b>{count}/{max(total,count)}</b></span>
        <span>|</span>
        <span style="color:{"#4CAF50" if count>=5 else "#FF9800"};">
            {"✅ Trainable" if count >= 5 else "❌ Need >=5 recordings"}
        </span>
    </div>
    '''


# ==================== Build Application ====================

def create_app():
    """Create and return the Gradio application."""

    with gr.Blocks(
        title="GPT-SoVITS Recording + Training WebUI",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        css="""
        /* ===== Centered pagination row ===== */
        .pagination-wrap { justify-content:center !important; align-items:center !important; gap:6px !important; }
        .pagination-wrap > * { flex:0 0 auto !important; min-width:0 !important; width:auto !important; }
        .pagination-wrap button {
            height:28px !important; min-height:28px !important;
            line-height:28px !important; padding:0 8px !important;
            font-size:13px !important;
        }

        /* ===== Jump input: completely flatten the container ===== */
        .page-jump {
            width:55px !important; min-width:55px !important; max-width:55px !important;
            flex:0 0 55px !important;
            padding:0 !important; margin:0 !important;
        }
        .page-jump > div,
        .page-jump .form {
            padding:0 !important; margin:0 !important; gap:0 !important;
            border:none !important; box-shadow:none !important;
            background:transparent !important;
        }
        .page-jump .gr-box,
        .page-jump .gr-input {
            padding:0 !important; margin:0 !important;
            height:28px !important; min-height:28px !important;
            width:100% !important; min-width:0 !important;
            border:none !important; box-shadow:none !important;
            background:transparent !important;
        }
        .page-jump input {
            height:26px !important; min-height:26px !important;
            padding:0 4px !important; margin:0 !important;
            text-align:center !important; font-size:13px !important;
            border:1px solid #ccc !important;
            border-radius:4px !important;
            box-shadow:none !important;
            width:100% !important; box-sizing:border-box !important;
        }

        /* ===== Project label: don't stretch ===== */
        .project-label { flex:0 0 auto !important; min-width:0 !important; width:auto !important; }

        /* ===== Recording tab: compact everything ===== */
        #recording-tab button {
            height:28px !important; min-height:28px !important;
            padding:0 6px !important; line-height:28px !important;
            font-size:12px !important;
        }
        #recording-tab .gr-dropdown {
            min-height:28px !important;
        }
        #recording-tab .gr-dropdown .gr-box {
            height:28px !important; min-height:28px !important;
        }
        #recording-tab .gr-dropdown .form {
            padding:0 !important; margin:0 !important; gap:0 !important;
        }
        #recording-tab .gr-dropdown .gr-box {
            padding:0 4px !important; margin:0 !important;
            min-height:28px !important; height:28px !important;
            border:1px solid #ccc !important; border-radius:4px !important;
        }
        #recording-tab .gr-dropdown input {
            height:26px !important; min-height:26px !important;
            font-size:13px !important; padding:0 4px !important;
        }
        #recording-tab .gr-form {
            gap:2px !important;
        }

        /* ===== Recording animation ===== */
        @keyframes recPulse { 0%{opacity:1;} 50%{opacity:0.2;} 100%{opacity:1;} }
        """
    ) as app:

        # ─── Global Status Bar (at top, below title but above tabs) ───
        status_bar = gr.HTML(value=get_status_html())

        # ─── Browser close → /quit beacon ───
        gr.HTML('''
        <script>
        window.addEventListener("beforeunload", function() {
            navigator.sendBeacon("http://127.0.0.1:17860/quit");
        });
        </script>
        ''')

        # ─── Main Tabs ───
        with gr.Tabs() as main_tabs:
            # ===== Tab 1: Training =====
            with gr.Tab("Training", id=0):
                # Sub-tabs for Recording and Training
                with gr.Tabs() as train_sub_tabs:
                    with gr.Tab("Recording", id=0) as rec_sub_tab:
                        rec_components = create_recording_tab()

                    with gr.Tab("Training", id=1) as train_sub_tab:
                        train_components = create_training_tab(rec_components)

            # ===== Tab 2: Inference =====
            with gr.Tab("Inference", id=1):
                infer_components = create_inference_tab()

        # ─── Status Bar Update Timer ───
        def update_status():
            return get_status_html()

        # Update status every 3 seconds
        app.load(update_status, None, status_bar, every=3)

        # ─── Auto-resfresh components on page load ───
        def on_app_load():
            """Initialize data when app loads."""
            # Ensure default project exists
            ProjectManager.init_default()
            # Load default project into recording tab
            project = AppState.get_current_project()
            rec_components["load_project"](project)
            # Build project UI state (2 values)
            projects = ProjectManager.list_projects()
            locked = ProjectManager.is_locked(project)
            proj_state = [
                gr.update(choices=projects, value=project),
                gr.update(value="\U0001f512" if locked else "\U0001f513",
                          variant="stop" if locked else "secondary"),
            ]
            # Return: project_ui(2) + current_page(1) + page_values(47) = 50
            return proj_state + [1] + rec_components["refresh_page"](1)

        app.load(
            fn=on_app_load,
            inputs=[],
            outputs=rec_components["all_outputs"],
        )

        # ─── Wire 完成录制 → Switch to Training sub-tab ───
        def on_done_click():
            """Handle 完成录制 button click."""
            print("[OK] Complete Recording clicked -> switching to Training tab")
            return 1  # 1 = Training sub-tab index

        rec_components["btn_done"].click(
            fn=on_done_click,
            inputs=[],
            outputs=[train_sub_tabs]
        )

        # ─── Periodic progress updates for training ───
        def periodic_train_update():
            """Update training progress if training is active."""
            if not AppState.get_training():
                return [None] * 4
            return train_components["update_progress"]()

        # Wire training update timer
        train_components.get("progress_bar") and None  # just needs to exist
        # We'll use a simple approach: update on every interaction
        # For real-time updates, Gradio's .every parameter works well

        # ─── Refresh inference models when switching to inference tab ───
        def on_inference_tab_select(evt: gr.SelectData):
            """Refresh inference components when tab is selected."""
            # Refresh model dropdowns
            infer_components.get("btn_refresh_models") and None
            return []

        # We'll wire this via the tab change event

    return app


# ==================== Main Entry ====================

def main():
    """Main entry point."""
    # Cleanup ports before starting
    startup_cleanup()

    # Start /quit server
    quit_server = start_quit_server(17860)

    # Initialize default project
    ProjectManager.init_default()

    # Create app
    app = create_app()

    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://127.0.0.1:7860')
        print("[OK] Browser opened to http://127.0.0.1:7860")

    threading.Thread(target=open_browser, daemon=True).start()

    # Launch
    print("[OK] Starting GPT-SoVITS Recording + Training WebUI on port 7860...")
    print("[OK] Open http://127.0.0.1:7860 in your browser")
    print("[OK] Close the browser tab to exit gracefully")

    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
        )
        # Keep main thread alive
        while not _should_exit:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[OK] Shutting down...")
    finally:
        # Cleanup
        print("[OK] Cleaning up...")
        try:
            stop_inference_api(9880)
        except Exception:
            pass
        cleanup_all_ports()
        print("[OK] Goodbye!")
        os._exit(0)


if __name__ == "__main__":
    main()
