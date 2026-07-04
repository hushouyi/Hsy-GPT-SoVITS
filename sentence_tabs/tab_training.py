"""
Training tab page.

Displays:
  - Data source selection (current project or multi-project merge)
  - Training configuration parameters
  - Training control: start button, progress bar, status display, log output
"""

import os
import threading
import time
import gradio as gr

from sentence_recorder.training_pipeline import TrainingPipeline, TrainingConfig
from sentence_recorder.project_manager import ProjectManager
from sentence_recorder.mapping import MappingManager
from sentence_recorder.state import AppState


def create_training_tab(recording_refs: dict = None):
    """Create the training tab UI and return components."""

    _pipeline = None
    _log_lines = []
    _start_time = 0
    _progress_val = 0
    _phase_text = ""
    _timer_active = False

    # ==================== Helper Functions ====================

    def _get_data_source_count(data_source_type: str, selected_projects: list) -> int:
        """Count confirmed recordings from data sources."""
        total = 0
        if data_source_type == "current":
            project = AppState.get_current_project()
            projects = [project]
        else:
            projects = selected_projects or [AppState.get_current_project()]

        for proj in projects:
            mapping_path = ProjectManager.get_mapping_path(proj)
            if os.path.exists(mapping_path):
                mm = MappingManager()
                mm.load(mapping_path)
                total += len(mm.get_confirmed())
        return total

    def _get_timer_text() -> str:
        """Get elapsed/estimated time text."""
        if _start_time == 0:
            return ""
        elapsed = time.time() - _start_time
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        if _progress_val > 0 and _progress_val < 100:
            estimated = int(elapsed / _progress_val * (100 - _progress_val))
            est_str = f"{estimated//60:02d}:{estimated%60:02d}"
            return f"Elapsed: {elapsed_str}  |  Estimated remaining: {est_str}"
        return f"Elapsed: {elapsed_str}"

    # ==================== Build UI ====================

    with gr.Column() as train_tab:
        # ─── Data Source ───
        with gr.Group():
            gr.Markdown("### Data Source")
            data_source = gr.Radio(
                choices=["Current project", "Multi-project merge"],
                value="Current project",
                label="Data source type"
            )
            # Multi-project selection (shown when multi-project is selected)
            project_select = gr.CheckboxGroup(
                choices=[],
                label="Select projects (you can select multiple)",
                visible=False
            )
            source_count = gr.Markdown("**Recordings available: 0**")

        # ─── Training Config ───
        with gr.Group():
            gr.Markdown("### Training Configuration")

            with gr.Row():
                model_version = gr.Dropdown(
                    choices=["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"],
                    value="v2Pro",
                    label="Model version",
                    info="Select the model version to train"
                )
                exp_name = gr.Textbox(
                    value="my_voice",
                    label="Experiment name",
                    info="Used as directory name in logs/"
                )

            s1_epochs = gr.Slider(
                minimum=2, maximum=50, value=15, step=1,
                label="S1 (GPT) training epochs",
                info="More epochs = better quality but longer training"
            )
            s2_epochs = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="S2 (SoVITS) training epochs",
                info="Usually 2-5 is enough, more may overfit"
            )
            batch_size = gr.Slider(
                minimum=1, maximum=32, value=8, step=1,
                label="Batch size (per GPU)",
                info="Auto-detected based on GPU memory. Lower if OOM."
            )

            # Advanced settings
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    text_low_lr_rate = gr.Slider(
                        minimum=0.2, maximum=0.6, value=0.4, step=0.05,
                        label="Text LR rate"
                    )
                    precision = gr.Dropdown(
                        choices=["32-bit", "16-mixed"],
                        value="16-mixed",
                        label="Precision mode"
                    )
                with gr.Row():
                    save_every_epoch = gr.Number(
                        value=1, minimum=1, maximum=50, step=1,
                        label="Save every N epochs"
                    )
                    gpu_numbers = gr.Textbox(
                        value="0",
                        label="GPU IDs (comma-separated)"
                    )
                with gr.Row():
                    if_save_every_weights = gr.Checkbox(
                        value=True, label="Save every weight (output to weights dir)"
                    )
                    if_save_latest = gr.Checkbox(
                        value=True, label="Keep only latest checkpoint"
                    )
                    if_dpo = gr.Checkbox(
                        value=False, label="Enable DPO (experimental)"
                    )

        # ─── Training Control ───
        with gr.Group():
            gr.Markdown("### Training Control")

            btn_train = gr.Button("🚀 Start Training", variant="primary")

            progress_bar = gr.HTML(value="""
            <div style="background:#e0e0e0;height:24px;border-radius:12px;overflow:hidden;">
                <div id="train-progress" style="background:linear-gradient(90deg,#4CAF50,#2196F3);width:0%;height:100%;
                     border-radius:12px;transition:width 0.5s;"></div>
            </div>
            """)

            with gr.Row():
                phase_text = gr.Markdown("**Status:** Idle")
                time_text = gr.Markdown("")

            log_output = gr.Textbox(
                value="Training log will appear here...\n",
                label="Training Log",
                lines=10,
                max_lines=50,
                interactive=False
            )

    # ==================== Event Handlers ====================

    def update_source_count(source_type: str, selected: list) -> str:
        """Update the displayed recording count."""
        count = _get_data_source_count(source_type, selected or [])
        return f"**Recordings available: {count}**"

    def toggle_project_select(source_type: str):
        """Show/hide multi-project selection based on data source type."""
        if source_type == "Multi-project merge":
            # Update project list
            projects = ProjectManager.list_projects()
            return gr.update(visible=True, choices=projects)
        else:
            return gr.update(visible=False)

    def start_training(
        source_type, selected_projects, model_ver, exp,
        s1_ep, s2_ep, batch, lr_rate, prec,
        save_epoch, gpu_ids, save_every, save_latest, dpo
    ):
        """Start training in background thread."""
        global _pipeline, _start_time, _progress_val, _log_lines, _timer_active

        if AppState.get_training():
            return [gr.update(interactive=False), "**Status:** Training already running", "", "A training session is already running.\n"]

        if AppState.get_inference_api():
            return [gr.update(interactive=False), "**Status:** Cannot train while inference API is running", "", "Please stop the inference service first.\n"]

        # Get data source projects
        if source_type == "Current project":
            projects = [AppState.get_current_project()]
        else:
            projects = selected_projects or [AppState.get_current_project()]

        # Count recordings
        count = _get_data_source_count(source_type, projects)
        if count < 5:
            return [gr.update(interactive=False),
                    "**Status:** Need at least 5 recordings",
                    "", f"Only {count} recordings found. Need at least 5.\n"]

        # Disable button and start
        _start_time = time.time()
        _log_lines = [f"[START] Training started at {time.strftime('%H:%M:%S')}"]
        _log_lines.append(f"[CONFIG] Model: {model_ver}, S1: {s1_ep}ep, S2: {s2_ep}ep, Batch: {batch}")
        _log_lines.append(f"[CONFIG] Data sources: {', '.join(projects)} ({count} recordings)")

        config = TrainingConfig(
            exp_name=exp,
            model_version=model_ver,
            s1_epochs=s1_ep,
            s2_epochs=s2_ep,
            batch_size=batch,
            text_low_lr_rate=lr_rate,
            precision=prec.replace("-bit", "").replace("-mixed", ""),
            save_every_epoch=int(save_epoch),
            if_save_every_weights=save_every,
            if_save_latest=save_latest,
            if_dpo=dpo,
            gpu_numbers=gpu_ids,
            data_sources=projects,
        )

        _pipeline = TrainingPipeline(config)
        progress_text = "**Status:** Training in progress..."
        log_text = '\n'.join(_log_lines) + '\n'

        def progress_callback(percent: float, phase: str, message: str):
            global _progress_val, _phase_text
            _progress_val = percent
            _phase_text = f"**Status:** [{phase}] {message}"
            _log_lines.append(f"[{time.strftime('%H:%M:%S')}] [{phase}] {message}")

        def train_thread():
            result = _pipeline.run(progress_callback=progress_callback)
            if result.get("success"):
                _log_lines.append(f"[DONE] Training completed successfully!")
                _log_lines.append(f"[DONE] Model saved to: {result.get('model_dir', '')}")
            else:
                _log_lines.append(f"[FAIL] Training failed: {result.get('error', 'Unknown error')}")

        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

        return [
            gr.update(interactive=False),
            "**Status:** Training in progress...",
            "",
            log_text,
        ]

    def update_progress():
        """Update progress bar and phase text periodically."""
        global _progress_val, _phase_text

        if not AppState.get_training() and _progress_val == 0:
            return [None] * 4  # No update needed

        # Build progress bar HTML
        pct = min(int(_progress_val), 100)
        bar_html = f'''
        <div style="background:#e0e0e0;height:24px;border-radius:12px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,#4CAF50,#2196F3);width:{pct}%;height:100%;
                 border-radius:12px;transition:width 0.5s;text-align:center;line-height:24px;color:white;
                 font-size:12px;font-weight:bold;">{pct}%</div>
        </div>'''

        if not _phase_text:
            _phase_text = "**Status:** Idle"

        time_text_val = _get_timer_text()
        log_text = '\n'.join(_log_lines[-30:]) if _log_lines else ""

        return [bar_html, _phase_text, time_text_val, log_text]

    # ==================== Wire Events ====================

    data_source.change(
        fn=toggle_project_select,
        inputs=[data_source],
        outputs=[project_select]
    )

    data_source.change(
        fn=update_source_count,
        inputs=[data_source, project_select],
        outputs=[source_count]
    )

    project_select.change(
        fn=update_source_count,
        inputs=[data_source, project_select],
        outputs=[source_count]
    )

    btn_train.click(
        fn=start_training,
        inputs=[
            data_source, project_select, model_version, exp_name,
            s1_epochs, s2_epochs, batch_size, text_low_lr_rate,
            precision, save_every_epoch, gpu_numbers,
            if_save_every_weights, if_save_latest, if_dpo
        ],
        outputs=[btn_train, phase_text, time_text, log_output]
    )

    # Return components for external wiring
    return {
        "train_tab": train_tab,
        "data_source": data_source,
        "project_select": project_select,
        "source_count": source_count,
        "model_version": model_version,
        "exp_name": exp_name,
        "s1_epochs": s1_epochs,
        "s2_epochs": s2_epochs,
        "batch_size": batch_size,
        "btn_train": btn_train,
        "progress_bar": progress_bar,
        "phase_text": phase_text,
        "time_text": time_text,
        "log_output": log_output,
        "update_progress": update_progress,
        "pipeline": lambda: _pipeline,
    }
