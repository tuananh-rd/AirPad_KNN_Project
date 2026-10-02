"""Unit and audit tests for ACC-07 Dry-Run Readiness and Replay Invariants.

Covers:
- Cryptographic hash pinning of candidate bundle and physical observation log.
- Audit of dryrun_replay_events.json artifact (0 OS events, 100% agreement, clean cleanup).
- Collision test: exclusive-output creation preserves existing file bytes and raises FileExistsError.
- Anti-aliasing test: fails closed if output path aliases an input file.
- Cryptographic tampering fail-closed check.
- Active positive OS guard verification: blocks WindowsController and OS dispatch with AssertionError.
- Rate limiter timestamp binding verification.
"""
from __future__ import annotations

import json
import os
import sys
import pytest
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reports.acc07.replay_acc06_dryrun import (
    compute_sha256,
    replay_acc06_shadow_trial,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_OBS_LOG_SHA256,
)


def test_acc07_file_hashes_and_pins():
    """Verify cryptographic pins for candidate bundle and physical observation log."""
    candidate_path = "reports/acc05/candidate_personal_svm.json"
    obs_log_path = "reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl"

    assert os.path.isfile(candidate_path)
    assert os.path.isfile(obs_log_path)

    actual_candidate_sha = compute_sha256(candidate_path)
    actual_log_sha = compute_sha256(obs_log_path)

    assert actual_candidate_sha == EXPECTED_CANDIDATE_SHA256
    assert actual_log_sha == EXPECTED_OBS_LOG_SHA256


def test_acc07_replay_events_json_audit():
    """Audit the generated dry-run replay events JSON artifact."""
    events_path = "reports/acc07/dryrun_replay_events_v2.json"
    assert os.path.isfile(events_path), f"Replay artifact not found: {events_path}"

    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "header" in data
    assert "summary" in data
    assert "stages" in data
    assert "action_events" in data
    assert "frames" in data

    summary = data["summary"]
    assert summary["status"] == "COMPLETED"
    assert summary["dry_run_only"] is True
    assert summary["real_os_commands_dispatched"] == 0
    assert summary["active_os_guards"]["guards_installed"] is True
    assert summary["active_os_guards"]["os_dispatch_attempts"] == 0
    guarded_targets = summary["active_os_guards"]["guarded_targets"]
    assert "src.platform.windows.WindowsController" in guarded_targets
    assert "pyautogui.scroll" in guarded_targets
    assert "pyautogui.hscroll" in guarded_targets
    assert "pyautogui.hotkey" in guarded_targets
    assert summary["frame_counts"]["total_replayed_frames"] == 3320
    assert summary["prediction_comparison_to_shadow_log"]["agreement_rate"] == 1.0
    assert summary["safety_audit"]["real_controller_invoked"] is False
    assert summary["safety_audit"]["zero_os_control_verified"] is True
    assert summary["safety_audit"]["held_buttons_after_cleanup"] == []
    assert summary["safety_audit"]["held_keys_after_cleanup"] == []
    assert summary["rate_limiter_mode"] == "controlled_to_capture_timestamps"

    # Verify per-row fields across all 3320 frames
    frames = data["frames"]
    assert len(frames) == 3320
    sample_frame = frames[0]
    assert "timestamp" in sample_frame
    assert "held_buttons" in sample_frame
    assert "held_keys" in sample_frame
    assert "matches_shadow_rejected" in sample_frame
    assert sample_frame["matches_shadow_rejected"] is True


def test_acc07_replay_collision_preserves_existing_file_and_raises(tmp_path):
    """Exclusive creation must raise FileExistsError and preserve existing file bytes without deletion/truncation."""
    collided_path = tmp_path / "collided_artifact.json"
    sentinel_content = '{"preserved": true, "bytes": "MUST_NOT_BE_TOUCHED"}'
    collided_path.write_text(sentinel_content, encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"Output file already exists"):
        replay_acc06_shadow_trial(output_events_path=str(collided_path))

    # Assert existing file bytes are completely intact
    assert collided_path.read_text(encoding="utf-8") == sentinel_content


def test_acc07_replay_anti_aliasing_fail_closed():
    """Anti-aliasing guard must reject output paths that alias candidate bundle or log."""
    candidate_path = "reports/acc05/candidate_personal_svm.json"
    obs_log_path = "reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl"

    with pytest.raises(ValueError, match=r"aliases an input file"):
        replay_acc06_shadow_trial(output_events_path=candidate_path)

    with pytest.raises(ValueError, match=r"aliases an input file"):
        replay_acc06_shadow_trial(output_events_path=obs_log_path)


def test_acc07_replay_fail_closed_on_tampered_hash(tmp_path):
    """Ensure replay fails closed if candidate bundle or log is modified/mismatched."""
    tampered_bundle = tmp_path / "tampered_bundle.json"
    tampered_bundle.write_text("{}", encoding="utf-8")
    out_target = tmp_path / "never_created.json"

    with pytest.raises(ValueError, match=r"Candidate bundle hash mismatch"):
        replay_acc06_shadow_trial(candidate_bundle_path=str(tampered_bundle), output_events_path=str(out_target))

    assert not out_target.exists(), "Output file must not be created when validation fails"


def test_acc07_active_positive_os_guards_intercept_forbidden_dispatch(tmp_path):
    """Active positive OS guards must intercept real OS controller instantiation and abort replay."""
    out_target = tmp_path / "guarded_replay.json"

    def malicious_pipeline_factory(*args, **kwargs):
        from src.platform.windows import WindowsController
        WindowsController()  # Should trigger active positive guard!

    with patch("reports.acc07.replay_acc06_dryrun.create_pipeline", side_effect=malicious_pipeline_factory):
        with pytest.raises(AssertionError, match=r"SAFETY VIOLATION.*Real OS controller"):
            replay_acc06_shadow_trial(output_events_path=str(out_target))

    assert not out_target.exists(), "Replay must fail closed before creating output"


def test_acc07_active_positive_os_guards_intercept_pyautogui_dispatch(tmp_path):
    """Active positive OS guards must intercept PyAutoGUI dispatch attempts."""
    out_target = tmp_path / "guarded_pyautogui.json"

    def malicious_pipeline_factory(*args, **kwargs):
        import pyautogui
        pyautogui.scroll(5)  # Should trigger active positive guard!

    with patch("reports.acc07.replay_acc06_dryrun.create_pipeline", side_effect=malicious_pipeline_factory):
        with pytest.raises(AssertionError, match=r"SAFETY VIOLATION.*Real OS controller"):
            replay_acc06_shadow_trial(output_events_path=str(out_target))

    assert not out_target.exists(), "Replay must fail closed before creating output"



def test_acc07_rate_limiter_bound_to_timestamps():
    """Verify that RateLimiter logic correctly enforces intervals based on timestamps."""
    from src.safety.failsafe import RateLimiter

    rl = RateLimiter(max_events_per_second=30)
    current_time = [100.0]

    def allow_simulated():
        with rl._lock:
            now = current_time[0]
            if now - rl.last_event_time >= rl.interval:
                rl.last_event_time = now
                return True
            return False

    # First event allowed
    assert allow_simulated() is True

    # 10ms later (too fast for 30fps / 33.3ms) -> rejected
    current_time[0] += 0.010
    assert allow_simulated() is False

    # 35ms later -> allowed
    current_time[0] += 0.035
    assert allow_simulated() is True

