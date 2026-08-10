#!/usr/bin/env python3
"""
Tests for SoupaWhisper dictate.py
"""

import pytest
import numpy as np
import queue
from unittest.mock import MagicMock, patch
from typing import Any
from types import SimpleNamespace
import threading

# Add 2 second timeout to all tests to prevent infinite loops
pytestmark = pytest.mark.timeout(2)

# Import the modules to test
import sys
import os

# Add the directory containing dictate.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock pynput and streaming audio deps before importing dictate
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
webrtcvad_mock = MagicMock()
vad_instance = MagicMock()
vad_instance.is_speech = MagicMock(return_value=False)
webrtcvad_mock.Vad = MagicMock(return_value=vad_instance)
sys.modules['webrtcvad'] = webrtcvad_mock
streamsad_mock = MagicMock()
streamsad_mock.SAD = MagicMock()
sys.modules['streamsad'] = streamsad_mock

# Now import dictate
import dictate

# Import real Segment and Word from faster_whisper
from faster_whisper.transcribe import Segment, Word


class MockWhisperModel:
    """Mock WhisperModel for testing."""
    def __init__(self, model_name="base.en", device="cpu", compute_type="int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []
        self.transcribe = MagicMock(side_effect=self._transcribe_impl)

    def detect_language(self, audio_path, **kwargs):
        """Stub: Spanish first, then English — used to test language_allowlist."""
        return (
            "es",
            0.9,
            [("es", 0.5), ("en", 0.4), ("ru", 0.05)],
        )
    
    def _transcribe_impl(self, audio_path, **kwargs):
        """Mock transcribe that records calls and returns test data."""
        self.transcribe_calls.append((audio_path, kwargs))
        
        # Return real segments based on audio path or kwargs
        if "word_timestamps" in kwargs and kwargs["word_timestamps"]:
            # Return segments with word timestamps for streaming tests
            words = [
                Word(word="hello", start=0.0, end=0.5, probability=0.9),
                Word(word="world", start=0.6, end=1.0, probability=0.9),
            ]
            segment = Segment(
                id=0, seek=0, start=0.0, end=1.0, text="hello world",
                tokens=[], avg_logprob=0.0, compression_ratio=0.0,
                no_speech_prob=0.0, words=words, temperature=None
            )
            return [segment], {"language": "en"}
        else:
            # Return simple segments for non-streaming tests
            segment = Segment(
                id=0, seek=0, start=0.0, end=1.0, text="test transcription",
                tokens=[], avg_logprob=0.0, compression_ratio=0.0,
                no_speech_prob=0.0, words=None, temperature=None
            )
            return [segment], SimpleNamespace(duration=1.0)


@pytest.fixture
def mock_config(tmp_path, monkeypatch):
    """Create a temporary config file and mock CONFIG_PATH."""
    config_dir = tmp_path / ".config" / "soupawhisper"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.ini"
    
    # Create a test config
    config_content = """[whisper]
model = base.en
device = cpu
compute_type = int8

[hotkey]
key = f10

[behavior]
auto_type = true
notifications = false
default_streaming = false
clipboard = true
tray_icon = false
tray_show_language = true

[streaming]
vad_silence_threshold_seconds = 1.0
vad_sample_rate = 16000
vad_chunk_size_ms = 30
vad_threshold = 0.5
"""
    config_file.write_text(config_content)
    
    # Mock the CONFIG_PATH
    monkeypatch.setattr(dictate, "CONFIG_PATH", config_file)
    return config_file


@pytest.fixture
def mock_whisper_model(monkeypatch):
    """Mock WhisperModel."""
    model = MockWhisperModel()

    def mock_init(model_name, device="cpu", compute_type="int8"):
        model.model_name = model_name
        model.device = device
        model.compute_type = compute_type
        return model

    monkeypatch.setattr(dictate.WhisperModel, "__new__", lambda cls, *args, **kwargs: mock_init(*args, **kwargs))

    return model


@pytest.fixture
def mock_pyaudio_stream(monkeypatch):
    """Mock pyaudio stream."""
    mock_stream = MagicMock()
    mock_stream.read.return_value = b'\x00' * 3200
    mock_stream.stop_stream = MagicMock()
    mock_stream.close = MagicMock()
    return mock_stream


@pytest.fixture
def mock_xdotool(monkeypatch):
    """Mock xdotool."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0)
    
    # Mock subprocess.run for "which" command and other calls
    def mock_run_with_which(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
            result = MagicMock()
            if len(cmd) > 1 and cmd[1] in ["xdotool", "xclip"]:
                result.returncode = 0
            else:
                result.returncode = 1
            return result
        # For other commands (xdotool type, etc.), use the mock
        return mock_run(cmd, **kwargs)
    
    monkeypatch.setattr(dictate.subprocess, "run", mock_run_with_which)
    
    # Store the mock so tests can access it
    mock_run_with_which._mock_run = mock_run
    return mock_run_with_which


class TestTyper:
    """Tests for Typer class."""
    
    def test_typer_init(self, mock_xdotool):
        """Test Typer initialization."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer(delay_ms=20, start_delay_ms=100)
        assert typer.delay_ms == 20
        assert typer.start_delay_ms == 100
        assert typer.enabled is True
    
    def test_typer_type_rewrite_append(self, mock_xdotool):
        """Test typing text (append mode with previous_length=0)."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
            typer.type_rewrite("hello world", 0)
        assert mock_xdotool._mock_run.called

    def test_typer_type_rewrite_incremental(self, mock_xdotool):
        """Test incremental typing using type_rewrite with previous_length=0."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
            # Simulate incremental: calculate suffix and type with previous_length=0
            previous_text = "hello"
            new_text = "hello world"
            suffix = new_text[len(previous_text):]
            typer.type_rewrite(suffix, 0)
        assert mock_xdotool._mock_run.called

    def test_typer_type_rewrite_correction(self, mock_xdotool):
        """Test rewrite typing with character removal."""
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
            typer.type_rewrite("new text", 5)
        assert mock_xdotool._mock_run.called


class TestTyperMacOS:
    """macOS types through Quartz (pynput), so no Apple Events consent is involved."""

    @staticmethod
    def _typer(**kwargs):
        with patch.object(dictate, "IS_MACOS", True):
            return dictate.Typer(delay_ms=1, start_delay_ms=0, **kwargs)

    def test_types_via_pynput_without_shelling_out(self):
        """AppleScript keystroke needs Automation access and stalls on its consent dialog."""
        typer = self._typer()
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            with patch.object(dictate.subprocess, "run") as run:
                typer.type_rewrite("привет", 0)

        typed = "".join(call.args[0] for call in typer._controller.type.call_args_list)
        assert typed == "привет"
        run.assert_not_called()

    def test_deletes_previous_text_with_backspace(self):
        typer = self._typer()
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            typer.type_rewrite("new", 3)
        assert typer._controller.tap.call_count == 3

    def test_zero_delay_types_the_whole_string_in_one_burst(self):
        """A per-character sleep dominates the cost; 0 means hand it all to pynput at once."""
        typer = self._typer()
        typer.delay_ms = 0
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            with patch.object(dictate.time, "sleep") as sleep:
                typer.type_rewrite("hello world", 0)

        typer._controller.type.assert_called_once_with("hello world")
        sleep.assert_not_called()

    def test_positive_delay_still_paces_characters(self):
        """Kept so typing_delay can still be raised to watch the text appear."""
        typer = self._typer()
        typer.delay_ms = 5
        typer._controller.reset_mock()
        with patch.object(dictate, "IS_MACOS", True):
            with patch.object(dictate.time, "sleep") as sleep:
                typer.type_rewrite("abc", 0)

        assert typer._controller.type.call_count == 3
        assert sleep.call_args_list == [((0.005,),)] * 3

    def test_zero_delay_survives_the_constructor(self):
        with patch.object(dictate, "IS_MACOS", True):
            assert dictate.Typer(delay_ms=0).delay_ms == 0

    def test_no_controller_built_on_linux(self, mock_xdotool):
        with patch.object(dictate, "IS_MACOS", False):
            typer = dictate.Typer()
        assert typer._controller is None


class TestStreamingDictation:
    """Tests for StreamingDictation class."""

    def test_streaming_initializes(self, mock_config, mock_whisper_model, mock_xdotool):
        """Basic sanity check that StreamingDictation can be created."""
        config = dictate.load_config()
        config["default_streaming"] = True
        config["auto_type"] = True  # Required for streaming mode
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        assert isinstance(dictation, dictate.StreamingDictation)

    def test_reject_phrases_normalization_ignores_punctuation(self):
        assert dictate._normalize_reject_phrase(" um... ") == "um"
        assert dictate._normalize_reject_phrase("Hmm..") == "hmm"
        assert dictate._normalize_reject_phrase("THANK YOU!!!") == "thank you"

    def test_reject_phrases_disabled_when_empty(self):
        d = dictate.StreamingDictation.__new__(dictate.StreamingDictation)
        d.config = {"reject_phrases": ""}
        d._reject_phrase_set = d._build_reject_phrase_set()
        assert d._reject_phrase_set == frozenset()
        assert d._should_reject_streaming_chunk("thank you") is False

    def test_reject_phrases_exact_whole_chunk_match(self):
        d = dictate.StreamingDictation.__new__(dictate.StreamingDictation)
        d.config = {"reject_phrases": "thank you, um, hmm"}
        d._reject_phrase_set = d._build_reject_phrase_set()

        assert d._should_reject_streaming_chunk("thank you") is True
        assert d._should_reject_streaming_chunk("Thank you!!!") is True
        assert d._should_reject_streaming_chunk("um...") is True
        assert d._should_reject_streaming_chunk("Hmm..") is True

        # Multiple phrases / extra words must NOT be rejected.
        assert d._should_reject_streaming_chunk("thank you thanks") is False
        assert d._should_reject_streaming_chunk("um well") is False


class TestCustomTerms:
    """Tests for custom-terms glossary (initial_prompt + hotwords biasing)."""

    def test_parse_custom_terms_empty(self):
        assert dictate._parse_custom_terms("") == []
        assert dictate._parse_custom_terms("   ") == []
        assert dictate._parse_custom_terms(",,, ,") == []

    def test_parse_custom_terms_comma_separated(self):
        assert dictate._parse_custom_terms("Claude, Kubernetes, GraphQL") == [
            "Claude", "Kubernetes", "GraphQL"
        ]

    def test_parse_custom_terms_newline_separated(self):
        assert dictate._parse_custom_terms("Claude\nKubernetes\nGraphQL") == [
            "Claude", "Kubernetes", "GraphQL"
        ]

    def test_parse_custom_terms_preserves_case_and_multiword(self):
        assert dictate._parse_custom_terms("ML repository, Claude") == [
            "ML repository", "Claude"
        ]

    def test_parse_custom_terms_dedup_preserves_order(self):
        assert dictate._parse_custom_terms("Claude, Kubernetes, Claude, GraphQL") == [
            "Claude", "Kubernetes", "GraphQL"
        ]

    def test_build_custom_terms_kwargs_empty_returns_empty_dict(self):
        assert dictate._build_custom_terms_kwargs([]) == {}

    def test_build_custom_terms_kwargs_populated(self):
        kw = dictate._build_custom_terms_kwargs(["Claude", "ML repository"])
        assert kw == {
            "initial_prompt": "Glossary: Claude, ML repository.",
            "hotwords": "Claude ML repository",
        }

    def test_load_config_custom_terms_default_empty(self, mock_config):
        config = dictate.load_config()
        assert config["custom_terms"] == ""

    def test_load_config_custom_terms_parsed_from_behavior(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "clipboard = true",
                "clipboard = true\ncustom_terms = Claude, Kubernetes, ML repository",
            )
        )
        config = dictate.load_config()
        assert config["custom_terms"] == "Claude, Kubernetes, ML repository"

    def test_dictation_caches_empty_kwargs_when_unset(self, mock_config, mock_whisper_model, mock_xdotool):
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model_loaded.wait(timeout=1.0)
        assert d._custom_terms_kwargs == {}

    def test_dictation_caches_kwargs_when_configured(self, mock_config, mock_whisper_model, mock_xdotool):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "clipboard = true",
                "clipboard = true\ncustom_terms = Claude, Kubernetes",
            )
        )
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model_loaded.wait(timeout=1.0)
        assert d._custom_terms_kwargs == {
            "initial_prompt": "Glossary: Claude, Kubernetes.",
            "hotwords": "Claude Kubernetes",
        }

    def test_custom_terms_passed_to_non_streaming_transcribe(self, mock_config, mock_whisper_model, mock_xdotool):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "clipboard = true",
                "clipboard = true\ncustom_terms = Claude, Kubernetes",
            )
        )
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model = mock_whisper_model
        d.model_error = None
        d.model_loaded.set()

        audio = np.zeros(1600, dtype=np.int16)
        d._transcribe_audio_array(audio)

        _, kwargs = mock_whisper_model.transcribe_calls[-1]
        assert kwargs.get("initial_prompt") == "Glossary: Claude, Kubernetes."
        assert kwargs.get("hotwords") == "Claude Kubernetes"

    def test_custom_terms_absent_when_disabled_non_streaming(self, mock_config, mock_whisper_model, mock_xdotool):
        config = dictate.load_config()
        d = dictate.Dictation(config)
        d.model = mock_whisper_model
        d.model_error = None
        d.model_loaded.set()

        audio = np.zeros(1600, dtype=np.int16)
        d._transcribe_audio_array(audio)

        _, kwargs = mock_whisper_model.transcribe_calls[-1]
        assert "initial_prompt" not in kwargs
        assert "hotwords" not in kwargs


class TestDictation:
    """Tests for non-streaming Dictation class (backward compatibility)."""
    
    @patch('dictate.pyaudio.PyAudio')
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_non_streaming_mode(self, mock_popen, mock_run, mock_pyaudio, mock_config, mock_whisper_model):
        """Test non-streaming mode (backward compatibility)."""
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_stream.read.return_value = b'\x00' * 3200
        mock_audio_stream.stop_stream = MagicMock()
        mock_audio_stream.close = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        # Ensure non-streaming mode
        config = dictate.load_config()
        config["default_streaming"] = False
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Start recording
        dictation.start_recording()
        assert dictation.recording is True
        
        # Stop recording
        dictation.stop_recording()
        assert dictation.recording is False
    
    def test_config_loading(self, mock_config):
        """Test configuration loading."""
        # Reload config to test loading
        config = dictate.load_config()
        assert "model" in config
        assert "default_streaming" in config
        assert "clipboard" in config
        assert config["model"] == "base.en"
        assert isinstance(config["default_streaming"], bool)
        assert config["clipboard"] is True

    def test_language_defaults_to_en(self, mock_config):
        config = dictate.load_config()
        assert config["language"] == "en"

    def test_language_auto_maps_to_none(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(content.replace("compute_type = int8", "compute_type = int8\nlanguage = auto"))
        config = dictate.load_config()
        assert config["language"] is None

    def test_language_allowlist_parsed(self, mock_config):
        content = mock_config.read_text()
        mock_config.write_text(
            content.replace(
                "compute_type = int8",
                "compute_type = int8\nlanguage = auto\nlanguage_allowlist = en, ru",
            )
        )
        config = dictate.load_config()
        assert config["language"] is None
        assert config["language_allowlist"] == ["en", "ru"]

    def test_resolve_transcription_language_allowlist_picks_best_of_two(self):
        model = MockWhisperModel()
        audio = np.zeros(1600, dtype=np.float32)
        # Whisper "top" is es, but allowlist is en,ru -> should pick ru (0.05) vs ... wait
        # filtered: en 0.4, ru 0.05 -> max is en
        assert (
            dictate.resolve_transcription_language(
                model, audio, None, ["en", "ru"]
            )
            == "en"
        )

    def test_resolve_transcription_language_fixed_skips_allowlist(self):
        model = MockWhisperModel()
        audio = np.zeros(1600, dtype=np.float32)
        assert (
            dictate.resolve_transcription_language(
                model, audio, "ru", ["en", "ru"]
            )
            == "ru"
        )

    def test_resolve_transcription_language_single_allowlist_no_detect_call(self):
        """One candidate needs no detect_language (same as fixed language)."""
        model = MagicMock()
        model.detect_language = MagicMock()
        audio = np.zeros(1600, dtype=np.float32)
        assert dictate.resolve_transcription_language(model, audio, None, ["ru"]) == "ru"
        model.detect_language.assert_not_called()

    def test_layout_language_map_parsing(self):
        parsed = dictate.Dictation._parse_layout_to_language_map(
            "com.apple.keylayout.US:en, com.apple.keylayout.Russian:ru, xkb:de:de"
        )
        assert parsed["com.apple.keylayout.US"] == "en"
        assert parsed["com.apple.keylayout.Russian"] == "ru"
        assert parsed["xkb"] == "de:de"

    def test_language_from_layout_direct_and_heuristic(self):
        m = {"com.apple.keylayout.US": "en"}
        assert dictate.language_from_layout("com.apple.keylayout.US", m) == "en"
        assert dictate.language_from_layout("com.apple.keylayout.Russian", {}) is None

    def test_linux_active_xkb_layout_uses_group_index(self, monkeypatch):
        monkeypatch.setattr(dictate, "_which_ok", lambda name: False)
        monkeypatch.setattr(
            dictate,
            "_linux_xkb_layouts_from_setxkbmap",
            lambda: ["us", "ru", "am"],
        )
        monkeypatch.setattr(dictate, "_linux_xkb_group_index", lambda: 1)
        assert dictate._linux_active_xkb_layout() == "ru"

    def test_parse_setxkbmap_layouts(self):
        assert dictate.parse_setxkbmap_layouts("rules: evdev\nlayout: us,ru,am\n") == [
            "us",
            "ru",
            "am",
        ]
        assert dictate.parse_setxkbmap_layouts("layout: ru\n") == ["ru"]
        assert dictate.parse_setxkbmap_layouts("") is None
        assert dictate.parse_setxkbmap_layouts("model: pc104\n") is None

    def test_linux_active_prefers_xkb_switch(self, monkeypatch):
        monkeypatch.setattr(
            dictate,
            "_which_ok",
            lambda name: name == "xkb-switch",
        )
        monkeypatch.setattr(dictate, "_run_cmd", lambda cmd, timeout_s=0.5: "ru")
        monkeypatch.setattr(
            dictate,
            "_linux_xkb_layouts_from_setxkbmap",
            lambda: (_ for _ in ()).throw(AssertionError("should not fall back")),
        )
        assert dictate._linux_active_xkb_layout() == "ru"

    def test_detect_current_keyboard_language_linux_maps_us(self, monkeypatch):
        monkeypatch.setattr(dictate, "IS_MACOS", False)
        monkeypatch.setattr(dictate, "detect_current_keyboard_layout", lambda: "us,ru,am")
        monkeypatch.setattr(dictate, "_linux_active_xkb_layout", lambda: "us")
        assert dictate.detect_current_keyboard_language({"us": "en", "ru": "ru"}) == "en"

    def test_capture_session_language_uses_active_layout_not_layout_list(
        self, mock_config, monkeypatch
    ):
        """setxkbmap layout_id is often 'us,ru,am'; capture must use active group."""
        content = mock_config.read_text()
        content = content.replace(
            "compute_type = int8",
            "compute_type = int8\nlanguage = auto\nlanguage_allowlist = en, ru",
        )
        content = content.replace(
            "clipboard = true",
            "clipboard = true\n"
            "enforce_language_from_layout = true\n"
            "layout_to_language = us:en, ru:ru",
        )
        mock_config.write_text(content)
        config = dictate.load_config()
        monkeypatch.setattr(dictate, "IS_MACOS", False)
        monkeypatch.setattr(dictate, "detect_current_keyboard_layout", lambda: "us,ru,am")
        monkeypatch.setattr(dictate, "_linux_active_xkb_layout", lambda: "ru")
        d = dictate.Dictation(config)
        d._begin_session_language()
        assert d._session_enforced_language == "ru"
        d._end_session_language()
        assert d._session_enforced_language is None

    def test_stop_recording_clears_session_language(
        self, mock_config, mock_whisper_model, mock_xdotool
    ):
        config = dictate.load_config()
        config["enforce_language_from_layout"] = True
        d = dictate.Dictation(config)
        d.model_loaded.set()
        d.recording = True
        d.audio_data = []
        d.audio_thread = None
        d.audio_stream = None
        d._session_enforced_language = "en"
        with patch.object(d, "_report_audio_problem"):
            d.stop_recording()
        assert d._session_enforced_language is None
        assert d.transcribing is False

    def test_enforce_language_from_layout_overrides_auto(self, mock_config, monkeypatch):
        # Set auto language + allowlist.
        content = mock_config.read_text()
        content = content.replace(
            "compute_type = int8",
            "compute_type = int8\nlanguage = auto\nlanguage_allowlist = en, ru",
        )
        # Inject behavior keys into the existing [behavior] section.
        content = content.replace(
            "clipboard = true",
            "clipboard = true\n"
            "enforce_language_from_layout = true\n"
            "layout_to_language = com.apple.keylayout.Russian:ru",
        )
        mock_config.write_text(content)
        config = dictate.load_config()
        # Pretend we're on macOS to avoid xdotool usage.
        monkeypatch.setattr(dictate, "IS_MACOS", True)
        # Force detector to return Russian layout.
        monkeypatch.setattr(dictate, "detect_current_keyboard_layout", lambda: "com.apple.keylayout.Russian")
        # Avoid depending on real HIToolbox parsing in tests.
        monkeypatch.setattr(dictate, "_macos_input_source_languages_for_id", lambda _id: ["ru"])

        d = dictate.Dictation(config)
        # Don't wait for real model thread; stub model directly.
        model = MockWhisperModel()
        d.model = model
        d.model_error = None
        d.model_loaded.set()

        # Mimic "hotkey pressed" behavior (capture once at session start).
        d._capture_session_enforced_language()

        audio = np.zeros(1600, dtype=np.int16)
        d._transcribe_audio_array(audio)
        # Ensure transcribe was called with language="ru".
        _, kwargs = model.transcribe_calls[-1]
        assert kwargs.get("language") == "ru"

    def test_detect_current_keyboard_language_macos_uses_input_source_languages(self, monkeypatch):
        monkeypatch.setattr(dictate, "IS_MACOS", True)
        # Match the real-world structure: AppleCurrentKeyboardLayoutInputSourceID exists, while
        # AppleSelectedInputSources entries may not contain InputSourceID or InputSourceLanguages.
        monkeypatch.setattr(
            dictate,
            "_macos_hitoolbox_plist",
            lambda: {
                "AppleCurrentKeyboardLayoutInputSourceID": "com.apple.keylayout.ABC",
                "AppleSelectedInputSources": [
                    {"Bundle ID": "com.apple.PressAndHold", "InputSourceKind": "Non Keyboard Input Method"},
                    {"InputSourceKind": "Keyboard Layout", "KeyboardLayout ID": 252, "KeyboardLayout Name": "ABC"},
                ],
            },
        )
        # We still expect language detection to use _macos_input_source_languages_for_id fallback
        # when no explicit mapping exists.
        monkeypatch.setattr(dictate, "_macos_input_source_languages_for_id", lambda _id: ["en"])
        assert dictate.detect_current_keyboard_language({}) == "en"

    def test_detect_current_keyboard_layout_macos_uses_current_layout_id(self, monkeypatch):
        monkeypatch.setattr(dictate, "IS_MACOS", True)
        monkeypatch.setattr(
            dictate,
            "_macos_hitoolbox_plist",
            lambda: {
                "AppleCurrentKeyboardLayoutInputSourceID": "com.apple.keylayout.ABC",
                "AppleSelectedInputSources": [
                    {"Bundle ID": "com.apple.PressAndHold", "InputSourceKind": "Non Keyboard Input Method"},
                    {"InputSourceKind": "Keyboard Layout", "KeyboardLayout ID": 252, "KeyboardLayout Name": "ABC"},
                ],
            },
        )
        assert dictate.detect_current_keyboard_layout() == "com.apple.keylayout.ABC"

    def test_config_clipboard_disabled(self, mock_config):
        """Test that clipboard=false is correctly loaded."""
        # Replace existing clipboard = true with clipboard = false
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        
        config = dictate.load_config()
        assert config["clipboard"] is False

class TestClipboardIntegration:
    """Tests specifically for the clipboard parameter integration."""

    @patch('dictate.pyaudio.PyAudio')
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_dictation_no_clipboard_call(self, mock_popen, mock_run, mock_pyaudio, mock_config, mock_whisper_model):
        """Test that Dictation doesn't call xclip when clipboard is disabled."""
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_stream.read.return_value = b'\x00' * 3200
        mock_audio_stream.stop_stream = MagicMock()
        mock_audio_stream.close = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Mock audio data
        dictation.audio_data = [np.array([0] * 1600, dtype=np.int16)]
        dictation.recording = True
        
        # Mock model return
        segment = Segment(
            id=0, seek=0, start=0.0, end=1.0, text="test text",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=None, temperature=None
        )
        model: Any = dictation.model
        model.transcribe.side_effect = None
        model.transcribe.return_value = ([segment], {})
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    @patch('dictate.pyaudio.PyAudio')
    def test_streaming_dictation_no_clipboard_call(self, mock_pyaudio, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test that StreamingDictation doesn't call xclip when clipboard is disabled."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        config["auto_type"] = True  # Required for streaming mode
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Clear queues to prevent timeout
        while not dictation.transcription_queue.empty():
            try:
                dictation.transcription_queue.get_nowait()
            except queue.Empty:
                break
        while not dictation.typing_queue.empty():
            try:
                dictation.typing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Set up state for stop_recording
        dictation.recording = True
        dictation.accumulated_text = "final text"
        dictation.audio_stream = mock_audio_stream
        dictation.audio_interface = mock_audio_instance
        dictation.audio_thread = None  # No thread to join
        dictation.transcription_thread = None
        dictation.typing_thread = None
        dictation.file_saving_thread = None
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    def test_check_dependencies_clipboard_optional(self, monkeypatch):
        """Test that xclip is optional in check_dependencies if clipboard is disabled."""
        # Force Linux path so we test the xclip/xdotool dependency checks.
        monkeypatch.setattr(dictate, "IS_MACOS", False)
        mock_run = MagicMock()

        # Mock 'which' to return 1 for xclip (missing)
        def side_effect(cmd, **kwargs):
            res = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "xclip":
                res.returncode = 1
            else:
                res.returncode = 0
            return res

        mock_run.side_effect = side_effect
        monkeypatch.setattr(dictate.subprocess, "run", mock_run)

        # Should NOT exit if clipboard is False (webrtcvad and pyaudio are already imported, so import check passes)
        dictate.check_dependencies({"clipboard": False, "auto_type": False, "default_streaming": False})

        # Should exit if clipboard is True (since xclip is missing)
        with pytest.raises(SystemExit):
            dictate.check_dependencies({"clipboard": True, "auto_type": False, "default_streaming": False})


class TestTrayStatus:
    def test_load_config_tray_defaults(self, mock_config):
        config = dictate.load_config()
        assert config["tray_icon"] is False  # mock_config sets false
        assert config["tray_show_language"] is True

    def test_load_config_tray_defaults_when_unset(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.ini"
        config_file.write_text("[whisper]\nmodel = base.en\n[behavior]\nnotifications = false\n")
        monkeypatch.setattr(dictate, "CONFIG_PATH", config_file)
        config = dictate.load_config()
        assert config["tray_icon"] is True
        assert config["tray_show_language"] is True

    def test_derive_state_priority(self):
        import tray_status

        d = SimpleNamespace(
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            recording=False,
            stopping=False,
            transcribing=False,
        )
        assert tray_status.derive_state(d) == "loading"
        d.model_loaded.set()
        assert tray_status.derive_state(d) == "idle"
        d.recording = True
        assert tray_status.derive_state(d) == "recording"
        d.recording = False
        d.stopping = True
        assert tray_status.derive_state(d) == "transcribing"
        d.stopping = False
        d.transcribing = True
        assert tray_status.derive_state(d) == "transcribing"
        d.transcribing = False
        d.model_error = "boom"
        assert tray_status.derive_state(d) == "error"

    def test_language_label_precedence(self):
        import tray_status

        d = SimpleNamespace(
            _session_enforced_language="ru",
            config={"language": "en"},
        )
        assert tray_status.language_label(d) == ("RU", True)
        d._session_enforced_language = None
        assert tray_status.language_label(d) == ("EN", False)
        d.config["language"] = None
        assert tray_status.language_label(d) == ("AUTO", False)

    def test_tooltip_idle_includes_hotkey(self):
        import tray_status

        d = SimpleNamespace(
            config={"model": "base.en"},
            get_hotkey_name=lambda: "f12",
            model_error=None,
            _tray_error=None,
        )
        tip = tray_status.tooltip_for(d, "idle")
        assert "F12" in tip
        assert "idle" in tip.lower()
        # AppIndicator titles are latin-1
        tip.encode("latin-1")

    def test_compose_language_badge_draws_pixels(self):
        import tray_status
        from PIL import Image

        base = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        out = tray_status.compose_language_badge(base, "AUTO")
        assert out.size == (64, 64)
        # Badge should paint some opaque pixels where the chip is
        assert max(px[3] for px in out.getdata()) > 200

    def test_latin1_safe_replaces_non_latin1(self):
        import tray_status

        assert tray_status.latin1_safe("SoupaWhisper - idle") == "SoupaWhisper - idle"
        out = tray_status.latin1_safe("ok — dash")
        out.encode("latin-1")
        assert "—" not in out

    def test_tray_title_includes_language(self):
        import tray_status

        d = SimpleNamespace(
            config={"model": "base", "language": None, "tray_show_language": True},
            get_hotkey_name=lambda: "f12",
            model_error=None,
            _tray_error=None,
            _session_enforced_language="en",
        )
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        assert "[EN]" in tray._title_for("recording")
        d.config["tray_show_language"] = False
        assert "[EN]" not in tray._title_for("recording")

    def test_badge_hidden_when_tray_show_language_false(self):
        import tray_status

        d = SimpleNamespace(
            config={"tray_show_language": False, "language": None},
            _session_enforced_language="ru",
        )
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        assert tray._language_badge_text() == ""

    def test_refresh_skips_update_menu_when_signature_unchanged(self):
        import tray_status

        d = SimpleNamespace(
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            recording=False,
            stopping=False,
            transcribing=False,
            config={"model": "base", "language": None, "tray_show_language": True},
            get_hotkey_name=lambda: "f12",
            _session_enforced_language=None,
        )
        d.model_loaded.set()
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        tray._images = {name: MagicMock() for name in tray_status.STATE_NAMES}
        tray._last_state = "idle"
        tray._last_title = tray._title_for("idle")
        tray._last_badge = "AUTO"
        tray._last_menu_signature = tray._menu_signature()
        tray._icon = None
        icon = MagicMock()
        with patch.object(tray, "_image_for", return_value=MagicMock()):
            tray._refresh(icon)
        icon.update_menu.assert_not_called()

    def test_compose_language_badge_short_labels_match_auto_width(self):
        """EN/RU use the same centered chip as AUTO (same left/right edges)."""
        import tray_status
        from PIL import Image

        base = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        auto = tray_status.compose_language_badge(base, "AUTO")
        en = tray_status.compose_language_badge(base, "EN")

        def chip_bounds(img: Image.Image) -> tuple[int, int]:
            px = img.load()
            left, right = img.width, -1
            for x in range(img.width):
                for y in range(img.height):
                    if px[x, y][3] > 200 and px[x, y][0] < 40:
                        left = min(left, x)
                        right = max(right, x)
            return left, right

        assert chip_bounds(auto) == chip_bounds(en)
        left, right = chip_bounds(auto)
        # Horizontally centered; vertically at the bottom.
        assert abs((left + right) / 2 - 31.5) < 3
        px = auto.load()
        bottom_has_chip = any(
            px[x, auto.height - 1][3] > 200 and px[x, auto.height - 1][0] < 40
            for x in range(auto.width)
        )
        assert bottom_has_chip

    def test_clear_session_language_resets_tray_to_auto(self):
        import tray_status

        d = SimpleNamespace(
            _session_enforced_language="ru",
            config={"language": None},
        )
        assert tray_status.language_label(d) == ("RU", True)
        d._session_enforced_language = None
        assert tray_status.language_label(d) == ("AUTO", False)

    def test_toggle_menu_text_and_action_schedules_immediately(self):
        import tray_status

        calls = []
        d = SimpleNamespace(
            recording=False,
            stopping=False,
            transcribing=False,
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            config={"model": "base.en", "tray_show_language": False},
            start_recording=lambda: calls.append("start"),
            stop_recording=lambda: calls.append("stop"),
            _schedule_hotkey_action=lambda fn, name: calls.append((name, fn)),
        )
        d.model_loaded.set()
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        tray._icon = None
        tray._last_state = None
        tray._last_title = None
        tray._images = {}
        tray._images_16 = {}

        assert tray._toggle_menu_text() == "Start dictation"
        assert tray._toggle_enabled() is True
        tray._on_toggle_dictation(MagicMock(), None)
        assert calls[0][0] == "start_recording"

        d.recording = True
        calls.clear()
        assert tray._toggle_menu_text() == "Stop dictation"
        tray._on_toggle_dictation(MagicMock(), None)
        assert calls[0][0] == "stop_recording"

    def test_missing_assets_raises(self, tmp_path):
        import tray_status

        with pytest.raises(tray_status.TrayStartError):
            tray_status.load_state_images(tmp_path)

    def test_run_with_tray_false_skips_tray(self, mock_config, mock_whisper_model, mock_xdotool):
        config = dictate.load_config()
        config["tray_icon"] = False
        d = dictate.Dictation(config)
        d.model_loaded.set()
        calls = []

        def fake_loop():
            calls.append("supervisor")
            d.running = False

        with patch.object(d, "_run_supervisor_loop", side_effect=fake_loop):
            with patch.dict(sys.modules, {"tray_status": MagicMock()}):
                d.run()
        assert calls == ["supervisor"]

    def test_run_with_tray_true_exits_on_prepare_failure(
        self, mock_config, mock_whisper_model, mock_xdotool
    ):
        import tray_status

        config = dictate.load_config()
        config["tray_icon"] = True
        d = dictate.Dictation(config)
        d.model_loaded.set()

        with patch.object(tray_status.TrayStatus, "prepare", side_effect=tray_status.TrayStartError("no host")):
            with pytest.raises(SystemExit) as ei:
                d.run()
            assert ei.value.code == 1

    def test_compose_empty_badge_returns_base(self):
        import tray_status
        from PIL import Image

        base = Image.new("RGBA", (64, 64), (1, 2, 3, 255))
        assert tray_status.compose_language_badge(base, "") is base

    def test_prepare_builds_icon_with_mocked_pystray(self, tmp_path):
        import tray_status
        from PIL import Image

        assets = tmp_path / "tray"
        assets.mkdir()
        for name in tray_status.STATE_NAMES:
            Image.new("RGBA", (64, 64), (0, 0, 0, 255)).save(assets / f"{name}.png")

        d = SimpleNamespace(
            model_error=None,
            _tray_error=None,
            model_loaded=threading.Event(),
            recording=False,
            stopping=False,
            transcribing=False,
            config={"model": "base", "language": None, "tray_show_language": True},
            get_hotkey_name=lambda: "f12",
            _session_enforced_language=None,
            running=True,
        )
        d.model_loaded.set()

        fake_icon = MagicMock()
        fake_pystray = MagicMock()
        fake_pystray.Icon.return_value = fake_icon
        fake_menu = MagicMock()
        fake_menu.SEPARATOR = object()
        fake_pystray.Menu = fake_menu
        fake_pystray.MenuItem = MagicMock()

        with patch.dict(sys.modules, {"pystray": fake_pystray}):
            # Re-import Menu inside prepare from pystray — inject via import
            import types

            mod = types.ModuleType("pystray")
            mod.Icon = fake_pystray.Icon
            mod.Menu = fake_menu
            mod.MenuItem = MagicMock(return_value=MagicMock())
            with patch.dict(sys.modules, {"pystray": mod}):
                tray = tray_status.TrayStatus(d, assets_dir=assets)
                tray.prepare()
        assert tray._icon is fake_icon
        mod.Icon.assert_called_once()

    def test_language_menu_text_from_layout(self):
        import tray_status

        d = SimpleNamespace(
            _session_enforced_language="ru",
            config={"language": None},
        )
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        assert tray._language_menu_text() == "Language: RU (from layout)"
        d._session_enforced_language = None
        assert tray._language_menu_text() == "Language: AUTO"


class TestMacOSMenuBarImage:
    """macOS draws its own pixmap: pystray squashes any image to the bar thickness."""

    @staticmethod
    def _mic() -> "Image.Image":
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((22, 6, 42, 40), radius=10, fill=(0, 0, 0, 255))
        draw.rectangle((30, 40, 34, 56), fill=(0, 0, 0, 255))
        return img

    @staticmethod
    def _rows_with_ink(img) -> list[int]:
        px = img.load()
        return [
            y
            for y in range(img.height)
            if any(px[x, y][3] > 40 for x in range(img.width))
        ]

    def test_rendered_at_retina_scale(self):
        import tray_status

        img = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        assert img.height == 44

    def test_glyph_does_not_touch_bar_edges(self):
        """22pt edge-to-edge is why the icon reads as oversized; leave breathing room."""
        import tray_status

        img = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        rows = self._rows_with_ink(img)
        assert rows, "image is empty"
        assert min(rows) >= 2
        assert max(rows) <= img.height - 3

    def test_badge_sits_below_the_glyph(self):
        import tray_status
        from PIL import Image

        # A solid block fills its glyph box exactly, so any ink lower down is the chip.
        block = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
        img = tray_status.macos_menu_bar_image(block, "AUTO", "idle", thickness=22, scale=2)
        pad = round(44 * tray_status.MACOS_EDGE_PADDING_RATIO)
        glyph_bottom = pad + round(44 * tray_status.MACOS_GLYPH_RATIO) - 1
        assert max(self._rows_with_ink(img)) > glyph_bottom

    def test_short_labels_stay_centered(self):
        import tray_status

        auto = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        en = tray_status.macos_menu_bar_image(self._mic(), "EN", "idle", thickness=22, scale=2)
        assert auto.size == en.size

    def test_template_states_render_opaque_mask(self):
        """setTemplate_ keeps only alpha, so badge text must be ink, not a knocked-out chip."""
        import tray_status

        img = tray_status.macos_menu_bar_image(self._mic(), "AUTO", "idle", thickness=22, scale=2)
        badge_band = img.crop((0, img.height // 2, img.width, img.height))
        assert max(px[3] for px in badge_band.getdata()) > 200

    @pytest.mark.parametrize("thickness,scale", [(22, 2), (24, 2), (37, 2), (22, 1), (22, 3)])
    def test_layout_scales_with_the_measured_bar(self, thickness, scale):
        """Bar thickness is not a constant across macOS releases and displays."""
        import tray_status

        img = tray_status.macos_menu_bar_image(
            self._mic(), "AUTO", "idle", thickness=thickness, scale=scale
        )
        height = round(thickness * scale)
        assert img.height == height

        rows = self._rows_with_ink(img)
        pad = max(1, round(height * tray_status.MACOS_EDGE_PADDING_RATIO))
        assert min(rows) >= pad, "glyph must not touch the top of the bar"
        assert max(rows) <= height - pad - 1, "chip must not touch the bottom of the bar"

        # The glyph keeps its share of the bar instead of a fixed point size.
        glyph_px = round(height * tray_status.MACOS_GLYPH_RATIO)
        assert max(self._rows_with_ink(img)) > pad + glyph_px - 1, "chip sits below the glyph"

    def test_colored_state_badge_uses_the_state_colour(self):
        """White-on-outline vanishes into a light menu bar; the chip tracks the glyph."""
        import tray_status

        img = tray_status.macos_menu_bar_image(
            self._mic(), "AUTO", "recording", thickness=22, scale=2
        )
        band = img.crop((0, img.height // 2, img.width, img.height))
        colours = {px[:3] for px in band.getdata() if px[3] > 200}
        assert tray_status.MACOS_BADGE_COLORS["recording"][:3] in colours

    def test_macos_install_image_hides_button_title(self):
        """The language chip lives in the pixmap; no second copy beside the icon."""
        import tray_status

        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = SimpleNamespace(config={"tray_show_language": True})
        tray._images = {name: self._mic() for name in tray_status.STATE_NAMES}
        button = MagicMock()
        status_item = MagicMock()
        status_item.button.return_value = button
        tray._icon = SimpleNamespace(_status_item=status_item)

        # AppKit only exists on macOS; the test runs everywhere, so stub the bridge.
        with patch.object(tray_status, "IS_MACOS", True):
            with patch.object(tray_status, "_ns_image_from", return_value="NSImage") as make:
                tray._install_macos_image("idle", "AUTO")

        make.assert_called_once()
        assert make.call_args.kwargs["template"] is True
        button.setImage_.assert_called_once_with("NSImage")
        button.setTitle_.assert_called_once_with("")

    def test_macos_colored_states_are_not_template_images(self):
        """Recording/error carry meaning in their colour; a template mask would erase it."""
        import tray_status

        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = SimpleNamespace(config={"tray_show_language": True})
        tray._images = {name: self._mic() for name in tray_status.STATE_NAMES}
        status_item = MagicMock()
        tray._icon = SimpleNamespace(_status_item=status_item)

        with patch.object(tray_status, "IS_MACOS", True):
            with patch.object(tray_status, "_ns_image_from", return_value="NSImage") as make:
                tray._install_macos_image("recording", "EN")

        assert make.call_args.kwargs["template"] is False

    def test_apply_language_badge_sets_no_title_on_macos(self):
        import tray_status

        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        button = MagicMock()
        status_item = MagicMock()
        status_item.button.return_value = button
        icon = MagicMock()
        icon._appindicator = None
        icon._status_item = status_item
        tray._icon = icon

        with patch.object(tray_status, "IS_MACOS", True):
            tray._apply_language_badge("AUTO")

        button.setTitle_.assert_not_called()

    def test_setup_installs_macos_image_before_polling(self):
        """Template mode was only applied on a state change, so the first icon was raw."""
        import tray_status

        d = SimpleNamespace(running=False)
        tray = tray_status.TrayStatus.__new__(tray_status.TrayStatus)
        tray.dictation = d
        tray._stop_poll = threading.Event()
        tray._stop_poll.set()
        tray._last_state = "idle"
        tray._last_badge = "AUTO"
        icon = MagicMock()

        with patch.object(tray_status, "IS_MACOS", True):
            with patch.object(tray, "_install_macos_image") as install:
                tray._setup(icon)

        install.assert_called_once_with("idle", "AUTO")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
