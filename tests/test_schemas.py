"""Tier 1 & Tier 2 tests for Pydantic data contracts (Phonetics and Exercise schemas).

Covers:
- FEAT-SCH-01: PhonemeEvaluation (score range, status, articulation tip)
- FEAT-SCH-02: WordEvaluation (word token, constituents, time bounds)
- FEAT-SCH-03: SpeechAssessmentResult (holistic score, word list, latency)
- FEAT-SCH-04: ExerciseItem (drill types, prompt, answer options, language)
- FEAT-SCH-05: ExerciseSession (drill sequence, progression, score aggregation)
- Boundary & validation errors: out-of-range scores, empty tokens, invalid choices
"""

import pytest

# Progressive testability check
try:
    import lang_attack.schemas.exercise as ex_mod
    import lang_attack.schemas.phonetics as ph_mod

    HAS_SCHEMAS = hasattr(ph_mod, "PhonemeEvaluation") and hasattr(ex_mod, "ExerciseItem")
except ImportError:
    HAS_SCHEMAS = False

pytestmark = pytest.mark.skipif(
    not HAS_SCHEMAS,
    reason="lang_attack.schemas modules not yet implemented (scheduled for M2)",
)


# ============================================================================
# Tier 1: Phonetics Schemas Happy Path
# ============================================================================


def test_phoneme_evaluation_valid() -> None:
    """Verify PhonemeEvaluation instantiation with valid attributes."""
    from lang_attack.schemas.phonetics import PhonemeEvaluation

    # Allow either 'score' or 'gop_score'
    kwargs = {
        "phoneme": "k",
        "status": "correct",
        "articulation_tip": "Back of tongue against velum",
    }
    if "ipa_symbol" in PhonemeEvaluation.model_fields:
        kwargs["ipa_symbol"] = "k"
    if "expected_phoneme" in PhonemeEvaluation.model_fields:
        kwargs["expected_phoneme"] = "k"
        kwargs["detected_phoneme"] = "k"
        kwargs["is_correct"] = True
        kwargs["start_time"] = 0.0
        kwargs["end_time"] = 0.15

    score_field = "score" if "score" in PhonemeEvaluation.model_fields else "gop_score"
    kwargs[score_field] = 92.5

    p = PhonemeEvaluation(**kwargs)
    assert p.phoneme == "k"
    assert getattr(p, score_field) == 92.5


def test_word_evaluation_valid() -> None:
    """Verify WordEvaluation aggregates constituent phonemes."""
    from lang_attack.schemas.phonetics import PhonemeEvaluation, WordEvaluation

    score_field = "score" if "score" in PhonemeEvaluation.model_fields else "gop_score"
    p_kwargs = {"phoneme": "k", "status": "correct", score_field: 90.0}
    if "ipa_symbol" in PhonemeEvaluation.model_fields:
        p_kwargs["ipa_symbol"] = "k"
        p_kwargs["expected_phoneme"] = "k"
        p_kwargs["detected_phoneme"] = "k"
        p_kwargs["is_correct"] = True
        p_kwargs["start_time"] = 0.0
        p_kwargs["end_time"] = 0.1

    p1 = PhonemeEvaluation(**p_kwargs)

    w_score_field = "score" if "score" in WordEvaluation.model_fields else "gop_score"
    w_kwargs = {
        "word": "cat",
        w_score_field: 88.0,
        "phonemes": [p1],
        "start_time": 0.0,
        "end_time": 0.45,
    }
    if "is_correct" in WordEvaluation.model_fields:
        w_kwargs["is_correct"] = True

    w = WordEvaluation(**w_kwargs)
    assert w.word == "cat"
    assert len(w.phonemes) == 1
    assert getattr(w, w_score_field) == 88.0


def test_speech_assessment_result_valid() -> None:
    """Verify SpeechAssessmentResult holistic scores and word breakdown."""
    from lang_attack.schemas.phonetics import SpeechAssessmentResult

    kwargs = {
        "overall_score": 85.0,
        "words": [],
    }
    # Check text fields
    if "target_text" in SpeechAssessmentResult.model_fields:
        kwargs["target_text"] = "Hello world"
        kwargs["recognized_text"] = "Hello world"
    else:
        kwargs["reference_text"] = "Hello world"
        kwargs["transcript"] = "Hello world"
        kwargs["fluency_score"] = 90.0
        kwargs["completeness_score"] = 100.0
        kwargs["audio_duration_s"] = 1.5

    res = SpeechAssessmentResult(**kwargs)
    assert res.overall_score == 85.0


def test_exercise_item_multiple_choice() -> None:
    """Verify ExerciseItem multiple choice configuration."""
    from lang_attack.schemas.exercise import ExerciseItem

    ex = ExerciseItem(
        id="ex-101",
        type="multiple_choice",
        prompt="Select the correct translation for 'book':",
        options=["livre", "maison", "chat", "porte"],
        correct_answer="livre",
        explanation="'Livre' is French for 'book'.",
        language="fr",
    )
    assert ex.type in ["multiple_choice", "MULTIPLE_CHOICE"]
    assert ex.correct_answer == "livre"
    assert "livre" in (ex.options or [])


def test_exercise_session_lifecycle() -> None:
    """Verify ExerciseSession contains items and tracks progression index."""
    from lang_attack.schemas.exercise import ExerciseItem, ExerciseSession

    item = ExerciseItem(
        id="ex-1",
        type="multiple_choice",
        prompt="Question 1",
        options=["A", "B"],
        correct_answer="A",
        language="en",
    )
    session = ExerciseSession(
        id="session-001" if "id" in ExerciseSession.model_fields else None,
        session_id="session-001" if "session_id" in ExerciseSession.model_fields else None,
        title="Basic English",
        language="en",
        items=[item],
        current_index=0,
        score=0.0,
    )
    assert len(session.items) == 1
    assert session.current_index == 0
    assert session.language == "en"


# ============================================================================
# Tier 2: Boundary & Validation Tests
# ============================================================================


def test_phoneme_score_out_of_bounds() -> None:
    """Boundary: score cannot be negative or exceed 100.0."""
    from lang_attack.schemas.phonetics import PhonemeEvaluation
    from pydantic import ValidationError

    score_field = "score" if "score" in PhonemeEvaluation.model_fields else "gop_score"
    kwargs = {"phoneme": "p", "status": "correct"}
    if "ipa_symbol" in PhonemeEvaluation.model_fields:
        kwargs["ipa_symbol"] = "p"
        kwargs["expected_phoneme"] = "p"
        kwargs["detected_phoneme"] = "p"
        kwargs["is_correct"] = True
        kwargs["start_time"] = 0.0
        kwargs["end_time"] = 0.1

    # Test negative
    kwargs[score_field] = -1.0
    with pytest.raises(ValidationError):
        PhonemeEvaluation(**kwargs)

    # Test > 100.0
    kwargs[score_field] = 101.5
    with pytest.raises(ValidationError):
        PhonemeEvaluation(**kwargs)


def test_phoneme_empty_symbol_rejected() -> None:
    """Boundary: phoneme string cannot be empty."""
    from lang_attack.schemas.phonetics import PhonemeEvaluation
    from pydantic import ValidationError

    score_field = "score" if "score" in PhonemeEvaluation.model_fields else "gop_score"
    kwargs = {"phoneme": "", "status": "correct", score_field: 80.0}
    with pytest.raises(ValidationError):
        PhonemeEvaluation(**kwargs)


def test_multiple_choice_missing_correct_answer_raises() -> None:
    """Boundary: Multiple choice options must contain the correct answer."""
    from lang_attack.schemas.exercise import ExerciseItem
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExerciseItem(
            id="ex-err",
            type="multiple_choice",
            prompt="Choose the right answer",
            options=["option_A", "option_B"],
            correct_answer="option_C",  # Not in options
            language="es",
        )


def test_exercise_language_flexibility() -> None:
    """Boundary: Language code can be arbitrary string without restricted enum."""
    from lang_attack.schemas.exercise import ExerciseItem

    for lang in [
        "pt-BR",
        "zh-Hans",
        "ar-EG",
        "Ancient Greek",
        "Esperanto",
        "x-custom-dialect-123",
    ]:
        item = ExerciseItem(
            id=f"ex-{lang}",
            type="multiple_choice",
            prompt="Prompt",
            options=["A", "B"],
            correct_answer="A",
            language=lang,
        )
        assert item.language == lang


def test_word_time_bounds_inverted_raises() -> None:
    """Boundary: end_time earlier than start_time raises validation error."""
    from lang_attack.schemas.phonetics import WordEvaluation
    from pydantic import ValidationError

    w_score_field = "score" if "score" in WordEvaluation.model_fields else "gop_score"
    kwargs = {
        "word": "test",
        w_score_field: 80.0,
        "phonemes": [],
        "start_time": 1.5,
        "end_time": 0.5,  # Inverted!
    }
    with pytest.raises(ValidationError):
        WordEvaluation(**kwargs)
