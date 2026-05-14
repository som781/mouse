"""Tests for the GroundingContext sidecar."""

from __future__ import annotations

from mouse.engine.grounding import (
    GroundingContext,
    empty_result_note,
    refusal_message,
)


# ─── Arg grounding ──────────────────────────────────────────────────


def test_no_args_is_always_grounded():
    g = GroundingContext()
    assert g.check_args({}) == []
    assert g.check_args(None) == []  # type: ignore[arg-type]


def test_ignores_non_id_keys():
    """Natural-language params must not be touched — otherwise the
    model can't pass a search query without triggering the check."""
    g = GroundingContext()  # empty, nothing grounded
    assert g.check_args({"query": "how to deploy", "status": "open"}) == []


def test_catches_ungrounded_id_value():
    g = GroundingContext()
    g.record_user_message("please fetch the latest records")
    complaints = g.check_args({"target_id": "557ff89444904763aef9e6d2d2b7"})
    assert complaints == ["target_id='557ff89444904763aef9e6d2d2b7'"]


def test_grounded_via_prior_tool_output():
    g = GroundingContext()
    g.record_tool_result(
        "svc_list_items",
        '[{"target_id":"abc123xyz","name":"Alpha"}]',
    )
    assert g.check_args({"target_id": "abc123xyz"}) == []


def test_grounded_via_user_message():
    g = GroundingContext()
    g.record_user_message("use target_id abc123xyz for this call")
    assert g.check_args({"target_id": "abc123xyz"}) == []


def test_tool_call_id_grounds_recovery_tools():
    """A tool_call_id arg (used by fetch_tool_output / search_tool_output /
    list_tool_outputs) must be grounded when the harness has recorded
    that id from a prior dispatch — without it, the recovery loop is
    blocked because the id lives in the message envelope, not in any
    tool result content."""
    g = GroundingContext()
    g.record_tool_result("svc_list_items", '[{"name":"Alpha"}]')
    # Before the tc_id is recorded, the recovery call would be refused.
    assert g.check_args({"tool_call_id": "call_Vq7ERKWD8zpIfi3Unsy3MJ7a"}) != []
    # After recording it (the agent does this right after dispatch),
    # the same call grounds cleanly.
    g.record_tool_call_id("call_Vq7ERKWD8zpIfi3Unsy3MJ7a")
    assert g.check_args({"tool_call_id": "call_Vq7ERKWD8zpIfi3Unsy3MJ7a"}) == []


def test_record_tool_call_id_ignores_empty():
    g = GroundingContext()
    g.record_tool_call_id("")
    g.record_tool_call_id("   ")
    assert g.harness_ids == set()
    assert g.seen == []


def test_tool_call_ids_do_not_pollute_content_haystack():
    """Harness-minted tool_call_ids land in a separate set so a long
    session full of them doesn't dilute the substring check used for
    genuine content-bearing ids."""
    g = GroundingContext()
    g.record_tool_result("svc_list_items", '[{"target_id":"abc123xyz"}]')
    for i in range(50):
        g.record_tool_call_id(f"call_{i:040d}")
    # Content haystack still holds exactly one entry.
    assert len(g.seen) == 1
    # And the real id is still groundable.
    assert g.check_args({"target_id": "abc123xyz"}) == []
    # But a tool_call_id still grounds via the harness-id set.
    assert g.check_args({"tool_call_id": "call_" + "0" * 39 + "5"}) == []


def test_list_valued_id_key_checks_each_element():
    g = GroundingContext()
    g.record_tool_result("svc_list_children", '{"children":[{"uuid":"real-child"}]}')
    complaints = g.check_args({
        "child_ids": ["real-child", "fake-child"],
    })
    assert complaints == ["child_ids='fake-child'"]


def test_short_values_are_skipped():
    """A 1-char id is trivially a substring of everything; we bail
    rather than burn cycles on meaningless checks."""
    g = GroundingContext()  # empty
    assert g.check_args({"user_id": "1"}) == []


def test_none_id_values_are_skipped():
    """``None`` is a common "not set" sentinel for optional id args —
    skip it rather than refuse the whole call."""
    g = GroundingContext()
    assert g.check_args({"project_id": None}) == []


def test_integer_id_values_are_checked():
    """REST APIs routinely use integer row ids; an ungrounded integer
    must be caught the same way an ungrounded string is."""
    g = GroundingContext()
    # Empty haystack — the stringified 99999 is ungrounded. `page` is
    # not an id-shaped key, so it's ignored regardless of value.
    complaints = g.check_args({"user_id": 99999, "page": 2})
    assert complaints == ["user_id='99999'"]


def test_integer_id_below_min_length_is_skipped():
    """A 2-digit integer stringifies to a 2-char value, which is below
    the min-length threshold — skip rather than flood the refusal."""
    g = GroundingContext()
    assert g.check_args({"user_id": 42}) == []


def test_integer_id_is_grounded_when_seen_in_prior_output():
    g = GroundingContext()
    g.record_tool_result(
        "svc_list_users", '[{"user_id": 12345, "name": "Alpha"}]'
    )
    assert g.check_args({"user_id": 12345}) == []


def test_bool_id_value_is_not_silently_stringified():
    """A stray ``True`` as an id value must not coerce to the string
    "True" and accidentally match anything — boolean id values are
    nonsensical, so skip rather than pretend they're grounded."""
    g = GroundingContext()
    g.record_user_message("please use True as the id")  # haystack has "True"
    # Even though "True" is in the haystack, bool values are skipped.
    assert g.check_args({"user_id": True}) == []


def test_id_key_suffix_variants_all_recognised():
    g = GroundingContext()
    for key in ("target_id", "child_ids", "uuid", "session_id", "gid", "slug"):
        assert g.check_args({key: "ZZZ-fake-value"}) == [
            f"{key}='ZZZ-fake-value'"
        ], f"{key} was not checked"


# ─── Error tracking ─────────────────────────────────────────────────


def test_records_tool_error():
    g = GroundingContext()
    g.record_tool_result(
        "svc_list_children",
        "Error calling tool 'list_children': missing required parent handle",
    )
    block = g.render_errors_block()
    assert "Recent tool errors" in block
    assert "svc_list_children" in block
    assert "parent handle" in block


def test_successful_call_clears_prior_error():
    """A stale error shouldn't keep shouting at the model once the
    same tool eventually succeeds."""
    g = GroundingContext()
    g.record_tool_result("svc_list_children", "Error: missing parent")
    g.record_tool_result("svc_list_children", '[{"uuid":"ok"}]')
    assert g.render_errors_block() == ""


def test_non_error_results_do_not_populate_errors_block():
    g = GroundingContext()
    g.record_tool_result("ok_tool", '{"result":"fine"}')
    assert g.render_errors_block() == ""


def test_errors_block_empty_when_no_errors():
    assert GroundingContext().render_errors_block() == ""


# ─── Reset ──────────────────────────────────────────────────────────


def test_reset_clears_everything():
    g = GroundingContext()
    g.record_user_message("hi")
    g.record_tool_result("x", "Error: bad")
    g.reset()
    assert g.seen == []
    assert g.last_errors == {}
    # Previously grounded id is now ungrounded again.
    assert g.check_args({"target_id": "hi-there"}) != []


# ─── Recency window ─────────────────────────────────────────────────


def test_grounding_haystack_is_scoped_to_recent_window():
    """An id that grounds while it's in the recent window must stop
    grounding once enough new entries push it out — older context
    is no longer trusted as a source of identifiers."""
    g = GroundingContext()
    g.record_tool_result("svc_list_items", '[{"target_id":"abc123xyz"}]')
    # Immediately groundable.
    assert g.check_args({"target_id": "abc123xyz"}) == []
    # Push the entry out of the window. The window is 80 by default,
    # so 120 fresh entries are more than enough.
    for i in range(120):
        g.record_user_message(f"msg-{i} no-id-here")
    # The id is still in ``seen`` (bound is 200) but no longer in the
    # recency window — grounding should now refuse it.
    assert "abc123xyz" in "\n".join(g.seen)
    assert g.check_args({"target_id": "abc123xyz"}) != []


# ─── Trimming ───────────────────────────────────────────────────────


def test_seen_is_trimmed_to_bound():
    g = GroundingContext()
    for i in range(500):
        g.record_user_message(f"msg-{i}")
    # Bound is 200 by default.
    assert len(g.seen) == 200
    # Oldest entries fall off the front.
    assert "msg-0" not in g.seen[0]
    assert g.seen[-1] == "msg-499"


# ─── Empty-result reflection ────────────────────────────────────────


def test_empty_result_note_fires_on_literal_empties():
    for payload in ("", "[]", "{}", "null", "(no output)", "  "):
        assert empty_result_note(payload), f"should nudge on {payload!r}"


def test_empty_result_note_fires_on_wrapped_empty_list():
    # The common "no rows returned" shapes from structured responses.
    assert empty_result_note('{"items":[]}')
    assert empty_result_note('{"items":[],"next_cursor":null}')
    assert empty_result_note('{"results": []}')
    assert empty_result_note('{"results": null}')


def test_empty_result_note_fires_on_paginated_shape_with_metadata():
    """``{"data": [], "meta": {"page": 1}}`` is the dominant real-world
    "zero rows" shape. The list is the data-bearing field; metadata
    siblings should not suppress the nudge."""
    assert empty_result_note('{"data":[],"meta":{"page":1,"total":0}}')
    assert empty_result_note('{"results":[],"next_cursor":"abc"}')
    # Nested wrapping — data lives one level down.
    assert empty_result_note('{"data":{"items":[]}}')


def test_empty_result_note_silent_on_single_record_dict():
    """A dict with a populated scalar is a real record, not empty."""
    assert empty_result_note('{"id":"abc","name":"Alpha"}') == ""
    assert empty_result_note('{"status":"ok"}') == ""


def test_empty_result_note_silent_on_non_empty():
    assert empty_result_note('{"items":[{"id":"t1"}]}') == ""
    assert empty_result_note('[{"target_id":"abc"}]') == ""
    assert empty_result_note("some plain text result") == ""


def test_empty_result_note_mentions_scope():
    note = empty_result_note("[]")
    assert "scope" in note.lower() or "discovery" in note.lower()


# ─── Refusal message ────────────────────────────────────────────────


def test_refusal_message_names_tool_and_args():
    msg = refusal_message("svc_list_items", ["child_ids='fake-child'"])
    assert "svc_list_items" in msg
    assert "child_ids='fake-child'" in msg
    assert "discovery tool" in msg.lower()


