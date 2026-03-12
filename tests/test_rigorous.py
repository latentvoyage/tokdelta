"""
Rigorous edge-case tests for PromptState.

Every test mutates a PromptState through the public API and then
compares the result against a from-scratch tokenization of the same
final text.  If they ever disagree, the test fails.
"""

import sys, os, string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import PromptState

TOK_NAME = "tiktoken"
MODEL = "gpt-4"


# ── helpers ────────────────────────────────────────────────────────

def fresh(text: str) -> list[int]:
    """Full tokenization from scratch — the ground truth."""
    return PromptState(text, TOK_NAME, MODEL).token_ids


def check(state: PromptState, label: str):
    expected = fresh(state.prompt)
    actual = state.token_ids
    assert actual == expected, (
        f"[{label}] mismatch\n"
        f"  prompt : {state.prompt!r}\n"
        f"  got    : {actual}\n"
        f"  expect : {expected}"
    )
    # offsets must always be in sync with token_ids
    assert len(state.token_ids) == len(state.token_offsets), (
        f"[{label}] len(token_ids)={len(state.token_ids)} "
        f"!= len(offsets)={len(state.token_offsets)}"
    )
    # offsets should tile contiguously from 0 to len(byte_buffer)
    if state.token_offsets:
        assert state.token_offsets[0][0] == 0, f"[{label}] first offset doesn't start at 0"
        assert state.token_offsets[-1][1] == len(state.byte_buffer), (
            f"[{label}] last offset end {state.token_offsets[-1][1]} "
            f"!= buffer length {len(state.byte_buffer)}"
        )
        for i in range(1, len(state.token_offsets)):
            assert state.token_offsets[i][0] == state.token_offsets[i - 1][1], (
                f"[{label}] gap between offset {i - 1} and {i}: "
                f"{state.token_offsets[i - 1]} -> {state.token_offsets[i]}"
            )


def make(text: str) -> PromptState:
    return PromptState(text, TOK_NAME, MODEL)


# ── 1. single-character operations ────────────────────────────────

def test_append_single_chars():
    state = make("x")
    for ch in "abcdefghij":
        state.append(ch)
        check(state, f"append char '{ch}'")
    print("  ✓ append single characters")


def test_insert_single_char_at_every_position():
    base = "Hello"
    for pos in range(len(base) + 1):
        state = make(base)
        state.insert(pos, "X")
        check(state, f"insert 'X' at pos {pos}")
    print("  ✓ insert single char at every position")


def test_delete_single_char_at_every_position():
    base = "Hello!"
    for pos in range(len(base)):
        state = make(base)
        state.delete(pos, pos + 1)
        check(state, f"delete char at pos {pos}")
    print("  ✓ delete single char at every position")


# ── 2. boundary edits ─────────────────────────────────────────────

def test_insert_at_start():
    state = make("world")
    state.insert(0, "Hello ")
    check(state, "insert at start")
    print("  ✓ insert at start")


def test_insert_at_end():
    state = make("Hello")
    state.insert(5, " world")
    check(state, "insert at end")
    print("  ✓ insert at end")


def test_delete_from_start():
    state = make("Hello world")
    state.delete(0, 6)  # "Hello "
    check(state, "delete from start")
    print("  ✓ delete from start")


def test_delete_to_end():
    state = make("Hello world")
    state.delete(5, 11)  # " world"
    check(state, "delete to end")
    print("  ✓ delete to end")


def test_delete_everything():
    state = make("Hello world")
    state.delete(0, 11)
    check(state, "delete everything")
    assert state.prompt == ""
    assert state.token_ids == []
    print("  ✓ delete everything")


# ── 3. no-op / empty edits ────────────────────────────────────────

def test_append_empty():
    state = make("Hello")
    before = list(state.token_ids)
    state.append("")
    check(state, "append empty")
    assert state.token_ids == before
    print("  ✓ append empty string")


def test_insert_empty():
    state = make("Hello")
    before = list(state.token_ids)
    state.insert(3, "")
    check(state, "insert empty")
    assert state.token_ids == before
    print("  ✓ insert empty string")


def test_delete_zero_range():
    state = make("Hello")
    before = list(state.token_ids)
    state.delete(2, 2)
    check(state, "delete zero range")
    assert state.token_ids == before
    print("  ✓ delete zero-length range")


# ── 4. empty prompt ───────────────────────────────────────────────

def test_empty_prompt_then_build_up():
    state = make("")
    check(state, "empty init")
    state.append("a")
    check(state, "empty + 'a'")
    state.append("bc")
    check(state, "empty + 'abc'")
    state.insert(0, "Z")
    check(state, "Zabc")
    state.delete(0, 1)
    check(state, "back to abc")
    print("  ✓ empty prompt then build up")


# ── 5. unicode / multi-byte characters ───────────────────────────

def test_emoji_only():
    state = make("🔥🌍🚀")
    check(state, "emoji init")
    state.append("✨")
    check(state, "emoji append")
    state.insert(2, "💡")  # between 🌍 and 🚀 (char positions)
    check(state, "emoji insert")
    state.delete(0, 1)  # remove 🔥
    check(state, "emoji delete first")
    print("  ✓ emoji-only strings")


def test_cjk_characters():
    state = make("你好世界")
    check(state, "cjk init")
    state.append("再见")
    check(state, "cjk append")
    state.insert(2, "美丽的")
    check(state, "cjk insert")
    state.delete(0, 2)  # remove 你好
    check(state, "cjk delete")
    print("  ✓ CJK characters")


def test_accented_latin():
    state = make("café résumé naïve")
    check(state, "accented init")
    state.append(" über")
    check(state, "accented append")
    state.insert(4, " crème")
    check(state, "accented insert")
    state.delete(0, 5)  # "café "
    check(state, "accented delete")
    print("  ✓ accented latin characters")


def test_mixed_scripts():
    state = make("Hello 你好 مرحبا 🌍")
    check(state, "mixed scripts init")
    state.append(" добрый день")
    check(state, "mixed scripts + russian")
    state.insert(6, "世界 ")
    check(state, "mixed scripts insert")
    print("  ✓ mixed scripts")


def test_surrogate_range_emoji():
    """Emoji that use ZWJ sequences — multiple code points per glyph."""
    state = make("👨‍👩‍👧‍👦 family")
    check(state, "zwj emoji init")
    state.append(" 👩‍💻")
    check(state, "zwj emoji append")
    print("  ✓ ZWJ / surrogate-range emoji")


def test_large_emoji_middle_replace():
    """Build a long emoji-dense prompt, then gut the middle and replace it."""
    # ~120 chars, heavy mix of 1-byte ascii and 4-byte emoji
    chunks = [
        "Hello 🌍 welcome to the 🔥 arena! ",
        "🚀 Let's go 🏁 and race 🏎️ around ",
        "the 🌈 rainbow 🦄 unicorn 🎠 track. ",
        "Watch out for 🐉 dragons and 🧙‍♂️ wizards! ",
        "🎵 Music plays 🎶 as the 🌟 stars ✨ shine. ",
        "🍕 Pizza 🍔 burgers 🌮 tacos for everyone! ",
        "💻 Code 📱 apps 🎮 games all day long. ",
        "🌊 Waves crash 🏖️ on the shore 🐚 shells. ",
        "📚 Books 🖊️ pens 📝 notes fill the desk. ",
        "🎉 Party 🥳 time 🎈 balloons everywhere! ",
    ]
    big_prompt = "".join(chunks)
    state = make(big_prompt)
    check(state, "large emoji init")

    # find the middle ~100 chars and replace them
    mid = len(big_prompt) // 2
    cut_start = mid - 50
    cut_end = mid + 50
    state.delete(cut_start, cut_end)
    check(state, "large emoji delete middle")

    replacement = "🤖 ROBOTS 🤖 have taken over this section 🛸🛸🛸 "
    state.insert(cut_start, replacement)
    check(state, "large emoji insert replacement")

    # one more round: append, then delete the first quarter
    state.append(" 🔚 THE END 🔚")
    check(state, "large emoji append ending")
    quarter = len(state.prompt) // 4
    state.delete(0, quarter)
    check(state, "large emoji delete first quarter")

    # insert emoji right at every remaining 20-char boundary
    for i in range(0, len(state.prompt), 20):
        state.insert(min(i, len(state.prompt)), "🔹")
        check(state, f"large emoji marker at {i}")

    print(f"  ✓ large emoji middle replace ({len(big_prompt)} chars original)")


# ── 6. whitespace and special characters ──────────────────────────

def test_whitespace_heavy():
    state = make("  \t\n  \t\n  ")
    check(state, "whitespace init")
    state.append("\n\n\n")
    check(state, "whitespace append newlines")
    state.insert(3, "    ")
    check(state, "whitespace insert tabs")
    state.delete(0, 4)
    check(state, "whitespace delete")
    print("  ✓ whitespace-heavy content")


def test_newline_variants():
    state = make("line1\nline2\r\nline3\rline4")
    check(state, "mixed newlines init")
    state.append("\n\nline5")
    check(state, "mixed newlines append")
    state.delete(5, 6)  # delete one \n
    check(state, "mixed newlines delete")
    print("  ✓ newline variants (\\n, \\r\\n, \\r)")


def test_punctuation_blast():
    state = make("!@#$%^&*()_+-=[]{}|;':\",./<>?")
    check(state, "punctuation init")
    state.append("```~~~!!!")
    check(state, "punctuation append")
    state.insert(5, "###")
    check(state, "punctuation insert")
    state.delete(10, 15)
    check(state, "punctuation delete")
    print("  ✓ dense punctuation")


# ── 7. repeated characters ────────────────────────────────────────

def test_repeated_chars():
    state = make("a" * 200)
    check(state, "200 a's")
    state.append("b" * 100)
    check(state, "200a + 100b")
    state.insert(100, "c" * 50)
    check(state, "insert 50c in middle")
    state.delete(50, 150)
    check(state, "delete middle chunk")
    print("  ✓ repeated characters")


def test_repeated_spaces():
    state = make(" " * 500)
    check(state, "500 spaces")
    state.insert(250, "X")
    check(state, "X in sea of spaces")
    state.delete(0, 250)
    check(state, "delete half the spaces")
    print("  ✓ repeated spaces")


# ── 8. long inputs ────────────────────────────────────────────────

def test_long_paragraph():
    para = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    ) * 20  # ~2600 chars
    state = make(para)
    check(state, "long paragraph init")
    state.append(" THE END.")
    check(state, "long paragraph append")
    state.insert(100, " [INSERTED] ")
    check(state, "long paragraph insert")
    state.delete(200, 250)
    check(state, "long paragraph delete")
    print(f"  ✓ long paragraph ({len(para)} chars)")


def test_very_long_single_append():
    state = make("start")
    big_chunk = "x" * 5000
    state.append(big_chunk)
    check(state, "5000-char append")
    print("  ✓ very long single append (5000 chars)")


def test_long_unicode_text():
    text = "日本語のテスト文です。" * 50  # 500 CJK chars
    state = make(text)
    check(state, "long cjk init")
    state.append("追加テキスト")
    check(state, "long cjk append")
    state.insert(100, "挿入")
    check(state, "long cjk insert")
    state.delete(50, 80)
    check(state, "long cjk delete")
    print(f"  ✓ long unicode ({len(text)} chars, {len(text.encode('utf-8'))} bytes)")


# ── 9. many small appends (simulates streaming) ──────────────────

def test_streaming_simulation():
    state = make("")
    words = "The quick brown fox jumps over the lazy dog".split()
    for i, word in enumerate(words):
        prefix = " " if i > 0 else ""
        state.append(prefix + word)
        check(state, f"stream word {i}: '{word}'")
    print("  ✓ streaming simulation (word-by-word)")


def test_character_by_character_append():
    state = make("")
    text = "Hello, world! 🌍"
    for i, ch in enumerate(text):
        state.append(ch)
        check(state, f"char-by-char append [{i}]: {ch!r}")
    print(f"  ✓ character-by-character append ({len(text)} chars)")


# ── 10. delete then rebuild ───────────────────────────────────────

def test_delete_all_then_rebuild():
    state = make("original content here")
    check(state, "before wipe")
    state.delete(0, len(state.prompt))
    check(state, "after wipe")
    assert state.prompt == ""
    state.append("brand new content")
    check(state, "rebuilt")
    state.insert(5, " shiny")
    check(state, "rebuilt + insert")
    print("  ✓ delete all then rebuild")


def test_shrink_and_grow():
    state = make("abcdefghijklmnopqrstuvwxyz")
    # repeatedly chop the end and grow back
    for _ in range(5):
        plen = len(state.prompt)
        state.delete(plen // 2, plen)
        check(state, "shrink")
        state.append("_GROW" * 10)
        check(state, "grow")
    print("  ✓ shrink-and-grow cycles")


# ── 11. consecutive inserts at the same position ─────────────────

def test_repeated_inserts_same_position():
    state = make("XY")
    for i in range(20):
        state.insert(1, str(i))
        check(state, f"insert '{i}' at pos 1")
    print("  ✓ 20 inserts at the same position")


# ── 12. alternating operations ────────────────────────────────────

def test_alternating_append_delete():
    state = make("base")
    for i in range(15):
        state.append(f" w{i}")
        check(state, f"alt append {i}")
        plen = len(state.prompt)
        cut = min(3, plen)
        state.delete(plen - cut, plen)
        check(state, f"alt delete {i}")
    print("  ✓ alternating append / delete")


def test_zigzag_insert_delete():
    state = make("0123456789")
    state.insert(5, "XXXXX")
    check(state, "zigzag insert")
    state.delete(3, 8)
    check(state, "zigzag delete")
    state.insert(2, "YY")
    check(state, "zigzag insert 2")
    state.delete(0, 3)
    check(state, "zigzag delete 2")
    state.append("ZZZZZ")
    check(state, "zigzag append")
    print("  ✓ zigzag insert/delete")


# ── 13. code-like input ───────────────────────────────────────────

def test_python_code():
    code = 'def hello():\n    print("Hello, world!")\n    return 42\n'
    state = make(code)
    check(state, "python code init")
    state.insert(len("def hello():"), "\n    x = 1")
    check(state, "python code insert line")
    state.append("\n\nclass Foo:\n    pass\n")
    check(state, "python code append class")
    state.delete(0, 4)  # "def "
    check(state, "python code delete keyword")
    print("  ✓ python code input")


def test_json_content():
    j = '{"name": "test", "values": [1, 2, 3], "nested": {"a": true}}'
    state = make(j)
    check(state, "json init")
    state.insert(len('{"name": "test"'), ', "extra": "field"')
    check(state, "json insert field")
    state.delete(0, 1)  # remove opening brace
    check(state, "json delete brace")
    state.insert(0, "{")  # put it back
    check(state, "json restore brace")
    print("  ✓ JSON content")


# ── 14. numbers and mixed content ─────────────────────────────────

def test_numeric_strings():
    state = make("3.14159265358979323846")
    check(state, "pi init")
    state.append(" 2.71828182845904523536")
    check(state, "pi + e")
    state.insert(0, "constants: ")
    check(state, "prefix")
    state.delete(11, 33)  # remove pi
    check(state, "remove pi")
    print("  ✓ numeric strings")


# ── 15. token-boundary stress ─────────────────────────────────────

def test_edit_at_likely_token_boundaries():
    """Words usually form token boundaries. Edit right at word junctions."""
    state = make("one two three four five six seven eight nine ten")
    # insert in the middle of "three"
    state.insert(10, "X")
    check(state, "mid-word insert")
    # delete across a word boundary
    state.delete(7, 15)
    check(state, "cross-boundary delete")
    # append something that merges with last token
    state.append("11")
    check(state, "merge-prone append")
    print("  ✓ token-boundary stress")


# ── 16. insert that creates or breaks multi-byte sequences ───────

def test_insert_between_multibyte():
    state = make("aéb")
    # 'é' is 2 bytes in UTF-8 (0xC3 0xA9), sits at char index 1
    state.insert(2, "X")  # insert between é and b
    check(state, "insert after multibyte")
    state.insert(1, "Y")  # insert between a and é
    check(state, "insert before multibyte")
    print("  ✓ insert around multi-byte chars")


def test_delete_partial_multibyte_word():
    state = make("naïve café")
    # delete "ïve " — ï is 2-byte
    state.delete(2, 6)
    check(state, "delete across multibyte")
    print("  ✓ delete across multi-byte boundaries")


# ── 17. very short prompts ────────────────────────────────────────

def test_single_char_prompt():
    state = make("a")
    check(state, "single char")
    state.append("b")
    check(state, "ab")
    state.delete(0, 1)
    check(state, "b only")
    state.insert(0, "c")
    check(state, "cb")
    print("  ✓ single-char prompt")


def test_two_char_prompt():
    state = make("ab")
    state.insert(1, "X")
    check(state, "aXb")
    state.delete(0, 1)
    check(state, "Xb")
    state.append("Y")
    check(state, "XbY")
    print("  ✓ two-char prompt")


# ── 18. large sequential edit chain ──────────────────────────────

def test_fifty_step_edit_chain():
    state = make("seed")
    for i in range(50):
        op = i % 3
        if op == 0:
            state.append(f"_{i}")
            check(state, f"chain append {i}")
        elif op == 1:
            plen = len(state.prompt)
            mid = plen // 2
            state.insert(mid, f"[{i}]")
            check(state, f"chain insert {i}")
        else:
            plen = len(state.prompt)
            if plen > 4:
                state.delete(1, 3)
                check(state, f"chain delete {i}")
    print("  ✓ 50-step edit chain")


# ── 19. offset integrity deep check ──────────────────────────────

def test_offsets_reconstruct_prompt():
    """Verify that decoding each token's byte span gives back the prompt."""
    state = make("Hello, world! How are you doing today? 🌍")
    state.append(" Fine, thanks!")
    state.insert(6, " beautiful")

    reconstructed = bytearray()
    for start, end in state.token_offsets:
        reconstructed.extend(state.byte_buffer[start:end])

    assert bytes(reconstructed) == bytes(state.byte_buffer), (
        "Offsets don't tile to cover the full buffer"
    )
    print("  ✓ offsets reconstruct prompt exactly")


# ── 20. error handling ────────────────────────────────────────────

def test_insert_out_of_range():
    state = make("Hello")
    caught = False
    try:
        state.insert(100, "X")
    except ValueError:
        caught = True
    assert caught, "Should raise ValueError for out-of-range insert"
    # state should be unchanged
    check(state, "after bad insert")
    print("  ✓ insert out-of-range raises ValueError")


def test_delete_invalid_range():
    state = make("Hello")
    for bad_range in [(-1, 2), (0, 100), (3, 2)]:
        caught = False
        try:
            state.delete(*bad_range)
        except ValueError:
            caught = True
        assert caught, f"Should raise ValueError for delete{bad_range}"
    check(state, "after bad deletes")
    print("  ✓ delete with invalid range raises ValueError")


def test_insert_negative_position():
    state = make("Hello")
    caught = False
    try:
        state.insert(-1, "X")
    except ValueError:
        caught = True
    assert caught, "Should raise ValueError for negative insert"
    check(state, "after negative insert")
    print("  ✓ insert at negative position raises ValueError")


# ── runner ─────────────────────────────────────────────────────────

ALL_TESTS = [
    # single-char ops
    test_append_single_chars,
    test_insert_single_char_at_every_position,
    test_delete_single_char_at_every_position,
    # boundaries
    test_insert_at_start,
    test_insert_at_end,
    test_delete_from_start,
    test_delete_to_end,
    test_delete_everything,
    # no-ops
    test_append_empty,
    test_insert_empty,
    test_delete_zero_range,
    # empty prompt
    test_empty_prompt_then_build_up,
    # unicode
    test_emoji_only,
    test_cjk_characters,
    test_accented_latin,
    test_mixed_scripts,
    test_surrogate_range_emoji,
    test_large_emoji_middle_replace,
    # whitespace / special
    test_whitespace_heavy,
    test_newline_variants,
    test_punctuation_blast,
    # repeated
    test_repeated_chars,
    test_repeated_spaces,
    # long inputs
    test_long_paragraph,
    test_very_long_single_append,
    test_long_unicode_text,
    # streaming
    test_streaming_simulation,
    test_character_by_character_append,
    # delete + rebuild
    test_delete_all_then_rebuild,
    test_shrink_and_grow,
    # positional stress
    test_repeated_inserts_same_position,
    test_alternating_append_delete,
    test_zigzag_insert_delete,
    # code-like
    test_python_code,
    test_json_content,
    # numbers
    test_numeric_strings,
    # token-boundary stress
    test_edit_at_likely_token_boundaries,
    # multibyte
    test_insert_between_multibyte,
    test_delete_partial_multibyte_word,
    # tiny prompts
    test_single_char_prompt,
    test_two_char_prompt,
    # long chain
    test_fifty_step_edit_chain,
    # offset integrity
    test_offsets_reconstruct_prompt,
    # error handling
    test_insert_out_of_range,
    test_delete_invalid_range,
    test_insert_negative_position,
]

if __name__ == "__main__":
    print(f"\n--- rigorous tests ({TOK_NAME} / {MODEL}) — {len(ALL_TESTS)} cases ---")
    passed = 0
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  ✗ {fn.__name__}: {exc}")

    print(f"\n{passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)
    print("all rigorous tests passed ✅")
