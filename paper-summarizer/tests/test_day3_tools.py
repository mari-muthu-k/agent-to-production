"""Day 3, Section 3: tools, the docstring demo (mock and offline fake), parallel calls, the tool-choice policy."""
from langchain_core.messages import AIMessage

from paper_agent.agent import policy, questions
from paper_agent.agent.fakes import FakeToolModel
from paper_agent.agent.memory import read_notes
from paper_agent.agent.model import make_chat_model
from paper_agent.agent.tools import (
    NOT_DEFINED,
    call_tool,
    define_term,
    flaky_search,
    get_section,
    get_section_vague,
    save_note,
    search_paper,
)

TODO1_STUB = ('"""✏️ TODO 1: write this docstring in 3 lines: what the tool does, when the model should pick it, '
              'and what it returns."""')


def test_schemas_come_from_type_hints_and_hide_the_runtime():
    assert search_paper.args == {"query": {"title": "Query", "type": "string"},
                                 "k": {"default": 4, "title": "K", "type": "integer"}}
    for t, arg in ((get_section, "section_id"), (define_term, "term"), (save_note, "note")):
        schema = t.tool_call_schema.model_json_schema()
        assert list(schema["properties"]) == [arg] and schema["required"] == [arg]
    assert flaky_search.name == "search_paper" and flaky_search.description == search_paper.description
    assert get_section_vague.name == get_section.name == "get_section"
    assert get_section_vague.description == "Gets stuff from the paper."


def test_search_paper_returns_tagged_chunks(day3):
    out = call_tool(search_paper, query="How does TinyCoder do on long files?")
    assert out.count('<chunk id="') == 4 and 'section="' in out and 'page="' in out


def test_get_section_unknown_id_is_a_message_listing_valid_ids(day3):
    out = call_tool(get_section, section_id="appendix_b")
    assert out.startswith("Section 'appendix_b' not found. Valid section ids:")
    assert "limitations" in out and "results" in out
    assert "Python only" in call_tool(get_section, section_id="limitations")
    assert "Python only" in call_tool(get_section, section_id="limitations", paper_id="tinycoder-injected")


def test_define_term(day3):
    found = call_tool(define_term, term="pass@1")
    assert '<chunk id="p3-c2" section="3 Training and evaluation" page="3">' in found and "We report pass@1:" in found
    assert call_tool(define_term, term="learning rate") == NOT_DEFINED
    assert call_tool(define_term, term="pass@1", paper_id="quickembed") == NOT_DEFINED


def test_save_note_writes_only_under_the_readers_namespace(day3):
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore()
    assert "Saved" in call_tool(save_note, user_id="alice", store=store, note="Prefers short answers.")
    assert read_notes(store, "alice") == ["Prefers short answers."] and read_notes(store, "bob") == []


def test_docstring_demo_with_mock(day3, mock):
    model = make_chat_model()
    vague = model.bind_tools([search_paper, get_section_vague]).invoke(questions.SECTION_Q)
    specific = model.bind_tools([search_paper, get_section]).invoke(questions.SECTION_Q)
    assert [c["name"] for c in vague.tool_calls] == ["search_paper"]
    assert specific.tool_calls[0]["name"] == "get_section"
    assert specific.tool_calls[0]["args"] == {"section_id": "limitations"}


def test_parallel_tool_calls_in_one_reply(day3, mock):
    reply = make_chat_model().bind_tools([search_paper, get_section]).invoke(questions.TWO_SECTIONS)
    assert reply.content == ""
    assert [(c["name"], c["args"]["section_id"]) for c in reply.tool_calls] == [("get_section", "results"),
                                                                                 ("get_section", "limitations")]
    assert len({c["id"] for c in reply.tool_calls}) == 2


def test_todo1_offline_fake_picks_by_docstring():
    """The 3.4 test: the stub docstring loses to search_paper; the answer-key docstring wins."""
    import langchain.tools as lc_tools

    def with_doc(doc):
        def define_term(term: str) -> str:
            return term
        define_term.__doc__ = doc
        return lc_tools.tool(define_term)

    stub = TODO1_STUB.strip('"')
    for doc, expected in ((stub, "search_paper"), (define_term.description, "define_term"),
                          ("Explains jargon. Use this when the reader asks what a word or term means.", "define_term")):
        reply = FakeToolModel().bind_tools([search_paper, get_section, with_doc(doc)]).invoke(questions.TERM_Q)
        assert reply.tool_calls[0]["name"] == expected, doc


def test_fake_model_scripts_and_records():
    fake = FakeToolModel(script=[AIMessage("first"), AIMessage("second")])
    assert fake.invoke("a").text == "first" and fake.bind_tools([search_paper]).invoke("b").text == "second"
    assert len(fake.requests) == 2 and fake.invoke("c").text == "Done."


def test_policy_rules():
    assert policy.sections_named("Compare the results and the limitations sections") == ["results", "limitations"]
    assert policy.sections_named("What are the results?") == []
    assert policy.term_asked("What does pass@1 mean in this paper?") == "pass@1"
    assert policy.is_profile(questions.PROFILE) and not policy.is_profile(questions.FOLLOW_UP)
    assert policy.note_from(questions.PROFILE) == "Backend engineer; prefers short, technical answers."
    assert policy.when_to_use(get_section.description) and not policy.when_to_use("Gets stuff from the paper.")
    assert policy.intent(questions.SUMMARY) == "search"
