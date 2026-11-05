import numpy as np

from tools_storage import ToolSelector, tool_name


def tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


class Embedder:
    model_name = "history-test"

    def __init__(self):
        self.queries = []

    def encode_query(self, text):
        self.queries.append(text)
        return np.array(
            {
                "brightness": [0.9, 0, 0],
                "music": [0, 0.9, 0],
                "timer": [0, 0, 0.9],
                "two actions": [0.95, 0.85, 0],
            }.get(text, [0.1, 0.1, 0.1])
        )

    def encode_passages(self, texts):
        return np.array(
            [
                [
                    float(f"Action: {name}" in text)
                    for name in ("light", "music", "timer")
                ]
                for text in texts
            ]
        )


def selector(max_tools=2, **kwargs):
    return ToolSelector(
        max_tools=max_tools,
        min_similarity=0.8,
        whitelisted_names=kwargs.pop("whitelisted_names", ()),
        embedder=Embedder(),
        cache_path=None,
        **kwargs,
    )


CATALOG = [tool(name) for name in ("light", "music", "timer")]


def users(*texts):
    return [{"role": "user", "content": text} for text in texts]


def selected(router, messages, **kwargs):
    return router.select(CATALOG, "ignored", messages=messages, **kwargs)


def names(result):
    return [tool_name(item) for item in result.tools]


def call(name="light", error=False):
    return [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call", "function": {"name": name}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call",
            "content": '{"error":"failed"}' if error else '{"result":"ok"}',
        },
    ]


def test_repeated_followups_keep_brightness_and_deduplicate_queries():
    router = selector()
    result = selected(router, users("brightness", "Again", "Again", "A little more"))
    assert names(result) == ["light"]
    assert router.embedder.queries.count("Again") == 1


def test_refresh_evicts_least_recent_not_first_added():
    result = selected(selector(), users("brightness", "music", "brightness", "timer"))
    assert names(result) == ["timer", "light"]
    assert result.trace[-1]["evicted"] == "music"


def test_best_match_survives_full_budget():
    assert names(selected(selector(1), users("timer", "two actions"))) == ["light"]


def test_success_refreshes_but_failure_does_not():
    prefix = users("brightness", "music")
    assert names(selected(selector(), prefix + call() + users("timer"))) == [
        "timer",
        "light",
    ]
    assert names(selected(selector(), prefix + call(error=True) + users("timer"))) == [
        "timer",
        "music",
    ]


def test_hallucinated_and_blacklisted_calls_do_not_enter_set():
    assert names(selected(selector(), call("invented"))) == []
    assert names(selected(selector(blacklisted_names=("light",)), call())) == []


def test_pinned_tools_count_toward_budget():
    router = selector(whitelisted_names=("light",))
    assert names(selected(router, users("music", "timer"))) == ["light", "timer"]


def test_no_cross_conversation_state_and_no_unavailable_tools():
    router = selector()
    assert names(selected(router, users("brightness"))) == ["light"]
    assert names(selected(router, users("Again"))) == []
    assert router.select([], "", messages=users("brightness") + call()).tools == []


def test_empty_zero_and_required_selection():
    assert names(selected(selector(), [])) == []
    assert names(selected(selector(0), users("brightness"))) == []
    assert names(selected(selector(), users("Again"), require_at_least_one=True)) == [
        "light"
    ]
    assert names(
        selected(selector(1), users("brightness"), required_names=("timer",))
    ) == ["timer"]


def test_multimodal_user_content():
    messages = [{"role": "user", "content": [{"type": "text", "text": "brightness"}]}]
    assert names(selected(selector(), messages)) == ["light"]
