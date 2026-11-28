"""Client tests: system-prompt injection for the prefix_cache arm."""

from __future__ import annotations

import unittest

from bench.client import ChatClient


class ChatClientTest(unittest.TestCase):
    def test_system_prompt_is_prepended_when_set(self) -> None:
        client = ChatClient("http://127.0.0.1:1", "m", system_prompt="be terse")
        body = client._body("hello", max_tokens=8, ignore_eos=False, stream=False)
        messages = body["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "be terse"})
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})

    def test_no_system_prompt_by_default(self) -> None:
        client = ChatClient("http://127.0.0.1:1", "m")
        body = client._body("hello", max_tokens=8, ignore_eos=False, stream=False)
        messages = body["messages"]
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
