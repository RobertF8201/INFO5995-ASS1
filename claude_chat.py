"""
Claude Multi-turn Conversation Script - API Key Rotation + Logging
"""

import anthropic
import itertools
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# ===================== Configuration =====================
load_dotenv()

API_KEYS = [v for k, v in sorted(os.environ.items()) if k.startswith("ANTHROPIC_API_KEY")]


MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
SYSTEM_PROMPT = "You are a helpful assistant. Please answer the user's questions clearly and concisely."
LOG_DIR = "ai-log"

MAX_RETRIES = 3
RETRY_DELAY = 1
# =========================================================


class Logger:
    """Conversation log manager"""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOG_DIR, f"chat_{timestamp}.log")
        self._write(f"=== Conversation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        print(f"📝 Log saved to: {self.log_path}")

    def _write(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def log_user(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"[{timestamp}] You: {message}")

    def log_assistant(self, message: str, key_index: int):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"[{timestamp}] Claude (Key#{key_index}): {message}")

    def log_end(self):
        self._write(f"\n=== Conversation Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")


class KeyRotator:
    """API Key rotation manager"""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("At least one API Key is required.")
        self.keys = keys
        self._cycle = itertools.cycle(enumerate(keys))
        self.current_index, self.current_key = next(self._cycle)

    def next_key(self):
        self.current_index, self.current_key = next(self._cycle)
        return self.current_key

    def get_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=self.current_key)


class ChatSession:
    """Multi-turn conversation session"""

    def __init__(self, rotator: KeyRotator, logger: Logger):
        self.rotator = rotator
        self.logger = logger
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Send a message and get a reply, with automatic API Key rotation."""
        self.history.append({"role": "user", "content": user_message})
        self.logger.log_user(user_message)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                client = self.rotator.get_client()
                key_display = f"Key #{self.rotator.current_index + 1}"
                print(f"  [Using {key_display}]", end=" ", flush=True)

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=self.history,
                )

                assistant_message = response.content[0].text
                self.history.append({"role": "assistant", "content": assistant_message})
                self.logger.log_assistant(assistant_message, self.rotator.current_index + 1)

                self.rotator.next_key()
                return assistant_message

            except anthropic.AuthenticationError:
                print(f"\n  ⚠️  Key #{self.rotator.current_index + 1} authentication failed, switching...")
                last_error = f"API Key #{self.rotator.current_index + 1} is invalid"
                self.rotator.next_key()
                time.sleep(RETRY_DELAY)

            except anthropic.RateLimitError:
                print(f"\n  ⚠️  Key #{self.rotator.current_index + 1} rate limited, switching...")
                last_error = "Rate limit reached"
                self.rotator.next_key()
                time.sleep(RETRY_DELAY)

            except anthropic.APIError as e:
                print(f"\n  ⚠️  API error: {e}, retrying...")
                last_error = str(e)
                time.sleep(RETRY_DELAY)

        # All retries failed, roll back history
        self.history.pop()
        raise RuntimeError(f"All API Keys failed. Last error: {last_error}")

    def clear_history(self):
        self.history.clear()
        print("✅ Conversation history cleared.")

    def show_history(self):
        if not self.history:
            print("(No conversation history yet)")
            return
        print("\n--- Conversation History ---")
        for msg in self.history:
            role = "You" if msg["role"] == "user" else "Claude"
            print(f"[{role}]: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        print("---------------------------\n")


def main():
    print("=" * 50)
    print("   Claude Multi-turn Chat (API Key Rotation)")
    print("=" * 50)
    print("Commands: /clear=clear history  /history=view history  /quit=exit\n")

    try:
        rotator = KeyRotator(API_KEYS)
        logger = Logger()
        session = ChatSession(rotator, logger)
        print(f"✅ Loaded {len(API_KEYS)} API Key(s). Start chatting!\n")
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            logger.log_end()
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("👋 Goodbye!")
            logger.log_end()
            break
        elif user_input == "/clear":
            session.clear_history()
            continue
        elif user_input == "/history":
            session.show_history()
            continue

        try:
            print("Claude:", end=" ")
            response = session.chat(user_input)
            print(response)
            print()
        except RuntimeError as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
