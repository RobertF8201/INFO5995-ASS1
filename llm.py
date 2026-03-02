"""
Claude 多轮对话脚本 - 支持多个 API Key 轮询
"""

import anthropic
import itertools
import time

# ==================== 配置区域 =====================
from dotenv import load_dotenv
import os

load_dotenv()

API_KEYS = [v for k, v in sorted(os.environ.items()) if k.startswith("ANTHROPIC_API_KEY")]

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
SYSTEM_PROMPT = "你是一个智能助手，请用中文回答用户的问题。"

# 请求失败时的重试次数
MAX_RETRIES = 3
# 切换 key 后的等待时间（秒）
RETRY_DELAY = 1
# ====================================================


class KeyRotator:
    """API Key 轮询管理器"""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("至少需要提供一个 API Key")
        self.keys = keys
        self._cycle = itertools.cycle(enumerate(keys))
        self.current_index, self.current_key = next(self._cycle)

    def next_key(self):
        self.current_index, self.current_key = next(self._cycle)
        return self.current_key

    def get_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=self.current_key)


class ChatSession:
    """多轮对话会话"""

    def __init__(self, rotator: KeyRotator):
        self.rotator = rotator
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """发送消息并获取回复，自动轮询 API Key"""
        self.history.append({"role": "user", "content": user_message})

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                client = self.rotator.get_client()
                key_display = f"Key #{self.rotator.current_index + 1}"
                print(f"  [使用 {key_display}]", end=" ", flush=True)

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=self.history,
                )

                assistant_message = response.content[0].text
                self.history.append({"role": "assistant", "content": assistant_message})

                # 成功后切换到下一个 key（轮询策略）
                self.rotator.next_key()
                return assistant_message

            except anthropic.AuthenticationError:
                print(f"\n  ⚠️  Key #{self.rotator.current_index + 1} 认证失败，切换下一个...")
                last_error = f"API Key #{self.rotator.current_index + 1} 无效"
                self.rotator.next_key()
                time.sleep(RETRY_DELAY)

            except anthropic.RateLimitError:
                print(f"\n  ⚠️  Key #{self.rotator.current_index + 1} 触发限流，切换下一个...")
                last_error = "触发速率限制"
                self.rotator.next_key()
                time.sleep(RETRY_DELAY)

            except anthropic.APIError as e:
                print(f"\n  ⚠️  API 错误: {e}，重试中...")
                last_error = str(e)
                time.sleep(RETRY_DELAY)

        # 所有重试失败，回滚历史
        self.history.pop()
        raise RuntimeError(f"所有 API Key 均请求失败。最后错误: {last_error}")

    def clear_history(self):
        self.history.clear()
        print("✅ 对话历史已清空")

    def show_history(self):
        if not self.history:
            print("（暂无对话历史）")
            return
        print("\n--- 对话历史 ---")
        for msg in self.history:
            role = "你" if msg["role"] == "user" else "Claude"
            print(f"[{role}]: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        print("----------------\n")


def main():
    print("=" * 50)
    print("   Claude 多轮对话 (多 API Key 轮询)")
    print("=" * 50)
    print("命令: /clear=清空历史  /history=查看历史  /quit=退出\n")

    try:
        rotator = KeyRotator(API_KEYS)
        session = ChatSession(rotator)
        print(f"✅ 已加载 {len(API_KEYS)} 个 API Key，开始对话！\n")
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return

    while True:
        try:
            user_input = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("👋 再见！")
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
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()