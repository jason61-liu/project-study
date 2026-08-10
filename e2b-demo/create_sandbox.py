from e2b import Sandbox


sandbox = Sandbox.create("base", timeout=15 * 60)
result = sandbox.commands.run(
    "printf 'template=%s\\nuser=%s\\n' \"$E2B_TEMPLATE_ID\" \"$(whoami)\""
)

print(f"sandbox_id={sandbox.sandbox_id}")
print(result.stdout, end="")
print("沙箱将在 15 分钟后自动销毁。")
