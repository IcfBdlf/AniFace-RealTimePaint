"""表情预设只在提交请求时附加，不改写用户输入的提示词。"""

EXPRESSION_PRESETS = {
    "不附加表情": "",
    "微笑": "smiling",
    "闭眼微笑": "closed eyes, smiling",
    "惊讶": "surprised, open mouth",
}


def compose_prompt(prompt, expression):
    suffix = EXPRESSION_PRESETS[expression]
    if not suffix:
        return prompt
    # 避免手动填写的同名标签重复；不删除用户写入的其他表情标签。
    tags = {tag.strip().casefold() for tag in prompt.split(",")}
    additions = [tag for tag in suffix.split(", ") if tag.casefold() not in tags]
    base = prompt.rstrip(" ,")
    return ", ".join([part for part in [base, *additions] if part])
