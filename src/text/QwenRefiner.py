# ---------- LLM 风格迁移 ----------
class QwenRefiner:

    def __init__(self, model=LLM_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def refine(self, raw_text: str) -> str:
        """
        调用 Qwen 进行风格迁移，若失败则返回原文本。
        """
        system_prompt = (
            "你是一位地理描述润色专家。请将输入的“机器生成的地理描述”改写为“自然的中文指路口语”。\n"
            "改写铁律（必须严格遵守）：\n"
            "1. 【空间逻辑铁律】严禁改变任何方位词（东/南/西/北）、距离（米/公里）和具体地名（道路名、地标名）。\n"
            "2. 【去冗余】若句子中重复出现同一个地名（如“朝阳路”），第二次出现必须替换为“这条路”、“这儿”、“该处”或“其”。\n"
            "3. 【句式打散】禁止使用“在...的...方向，距离...”这种长定语结构。应拆分为短句，或改为“往...走”、“顺着...”的动作引导句式。\n"
            "4. 【口语化】适当加入“您呐”、“其实吧”、“顺着”、“拐个弯”等口语词，但不要过度。\n"
            "5. 只输出改写后的中文文本，不要输出任何解释或额外内容。"
        )
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n待改写的文本：{raw_text}",
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 200,
            "timeout": 60
        }
        try:
            response = requests.post(self.url, json=payload, timeout=70)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", raw_text).strip()
            else:
                logger.warning(f"LLM API 返回非200: {response.status_code}")
                return raw_text
        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            return raw_text