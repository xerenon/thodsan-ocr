from thaiocr.data.format import OCR_INSTRUCTION, to_conversation


def test_conversation_has_image_and_instruction():
    msg = to_conversation("path/to/img.png", "สวัสดี")
    user = msg["messages"][0]
    assert user["role"] == "user"
    assert OCR_INSTRUCTION in user["content"]
    assert msg["images"] == ["path/to/img.png"]


def test_conversation_assistant_is_ground_truth():
    msg = to_conversation("x.png", "ข้อความเฉลย")
    assert msg["messages"][1] == {"role": "assistant", "content": "ข้อความเฉลย"}
