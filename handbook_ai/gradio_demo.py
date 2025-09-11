import gradio as gr


def add_msg(history, message):
    msg = None
    for x in message["files"]:
        print(x)
        if x.split('.')[-1] != 'pdf':
            raise gr.Error("不支持上传非PDF格式文件！")
        msg = gr.ChatMessage(
            role='user',
            content={"path": x}
        )
        history.append(msg)

    if message["text"] is not None:
        msg = gr.ChatMessage(
            role='user',
            content=message["text"]
        )
        history.append(msg)

    return history, ''


def bot(history, checkbox):
    print("is_use")
    print(checkbox)

    history.append(gr.ChatMessage(
        role='assistant',
        content=history[-1]["content"]
    ))
    yield history

    history.append(gr.ChatMessage(
        role='assistant',
        content="""项目     | Value
 -------- | -----
 电脑&#124;平板  | $1600
 _手机_  | $12
 __导管__   | $1"""
    ))
    yield history

    history.append(gr.ChatMessage(
        role='assistant',
        content="""**加粗**
<font size=5>我是尺寸</font>
<font size=10>我是尺寸</font>
<table><tr><td bgcolor=green>背景色yellow</td></tr></table>"""
    ))
    yield history

    history.append(gr.ChatMessage(
        role='assistant',
        content='Here are the pictures for cat'
    ))
    yield history

    history.append(gr.ChatMessage(
        role='assistant',
        content={"path": "data/dataset/upload/cat1.jpg", "alt_text": "测试图片"}
    )
    )

    yield history


# my_theme = gr.Theme.from_hub("gstaff/sketch")
# with gr.Blocks(theme=my_theme) as demo:

EXAMPLES = ["新员工入职报到时需提供哪些资料", "请假的批准权限是怎样的"]
formatted_examples = [{"text": example} for example in EXAMPLES]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="title"):
        gr.Markdown(" # <center>员工AI小助手</center>")
        gr.Markdown(" ###### <center>可上传PDF文件问答</center>")

    with gr.Column():
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            show_copy_button=True,
            show_copy_all_button=True,
            avatar_images=(None, "bot_avatar.png"),
            examples=formatted_examples
        )

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="输入提问，或上传文件...",
            show_label=False
        )
        checkbox = gr.Checkbox(label="使用", info="是否使用上传的文件进行回答")

        clear = gr.Button("清除")
        chat_msg = chat_input.submit(
            add_msg, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(bot, [chatbot, checkbox], chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


        def add_example(evt: gr.SelectData):
            return evt.value


        def vote(data: gr.LikeData):
            if data.liked:
                print("You upvoted this response: ")
                print(data.value)
            else:
                print("You downvoted this response: ")
                print(data.value)


        chatbot.example_select(add_example, None, chat_input)
        chatbot.like(vote, None, None)
        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(default_concurrency_limit=5)
demo.launch(debug=True)